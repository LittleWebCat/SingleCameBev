#!/usr/bin/env python3
"""Show the map route, driven path and kept landmarks in RViz around the localized vehicle.

Subscribes to /camera_lidar_bev/localization/odometry (enu -> os_sensor) and
  * broadcasts TF enu -> os_sensor, so map-frame data can be drawn in the sensor-fixed views;
  * publishes the saved route (MAP/gps_alignment/trajectory_enu.csv) as nav_msgs/Path on
    /camera_lidar_bev/route/reference (frame enu, ~1 m spacing);
  * publishes the driven path on /camera_lidar_bev/route/driven (accepted poses, 0.5 m spacing);
  * publishes the stop signs / traffic lights kept in tools/review_landmarks.py
    (MAP/landmarks/landmarks_reviewed.json) on /camera_lidar_bev/landmarks/kept_markers (frame enu),
    and 4x larger copies for the whole-route map view on /camera_lidar_bev/landmarks/kept_markers_map;
  * with MAP/gps_alignment/tree_voxels.npz from tools/remove_map_trees.py: publishes the saved map once
    on /camera_lidar_bev/map/points (frame enu, field "height" = metres above the local ground, for
    colouring);
  * only with --remove-trees: that map without trees, and every localized scan without tree points on
    /camera_lidar_bev/points_no_trees (frame os_sensor). A scan is filtered with the pose estimated for
    that same scan (matched by scan-end time), so it appears at the localization rate.
Both paths are republished at 10 Hz stamped with the latest localization time: RViz places a
Path with the transform at the time it is received, so a path published once would stay where
the vehicle was at that moment instead of moving with the map. Before localization they carry
stamp 0, which only places them in a map-fixed view (config/rviz/landmarks_map.rviz); a static
identity TF enu -> enu_map_view makes the enu frame exist in TF for that view.
Display only: it does not change localization, detection or control (they keep using /points).

  /usr/bin/python3 tools/localization_rviz_bridge.py [--map maps/local_gps_recording_002]
"""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path

# Per-scan products are (N x 3) @ (3 x 3): threaded BLAS only adds idle spinning on every core.
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from saved_map_source import read_pcd_xyz, scan_end_nanoseconds

SENSOR_FROM_LIDAR = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.038195], [0, 0, 0, 1]], float)


def pose(frame, x, y, z, q=(0.0, 0.0, 0.0, 1.0)):
    p = PoseStamped()
    p.header.frame_id = frame
    p.pose.position.x, p.pose.position.y, p.pose.position.z = float(x), float(y), float(z)
    p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = (float(v) for v in q)
    return p


def float_cloud(header, columns, names='xyz'):
    """PointCloud2 with one float32 field per column of `columns`."""
    msg = PointCloud2()
    msg.header = header
    msg.height, msg.width = 1, len(columns)
    msg.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(names)]
    msg.is_bigendian, msg.point_step, msg.is_dense = False, 4 * len(names), True
    msg.row_step = msg.point_step * len(columns)
    msg.data = np.ascontiguousarray(columns, '<f4').tobytes()
    return msg


def landmark_markers(landmarks, scale=1.0):
    out = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    out.markers.append(clear)

    def marker(ns, mid, kind):
        m = Marker()
        m.header.frame_id, m.ns, m.id, m.type, m.frame_locked = 'enu', ns, mid, kind, True
        m.pose.orientation.w = 1.0
        m.color.a = 1.0
        out.markers.append(m)
        return m

    for i, l in enumerate(landmarks):
        x, y, z = l['position_enu_m']
        # Stack labels of landmarks within 6 m of each other (e.g. two signal heads on one arm).
        stacked = sum(math.dist((x, y), o['position_enu_m'][:2]) < 6.0 for o in landmarks[:i])
        stop = l['type'] == 'stop_sign'
        m = marker(l['type'], l['id'], Marker.CYLINDER if stop else Marker.CUBE)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), float(z)
        size, (m.color.r, m.color.g, m.color.b) = ((0.9, 0.9, 0.9), (1.0, 0.0, 0.0)) if stop else ((0.5, 0.5, 1.2), (1.0, 0.85, 0.0))
        m.scale.x, m.scale.y, m.scale.z = (v * scale for v in size)
        label = marker('label', l['id'], Marker.TEXT_VIEW_FACING)
        label.pose.position.x, label.pose.position.y, label.pose.position.z = float(x), float(y), float(z) + (1.6 + 1.5 * stacked) * scale
        label.scale.z = 1.2 * scale
        label.color.r = label.color.g = label.color.b = 1.0
        label.text = f"{'STOP' if stop else 'light'} #{l['id']}"
        if stop and l.get('stop_line_estimated'):
            line = marker('stop_line_estimated', l['id'], Marker.LINE_STRIP)
            line.scale.x = 0.45 * scale
            line.points = [Point(x=float(a), y=float(b), z=float(c)) for a, b, c in l['stop_line_estimated']['endpoints_enu_m']]
            line.color.r = line.color.g = line.color.b = 1.0
    return out


class TreeFilter:
    """Drop scan points that fall in tree voxels of the saved map (grown by one voxel)."""

    def __init__(self, path):
        t = np.load(path)
        self.voxel_m, self.origin, self.dims = float(t['voxel_m']), t['origin'], t['dims']
        bits = np.zeros(int(np.prod(self.dims)), bool)  # one bit per voxel of the map: ~45 MB packed
        bits[t['live_keys']] = True
        self.bits = np.packbits(bits)

    def __call__(self, scan, rotation, position):
        """Return the scan's valid points in os_sensor without tree points, and the number removed."""
        f = {x.name: x for x in scan.fields}
        raw = np.frombuffer(scan.data, np.uint8).reshape(scan.height, scan.row_step)[:, :scan.width * scan.point_step]
        raw = raw.reshape(-1, scan.point_step)
        xyz = np.stack([raw[:, f[k].offset:f[k].offset + 4].view('<f4')[:, 0] for k in 'xyz'], 1)
        xyz = xyz[np.isfinite(xyz).all(1) & (np.einsum('ij,ij->i', xyz, xyz) > 1.0)].astype(np.float64)
        if scan.header.frame_id == 'os_lidar':
            xyz = xyz @ SENSOR_FROM_LIDAR[:3, :3].T + SENSOR_FROM_LIDAR[:3, 3]
        ijk = np.floor((xyz @ rotation.T + position - self.origin) / self.voxel_m).astype(np.int64) + 1
        inside = (ijk >= 0).all(1) & (ijk < self.dims).all(1)
        key = np.where(inside, (ijk[:, 2] * self.dims[1] + ijk[:, 1]) * self.dims[0] + ijk[:, 0], 0)
        tree = inside & ((self.bits[key >> 3] >> (7 - (key & 7))) & 1).astype(bool)
        return xyz[~tree], int(tree.sum())


def saved_map_cloud(path, ground, route_xy, margin_m=50.0, voxel_m=0.2):
    """Saved map near the route, one point per voxel, as columns x y z height-above-ground."""
    xyz = np.asarray(read_pcd_xyz(path), np.float64)
    lo, hi = route_xy.min(0) - margin_m, route_xy.max(0) + margin_m
    xyz = xyz[((xyz[:, :2] >= lo) & (xyz[:, :2] <= hi)).all(1)]
    _, first = np.unique(np.floor(xyz / voxel_m).astype(np.int64), axis=0, return_index=True)
    xyz = xyz[first]
    grid = ground['ground']
    ij = np.floor((xyz[:, :2] - ground['ground_origin']) / float(ground['ground_cell_m'])).astype(np.int64)
    ij = np.clip(ij, 0, np.array(grid.shape[::-1]) - 1)
    return np.column_stack([xyz, xyz[:, 2] - grid[ij[:, 1], ij[:, 0]]])


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', type=Path, default=root / 'maps/local_gps_recording_002')
    parser.add_argument('--route-spacing-m', type=float, default=1.0)
    parser.add_argument('--driven-spacing-m', type=float, default=0.5)
    parser.add_argument('--driven-max-points', type=int, default=5000)
    parser.add_argument('--landmarks', type=Path, help='default: MAP/landmarks/landmarks_reviewed.json')
    parser.add_argument('--remove-trees', action='store_true',
                        help='also publish the map and live scans without trees (tools/remove_map_trees.py)')
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node('localization_rviz_bridge')
    log = node.get_logger()
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    route_pub = node.create_publisher(PathMsg, '/camera_lidar_bev/route/reference', latched)
    driven_pub = node.create_publisher(PathMsg, '/camera_lidar_bev/route/driven', latched)
    landmark_pub = node.create_publisher(MarkerArray, '/camera_lidar_bev/landmarks/kept_markers', latched)
    landmark_map_pub = node.create_publisher(MarkerArray, '/camera_lidar_bev/landmarks/kept_markers_map', latched)
    map_pub = node.create_publisher(PointCloud2, '/camera_lidar_bev/map/points', latched)
    scan_pub = node.create_publisher(PointCloud2, '/camera_lidar_bev/points_no_trees', qos_profile_sensor_data)
    tf = TransformBroadcaster(node)
    static_tf = StaticTransformBroadcaster(node)
    anchor = TransformStamped()
    anchor.header.frame_id, anchor.child_frame_id = 'enu', 'enu_map_view'
    anchor.transform.rotation.w = 1.0
    static_tf.sendTransform(anchor)

    # Saved route, thinned to about one pose per metre.
    t = np.genfromtxt(args.map / 'gps_alignment/trajectory_enu.csv', delimiter=',', names=True)
    route = PathMsg()
    route.header.frame_id = 'enu'
    last = None
    for row in t:
        xyz = (row['x_m'], row['y_m'], row['z_m'])
        if last is None or math.dist(xyz, last) >= args.route_spacing_m:
            route.poses.append(pose('enu', *xyz, (row['qx'], row['qy'], row['qz'], row['qw'])))
            last = xyz
    log.info(f'Route: {len(route.poses)} poses over {len(t)} trajectory samples from {args.map.name}')

    landmark_file = args.landmarks or args.map / 'landmarks/landmarks_reviewed.json'
    if landmark_file.exists():
        kept = [l for l in json.loads(landmark_file.read_text())['landmarks'] if l.get('keep')]
        landmark_pub.publish(landmark_markers(kept))
        landmark_map_pub.publish(landmark_markers(kept, scale=4.0))
        log.info(f"Landmarks: {sum(l['type'] == 'stop_sign' for l in kept)} stop signs and "
                 f"{sum(l['type'] == 'traffic_light' for l in kept)} traffic lights kept in {landmark_file.name}")
    else:
        log.warn(f'No {landmark_file}: review the mapped landmarks with tools/review_landmarks.py to show them')

    tree_file = args.map / 'gps_alignment/tree_voxels.npz'
    if not tree_file.exists():
        log.warn(f'No {tree_file}: run tools/remove_map_trees.py to publish the saved map (and for --remove-trees)')
    trees = TreeFilter(tree_file) if args.remove_trees and tree_file.exists() else None
    if tree_file.exists():
        name = 'pointcloud_map_enu_no_trees.pcd' if trees else 'pointcloud_map_enu.pcd'
        cloud = saved_map_cloud(args.map / 'gps_alignment' / name, np.load(tree_file), np.stack([t['x_m'], t['y_m']], 1))
        map_pub.publish(float_cloud(Header(frame_id='enu'), cloud, ('x', 'y', 'z', 'height')))
        log.info(f'Saved map{" without trees" if trees else ""}: {len(cloud)} points' +
                 ('; live scans without trees on /camera_lidar_bev/points_no_trees' if trees else ''))

    driven = PathMsg()
    driven.header.frame_id = 'enu'
    latest = {'stamp': None}

    def republish():
        if latest['stamp'] is not None:
            route.header.stamp = driven.header.stamp = latest['stamp']
        route_pub.publish(route)  # stamp 0 until localized: placed only in map-fixed views
        driven_pub.publish(driven)

    node.create_timer(0.1, republish)
    scans = deque(maxlen=20)  # (scan-end ns, scan) waiting for their localization result

    def on_points(m):
        try:
            scans.append((scan_end_nanoseconds(m), m))
        except ValueError as e:
            log.warn(f'scan not usable for the tree filter: {e}', throttle_duration_sec=10.0)

    def on_odometry(m):
        if m.child_frame_id != 'os_sensor':
            return
        p, q = m.pose.pose.position, m.pose.pose.orientation
        msg = TransformStamped()
        msg.header = m.header
        msg.child_frame_id = 'os_sensor'
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = p.x, p.y, p.z
        msg.transform.rotation = q
        tf.sendTransform(msg)
        latest['stamp'] = m.header.stamp
        if not driven.poses or math.dist((p.x, p.y, p.z), (driven.poses[-1].pose.position.x, driven.poses[-1].pose.position.y,
                                                            driven.poses[-1].pose.position.z)) >= args.driven_spacing_m:
            driven.poses.append(pose('enu', p.x, p.y, p.z, (q.x, q.y, q.z, q.w)))
            del driven.poses[:-args.driven_max_points]
        stamp = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
        scan = next((s for end, s in scans if end == stamp), None)
        if trees and scan is not None:
            xyz, _ = trees(scan, Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([p.x, p.y, p.z]))
            scan_pub.publish(float_cloud(Header(stamp=scan.header.stamp, frame_id='os_sensor'), xyz))

    node.create_subscription(Odometry, '/camera_lidar_bev/localization/odometry', on_odometry, 10)
    if trees:
        node.create_subscription(PointCloud2, '/points', on_points, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
