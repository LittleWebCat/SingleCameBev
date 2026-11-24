#!/usr/bin/env python3
"""Map stop signs and traffic lights from camera YOLO + LiDAR + map localization while driving.

For every camera 2D detection of "stop sign" or "traffic light", the LiDAR scan closest in time
is projected into that camera with the rig calibration; the nearest consistent depth cluster
inside the box gives the object's 3D position in os_sensor, which the localization pose at that
scan's end time converts to the map (enu). Sightings within --merge-radius-m merge into one
landmark; a landmark counts as confirmed after --min-sightings. Each confirmed stop sign also gets
an ESTIMATED stop line: across the mapped route at the route point nearest the sign (no paint is
detected). Publishes /camera_lidar_bev/landmarks/markers (enu, frame-locked) and writes
MAP/landmarks/landmarks.json every 10 s and on exit.

  /usr/bin/python3 tools/map_landmarks.py [--map maps/local_gps_recording_002]
"""
import argparse
from collections import deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(__file__).resolve().parents[1]
RIG = ROOT / 'config/calibration/rig_20261124'
CAMERAS = {  # name: (detections topic, intrinsics, extrinsics)
    'front_wide': ('/camera_lidar_bev/front_wide/detections_2d', RIG / 'front_wide_1920x1080.yaml',
                   RIG / 'front_wide_T_os_sensor.yaml'),
    # front_narrow (main front camera, /camera_lidar_bev/detections_2d) is not connected now.
    'front_left': ('/camera_lidar_bev/front_left/detections_2d', RIG / 'front_left_1600x1200.yaml',
                   RIG / 'front_left_T_os_sensor.yaml'),
    'front_right': ('/camera_lidar_bev/front_right/detections_2d', RIG / 'front_right_1600x1200.yaml',
                    RIG / 'front_right_T_os_sensor.yaml'),
}
SENSOR_FROM_LIDAR = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.038195], [0, 0, 0, 1]], float)
KINDS = {'stop sign': 'stop_sign', 'traffic light': 'traffic_light'}
HEIGHT_RANGE = {'stop_sign': (-1.6, 1.2), 'traffic_light': (-0.8, 6.0)}  # m relative to the LiDAR (~1.8 m up)


def stamp_ns(h):
    return h.stamp.sec * 1_000_000_000 + h.stamp.nanosec


def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Camera:
    def __init__(self, name, topic, intrinsics, extrinsics):
        intr = yaml.safe_load(Path(intrinsics).read_text())
        self.name, self.topic = name, topic
        self.k = np.array(intr['camera_matrix']['data'], float).reshape(3, 3)
        self.d = np.array(intr['distortion_coefficients']['data'], float)
        self.w, self.h = intr['image_width'], intr['image_height']
        ext = yaml.safe_load(Path(extrinsics).read_text())
        self.cam_from_sensor = np.array(ext['transforms']['T_camera_from_os_sensor']['matrix'], float)

    def locate(self, xyz, box):
        """3D os_sensor position of the nearest consistent depth cluster inside box (u0, v0, u1, v1)."""
        pc = xyz @ self.cam_from_sensor[:3, :3].T + self.cam_from_sensor[:3, 3]
        ahead = pc[:, 2] > 1.0
        if ahead.sum() < 10:
            return None
        uv, _ = cv2.projectPoints(pc[ahead].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.k, self.d)
        uv = uv.reshape(-1, 2)
        u0, v0, u1, v1 = box
        shrink_u, shrink_v = 0.15 * (u1 - u0), 0.15 * (v1 - v0)  # stay off the box edges (background)
        inside = (uv[:, 0] > u0 + shrink_u) & (uv[:, 0] < u1 - shrink_u) & (uv[:, 1] > v0 + shrink_v) & (uv[:, 1] < v1 - shrink_v)
        if inside.sum() < 3:
            return None
        depth = pc[ahead][inside, 2]
        near = np.percentile(depth, 20)
        cluster = np.abs(depth - near) < 1.0  # the object, not what is behind it
        if cluster.sum() < 3:
            return None
        return xyz[ahead][inside][cluster].mean(0), int(cluster.sum()), float(near)


class Landmark:
    def __init__(self, kind, enu, obs):
        self.kind, self.points, self.observations = kind, [enu], [obs]

    @property
    def position(self):
        return np.median(np.array(self.points), axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--min-score', type=float, default=0.45)
    parser.add_argument('--max-range-m', type=float, default=40.0)
    parser.add_argument('--merge-radius-m', type=float, default=2.5)
    parser.add_argument('--min-sightings', type=int, default=3)
    parser.add_argument('--lane-width-m', type=float, default=3.6)
    args = parser.parse_args()
    out_dir = args.map / 'landmarks'
    out_dir.mkdir(exist_ok=True)
    route = np.genfromtxt(args.map / 'gps_alignment/trajectory_enu.csv', delimiter=',', names=True)
    route_xy = np.stack([route['x_m'], route['y_m']], 1)
    route_z = route['z_m']

    rclpy.init()
    node = rclpy.create_node('landmark_mapper')
    scans = deque(maxlen=6)    # (start_ns, end_ns, xyz os_sensor)
    poses = deque(maxlen=400)  # (stamp_ns, R, t) enu <- os_sensor, stamped at scan end
    landmarks = []
    stats = {'detections': 0, 'located': 0, 'no_scan': 0, 'no_pose': 0, 'no_points': 0, 'out_of_range': 0}

    def on_points(m):
        f = {x.name: x for x in m.fields}
        raw = np.frombuffer(m.data, np.uint8).reshape(m.height, m.row_step)[:, :m.width * m.point_step].reshape(-1, m.point_step)
        xyz = np.stack([raw[:, f[k].offset:f[k].offset + 4].view('<f4')[:, 0] for k in 'xyz'], 1).astype(np.float32)
        t_end = int(raw[:, f['t'].offset:f['t'].offset + 4].view('<u4').max()) if 't' in f else 0
        good = np.isfinite(xyz).all(1) & (np.einsum('ij,ij->i', xyz, xyz) > 1.0)
        xyz = xyz[good]
        if m.header.frame_id == 'os_lidar':
            xyz = xyz @ SENSOR_FROM_LIDAR[:3, :3].T.astype(np.float32) + SENSOR_FROM_LIDAR[:3, 3].astype(np.float32)
        start = stamp_ns(m.header)
        scans.append((start, start + t_end, xyz))

    def on_odometry(m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        poses.append((stamp_ns(m.header), quat_to_matrix((q.x, q.y, q.z, q.w)), np.array([p.x, p.y, p.z])))

    def pose_at(t):
        for (t0, r0, p0), (t1, r1, p1) in zip(list(poses)[:-1], list(poses)[1:]):
            if t0 <= t <= t1 and t1 - t0 <= 300_000_000:
                f = (t - t0) / max(t1 - t0, 1)
                return (r0 if f < 0.5 else r1), p0 + f * (p1 - p0)  # nearest rotation; 0.05 s apart
        return None

    pending = deque(maxlen=200)  # detections waiting for their scan's pose (localization lags ~0.1 s)

    def on_detections(m, cam):
        t_cam = stamp_ns(m.header)
        for d in m.detections:
            if not d.results:
                continue
            h = d.results[0].hypothesis
            kind = KINDS.get(h.class_id)
            if kind is None or h.score < args.min_score:
                continue
            stats['detections'] += 1
            scan = min(scans, key=lambda s: abs((s[0] + s[1]) // 2 - t_cam), default=None)
            if scan is None or abs((scan[0] + scan[1]) // 2 - t_cam) > 80_000_000:
                stats['no_scan'] += 1
                continue
            b = d.bbox
            box = (b.center.position.x - b.size_x / 2, b.center.position.y - b.size_y / 2,
                   b.center.position.x + b.size_x / 2, b.center.position.y + b.size_y / 2)
            pending.append((time.monotonic(), cam, kind, float(h.score), box, scan))

    def process_pending():
        keep = deque(maxlen=pending.maxlen)
        while pending:
            item = pending.popleft()
            queued, cam, kind, score, box, scan = item
            pose = pose_at(scan[1])
            if pose is None:
                if time.monotonic() - queued < 1.5:
                    keep.append(item)
                else:
                    stats['no_pose'] += 1
                continue
            locate(cam, kind, score, box, scan, pose)
        pending.extend(keep)

    def locate(cam, kind, score, box, scan, pose):
        hit = cam.locate(scan[2], box)
        if hit is None:
            stats['no_points'] += 1
            return
        p_sensor, n_points, depth = hit
        lo, hi = HEIGHT_RANGE[kind]
        if depth > args.max_range_m or not lo <= p_sensor[2] <= hi:
            stats['out_of_range'] += 1
            return
        r, t = pose
        enu = r @ p_sensor + t
        stats['located'] += 1
        obs = {'utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'camera': cam.name,
               'score': round(score, 3), 'range_m': round(depth, 2), 'lidar_points': n_points}
        match = min((lm for lm in landmarks if lm.kind == kind), default=None,
                    key=lambda lm: np.linalg.norm(lm.position[:2] - enu[:2]))
        if match is not None and np.linalg.norm(match.position[:2] - enu[:2]) <= args.merge_radius_m:
            match.points.append(enu)
            match.observations.append(obs)
        else:
            landmarks.append(Landmark(kind, enu, obs))
            node.get_logger().info(f'new {kind} candidate at enu ({enu[0]:.1f}, {enu[1]:.1f}) from {cam.name}, {depth:.1f} m away')

    def stop_line(sign):
        """Estimated stop line: across the route at the route point nearest the sign."""
        i = int(np.argmin(np.linalg.norm(route_xy - sign[:2], axis=1)))
        a, b = route_xy[max(i - 2, 0)], route_xy[min(i + 2, len(route_xy) - 1)]
        heading = (b - a) / max(np.linalg.norm(b - a), 1e-6)
        normal = np.array([-heading[1], heading[0]])
        half = args.lane_width_m / 2
        z = route_z[i] - 1.7  # route poses are at LiDAR height; paint is on the ground
        return [list(map(float, (*(route_xy[i] + s * half * normal), z))) for s in (-1, 1)], i

    def confirmed():
        return [lm for lm in landmarks if len(lm.observations) >= args.min_sightings]

    def save():
        doc = {'map': args.map.name, 'frame': 'enu', 'updated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'method': 'camera YOLO box + LiDAR points inside it + map localization; median of sightings',
               'min_sightings': args.min_sightings, 'stats': stats, 'landmarks': []}
        for i, lm in enumerate(landmarks):
            p = lm.position
            item = {'id': i, 'type': lm.kind, 'confirmed': len(lm.observations) >= args.min_sightings,
                    'position_enu_m': [round(float(v), 2) for v in p], 'sightings': len(lm.observations),
                    'spread_m': round(float(np.linalg.norm(np.std(np.array(lm.points)[:, :2], axis=0))), 2),
                    'cameras': sorted({o['camera'] for o in lm.observations}),
                    'best_score': max(o['score'] for o in lm.observations)}
            if lm.kind == 'stop_sign':
                line, idx = stop_line(p)
                item['stop_line_estimated'] = {'endpoints_enu_m': [[round(v, 2) for v in e] for e in line],
                                               'route_index': idx, 'source': 'nearest mapped-route point to the sign'}
            doc['landmarks'].append(item)
        (out_dir / 'landmarks.json').write_text(json.dumps(doc, indent=2) + '\n')

    marker_pub = node.create_publisher(MarkerArray, '/camera_lidar_bev/landmarks/markers',
                                       QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                                  durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def publish():
        out = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        out.markers.append(clear)
        for i, lm in enumerate(landmarks):
            ok = len(lm.observations) >= args.min_sightings
            p = lm.position
            m = Marker()
            m.header.frame_id = 'enu'
            m.ns, m.id, m.frame_locked = lm.kind, i * 3, True
            m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in p)
            m.pose.orientation.w = 1.0
            if lm.kind == 'stop_sign':
                m.type, (m.scale.x, m.scale.y, m.scale.z) = Marker.CYLINDER, (0.75, 0.75, 0.75)
                m.color.r, m.color.g, m.color.b = 1.0, 0.0, 0.0
            else:
                m.type, (m.scale.x, m.scale.y, m.scale.z) = Marker.CUBE, (0.4, 0.4, 1.1)
                m.color.r, m.color.g, m.color.b = 1.0, 0.85, 0.0
            m.color.a = 1.0 if ok else 0.35  # faint until confirmed
            out.markers.append(m)
            label = Marker()
            label.header.frame_id, label.ns, label.id, label.frame_locked = 'enu', lm.kind + '_label', i * 3 + 1, True
            label.type, label.scale.z = Marker.TEXT_VIEW_FACING, 0.7
            label.pose.position.x, label.pose.position.y, label.pose.position.z = float(p[0]), float(p[1]), float(p[2]) + 1.2
            label.pose.orientation.w = 1.0
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f"{lm.kind.replace('_', ' ')} x{len(lm.observations)}" + ('' if ok else ' (unconfirmed)')
            out.markers.append(label)
            if lm.kind == 'stop_sign' and ok:
                (a, b), _ = stop_line(p)
                line = Marker()
                line.header.frame_id, line.ns, line.id, line.frame_locked = 'enu', 'stop_line_estimated', i * 3 + 2, True
                line.type, line.scale.x = Marker.LINE_STRIP, 0.45
                line.pose.orientation.w = 1.0
                line.points = [Point(x=a[0], y=a[1], z=a[2]), Point(x=b[0], y=b[1], z=b[2])]
                line.color.r = line.color.g = line.color.b = line.color.a = 1.0
                out.markers.append(line)
        marker_pub.publish(out)

    cameras = [Camera(name, *cfg) for name, cfg in CAMERAS.items()]
    node.create_subscription(PointCloud2, '/points', on_points, qos_profile_sensor_data)
    node.create_subscription(Odometry, '/camera_lidar_bev/localization/odometry', on_odometry, 50)
    for cam in cameras:
        node.create_subscription(Detection2DArray, cam.topic, lambda m, cam=cam: on_detections(m, cam), 10)
    node.create_timer(0.1, process_pending)
    node.create_timer(1.0, publish)
    node.create_timer(10.0, save)
    node.get_logger().info(f'Mapping stop signs/traffic lights from {", ".join(c.name for c in cameras)}; '
                           f'writing {out_dir / "landmarks.json"}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        save()
        c = confirmed()
        print(f'Saved {len(landmarks)} landmarks ({len(c)} confirmed: '
              f'{sum(l.kind == "stop_sign" for l in c)} stop signs, {sum(l.kind == "traffic_light" for l in c)} traffic lights) '
              f'to {out_dir / "landmarks.json"}; stats {stats}')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
