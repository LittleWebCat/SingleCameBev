#!/usr/bin/env python3
"""LiDAR-only lane lines, road boundaries and lane centre, from each single Ouster scan (no map).

Per /points scan (os_lidar or os_sensor):
  * road plane: least-squares plane z = a + b x + c y through low points within 40 m, refitted twice on its
    inliers (follows the car's pitch / roll and road slope);
  * lane paint: road points (|height| < 0.10 m) whose reflectivity is well above the road around them at the
    same range (median + max(4 MAD, --paint-contrast) and >= 1.6 x median, per 1 m range bin);
  * road boundary: on a 0.25 m grid, walk left and right from the car along each 0.5 m row until a raised cell
    (kerb or obstacle, 0.08-2.5 m above the plane) or a 1.5 m gap with no road returns;
  * lines, ego lane and lane centre: tools/road_features.py, identical for the camera method.
Publishes /camera_lidar_bev/lanes/lidar (JSON, os_sensor, x forward / y left) and .../lidar_markers.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/lane_lidar.py [--paint-contrast 12]
"""
import argparse
import os
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

import road_features as rf

SENSOR_FROM_LIDAR = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.038195], [0, 0, 0, 1]], float)
DTYPES = {1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8'}
STYLE = {'lane_line': (1.0, 0.55, 0.0, 0.25), 'road_boundary': (1.0, 0.15, 0.15, 0.3), 'centre': (0.2, 1.0, 0.3, 0.35)}  # LiDAR: orange / red
CAMERA_STYLE = {'lane_line': (0.2, 0.6, 1.0, 0.25), 'road_boundary': (0.6, 0.3, 1.0, 0.3), 'centre': (0.2, 1.0, 1.0, 0.35)}  # camera: blue / violet


def cloud_arrays(m, intensity_field):
    """xyz in os_sensor and intensity of the valid points of a PointCloud2."""
    f = {x.name: x for x in m.fields}
    raw = np.frombuffer(m.data, np.uint8).reshape(m.height, m.row_step)[:, :m.width * m.point_step].reshape(-1, m.point_step)
    col = lambda name: raw[:, f[name].offset:f[name].offset + np.dtype(DTYPES[f[name].datatype]).itemsize].copy().view(  # noqa: E731
        DTYPES[f[name].datatype])[:, 0].astype(np.float64)
    xyz = np.stack([col('x'), col('y'), col('z')], 1)
    inten = col(intensity_field if intensity_field in f else 'intensity')
    ok = np.isfinite(xyz).all(1) & (np.einsum('ij,ij->i', xyz, xyz) > 1.0)
    xyz, inten = xyz[ok], inten[ok]
    if m.header.frame_id == 'os_lidar':
        xyz = xyz @ SENSOR_FROM_LIDAR[:3, :3].T + SENSOR_FROM_LIDAR[:3, 3]
    return xyz, inten


def road_plane(xyz, sensor_height_m=1.9):
    """Coefficients (a, b, c) of z = a + b x + c y, or None."""
    low = xyz[(xyz[:, 0] > 2) & (xyz[:, 0] < 40) & (np.abs(xyz[:, 1]) < 15) & (xyz[:, 2] < -sensor_height_m + 0.6)]
    if len(low) < 200:
        return None
    sel = low
    coef = None
    for tol in (0.3, 0.15, 0.08):
        a = np.c_[np.ones(len(sel)), sel[:, 0], sel[:, 1]]
        coef = np.linalg.lstsq(a, sel[:, 2], rcond=None)[0]
        res = low[:, 2] - (coef[0] + coef[1] * low[:, 0] + coef[2] * low[:, 1])
        sel = low[np.abs(res) < tol]
        if len(sel) < 100:
            return None
    return coef


def detect(xyz, inten, paint_contrast=12.0, cell=0.25):
    """(lane points (N, 2), boundary points (M, 2), road-plane height at the car)."""
    coef = road_plane(xyz)
    if coef is None:
        return np.zeros((0, 2)), np.zeros((0, 2)), None
    roi = (xyz[:, 0] > 1.0) & (xyz[:, 0] < 40) & (np.abs(xyz[:, 1]) < 15)
    p, i = xyz[roi], inten[roi]
    h = p[:, 2] - (coef[0] + coef[1] * p[:, 0] + coef[2] * p[:, 1])
    ground = np.abs(h) < 0.10
    g, gi = p[ground], i[ground]
    # Paint: bright relative to the road at the same range.
    rng = np.hypot(g[:, 0], g[:, 1])
    b = np.minimum(rng.astype(int), 39)
    lane = np.zeros(len(g), bool)
    for k in np.unique(b):
        s = b == k
        if s.sum() < 30:
            continue
        med = np.median(gi[s])
        mad = np.median(np.abs(gi[s] - med)) + 1e-6
        lane[s] = (gi[s] > med + max(4 * mad, paint_contrast)) & (gi[s] >= 1.6 * med)
    # Boundary: walk out from the car on a grid until raised cells or a long gap.
    nx, ny = int(39 / cell), int(30 / cell)
    ix = ((p[:, 0] - 1.0) / cell).astype(int)
    iy = ((p[:, 1] + 15.0) / cell).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    road = np.zeros((nx, ny), bool)
    raised = np.zeros((nx, ny), np.int16)
    road[ix[ok & ground], iy[ok & ground]] = True
    up = ok & (h > 0.08) & (h < 2.5)
    np.add.at(raised, (ix[up], iy[up]), 1)
    blocked = raised >= 2
    centre, gap_cells = int(15.0 / cell), int(1.5 / cell)
    edges = []
    for r in range(0, nx, 2):  # every 0.5 m
        rows = slice(r, min(r + 2, nx))
        rr, bb = road[rows].any(0), blocked[rows].any(0)
        if not rr[centre - 2:centre + 3].any():
            continue
        for step in (1, -1):
            last, gap, c = None, 0, centre
            while 0 <= c < ny:
                if bb[c]:
                    break
                if rr[c]:
                    last, gap = c, 0
                else:
                    gap += 1
                    if gap > gap_cells:
                        break
                c += step
            if last is not None and 0 < c < ny - 1:
                edges.append((1.0 + (r + 1) * cell, (last + 0.5) * cell - 15.0 + step * cell / 2))
    return g[lane, :2], np.array(edges).reshape(-1, 2), coef[0]


def markers(msg, frame, stamp, style=STYLE):
    out = MarkerArray()
    clear = Marker()
    clear.header.frame_id, clear.header.stamp, clear.action = frame, stamp, Marker.DELETEALL
    out.markers.append(clear)
    k = 0
    z0 = msg['road_z'] if msg['road_z'] is not None else -1.9
    groups = [('lane_line', l, z0) for l in msg['lines']] + [('road_boundary', l, z0) for l in msg['boundaries']]
    if msg['lane_centre']:
        groups.append(('centre', {'points': msg['lane_centre']}, z0))
    for kind, line, z in groups:
        k += 1
        m = Marker()
        m.header.frame_id, m.header.stamp, m.ns, m.id, m.type = frame, stamp, kind, k, Marker.LINE_STRIP
        m.pose.orientation.w = 1.0
        r, g, b, w = style[kind]
        m.scale.x = w
        m.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
        m.points = [Point(x=float(x), y=float(y), z=float(z) + 0.05) for x, y in line['points']]
        out.markers.append(m)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--points-topic', default='/points')
    parser.add_argument('--intensity-field', default='reflectivity')
    parser.add_argument('--paint-contrast', type=float, default=12.0, help='minimum reflectivity above the local road')
    parser.add_argument('--method', default='lidar', help='output topic name: /camera_lidar_bev/lanes/<method>')
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node(f'lanes_{args.method}')
    pub = node.create_publisher(String, f'/camera_lidar_bev/lanes/{args.method}', 10)
    mpub = node.create_publisher(MarkerArray, f'/camera_lidar_bev/lanes/{args.method}_markers', 5)
    widths = []

    def on_points(m):
        t0 = time.perf_counter()
        xyz, inten = cloud_arrays(m, args.intensity_field)
        lane_pts, edge_pts, road_z = detect(xyz, inten, args.paint_contrast)
        lines = rf.fit_lines(lane_pts, 'lane_line')
        bounds = rf.fit_lines(edge_pts, 'road_boundary', link_m=1.2)
        width = float(np.median(widths)) if widths else 3.5
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        compute_ms = (time.perf_counter() - t0) * 1000
        latency = (node.get_clock().now().nanoseconds * 1e-9 - stamp) * 1000
        text = rf.to_message(stamp, args.method, lines, bounds, width, latency,
                             {'compute_ms': round(compute_ms, 1), 'road_z': None if road_z is None else round(float(road_z), 3),
                              'lane_points': int(len(lane_pts)), 'boundary_points': int(len(edge_pts))})
        msg = rf.from_message(text)
        if msg['lane_width_m'] and 2.5 < msg['lane_width_m'] < 5.0:
            widths.append(msg['lane_width_m'])
            del widths[:-50]
        pub.publish(String(data=text))
        if road_z is not None:
            mpub.publish(markers(msg, 'os_sensor', m.header.stamp))

    node.create_subscription(PointCloud2, args.points_topic, on_points, qos_profile_sensor_data)
    node.get_logger().info(f'LiDAR lanes from {args.points_topic} -> /camera_lidar_bev/lanes/{args.method}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
