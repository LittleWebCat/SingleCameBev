#!/usr/bin/env python3
"""Show the road markings of MAP/lane_markings/lane_markings.json (tools/extract_lane_markings.py) in RViz.

Display only. Publishes one MarkerArray, latched (transient local) and republished every --period-s so a
late RViz still gets it, on /camera_lidar_bev/lane_markings/markers, frame enu. Markings lie flat on the
road (triangle ribbons at the json z, lifted --lift-m against z-fighting with the map cloud):
  lane and edge lines white 0.15 m, centre lines yellow 0.3 m (double line), stop lines white 0.45 m bars,
  crosswalk stripes white 0.5 m bars with a thin outline per crosswalk, small labels on stop lines and
  crosswalks (--no-labels hides them). One marker per type (ns lane_markings/<type>, id 0).

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/lane_markings_publisher.py [--map maps/local_gps_recording_002]
  RViz: MarkerArray display on /camera_lidar_bev/lane_markings/markers (durability Transient Local), fixed frame enu
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(__file__).resolve().parents[1]
TOPIC = '/camera_lidar_bev/lane_markings/markers'
WHITE, YELLOW = (0.95, 0.95, 0.95), (1.0, 0.8, 0.1)
STYLE = {'lane_line': (WHITE, 0.15), 'edge_line': (WHITE, 0.15), 'centre_line': (YELLOW, 0.30),
         'stop_line': (WHITE, 0.45), 'crosswalk': (WHITE, 0.50)}  # colour, ribbon width (m)


def ribbon(points, width):
    """Triangles (6 vertices per segment) of a flat ribbon of the given width along an xyz polyline."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return np.empty((0, 3))
    seg = np.diff(p[:, :2], axis=0)
    seg /= np.maximum(np.linalg.norm(seg, axis=1, keepdims=True), 1e-9)
    normals = np.stack([-seg[:, 1], seg[:, 0]], 1)
    vertex = np.vstack([normals[:1], normals[:-1] + normals[1:], normals[-1:]])  # mitre at the joints
    vertex /= np.maximum(np.linalg.norm(vertex, axis=1, keepdims=True), 1e-9)
    scale = 1 / np.maximum(np.einsum('ij,ij->i', vertex, np.vstack([normals, normals[-1:]])), 0.5)
    offset = np.zeros((len(p), 3))
    offset[:, :2] = vertex * (width / 2 * scale)[:, None]
    left, right = p + offset, p - offset
    return np.stack([left[:-1], right[:-1], left[1:], right[:-1], right[1:], left[1:]], 1).reshape(-1, 3)


def marker(ns, kind, rgb, alpha=1.0):
    m = Marker(ns=ns, id=0, type=kind, action=Marker.ADD)
    m.header.frame_id, m.header.stamp = 'enu', Time()
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = 1.0
    m.color = ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(alpha))
    return m


def build(data, lift_m, labels):
    markers = []
    for kind, (rgb, width) in STYLE.items():
        tris = [ribbon(np.array(m['points_enu_m']) + [0, 0, lift_m], width) for m in data['markings'] if m['type'] == kind]
        tris = np.vstack(tris) if tris else np.empty((0, 3))
        if len(tris):
            m = marker(f'lane_markings/{kind}', Marker.TRIANGLE_LIST, rgb)
            m.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in tris]
            markers.append(m)
    outlines = marker('lane_markings/crosswalk_outline', Marker.LINE_LIST, WHITE, 0.7)
    outlines.scale.x = 0.05
    for c in data.get('crosswalks', []):
        corners = np.array(c['outline_enu_m']) + [0, 0, lift_m]
        for a, b in zip(corners, np.roll(corners, -1, axis=0)):
            outlines.points += [Point(x=float(a[0]), y=float(a[1]), z=float(a[2])), Point(x=float(b[0]), y=float(b[1]), z=float(b[2]))]
    if outlines.points:
        markers.append(outlines)
    if labels:
        spots = [('stop line', np.mean(np.array(m['points_enu_m']), 0)) for m in data['markings'] if m['type'] == 'stop_line']
        spots += [('crosswalk', np.mean(np.array(c['outline_enu_m']), 0)) for c in data.get('crosswalks', [])]
        for i, (text, xyz) in enumerate(spots):
            m = marker('lane_markings/labels', Marker.TEXT_VIEW_FACING, WHITE, 0.9)
            m.id, m.text = i, text
            m.scale.x = m.scale.y = m.scale.z = 0.8
            m.pose.position = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2] + 1.0))
            markers.append(m)
    return MarkerArray(markers=markers)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--lift-m', type=float, default=0.03, help='height above the road surface')
    parser.add_argument('--period-s', type=float, default=2.0)
    parser.add_argument('--no-labels', action='store_true')
    args = parser.parse_args()
    data = json.loads((args.map / 'lane_markings/lane_markings.json').read_text())
    if data.get('frame') != 'enu':
        raise SystemExit('lane_markings.json must be in frame enu')
    markers = build(data, args.lift_m, not args.no_labels)

    rclpy.init()
    node = rclpy.create_node('lane_markings_publisher')
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(MarkerArray, TOPIC, latched)
    pub.publish(markers)
    node.create_timer(args.period_s, lambda: pub.publish(markers))
    counts = {k: sum(m['type'] == k for m in data['markings']) for k in STYLE}
    node.get_logger().info(f'{TOPIC}: {len(markers.markers)} markers from {len(data["markings"])} markings {counts}')
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
