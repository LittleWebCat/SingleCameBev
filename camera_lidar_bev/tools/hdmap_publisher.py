#!/usr/bin/env python3
"""Publish the aligned HD map (MAP/hdmap/hdmap3d_enu.json, from tools/align_hdmap.py) for RViz.

Every element is a LINE_STRIP in frame enu on /camera_lidar_bev/hdmap/markers (latched, re-sent every 2 s), one
namespace per layer: lane_line, curb, parking_edge, lane_centerline (green with the mapped drive direction, orange
against it, magenta for intersection turn lanes), lane_boundary (off by default: it repeats the lines and kerbs),
crosswalk, stop_line, intersection, in the HD map's own colours. In a view whose fixed frame is os_sensor the map
follows the car through the localization TF enu -> os_sensor (tools/localization_rviz_bridge.py).

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/hdmap_publisher.py [--map maps/local_gps_recording_002] [--layers lane_line curb ...]
"""
import argparse
import json
from pathlib import Path

import rclpy
from geometry_msgs.msg import Point
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(__file__).resolve().parents[1]
WIDTH = {'lane_line': 0.15, 'curb': 0.2, 'parking_edge': 0.1, 'lane_centerline': 0.12, 'lane_boundary': 0.08,
         'crosswalk': 0.15, 'stop_line': 0.35, 'intersection': 0.1}
DEFAULT_LAYERS = ['lane_line', 'curb', 'parking_edge', 'lane_centerline', 'crosswalk', 'stop_line', 'intersection']


def markers(layers, names, lift=0.05):
    out = MarkerArray()
    clear = Marker()
    clear.header.frame_id, clear.action = 'enu', Marker.DELETEALL
    out.markers.append(clear)
    k = 0
    for name in names:
        for e in layers.get(name, []):
            pts = e['points'] + ([e['points'][0]] if e.get('closed') else [])
            if len(pts) < 2:
                continue
            k += 1
            m = Marker()
            m.header.frame_id, m.ns, m.id, m.type, m.frame_locked = 'enu', f'hdmap_{name}', k, Marker.LINE_STRIP, True
            m.pose.orientation.w = 1.0
            m.scale.x = WIDTH.get(name, 0.1)
            r, g, b = (e.get('color') or [255, 255, 255])[:3]
            m.color = ColorRGBA(r=r / 255, g=g / 255, b=b / 255, a=0.9)
            m.points = [Point(x=float(x), y=float(y), z=float(z) + lift) for x, y, z in pts]
            out.markers.append(m)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--layers', nargs='+', default=DEFAULT_LAYERS)
    args = parser.parse_args()
    path = args.map / 'hdmap/hdmap3d_enu.json'
    hd = json.loads(path.read_text())
    msg = markers(hd['layers'], args.layers)
    rclpy.init()
    node = rclpy.create_node('hdmap_publisher')
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(MarkerArray, '/camera_lidar_bev/hdmap/markers', latched)
    pub.publish(msg)
    node.create_timer(2.0, lambda: pub.publish(msg))
    node.get_logger().info(f'HD map {hd.get("version")} ({len(msg.markers) - 1} elements of {args.layers}) from {path}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
