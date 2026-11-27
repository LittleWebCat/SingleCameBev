#!/usr/bin/env python3
"""Combine per-camera lane results into camera combinations (light; the cameras run in their own processes).

Subscribes to /camera_lidar_bev/lanes/camera_<name>/points from one tools/lane_camera.py process per camera: its road
points (lane_xy, edge_xy) in os_sensor, image stamp and compute time. Whenever a camera result arrives, every combination whose cameras all
have a result from the last --max-age-s is rebuilt from the pooled points with tools/road_features.py and published
on /camera_lidar_bev/lanes/camera_<combination> (latency from the oldest image used; compute_ms = the members'
compute + this fit, i.e. what the combination costs), plus /camera_lidar_bev/lanes/camera_<first combination>_markers.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/fuse_camera_lanes.py [--combo narrow_wide=front_narrow,front_wide ...]
"""
import argparse
import os
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import rclpy
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray

import road_features as rf
from lane_lidar import CAMERA_STYLE, markers

DEFAULT = ['narrow_wide=front_narrow,front_wide']


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--combo', action='append', help='NAME=camera,camera,...; default: ' + ' '.join(DEFAULT))
    parser.add_argument('--max-age-s', type=float, default=0.1, help='member results must be this close in time')
    args = parser.parse_args()
    combos = {k: v.split(',') for k, v in (c.split('=', 1) for c in (args.combo or DEFAULT))}
    cameras = sorted({c for v in combos.values() for c in v})
    rclpy.init()
    node = rclpy.create_node('fuse_camera_lanes')
    pubs = {k: node.create_publisher(String, f'/camera_lidar_bev/lanes/camera_{k}', 10) for k in combos}
    first = next(iter(combos))
    mpub = node.create_publisher(MarkerArray, f'/camera_lidar_bev/lanes/camera_{first}_markers', 5)
    latest, widths = {}, {k: [] for k in combos}

    def on_camera(camera, text):
        latest[camera] = rf.from_message(text)
        newest = latest[camera]['stamp']
        for k, members in combos.items():
            if camera not in members or not all(c in latest and newest - latest[c]['stamp'] <= args.max_age_s for c in members):
                continue
            t0 = time.perf_counter()
            res = [latest[c] for c in members]
            lane = np.array([p for r in res for p in r['lane_xy']]).reshape(-1, 2)
            edge = np.array([p for r in res for p in r['edge_xy']]).reshape(-1, 2)
            lines = rf.fit_lines(lane, 'lane_line')
            bounds = rf.fit_lines(edge, 'road_boundary', link_m=1.2)
            oldest = min(r['stamp'] for r in res)
            w = widths[k]
            text = rf.to_message(newest, f'camera_{k}', lines, bounds, float(np.median(w)) if w else 3.5,
                                 (node.get_clock().now().nanoseconds * 1e-9 - oldest) * 1000,
                                 {'compute_ms': round(sum(r['compute_ms'] for r in res) + (time.perf_counter() - t0) * 1000, 1),
                                  'road_z': res[0]['road_z'], 'cameras': members,
                                  'image_spread_ms': round((newest - oldest) * 1000, 1)})
            msg = rf.from_message(text)
            if msg['lane_width_m'] and 2.5 < msg['lane_width_m'] < 5.0:
                w.append(msg['lane_width_m'])
                del w[:-50]
            pubs[k].publish(String(data=text))
            if k == first:
                stamp = node.get_clock().now().to_msg()
                stamp.sec, stamp.nanosec = int(newest), int((newest % 1) * 1e9)
                mpub.publish(markers(msg, 'os_sensor', stamp, CAMERA_STYLE))

    for c in cameras:
        node.create_subscription(String, f'/camera_lidar_bev/lanes/camera_{c}/points', lambda m, c=c: on_camera(c, m.data), 1)  # newest only
    node.get_logger().info(f'Combining {combos}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
