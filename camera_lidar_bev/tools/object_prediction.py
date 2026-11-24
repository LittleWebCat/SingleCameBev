#!/usr/bin/env python3
"""Track CenterPoint objects on the map and predict where moving vehicles, cyclists and pedestrians go next.

Display and logging only; nothing here commands the vehicle.

Each /camera_lidar_bev/lidar_detections_3d message (os_sensor) is placed in enu with the localization
pose of the same scan (/camera_lidar_bev/localization/odometry, stamps within 30 ms), so the car's own
motion is removed. Tracking and prediction come from tools/prediction_models.py, the same code that
tools/eval_prediction.py scores on recorded drives:
  * --model imm (default): vehicles use a straight + turning IMM filter with CenterPoint's box heading;
    cyclists and pedestrians a constant-velocity Kalman filter. --model cv is the first version.
  * --lanes: also predict vehicles along the mapped route when they drive along it (helped turns,
    not straight driving, in the benchmark, so it is off by default).
  * tracks slower than 1.5 (vehicle) / 1.0 (cyclist) / 0.5 (pedestrian) m/s are shown as parked / still;
  * horizons --vehicle-horizon-s 5 (vehicles, cyclists) and --pedestrian-horizon-s 4.
Publishes /camera_lidar_bev/prediction/markers (enu: fading predicted path, 2-sigma discs every second,
translucent box at the horizon, "#id speed" label) and /camera_lidar_bev/prediction/objects (JSON: tracks
with id, class, position, velocity and the predicted points) for recording and later planner use.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/object_prediction.py [--model imm] [--lanes]
"""
import argparse
from collections import deque
import itertools
import json
import math
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # small filters; threaded BLAS only spins
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from std_msgs.msg import ColorRGBA, String
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from prediction_models import RouteLanes, Tracker, detections_in_enu, quat_matrix

ROOT = Path(__file__).resolve().parents[1]
# Matches tools/bev_scene_markers.py: vehicles violet, pedestrians amber, cyclists pink.
COLOURS = {'vehicle': (0.62, 0.45, 1.0), 'pedestrian': (1.0, 0.72, 0.1), 'cyclist': (1.0, 0.35, 0.75)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', choices=('imm', 'cv'), default='imm')
    parser.add_argument('--lanes', action='store_true', help='vehicles driving along the mapped route follow it')
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--vehicle-horizon-s', type=float, default=5.0, help='vehicles and cyclists')
    parser.add_argument('--pedestrian-horizon-s', type=float, default=4.0)
    parser.add_argument('--min-score', type=float, default=0.35)
    parser.add_argument('--max-range-m', type=float, default=80.0)
    args = parser.parse_args()

    lanes = None
    if args.lanes:
        from route_geometry import Route
        lanes = RouteLanes(Route(args.map / 'gps_alignment/trajectory_enu.csv'))
    tracker = Tracker(args.model, lanes=lanes)
    rclpy.init()
    node = rclpy.create_node('object_prediction')
    marker_pub = node.create_publisher(MarkerArray, '/camera_lidar_bev/prediction/markers', 5)
    objects_pub = node.create_publisher(String, '/camera_lidar_bev/prediction/objects', 5)
    poses = deque(maxlen=100)  # (stamp ns, R, p) from localization

    def on_odometry(m):
        if m.child_frame_id == 'os_sensor':
            p = m.pose.pose.position
            poses.append((m.header.stamp.sec * 10**9 + m.header.stamp.nanosec, quat_matrix(m.pose.pose.orientation),
                          np.array([p.x, p.y, p.z])))

    def on_detections(m):
        stamp = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
        best = min(poses, key=lambda p: abs(p[0] - stamp), default=None)
        if best is None or abs(best[0] - stamp) >= 30_000_000:
            return  # no localization for this scan: cannot remove ego motion
        t = stamp * 1e-9
        tracker.step(t, detections_in_enu(m, best[1], best[2], args.min_score, args.max_range_m))
        publish(m.header.stamp, t)

    def publish(stamp, t):
        out = MarkerArray()
        clear = Marker()
        clear.header.frame_id, clear.header.stamp, clear.ns, clear.action = 'enu', stamp, 'prediction', Marker.DELETEALL
        out.markers.append(clear)
        report = []
        mid = itertools.count(1)  # id 0 is the DELETEALL marker; RViz rejects duplicate ids

        def marker(kind):
            mk = Marker()
            mk.header.frame_id, mk.header.stamp = 'enu', stamp
            mk.ns, mk.id, mk.type = 'prediction', next(mid), kind
            mk.pose.orientation.w = 1.0
            out.markers.append(mk)
            return mk

        for tr in tracker.confirmed(t):
            horizon = args.pedestrian_horizon_s if tr.group == 'pedestrian' else args.vehicle_horizon_s
            moving, path = tracker.predict_track(tr, horizon)
            (x0, y0), (vx, vy) = tr.position(), tr.velocity()
            speed = math.hypot(vx, vy)
            report.append({'id': tr.id, 'class': tr.cls, 'group': tr.group, 'x': round(float(x0), 2), 'y': round(float(y0), 2),
                           'vx': round(float(vx), 2), 'vy': round(float(vy), 2), 'moving': moving,
                           'size': [round(v, 2) for v in tr.size],
                           'predicted': [[round(a, 1), round(float(b), 2), round(float(c), 2), round(r, 2)]
                                         for a, b, c, _, r in path[::5]] if moving else []})
            r, g, b = COLOURS[tr.group]
            ground = tr.z - tr.size[2] / 2 + 0.15
            label = marker(Marker.TEXT_VIEW_FACING)
            label.pose.position.x, label.pose.position.y, label.pose.position.z = float(x0), float(y0), tr.z + tr.size[2] / 2 + 1.0
            label.scale.x = label.scale.y = 1.0  # RViz rejects a zero x/y scale even for text
            label.scale.z = 0.5
            label.color.r, label.color.g, label.color.b, label.color.a = r, g, b, 1.0
            label.text = f'#{tr.id} {speed:.1f} m/s' if moving else f'#{tr.id} parked' if tr.group == 'vehicle' else f'#{tr.id} still'
            if not moving:
                continue
            line = marker(Marker.LINE_STRIP)
            line.scale.x = 0.25 if tr.group == 'pedestrian' else 0.4
            line.color.r, line.color.g, line.color.b, line.color.a = r, g, b, 1.0  # per-point colours below fade it
            for ft, x, y, _, _ in path:
                line.points.append(Point(x=float(x), y=float(y), z=ground))
                line.colors.append(ColorRGBA(r=r, g=g, b=b, a=float(1.0 - 0.75 * ft / horizon)))  # fades with time
            for ft, x, y, _, radius in path[10::10]:  # every second: 2-sigma uncertainty disc
                disc = marker(Marker.CYLINDER)
                disc.pose.position.x, disc.pose.position.y, disc.pose.position.z = float(x), float(y), ground - 0.05
                disc.scale.x = disc.scale.y = max(2 * radius, 0.3)
                disc.scale.z = 0.03
                disc.color.r, disc.color.g, disc.color.b, disc.color.a = r, g, b, float(max(0.05, 0.22 - 0.03 * ft))
            _, x, y, heading, _ = path[-1]
            ghost = marker(Marker.CUBE)  # where it is predicted to be at the horizon
            ghost.pose.position.x, ghost.pose.position.y, ghost.pose.position.z = float(x), float(y), tr.z
            ghost.pose.orientation.z, ghost.pose.orientation.w = math.sin(heading / 2), math.cos(heading / 2)
            length, width = max(tr.size[:2]), min(tr.size[:2])
            ghost.scale.x, ghost.scale.y, ghost.scale.z = length, width, tr.size[2]
            ghost.color.r, ghost.color.g, ghost.color.b, ghost.color.a = r, g, b, 0.18
        marker_pub.publish(out)
        objects_pub.publish(String(data=json.dumps({'stamp': round(t, 3), 'model': args.model + ('+lanes' if lanes else ''),
                                                    'tracks': report})))

    node.create_subscription(Odometry, '/camera_lidar_bev/localization/odometry', on_odometry, 50)
    node.create_subscription(Detection3DArray, '/camera_lidar_bev/lidar_detections_3d', on_detections, 10)
    node.get_logger().info(f'Predicting with {args.model}{" + lane following" if lanes else ""}: vehicles / cyclists '
                           f'{args.vehicle_horizon_s:g} s, pedestrians {args.pedestrian_horizon_s:g} s ahead (display and logging only)')
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
