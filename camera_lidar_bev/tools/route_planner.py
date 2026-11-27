#!/usr/bin/env python3
"""Plan along the mapped route (navtraj) in real time, stop at stop lines, keep a safe gap, and record driving.

Planning only: publishes the plan for display and logging and never commands the vehicle (no DBW topic).

Each accepted localization pose (/camera_lidar_bev/localization/odometry, ~18 Hz) re-plans from the
measured state:
  * path: the saved route (MAP/gps_alignment/trajectory_enu.csv, LiDAR positions), starting at the
    vehicle and blending onto the route over max(8 m, 2 s of travel);
  * speed: min(--max-speed-mps, sqrt(lateral accel limit / curvature)), reachable with the comfort
    deceleration; a full stop before every kept stop line (MAP/landmarks/landmarks_reviewed.json,
    stop_line_estimated), held --stop-dwell-s once stopped, then released; a stop at the route end;
    traffic lights (kept lights grouped by HD-map intersection, planner_core.load_signals): stop on red, on yellow
    if --yellow-max-decel-mps2 is enough; there is no live colour input yet, so live they are 'unknown' and
    --unknown-light go (default) ignores them (tested in tools/simulate_planner.py light_* scenarios);
  * front vehicle: CenterPoint objects (/camera_lidar_bev/lidar_detections_3d), tracked (constant-velocity
    Kalman filter, kept through dropouts of up to 1 s, confirmed in consecutive frames), inside the lane corridor
    ahead; IDM car following with a standstill gap and a time gap, the lead speed from its track; moving objects
    whose predicted path enters the lane ahead (crossing pedestrians and cars, cut-ins) are yielded to; braking
    harder than --max-decel-mps2 (up to --emergency-decel-mps2) only when needed to stop in time;
  * horizon --horizon-s at 0.1 s: time, position, speed, longitudinal / lateral acceleration.
Speeds and accelerations are in the vehicle frame (x forward, y left) at the rear axle; the planned
vy is 0 (kinematic, no slip) and the planned ay = v^2 * curvature. Measured values come from fits to
localized rear-axle positions (velocity: linear over 0.5 s; acceleration: quadratic over 1 s; both
brought to the newest pose), plus the Ouster IMU (gravity removed with the localized attitude, 0.2 s
low-pass). Each plan starts from the measured speed (shadow mode: you drive, the plan shows what it
would do from here). With --continuity-mps > 0 a plan instead continues the previous plan's speed when
that is within the tolerance of the measurement (for when the plan drives the vehicle, e.g. a closed-loop
simulation); the plan_start column says which. Geometry comes from config/tracking/pure_pursuit.yaml. The decisions are in tools/planner_core.py (ROS-free,
shared with tools/simulate_planner.py).

Publishes /camera_lidar_bev/planner/trajectory (nav_msgs/Path, enu), /planner/markers (speed-coloured
plan, stop target, lead vehicle, status) and /planner/state (JSON for tools/planner_plots.py).
Records data/drive_<UTC>/: drive.csv (one row per cycle: pose, measured and planned kinematics, IMU,
lead, stop line), plans.jsonl (every full plan), metadata.json, and a ros2 bag (mcap) of odometry,
IMU, 3D detections, TF, planner and object-prediction topics (--no-bag to skip, --bag-points to add the raw scan).

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/route_planner.py [--map maps/local_gps_recording_002] [--max-speed-mps 4.4704]
"""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # small per-cycle arrays; threaded BLAS only spins
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import ColorRGBA, String
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from planner_core import PlannerConfig, RoutePlanner, load_signals, load_stops
from route_geometry import Route

ROOT = Path(__file__).resolve().parents[1]
GRAVITY = 9.80665
CSV_FIELDS = [
    'stamp_s', 'x_m', 'y_m', 'yaw_rad', 'route_s_m', 'lateral_error_m', 'heading_error_rad', 'on_route',
    'vx_mps', 'vy_mps', 'ax_mps2', 'ay_mps2', 'yaw_rate_rad_s', 'imu_ax_mps2', 'imu_ay_mps2',
    'plan_mode', 'plan_start', 'plan_v_now_mps', 'plan_ax_now_mps2', 'plan_ay_now_mps2', 'plan_v_1s_mps', 'plan_v_3s_mps',
    'route_curvature_per_m', 'speed_limit_mps', 'stop_id', 'stop_distance_m', 'stop_state',
    'lead_class', 'lead_gap_m', 'lead_speed_mps', 'desired_gap_m', 'plan_ms']


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def quat_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class MotionEstimate:
    """Rear-axle velocity / acceleration in the vehicle frame from fits to recent poses."""

    def __init__(self, velocity_window_s=0.5, accel_window_s=1.0):
        self.v_window, self.window = velocity_window_s, accel_window_s
        self.samples = deque()

    def add(self, t, x, y, yaw):
        if self.samples and (t <= self.samples[-1][0] or t - self.samples[-1][0] > 0.5):
            self.samples.clear()  # out of order or a gap: start over
        self.samples.append((t, x, y, yaw))
        while self.samples and t - self.samples[0][0] > self.window:
            self.samples.popleft()

    def get(self):
        if len(self.samples) < 6 or self.samples[-1][0] - self.samples[0][0] < 0.6 * self.window:
            return None
        a = np.array(self.samples)
        t = a[:, 0] - a[-1, 0]
        ax_e, ay_e = 2 * np.polyfit(t, a[:, 1], 2)[0], 2 * np.polyfit(t, a[:, 2], 2)[0]
        recent = t >= -self.v_window
        tc = t[recent].mean()  # a linear fit's slope is the velocity at the window's mean time
        vx_e = np.polyfit(t[recent], a[recent, 1], 1)[0] - ax_e * tc
        vy_e = np.polyfit(t[recent], a[recent, 2], 1)[0] - ay_e * tc
        yaw = a[-1, 3]
        c, s = math.cos(yaw), math.sin(yaw)
        yaw_rate = np.polyfit(t, np.unwrap(a[:, 3]), 1)[0]
        return {'vx': c * vx_e + s * vy_e, 'vy': -s * vx_e + c * vy_e,
                'ax': c * ax_e + s * ay_e, 'ay': -s * ax_e + c * ay_e, 'yaw_rate': float(yaw_rate)}


def main():
    tracking = yaml.safe_load((ROOT / 'config/tracking/pure_pursuit.yaml').read_text())['path_tracker']['ros__parameters']
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--max-speed-mps', type=float, default=tracking['preview_speed_mps'])
    parser.add_argument('--max-accel-mps2', type=float, default=1.0)
    parser.add_argument('--comfort-decel-mps2', type=float, default=1.5)
    parser.add_argument('--max-decel-mps2', type=float, default=4.0, help='hardest planned braking')
    parser.add_argument('--lateral-accel-mps2', type=float, default=1.5)
    parser.add_argument('--time-gap-s', type=float, default=1.8)
    parser.add_argument('--standstill-gap-m', type=float, default=4.0, help='bumper to bumper when stopped')
    parser.add_argument('--stop-margin-m', type=float, default=0.5, help='front bumper stops this far before the line')
    parser.add_argument('--stop-dwell-s', type=float, default=2.0)
    parser.add_argument('--corridor-half-width-m', type=float, default=1.6, help='lane half-width checked for objects')
    parser.add_argument('--min-detection-score', type=float, default=0.4)
    parser.add_argument('--horizon-s', type=float, default=8.0)
    parser.add_argument('--continuity-mps', type=float, default=0.0,
                        help='continue the previous plan when it is this close to the measured speed (0 = always measured)')
    parser.add_argument('--lidar-to-front-m', type=float, default=2.35, help='measured LiDAR to front bumper')
    parser.add_argument('--emergency-decel-mps2', type=float, default=6.5, help='only when max decel cannot stop in time')
    parser.add_argument('--no-track-objects', dest='track_objects', action='store_false',
                        help='first version: raw detections, no tracking or crossing checks')
    parser.add_argument('--conflict-horizon-s', type=float, default=4.0, help='predicted paths checked for entering the lane')
    parser.add_argument('--unknown-light', choices=('go', 'stop'), default='go',
                        help='traffic light not seen: go (map only; live has no colour input yet) or stop within 30 m')
    parser.add_argument('--yellow-max-decel-mps2', type=float, default=3.0, help='on yellow, stop if this braking is enough')
    parser.add_argument('--out', type=Path, help='default: data/drive_<UTC>')
    parser.add_argument('--no-bag', action='store_true')
    parser.add_argument('--bag-points', action='store_true', help='also bag the raw /points scan (large)')
    args = parser.parse_args()

    route = Route(args.map / 'gps_alignment/trajectory_enu.csv')
    lever = (tracking['lidar_forward_from_rear_axle_m'], tracking['lidar_left_from_rear_axle_m'])
    half_width = 1.791 / 2  # 2025 Nissan LEAF body width without mirrors
    cfg = PlannerConfig.from_args(args)
    stops = load_stops(route, args.map, cfg)
    core = RoutePlanner(route, stops, cfg, load_signals(route, args.map, cfg))
    out = args.out or ROOT / 'data' / f"drive_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=False)
    rows = (out / 'drive.csv').open('w', newline='')
    table = csv.DictWriter(rows, fieldnames=CSV_FIELDS)
    table.writeheader()
    plans = (out / 'plans.jsonl').open('w')
    (out / 'metadata.json').write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'map': str(args.map),
        'route_length_m': round(route.length, 1), 'stop_lines': stops,
        'parameters': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'vehicle': {'lidar_forward_from_rear_axle_m': lever[0], 'lidar_left_from_rear_axle_m': lever[1],
                    'width_m': 2 * half_width, 'lidar_to_front_m': args.lidar_to_front_m},
        'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        'note': 'planning only; no vehicle commands were published'}, indent=2) + '\n')

    rclpy.init()
    node = rclpy.create_node('route_planner')
    log = node.get_logger()
    path_pub = node.create_publisher(PathMsg, '/camera_lidar_bev/planner/trajectory', 5)
    marker_pub = node.create_publisher(MarkerArray, '/camera_lidar_bev/planner/markers', 5)
    state_pub = node.create_publisher(String, '/camera_lidar_bev/planner/state', 5)
    bag = None
    if not args.no_bag:
        topics = ['/camera_lidar_bev/localization/odometry', '/camera_lidar_bev/localization/status', '/imu', '/tf',
                  '/tf_static', '/camera_lidar_bev/lidar_detections_3d', '/camera_lidar_bev/planner/trajectory',
                  '/camera_lidar_bev/planner/state', '/camera_lidar_bev/prediction/objects'] + (['/points'] if args.bag_points else [])
        bag = subprocess.Popen(['ros2', 'bag', 'record', '-s', 'mcap', '-o', str(out / 'bag')] + topics,
                               stdout=(out / 'bag_record.log').open('w'), stderr=subprocess.STDOUT, start_new_session=True)

    state = {'s': None, 'detections': None, 'det_pose': None, 'poses': deque(maxlen=40), 'pose': None, 'imu': None, 'imu_t': None, 'lead': None, 'last_plan': None}
    motion = MotionEstimate()

    def on_imu(m):
        if state['pose'] is None:
            return
        a = np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        # Specific force minus gravity in os_sensor axes (os_imu is a translation of os_sensor on Ouster).
        a = a - state['pose'][0].T @ np.array([0.0, 0.0, GRAVITY])
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if state['imu'] is None or state['imu_t'] is None or not 0 < t - state['imu_t'] < 0.5:
            state['imu'] = a
        else:
            k = min(1.0, (t - state['imu_t']) / 0.2)
            state['imu'] = state['imu'] + k * (a - state['imu'])
        state['imu_t'] = t

    def on_detections(m):
        state['detections'] = m
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        near = min(state['poses'], key=lambda p: abs(p[0] - t), default=None)
        state['det_pose'] = near[1:] if near is not None and abs(near[0] - t) < 0.1 else None

    def lead_vehicle(t, rot, pos, s_ego, v_ego):
        """Closest object in the corridor ahead (or a crossing / cutting-in object to yield to):
        (gap bumper-to-bumper m, speed along route m/s, class). Detections are placed with the pose at their stamp."""
        m = state['detections']
        if m is None:
            return None
        t_det = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if abs(t_det - t) > 0.5:
            objects, t_det = [], t  # stale: no detections this cycle (tracks coast, then expire)
        else:
            rot_d, pos_d = state['det_pose'] or (rot, pos)
            ego_yaw = math.atan2(rot_d[1, 0], rot_d[0, 0])
            objects = []
            for d in m.detections:
                if not d.results:
                    continue
                c, q = d.bbox.center.position, d.bbox.center.orientation
                p = rot_d @ np.array([c.x, c.y, c.z]) + pos_d
                objects.append((p[0], p[1], p[2], max(d.bbox.size.x, d.bbox.size.y), min(d.bbox.size.x, d.bbox.size.y),
                                d.results[0].hypothesis.class_id, d.results[0].hypothesis.score, yaw_of(q) + ego_yaw))
        lead = core.lead_vehicle(t_det, objects, s_ego, now=t, v_ego=v_ego)
        state['lead'] = core.lead_state
        return lead

    def colour(v):
        f = min(1.0, v / max(args.max_speed_mps, 0.1))  # red (stopping) -> yellow -> green (full speed)
        return ColorRGBA(r=float(min(1.0, 2 * (1 - f))), g=float(min(1.0, 2 * f)), b=0.1, a=0.75)  # see the scan through the band

    def publish(header, p, ego, status_text, lead_pose):
        msg = PathMsg()
        msg.header = header
        msg.header.frame_id = 'enu'
        for x, y, z, hh in zip(p['x'], p['y'], p['z'], np.arctan2(np.gradient(p['y']), np.gradient(p['x']) + 1e-9)):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(x), float(y), float(z)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(hh / 2), math.cos(hh / 2)
            msg.poses.append(ps)
        path_pub.publish(msg)
        out_m = MarkerArray()

        def marker(ns, kind):
            m = Marker()
            m.header = msg.header
            m.ns, m.id, m.type = ns, 0, kind
            m.pose.orientation.w = 1.0
            m.color.a = 1.0
            out_m.markers.append(m)
            return m

        line = marker('planned_speed', Marker.LINE_STRIP)
        line.scale.x = 1.8  # about the car's width: a lane-style band, wider than the route line
        line.points = [Point(x=float(x), y=float(y), z=float(z) - 1.5) for x, y, z in zip(p['x'], p['y'], p['z'])]
        line.colors = [colour(v) for v in p['v']]
        stop_s, stop_kind, stop_id = p['stop']
        stop = marker('stop_target', Marker.LINE_STRIP)
        if stop_s - p['s'][0] < 80:
            sx, sy, sz, sh, _ = route.at(stop_s)
            nx, ny = -math.sin(float(sh)), math.cos(float(sh))
            stop.points = [Point(x=float(sx + d * nx), y=float(sy + d * ny), z=float(sz) - 1.5) for d in (-2.0, 2.0)]
            stop.scale.x = 0.3
            stop.color.r, stop.color.g, stop.color.b = 1.0, 0.2, 0.2
        else:
            stop.action = Marker.DELETE
        lead_m = marker('lead_vehicle', Marker.CYLINDER)
        if lead_pose is not None:
            lead_m.pose.position.x, lead_m.pose.position.y, lead_m.pose.position.z = (float(v) for v in lead_pose)
            lead_m.scale.x = lead_m.scale.y = 2.5
            lead_m.scale.z = 0.3
            lead_m.color.r, lead_m.color.g, lead_m.color.b, lead_m.color.a = 1.0, 0.3, 1.0, 0.6
        else:
            lead_m.action = Marker.DELETE
        label = marker('status', Marker.TEXT_VIEW_FACING)
        label.pose.position.x, label.pose.position.y, label.pose.position.z = ego[0], ego[1], ego[2] + 3.0
        label.scale.z = 0.7
        label.color.r = label.color.g = label.color.b = 1.0
        label.text = status_text
        marker_pub.publish(out_m)

    def on_odometry(m):
        if m.child_frame_id != 'os_sensor':
            return
        t0 = time.perf_counter()
        q, pp = m.pose.pose.orientation, m.pose.pose.position
        rot, pos = quat_matrix(q), np.array([pp.x, pp.y, pp.z])
        state['pose'] = (rot, pos)
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        state['poses'].append((t, rot, pos))
        yaw = yaw_of(q)
        rear = (pos[0] - math.cos(yaw) * lever[0] + math.sin(yaw) * lever[1],
                pos[1] - math.sin(yaw) * lever[0] - math.cos(yaw) * lever[1])
        motion.add(t, rear[0], rear[1], yaw)
        meas = motion.get()
        proj = route.project(pos[0], pos[1], yaw, state['s'])
        if proj is None and state['s'] is not None:
            proj = route.project(pos[0], pos[1], yaw)  # lost the previous section: re-acquire anywhere
        row = {'stamp_s': f'{t:.3f}', 'x_m': round(pos[0], 3), 'y_m': round(pos[1], 3), 'yaw_rad': round(yaw, 4),
               'on_route': int(proj is not None)}
        if meas:
            row.update(vx_mps=round(meas['vx'], 3), vy_mps=round(meas['vy'], 3), ax_mps2=round(meas['ax'], 3),
                       ay_mps2=round(meas['ay'], 3), yaw_rate_rad_s=round(meas['yaw_rate'], 4))
        if state['imu'] is not None:
            row.update(imu_ax_mps2=round(float(state['imu'][0]), 3), imu_ay_mps2=round(float(state['imu'][1]), 3))
        report = {'stamp': t, 'measured': meas, 'imu': None if state['imu'] is None else
                  {'ax': float(state['imu'][0]), 'ay': float(state['imu'][1])}}
        if proj is None:
            state['s'] = state['last_plan'] = None
            row['plan_mode'] = 'off_route'
            report.update(mode='off_route', plan=None)
            table.writerow(row)
            state_pub.publish(String(data=json.dumps(report)))
            return
        s_ego, lateral, heading_error = proj
        state['s'] = s_ego
        v_meas = max(0.0, meas['vx']) if meas else 0.0
        v0, plan_start = v_meas, 'measured'
        last = state['last_plan']
        if last is not None and 0 <= t - last[0] <= last[1][-1]:
            v_prev = float(np.interp(t - last[0], last[1], last[2]))
            if args.continuity_mps > 0 and abs(v_prev - v_meas) <= args.continuity_mps:
                v0, plan_start = v_prev, 'continued'
        core.update_stops(s_ego, v_meas, t)
        core.update_signals(s_ego, v_meas, t)  # no colour input live yet: signals stay 'unknown'
        lead = lead_vehicle(t, rot, pos, s_ego, v_meas)
        p = core.plan(s_ego, lateral, v0, lead)
        state['last_plan'] = (t, p['t'], p['v'])
        stop_s, stop_kind, stop_id = p['stop']
        nearest_stop = core.nearest_stop(s_ego)
        signal = min((sg for sg in core.signals if -1.0 < sg['stop_s'] - s_ego < 80), key=lambda sg: sg['stop_s'], default=None)
        mode = p['reason']
        if nearest_stop is not None and nearest_stop['state'] == 'stopping':
            mode = 'stopped_at_stop_line'
        i1, i3 = min(10, len(p['v']) - 1), min(30, len(p['v']) - 1)
        plan_ms = (time.perf_counter() - t0) * 1000
        row.update(route_s_m=round(s_ego, 2), lateral_error_m=round(lateral, 3), heading_error_rad=round(heading_error, 4),
                   plan_mode=mode, plan_start=plan_start, plan_v_now_mps=round(float(p['v'][1]), 3), plan_ax_now_mps2=round(float(p['ax'][0]), 3),
                   plan_ay_now_mps2=round(float(p['ay'][0]), 3), plan_v_1s_mps=round(float(p['v'][i1]), 3),
                   plan_v_3s_mps=round(float(p['v'][i3]), 3), route_curvature_per_m=round(float(p['kappa'][0]), 5),
                   speed_limit_mps=args.max_speed_mps, plan_ms=round(plan_ms, 2))
        if nearest_stop is not None:
            row.update(stop_id=nearest_stop['id'], stop_distance_m=round(nearest_stop['stop_s'] - s_ego, 2),
                       stop_state=nearest_stop['state'])
        if lead is not None:
            gap, v_lead, cls = lead
            row.update(lead_class=cls, lead_gap_m=round(gap, 2), lead_speed_mps=round(v_lead, 2),
                       desired_gap_m=round(args.standstill_gap_m + v_meas * args.time_gap_s, 2))
        table.writerow(row)
        plan_record = {'t': [round(v, 2) for v in p['t']], 'x': [round(float(v), 2) for v in p['x']],
                       'y': [round(float(v), 2) for v in p['y']], 'v': [round(float(v), 3) for v in p['v']],
                       'ax': [round(float(v), 3) for v in p['ax']], 'ay': [round(float(v), 3) for v in p['ay']]}
        plans.write(json.dumps({'stamp': round(t, 3), 'mode': mode, 's': round(s_ego, 2), **plan_record}) + '\n')
        report.update(mode=mode, plan_start=plan_start, route_s=s_ego, route_length=route.length, lateral_error=lateral,
                      heading_error=heading_error, plan=plan_record, plan_ms=plan_ms, max_speed=args.max_speed_mps,
                      stop=None if nearest_stop is None else {'id': nearest_stop['id'], 'distance': nearest_stop['stop_s'] - s_ego,
                                                               'state': nearest_stop['state']},
                      lead=None if lead is None else {'gap': lead[0], 'speed': lead[1], 'class': lead[2],
                                                      'desired_gap': args.standstill_gap_m + v_meas * args.time_gap_s},
                      conflict=core.conflict)
        state_pub.publish(String(data=json.dumps(report)))
        parts = [mode.replace('_', ' ').upper(), f"v {v_meas:.1f} -> {p['v'][i3]:.1f} m/s in 3 s"]
        if nearest_stop is not None and nearest_stop['stop_s'] - s_ego < 60:
            parts.append(f"stop #{nearest_stop['id']} {nearest_stop['stop_s'] - s_ego:.0f} m ({nearest_stop['state']})")
        if signal is not None:
            parts.append(f"light #{signal['id']} {signal['stop_s'] - s_ego:.0f} m ({signal['state']}{', stopping' if signal['target'] else ''})")
        if core.conflict is not None:
            parts.append(f"yield: {core.conflict['class']} entering lane in {core.conflict['enters_in_s']:.1f} s")
        elif lead is not None:
            parts.append(f'{lead[2]} {lead[0]:.0f} m @ {lead[1]:.1f} m/s')
        publish(m.header, p, pos, ' | '.join(parts), None if state['lead'] is None else state['lead']['p'])

    node.create_subscription(Odometry, '/camera_lidar_bev/localization/odometry', on_odometry, 10)
    node.create_subscription(Detection3DArray, '/camera_lidar_bev/lidar_detections_3d', on_detections, 10)
    node.create_subscription(Imu, '/imu', on_imu, qos_profile_sensor_data)
    log.info(f'Route {route.length:.0f} m, {len(stops)} stop lines {[s["id"] for s in stops]}, '
             f'max {args.max_speed_mps:.2f} m/s; planning only (no vehicle commands); recording {out}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rows.close()
        plans.close()
        if bag is not None:
            os.killpg(bag.pid, signal.SIGINT)
            try:
                bag.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(bag.pid, signal.SIGTERM)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(f'Recorded {out}', flush=True)


if __name__ == '__main__':
    main()
