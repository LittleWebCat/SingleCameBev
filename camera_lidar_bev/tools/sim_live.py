#!/usr/bin/env python3
"""Watch simulated episodes of tools/simulate_planner.py live in the 3D view (real time, ROS 2).

Runs the same episodes as the batch simulation (same planner code, traffic, perception and traffic-light models)
one step at a time at wall-clock speed and publishes them as the car's own topics would be, so the normal 3D view,
predictions and charts show them:
  /tf enu -> os_sensor and /camera_lidar_bev/localization/odometry (the simulated car), /camera_lidar_bev/lidar_detections_3d
  (what the simulated perception sees, in os_sensor), /camera_lidar_bev/planner/trajectory and /planner/state (the plan),
  and /camera_lidar_bev/sim/markers: each signal's true colour (big lamp), what the planner believes (small lamp),
  its stop line in the planner's decision colour, the true road users (thin outlines) and an episode status line.
Use it through tools/run_sim_live.sh (own ROS domain 78). Simulation only: nothing commands the vehicle.

  /usr/bin/python3 tools/sim_live.py [--scenarios light_cycle light_yellow ...] [--seed 0] [--episodes 5] [--rate 1.0]
                                     [--max-speed-mps 8.94] [--unknown-light stop] [--light-perfect]
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from std_msgs.msg import ColorRGBA, String
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

import simulate_planner as sim
from planner_core import PlannerConfig

ROOT = Path(__file__).resolve().parents[1]
LAMP = {'red': (1.0, 0.1, 0.1), 'yellow': (1.0, 0.85, 0.0), 'green': (0.1, 0.95, 0.2), 'unknown': (0.55, 0.55, 0.55)}
LIDAR_HEIGHT = 1.75  # m above the road: route z is the LiDAR, boxes sit on the road


def quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--scenarios', nargs='+', default=['light_cycle', 'light_yellow', 'light_red', 'light_red_queue'],
                        help=f'any of {sim.SCENARIOS + sim.LIGHT_SCENARIOS}; played in turn')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=0, help='0: keep going (new seeds) until Ctrl+C')
    parser.add_argument('--duration-s', type=float, default=45.0)
    parser.add_argument('--rate', type=float, default=1.0, help='playback speed (2 = twice as fast)')
    parser.add_argument('--pause-s', type=float, default=3.0, help='between episodes')
    parser.add_argument('--light-perfect', action='store_true')
    parser.add_argument('--latency-s', type=float, default=0.1)
    parser.add_argument('--miss', type=float, default=0.1)
    for f in PlannerConfig.__dataclass_fields__.values():
        if f.type is bool:
            parser.add_argument('--' + f.name.replace('_', '-'), type=lambda v: v.lower() in ('1', 'true', 'yes'), default=f.default)
        elif f.type is str:
            parser.add_argument('--' + f.name.replace('_', '-'), default=f.default)
        else:
            parser.add_argument('--' + f.name.replace('_', '-'), type=float, default=f.default)
    args = parser.parse_args()
    cfg = PlannerConfig.from_args(args)

    rclpy.init()
    node = rclpy.create_node('sim_live')
    log = node.get_logger()
    tf = TransformBroadcaster(node)
    odom_pub = node.create_publisher(Odometry, '/camera_lidar_bev/localization/odometry', 10)
    det_pub = node.create_publisher(Detection3DArray, '/camera_lidar_bev/lidar_detections_3d', 5)
    path_pub = node.create_publisher(PathMsg, '/camera_lidar_bev/planner/trajectory', 5)
    state_pub = node.create_publisher(String, '/camera_lidar_bev/planner/state', 5)
    mark_pub = node.create_publisher(MarkerArray, '/camera_lidar_bev/sim/markers', 5)
    clock = {'wall0': None, 'sim0': 0.0, 'prev': None}

    def publish(st):
        if not rclpy.ok():
            raise KeyboardInterrupt
        t = st['t']
        if clock['wall0'] is None:
            clock['wall0'] = time.monotonic()
        wait = clock['wall0'] + t / args.rate - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        stamp = node.get_clock().now().to_msg()
        route, s, v = st['route'], st['s'], st['v']
        x, y, z, h, _ = (float(q) for q in route.at(s))
        qx, qy, qz, qw = quat(h)
        tr = TransformStamped()
        tr.header.stamp, tr.header.frame_id, tr.child_frame_id = stamp, 'enu', 'os_sensor'
        tr.transform.translation.x, tr.transform.translation.y, tr.transform.translation.z = x, y, z
        tr.transform.rotation.x, tr.transform.rotation.y, tr.transform.rotation.z, tr.transform.rotation.w = qx, qy, qz, qw
        tf.sendTransform(tr)
        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, 'enu', 'os_sensor'
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = x, y, z
        od.pose.pose.orientation.z, od.pose.pose.orientation.w = qz, qw
        od.twist.twist.linear.x = v
        odom_pub.publish(od)

        # what the simulated perception sees, in the car's frame
        c, sn = math.cos(h), math.sin(h)
        arr = Detection3DArray()
        arr.header.stamp, arr.header.frame_id = stamp, 'os_sensor'
        for ox, oy, _, length, width, cls, score, *rest in st['frame']:
            d = Detection3D()
            dx, dy = ox - x, oy - y
            d.bbox.center.position.x, d.bbox.center.position.y = c * dx + sn * dy, -sn * dx + c * dy
            tall = 1.7 if cls == 'person' else 1.5
            d.bbox.center.position.z = -LIDAR_HEIGHT + tall / 2
            _, _, d.bbox.center.orientation.z, d.bbox.center.orientation.w = quat((rest[0] if rest else 0.0) - h)
            d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = length, width, tall
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id, hyp.hypothesis.score = cls, float(score)
            d.results.append(hyp)
            arr.detections.append(d)
        det_pub.publish(arr)

        p, cfg_, planner = st['plan'], st['cfg'], st['planner']
        path = PathMsg()
        path.header.stamp, path.header.frame_id = stamp, 'enu'
        for px, py, pz in zip(p['x'], p['y'], p['z']):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(px), float(py), float(pz)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        path_pub.publish(path)
        nearest = planner.nearest_stop(s)
        lead = st['lead']
        mode = p['reason']
        if planner.conflict:
            mode = 'yield'
        state = {'stamp': stamp.sec + stamp.nanosec * 1e-9, 'mode': mode, 'plan_start': 'measured', 'max_speed': cfg_.max_speed_mps,
                 'measured': {'vx': v, 'vy': 0.0, 'ax': st['a'], 'ay': 0.0, 'yaw_rate': 0.0}, 'imu': None,
                 'route_s': s, 'route_length': route.length, 'lateral_error': 0.0, 'heading_error': 0.0, 'plan_ms': None,
                 'plan': {k: [round(float(q), 3) for q in p[k]] for k in ('t', 'x', 'y', 'v', 'ax', 'ay')},
                 'stop': None if nearest is None else {'id': nearest['id'], 'distance': nearest['stop_s'] - s, 'state': nearest['state']},
                 'lead': None if lead is None else {'gap': lead[0], 'speed': lead[1], 'class': lead[2],
                                                    'desired_gap': cfg_.standstill_gap_m + v * cfg_.time_gap_s}}
        state_pub.publish(String(data=json.dumps(state)))
        mark_pub.publish(markers(st, stamp, x, y, z, h))

    def markers(st, stamp, ex, ey, ez, eh):
        out = MarkerArray()
        clear = Marker()
        clear.header.frame_id, clear.action = 'enu', Marker.DELETEALL
        out.markers.append(clear)
        nid = iter(range(1, 10 ** 6))

        def mk(kind, rgb, alpha=1.0):
            m = Marker()
            m.header.frame_id, m.header.stamp, m.ns, m.id, m.type = 'enu', stamp, 'sim', next(nid), kind
            m.pose.orientation.w = 1.0
            m.color = ColorRGBA(r=rgb[0], g=rgb[1], b=rgb[2], a=alpha)
            out.markers.append(m)
            return m

        def text(pos, words, size, rgb):
            m = mk(Marker.TEXT_VIEW_FACING, rgb)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
            m.scale.x = m.scale.y = 1.0
            m.scale.z = size
            m.text = words
            return m

        route, cfg_, s = st['route'], st['cfg'], st['s']
        road = ez - LIDAR_HEIGHT
        status = [f"{st['name']}  seed {st['seed']}  t {st['t']:.1f} s  v {st['v']:.1f} m/s"]
        for sig in st['signals']:
            true = st['truth'][sig['id']]
            believed = sig['state']
            lx, ly = (float(q) for q in sig['light_xy'])
            pole = mk(Marker.CYLINDER, (0.3, 0.3, 0.3))
            pole.pose.position.x, pole.pose.position.y, pole.pose.position.z = lx, ly, road + 2.5
            pole.scale.x = pole.scale.y = 0.15
            pole.scale.z = 5.0
            lamp = mk(Marker.SPHERE, LAMP[true])
            lamp.pose.position.x, lamp.pose.position.y, lamp.pose.position.z = lx, ly, road + 5.4
            lamp.scale.x = lamp.scale.y = lamp.scale.z = 1.2
            seen = mk(Marker.SPHERE, LAMP.get(believed, LAMP['unknown']))
            seen.pose.position.x, seen.pose.position.y, seen.pose.position.z = lx, ly, road + 4.3
            seen.scale.x = seen.scale.y = seen.scale.z = 0.6
            text((lx, ly, road + 6.6), f'light #{sig["id"]}: {true.upper()}  (planner sees {believed})', 0.6, LAMP[true])
            sx, sy, _, sh, _ = (float(q) for q in route.at(sig['line_s']))
            line = mk(Marker.LINE_STRIP, (1.0, 0.15, 0.15) if sig['target'] else (0.2, 0.9, 0.3), 0.95)
            line.scale.x = 0.45
            nx, ny = -math.sin(sh), math.cos(sh)
            line.points = [Point(x=sx + d * nx, y=sy + d * ny, z=road + 0.05) for d in (-2.2, 2.2)]
            to_line = sig['line_s'] - s - cfg_.lidar_to_front_m
            if -5 < to_line < 120:
                decision = 'STOP' if sig['target'] else 'GO'
                text((sx, sy, road + 1.2), f'{decision}  {to_line:.0f} m', 0.7, (1.0, 0.3, 0.3) if sig['target'] else (0.3, 1.0, 0.4))
                status.append(f"light #{sig['id']} {true} (sees {believed}) {to_line:.0f} m -> {decision}")
        for act in st['actors']:  # the true road users, thin outlines (detections are the filled boxes)
            if not act.present(st['t']):
                continue
            ax_, ay_, ah = act.pose()
            box = mk(Marker.LINE_STRIP, (1.0, 1.0, 1.0), 0.6)
            box.scale.x = 0.05
            box.points = [Point(x=float(px), y=float(py), z=road + 0.1)
                          for px, py in sim.box_corners(ax_, ay_, ah, act.length, act.width)[[0, 1, 2, 3, 0]]]
        m, lm = st['metrics'], st['light_metrics']
        if st['light_sig'] is not None and lm.get('crossed_on'):
            status.append(f"crossed on {lm['crossed_on'].upper()}" + ('  RED-LIGHT RUN' if lm['red_runs'] else ''))
        if m.get('collision'):
            status.append('COLLISION')
        text((ex, ey, ez + 4.2), '\n'.join(status), 0.55, (1.0, 1.0, 1.0))
        return out

    names = args.scenarios
    k = 0
    try:
        while rclpy.ok() and (args.episodes == 0 or k < args.episodes):
            name = names[k % len(names)]
            seed = args.seed * 100000 + k
            opts = {'map': str(args.map), 'cfg': cfg.__dict__, 'duration_s': args.duration_s, 'trace': None, 'dataset': '',
                    'light_perfect': args.light_perfect, 'on_step': publish,
                    'perception': {'latency_s': args.latency_s, 'pos_std': 0.15, 'miss': args.miss, 'burst': 0.02, 'ghosts': 0.0}}
            log.info(f'episode {k + 1}: {name} seed {seed} (max {cfg.max_speed_mps:.1f} m/s, unknown light: {cfg.unknown_light})')
            clock['wall0'] = None
            _, _, result = sim.run_episode((name, seed, opts))
            keys = ('collision', 'red_runs', 'crossed_on', 'yellow_decision', 'stop_gap_m', 'green_wait_s', 'min_gap_m', 'max_decel')
            log.info('  result: ' + ', '.join(f'{q}={result[q]}' for q in keys if q in result))
            time.sleep(args.pause_s)
            k += 1
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
