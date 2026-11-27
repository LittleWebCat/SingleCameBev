#!/usr/bin/env python3
"""Clean 3D planning scene for RViz: detected objects, kept stop signs and stop lines, mapped route and plan.

Display only (never commands the vehicle). Publishes two MarkerArrays for config/rviz/bev_planning_3d.rviz:
  * /camera_lidar_bev/bev_scene/static_markers (latched, frame enu, frame-locked; republished every 2 s):
    the mapped route (navtraj) as a thin centre line at road level, every kept stop
    sign of MAP/landmarks/landmarks_reviewed.json as a red octagon on a pole facing the approaching
    lane with its estimated stop line as a white bar on the road (traffic lights are never drawn), and a
    translucent body for the own vehicle (os_sensor, from the detector's self-mask envelope).
  * /camera_lidar_bev/bev_scene/markers (rebuilt with DELETEALL on every detection message, and at
    least every 0.15 s otherwise):
      - objects of /camera_lidar_bev/lidar_detections_3d (os_sensor) as translucent boxes with edges and a
        heading tick, coloured by class: vehicles violet, pedestrians amber, cyclists / motorcycles pink,
        others grey; label "class distance". Objects within the planned corridor (via the localization
        pose) are drawn brighter and labelled "in path" (or "LEAD" when the planner reports a lead);
      - the planned trajectory (/camera_lidar_bev/planner/trajectory) as a lane-width ribbon coloured
        by planned speed (green full speed -> yellow -> red stopping, speeds from
        /camera_lidar_bev/planner/state), white ticks every second;
      - a red stop wall on the planner's active stop line (distance and state) and a one-line planner
        status behind the vehicle.
Route and plan poses are at LiDAR height; --ground-offset-m (-1.7) puts them on the road.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/bev_scene_markers.py [--map maps/local_gps_recording_002]
  rviz2 -d config/rviz/bev_planning_3d.rviz
"""
import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # tiny arrays per message; threaded BLAS only spins
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.duration import Duration
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, String
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(__file__).resolve().parents[1]
VEHICLES = {'car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'van'}
PEDESTRIANS = {'person', 'pedestrian'}
CYCLISTS = {'bicycle', 'motorcycle', 'cyclist'}
COLOURS = {'vehicle': (0.66, 0.50, 1.0), 'pedestrian': (1.0, 0.78, 0.12), 'cyclist': (1.0, 0.36, 0.78),
           'other': (0.7, 0.7, 0.7)}
EGO_RGB = (0.35, 0.75, 1.0)


def category(name):
    return ('vehicle' if name in VEHICLES else 'pedestrian' if name in PEDESTRIANS else
            'cyclist' if name in CYCLISTS else 'other')


def rgba(rgb, a=1.0):
    return ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(a))


def pt(x, y, z):
    return Point(x=float(x), y=float(y), z=float(z))


def speed_rgb(v, v_max):
    f = min(1.0, max(0.0, v / max(v_max, 0.1)))  # red (stopping) -> yellow -> green (full speed), as the planner
    return (min(1.0, 2 * (1 - f)), min(1.0, 2 * f) * 0.9 + 0.1, 0.15)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def headings(xy, start=None):
    """Heading of each polyline point; short segments keep the previous heading (stopped plans)."""
    h, last, first = np.zeros(len(xy)), start, None
    for i in range(len(xy)):
        d = xy[min(i + 1, len(xy) - 1)] - xy[max(i - 1, 0)]
        if np.hypot(*d) > 0.02:
            last = math.atan2(d[1], d[0])
            first = i if first is None else first
        h[i] = last if last is not None else 0.0
    if start is None and first:
        h[:first] = h[first]
    return h


def ribbon(m, xyz, h, width, colours):
    """Fill `m` (TRIANGLE_LIST) with a strip of `width` along xyz, one colour per centre point."""
    n = np.stack([-np.sin(h), np.cos(h), np.zeros_like(h)], 1) * width / 2
    left, right = xyz + n, xyz - n
    for i in range(len(xyz) - 1):
        a, b, c, d = left[i], right[i], left[i + 1], right[i + 1]
        m.points += [pt(*a), pt(*b), pt(*c), pt(*c), pt(*b), pt(*d)]
        ci, cj = colours[i], colours[i + 1]
        m.colors += [ci, ci, cj, cj, ci, cj]


class Scene:
    def __init__(self, frame, stamp):
        self.out, self.frame, self.stamp, self.ids = MarkerArray(), frame, stamp, {}

    def add(self, ns, kind, rgb=(1, 1, 1), a=1.0, frame=None, stamp=None, locked=False):
        m = Marker()
        m.header.frame_id, m.header.stamp = frame or self.frame, stamp or self.stamp
        m.ns, m.type, m.frame_locked = ns, kind, locked
        m.id = self.ids[ns] = self.ids.get(ns, -1) + 1
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 1.0
        m.color = rgba(rgb, a)
        self.out.markers.append(m)
        return m

    def text(self, ns, xyz, text, size, rgb=(1, 1, 1), **kw):
        m = self.add(ns, Marker.TEXT_VIEW_FACING, rgb, **kw)
        m.pose.position = pt(*xyz)
        m.scale.z, m.text = size, text
        return m


def static_scene(route, stops, ego, args):
    s = Scene('enu', rclpy.time.Time().to_msg())
    lift = args.ground_offset_m
    road = route + [0, 0, lift + 0.03]
    h = headings(road[:, :2])
    line = s.add('route', Marker.LINE_STRIP, (0.62, 0.72, 0.88), 0.8, locked=True)
    line.scale.x = args.route_width_m
    line.points = [pt(x, y, z + 0.03) for x, y, z in road]
    for st in stops:
        x, y, z = st['position_enu_m']
        sl = st.get('stop_line_estimated')
        ends = np.array(sl['endpoints_enu_m'], float) if sl else None
        i = int(np.argmin(np.hypot(*(route[:, :2] - (ends.mean(0)[:2] if sl else (x, y))).T)))
        travel = h[i]
        ground = float(ends[:, 2].mean()) if sl else route[i, 2] + lift
        r = args.sign_radius_m
        top = z - r  # pole up to the bottom of the octagon
        pole = s.add('stop_sign', Marker.CYLINDER, (0.75, 0.77, 0.80), locked=True)
        pole.pose.position = pt(x, y, (ground + top) / 2)
        pole.scale.x = pole.scale.y = 0.1
        pole.scale.z = max(0.1, top - ground)
        face = np.array([-math.cos(travel), -math.sin(travel), 0.0])  # toward the approaching vehicle
        u, w = np.array([-face[1], face[0], 0.0]), np.array([0.0, 0.0, 1.0])
        centre = np.array([x, y, z])
        for rad, rgb, off in ((r * 1.1, (1, 1, 1), -0.03), (r, (0.85, 0.05, 0.05), 0.0)):
            octagon = s.add('stop_sign', Marker.TRIANGLE_LIST, rgb, locked=True)
            c = centre + face * off
            corners = [c + rad * (math.cos(t) * u + math.sin(t) * w) for t in np.radians(22.5 + 45 * np.arange(9))]
            for k in range(8):
                octagon.points += [pt(*c), pt(*corners[k]), pt(*corners[k + 1])]
        s.text('stop_sign', centre + face * 0.08, 'STOP', r * 0.55, locked=True)
        if sl:
            bar = s.add('stop_line', Marker.CUBE, (0.95, 0.95, 0.95), 0.95, locked=True)
            mid, d = ends.mean(0), ends[1] - ends[0]
            yaw = math.atan2(d[1], d[0])
            bar.pose.position = pt(mid[0], mid[1], mid[2] + 0.04)
            bar.pose.orientation.z, bar.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            bar.scale.x, bar.scale.y, bar.scale.z = float(np.hypot(*d[:2])), args.stop_line_width_m, 0.04
    if ego is not None:
        (x0, y0, z0), (x1, y1, z1) = ego
        body = s.add('ego_body', Marker.CUBE, EGO_RGB, 0.28, frame='os_sensor', locked=True)
        body.pose.position = pt((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
        body.scale.x, body.scale.y, body.scale.z = x1 - x0, y1 - y0, z1 - z0
    return s.out


def ego_envelope(path):
    try:
        p = next(iter(yaml.safe_load(path.read_text()).values()))['ros__parameters']
        return tuple(tuple(float(p[f'self_mask_{e}_{a}_m']) for a in 'xyz') for e in ('min', 'max'))
    except (OSError, KeyError, StopIteration, AttributeError, TypeError, ValueError):
        return (-2.131, -0.905, -1.8), (2.35, 0.885, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--landmarks', type=Path, help='default: MAP/landmarks/landmarks_reviewed.json')
    parser.add_argument('--detector-config', type=Path, default=ROOT / 'config/perception/lidar_vehicle_detector.yaml',
                        help='own-vehicle body from its self_mask_* envelope')
    parser.add_argument('--ground-offset-m', type=float, default=-1.7, help='route/plan pose height -> road')
    parser.add_argument('--route-width-m', type=float, default=0.12, help='mapped route centre line')
    parser.add_argument('--plan-width-m', type=float, default=2.6, help='planned trajectory ribbon')
    parser.add_argument('--plan-dt-s', type=float, default=0.1, help='plan sample time when no planner state')
    parser.add_argument('--max-speed-mps', type=float, default=4.4704, help='speed colour scale (planner state wins)')
    parser.add_argument('--sign-radius-m', type=float, default=0.55, help='octagon size (real signs are ~0.38)')
    parser.add_argument('--stop-line-width-m', type=float, default=0.5)
    parser.add_argument('--corridor-m', type=float, default=1.25,
                        help='in path: object centre within this plus its half width of the planned centre line')
    parser.add_argument('--stale-s', type=float, default=0.5)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node('bev_scene_markers')
    log = node.get_logger()
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    static_pub = node.create_publisher(MarkerArray, '/camera_lidar_bev/bev_scene/static_markers', latched)
    scene_pub = node.create_publisher(MarkerArray, '/camera_lidar_bev/bev_scene/markers', 2)

    t = np.genfromtxt(args.map / 'gps_alignment/trajectory_enu.csv', delimiter=',', names=True)
    raw = np.stack([t['x_m'], t['y_m'], t['z_m']], 1)
    keep = [0]
    for i in range(1, len(raw)):  # about one route point per metre
        if np.linalg.norm(raw[i] - raw[keep[-1]]) >= 1.0:
            keep.append(i)
    route = raw[keep]
    landmark_file = args.landmarks or args.map / 'landmarks/landmarks_reviewed.json'
    stops = []
    if landmark_file.exists():
        stops = [l for l in json.loads(landmark_file.read_text())['landmarks'] if l.get('keep') and l['type'] == 'stop_sign']
    else:
        log.warn(f'No {landmark_file}: no stop signs shown (review them with tools/review_landmarks.py)')
    stop_lines = {l['id']: np.array(l['stop_line_estimated']['endpoints_enu_m'], float)
                  for l in stops if l.get('stop_line_estimated')}
    static = static_scene(route, stops, ego_envelope(args.detector_config), args)
    static_pub.publish(static)
    node.create_timer(2.0, lambda: static_pub.publish(static))  # for viewers that dropped it before TF existed
    log.info(f'Route {len(route)} points, stop signs {[l["id"] for l in stops]} from {landmark_file.name}; '
             'traffic lights are not drawn')

    latest = {'det': None, 'det_at': None, 'path': None, 'path_at': None, 'state': None, 'state_at': None,
              'odom': None, 'published': None}

    def now_s():
        return node.get_clock().now().nanoseconds * 1e-9

    def fresh(key, limit):
        return latest[key] is not None and now_s() - latest[key + '_at'] <= limit

    def plan_in_sensor(xyz):
        if latest['odom'] is None:
            return None
        p, q = latest['odom'].pose.pose.position, latest['odom'].pose.pose.orientation
        yaw = yaw_of(q)
        d = xyz[:, :2] - (p.x, p.y)
        c, s_ = math.cos(yaw), math.sin(yaw)
        return np.stack([c * d[:, 0] + s_ * d[:, 1], -s_ * d[:, 0] + c * d[:, 1]], 1)

    def publish():
        stamp = node.get_clock().now().to_msg()
        sc = Scene('os_sensor', stamp)
        clear = Marker()
        clear.action = Marker.DELETEALL
        sc.out.markers.append(clear)
        state = latest['state'] if fresh('state', 1.0) else None
        v_max = float(state.get('max_speed') or args.max_speed_mps) if state else args.max_speed_mps

        plan_xyz = plan_xy_sensor = None
        if fresh('path', 1.0) and len(latest['path'].poses) >= 2:
            path = latest['path']
            plan_xyz = np.array([[p.pose.position.x, p.pose.position.y, p.pose.position.z] for p in path.poses])
            plan = (state or {}).get('plan') or {}
            if len(plan.get('v', [])) == len(plan_xyz):
                v, tt = np.array(plan['v'], float), np.array(plan['t'], float)
            else:
                tt = np.arange(len(plan_xyz)) * args.plan_dt_s
                v = np.hypot(*np.gradient(plan_xyz[:, :2], axis=0).T) / args.plan_dt_s
            start = yaw_of(latest['odom'].pose.pose.orientation) if latest['odom'] is not None else None
            h = headings(plan_xyz[:, :2], start)
            road = plan_xyz + [0, 0, args.ground_offset_m + 0.09]
            kw = dict(frame=path.header.frame_id or 'enu', stamp=path.header.stamp, locked=True)
            if np.hypot(*(plan_xyz[-1, :2] - plan_xyz[0, :2])) > 0.3:
                band = sc.add('plan', Marker.TRIANGLE_LIST, **kw)
                ribbon(band, road, h, args.plan_width_m, [rgba(speed_rgb(x, v_max), 0.55) for x in v])
                edge = np.stack([-np.sin(h), np.cos(h), np.zeros_like(h)], 1) * args.plan_width_m / 2
                for sign in (1, -1):
                    rim = sc.add('plan', Marker.LINE_STRIP, **kw)
                    rim.scale.x = 0.07
                    rim.points = [pt(*p) for p in road + sign * edge + [0, 0, 0.01]]
                    rim.colors = [rgba(speed_rgb(x, v_max), 0.95) for x in v]
                ticks = sc.add('plan', Marker.LINE_LIST, (1, 1, 1), 0.8, **kw)
                ticks.scale.x = 0.06
                for sec in range(1, int(tt[-1] + 1e-6) + 1):
                    k = int(np.argmin(abs(tt - sec)))
                    a, b = road[k] + edge[k] * 0.8 + [0, 0, 0.02], road[k] - edge[k] * 0.8 + [0, 0, 0.02]
                    ticks.points += [pt(*a), pt(*b)]
            plan_xy_sensor = plan_in_sensor(plan_xyz)

        lead_class = ((state or {}).get('lead') or {}).get('class')
        if fresh('det', args.stale_s):
            det = latest['det']
            objects = []
            for d in det.detections:
                name = d.results[0].hypothesis.class_id if d.results else 'object'
                c, sz = d.bbox.center, d.bbox.size
                rng = math.hypot(c.position.x, c.position.y)
                gap = (float(np.min(np.hypot(*(plan_xy_sensor - (c.position.x, c.position.y)).T)))
                       if plan_xy_sensor is not None else math.inf)
                objects.append((name, c, sz, rng, c.position.x > 0 and gap < args.corridor_m + min(sz.x, sz.y) / 2))
            lead = min((o for o in objects if o[4] and o[0] == lead_class), key=lambda o: o[3], default=None)
            kw = dict(frame=det.header.frame_id or 'os_sensor', stamp=det.header.stamp)
            for o in objects:
                name, c, sz, rng, in_path = o
                rgb = COLOURS[category(name)]
                pos = (c.position.x, c.position.y, c.position.z)
                fill = sc.add('objects', Marker.CUBE, rgb, 0.5 if in_path else 0.28, **kw)
                fill.pose = c
                fill.scale.x, fill.scale.y, fill.scale.z = max(sz.x, 0.1), max(sz.y, 0.1), max(sz.z, 0.1)
                edges = sc.add('objects', Marker.LINE_LIST, rgb, **kw)
                edges.pose = c
                edges.scale.x = 0.09 if in_path else 0.05
                hx, hy, hz = sz.x / 2, sz.y / 2, sz.z / 2
                corners = [(sx * hx, sy * hy, sz_ * hz) for sz_ in (-1, 1) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
                for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]:
                    edges.points += [pt(*corners[a]), pt(*corners[b])]
                edges.points += [pt(hx * 0.4, 0, hz), pt(hx, 0, hz)]  # heading tick on the roof
                for m in (fill, edges):
                    m.lifetime = Duration(seconds=args.stale_s).to_msg()
                tag = ' LEAD' if o is lead else ' in path' if in_path else ''
                label = sc.text('objects', (pos[0], pos[1], pos[2] + sz.z / 2 + 0.45), f'{name} {rng:.0f} m{tag}',
                                0.5 if in_path else 0.42, tuple(0.55 + 0.45 * x for x in rgb), **kw)
                label.lifetime = Duration(seconds=args.stale_s).to_msg()

        if state:
            stop = state.get('stop')
            if stop and stop.get('id') in stop_lines and stop.get('distance', 1e9) < 80:
                ends = stop_lines[stop['id']]
                hold = stop.get('state') == 'stopping'
                mid, d = ends.mean(0), ends[1] - ends[0]
                wall = sc.add('stop_target', Marker.CUBE, (1.0, 0.15, 0.1), 0.35, frame='enu', locked=True)
                wall.pose.position = pt(mid[0], mid[1], mid[2] + 0.6)
                yaw = math.atan2(d[1], d[0])
                wall.pose.orientation.z, wall.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
                wall.scale.x, wall.scale.y, wall.scale.z = float(np.hypot(*d[:2])) + 0.4, 0.08, 1.2
                sc.text('stop_target', mid + [0, 0, 1.6], f"{'HOLD' if hold else 'STOP'} {stop['distance']:.0f} m"
                        f" ({stop.get('state', '')})", 0.5, (1.0, 0.45, 0.4), frame='enu', locked=True)
            parts = [str(state.get('mode', '')).replace('_', ' ').upper()]
            meas = state.get('measured') or {}
            plan = state.get('plan') or {}
            if plan.get('v'):
                v3 = plan['v'][min(30, len(plan['v']) - 1)]
                parts.append(f"{max(0.0, meas.get('vx', 0.0)):.1f}->{v3:.1f} m/s (3 s)")
            lead = state.get('lead')
            if lead:
                parts.append(f"lead {lead.get('class', '')} {lead.get('gap', 0):.0f} m")
            sc.text('status', (-3.0, 0.0, 0.3), '  '.join(parts), 0.25, (0.92, 0.95, 1.0))
        scene_pub.publish(sc.out)
        latest['published'] = now_s()

    def on_detections(m):
        latest['det'], latest['det_at'] = m, now_s()
        publish()

    def on_path(m):
        latest['path'], latest['path_at'] = m, now_s()

    def on_state(m):
        try:
            latest['state'], latest['state_at'] = json.loads(m.data), now_s()
        except ValueError:
            log.warn('planner state is not JSON', throttle_duration_sec=10.0)

    def on_odometry(m):
        if m.child_frame_id == 'os_sensor':
            latest['odom'] = m

    def tick():  # keep the plan and status moving when no detections arrive
        if latest['published'] is None or now_s() - latest['published'] > 0.15:
            publish()

    node.create_subscription(Detection3DArray, '/camera_lidar_bev/lidar_detections_3d', on_detections, 2)
    node.create_subscription(PathMsg, '/camera_lidar_bev/planner/trajectory', on_path, 2)
    node.create_subscription(String, '/camera_lidar_bev/planner/state', on_state, 2)
    node.create_subscription(Odometry, '/camera_lidar_bev/localization/odometry', on_odometry, 10)
    node.create_timer(0.1, tick)
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
