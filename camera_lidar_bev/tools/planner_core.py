"""Route planner decisions without ROS: stop lines, traffic lights, front vehicle, speed plan (numpy only).

Shared by tools/route_planner.py (live, shadow mode) and tools/simulate_planner.py (closed loop with simulated
traffic), so what is tested in simulation is what runs on the car. Positions are enu metres; route positions are
the LiDAR (MAP/gps_alignment/trajectory_enu.csv), so the front bumper is lidar_to_front_m ahead of s.
"""
from dataclasses import dataclass, fields
import json
import math

import numpy as np

from prediction_models import GROUPS, MIN_SPEED, Tracker


@dataclass
class PlannerConfig:
    max_speed_mps: float = 4.4704
    max_accel_mps2: float = 1.0
    comfort_decel_mps2: float = 1.5
    max_decel_mps2: float = 4.0  # hardest planned braking
    lateral_accel_mps2: float = 1.5
    time_gap_s: float = 1.8
    standstill_gap_m: float = 4.0  # bumper to bumper when stopped
    stop_margin_m: float = 0.5  # front bumper stops this far before the line
    stop_dwell_s: float = 2.0
    corridor_half_width_m: float = 1.6  # lane half-width checked for objects
    min_detection_score: float = 0.4
    horizon_s: float = 8.0
    lidar_to_front_m: float = 2.35
    emergency_decel_mps2: float = 6.5  # allowed only when max_decel cannot stop before the object ahead
    track_objects: bool = True  # tracker-based lead and crossing checks (False: the first version, raw detections)
    conflict_horizon_s: float = 4.0  # predicted paths of moving objects checked this far for entering the lane
    conflict_persist_s: float = 0.4  # pedestrians / cyclists: the predicted entry must hold this long (vehicles 0.1 s)
    yellow_max_decel_mps2: float = 3.0  # on yellow: stop if this braking is enough, else go through (latched per yellow)
    unknown_light: str = 'go'  # a signal not seen recently: 'go' (map only) or 'stop' (treat as red within light_zone_m)
    light_zone_m: float = 30.0  # 'stop' policy: an unknown signal closer than this is a stop target
    light_hold_s: float = 2.0  # keep the last accepted colour this long without new observations

    @classmethod
    def from_args(cls, args):
        return cls(**{f.name: f.type(getattr(args, f.name)) if f.type in (int, float, bool) else getattr(args, f.name)
                      for f in fields(cls) if hasattr(args, f.name)})


def load_stops(route, map_dir, cfg):
    """Kept stop signs with an estimated stop line (MAP/landmarks/landmarks_reviewed.json)."""
    reviewed = map_dir / 'landmarks/landmarks_reviewed.json'
    stops = []
    for l in json.loads(reviewed.read_text())['landmarks'] if reviewed.exists() else []:
        if l.get('keep') and l['type'] == 'stop_sign' and l.get('stop_line_estimated'):
            line_s = float(route.raw_s[l['stop_line_estimated']['route_index']])
            # Route positions are the LiDAR; stop with the front bumper stop_margin before the line.
            stops.append({'id': l['id'], 'line_s': line_s, 'stop_s': line_s - cfg.lidar_to_front_m - cfg.stop_margin_m,
                          'state': 'armed', 'stopped_since': None})
    return stops


def inside(xy, poly):
    """Points in a polygon (even-odd ray casting)."""
    x, y = xy[:, 0][:, None], xy[:, 1][:, None]
    x0, y0, x1, y1 = poly[:, 0], poly[:, 1], np.roll(poly[:, 0], -1), np.roll(poly[:, 1], -1)
    cond = (y0 > y) != (y1 > y)
    xi = x0 + (y - y0) * (x1 - x0) / np.where(y1 == y0, 1e-12, y1 - y0)
    return (cond & (x < xi)).sum(axis=1) % 2 == 1


def load_signals(route, map_dir, cfg):
    """Signalized intersections on the route: kept traffic lights (MAP/landmarks/landmarks_reviewed.json) grouped by the
    nearest HD-map intersection (MAP/hdmap/hdmap3d_enu.json). The line is the HD stop line the route crosses just
    before entering the intersection, else 1 m before the route enters the intersection polygon."""
    reviewed, hd_path = map_dir / 'landmarks/landmarks_reviewed.json', map_dir / 'hdmap/hdmap3d_enu.json'
    if not reviewed.exists() or not hd_path.exists():
        return []
    lights = [l for l in json.loads(reviewed.read_text())['landmarks'] if l.get('keep') and l['type'] == 'traffic_light']
    layers = json.loads(hd_path.read_text())['layers']
    stop_lines = {e['id']: e for e in layers.get('stop_line', [])}
    xy = np.c_[route.x, route.y]
    groups = {}
    for l in lights:
        p = np.array(l['position_enu_m'][:2])
        inter = min(layers.get('intersection', []), key=lambda e: np.linalg.norm(np.mean(np.array(e['points'])[:, :2], 0) - p))
        if np.linalg.norm(np.mean(np.array(inter['points'])[:, :2], 0) - p) < 30.0:
            groups.setdefault(inter['id'], (inter, []))[1].append(l)
    signals = []
    for iid, (inter, ls) in sorted(groups.items()):
        ins = inside(xy, np.array(inter['points'])[:, :2])
        s_light = min(float(route.s[np.argmin(np.hypot(route.x - l['position_enu_m'][0], route.y - l['position_enu_m'][1]))]) for l in ls)
        starts = [i for i in range(1, len(ins)) if ins[i] and not ins[i - 1]]
        starts = [i for i in starts if route.s[i] <= s_light + 5.0]
        if not starts:
            continue
        entry = float(route.s[max(starts)])  # the last entry before the lights (they face the approach)
        line_s = entry - 1.0
        for sid in inter.get('stop_line_ids', []):
            if sid not in stop_lines:
                continue
            a, b = (np.array(q[:2]) for q in (stop_lines[sid]['points'][0], stop_lines[sid]['points'][-1]))
            side = (xy[:, 0] - a[0]) * (b[1] - a[1]) - (xy[:, 1] - a[1]) * (b[0] - a[0])
            for i in np.nonzero((side[:-1] > 0) != (side[1:] > 0))[0]:
                u = ((xy[i] - a) @ (b - a)) / max((b - a) @ (b - a), 1e-9)
                if -0.15 <= u <= 1.15 and entry - 6.0 <= route.s[i] <= entry + 1.0:
                    line_s = float(route.s[i])
        signals.append({'id': int(iid), 'lights': [l['id'] for l in ls], 'line_s': line_s,
                        'light_xy': np.mean([l['position_enu_m'][:2] for l in ls], axis=0),
                        'stop_s': line_s - cfg.lidar_to_front_m - cfg.stop_margin_m})
    return signals


class RoutePlanner:
    def __init__(self, route, stops, cfg=None, signals=None):
        self.route, self.stops, self.cfg = route, stops, cfg or PlannerConfig()
        self.signals = signals or []
        for sig in self.signals:
            sig.update(seen=[], state='unknown', accepted_t=None, yellow_go=None, target=False)
        self.lead_state = None  # {'t', 's', 'p', 'speed'} of the front object last cycle
        self.tracker = Tracker('cv', confirm_hits=2, max_coast_s=1.0) if self.cfg.track_objects else None
        self.frame_t = None
        self.conflict = None  # last crossing / cut-in object the plan yields to (for display and logs)

    def update_stops(self, s_ego, v_meas, t):
        """Stop-line state machine (re-armed when far behind the line again)."""
        for st in self.stops:
            to_go = st['stop_s'] - s_ego
            if st['state'] in ('released', 'passed') and to_go > 30:
                st['state'], st['stopped_since'] = 'armed', None
            elif st['state'] in ('armed', 'stopping'):
                if -3.0 < to_go < 2.0 and v_meas < 0.3:
                    st['state'] = 'stopping'
                    st['stopped_since'] = st['stopped_since'] or t
                    if t - st['stopped_since'] >= self.cfg.stop_dwell_s:
                        st['state'] = 'released'
                elif to_go < -3.0:
                    st['state'] = 'passed'  # drove through without the planned stop (logged)
                else:
                    st['stopped_since'] = None

    def observe_light(self, signal_id, colour, t):
        """A traffic-light classification for a signal: 'red', 'yellow', 'green' or None (not seen)."""
        for sig in self.signals:
            if sig['id'] == signal_id and colour is not None:
                sig['seen'].append((t, colour))

    def update_signals(self, s_ego, v_ego, t):
        """Filter the light colours (majority of the last 0.5 s, at least 2 alike; held light_hold_s) and decide which
        signals are stop targets: red; yellow if stopping needs at most yellow_max_decel (decided once per yellow);
        unknown only with unknown_light='stop' within light_zone_m. Signals the front bumper has passed are ignored."""
        cfg = self.cfg
        for sig in self.signals:
            sig['seen'] = [o for o in sig['seen'] if t - o[0] <= 0.5]
            votes = {}
            for _, c in sig['seen']:
                votes[c] = votes.get(c, 0) + 1
            best = max(votes, key=votes.get) if votes else None
            if best is not None and votes[best] >= 2:
                if best != sig['state']:
                    sig['yellow_go'] = None
                sig['state'], sig['accepted_t'] = best, t
            elif sig['accepted_t'] is None or t - sig['accepted_t'] > cfg.light_hold_s:
                sig['state'] = 'unknown'
            to_go = sig['stop_s'] - s_ego
            if to_go < -1.0 or to_go > 120.0:
                sig['target'] = False
                continue
            if sig['state'] == 'red':
                sig['target'] = True
            elif sig['state'] == 'yellow':
                if sig['yellow_go'] is None:
                    sig['yellow_go'] = v_ego * v_ego / (2 * max(to_go, 0.1)) > cfg.yellow_max_decel_mps2 and to_go > -1.0
                sig['target'] = not sig['yellow_go']
            elif sig['state'] == 'unknown':
                sig['target'] = cfg.unknown_light == 'stop' and to_go < cfg.light_zone_m
            else:
                sig['target'] = False

    def nearest_stop(self, s_ego):
        return min((st for st in self.stops if st['stop_s'] > s_ego - 3.0), key=lambda st: st['stop_s'], default=None)

    def lead_vehicle(self, t, objects, s_ego, now=None, v_ego=0.0):
        """Closest object in the corridor ahead: (gap bumper-to-bumper m, speed along route m/s, class) or None.

        objects: [(x, y, z, length, width, class, score[, yaw])] in enu, from one detection frame at time t; now is
        the planning time (default t). With track_objects, objects are tracked (kept through short dropouts, speed from
        the track) and moving objects whose predicted path enters the lane ahead are yielded to (crossing, cut-in)."""
        if self.tracker is not None:
            return self.tracked_lead(t, objects, s_ego, t if now is None else now, v_ego)
        cfg, best = self.cfg, None
        for x, y, z, length, width, cls, score, *_ in objects:
            if score < cfg.min_detection_score:
                continue
            proj = self.route.project(x, y, 0.0, s_prev=s_ego, max_dist=cfg.corridor_half_width_m + 3.0,
                                      max_heading=math.pi + 0.1, ahead_m=80.0)
            if proj is None or proj[0] <= s_ego:
                continue
            if abs(proj[1]) - min(length, width) / 2 > cfg.corridor_half_width_m:
                continue
            gap = proj[0] - s_ego - cfg.lidar_to_front_m - max(length, width) / 2
            if best is None or gap < best[0]:
                best = (gap, proj[0], np.array([x, y, z]), cls)
        if best is None:
            self.lead_state = None
            return None
        gap, s_obj, p, cls = best
        prev, speed = self.lead_state, 0.0
        if prev is not None and np.linalg.norm(prev['p'][:2] - p[:2]) < 3.0 and t > prev['t']:
            raw = (s_obj - prev['s']) / (t - prev['t'])
            speed = prev['speed'] + min(1.0, (t - prev['t']) / 0.5) * (raw - prev['speed'])
        self.lead_state = {'t': t, 's': s_obj, 'p': p, 'speed': speed}
        return max(gap, 0.0), max(speed, 0.0), cls

    def tracked_lead(self, t, objects, s_ego, now, v_ego):
        cfg, route = self.cfg, self.route
        if t != self.frame_t:  # a new detection frame
            self.frame_t = t
            dets = [(GROUPS.get(cls, 'vehicle'), cls, np.array([x, y, z]), (length, width, 1.5), rest[0] if rest else 0.0)
                    for x, y, z, length, width, cls, score, *rest in objects if score >= cfg.min_detection_score]
            self.tracker.step(t, dets)
            for tr in (tr for tr in self.tracker.tracks if tr.last_seen == t):  # matched or new this frame
                tr.streak = getattr(tr, 'streak', 0) + 1 if t - getattr(tr, 'prev_seen', -1.0) < 0.15 else 1
                tr.prev_seen = t
            # a track is real after detections in consecutive frames; young tracks are dropped after a short gap, so
            # scattered false detections cannot chain into a confirmed obstacle
            self.tracker.tracks = [tr for tr in self.tracker.tracks
                                   if t - tr.last_seen <= (self.tracker.max_coast_s if tr.hits >= 5 else 0.3)]
        best, conflict = None, None
        for tr in self.tracker.tracks:
            if tr.hits < self.tracker.confirm_hits or (tr.hits < 5 and getattr(tr, 'streak', 1) < 2):
                continue
            age = max(0.0, now - tr.t)
            vel = tr.velocity()
            x, y = tr.position() + vel * age
            length, width = max(tr.size[:2]), min(tr.size[:2])
            proj = route.project(x, y, 0.0, s_prev=s_ego, max_dist=cfg.corridor_half_width_m + 3.0,
                                 max_heading=math.pi + 0.1, ahead_m=80.0)
            speed = float(np.hypot(*vel))
            if proj is not None and proj[0] > s_ego and abs(proj[1]) - width / 2 <= cfg.corridor_half_width_m:
                h = float(route.at(proj[0])[3])
                along = vel[0] * math.cos(h) + vel[1] * math.sin(h)
                v_std = math.sqrt(max(float(np.trace(tr.P[2:4, 2:4])) / 2, 0.0)) if hasattr(tr, 'P') and tr.P.shape[0] >= 4 else 0.5
                v_lead = max(0.0, along - v_std)  # conservative: a new or uncertain track counts as slower
                gap = proj[0] - s_ego - cfg.lidar_to_front_m - length / 2
                if best is None or gap < best[0]:
                    best = (gap, v_lead, tr.cls, np.array([x, y, tr.z]), proj[0])
                continue
            # moving and heading into the lane ahead: yield where its predicted path enters the corridor
            if speed < MIN_SPEED[tr.group] or tr.hits < 4:
                continue
            horizon = cfg.conflict_horizon_s if tr.group != 'pedestrian' else min(cfg.conflict_horizon_s, 3.0)
            entry = None
            for k in range(1, int(horizon / 0.2) + 1):
                tau = k * 0.2
                px, py = x + vel[0] * tau, y + vel[1] * tau
                pr = route.project(px, py, 0.0, s_prev=s_ego, max_dist=cfg.corridor_half_width_m + 3.0,
                                   max_heading=math.pi + 0.1, ahead_m=60.0)
                if pr is None or pr[0] <= s_ego + cfg.lidar_to_front_m:
                    continue
                h = float(route.at(pr[0])[3])
                crossing = abs(-vel[0] * math.sin(h) + vel[1] * math.cos(h))  # speed across the lane
                v_std = math.sqrt(max(float(np.trace(tr.P[2:4, 2:4])) / 2, 0.0)) if hasattr(tr, 'P') and tr.P.shape[0] >= 4 else 0.5
                if crossing < max(0.4 if tr.group == 'pedestrian' else 0.5, 2.0 * v_std):
                    break  # moving along the road (oncoming or parked-lane traffic, cyclists, sidewalk) or too uncertain to tell
                if abs(pr[1]) - width / 2 <= cfg.corridor_half_width_m + 0.3:
                    entry = (tau, pr[0])
                    break
            if entry is None:
                tr.conflict_since = None
                continue
            tr.conflict_since = getattr(tr, 'conflict_since', None) or now
            if now - tr.conflict_since < (cfg.conflict_persist_s if tr.group != 'vehicle' else 0.1):
                continue  # must keep heading for the lane (one noisy velocity is not enough)
            tau, s_c = entry
            to_go = s_c - s_ego - cfg.lidar_to_front_m - length / 2
            t_ego = to_go / max(v_ego, 0.5)
            if t_ego < tau - 1.0:  # the ego is through well before it arrives
                continue
            if best is None or to_go < best[0]:
                best = (max(to_go, 0.0), 0.0, tr.cls, np.array([x, y, tr.z]), s_c)
                conflict = {'class': tr.cls, 'enters_in_s': round(tau, 1), 'at_s': round(s_c, 1)}
        self.conflict = conflict
        if best is None:
            self.lead_state = None
            return None
        gap, v_lead, cls, p, s_obj = best
        self.lead_state = {'t': now, 's': s_obj, 'p': p, 'speed': v_lead}
        return max(gap, 0.0), v_lead, cls

    def plan(self, s_ego, lateral, v0, lead):
        """Speed and path over the horizon from the measured state; returns arrays and the active reason."""
        cfg, route = self.cfg, self.route
        dt, n = 0.1, int(round(cfg.horizon_s / 0.1))
        ds = 0.5
        reach = s_ego + np.arange(0.0, cfg.max_speed_mps * cfg.horizon_s + 60.0, ds)
        reach = reach[reach <= route.length]
        _, _, _, _, kap = route.at(reach)
        v_static = np.minimum(cfg.max_speed_mps, np.sqrt(cfg.lateral_accel_mps2 / np.maximum(np.abs(kap), 1e-4)))
        targets = [(route.length - 1.0, 'route_end', None)]
        for st in self.stops:
            if st['state'] == 'stopping' or (st['state'] == 'armed' and st['stop_s'] > s_ego - 3.0):
                targets.append((st['stop_s'], 'stop_line', st['id']))
        for sig in self.signals:
            if sig['target']:
                targets.append((sig['stop_s'], 'traffic_light', sig['id']))
        stop_s, stop_kind, stop_id = min(targets)

        def reachable(v_max):  # backward pass: every speed can slow to the next with comfort deceleration
            v_max = v_max.copy()
            for i in range(len(v_max) - 2, -1, -1):
                v_max[i] = min(v_max[i], math.sqrt(v_max[i + 1] ** 2 + 2 * cfg.comfort_decel_mps2 * ds))
            return v_max
        v_curve = reachable(v_static)  # curves only, to label why the plan slows
        v_allow = reachable(np.where(reach >= stop_s, 0.0, v_static))
        t_arr, s_arr, v_arr, a_arr, reason = [0.0], [s_ego], [v0], [], []
        s, v = s_ego, v0
        for k in range(n):
            step = max(v * dt + 0.5 * cfg.max_accel_mps2 * dt * dt, 0.05)
            target = float(np.interp(s + step, reach, v_allow)) if len(reach) > 1 else 0.0
            a = (target * target - v * v) / (2 * step)  # exactly the allowed speed one step ahead
            here = float(np.interp(s + step, reach, v_curve)) if len(reach) > 1 else 0.0
            why = stop_kind if target < here - 0.05 else ('curve' if target < cfg.max_speed_mps - 0.05 else 'cruise')
            if lead is not None:
                gap0, v_lead, _ = lead
                gap = gap0 + v_lead * (k * dt) - (s - s_ego)
                desired = cfg.standstill_gap_m + v * cfg.time_gap_s + v * (v - v_lead) / (
                    2 * math.sqrt(cfg.max_accel_mps2 * cfg.comfort_decel_mps2))
                interaction = cfg.max_accel_mps2 * (max(desired, cfg.standstill_gap_m) / max(gap, 0.1)) ** 2
                a_idm = cfg.max_accel_mps2 * (1 - (v / max(target, 0.1)) ** 4) - interaction
                if a_idm < a:
                    a = a_idm
                    why = 'follow' if interaction > 0.1 else why  # the front vehicle is what holds the plan back
            if why in ('cruise', 'curve'):  # above the cap or a curve speed: slow comfortably, brake hard only to stop
                a = max(a, -cfg.comfort_decel_mps2)
            hardest = cfg.max_decel_mps2
            if lead is not None and why == 'follow':  # cannot stop in time with max_decel: brake harder
                room = lead[0] + lead[1] * (k * dt) - (s - s_ego) - 0.5
                if v * v > 2 * cfg.max_decel_mps2 * max(room, 0.01):
                    hardest = cfg.emergency_decel_mps2
            a = float(np.clip(a, -hardest, cfg.max_accel_mps2))
            if v <= 0.0 and a < 0:
                a = 0.0
            v_next = max(0.0, v + a * dt)
            s += 0.5 * (v + v_next) * dt
            v = v_next
            t_arr.append((k + 1) * dt)
            s_arr.append(min(s, route.length))
            v_arr.append(v)
            a_arr.append(a)
            reason.append(why)
        a_arr.append(a_arr[-1] if a_arr else 0.0)
        s_arr = np.array(s_arr)
        x, y, z, h, kap = route.at(s_arr)
        # Start at the vehicle and blend onto the route (smoothstep over max(8 m, 2 s of travel)).
        u = np.clip((s_arr - s_ego) / max(8.0, 2.0 * v0), 0, 1)
        offset = lateral * (1 - u * u * (3 - 2 * u))
        x, y = x - np.sin(h) * offset, y + np.cos(h) * offset
        v_arr = np.array(v_arr)
        return {'t': np.array(t_arr), 's': s_arr, 'x': x, 'y': y, 'z': z, 'v': v_arr, 'ax': np.array(a_arr),
                'ay': v_arr ** 2 * kap, 'kappa': kap, 'reason': reason[0] if reason else 'cruise',
                'stop': (stop_s, stop_kind, stop_id)}
