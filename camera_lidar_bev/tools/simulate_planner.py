#!/usr/bin/env python3
"""Closed-loop safety test of the route planner with simulated traffic (ROS-free; numpy + scipy).

The ego car drives the mapped route with tools/planner_core.py, exactly the decision code of the live
tools/route_planner.py: every 0.05 s (20 Hz) it plans from the measured speed and follows the plan's first
acceleration through a 0.3 s actuator lag. Simulated road users move on their own scripts (they do not react) and
are seen through a perception model: detection latency, position noise, missed frames (and bursts of them), a
range limit and optional ghost detections. Randomized episodes per scenario, all along the route (with its curves
and stop lines):

  slow_lead     a slower car ahead                      lead_brakes  the car ahead brakes hard to a stop
  stopped_car   a car stopped in the lane                cut_in       a car from the next lane cuts in close
  pull_out      a parked car pulls out into the lane     crossing_ped a pedestrian walks across in front
  crossing_car  a car crosses the route ahead            oncoming     oncoming cars and parked cars (no braking wanted)
  stop_and_go   the car ahead speeds up and slows down   dense        several of the above at once
  replay        the recorded traffic of a drive (datasets/prediction/*.npz) around the ego
  light_red / light_green / light_yellow (turns yellow 0.5-6 s before the car arrives) / light_cycle (random
  fixed-time cycle) / light_red_queue (red with stopped cars at the line): the signalized intersections of
  planner_core.load_signals; colours seen like the front camera (88 / 34 / 20 % per frame at 0-20 / 20-40 / 40-60 m,
  3 % wrong colour; --light-perfect for always right). Scored: red-light runs (front bumper over the line on red),
  the colour it crossed on, yellow decisions, stop distance to the line, time to move after green, stops on green.

Scored per episode (all with the true positions): collision (ego and object boxes overlap; objects that drive
into the ego from behind or the side while it is stopped are not counted against it) and whether it was avoidable
(when the object first got in front of the ego, stopping relative to it after 0.3 s needed at most 6.5 m/s^2), smallest bumper gap and
time to collision to objects in the lane ahead, hardest braking and jerk, unnecessary hard braking (oncoming),
stop lines run, and progress. Writes evaluation/planner_sim_<label>.json and prints a table.

  python3 tools/simulate_planner.py [--episodes 30] [--scenarios lead_brakes cut_in ...] [--max-speed-mps 8.94]
  python3 tools/simulate_planner.py --scenarios replay --dataset datasets/prediction/drive_20261126T013444Z.npz
  python3 tools/simulate_planner.py --scenarios stopped_car --episode-seed 2007 --trace trace.csv   # one episode
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from planner_core import PlannerConfig, RoutePlanner, load_signals, load_stops
from route_geometry import Route

ROOT = Path(__file__).resolve().parents[1]
DT = 0.05
EGO_LENGTH, EGO_WIDTH = 4.48, 1.791  # 2025 Nissan LEAF
SCENARIOS = ['slow_lead', 'lead_brakes', 'stopped_car', 'cut_in', 'pull_out', 'crossing_ped', 'crossing_car',
             'oncoming', 'stop_and_go', 'dense']
LIGHT_SCENARIOS = ['light_red', 'light_green', 'light_yellow', 'light_cycle', 'light_red_queue']
CAR, PED = (4.6, 1.85), (0.6, 0.6)
LIGHT_DETECT = ((20.0, 0.88), (40.0, 0.34), (60.0, 0.20))  # front-camera colour found per frame by range (measured)
LANE_W = 3.4


class Actor:
    """A scripted road user. Route-relative (s, lateral) or free (x, y); update(t, dt) advances it."""

    def __init__(self, kind, size, route=None, s=0.0, lat=0.0, v=0.0, xy=None, heading=0.0, start=0.0):
        self.kind, (self.length, self.width), self.route = kind, size, route
        self.s, self.lat, self.v, self.a, self.start = s, lat, v, 0.0, start
        self.xy, self.heading = (None if xy is None else np.array(xy, float)), heading
        self.direction = 1.0  # -1: drives against the route
        self.script = []  # [(t_from, fn(actor, t, dt))]
        self.lat_target, self.lat_rate = lat, 0.0

    def update(self, t, dt):
        if t < self.start:
            return
        for t0, fn in self.script:
            if t >= t0:
                fn(self, t, dt)
        self.v = max(0.0, self.v + self.a * dt)
        if self.xy is not None:
            self.xy += self.v * dt * np.array([math.cos(self.heading), math.sin(self.heading)])
        else:
            self.s += self.direction * self.v * dt
            step = self.lat_rate * dt
            self.lat = self.lat_target if abs(self.lat_target - self.lat) <= step else self.lat + math.copysign(step, self.lat_target - self.lat)

    def present(self, t):
        return t >= self.start and (self.xy is not None or 0.0 <= self.s <= self.route.length)

    def pose(self):
        if self.xy is not None:
            return self.xy[0], self.xy[1], self.heading
        x, y, _, h, _ = self.route.at(self.s)
        h = float(h)
        heading = h + (math.pi if self.direction < 0 else 0.0)
        if self.lat_rate and abs(self.lat_target - self.lat) > 1e-3:  # changing lanes: turned towards the target
            heading += math.copysign(math.atan2(self.lat_rate, max(self.v, 1.0)), self.lat_target - self.lat) * self.direction
        return float(x) - math.sin(h) * self.lat, float(y) + math.cos(h) * self.lat, heading


def box_corners(x, y, h, length, width):
    c, s = math.cos(h), math.sin(h)
    d = np.array([[length / 2, width / 2], [length / 2, -width / 2], [-length / 2, -width / 2], [-length / 2, width / 2]])
    return np.c_[x + d[:, 0] * c - d[:, 1] * s, y + d[:, 0] * s + d[:, 1] * c]


def overlap(a, b):
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (a, b):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def free_actor_at(route, s, side, dist, kind, size, speed, start):
    """An actor that crosses the route at arc length s, starting dist metres to one side."""
    x, y, _, h, _ = route.at(s)
    h = float(h)
    nx, ny = -math.sin(h), math.cos(h)
    sign = 1.0 if side > 0 else -1.0
    a = Actor(kind, size, xy=(float(x) + sign * dist * nx, float(y) + sign * dist * ny), heading=math.atan2(-sign * ny, -sign * nx),
              v=speed, start=start)
    return a


def make_scenario(name, route, stops, rng, cfg):
    """(ego start s, ego start speed, [actors]) for one random episode."""
    vmax = cfg.max_speed_mps
    s0 = float(rng.uniform(20.0, route.length - 260.0))
    v0 = float(rng.uniform(0.6, 1.0) * vmax)
    front = s0 + cfg.lidar_to_front_m
    actors = []

    def car_ahead(gap, v, lat=0.0):
        a = Actor('car', CAR, route, s=front + gap + CAR[0] / 2, lat=lat, v=v)
        actors.append(a)
        return a

    if name == 'slow_lead':
        car_ahead(rng.uniform(15, 40), rng.uniform(0.2, 0.6) * vmax)
    elif name == 'lead_brakes':
        v0 = vmax
        lead = car_ahead(cfg.standstill_gap_m + v0 * cfg.time_gap_s * rng.uniform(0.6, 1.2), vmax)
        tb, dec = rng.uniform(3, 10), rng.uniform(3.0, 8.0)
        lead.script.append((tb, lambda a, t, dt, dec=dec: setattr(a, 'a', -dec if a.v > 0 else 0.0)))
    elif name == 'stopped_car':
        car_ahead(rng.uniform(20, 60), 0.0, lat=rng.uniform(-0.5, 0.5))
    elif name == 'cut_in':
        side = rng.choice([-1, 1])
        c = car_ahead(rng.uniform(2, 12), v0 * rng.uniform(0.5, 1.0), lat=side * LANE_W)
        c.lat_rate, tc = LANE_W / rng.uniform(1.5, 3.0), rng.uniform(0.5, 4.0)
        c.script.append((tc, lambda a, t, dt: setattr(a, 'lat_target', 0.0)))
    elif name == 'pull_out':
        c = car_ahead(rng.uniform(8, 25), 0.0, lat=rng.choice([-1, 1]) * 2.8)
        c.lat_rate, tp = 1.2, rng.uniform(0.5, 4.0)
        c.script.append((tp, lambda a, t, dt: (setattr(a, 'lat_target', 0.0), setattr(a, 'a', 1.5 if a.v < 0.6 * vmax else 0.0))))
    elif name in ('crossing_ped', 'crossing_car'):
        d = rng.uniform(10, 45)
        speed = rng.uniform(0.8, 1.8) if name == 'crossing_ped' else rng.uniform(3, 9)
        size = PED if name == 'crossing_ped' else CAR
        side_dist = rng.uniform(4, 8) if name == 'crossing_ped' else rng.uniform(12, 25)
        arrive = d / max(v0, 0.5)  # when the ego would reach the crossing at its current speed
        start = max(0.0, arrive - side_dist / speed + rng.uniform(-1.5, 1.5))
        actors.append(free_actor_at(route, front + d, rng.choice([-1, 1]), side_dist, name.split('_')[1], size, speed, start))
    elif name == 'oncoming':
        for _ in range(int(rng.integers(2, 5))):
            o = Actor('car', CAR, route, s=front + rng.uniform(20, 120), lat=-LANE_W, v=rng.uniform(3, 9))
            o.direction = -1.0
            actors.append(o)
        for _ in range(int(rng.integers(2, 6))):
            actors.append(Actor('car', CAR, route, s=front + rng.uniform(5, 120), lat=rng.choice([-1, 1]) * rng.uniform(2.9, 3.6)))
    elif name == 'stop_and_go':
        lead = car_ahead(rng.uniform(8, 20), rng.uniform(0, vmax))
        period, amp = rng.uniform(6, 15), rng.uniform(1.0, 2.5)
        lead.script.append((0.0, lambda a, t, dt: setattr(a, 'a', amp * math.sin(2 * math.pi * t / period)
                                                         if not (a.v <= 0 and math.sin(2 * math.pi * t / period) < 0) else 0.0)))
    elif name == 'dense':
        for sub in ('stop_and_go', 'oncoming', 'crossing_ped', 'cut_in'):
            actors += make_scenario(sub, route, stops, rng, cfg)[2]
        return regroup(actors, route, rng, cfg, v0)
    return s0, v0, actors


def phases(*spans):
    """Light state at t from [(colour, duration), ...]; the last colour holds."""
    def state(t):
        for colour, dur in spans:
            if t < dur:
                return colour
            t -= dur
        return spans[-1][0]
    return state


class DilemmaLight:
    """Green until the car is tau seconds from the line at its current speed, then 3 s yellow, red, green."""

    def __init__(self, tau, red_s):
        self.tau, self.red_s, self.t_y = tau, red_s, None

    def update(self, t, to_line, v):
        if self.t_y is None and to_line > 0 and to_line / max(v, 0.1) <= self.tau:
            self.t_y = t

    def __call__(self, t):
        if self.t_y is None:
            return 'green'
        return phases(('yellow', 3.0), ('red', self.red_s), ('green', 1e9))(t - self.t_y)


def make_light_scenario(name, route, signals, rng, cfg):
    """(ego start s, speed, actors, {signal id: state(t)}, the signal approached) for a traffic-light episode."""
    sig = signals[int(rng.integers(len(signals)))]
    vmax = cfg.max_speed_mps
    v0 = float(rng.uniform(0.6, 1.0) * vmax)
    s0 = sig['stop_s'] - float(rng.uniform(40, 110))
    actors, script = [], None
    if name == 'light_red':
        script = phases(('red', float(rng.uniform(5, 25))), ('green', 1e9))
    elif name == 'light_green':
        script = phases(('green', 1e9))
    elif name == 'light_yellow':  # turns yellow when the car is 0.5-6 s from the line (its real speed): the dilemma zone
        script = DilemmaLight(float(rng.uniform(0.5, 6.0)), float(rng.uniform(10, 20)))
    elif name == 'light_cycle':
        g, r, off = float(rng.uniform(10, 25)), float(rng.uniform(10, 25)), float(rng.uniform(0, 60))
        cycle = (('green', g), ('yellow', 3.0), ('red', r))
        period = g + 3.0 + r
        script = lambda t: phases(*cycle)((t + off) % period)
    elif name == 'light_red_queue':
        t_g = float(rng.uniform(8, 20))
        script = phases(('red', t_g), ('green', 1e9))
        for k in range(int(rng.integers(1, 4))):  # stopped cars queued at the line, leaving one by one after green
            c = Actor('car', CAR, route, s=sig['line_s'] - 1.0 - CAR[0] / 2 - k * (CAR[0] + 2.5), v=0.0)
            go = t_g + 1.5 + 1.2 * k
            c.script.append((go, lambda a, t, dt: setattr(a, 'a', 2.0 if a.v < 8.0 else 0.0)))
            actors.append(c)
    return s0, v0, actors, {sig['id']: script}, sig


def regroup(actors, route, rng, cfg, v0):
    """Move a mixed set of actors so they all start around one ego start."""
    s0 = float(rng.uniform(20.0, route.length - 260.0))
    front = s0 + cfg.lidar_to_front_m
    for a in actors:
        if a.xy is None:
            a.s = front + float(rng.uniform(5, 80)) * (1 if a.direction > 0 else 1.5)
        else:  # a crossing: re-place it across the route ahead
            d, side = float(rng.uniform(15, 50)), float(rng.choice([-1, 1]))
            b = free_actor_at(route, front + d, side, 6.0 if a.kind == 'ped' else 15.0, a.kind, (a.length, a.width), a.v,
                              max(0.0, d / max(v0, 0.5) - (6.0 if a.kind == 'ped' else 15.0) / max(a.v, 0.5) + float(rng.uniform(-2, 2))))
            a.xy, a.heading, a.start = b.xy, b.heading, b.start
    return s0, v0, actors


class Perception:
    """True actors -> delayed, noisy, sometimes missing detections in enu."""

    def __init__(self, rng, latency_s=0.1, pos_std=0.15, miss=0.1, burst=0.02, range_m=60.0, ghosts=0.0):
        self.rng, self.latency, self.pos_std, self.miss, self.burst, self.range, self.ghosts = rng, latency_s, pos_std, miss, burst, range_m, ghosts
        self.history, self.blind = [], {}

    def observe(self, t, actors, ego_xy):
        frame = []
        for i, a in enumerate(actors):
            if not a.present(t):
                continue
            x, y, h = a.pose()
            if math.hypot(x - ego_xy[0], y - ego_xy[1]) > (self.range if a.kind != 'ped' else 0.6 * self.range):
                continue
            if self.blind.get(i, -1) > t:
                continue
            if self.rng.random() < self.burst:
                self.blind[i] = t + self.rng.uniform(0.2, 1.0)  # lost for a while (occlusion)
                continue
            if self.rng.random() < self.miss:
                continue
            n = self.rng.normal(0, self.pos_std, 2)
            frame.append((x + n[0], y + n[1], 0.0, a.length, a.width, 'person' if a.kind == 'ped' else 'car', 0.7, h))
        if self.ghosts and self.rng.random() < self.ghosts:
            frame.append((ego_xy[0] + self.rng.uniform(-10, 10), ego_xy[1] + self.rng.uniform(-10, 10), 0.0, 4.0, 1.8, 'car', 0.45, 0.0))
        self.history.append((t, frame))
        while self.history and self.history[0][0] < t - self.latency - 1.0:
            self.history.pop(0)
        old = [f for f in self.history if f[0] <= t - self.latency + 1e-9]
        return old[-1] if old else None


def run_episode(job):
    name, seed, opts = job
    rng = np.random.default_rng(seed)
    route = Route(Path(opts['map']) / 'gps_alignment/trajectory_enu.csv')
    cfg = PlannerConfig(**opts['cfg'])
    stops = load_stops(route, Path(opts['map']), cfg)
    signals = load_signals(route, Path(opts['map']), cfg)
    scripts, light_sig, replay = {}, None, None
    if name == 'replay':
        s0, v0, actors, replay = replay_scenario(opts['dataset'], route, rng, cfg)
    elif name.startswith('light_'):
        s0, v0, actors, scripts, light_sig = make_light_scenario(name, route, signals, rng, cfg)
    else:
        s0, v0, actors = make_scenario(name, route, stops, rng, cfg)
    planner = RoutePlanner(route, stops, cfg, signals)
    truth = {sig['id']: scripts.get(sig['id'], lambda t: 'green') for sig in signals}
    lm = {'red_runs': 0, 'crossed_on': None, 'stop_gap_m': None, 'green_wait_s': None, 'stopped_on_green_s': 0.0,
          'yellow_decision': None}
    green_at, stopped_at_red = None, False
    perception = Perception(rng, **opts['perception'])
    s, v, a = s0, v0, 0.0
    m = {'collision': 0, 'collision_with': None, 'min_gap_m': 99.0, 'min_ttc_s': 99.0, 'max_decel': 0.0, 'max_jerk': 0.0,
         'hard_brakes': 0, 'stops_run': 0, 'progress_m': 0.0}
    trace, braking, stop_min_v, entered = [], False, {}, {}
    near_stop = {st['id'] for st in stops if 0 < st['stop_s'] - s0 < 12}  # starts too close to stop comfortably
    a_prev_cmd = 0.0
    for k in range(int(opts['duration_s'] / DT)):
        t = k * DT
        if s >= route.length - 5:
            break
        ex, ey, _, eh, _ = route.at(s)
        ex, ey, eh = float(ex), float(ey), float(eh)
        for act in actors:
            act.update(t, DT)
        objects = replay(t, (ex, ey)) if replay else []
        seen = perception.observe(t, actors, (ex, ey))
        frame_t, frame = seen if seen else (t, [])
        planner.update_stops(s, v, t)
        for sig in signals:  # traffic-light colours through the front-camera model
            d = math.hypot(sig['light_xy'][0] - ex, sig['light_xy'][1] - ey)
            seen_p = 1.0 if opts.get('light_perfect') else next((pr for r_, pr in LIGHT_DETECT if d < r_), 0.0)
            colour = truth[sig['id']](t) if rng.random() < seen_p else None
            if colour is not None and not opts.get('light_perfect') and rng.random() < 0.03:
                colour = str(rng.choice([c for c in ('red', 'yellow', 'green') if c != colour]))  # misclassified
            planner.observe_light(sig['id'], colour, t)
        planner.update_signals(s, v, t)
        lead = planner.lead_vehicle(frame_t, frame + objects, s, now=t, v_ego=v)
        p = planner.plan(s, 0.0, max(0.0, v + rng.normal(0, 0.05)), lead)
        a_cmd = float(p['ax'][0])
        m['max_jerk'] = max(m['max_jerk'], abs(a_cmd - a_prev_cmd) / DT)
        a_prev_cmd = a_cmd
        a += (a_cmd - a) * DT / 0.3
        a = min(max(a, -7.0), 2.5)
        v = max(0.0, v + a * DT)
        front_before = s + cfg.lidar_to_front_m
        s += v * DT
        if light_sig is not None:  # the approached signal, scored with the true colour
            to_line = light_sig['line_s'] - (s + cfg.lidar_to_front_m)
            if hasattr(truth[light_sig['id']], 'update'):
                truth[light_sig['id']].update(t, to_line, v)
            state = truth[light_sig['id']](t)
            if front_before < light_sig['line_s'] <= s + cfg.lidar_to_front_m:
                lm['crossed_on'] = state
                if state == 'yellow':
                    lm['yellow_decision'] = 'went'
                elif state == 'red' and lm['yellow_decision']:
                    lm['yellow_decision'] = 'ran_red'
                lm['red_runs'] += state == 'red'
            if state == 'yellow' and to_line > 0:
                lm['yellow_decision'] = 'stopped'  # turned yellow before the line; 'went' if it then crosses on yellow
            if v < 0.1 and 0 <= to_line < 12 and state == 'red':
                lm['stop_gap_m'] = round(to_line, 2) if lm['stop_gap_m'] is None else min(lm['stop_gap_m'], round(to_line, 2))
                stopped_at_red = True
            if state == 'green' and green_at is None and stopped_at_red:
                green_at = t
            if green_at is not None and lm['green_wait_s'] is None and v > 1.0:
                lm['green_wait_s'] = round(t - green_at, 2)
            if state == 'green' and v < 0.5 and 0 <= to_line < 15 and name != 'light_red_queue' and not stopped_at_red:
                lm['stopped_on_green_s'] += DT
        m['max_decel'] = max(m['max_decel'], -a)
        if a < -3.0 and not braking:
            m['hard_brakes'] += 1
        braking = a < -3.0
        for st in stops:
            if -3.0 < st['stop_s'] - s < 2.0:
                stop_min_v[st['id']] = min(stop_min_v.get(st['id'], 99.0), v)
        # true safety measures
        cx, cy = ex + math.cos(eh) * (cfg.lidar_to_front_m - EGO_LENGTH / 2), ey + math.sin(eh) * (cfg.lidar_to_front_m - EGO_LENGTH / 2)
        ego_box = box_corners(cx, cy, eh, EGO_LENGTH, EGO_WIDTH)
        bodies = [(act.pose(), act.length, act.width, act.kind, act.v, idx) for idx, act in enumerate(actors) if act.present(t)]
        if replay:
            bodies += [((x, y, h), L, W, c, 0.0, None) for x, y, h, L, W, c in replay.truth(t, (ex, ey))]
        for (x, y, h), L, W, kind, va, idx in bodies:
            dx, dy = x - ex, y - ey
            along, lat = dx * math.cos(eh) + dy * math.sin(eh), -dx * math.sin(eh) + dy * math.cos(eh)
            proj = route.project(x, y, 0.0, s_prev=s, max_dist=4.0, max_heading=4.0, ahead_m=60.0) if along > 0 else None
            if proj is not None and proj[0] > s and abs(proj[1]) < (EGO_WIDTH + W) / 2:  # in the ego's lane ahead
                gap = proj[0] - s - cfg.lidar_to_front_m - L / 2
                m['min_gap_m'] = min(m['min_gap_m'], gap)
                if idx is not None and idx not in entered:  # when it first got in front of the ego
                    entered[idx] = (gap, v, va * math.cos(h - float(route.at(proj[0])[3])))
                closing = v - va * math.cos(h - float(route.at(proj[0])[3]))
                if closing > 0.1 and gap > 0:
                    m['min_ttc_s'] = min(m['min_ttc_s'], gap / closing)
            if abs(dx) < 10 and abs(dy) < 10 and overlap(ego_box, box_corners(x, y, h, L, W)):
                ahead = along > cfg.lidar_to_front_m - 1.0 or (along > 0 and abs(lat) < EGO_WIDTH / 2 + 0.3)
                if v > 0.3 and ahead and not m['collision']:  # the ego moved into it
                    m['collision'], m['collision_with'], m['collision_t'] = 1, kind, round(t, 2)
                    # avoidable: when it entered the lane ahead, stopping (relative to it) needed at most 6.5 m/s^2 after 0.3 s
                    g0, v_e, v_o = entered.get(idx, (0.0, v, 0.0))
                    rel = max(v_e - max(v_o, 0.0), 0.0)
                    room = g0 - rel * 0.3
                    m['avoidable'] = int(rel < 0.1 or (room > 0 and rel * rel / (2 * room) <= 6.5))
        if opts.get('on_step'):  # live view (tools/sim_live.py): everything about this step, in process only
            opts['on_step']({'t': t, 's': s, 'v': v, 'a': a, 'a_cmd': a_cmd, 'route': route, 'cfg': cfg, 'actors': actors,
                             'frame': frame, 'plan': p, 'lead': lead, 'planner': planner, 'signals': signals,
                             'truth': {k: f(t) for k, f in truth.items()}, 'light_sig': light_sig, 'metrics': m,
                             'light_metrics': lm, 'name': name, 'seed': seed})
        if opts.get('trace'):
            trace.append({'t': round(t, 2), 's': round(s, 2), 'v': round(v, 3), 'a': round(a, 3), 'a_cmd': round(a_cmd, 3),
                          'mode': p['reason'], 'lead_gap': None if lead is None else round(lead[0], 2),
                          'lead_v': None if lead is None else round(lead[1], 2), 'seen': len(frame), 'true_gap': round(m['min_gap_m'], 2),
                          'light': truth[light_sig['id']](t) if light_sig else None,
                          'light_seen': next((sg['state'] for sg in planner.signals if light_sig and sg['id'] == light_sig['id']), None),
                          'to_line': round(light_sig['line_s'] - s - cfg.lidar_to_front_m, 2) if light_sig else None})
    m['progress_m'] = s - s0
    if light_sig is not None:
        m.update(lm, signal=light_sig['id'], stopped_on_green_s=round(lm['stopped_on_green_s'], 2))
    passed = {st['id'] for st in stops if s > st['stop_s'] + 2.0}  # an episode ending mid-stop is not a run line
    m['stops_run'] = sum(1 for k, v_min in stop_min_v.items() if v_min > 0.3 and k not in near_stop and k in passed)
    m['min_gap_m'] = round(m['min_gap_m'], 2)
    m['min_ttc_s'] = round(m['min_ttc_s'], 2)
    m['max_decel'], m['max_jerk'] = round(m['max_decel'], 2), round(m['max_jerk'], 1)
    if opts.get('trace'):
        with open(opts['trace'], 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(trace[0]))
            w.writeheader()
            w.writerows(trace)
    return name, seed, m


class Replay:
    """Recorded detections (dataset frames, enu) replayed around the simulated ego; truth = the detections."""

    def __init__(self, frames, t0):
        self.frames, self.t0 = frames, t0
        self.times = np.array([f[0] for f in frames])

    def at(self, t):
        i = int(np.searchsorted(self.times, self.t0 + t))
        return self.frames[min(i, len(self.frames) - 1)][1]

    def __call__(self, t, ego_xy):
        return [(p[0], p[1], p[2], max(sz[:2]), min(sz[:2]), cls, 0.7, yaw) for g, cls, p, sz, yaw in self.at(t)
                if math.hypot(p[0] - ego_xy[0], p[1] - ego_xy[1]) < 60]

    def truth(self, t, ego_xy):
        return [(p[0], p[1], yaw, max(sz[:2]), min(sz[:2]), 'ped' if g == 'pedestrian' else 'car') for g, cls, p, sz, yaw in self.at(t)
                if math.hypot(p[0] - ego_xy[0], p[1] - ego_xy[1]) < 15]


def replay_scenario(dataset, route, rng, cfg):
    """Start the ego where the recording car was at a random moment; the recorded traffic is replayed around it."""
    from prediction_dataset import load_dataset
    frames, _, _, ego_poses = load_dataset(Path(dataset), 0.35)
    i = int(rng.integers(0, max(1, len(frames) - 700)))
    t0 = frames[i][0]
    x, y, _, yaw = np.asarray(ego_poses)[i]  # the recording car's LiDAR pose per frame
    proj = route.project(float(x), float(y), float(yaw), max_dist=4.0)
    s0 = proj[0] if proj else float(rng.uniform(20, route.length - 260))
    return s0, 0.6 * cfg.max_speed_mps, [], Replay(frames, t0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS)
    parser.add_argument('--episodes', type=int, default=30, help='per scenario')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episode-seed', type=int, help="rerun one episode by the 'seed' in a report (first --scenarios)")
    parser.add_argument('--duration-s', type=float, default=30.0)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/prediction/drive_20261126T013444Z.npz', help='for replay')
    parser.add_argument('--latency-s', type=float, default=0.1)
    parser.add_argument('--pos-std-m', type=float, default=0.15)
    parser.add_argument('--miss', type=float, default=0.1, help='chance a detection is missing in a frame')
    parser.add_argument('--burst', type=float, default=0.02, help='chance per frame an object is lost for 0.2-1 s')
    parser.add_argument('--ghosts', type=float, default=0.0, help='chance per frame of a false detection near the ego')
    parser.add_argument('--label', default='default')
    parser.add_argument('--light-perfect', action='store_true', help='traffic-light colours always seen correctly')
    parser.add_argument('--trace', type=Path, help='per-step CSV (use with one episode)')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    for f in PlannerConfig.__dataclass_fields__.values():
        if f.type is bool:
            parser.add_argument('--' + f.name.replace('_', '-'), type=lambda v: v.lower() in ('1', 'true', 'yes'), default=f.default)
        elif f.type is str:
            parser.add_argument('--' + f.name.replace('_', '-'), default=f.default)
        else:
            parser.add_argument('--' + f.name.replace('_', '-'), type=float, default=f.default)
    args = parser.parse_args()
    cfg = PlannerConfig.from_args(args)
    opts = {'map': str(args.map), 'cfg': cfg.__dict__, 'duration_s': args.duration_s, 'dataset': str(args.dataset),
            'trace': str(args.trace) if args.trace else None, 'light_perfect': args.light_perfect,
            'perception': {'latency_s': args.latency_s, 'pos_std': args.pos_std_m, 'miss': args.miss, 'burst': args.burst,
                           'ghosts': args.ghosts}}
    if args.episode_seed is not None:  # one episode exactly as in a report (its 'seed')
        jobs = [(args.scenarios[0], args.episode_seed, opts)]
    else:
        jobs = [(name, args.seed * 100000 + i * 1000 + j, opts) for i, name in enumerate(args.scenarios) for j in range(args.episodes)]
    t0 = time.time()
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(run_episode, jobs, chunksize=2))
    else:
        results = [run_episode(j) for j in jobs]
    by = {}
    for name, seed, m in results:
        by.setdefault(name, []).append({'seed': seed, **m})
    summary = {}
    for name, eps in by.items():
        col = lambda k: np.array([e[k] for e in eps], float)
        gaps = col('min_gap_m')
        summary[name] = {'episodes': len(eps), 'collisions': int(col('collision').sum()),
                         'avoidable_collisions': int(sum(e.get('avoidable', 0) for e in eps)),
                         'collision_seeds': [e['seed'] for e in eps if e['collision']],
                         'min_gap_m_p5': round(float(np.percentile(gaps[gaps < 99], 5)), 2) if (gaps < 99).any() else None,
                         'ttc_under_1_5s': int((col('min_ttc_s') < 1.5).sum()),
                         'hard_brake_episodes': int((col('hard_brakes') > 0).sum()),
                         'max_decel_p95': round(float(np.percentile(col('max_decel'), 95)), 2),
                         'stops_run': int(col('stops_run').sum()),
                         'mean_progress_m': round(float(col('progress_m').mean()), 1)}
        if name.startswith('light_'):
            gaps_ = [e['stop_gap_m'] for e in eps if e.get('stop_gap_m') is not None]
            waits = [e['green_wait_s'] for e in eps if e.get('green_wait_s') is not None]
            summary[name].update({
                'red_runs': int(sum(e['red_runs'] for e in eps)),
                'crossed_on': dict(Counter(str(e['crossed_on']) for e in eps)),
                'yellow_decisions': dict(Counter(e['yellow_decision'] for e in eps if e.get('yellow_decision'))),
                'stop_gap_to_line_m_median': round(float(np.median(gaps_)), 2) if gaps_ else None,
                'green_wait_s_median': round(float(np.median(waits)), 2) if waits else None,
                'episodes_stopped_on_green': int(sum(e['stopped_on_green_s'] > 1.0 for e in eps))})
    report = {'label': args.label, 'config': cfg.__dict__, 'perception': opts['perception'], 'episodes': args.episodes,
              'duration_s': args.duration_s, 'seed': args.seed, 'summary': summary, 'results': by}
    out = ROOT / 'evaluation' / f'planner_sim_{args.label}.json'
    if not args.trace:
        out.write_text(json.dumps(report, indent=1) + '\n')
    keys = ['collisions', 'avoidable_collisions', 'ttc_under_1_5s', 'min_gap_m_p5', 'hard_brake_episodes', 'max_decel_p95', 'stops_run', 'mean_progress_m']
    print(f'{len(jobs)} episodes in {time.time() - t0:.0f} s, max speed {cfg.max_speed_mps:.2f} m/s')
    print(f"{'scenario':14s}" + ''.join(f'{k:>20s}' for k in keys))
    for name in args.scenarios:
        if name in summary:
            print(f'{name:14s}' + ''.join(f"{str(summary[name][k]):>20s}" for k in keys))
    lights = [n for n in args.scenarios if n.startswith('light_') and n in summary]
    if lights:
        lk = ['red_runs', 'crossed_on', 'yellow_decisions', 'stop_gap_to_line_m_median', 'green_wait_s_median', 'episodes_stopped_on_green']
        print(f"\ntraffic lights (unknown light: {cfg.unknown_light}{', colours perfect' if args.light_perfect else ''})")
        for name in lights:
            print(f'{name:16s}' + '  '.join(f'{k}={summary[name][k]}' for k in lk))
    for name in args.scenarios:
        if summary.get(name, {}).get('collision_seeds'):
            print(f"  {name} collisions: seeds {summary[name]['collision_seeds'][:10]}")
    if not args.trace:
        print(f'wrote {out}')


if __name__ == '__main__':
    main()
