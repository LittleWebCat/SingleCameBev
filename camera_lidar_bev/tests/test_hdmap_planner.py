"""HD-map rule-based prediction and planner core / traffic simulation checks without ROS (numpy + scipy only).

  python3 -m unittest discover -s tests -p 'test_hdmap_planner.py'
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from hdmap_lanes import HDMapLanes  # noqa: E402
from planner_core import PlannerConfig, RoutePlanner  # noqa: E402
from route_geometry import Route  # noqa: E402

MAP = ROOT / 'maps/local_gps_recording_002'


def lane(i, pts, successors=(), width=3.5):
    return {'id': i, 'lane_type': 'VEHICLE', 'successors': list(successors), 'width_m': width,
            'points': [[x, y, 0.0] for x, y in pts]}


def fork_map(stop=False):
    """Lane 0 east along y=0 to x=50, then lane 1 straight on or lane 2 turning left (radius 10 m)."""
    straight = [(x, 0.0) for x in np.arange(0, 50.1, 0.5)]
    on = [(x, 0.0) for x in np.arange(50, 100.1, 0.5)]
    turn = [(50 + 10 * math.sin(a), 10 - 10 * math.cos(a)) for a in np.linspace(0, math.pi / 2, 40)]
    turn += [(60.0, y) for y in np.arange(10.5, 50, 0.5)]
    layers = {'lane_centerline': [lane(0, straight, (1, 2)), lane(1, on), lane(2, turn)],
              'stop_line': [{'id': 0, 'points': [[48.0, -1.8, 0], [48.0, 1.8, 0]]}] if stop else []}
    return {'layers': layers}


class HDMapLanesTest(unittest.TestCase):
    def test_straight_car_takes_straight_branch(self):
        path = HDMapLanes(fork_map()).follow(30.0, 0.2, 0.0, 10.0, 4.0, 0.5, turn_rate=0.0)
        self.assertLess(abs(path[-1][2]), 0.5)  # still on y = 0
        self.assertAlmostEqual(path[-1][1], 70.0, delta=1.0)  # 40 m along the lanes

    def test_turning_car_takes_turn_branch(self):
        path = HDMapLanes(fork_map()).follow(45.0, 0.3, 0.2, 6.0, 4.0, 0.5, turn_rate=0.4)
        self.assertGreater(path[-1][2], 5.0)  # went north into the turn lane

    def test_stop_line_stop_and_go(self):
        path = HDMapLanes(fork_map(stop=True)).follow(30.0, 0.0, 0.0, 6.0, 8.0, 0.1)
        xs = np.array([p[1] for p in path])
        stopped = np.nonzero(np.abs(np.diff(xs)) < 1e-3)[0]
        self.assertTrue(len(stopped) > 0, 'predicted a stop')
        self.assertAlmostEqual(xs[stopped[0]], 48.0, delta=0.6)  # at the line
        self.assertGreater(xs[-1], 48.5)  # and drove on after waiting

    def test_past_lane_end_matches_next_lane(self):
        lane_, s, _ = HDMapLanes(fork_map()).match(53.0, 0.0, 0.0)
        self.assertEqual(lane_.id, 1)
        self.assertAlmostEqual(s, 3.0, delta=0.5)

    def test_wrong_direction_not_matched(self):
        self.assertIsNone(HDMapLanes(fork_map()).follow(30.0, 0.0, math.pi, 8.0, 4.0))

    def test_aligned_map_loads(self):
        path = MAP / 'hdmap/hdmap3d_enu.json'
        if not path.exists():
            self.skipTest('no aligned HD map')
        hd = HDMapLanes(path)
        self.assertGreater(len(hd.lanes), 50)
        self.assertGreater(sum(len(l.stops) for l in hd.lanes.values()), 3)


class PlannerCoreTest(unittest.TestCase):
    def setUp(self):
        xs = np.arange(0, 300, 1.0)
        self.route = Route(np.c_[xs, np.zeros_like(xs), np.zeros_like(xs)])

    def car(self, x, y=0.0):
        return (x, y, 0.0, 4.6, 1.85, 'car', 0.8, 0.0)

    def test_lead_kept_through_dropout(self):
        pl = RoutePlanner(self.route, [], PlannerConfig())
        t, lead = 0.0, None
        for k in range(10):
            t = k * 0.05
            lead = pl.lead_vehicle(t, [self.car(60.0)], 20.0, now=t, v_ego=4.0)
        self.assertIsNotNone(lead)
        for k in range(10, 16):  # 0.3 s with no detection
            t = k * 0.05
            lead = pl.lead_vehicle(t, [], 20.0, now=t, v_ego=4.0)
        self.assertIsNotNone(lead, 'the stopped car is remembered through a short dropout')
        self.assertAlmostEqual(lead[0], 60.0 - 20.0 - 2.35 - 2.3, delta=1.0)

    def test_old_mode_forgets(self):
        pl = RoutePlanner(self.route, [], PlannerConfig(track_objects=False))
        pl.lead_vehicle(0.0, [self.car(60.0)], 20.0)
        self.assertIsNone(pl.lead_vehicle(0.05, [], 20.0))

    def test_single_ghost_ignored(self):
        pl = RoutePlanner(self.route, [], PlannerConfig())
        self.assertIsNone(pl.lead_vehicle(0.0, [self.car(30.0)], 20.0, now=0.0, v_ego=4.0))
        self.assertIsNone(pl.lead_vehicle(0.05, [], 20.0, now=0.05, v_ego=4.0))

    def test_crossing_pedestrian_yielded_to(self):
        pl = RoutePlanner(self.route, [], PlannerConfig())
        lead = None
        for k in range(30):  # walking at 1.4 m/s towards the lane, 6 m to the side, 20 m ahead
            t = k * 0.05
            ped = (45.0, 6.0 - 1.4 * t, 0.0, 0.6, 0.6, 'person', 0.8, -math.pi / 2)
            lead = pl.lead_vehicle(t, [ped], 20.0, now=t, v_ego=4.0)
        self.assertIsNotNone(lead)
        self.assertIsNotNone(pl.conflict)

    def test_parallel_traffic_not_yielded_to(self):
        pl = RoutePlanner(self.route, [], PlannerConfig())
        for k in range(30):  # oncoming car in the next lane
            t = k * 0.05
            pl.lead_vehicle(t, [(80.0 - 8.0 * t, -3.4, 0.0, 4.6, 1.85, 'car', 0.8, math.pi)], 20.0, now=t, v_ego=4.0)
        self.assertIsNone(pl.conflict)

    def test_emergency_braking_only_when_needed(self):
        pl = RoutePlanner(self.route, [], PlannerConfig())
        close = pl.plan(20.0, 0.0, 4.4, (2.0, 0.0, 'car'))
        far = pl.plan(20.0, 0.0, 4.4, (25.0, 0.0, 'car'))
        self.assertLess(close['ax'][0], -PlannerConfig().max_decel_mps2 - 0.1)
        self.assertGreaterEqual(far['ax'][0], -PlannerConfig().max_decel_mps2)


class TrafficLightTest(unittest.TestCase):
    def setUp(self):
        xs = np.arange(0, 300, 1.0)
        self.route = Route(np.c_[xs, np.zeros_like(xs), np.zeros_like(xs)])

    def planner(self, **cfg):
        sig = {'id': 7, 'lights': [1], 'line_s': 100.0, 'stop_s': 100.0 - 2.35 - 0.5, 'light_xy': np.array([105.0, 3.0])}
        return RoutePlanner(self.route, [], PlannerConfig(**cfg), [sig])

    def see(self, pl, colour, t0, s, v, frames=4):
        for k in range(frames):
            pl.observe_light(7, colour, t0 + k * 0.05)
        pl.update_signals(s, v, t0 + (frames - 1) * 0.05)

    def test_red_is_a_stop_green_is_not(self):
        pl = self.planner()
        self.see(pl, 'red', 0.0, 80.0, 4.0)
        p = pl.plan(80.0, 0.0, 4.0, None)
        self.assertEqual(p['stop'][1], 'traffic_light')
        self.assertLess(p['v'][-1], 0.1)  # stopped within the 8 s plan
        self.assertLess(p['s'][-1] + 2.35, 100.0)  # front bumper before the line
        self.see(pl, 'green', 1.0, 80.0, 4.0)
        self.assertNotEqual(pl.plan(80.0, 0.0, 4.0, None)['stop'][1], 'traffic_light')

    def test_one_wrong_colour_is_filtered(self):
        pl = self.planner()
        self.see(pl, 'green', 0.0, 60.0, 4.0)
        self.see(pl, 'red', 0.2, 60.0, 4.0, frames=1)
        self.assertFalse(pl.signals[0]['target'])

    def test_yellow_stop_far_go_close(self):
        far, close = self.planner(), self.planner()
        self.see(far, 'yellow', 0.0, 50.0, 8.0)  # 47 m to go: 0.7 m/s^2 is enough
        self.see(close, 'yellow', 0.0, 91.0, 8.0)  # 6 m to go: needs 5 m/s^2 -> go
        self.assertTrue(far.signals[0]['target'])
        self.assertFalse(close.signals[0]['target'])

    def test_unknown_policy(self):
        go, stop = self.planner(), self.planner(unknown_light='stop')
        for pl in (go, stop):
            pl.update_signals(80.0, 4.0, 0.0)
        self.assertFalse(go.signals[0]['target'])
        self.assertTrue(stop.signals[0]['target'])

    def test_map_signals(self):
        if not (MAP / 'hdmap/hdmap3d_enu.json').exists():
            self.skipTest('no aligned HD map')
        from planner_core import load_signals
        route = Route(MAP / 'gps_alignment/trajectory_enu.csv')
        signals = load_signals(route, MAP, PlannerConfig())
        self.assertEqual(len(signals), 3)

    def test_red_light_simulation(self):
        if not (MAP / 'hdmap/hdmap3d_enu.json').exists():
            self.skipTest('no aligned HD map')
        from simulate_planner import run_episode
        opts = {'map': str(MAP), 'cfg': PlannerConfig().__dict__, 'duration_s': 40.0, 'dataset': '', 'trace': None,
                'perception': {'latency_s': 0.1, 'pos_std': 0.15, 'miss': 0.1, 'burst': 0.02, 'ghosts': 0.0}}
        for seed in range(3):
            _, _, m = run_episode(('light_red', seed, opts))
            self.assertEqual(m['red_runs'], 0)


class SimulationTest(unittest.TestCase):
    def test_lead_brakes_no_collision(self):
        if not (MAP / 'gps_alignment/trajectory_enu.csv').exists():
            self.skipTest('no map')
        from simulate_planner import run_episode
        opts = {'map': str(MAP), 'cfg': PlannerConfig().__dict__, 'duration_s': 15.0, 'dataset': '', 'trace': None,
                'perception': {'latency_s': 0.1, 'pos_std': 0.15, 'miss': 0.1, 'burst': 0.02, 'ghosts': 0.0}}
        for seed in range(3):
            _, _, m = run_episode(('lead_brakes', 1000 + seed, opts))
            self.assertEqual(m['collision'], 0)
            self.assertGreater(m['min_gap_m'], 0.5)


if __name__ == '__main__':
    unittest.main()
