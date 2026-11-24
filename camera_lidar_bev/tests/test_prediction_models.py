"""Object tracking / prediction checks without ROS (numpy + scipy only).

  python3 -m unittest discover -s tests -p 'test_prediction_models.py'
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prediction_dataset import load_dataset  # noqa: E402
from prediction_models import MODELS, RouteLanes, Tracker, register_model  # noqa: E402
from route_geometry import Route  # noqa: E402

DATASET = ROOT / 'datasets/prediction/drive_20261124T213905Z.npz'


def scene(motion, seconds=6.0, dt=0.05, noise=0.1, group='vehicle', cls='car', seed=0):
    """Frames of one object whose enu position at time t is motion(t) -> (x, y, heading)."""
    rng = np.random.default_rng(seed)
    size = (0.6, 0.6, 1.7) if group == 'pedestrian' else (4.5, 1.9, 1.6)
    frames = []
    for k in range(int(seconds / dt)):
        t = k * dt
        x, y, h = motion(t)
        frames.append((t, [(group, cls, np.array([x + rng.normal(0, noise), y + rng.normal(0, noise), 0.0]), size, h)]))
    return frames


def predict_after(model, frames, horizon=3.0):
    tracker = Tracker(model)
    for t, dets in frames:
        tracker.step(t, dets)
    (tr,) = tracker.confirmed(frames[-1][0])
    return tracker.predict_track(tr, horizon)


class PredictionTests(unittest.TestCase):
    def test_straight_car_both_models(self):
        frames = scene(lambda t: (8.0 * t, 0.0, 0.0))
        t_end = frames[-1][0]
        for model in ('cv', 'imm'):
            moving, path = predict_after(model, frames)
            self.assertTrue(moving)
            error = math.hypot(path[-1][1] - 8.0 * (t_end + 3.0), path[-1][2])
            self.assertLess(error, 1.0, model)

    def test_parked_car_stays(self):
        for model in ('cv', 'imm'):
            moving, path = predict_after(model, scene(lambda t: (5.0, 2.0, 0.3)))
            self.assertFalse(moving, model)
            self.assertAlmostEqual(path[-1][1], 5.0, delta=0.3)

    def test_crossing_pedestrian(self):
        frames = scene(lambda t: (3.0, -4.0 + 1.3 * t, math.pi / 2), group='pedestrian', cls='person', noise=0.05)
        moving, path = predict_after('imm', frames)  # pedestrians always use the constant-velocity filter
        self.assertTrue(moving)
        error = math.hypot(path[-1][1] - 3.0, path[-1][2] - (-4.0 + 1.3 * (frames[-1][0] + 3.0)))
        self.assertLess(error, 0.8)

    def test_turning_car_imm_follows_the_turn(self):
        w, v = 0.2, 6.0

        def turn(t):
            h = w * t
            return v / w * math.sin(h), v / w * (1 - math.cos(h)), h
        frames = scene(turn, seconds=8.0)
        t_end = frames[-1][0]
        x, y, _ = turn(t_end + 3.0)
        x0, y0, h0 = turn(t_end)
        straight = math.hypot(x0 + v * 3.0 * math.cos(h0) - x, y0 + v * 3.0 * math.sin(h0) - y)  # ~5 m
        _, path = predict_after('imm', frames)
        error = math.hypot(path[-1][1] - x, path[-1][2] - y)  # 2.2 m on 2026-11-24
        self.assertLess(error, 0.5 * straight)

    def test_registered_model_is_used(self):
        class StayTrack(MODELS['cv'](0.0, ('vehicle', 'car', np.zeros(3), (4, 2, 1.5), 0.0)).__class__):
            def future(self, horizon_s, step_s=0.1):
                x, y = self.position()
                return [(k * step_s, x, y, 0.0, 0.0) for k in range(int(round(horizon_s / step_s)) + 1)]
        register_model('stay_test', lambda t, det: StayTrack(t, det))
        frames = scene(lambda t: (8.0 * t, 0.0, 0.0))
        _, path = predict_after('stay_test', frames)
        self.assertAlmostEqual(path[-1][1], path[0][1])
        with self.assertRaises(ValueError):
            Tracker('no_such_model')

    def test_lane_following_along_a_straight_route(self):
        route = Route(np.c_[np.linspace(-50, 150, 400), np.full(400, 1.0), np.zeros(400)])
        tracker = Tracker('cv', lanes=RouteLanes(route))
        for t, dets in scene(lambda t: (6.0 * t, 0.0, 0.0)):
            tracker.step(t, dets)
        (tr,) = tracker.confirmed(t)
        _, path = tracker.predict_track(tr, 3.0)
        self.assertAlmostEqual(path[-1][2], 0.0, delta=0.3)  # keeps its 1 m offset from the route


@unittest.skipUnless(DATASET.exists(), 'replay dataset not present')
class ReplayDatasetTests(unittest.TestCase):
    def test_dataset_replays(self):
        frames, route_xyz, meta, ego = load_dataset(DATASET)
        self.assertEqual(len(frames), meta['frames'])
        self.assertGreater(len(route_xyz), 100)
        tracker = Tracker('imm')
        confirmed = 0
        for t, dets in frames[:600]:  # about 30 s
            tracker.step(t, dets)
            confirmed = max(confirmed, len(tracker.confirmed(t)))
        self.assertGreater(confirmed, 3)


if __name__ == '__main__':
    unittest.main()
