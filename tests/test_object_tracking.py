import math
import unittest

import numpy as np

from bev_depth.object_tracking import BEVObjectTracker, CameraTrackIds, TrackerConfig, _Track
from bev_depth.types import Detection, ProjectedDetection

DT = 0.05  # 20 Hz


def det(x, z, name="car", conf=0.8, camera="middle", camera_track_id=None):
    return ProjectedDetection(
        Detection(name, conf, (0.0, 0.0, 10.0, 10.0)),
        math.hypot(x, z), x, z, (0, 0), "depth_map", camera, camera_track_id,
    )


class BEVObjectTrackerTest(unittest.TestCase):
    def test_moving_car_keeps_one_id_through_depth_noise(self):
        rng = np.random.default_rng(0)
        tracker = BEVObjectTracker()
        seen = set()
        for n in range(60):
            x, z = -8.0 + 5.0 * n * DT, 15.0          # 5 m/s across the road
            noise = 0.10 * math.hypot(x, z)            # range-proportional depth noise
            _, ids = tracker.step([det(x, z + rng.normal(0, noise))], n * DT)
            if ids[0] is not None:
                seen.add(ids[0])
        self.assertEqual(len(seen), 1)

    def test_track_is_confirmed_after_min_hits(self):
        tracker = BEVObjectTracker(TrackerConfig(min_hits=3))
        results = [tracker.step([det(1.0, 10.0)], n * DT)[1][0] for n in range(4)]
        self.assertEqual(results[:2], [None, None])
        self.assertIsNotNone(results[2])
        self.assertEqual(results[2], results[3])

    def test_coasts_then_expires_and_a_return_gets_a_new_id(self):
        tracker = BEVObjectTracker(TrackerConfig(max_coast_s=1.0))
        for n in range(5):
            _, ids = tracker.step([det(0.0, 12.0)], n * DT)
        first_id = ids[0]
        t = 4 * DT
        tracks, _ = tracker.step([], t + 0.5)
        self.assertEqual([(tr.track_id, tr.coasting) for tr in tracks], [(first_id, True)])
        tracks, _ = tracker.step([], t + 1.2)
        self.assertEqual(tracks, [])
        for n in range(4):
            _, ids = tracker.step([det(0.0, 12.0)], t + 1.3 + n * DT)
        self.assertNotEqual(ids[0], first_id)

    def test_two_crossing_people_keep_their_ids(self):
        tracker = BEVObjectTracker()
        ids_a, ids_b = set(), set()
        for n in range(80):
            t = n * DT
            a = det(-4.0 + 2.0 * t, 8.0, "person")    # walking right
            b = det(4.0 - 2.0 * t, 9.0, "person")     # walking left, 1 m further
            _, ids = tracker.step([a, b], t)
            if ids[0] is not None:
                ids_a.add(ids[0])
            if ids[1] is not None:
                ids_b.add(ids[1])
        self.assertEqual(len(ids_a), 1)
        self.assertEqual(len(ids_b), 1)
        self.assertNotEqual(ids_a, ids_b)

    def test_person_does_not_take_over_a_car_track(self):
        tracker = BEVObjectTracker()
        for n in range(5):
            _, ids = tracker.step([det(2.0, 10.0, "car")], n * DT)
        car_id = ids[0]
        _, ids = tracker.step([det(2.1, 10.0, "person")], 5 * DT)
        self.assertNotEqual(ids[0], car_id)

    def test_camera_track_hint_survives_a_large_depth_jump(self):
        def run(camera_track_id):
            tracker = BEVObjectTracker()
            for n in range(10):
                _, ids = tracker.step([det(0.0, 15.0, camera_track_id=camera_track_id)], n * DT)
            before = ids[0]
            # Monocular depth suddenly doubles (15 -> 30 m): outside the normal
            # gate, inside the looser gate used when the camera track agrees.
            for n in range(10, 14):
                _, ids = tracker.step([det(0.0, 30.0, camera_track_id=camera_track_id)], n * DT)
            return before, ids[0]

        before, after = run(camera_track_id=7)
        self.assertEqual(before, after)
        before, after = run(camera_track_id=None)
        self.assertNotEqual(before, after)

    def test_object_keeps_its_id_moving_between_cameras(self):
        tracker = BEVObjectTracker()
        ids = set()
        for n in range(40):
            x = -6.0 + 6.0 * n * DT                   # passes from the left camera to the middle one
            camera, local = ("left", 3) if x < -2.0 else ("middle", 11)
            _, got = tracker.step([det(x, 10.0, camera=camera, camera_track_id=local)], n * DT)
            if got[0] is not None:
                ids.add(got[0])
        self.assertEqual(len(ids), 1)

    def test_ego_motion_compensation_keeps_parked_objects_still(self):
        tracker = BEVObjectTracker()
        for n in range(20):
            t = n * DT
            tracks, _ = tracker.step([det(3.0, 30.0 - 10.0 * t)], t,
                                     ego_speed_mps=10.0, ego_yaw_rate_rps=0.0)
        self.assertLess(math.hypot(tracks[0].vx_mps, tracks[0].vz_mps), 0.5)

    def test_left_turn_moves_objects_ahead_to_the_right(self):
        tracker = BEVObjectTracker()
        track = _Track(1, "vehicle", np.array([0.0, 20.0, 0.0, 0.0]), np.eye(4), 0.0)
        tracker._compensate_ego(track, speed_mps=0.0, yaw_rate_rps=math.radians(90), dt=0.1)
        self.assertGreater(track.x[0], 0.0)
        self.assertAlmostEqual(math.hypot(track.x[0], track.x[1]), 20.0, places=6)


class CameraTrackIdsTest(unittest.TestCase):
    def test_bytetrack_ids_follow_boxes(self):
        ids_per_box = {0: set(), 1: set()}
        tracker = CameraTrackIds()
        for n in range(20):
            boxes = [
                Detection("car", 0.9, (100 + 3 * n, 200, 180 + 3 * n, 260)),
                Detection("person", 0.8, (400, 150 - n, 430, 230 - n)),
            ]
            for index, track_id in enumerate(tracker.assign(boxes)):
                if track_id is not None:
                    ids_per_box[index].add(track_id)
        self.assertEqual(len(ids_per_box[0]), 1)
        self.assertEqual(len(ids_per_box[1]), 1)
        self.assertNotEqual(ids_per_box[0], ids_per_box[1])

    def test_no_detections_is_fine(self):
        self.assertEqual(CameraTrackIds().assign([]), [])


if __name__ == "__main__":
    unittest.main()
