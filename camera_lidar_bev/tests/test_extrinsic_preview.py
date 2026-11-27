"""Check independent spherical landmarks, freshness pairing, and stillness gates."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from capture_extrinsics import CloudRecord, ImageRecord, camera_info_matches, choose_pair
from extrinsic_preview import CornerStability, render_range_view


def header(sec, nanos, frame="camera"):
    return {"stamp_sec": sec, "stamp_nanosec": nanos, "frame_id": frame}


class SphericalPreviewTests(unittest.TestCase):
    def test_auto_extent_uses_full_measured_vertical_span_instead_of_a_thin_band(self):
        angles = np.radians(np.linspace(-22, 22, 100))
        points = np.column_stack((3 * np.cos(angles), np.zeros(100), 3 * np.sin(angles)))
        raster = render_range_view(points)
        self.assertEqual(raster.elevation_extent_mode, "auto")
        self.assertGreater(raster.elevation_span_deg, 44)
        self.assertLess(raster.elevation_span_deg, 60)
        rows, _ = np.nonzero(raster.point_indices >= 0)
        self.assertGreater(np.ptp(rows), raster.plot_height * 0.8)
        self.assertEqual(len(np.unique(raster.point_indices[raster.point_indices >= 0])), 100)

    def test_auto_extent_preserves_sparse_edge_returns_and_both_poles(self):
        points = np.vstack((np.tile([2., 0., 0.], (1000, 1)),
                            [0., 0., 3.], [0., 0., -4.], [np.nan, 0., 0.]))
        raster = render_range_view(points)
        self.assertEqual(raster.elevation_span_deg, 180)
        self.assertEqual(raster.sample_direction(0, 90)["decoded_xyz_index"], 1000)
        self.assertEqual(raster.sample_direction(0, -90)["decoded_xyz_index"], 1001)

    def test_selected_xyz_stays_correct_when_auto_extent_changes(self):
        azimuth, elevation = np.radians([13, 8])
        selected = 3 * np.array([np.cos(elevation) * np.cos(azimuth),
                                  np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
        first = render_range_view(np.array([selected, [3, 0, -1], [3, 0, 1]]))
        second = render_range_view(np.array([selected, [1, 0, -3], [1, 0, 3]]))
        self.assertNotEqual(first.elevation_span_deg, second.elevation_span_deg)
        for raster in (first, second):
            sample = raster.sample_direction(13, 8)
            self.assertEqual(sample["decoded_xyz_index"], 0)
            np.testing.assert_array_equal(sample["xyz_m"], selected)

    def test_default_white_display_has_white_measured_points_and_black_empty_pixels(self):
        raster = render_range_view(np.array([[2., 0., 0.]]))
        plot = raster.image[raster.plot_y:raster.plot_y + raster.plot_height,
                            raster.plot_x:raster.plot_x + raster.plot_width]
        valid = raster.point_indices >= 0
        self.assertEqual(np.count_nonzero(valid), 1)
        self.assertTrue(np.all(plot[valid] == 255))
        self.assertTrue(np.all(plot[~valid] == 0))
        self.assertTrue(np.array_equal(raster.image[:, :, 0], raster.image[:, :, 1]))
        self.assertTrue(np.array_equal(raster.image[:, :, 1], raster.image[:, :, 2]))
        colored = render_range_view(np.array([[2., 0., 0.]]), color_mode="range")
        self.assertFalse(np.array_equal(colored.image[:, :, 0], colored.image[:, :, 1]))

    def test_optional_display_markers_repeat_exact_original_return_and_keep_larger_gaps_empty(self):
        points = np.array([[2., 0., 0.]])
        raster = render_range_view(points, point_radius_px=1)
        valid = raster.point_indices >= 0
        self.assertEqual(np.count_nonzero(valid), 9)
        self.assertTrue(np.all(raster.point_indices[valid] == 0))
        rows, cols = np.nonzero(valid)
        for row, col in zip(rows, cols):
            sample = raster.sample(col + raster.plot_x, row + raster.plot_y, radius_px=0)
            np.testing.assert_array_equal(sample["xyz_m"], points[0])
        self.assertIsNone(raster.sample(cols.max() + raster.plot_x + 5,
                                        rows.max() + raster.plot_y, radius_px=0))

    def test_xyz_axis_landmarks_have_correct_spherical_directions_and_original_indices(self):
        points = np.array([[2., 0., 0.], [0., 3., 0.], [0., -4., 0.], [-5., 0., 0.],
                           [1., 0., 1.], [1., 0., -1.]])
        raster = render_range_view(points)
        for index, azimuth, elevation, distance in ((0, 0, 0, 2), (1, 90, 0, 3),
                                                  (2, -90, 0, 4), (3, -180, 0, 5),
                                                  (4, 0, 45, np.sqrt(2)), (5, 0, -45, np.sqrt(2))):
            with self.subTest(index=index):
                sample = raster.sample_direction(azimuth, elevation)
                self.assertIsNotNone(sample)
                self.assertEqual(sample["decoded_xyz_index"], index)
                np.testing.assert_array_equal(sample["xyz_m"], points[index])
                self.assertAlmostEqual(sample["range_m"], distance)
                self.assertAlmostEqual(sample["azimuth_deg"], azimuth if azimuth != -180 else 180)
                self.assertAlmostEqual(sample["elevation_deg"], elevation)

    def test_pixel_collision_keeps_nearest_valid_return_without_mutating_xyz(self):
        points = np.array([[np.nan, 0., 0.], [0., 0., 0.], [8., 0., 0.], [2., 0., 0.],
                           [2., 0., 0.], [np.inf, 0., 0.]])
        before = points.copy()
        raster = render_range_view(points)
        sample = raster.sample_direction(0, 0)
        self.assertEqual(sample["decoded_xyz_index"], 3)
        self.assertEqual(sample["range_m"], 2)
        np.testing.assert_array_equal(points, before)

    def test_zoom_wraps_through_plus_minus_180_and_excludes_far_directions(self):
        angles = np.radians([179., -179., 145., 0.])
        points = np.column_stack((3 * np.cos(angles), 3 * np.sin(angles), np.zeros(4)))
        raster = render_range_view(points, zoom_direction=(179., 0.))
        self.assertEqual(raster.azimuth_span_deg, 60)
        self.assertIsNotNone(raster.sample_direction(179, 0))
        self.assertIsNotNone(raster.sample_direction(-179, 0))
        self.assertIsNone(raster.sample_direction(145, 0))
        self.assertIsNone(raster.sample_direction(0, 0))

    def test_angular_selection_does_not_jump_to_a_distant_return(self):
        angle = np.radians(1.8)
        raster = render_range_view(np.array([[3 * np.cos(angle), 3 * np.sin(angle), 0.]]))
        self.assertIsNone(raster.sample_direction(0, 0, tolerance_deg=0.8))
        self.assertIsNotNone(raster.sample_direction(1.8, 0))

    def test_empty_and_invalid_inputs_are_handled(self):
        raster = render_range_view(np.empty((0, 3)))
        self.assertIsNone(raster.sample_direction(0, 0))
        self.assertIsNone(raster.sample(-1, -1))
        for points in (np.zeros(3), np.zeros((4, 4)), np.array([["1", "2", "3"]])):
            with self.subTest(shape=points.shape):
                with self.assertRaises(ValueError):
                    render_range_view(points)
        with self.assertRaises(ValueError):
            render_range_view(np.ones((2, 3)), color_max_m=float("nan"))
        with self.assertRaises(ValueError):
            render_range_view(np.ones((2, 3)), color_mode="unknown")
        with self.assertRaises(ValueError):
            render_range_view(np.ones((2, 3)), elevation_limits_deg=(20, -20))


class StabilityTests(unittest.TestCase):
    def setUp(self):
        self.corners = np.array([[40., 50.], [60., 50.], [40., 70.], [60., 70.]])

    def test_stationary_reversed_correspondences_accumulate_two_seconds(self):
        tracker = CornerStability(settle_seconds=2, max_rms_motion_px=2)
        self.assertFalse(tracker.observe(self.corners, 10).ready)
        for index in range(1, 11):
            corners = self.corners if index % 2 else self.corners[::-1]
            result = tracker.observe(corners + [0.2, -0.1], 10 + index * 0.2)
        self.assertTrue(result.ready)
        self.assertAlmostEqual(result.elapsed_s, 2)

    def test_gradual_drift_cannot_accumulate_a_stillness_interval(self):
        tracker = CornerStability(settle_seconds=2, max_rms_motion_px=2)
        for index in range(21):
            result = tracker.observe(self.corners + [index, 0], index * 0.2)
            self.assertFalse(result.ready)
        self.assertTrue(result.reset or result.elapsed_s < 2)

    def test_missing_corners_and_receipt_gaps_reset_stability(self):
        tracker = CornerStability()
        tracker.observe(self.corners, 1)
        tracker.observe(self.corners, 1.5)
        result = tracker.observe(self.corners, 2.5)
        self.assertTrue(result.reset)
        self.assertEqual(result.elapsed_s, 0)
        missing = tracker.observe(None, 2.6)
        self.assertFalse(missing.ready)
        self.assertTrue(missing.reset)
        self.assertTrue(tracker.observe(self.corners, 2.7).reset)


class PairSelectionTests(unittest.TestCase):
    def test_nearest_image_requires_exact_header_info_and_precise_nanoseconds(self):
        # Epoch-sized seconds must not lose precision in an absolute float timestamp.
        images = [ImageRecord(np.zeros((1, 1, 3)), header(1_780_000_000, nanos), 0)
                  for nanos in (10, 80, 120)]
        cloud = CloudRecord({"header": header(1_780_000_000, 90, "lidar")}, b"", 0)
        infos = {(1_780_000_000, 10, "camera"): {"id": "old"},
                 (1_780_000_000, 120, "camera"): {"id": "matched"}}
        pair = choose_pair(images, infos, cloud, 0.15)
        self.assertIs(pair[0], images[2])
        self.assertEqual(pair[1]["id"], "matched")
        self.assertEqual(pair[3], 30e-9)

    def test_header_difference_limit_missing_info_and_missing_cloud_reject(self):
        image = ImageRecord(np.zeros((1, 1, 3)), header(10, 0), 0)
        cloud = CloudRecord({"header": header(10, 200_000_000, "lidar")}, b"", 0)
        info = {(10, 0, "camera"): {}}
        self.assertIsNone(choose_pair([image], info, cloud, 0.15))
        self.assertIsNone(choose_pair([image], {}, cloud, 0.3))
        self.assertIsNone(choose_pair([image], info, None, 0.3))

    def test_calibration_identity_and_nonfinite_values(self):
        expected = {"width": 640, "height": 480, "distortion_model": "plumb_bob",
                    "k": list(range(9)), "d": [0.] * 5, "r": list(range(9)), "p": list(range(12))}
        self.assertTrue(camera_info_matches(expected, expected))
        bad = dict(expected, width=1280)
        self.assertFalse(camera_info_matches(bad, expected))
        bad = dict(expected, d=[0, 0, float("nan"), 0, 0])
        self.assertFalse(camera_info_matches(bad, expected))
        bad = dict(expected, k=[1, 2])
        self.assertFalse(camera_info_matches(bad, expected))


if __name__ == "__main__":
    unittest.main()
