"""Input and TF checks for the live imported-calibration diagnostic."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from capture_extrinsics import CloudRecord, ImageRecord, choose_pair, header_key
from import_projection import render_calibration_preview, render_two_hypotheses
from view_imported_projection import (expected_camera_info, projection_report,
                                      transform_from_tf, validate_pair)


class LiveProjectionChecks(unittest.TestCase):
    def setUp(self):
        self.expected = expected_camera_info({"image_size": (1024, 768),
            "K": np.array([[800., 0., 512.], [0., 810., 384.], [0., 0., 1.]]), "D": np.zeros(5)})
        self.header = {"frame_id": "camera_front_optical_frame", "stamp_sec": 10, "stamp_nanosec": 100_000_000}
        self.image = ImageRecord(np.zeros((768, 1024, 3), dtype=np.uint8), dict(self.header), 50.0)
        self.info = dict(self.expected, header=dict(self.header), _received_at=50.0)
        self.cloud = CloudRecord({"header": {"frame_id": "os_lidar", "stamp_sec": 10, "stamp_nanosec": 0}}, b"", 50.0)

    def test_observed_parent_child_transform_at_180_degrees(self):
        record = {"parent_frame": "os_sensor", "child_frame": "os_lidar",
                  "translation_m": [.1, -.2, .038195], "quaternion_xyzw": [0., 0., 1., 0.]}
        result = transform_from_tf(record)
        np.testing.assert_allclose(result @ [2., 3., 4., 1.], [-1.9, -3.2, 4.038195, 1.], atol=1e-12)
        # Sign-equivalent unit quaternions must describe the same measured TF.
        np.testing.assert_allclose(transform_from_tf(dict(record, quaternion_xyzw=[0., 0., -1., 0.])), result)
        for change in ({"parent_frame": "os_lidar", "child_frame": "os_sensor"},
                       {"quaternion_xyzw": [0., 0., 0., 0.]},
                       {"translation_m": [0., 0., float("nan")]},
                       {"quaternion_xyzw": [0., 0., 2., 0.]}):
            with self.assertRaises(ValueError):
                transform_from_tf(dict(record, **change))

    def test_exact_header_pairing_and_age_guards(self):
        pair = choose_pair([self.image], {header_key(self.header): self.info}, self.cloud, .15)
        self.assertAlmostEqual(pair[3], .1)
        self.assertAlmostEqual(validate_pair(pair, self.expected, 50.2)["cloud"], .2)
        with self.assertRaisesRegex(ValueError, "STALE"):
            validate_pair(pair, self.expected, 51.01)
        self.info["_received_at"] = 48.0
        with self.assertRaisesRegex(ValueError, "STALE"):
            validate_pair(pair, self.expected, 50.2)
        self.info["_received_at"] = 50.0
        self.info["header"]["stamp_nanosec"] += 1
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_pair(pair, self.expected, 50.2)
        self.assertIsNone(choose_pair([self.image], {}, self.cloud, .15))

    def test_native_intrinsics_and_cloud_frame_are_required(self):
        pair = (self.image, self.info, self.cloud, .1)
        self.image.frame = self.image.frame[::2, ::2]
        with self.assertRaisesRegex(ValueError, "native"):
            validate_pair(pair, self.expected, 50.1)
        self.image.frame = np.zeros((768, 1024, 3), dtype=np.uint8)
        self.info["k"] = list(self.info["k"])
        self.info["k"][0] *= .625
        with self.assertRaisesRegex(ValueError, "K/D/R/P"):
            validate_pair(pair, self.expected, 50.1)
        self.info["k"] = self.expected["k"]
        self.cloud.metadata["header"]["frame_id"] = "os_sensor"
        with self.assertRaisesRegex(ValueError, "os_lidar"):
            validate_pair(pair, self.expected, 50.1)
        self.cloud.metadata["header"]["frame_id"] = "os_lidar"
        self.info["roi"] = {"x_offset": 8, "y_offset": 0, "width": 1024, "height": 768}
        with self.assertRaisesRegex(ValueError, "ROI"):
            validate_pair(pair, self.expected, 50.1)

    def test_b_report_records_visual_preference_without_claiming_accuracy_or_tf(self):
        record = {"parent_frame": "os_sensor", "child_frame": "os_lidar",
                  "translation_m": [0., 0., .038195], "quaternion_xyzw": [0., 0., 1., 0.]}
        camera = {"model": "pinhole", "image_size": [1024, 768],
                  "K": np.array(self.expected["k"]).reshape(3, 3).tolist(), "D": self.expected["d"]}
        external = np.eye(4)
        external[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
        extrinsics = {"direction": "lidar_to_camera", "units": "m", "T_camera_from_lidar": external.tolist()}
        source_points = np.array([[3., 0., 0.], [-3., 0., 0.]])
        sensor_from_lidar = transform_from_tf(record)
        preview = render_calibration_preview(self.image.frame, source_points, camera, extrinsics,
                                             sensor_from_lidar, view="b")
        pair = (self.image, self.info, self.cloud, .1)
        ages = validate_pair(pair, self.expected, 50.1)
        sources = {"extrinsics": {"sha256": "preserved-source-hash"}}
        report = projection_report(pair, ages, preview, record, sources, 12000)
        self.assertFalse(report["frame_unselected"])
        self.assertEqual(report["selected_source_frame"], "os_sensor")
        self.assertEqual(report["selected_hypothesis"], "via_os_sensor")
        self.assertEqual(report["selection_basis"], "user_visual_preference")
        for field in ("tf_published", "fusion_ready", "validation_passed", "exposure_synchronization_verified"):
            self.assertFalse(report[field])
        self.assertIn("provisional", report["accuracy_qualification"])
        self.assertEqual(report["source_files"], sources)
        self.assertEqual(set(report["hypotheses"]), {"via_os_sensor"})
        np.testing.assert_array_equal(report["hypotheses"]["via_os_sensor"]["T_camera_from_os_lidar"],
                                      external @ sensor_from_lidar)
        # Default comparison keeps its original unselected source-frame status.
        comparison = render_two_hypotheses(self.image.frame, source_points, camera, extrinsics, sensor_from_lidar)
        comparison_report = projection_report(pair, ages, comparison, record, sources, 12000)
        self.assertTrue(comparison_report["frame_unselected"])
        self.assertIsNone(comparison_report["selected_source_frame"])
        self.assertIsNone(comparison_report["selection_basis"])
        self.assertEqual(len(comparison_report["hypotheses"]), 2)


if __name__ == "__main__":
    unittest.main()
