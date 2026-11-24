import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import numpy as np

from bev_depth.config import BevConfig
from bev_depth.temporal_tracking import (
    TemporalBEVTracker,
    _load_tracker_viewer,
    validate_tracker_geometry,
)


class TemporalTrackingIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.bev = BevConfig(
            forward_range_m=80.0,
            rear_range_m=40.0,
            lateral_range_m=80.0,
            pixels_per_meter=5,
        )

    def test_three_camera_geometry_matches_tracker_contract(self):
        validate_tracker_geometry(self.bev)

        incompatible = BevConfig(
            forward_range_m=80.0,
            rear_range_m=0.0,
            lateral_range_m=40.0,
            pixels_per_meter=5,
        )
        with self.assertRaisesRegex(ValueError, "Temporal tracking requires"):
            validate_tracker_geometry(incompatible)

    def test_bridge_rate_limits_and_returns_cleaned_matrix(self):
        tracker = TemporalBEVTracker(self.bev, update_hz=10.0)
        matrix = np.zeros((120, 80), dtype=np.uint8)
        matrix[60, 40] = 3

        first = tracker.maybe_update(matrix, monotonic_s=10.0)
        skipped = tracker.maybe_update(matrix, monotonic_s=10.05)
        second = tracker.maybe_update(matrix, monotonic_s=10.10)

        self.assertIsNotNone(first)
        self.assertIsNone(skipped)
        self.assertIsNotNone(second)
        assert first is not None
        self.assertEqual(first.cleaned_matrix.shape, (120, 80))
        self.assertEqual(first.cleaned_matrix.dtype, np.uint8)
        self.assertGreater(len(first.objects), 0)
        self.assertGreaterEqual(first.raw_target_speed_mph, 0.0)
        self.assertGreaterEqual(first.cleaned_target_speed_mph, 0.0)

    def test_bridge_uses_tracker_projects_raw_vs_causal_viewer(self):
        tracker = TemporalBEVTracker(self.bev, update_hz=10.0)
        matrix = np.zeros((120, 80), dtype=np.uint8)
        matrix[60, 40] = 3
        tracker.maybe_update(matrix, monotonic_s=10.0)

        canvas = tracker.render_viewer()

        self.assertEqual(canvas.shape, (692, 960, 3))
        self.assertEqual(canvas.dtype, np.uint8)

        tracker.reset()
        with self.assertRaisesRegex(RuntimeError, "has not produced"):
            tracker.render_viewer()

    def test_viewer_can_import_sibling_modules_from_tracker_directory(self):
        temporal = ModuleType("test_temporal")
        target_speed = ModuleType("test_target_speed")
        control = ModuleType("test_control")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "longitudinal_safety.py").write_text(
                "SIBLING_VALUE = 42\n", encoding="utf-8"
            )
            (root / "live_bev_viewer.py").write_text(
                "from longitudinal_safety import SIBLING_VALUE\n",
                encoding="utf-8",
            )

            viewer = _load_tracker_viewer(
                root, temporal, target_speed, control
            )

        self.assertEqual(viewer.SIBLING_VALUE, 42)


if __name__ == "__main__":
    unittest.main()
