import unittest

import numpy as np

from bev_depth.config import DepthConfig
from bev_depth.depth import sample_detection_depth
from bev_depth.types import Detection


class DepthSamplingTest(unittest.TestCase):
    def test_samples_median_from_lower_center(self):
        depth = np.full((100, 100), 12.0, dtype=np.float32)
        depth[70:85, 40:60] = 18.0
        config = DepthConfig(
            backend="constant",
            path="",
            input_size=384,
            cadence=2,
            metric_scale=1.0,
            metric_bias=0.0,
            inverse_depth=False,
        )
        detection = Detection("car", 0.9, (30.0, 20.0, 70.0, 90.0))
        value, source = sample_detection_depth(depth, detection, config)
        self.assertEqual(source, "depth_map")
        self.assertAlmostEqual(value, 18.0, places=4)

    def test_samples_depth_for_traffic_control_classes(self):
        depth = np.full((100, 100), 24.0, dtype=np.float32)
        config = DepthConfig(
            backend="constant",
            path="",
            input_size=384,
            cadence=2,
            metric_scale=1.0,
            metric_bias=0.0,
            inverse_depth=False,
        )

        for class_name in ("traffic light", "stop sign"):
            with self.subTest(class_name=class_name):
                detection = Detection(class_name, 0.9, (30.0, 20.0, 70.0, 90.0))
                value, source = sample_detection_depth(depth, detection, config)
                self.assertEqual(source, "depth_map")
                self.assertAlmostEqual(value, 24.0, places=4)

    def test_rejects_invalid_depth(self):
        depth = np.zeros((100, 100), dtype=np.float32)
        config = DepthConfig(
            backend="constant",
            path="",
            input_size=384,
            cadence=2,
            metric_scale=1.0,
            metric_bias=0.0,
            inverse_depth=False,
        )
        detection = Detection("car", 0.9, (30.0, 20.0, 70.0, 90.0))
        value, source = sample_detection_depth(depth, detection, config)
        self.assertIsNone(value)
        self.assertEqual(source, "no_valid_depth")


if __name__ == "__main__":
    unittest.main()
