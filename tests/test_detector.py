import unittest

import numpy as np

from bev_depth.config import load_config
from bev_depth.detector import CompositeDetectorBackend
from bev_depth.types import Detection


class StaticDetector:
    def __init__(self, detections):
        self.detections = detections

    def infer(self, frame_bgr):
        return list(self.detections)

    def infer_batch(self, frames_bgr):
        return [self.infer(frame) for frame in frames_bgr]


class CompositeDetectorTest(unittest.TestCase):
    def test_combines_primary_and_traffic_state_detections(self):
        vehicle = Detection("car", 0.9, (1.0, 2.0, 3.0, 4.0))
        traffic_state = Detection("green", 0.8, (5.0, 6.0, 7.0, 8.0))
        detector = CompositeDetectorBackend(
            (StaticDetector([vehicle]), StaticDetector([traffic_state]))
        )

        detections = detector.infer(np.zeros((10, 10, 3), dtype=np.uint8))

        self.assertEqual(detections, [vehicle, traffic_state])

    def test_combines_batch_results_per_frame(self):
        vehicle = Detection("car", 0.9, (1.0, 2.0, 3.0, 4.0))
        traffic_state = Detection("green", 0.8, (5.0, 6.0, 7.0, 8.0))
        detector = CompositeDetectorBackend(
            (StaticDetector([vehicle]), StaticDetector([traffic_state]))
        )
        frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]

        detections = detector.infer_batch(frames)

        self.assertEqual(detections, [[vehicle, traffic_state]] * 3)

    def test_thor_config_replaces_generic_traffic_light_with_state_detector(self):
        config = load_config("configs/thor_1080p.yaml")

        self.assertNotIn("traffic light", config.detector.classes)
        self.assertIsNotNone(config.traffic_detector)
        self.assertEqual(config.traffic_detector.classes, ("red", "yellow", "green"))
        self.assertEqual(
            config.traffic_detector.path,
            "models/traffic_light_state_fp32.engine",
        )

    def test_new_camera_config_combines_vehicle_traffic_depth_and_bev(self):
        config = load_config("configs/new_camera_bev.yaml")

        self.assertEqual(
            (config.input.width, config.input.height, config.input.fps, config.input.fourcc),
            (1920, 1200, 90, "MJPG"),
        )
        self.assertEqual(config.detector.path, "models/yolo_vehicle_fp16.engine")
        self.assertNotIn("traffic light", config.detector.classes)
        self.assertIsNotNone(config.traffic_detector)
        self.assertEqual(config.traffic_detector.path, "untested_model/best.engine")
        self.assertEqual(config.traffic_detector.classes, ("red", "yellow", "green"))
        self.assertEqual(
            config.depth.path,
            "models/depth_anything_v2_small_fp16.engine",
        )
        self.assertEqual(config.depth.cadence, 2)
        self.assertEqual(config.bev.forward_range_m, 80.0)
        self.assertEqual(config.bev.lateral_range_m, 40.0)


if __name__ == "__main__":
    unittest.main()
