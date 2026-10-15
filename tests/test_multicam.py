import unittest

from bev_depth.config import load_config
from bev_depth.multicam_infer import fuse_detections, load_camera_specs
from bev_depth.types import Detection, ProjectedDetection


def projected(name, confidence, x_m, z_m, camera):
    detection = Detection(name, confidence, (10.0, 10.0, 20.0, 20.0))
    return ProjectedDetection(
        detection, z_m, x_m, z_m, (10, 10), "depth_map", camera
    )


class MultiCameraConfigTest(unittest.TestCase):
    def test_verified_camera_order_and_center_only_traffic_color(self):
        path = "configs/three_camera_bev.yaml"
        config = load_config(path)
        cameras = load_camera_specs(path, config)

        self.assertEqual([camera.name for camera in cameras], ["left", "middle", "right"])
        self.assertEqual([camera.source for camera in cameras], [0, 4, 2])
        self.assertEqual(
            [camera.traffic_color for camera in cameras], [False, True, False]
        )
        self.assertEqual(
            [camera.extrinsics.yaw_deg for camera in cameras], [-90.0, 0.0, 90.0]
        )
        self.assertEqual(config.bev.lateral_range_m, 80.0)
        self.assertEqual(config.bev.forward_range_m, 80.0)
        self.assertEqual(config.bev.rear_range_m, 40.0)
        self.assertEqual(config.detector.path, "models/yolo_vehicle_fp16.engine")

    def test_fusion_keeps_highest_confidence_nearby_duplicate(self):
        detections = [
            projected("car", 0.7, 1.0, 12.0, "left"),
            projected("car", 0.9, 1.4, 12.2, "middle"),
            projected("person", 0.8, 1.2, 12.1, "right"),
        ]

        fused = fuse_detections(detections, merge_distance_m=1.5)

        self.assertEqual(len(fused), 2)
        cars = [item for item in fused if item.detection.class_name == "car"]
        self.assertEqual(len(cars), 1)
        self.assertEqual(cars[0].camera_name, "middle")


if __name__ == "__main__":
    unittest.main()
