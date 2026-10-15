import unittest

from bev_depth.config import BevConfig
from bev_depth.render import render_bev
from bev_depth.types import Detection, ProjectedDetection


def projected(class_name, bev_xy):
    detection = Detection(class_name, 0.9, (10.0, 10.0, 20.0, 20.0))
    return ProjectedDetection(detection, 10.0, 0.0, 10.0, bev_xy, "depth_map")


class BevMarkerTest(unittest.TestCase):
    def test_traffic_controls_have_distinct_symbols_and_colors(self):
        config = BevConfig(forward_range_m=40.0, lateral_range_m=40.0, pixels_per_meter=10)
        image = render_bev(
            config,
            [
                projected("stop sign", (100, 100)),
                projected("traffic light", (200, 100)),
                projected("car", (300, 100)),
            ],
        )

        self.assertEqual(tuple(image[100, 100]), (35, 35, 210))
        self.assertEqual(tuple(image[94, 200]), (45, 45, 225))
        self.assertEqual(tuple(image[100, 200]), (35, 190, 235))
        self.assertEqual(tuple(image[106, 200]), (75, 190, 45))
        self.assertEqual(tuple(image[100, 300]), (0, 80, 255))

    def test_traffic_state_marker_only_highlights_detected_state(self):
        config = BevConfig(forward_range_m=40.0, lateral_range_m=40.0, pixels_per_meter=10)
        image = render_bev(config, [projected("green", (200, 100))])

        self.assertEqual(tuple(image[94, 200]), (25, 25, 75))
        self.assertEqual(tuple(image[100, 200]), (20, 60, 70))
        self.assertEqual(tuple(image[106, 200]), (75, 190, 45))


if __name__ == "__main__":
    unittest.main()
