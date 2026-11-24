import unittest
from unittest.mock import patch

import numpy as np

from bev_depth.config import BevConfig
from bev_depth.render import class_color_bgr, draw_annotated_frame, render_bev
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
        self.assertEqual(tuple(image[100, 300]), (255, 120, 30))  # car: blue (BGR)

    def test_traffic_state_marker_only_highlights_detected_state(self):
        config = BevConfig(forward_range_m=40.0, lateral_range_m=40.0, pixels_per_meter=10)
        image = render_bev(config, [projected("green", (200, 100))])

        self.assertEqual(tuple(image[94, 200]), (25, 25, 75))
        self.assertEqual(tuple(image[100, 200]), (20, 60, 70))
        self.assertEqual(tuple(image[106, 200]), (75, 190, 45))


class ClassColorTest(unittest.TestCase):
    def test_each_class_gets_its_own_bev_color(self):
        config = BevConfig(forward_range_m=40.0, lateral_range_m=40.0, pixels_per_meter=10)
        classes = ["car", "truck", "bus", "person", "bicycle", "motorcycle"]
        image = render_bev(
            config, [projected(name, (60 + 40 * i, 100)) for i, name in enumerate(classes)]
        )
        colors = {tuple(image[100, 60 + 40 * i]) for i in range(len(classes))}
        self.assertEqual(len(colors), len(classes))

    def test_bev_writes_no_distance_or_class_text(self):
        config = BevConfig(forward_range_m=40.0, lateral_range_m=40.0, pixels_per_meter=10)
        with patch("bev_depth.render.ImageDraw.ImageDraw.text") as text:
            render_bev(config, [projected("car", (200, 100))])
        written = [call.args[1] for call in text.call_args_list]
        self.assertFalse(any(label.endswith("m") and "." in label for label in written))
        self.assertNotIn("M:car", written)


class AnnotatedFrameTest(unittest.TestCase):
    def test_boxes_are_colored_by_class_without_distance_text(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        car = projected("car", None)
        person = ProjectedDetection(
            Detection("person", 0.9, (100.0, 50.0, 150.0, 150.0)), 5.0, 0.0, 5.0, None, "depth_map"
        )

        with patch("bev_depth.render.cv2.putText") as put_text:
            image = draw_annotated_frame(frame, [car, person])

        put_text.assert_not_called()
        self.assertEqual(tuple(image[10, 15]), class_color_bgr("car"))
        self.assertEqual(tuple(image[50, 125]), class_color_bgr("person"))
        self.assertNotEqual(class_color_bgr("car"), class_color_bgr("person"))

    def test_track_id_is_the_only_label(self):
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        detection = Detection("car", 0.9, (10.0, 10.0, 20.0, 20.0))
        item = ProjectedDetection(detection, 10.0, 0.0, 10.0, None, "depth_map", track_id=7)

        with patch("bev_depth.render.cv2.putText") as put_text:
            draw_annotated_frame(frame, [item])

        self.assertEqual(put_text.call_count, 1)
        self.assertEqual(put_text.call_args.args[1], "#7")


if __name__ == "__main__":
    unittest.main()
