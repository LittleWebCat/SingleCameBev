import unittest
from unittest.mock import Mock, patch

import cv2

from bev_depth.config import InputConfig
from bev_depth.io import _iter_capture


class CaptureInputTest(unittest.TestCase):
    def test_video_file_does_not_use_camera_backend_or_capture_settings(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 10.0
        capture.read.return_value = (False, None)
        config = InputConfig(
            width=800,
            height=600,
            fps=30,
            fourcc="MJPG",
            camera_index_backend="v4l2",
        )

        with patch("bev_depth.io.cv2.VideoCapture", return_value=capture) as constructor:
            self.assertEqual(list(_iter_capture("video.mkv", config)), [])

        constructor.assert_called_once_with("video.mkv", cv2.CAP_ANY)
        capture.set.assert_not_called()
        capture.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
