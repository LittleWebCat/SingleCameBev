"""Exercise dataset integrity without ROS, a camera, or a display."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from calibration_capture import BoardConfig, CaptureSession


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.board = BoardConfig(5, 4, 65.0)
        self.session = CaptureSession(self.root / "dataset", self.board, "/camera/front/image_raw")
        self.frame = np.random.default_rng(4).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        self.corners = np.array([(160 + x * 40, 120 + y * 40)
                                 for y in range(4) for x in range(5)], dtype=np.float32).reshape(-1, 1, 2)

    def capture(self, corners=None, frame=None, stamp=1, frame_id="front_optical", age=0.01):
        return self.session.capture(self.frame if frame is None else frame,
                                    self.corners if corners is None else corners,
                                    {"frame_id": frame_id, "stamp_sec": stamp, "stamp_nanosec": 2,
                                     "timestamp_source": "host_receipt_after_read"}, age)

    def test_raw_pixels_and_manifest_provenance(self):
        result = self.capture()
        self.assertTrue(result.accepted)
        np.testing.assert_array_equal(cv2.imread(str(result.path)), self.frame)
        manifest = json.loads((self.session.output_dir / "session.json").read_text())
        self.assertEqual(manifest["calibration_status"], "unmeasured")
        self.assertEqual(manifest["board"]["square_size_source"], "user_provided")
        self.assertEqual(manifest["board"]["square_size_mm"], 65.0)
        self.assertEqual((manifest["camera_width_px"], manifest["camera_height_px"]), (640, 480))
        self.assertEqual(len(manifest["saved_frames"][0]["corners_px"]), 20)
        self.assertEqual(manifest["saved_frames"][0]["header"]["timestamp_source"],
                         "host_receipt_after_read")
        self.assertTrue(manifest["saved_frames"][0]["saved_at_utc"].endswith("+00:00"))
        self.assertEqual(manifest["min_board_span_px"], 0.0)
        self.assertEqual(manifest["saved_frames"][0]["feedback"]["corner_span_px"], [160.0, 120.0])

    def test_board_size_guide_rejects_below_and_accepts_boundary(self):
        self.session = CaptureSession(self.root / "size_guide", self.board, "/camera",
                                      min_board_span_px=140)
        small = np.array([(160 + x * 34.875, 120 + y * 30)
                          for y in range(4) for x in range(5)], dtype=np.float32).reshape(-1, 1, 2)
        rejected = self.capture(corners=small)
        self.assertFalse(rejected.accepted)
        self.assertIn("Move closer", rejected.reason)
        self.assertIn("139.5", rejected.reason)
        self.assertEqual(self.session.count, 0)
        self.assertEqual(list((self.session.output_dir / "images").iterdir()), [])
        boundary = small.copy()
        boundary[:, :, 0] = 160 + (boundary[:, :, 0] - 160) * (140 / 139.5)
        self.assertTrue(self.capture(corners=boundary).accepted)
        manifest = json.loads((self.session.output_dir / "session.json").read_text())
        self.assertEqual(manifest["min_board_span_px"], 140.0)

    def test_board_size_guide_accepts_larger_vertical_span(self):
        self.session = CaptureSession(self.root / "portrait", self.board, "/camera",
                                      min_board_span_px=140)
        portrait = np.array([(160 + x * 20, 120 + y * 50)
                             for y in range(4) for x in range(5)], dtype=np.float32).reshape(-1, 1, 2)
        self.assertTrue(self.capture(corners=portrait).accepted)

    def test_board_size_guide_is_disabled_by_default(self):
        tiny = np.array([(160 + x * 3, 120 + y * 3)
                         for y in range(4) for x in range(5)], dtype=np.float32).reshape(-1, 1, 2)
        self.assertTrue(self.capture(corners=tiny).accepted)

    def test_invalid_board_size_guide_does_not_create_dataset(self):
        path = self.root / "invalid_size"
        for value in [-1, float("nan"), float("inf"), True, "140", None]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "min_board_span_px"):
                    CaptureSession(path, self.board, "/camera", min_board_span_px=value)
                self.assertFalse(path.exists())

    def test_duplicate_pose_and_reversed_correspondence(self):
        self.assertTrue(self.capture().accepted)
        self.assertFalse(self.capture(stamp=2).accepted)
        self.assertFalse(self.capture(corners=self.corners[::-1], stamp=3).accepted)
        result = self.capture(corners=self.corners + [50, 0], stamp=4)
        self.assertTrue(result.accepted)
        self.assertEqual(result.count, 2)

    def test_tilt_with_same_center_is_distinct(self):
        self.assertTrue(self.capture().accepted)
        tilted = self.corners.copy()
        tilted[:, :, 0] = 240 + (tilted[:, :, 0] - 240) * 1.4
        self.assertTrue(self.capture(corners=tilted, stamp=2).accepted)

    def test_reused_stamp_is_rejected_even_after_pose_motion(self):
        self.assertTrue(self.capture().accepted)
        self.assertFalse(self.capture(corners=self.corners + [50, 0]).accepted)
        self.assertEqual(self.session.count, 1)

    def test_stale_nonfinite_and_outside_corners(self):
        self.assertFalse(self.capture(age=1.01).accepted)
        self.assertFalse(self.capture(age=float("nan")).accepted)
        invalid = self.corners.copy()
        invalid[0, 0, 0] = np.nan
        self.assertFalse(self.capture(corners=invalid).accepted)
        invalid[0, 0, 0] = 640
        self.assertFalse(self.capture(corners=invalid).accepted)
        self.assertFalse(self.capture(corners=self.corners[:19]).accepted)
        self.assertEqual(self.session.count, 0)

    def test_image_dimensions_dtype_and_frame_id_are_locked(self):
        self.assertFalse(self.capture(frame=self.frame.astype(np.float32)).accepted)
        self.assertTrue(self.capture().accepted)
        self.assertFalse(self.capture(frame=np.zeros((720, 1280, 3), np.uint8), stamp=2).accepted)
        self.assertFalse(self.capture(corners=self.corners + [50, 0], frame_id="other", stamp=3).accepted)

    def test_invalid_timestamps(self):
        for sec, nanos in [(0, 0), (-1, 0), (1, 1_000_000_000), (True, 0), (1.1, 0)]:
            with self.subTest(sec=sec, nanos=nanos):
                result = self.session.capture(self.frame, self.corners,
                                              {"frame_id": "front", "stamp_sec": sec,
                                               "stamp_nanosec": nanos}, 0.1)
                self.assertFalse(result.accepted)

    def test_existing_directory_file_and_symlink_are_refused(self):
        marker = self.session.output_dir / "keep.txt"
        marker.write_text("do not overwrite")
        file_path = self.root / "existing.txt"
        file_path.write_text("existing")
        link = self.root / "dangling"
        link.symlink_to(self.root / "missing")
        for path in [self.session.output_dir, file_path, link]:
            with self.subTest(path=path):
                with self.assertRaises(FileExistsError):
                    CaptureSession(path, self.board, "/camera")
        self.assertEqual(marker.read_text(), "do not overwrite")

    def test_invalid_board_configuration(self):
        valid = {"pattern": "checkerboard", "squares_columns": 6, "squares_rows": 5,
                 "inner_corners_columns": 5, "inner_corners_rows": 4, "square_size_mm": 65.0}
        bad_values = [("squares_columns", True), ("squares_rows", 5.0),
                      ("inner_corners_columns", 6), ("inner_corners_rows", 0),
                      ("square_size_mm", True), ("square_size_mm", float("inf")),
                      ("square_size_mm", 0), ("pattern", "circles")]
        path = self.root / "board.yaml"
        for key, value in bad_values:
            with self.subTest(key=key, value=value):
                path.write_text(yaml.safe_dump(dict(valid, **{key: value})))
                with self.assertRaises(ValueError):
                    BoardConfig.from_yaml(path)
        path.write_text(yaml.safe_dump(valid))
        self.assertEqual(BoardConfig.from_yaml(path).pattern, (5, 4))


if __name__ == "__main__":
    unittest.main()
