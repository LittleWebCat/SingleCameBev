"""Verify calibration observation integrity without ROS or connected sensors."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from extrinsic_capture import BoardConfig, ExtrinsicCaptureSession, decode_point_cloud


def cloud_fixture(big_endian=False, doubles=False):
    scalar = 8 if doubles else 4
    datatype = 8 if doubles else 7
    width, height, point_step = 2, 2, 3 * scalar + 8
    row_step = width * point_step + 7
    metadata = {"header": {"frame_id": "os_lidar", "stamp_sec": 10, "stamp_nanosec": 0},
                "width": width, "height": height, "point_step": point_step,
                "row_step": row_step, "is_bigendian": big_endian, "is_dense": False,
                "fields": [{"name": name, "offset": index * scalar, "datatype": datatype,
                            "count": 1} for index, name in enumerate(("x", "y", "z"))]
                + [{"name": "intensity", "offset": 3 * scalar, "datatype": 4, "count": 1}]}
    payload = bytearray([0xA5] * (height * row_step))
    xyz = np.array([[1, 2, 3], [4, 5, 6], [np.nan, 0, 1], [0, 0, 0]], dtype=np.float64)
    endian = ">" if big_endian else "<"
    for index, name in enumerate(("x", "y", "z")):
        view = np.ndarray((height, width), dtype=endian + ("f8" if doubles else "f4"),
                          buffer=payload, offset=index * scalar, strides=(row_step, point_step))
        view[:] = xyz[:, index].reshape(height, width)
    intensities = np.ndarray((height, width), dtype=endian + "u2", buffer=payload,
                             offset=3 * scalar, strides=(row_step, point_step))
    intensities[:] = [[100, 200], [300, 400]]
    return metadata, payload, xyz


class DecodeTests(unittest.TestCase):
    def test_both_byte_orders_and_scalar_widths_preserve_point_order_and_padding(self):
        for big_endian in (False, True):
            for doubles in (False, True):
                with self.subTest(big_endian=big_endian, doubles=doubles):
                    metadata, payload, expected = cloud_fixture(big_endian, doubles)
                    cloud = decode_point_cloud(metadata, payload)
                    np.testing.assert_allclose(cloud.xyz, expected, equal_nan=True)
                    np.testing.assert_allclose(cloud.organized_xyz, expected.reshape(2, 2, 3),
                                               equal_nan=True)
                    np.testing.assert_array_equal(cloud.intensity, [100, 200, 300, 400])
                    np.testing.assert_array_equal(cloud.valid_mask, [True, True, False, False])

    def test_optional_intensity_and_all_layout_fields_validated(self):
        metadata, payload, _ = cloud_fixture()
        metadata["fields"].pop()
        self.assertIsNone(decode_point_cloud(metadata, payload).intensity)
        cases = [("width", 0), ("height", True), ("point_step", 0),
                 ("row_step", metadata["width"] * metadata["point_step"] - 1),
                 ("is_bigendian", 0), ("is_dense", None), ("fields", [])]
        for key, value in cases:
            with self.subTest(key=key):
                altered = copy.deepcopy(metadata)
                altered[key] = value
                with self.assertRaises(ValueError):
                    decode_point_cloud(altered, payload)
        for altered_payload in (payload[:-1], payload + b"x", None):
            with self.assertRaises(ValueError):
                decode_point_cloud(metadata, altered_payload)

    def test_invalid_point_fields_cannot_read_outside_layout(self):
        metadata, payload, _ = cloud_fixture()
        for key, value in [("offset", metadata["point_step"]), ("datatype", 99),
                           ("datatype", 6), ("count", 2), ("count", 0), ("name", "y")]:
            with self.subTest(key=key, value=value):
                altered = copy.deepcopy(metadata)
                altered["fields"][0][key] = value
                with self.assertRaises(ValueError):
                    decode_point_cloud(altered, payload)
        missing = copy.deepcopy(metadata)
        missing["fields"] = missing["fields"][1:]
        with self.assertRaisesRegex(ValueError, "x is missing"):
            decode_point_cloud(missing, payload)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.calibration = self.root / "camera.yaml"
        k = [500., 0., 320., 0., 505., 240., 0., 0., 1.]
        document = {"image_width": 640, "image_height": 480, "distortion_model": "plumb_bob",
                    "camera_matrix": {"rows": 3, "cols": 3, "data": k},
                    "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0.] * 5},
                    "rectification_matrix": {"rows": 3, "cols": 3,
                                             "data": np.eye(3).reshape(-1).tolist()},
                    "projection_matrix": {"rows": 3, "cols": 4,
                                          "data": [500., 0., 320., 0., 0., 505., 240., 0.,
                                                   0., 0., 1., 0.]}}
        self.calibration.write_text(yaml.safe_dump(document))
        self.board = BoardConfig(5, 4, 65.)
        self.session = ExtrinsicCaptureSession(self.root / "dataset", self.board, self.calibration)
        self.frame = np.random.default_rng(7).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        self.corners = np.array([(160 + 40 * x, 120 + 40 * y)
                                 for y in range(4) for x in range(5)], dtype=np.float32)
        self.header = {"frame_id": "front_optical", "stamp_sec": 10, "stamp_nanosec": 100_000_000,
                       "timestamp_source": "host_receipt_after_read"}
        self.info = dict(copy.deepcopy(self.session.expected_camera_info), header=dict(self.header))
        self.cloud_metadata, self.payload, _ = cloud_fixture()

    def capture(self, **changes):
        arguments = {"frame": self.frame, "corners": self.corners,
                     "image_header": self.header, "camera_info_metadata": self.info,
                     "cloud_metadata": self.cloud_metadata, "cloud_data": self.payload,
                     "image_age_s": 0.1, "cloud_age_s": 0.2, "board_stationary_confirmed": True}
        arguments.update(changes)
        return self.session.capture(**arguments)

    def test_saves_raw_pixels_payload_calibration_and_truthful_manifest(self):
        result = self.capture()
        self.assertTrue(result.accepted, result.reason)
        np.testing.assert_array_equal(cv2.imread(str(result.path)), self.frame)
        manifest = json.loads((self.session.output_dir / "session.json").read_text())
        self.assertEqual(manifest["calibration_status"], "unmeasured")
        self.assertNotIn("camera_from_lidar", manifest)
        self.assertEqual(manifest["board"]["square_size_mm"], 65.)
        self.assertEqual(manifest["board"]["squares_columns"], 6)
        self.assertEqual(manifest["pairing"]["qualification"], "header_pairing_heuristic_only")
        calibration = manifest["camera_calibration"]
        self.assertEqual(calibration["sha256"], hashlib.sha256(self.calibration.read_bytes()).hexdigest())
        self.assertEqual((self.session.output_dir / calibration["copied_file"]).read_bytes(),
                         self.calibration.read_bytes())
        pair = manifest["saved_pairs"][0]
        self.assertEqual(pair["absolute_header_difference_s"], 0.1)
        self.assertEqual(pair["camera_info"], self.info)
        self.assertIsNone(pair["board_pose_camera_from_board"])
        self.assertTrue(pair["board_stationary_user_confirmed"])
        self.assertEqual((self.session.output_dir / pair["point_cloud_file"]).read_bytes(), self.payload)
        cloud = json.loads((self.session.output_dir / pair["point_cloud_metadata_file"]).read_text())
        for key in ("width", "height", "point_step", "row_step", "fields", "is_bigendian"):
            self.assertEqual(cloud[key], self.cloud_metadata[key])
        self.assertEqual(cloud["payload_sha256"], hashlib.sha256(self.payload).hexdigest())
        preview = np.load(self.session.output_dir / pair["point_cloud_preview_file"])
        np.testing.assert_array_equal(preview["point_indices"], [0, 1])
        self.assertTrue(np.isfinite(preview["xyz"]).all())
        self.assertLessEqual(len(preview["xyz"]), 4096)

    def test_confirmation_is_required_by_default(self):
        result = self.session.capture(self.frame, self.corners, self.header, self.info,
                                      self.cloud_metadata, self.payload, 0.1, 0.2)
        self.assertFalse(result.accepted)
        self.assertIn("stationary", result.reason)
        self.assertEqual(self.session.count, 0)
        self.assertEqual(list((self.session.output_dir / "observations").iterdir()), [])

    def test_exact_image_info_header_and_calibration_matching(self):
        for key, value in [("width", 1280), ("width", 640.0), ("distortion_model", "equidistant"),
                           ("k", [0.] * 9), ("d", [float("nan")] * 5)]:
            with self.subTest(key=key):
                info = copy.deepcopy(self.info)
                info[key] = value
                self.assertFalse(self.capture(camera_info_metadata=info).accepted)
        for key, value in [("frame_id", "other"), ("stamp_nanosec", 100_000_001),
                           ("stamp_sec", 0)]:
            info = copy.deepcopy(self.info)
            info["header"][key] = value
            self.assertFalse(self.capture(camera_info_metadata=info).accepted)
        self.assertTrue(self.capture().accepted)

    def test_pairing_boundary_freshness_and_nonzero_stamps(self):
        cloud = copy.deepcopy(self.cloud_metadata)
        cloud["header"]["stamp_nanosec"] = 250_000_000
        self.assertTrue(self.capture(cloud_metadata=cloud).accepted)
        self.session = ExtrinsicCaptureSession(self.root / "second", self.board, self.calibration)
        cloud["header"]["stamp_nanosec"] += 1
        self.assertFalse(self.capture(cloud_metadata=cloud).accepted)
        for key in ("image_age_s", "cloud_age_s"):
            for value in (1.001, -0.1, float("nan"), True):
                self.assertFalse(self.capture(**{key: value}).accepted)
        for sec, ns in [(0, 0), (True, 0), (1, 1_000_000_000), (-1, 0)]:
            cloud = copy.deepcopy(self.cloud_metadata)
            cloud["header"].update(stamp_sec=sec, stamp_nanosec=ns)
            self.assertFalse(self.capture(cloud_metadata=cloud).accepted)

    def test_all_zero_or_nonfinite_cloud_and_invalid_corners_rejected(self):
        cloud, payload, _ = cloud_fixture()
        for x in (0., float("nan")):
            for field in cloud["fields"][:3]:
                array = np.ndarray((2, 2), dtype="<f4", buffer=payload, offset=field["offset"],
                                   strides=(cloud["row_step"], cloud["point_step"]))
                array[:] = x
            self.assertFalse(self.capture(cloud_data=payload).accepted)
        for corners in (self.corners[:19], np.full((20, 2), np.nan), self.corners + [640, 0],
                        self.corners * 0.1):
            self.assertFalse(self.capture(corners=corners).accepted)
        self.assertFalse(self.capture(frame=self.frame.astype(np.float32)).accepted)
        self.assertFalse(self.capture(frame=self.frame[:200]).accepted)
        self.assertEqual(self.session.count, 0)

    def test_pose_reverse_repeats_timestamp_reuse_and_frame_change_rejected(self):
        self.assertTrue(self.capture().accepted)
        header = dict(self.header, stamp_sec=11)
        info = dict(self.info, header=header)
        cloud = copy.deepcopy(self.cloud_metadata)
        cloud["header"]["stamp_sec"] = 11
        self.assertFalse(self.capture(corners=self.corners[::-1], image_header=header,
                                     camera_info_metadata=info, cloud_metadata=cloud).accepted)
        self.assertFalse(self.capture(corners=self.corners + [50, 0]).accepted)
        cloud["header"]["frame_id"] = "other_lidar"
        self.assertFalse(self.capture(corners=self.corners + [50, 0], image_header=header,
                                     camera_info_metadata=info, cloud_metadata=cloud).accepted)
        cloud["header"]["frame_id"] = "os_lidar"
        self.assertTrue(self.capture(corners=self.corners + [50, 0], image_header=header,
                                    camera_info_metadata=info, cloud_metadata=cloud).accepted)
        self.assertEqual(self.session.count, 2)

    def test_hint_provenance_is_saved_and_bound_to_actual_cloud_point(self):
        hint = {"xyz_m": [1., 2., 3.], "range_m": float(np.sqrt(14)), "point_index": 0,
                "header": dict(self.cloud_metadata["header"]), "azimuth_deg": 63.435,
                "elevation_deg": 53.301, "board_return_verified": True}
        for altered in [dict(hint, point_index=1), dict(hint, range_m=0.),
                        dict(hint, xyz_m=[0., 0., 0.]), dict(hint, azimuth_deg=float("nan")),
                        dict(hint, azimuth_deg=0.), dict(hint, elevation_deg=0.),
                        dict(hint, header=dict(hint["header"], stamp_sec=11))]:
            self.assertFalse(self.capture(lidar_target_hint=altered).accepted)
        self.assertTrue(self.capture(lidar_target_hint=hint).accepted)
        manifest = json.loads((self.session.output_dir / "session.json").read_text())
        saved = manifest["saved_pairs"][0]["lidar_target_hint"]
        self.assertEqual(saved["source"], "user_selected_return")
        self.assertFalse(saved["board_return_verified"])
        self.assertEqual(saved["point_index"], 0)

    def test_manifest_failure_rolls_back_observation_and_allows_retry(self):
        with patch.object(self.session, "_write_manifest", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.capture()
        self.assertEqual(self.session.count, 0)
        self.assertEqual(list((self.session.output_dir / "observations").iterdir()), [])
        self.assertEqual(list(self.session.output_dir.glob(".pose-*")), [])
        self.assertTrue(self.capture().accepted)

    def test_existing_paths_never_overwritten_and_invalid_config_never_creates_dataset(self):
        original = (self.session.output_dir / "session.json").read_bytes()
        for path in [self.session.output_dir, self.calibration]:
            with self.assertRaises(FileExistsError):
                ExtrinsicCaptureSession(path, self.board, self.calibration)
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            ExtrinsicCaptureSession(dangling, self.board, self.calibration)
        self.assertEqual((self.session.output_dir / "session.json").read_bytes(), original)
        invalid = self.root / "invalid"
        with self.assertRaises(ValueError):
            ExtrinsicCaptureSession(invalid, self.board, self.calibration, min_board_span_px=-1)
        self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()
