"""Verify strict model loading and native-size candidate undistortion without ROS."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from camera_preview import load_calibration


class CameraPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "candidate.yaml"
        self.document = {
            "image_width": 160, "image_height": 120, "distortion_model": "plumb_bob",
            "camera_matrix": {"rows": 3, "cols": 3,
                              "data": [120., 0., 82., 0., 118., 59., 0., 0., 1.]},
            "distortion_coefficients": {"rows": 1, "cols": 5,
                                        "data": [0., 0., 0., 0., 0.]},
        }

    def load(self, document=None):
        self.path.write_text(yaml.safe_dump(self.document if document is None else document))
        return load_calibration(self.path)

    def test_zero_distortion_preserves_pixels_and_native_pane_dimensions(self):
        preview = self.load()
        raw = np.random.default_rng(2).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        before = raw.copy()
        comparison = preview.comparison(raw)
        self.assertEqual(comparison.shape, (156, 320, 3))
        np.testing.assert_array_equal(comparison[36:, :160], raw)
        np.testing.assert_array_equal(comparison[36:, 160:], raw)
        np.testing.assert_array_equal(raw, before)
        self.assertTrue(np.any(comparison[:36]))

    def test_nonzero_distortion_maps_follow_original_intrinsics(self):
        self.document["distortion_coefficients"]["data"] = [-0.22, 0.04, 0.003, -0.002, -0.01]
        preview = self.load()
        # A corrected pixel samples the distorted projection of its original-K ray.
        pixels = np.array([[15, 15], [140, 18], [25, 105], [135, 98], [82, 59]], dtype=float)
        matrix = preview.camera_matrix
        rays = np.column_stack(((pixels[:, 0] - matrix[0, 2]) / matrix[0, 0],
                                (pixels[:, 1] - matrix[1, 2]) / matrix[1, 1],
                                np.ones(len(pixels))))
        expected, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), matrix,
                                        preview.distortion_coefficients)
        actual = np.array([[preview.map_x[int(y), int(x)], preview.map_y[int(y), int(x)]]
                           for x, y in pixels])
        np.testing.assert_allclose(actual, expected.reshape(-1, 2), atol=1e-5, rtol=0)
        self.assertGreater(np.linalg.norm(actual[0] - pixels[0]), 1)

    def test_source_size_mismatch_and_non_bgr_input_fail(self):
        preview = self.load()
        with self.assertRaisesRegex(ValueError, "Calibration is 160x120, but image is 320x240"):
            preview.comparison(np.zeros((240, 320, 3), dtype=np.uint8))
        for bad in (np.zeros((120, 160), dtype=np.uint8),
                    np.zeros((120, 160, 4), dtype=np.uint8),
                    np.zeros((120, 160, 3), dtype=float), None):
            with self.subTest(shape=getattr(bad, "shape", None)):
                with self.assertRaisesRegex(ValueError, "uint8 BGR"):
                    preview.comparison(bad)

    def test_invalid_dimensions_and_models_fail(self):
        for name, values in (("image_width", [0, -1, True, 160., "160", None]),
                             ("image_height", [0, -1, False, 120., "120", None]),
                             ("distortion_model", ["rational_polynomial", "equidistant", None])):
            for value in values:
                with self.subTest(name=name, value=value):
                    document = copy.deepcopy(self.document)
                    document[name] = value
                    with self.assertRaises(ValueError):
                        self.load(document)

    def test_malformed_matrices_and_nonfinite_coefficients_fail(self):
        for name in ("camera_matrix", "distortion_coefficients"):
            for change in (None, {"rows": True}, {"rows": 2}, {"cols": 8},
                           {"data": []}, {"data": [1]}, {"data": "bad"}):
                with self.subTest(name=name, change=change):
                    document = copy.deepcopy(self.document)
                    if change is None:
                        document[name] = None
                    else:
                        document[name].update(change)
                    with self.assertRaises(ValueError):
                        self.load(document)
            for value in (float("nan"), float("inf"), True, "0"):
                with self.subTest(name=name, value=value):
                    document = copy.deepcopy(self.document)
                    document[name]["data"][0] = value
                    with self.assertRaises(ValueError):
                        self.load(document)

    def test_non_pinhole_camera_matrix_fails(self):
        for index, value in ((0, 0), (4, -1), (1, 0.01), (3, 0.1), (6, 1), (8, 2)):
            with self.subTest(index=index):
                document = copy.deepcopy(self.document)
                document["camera_matrix"]["data"][index] = value
                with self.assertRaises(ValueError):
                    self.load(document)

    def test_missing_invalid_yaml_and_wrong_root_fail(self):
        with self.assertRaisesRegex(ValueError, "Cannot read calibration"):
            load_calibration(self.path)
        self.path.write_text("image_width: [unterminated")
        with self.assertRaisesRegex(ValueError, "Cannot read calibration"):
            load_calibration(self.path)
        for root in (None, [], "invalid"):
            with self.subTest(root=root):
                with self.assertRaisesRegex(ValueError, "must contain a mapping"):
                    self.load(root if root is not None else [])


if __name__ == "__main__":
    unittest.main()
