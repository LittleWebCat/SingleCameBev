"""Offline checks for external calibration consistency and preservation."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import import_camera_lidar_calibration as importer


def example_documents():
    camera = {
        "model": "pinhole", "image_size": [1024, 768],
        "K": [[1410.0, 0.0, 548.0], [0.0, 1400.0, 378.0], [0.0, 0.0, 1.0]],
        "D": [-0.44, 0.76, 0.0035, -0.0002, -1.76],
        "rms_px": 0.32, "per_view_rms_px": [0.2, 0.3], "images": ["board1.png", "board2.png"],
        "board": {"cols": 5, "rows": 4, "square_m": 0.065},
    }
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.array([0.01, -0.19, -0.47])
    forward, inverse = np.eye(4), np.eye(4)
    forward[:3, :3], forward[:3, 3] = rotation, translation
    inverse[:3, :3], inverse[:3, 3] = rotation.T, -rotation.T @ translation
    errors = np.array([1., 2., 3., 4., 5., 6., 9., 11.])
    inliers = [0, 1, 2, 3, 4, 5, 6]
    external = {
        "direction": "lidar_to_camera", "units": "m", "R": rotation.tolist(), "t": translation.tolist(),
        "T_camera_from_lidar": forward.tolist(), "T_lidar_from_camera": inverse.tolist(),
        "point_count": len(errors), "point_errors_px": errors.tolist(), "inlier_indices": inliers,
        "all_points_rms_px": float(np.sqrt(np.mean(errors ** 2))),
        "inlier_rms_px": float(np.sqrt(np.mean(errors[inliers] ** 2))), "ransac_threshold_px": 8.0,
        "camera": copy.deepcopy(camera), "pair_files": ["pose1.json", "pose4.json", "pose4.json"],
    }
    return camera, external


class ExternalCalibrationImportTests(unittest.TestCase):
    def setUp(self):
        self.camera, self.external = example_documents()

    def test_candidate_keeps_native_geometry_without_promoting_validation(self):
        camera, transform, report = importer.candidate_documents(self.camera, self.external)
        self.assertEqual((camera["image_width"], camera["image_height"]), (1024, 768))
        self.assertEqual(camera["camera_matrix"]["data"], np.array(self.camera["K"]).reshape(-1).tolist())
        self.assertEqual(camera["distortion_coefficients"]["data"], self.camera["D"])
        self.assertEqual(camera["distortion_model"], "plumb_bob")
        np.testing.assert_array_equal(np.array(camera["rectification_matrix"]["data"]).reshape(3, 3), np.eye(3))
        projection = np.array(camera["projection_matrix"]["data"]).reshape(3, 4)
        np.testing.assert_array_equal(projection[:, :3], self.camera["K"])
        np.testing.assert_array_equal(projection[:, 3], np.zeros(3))
        self.assertFalse(camera["calibrated"])
        self.assertFalse(transform["calibrated"])
        self.assertEqual(transform["transforms"]["T_camera_from_lidar"]["source_frame"], "UNKNOWN")
        self.assertEqual(transform["transforms"]["T_camera_from_lidar"]["target_frame"], "camera_front_optical_frame")
        for field in ("validation_passed", "fusion_ready", "hardware_changed", "intrinsics_rescaled", "tf_published"):
            self.assertFalse(report[field])
        self.assertIsNone(report["same_sensor_current_mounting_user_confirmed"])
        self.assertEqual(report["extrinsics"]["duplicate_pair_files"], {"pose4.json": 2})
        self.assertEqual(report["extrinsics"]["inlier_count"], 7)
        self.assertTrue(any("Repeated pair_files" in warning for warning in report["warnings"]))
        self.assertTrue(any("after refinement" in warning for warning in report["warnings"]))
        json.dumps(report, allow_nan=False)

    def test_point_mapping_inverse_and_quaternion_have_correct_direction(self):
        _, transform, report = importer.candidate_documents(self.camera, self.external, "os_sensor")
        candidate = transform["transforms"]["T_camera_from_lidar"]
        self.assertEqual(candidate["source_frame"], "os_sensor")
        quaternion = np.array(candidate["quaternion_xyzw"])
        np.testing.assert_allclose(quaternion, [0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-12)
        point = np.array([1.2, 0.8, -0.3, 1.0])
        mapped = np.array(candidate["matrix"]) @ point
        expected = np.array(self.external["R"]) @ point[:3] + self.external["t"]
        np.testing.assert_allclose(mapped[:3], expected)
        np.testing.assert_allclose(np.array(transform["T_lidar_from_camera"]) @ mapped, point)
        self.assertLess(report["extrinsics"]["inverse_product_max_abs_error"], 1e-12)

    def test_quaternion_covers_half_turns_without_trace_division(self):
        for axis in range(3):
            diagonal = [-1., -1., -1.]
            diagonal[axis] = 1.
            result = importer._quaternion_xyzw(np.diag(diagonal))
            expected = [0., 0., 0., 0.]
            expected[axis] = 1.
            np.testing.assert_allclose(result, expected)

    def test_inconsistent_frames_units_matrices_and_camera_are_rejected(self):
        mutations = [
            lambda doc: doc.update(direction="camera_to_lidar"),
            lambda doc: doc.update(units="mm"),
            lambda doc: doc["R"][0].__setitem__(0, 0.1),
            lambda doc: doc["T_camera_from_lidar"][0].__setitem__(3, 0.2),
            lambda doc: doc["T_lidar_from_camera"][2].__setitem__(3, 0.0),
            lambda doc: doc["T_camera_from_lidar"][3].__setitem__(0, 0.1),
            lambda doc: doc["camera"]["K"][0].__setitem__(0, 1410.0000001),
            lambda doc: doc["camera"]["D"].__setitem__(0, -0.43),
            lambda doc: doc["camera"].update(image_size=[640, 480]),
        ]
        for index, mutation in enumerate(mutations):
            document = copy.deepcopy(self.external)
            mutation(document)
            with self.subTest(mutation=index), self.assertRaises(importer.ImportError):
                importer.validate_documents(self.camera, document)
        reflection = copy.deepcopy(self.external)
        reflection["R"] = np.diag([1., 1., -1.]).tolist()
        with self.assertRaisesRegex(importer.ImportError, "proper orthonormal"):
            importer.validate_documents(self.camera, reflection)

    def test_corrupt_residual_summaries_and_indices_are_rejected(self):
        mutations = [
            lambda doc: doc.update(all_points_rms_px=0.1),
            lambda doc: doc.update(inlier_rms_px=0.1),
            lambda doc: doc.update(point_count=9),
            lambda doc: doc.update(inlier_indices=[0, 1, 2, 2]),
            lambda doc: doc.update(inlier_indices=[0, 1, 2, 8]),
            lambda doc: doc.update(inlier_indices=[0, 1, 2, True]),
            lambda doc: doc["point_errors_px"].__setitem__(0, -1),
            lambda doc: doc["point_errors_px"].__setitem__(0, float("nan")),
        ]
        for index, mutation in enumerate(mutations):
            document = copy.deepcopy(self.external)
            mutation(document)
            with self.subTest(mutation=index), self.assertRaises(importer.ImportError):
                importer.validate_documents(self.camera, document)

    def test_non_pinhole_invalid_focal_and_coerced_numbers_are_rejected(self):
        mutations = [
            lambda doc: doc.update(model="fisheye"),
            lambda doc: doc.update(image_size=[1024., 768]),
            lambda doc: doc.update(image_size=[True, 768]),
            lambda doc: doc["K"][0].__setitem__(0, -1),
            lambda doc: doc["K"][0].__setitem__(1, 0.1),
            lambda doc: doc["K"][0].__setitem__(0, "1410"),
            lambda doc: doc["K"][2].__setitem__(2, True),
            lambda doc: doc.update(D=[0, 0, 0, 0]),
            lambda doc: doc["D"].__setitem__(0, float("inf")),
        ]
        for index, mutation in enumerate(mutations):
            document = copy.deepcopy(self.camera)
            mutation(document)
            with self.subTest(mutation=index), self.assertRaises(importer.ImportError):
                importer.validate_camera(document)

    def test_import_preserves_source_bytes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera_raw = (json.dumps(self.camera, indent=3) + "\n\n").encode()
            external_raw = (json.dumps(self.external) + "\n").encode()
            intrinsics, extrinsics, output = root / "i.json", root / "e.json", root / "new"
            intrinsics.write_bytes(camera_raw)
            extrinsics.write_bytes(external_raw)
            report = importer.import_calibration(intrinsics, extrinsics, output)
            self.assertEqual((output / "front.json").read_bytes(), camera_raw)
            self.assertEqual((output / "front_external.json").read_bytes(), external_raw)
            self.assertEqual(intrinsics.read_bytes(), camera_raw)
            self.assertEqual(extrinsics.read_bytes(), external_raw)
            self.assertEqual(report["source_files"]["intrinsics"]["sha256"], hashlib.sha256(camera_raw).hexdigest())
            candidate = yaml.safe_load((output / report["candidate_camera_file"]).read_text())
            self.assertEqual(candidate["image_width"], 1024)
            first_outputs = {path.name: path.read_bytes() for path in output.iterdir()}
            with self.assertRaisesRegex(importer.ImportError, "refusing to overwrite"):
                importer.import_calibration(intrinsics, extrinsics, output)
            self.assertEqual({path.name: path.read_bytes() for path in output.iterdir()}, first_outputs)

    def test_invalid_cli_and_duplicate_json_keys_leave_no_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intrinsics, extrinsics, output = root / "i.json", root / "e.json", root / "new"
            intrinsics.write_text(json.dumps(self.camera))
            broken = copy.deepcopy(self.external)
            broken["all_points_rms_px"] = 0
            extrinsics.write_text(json.dumps(broken))
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                result = importer.main(["--intrinsics", str(intrinsics), "--extrinsics", str(extrinsics), "--output-dir", str(output)])
            self.assertEqual(result, 1)
            self.assertFalse(output.exists())
            intrinsics.write_text('{"model":"pinhole", "model":"fisheye"}')
            with self.assertRaisesRegex(importer.ImportError, "Duplicate JSON key"):
                importer.import_calibration(intrinsics, extrinsics, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
