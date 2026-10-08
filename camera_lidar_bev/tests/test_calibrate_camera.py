"""Offline numerical and output safety checks for checkerboard camera fitting."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import calibrate_camera as calibration


def synthetic_views(board, count=25, noise_px=0.0):
    """Known 3D planar target poses spanning position, scale and both tilt axes."""
    matrix = np.array([[900.0, 0.0, 640.0], [0.0, 920.0, 480.0], [0.0, 0.0, 1.0]])
    distortion = np.array([-0.12, 0.035, 0.001, -0.002, 0.007])
    rng = np.random.default_rng(451)
    points = board.object_points()
    center = points.mean(axis=0)
    views = []
    for index in range(count):
        rotation = rng.uniform([-0.65, -0.65, -0.3], [0.65, 0.65, 0.3])
        target_center = rng.uniform([-0.25, -0.19, 0.55], [0.25, 0.19, 1.1])
        translation = target_center - cv2.Rodrigues(rotation)[0] @ center
        corners = cv2.projectPoints(points, rotation, translation, matrix, distortion)[0].reshape(-1, 2)
        corners = corners + rng.normal(0, noise_px, corners.shape)
        views.append(calibration.View(f"view_{index:04d}", (1280, 960), corners))
    return views, matrix, distortion


class CameraCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.board = calibration.BoardSpec(5, 4, 65.0)
        cls.views, cls.matrix, cls.distortion = synthetic_views(cls.board)
        cls.training, cls.held_out = calibration.split_views(cls.views)
        cls.fit = calibration.fit_camera(cls.training, cls.board)

    def test_known_camera_recovered_from_independent_poses(self):
        # Full focal/center recovery is the meaningful target, not low training loss alone.
        np.testing.assert_allclose(self.fit["camera_matrix"], self.matrix, atol=0.01)
        np.testing.assert_allclose(self.fit["distortion"], self.distortion, atol=0.002)
        self.assertLess(self.fit["rms_px"], 0.001)
        self.assertEqual(len(self.fit["per_view_rms_px"]), 20)
        self.assertEqual(len(self.fit["intrinsic_stddev"]), 18)

    def test_held_out_warped_corner_is_detected(self):
        clean = calibration.evaluate_held_out(self.held_out, self.board, self.matrix, self.distortion)
        self.assertLess(clean["rms_px"], 0.001)
        warped = self.held_out[0].corners.copy()
        warped[6] += [20, -16]
        bad_view = calibration.View("warped_independent", self.held_out[0].image_size, warped)
        result = calibration.evaluate_held_out([bad_view] + self.held_out[1:], self.board,
                                             self.matrix, self.distortion)
        self.assertGreater(result["max_view_rms_px"], 4.0)
        self.assertGreater(result["views"][0]["max_corner_error_px"], 15.0)

    def test_split_is_repeatable_disjoint_and_uses_all_views(self):
        training, held_out = calibration.split_views(list(reversed(self.views)))
        self.assertEqual([view.name for view in training], [view.name for view in self.training])
        self.assertEqual([view.name for view in held_out], [view.name for view in self.held_out])
        self.assertEqual(len(training), 20)
        self.assertEqual(len(held_out), 5)
        self.assertFalse({v.name for v in training} & {v.name for v in held_out})
        self.assertEqual({v.name for v in training + held_out}, {v.name for v in self.views})
        self.assertIn(self.views[0].name, {v.name for v in training})

    def test_report_does_not_promote_low_residual_to_validated(self):
        _, report = calibration.calibrate_views(self.training, self.held_out, self.board,
                                                include_stability=False, max_validation_rms_px=0.01)
        self.assertTrue(report["held_out"]["residual_screen_passed"])
        self.assertFalse(report["validation_passed"])
        self.assertFalse(report["fusion_ready"])
        self.assertEqual(report["calibration_status"], "candidate_unvalidated")
        self.assertEqual(len(report["fit"]["intrinsic_standard_deviation"]), 18)
        json.dumps(report, allow_nan=False)

    def test_camera_info_has_valid_monocular_ros_matrices(self):
        document = calibration.camera_info_yaml(self.fit)
        self.assertEqual(document["image_width"], 1280)
        self.assertEqual(document["image_height"], 960)
        self.assertEqual(document["distortion_model"], "plumb_bob")
        self.assertEqual(len(document["distortion_coefficients"]["data"]), 5)
        np.testing.assert_array_equal(np.array(document["rectification_matrix"]["data"]).reshape(3, 3), np.eye(3))
        projection = np.array(document["projection_matrix"]["data"]).reshape(3, 4)
        np.testing.assert_allclose(projection[:, :3], self.matrix, atol=0.01)
        np.testing.assert_array_equal(projection[:, 3], np.zeros(3))

    def test_invalid_board_view_and_camera_inputs_are_rejected(self):
        for arguments in ((1, 4, 65), (5, True, 65), (5, 4, 0), (5, 4, float("nan"))):
            with self.subTest(board=arguments), self.assertRaises(calibration.CalibrationError):
                calibration.BoardSpec(*arguments)
        for corners in (np.zeros((19, 2)), np.full((20, 2), np.nan),
                        np.full((20, 2), 2000), np.column_stack([np.arange(20), np.arange(20)])):
            with self.subTest(corners=corners.shape), self.assertRaises(calibration.CalibrationError):
                calibration.validate_views([calibration.View("bad", (1280, 960), corners)], self.board)
        with self.assertRaisesRegex(calibration.CalibrationError, "Mixed image"):
            calibration.validate_views([self.views[0], calibration.View("different", (640, 480), self.views[1].corners)], self.board)
        with self.assertRaisesRegex(calibration.CalibrationError, "Duplicate checkerboard"):
            calibration.validate_views([self.views[0], calibration.View("duplicate", (1280, 960), self.views[0].corners)], self.board)
        negative_focal = self.matrix.copy()
        negative_focal[0, 0] = -900
        with self.assertRaises(calibration.CalibrationError):
            calibration.validate_camera_model(negative_focal, self.distortion, (1280, 960))
        with self.assertRaises(calibration.CalibrationError):
            calibration.validate_camera_model(self.matrix, [0, 0, np.inf, 0, 0], (1280, 960))

    def test_insufficient_views_and_frontal_degeneracy_are_rejected(self):
        with self.assertRaisesRegex(calibration.CalibrationError, "training views"):
            calibration.split_views(self.views[:17])
        points = self.board.object_points()
        frontal = []
        for index in range(8):
            corners = cv2.projectPoints(points, np.zeros(3), np.array([-0.15 + index * 0.02, -0.1, 0.8]),
                                         self.matrix, np.zeros(5))[0]
            frontal.append(calibration.View(f"frontal{index}", (1280, 960), corners))
        with self.assertRaisesRegex(calibration.CalibrationError, "do not constrain"):
            calibration.fit_camera(frontal, self.board)

    def test_radial_fold_is_rejected_without_coefficient_magnitude_rules(self):
        # r_d=r-2*r^3 folds at r=sqrt(1/6), inside this camera image domain.
        diagnostic = calibration.radial_domain_diagnostic(self.matrix, [-2, 0, 0, 0, 0], (1280, 960))
        self.assertFalse(diagnostic["radial_mapping_valid_in_nominal_domain"])
        self.assertLess(diagnostic["radial_mapping_derivative"]["minimum"], 0)
        self.assertAlmostEqual(diagnostic["radial_mapping_derivative"]["zero_radii_within_domain"][0], np.sqrt(1 / 6))
        invalid_fit = dict(self.fit, distortion=np.array([-2, 0, 0, 0, 0]))
        with self.assertRaisesRegex(calibration.CalibrationError, "Refusing loadable"):
            calibration.camera_info_yaml(invalid_fit)
        # Large coefficients can remain monotone: do not reject based on magnitude alone.
        self.assertTrue(calibration.radial_domain_diagnostic(self.matrix, [10, 20, 0, 0, 30], (1280, 960))[
            "radial_mapping_valid_in_nominal_domain"])

    def test_radial_extremum_between_endpoints_is_checked(self):
        # derivative 1-10*s+10*s^2: positive at s=0 and s=1, negative near s=.5.
        matrix = np.array([[320., 0., 320.], [0., 320., 240.], [0., 0., 1.]])
        diagnostic = calibration.radial_domain_diagnostic(matrix, [-10/3, 2, 0, 0, 0], (640, 480))
        self.assertFalse(diagnostic["radial_mapping_valid_in_nominal_domain"])
        self.assertAlmostEqual(diagnostic["radial_mapping_derivative"]["minimum"], -1.5)
        self.assertAlmostEqual(diagnostic["radial_mapping_derivative"]["minimum_at_radius"], np.sqrt(.5))

    def test_training_subset_refits_never_use_held_out(self):
        result = calibration.subset_stability(self.training, self.board, self.fit)
        self.assertEqual(result["successful_subsets"], 3)
        held_names = {view.name for view in self.held_out}
        for subset in result["subsets"] + [result["largest_board_view_removed"]]:
            self.assertFalse(set(subset["training_images"]) & held_names)
        self.assertLess(result["maximum_absolute_focal_relative_change"], 0.001)
        self.assertLess(max(abs(x) for x in result["largest_board_view_removed"]["focal_relative_change"]), .001)

    def test_rejected_model_writes_report_without_loadable_yaml(self):
        _, report = calibration.calibrate_views(self.training, self.held_out, self.board, include_stability=False)
        report.update(candidate_yaml_permitted=False, calibration_status="rejected_camera_model",
                      model_rejection_reasons=["Radial mapping folds inside image."])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.yaml"
            with patch.object(calibration, "load_image_views", return_value=(self.views, {})), \
                    patch.object(calibration, "calibrate_views", return_value=(self.fit, report)), \
                    redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                code = calibration.main(["--images", "unused", "--columns", "5", "--rows", "4",
                                         "--square-size-mm", "65", "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertFalse(output.exists())
            saved = json.loads(output.with_suffix(".report.json").read_text())
            self.assertIsNone(saved["candidate_yaml"])
            self.assertEqual(saved["model_rejection_reasons"], ["Radial mapping folds inside image."])

    def test_existing_output_is_refused_before_loading_images(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.yaml"
            output.write_text("keep original\n")
            with patch.object(calibration, "load_image_views") as loader, redirect_stderr(io.StringIO()):
                code = calibration.main(["--images", "unused", "--columns", "5", "--rows", "4",
                                         "--square-size-mm", "65", "--output", str(output)])
            self.assertEqual(code, 1)
            loader.assert_not_called()
            self.assertEqual(output.read_text(), "keep original\n")

    def test_forced_rejection_removes_only_requested_file_or_symlink(self):
        for kind in ("file", "symlink", "directory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "candidate.yaml"
                target = Path(directory) / "underlying.yaml"
                if kind == "file":
                    output.write_text("old loadable camera model\n")
                elif kind == "symlink":
                    target.write_text("preserve symlink target\n")
                    output.symlink_to(target)
                else:
                    output.mkdir()
                report = {"candidate_yaml_permitted": False, "split": {},
                          "model_rejection_reasons": ["Radial mapping folds inside image."]}
                with patch.object(calibration, "load_image_views", return_value=(self.views, {})), \
                        patch.object(calibration, "calibrate_views", return_value=(self.fit, report)), \
                        redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                    code = calibration.main(["--images", "unused", "--columns", "5", "--rows", "4",
                                             "--square-size-mm", "65", "--output", str(output), "--force"])
                self.assertEqual(code, 2)
                saved = json.loads(output.with_suffix(".report.json").read_text())
                self.assertIsNone(saved["candidate_yaml"])
                self.assertEqual(saved["model_rejection_reasons"], ["Radial mapping folds inside image."])
                if kind == "directory":
                    self.assertTrue(output.is_dir())
                else:
                    self.assertFalse(output.exists())
                    self.assertFalse(output.is_symlink())
                if kind == "symlink":
                    self.assertEqual(target.read_text(), "preserve symlink target\n")

    def test_forced_rejection_preserves_old_yaml_if_report_cannot_be_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "candidate.yaml"
            output.write_text("preserve until rejection report is saved\n")
            report_directory = Path(directory) / "report_directory"
            report_directory.mkdir()
            report = {"candidate_yaml_permitted": False, "split": {},
                      "model_rejection_reasons": ["Radial mapping folds inside image."]}
            with patch.object(calibration, "load_image_views", return_value=(self.views, {})), \
                    patch.object(calibration, "calibrate_views", return_value=(self.fit, report)), \
                    redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                code = calibration.main(["--images", "unused", "--columns", "5", "--rows", "4",
                                         "--square-size-mm", "65", "--output", str(output),
                                         "--report", str(report_directory), "--force"])
            self.assertEqual(code, 1)
            self.assertEqual(output.read_text(), "preserve until rejection report is saved\n")


class SavedImageCompatibilityTests(unittest.TestCase):
    def test_capture_manifest_board_and_dimensions_are_checked(self):
        board = calibration.BoardSpec(5, 4, 65)
        image = np.full((300, 400, 3), 255, dtype=np.uint8)
        for row in range(5):
            for column in range(6):
                if (row + column) % 2 == 0:
                    cv2.rectangle(image, (50 + column * 40, 50 + row * 40),
                                  (89 + column * 40, 89 + row * 40), (0, 0, 0), -1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            cv2.imwrite(str(root / "images/frame_0001.png"), image)
            manifest = {"board": board.as_dict(), "camera_width_px": 400, "camera_height_px": 300,
                        "source_topic": "/camera/front/image_raw", "frame_id": "camera_front_optical_frame",
                        "saved_frames": [{"image_file": "images/frame_0001.png", "corners_px": "ignored"}]}
            (root / "session.json").write_text(json.dumps(manifest))
            views, metadata = calibration.load_image_views(str(root), board)
            self.assertEqual(len(views), 1)
            self.assertEqual(views[0].corners.reshape(-1, 2).shape, (20, 2))
            self.assertEqual(metadata["detected_view_count"], 1)
            self.assertEqual(metadata["source_topic"], manifest["source_topic"])
            with self.assertRaisesRegex(calibration.CalibrationError, "board dimensions"):
                calibration.load_image_views(str(root), calibration.BoardSpec(5, 4, 66))
            manifest["camera_width_px"] = 640
            (root / "session.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(calibration.CalibrationError, "dimensions disagree"):
                calibration.load_image_views(str(root), board)


if __name__ == "__main__":
    unittest.main()
