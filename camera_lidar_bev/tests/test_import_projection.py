"""Independent numerical checks for raw-image diagnostic frame hypotheses."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from import_projection import (project_cloud, radial_domain_radius_squared,
                               render_calibration_preview, render_two_hypotheses)


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.K = np.array([[100., 0., 320.], [0., 120., 240.], [0., 0., 1.]])
        self.D = np.zeros(5)
        self.T = np.eye(4)

    def test_four_known_rays_and_forward_transform_direction(self):
        # R sends source +X to optical +Z, +Y to optical -X, +Z to optical -Y.
        self.T[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
        self.T[:3, 3] = [.1, .2, .5]
        xyz = np.array([[1.5, .1, .2], [1.5, -.9, .2], [1.5, .1, -1.8], [1.5, .1, 2.2]])
        result = project_cloud(xyz, self.K, self.D, self.T, (640, 480))
        by_index = dict(zip(result.point_indices, result.pixels_px))
        for index, expected in enumerate(([320, 240], [370, 240], [320, 360], [320, 120])):
            np.testing.assert_allclose(by_index[index], expected, atol=1e-12)
        wrong = project_cloud(xyz, self.K, self.D, np.linalg.inv(self.T), (640, 480))
        self.assertNotEqual(set(wrong.point_indices), set(result.point_indices))

    def test_raw_distortion_uses_radial_and_tangential_terms(self):
        # x=.2, y=.1; compute the model directly, independently of OpenCV.
        self.D = np.array([.2, -.1, .01, -.02, .03])
        x, y = .2, .1
        r2 = x*x + y*y
        radial = 1 + .2*r2 - .1*r2*r2 + .03*r2*r2*r2
        xd = x*radial + 2*.01*x*y - .02*(r2 + 2*x*x)
        yd = y*radial + .01*(r2 + 2*y*y) - 2*.02*x*y
        result = project_cloud(np.array([[.4, .2, 2.]]), self.K, self.D, self.T, (640, 480))
        np.testing.assert_allclose(result.pixels_px[0], [100*xd + 320, 120*yd + 240], atol=1e-12)
        self.assertFalse(np.allclose(result.pixels_px[0], [340, 252], atol=.01))

    def test_original_indices_nearest_optical_depth_and_invalid_returns(self):
        xyz = np.array([[np.nan, 0, 2], [0, 0, 0], [0, 0, -2], [0, 0, 6],
                        [0, 0, 2], [0, 0, 2], [100, 0, 2], [np.inf, 0, 2]])
        before = xyz.copy()
        result = project_cloud(xyz, self.K, self.D, self.T, (640, 480))
        self.assertEqual(result.input_count, 8)
        self.assertEqual(result.valid_count, 5)
        self.assertEqual(result.front_count, 4)
        self.assertEqual(result.in_frame_count, 3)
        np.testing.assert_array_equal(result.point_indices, [4])
        np.testing.assert_array_equal(result.camera_depth_m, [2.])
        np.testing.assert_array_equal(xyz, before)

    def test_folded_rays_cannot_occlude_the_real_central_return(self):
        # r=1 maps to the center for k1=-1, yet is beyond the monotonic branch.
        # Without the domain guard its nearer depth would hide the genuine center.
        distortion = np.array([-1., 0., 0., 0., 0.])
        xyz = np.array([[1., 0., 1.], [0., 0., 2.], [.3, 0., 1.], [.6, 0., 1.]])
        result = project_cloud(xyz, self.K, distortion, self.T, (640, 480))
        self.assertAlmostEqual(radial_domain_radius_squared(distortion), 1/3)
        self.assertEqual(result.front_count, 4)
        self.assertEqual(result.radial_domain_excluded_count, 2)
        np.testing.assert_array_equal(result.point_indices, [1, 2])
        np.testing.assert_array_equal(result.camera_depth_m, [2., 1.])
        np.testing.assert_allclose(result.pixels_px, [[320., 240.], [347.3, 240.]], atol=1e-12)

    def test_high_order_radial_domain_and_no_guessed_fov_cap(self):
        distortion = np.array([0., 0., 0., 0., -1.])
        self.assertAlmostEqual(radial_domain_radius_squared(distortion), (1/7)**(1/3))
        # Both rays would have in-image positive radial projections, but r=.8
        # lies beyond the cubic radial derivative's first zero.
        result = project_cloud(np.array([[.5, 0., 1.], [.8, 0., 1.]]),
                               self.K, distortion, self.T, (640, 480))
        np.testing.assert_array_equal(result.point_indices, [0])
        self.assertEqual(result.radial_domain_excluded_count, 1)
        monotonic = np.array([.1, 0., 0., 0., 0.])
        self.assertTrue(np.isinf(radial_domain_radius_squared(monotonic)))
        result = project_cloud(np.array([[2., 0., 1.]]), self.K, monotonic, self.T, (640, 480))
        np.testing.assert_array_equal(result.point_indices, [0])
        np.testing.assert_allclose(result.pixels_px, [[600., 240.]])

    def test_empty_and_bad_matrices(self):
        empty = project_cloud(np.empty((0, 3)), self.K, self.D, self.T, (640, 480))
        self.assertEqual(empty.pixels_px.shape, (0, 2))
        for matrix in (np.diag([1., 1., -1., 1.]), np.diag([2., 1., 1., 1.]), np.zeros((4, 4))):
            with self.assertRaises(ValueError):
                project_cloud(np.array([[1., 1., 2.]]), self.K, self.D, matrix, (640, 480))
        with self.assertRaises(ValueError):
            project_cloud(np.array([[1., 1., 2.]]), self.K, self.D, self.T, (640., 480))
        with self.assertRaises(ValueError):
            project_cloud(np.array([["1", "1", "2"]]), self.K, self.D, self.T, (640, 480))

    def test_explicit_sensor_composition_changes_measured_source_indices(self):
        K = np.array([[100., 0., 512.], [0., 120., 384.], [0., 0., 1.]])
        camera = {"model": "pinhole", "image_size": [1024, 768], "K": K.tolist(), "D": self.D.tolist()}
        external = np.eye(4)
        external[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
        extrinsics = {"direction": "lidar_to_camera", "units": "m", "T_camera_from_lidar": external.tolist()}
        sensor = np.diag([-1., -1., 1., 1.])
        sensor[2, 3] = .038195
        image = np.full((768, 1024, 3), [7, 91, 163], dtype=np.uint8)
        xyz = np.array([[3., 0., 0.], [-3., 0., 0.]])
        original_image, original_xyz = image.copy(), xyz.copy()
        result = render_two_hypotheses(image, xyz, camera, extrinsics, sensor)
        np.testing.assert_array_equal(result.projections["direct_os_lidar"].point_indices, [0])
        np.testing.assert_array_equal(result.projections["via_os_sensor"].point_indices, [1])
        self.assertAlmostEqual(result.projections["via_os_sensor"].pixels_px[0, 1],
                               384 - 120*.038195/3)
        self.assertEqual(result.image.shape, (828, 2048, 3))
        # Camera colors remain intact away from monochrome markers and headers.
        np.testing.assert_array_equal(result.image[160, 100], [7, 91, 163])
        np.testing.assert_array_equal(image, original_image)
        np.testing.assert_array_equal(xyz, original_xyz)
        with self.assertRaises(ValueError):
            render_two_hypotheses(image[::2, ::2], xyz, camera, extrinsics, sensor)

    def test_selected_b_is_single_native_pane_with_the_same_composed_projection(self):
        K = np.array([[100., 0., 512.], [0., 120., 384.], [0., 0., 1.]])
        camera = {"model": "pinhole", "image_size": [1024, 768], "K": K.tolist(), "D": self.D.tolist()}
        external = np.eye(4)
        external[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
        extrinsics = {"direction": "lidar_to_camera", "units": "m", "T_camera_from_lidar": external.tolist()}
        sensor = np.diag([-1., -1., 1., 1.])
        sensor[2, 3] = .038195
        image = np.full((768, 1024, 3), [7, 91, 163], dtype=np.uint8)
        xyz = np.array([[3., 0., 0.], [-3., .3, .2]])
        comparison = render_two_hypotheses(image, xyz, camera, extrinsics, sensor)
        selected = render_calibration_preview(image, xyz, camera, extrinsics, sensor, view="b")
        self.assertEqual(selected.view, "b")
        self.assertEqual(selected.image.shape, (828, 1024, 3))
        self.assertEqual(set(selected.projections), {"via_os_sensor"})
        np.testing.assert_array_equal(selected.projections["via_os_sensor"].point_indices, [1])
        np.testing.assert_allclose(selected.projections["via_os_sensor"].pixels_px,
                                   [[522., 384 - 120*.238195/3]], atol=1e-12)
        np.testing.assert_array_equal(selected.image[60:], comparison.image[60:, 1024:])
        np.testing.assert_array_equal(selected.hypotheses["via_os_sensor"], external @ sensor)
        np.testing.assert_array_equal(image[100, 100], [7, 91, 163])
        with self.assertRaisesRegex(ValueError, "compare or b"):
            render_calibration_preview(image, xyz, camera, extrinsics, sensor, view="automatic")


if __name__ == "__main__":
    unittest.main()
