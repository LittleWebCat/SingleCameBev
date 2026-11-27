"""Numerical reference checks for native RGB association and measured BEV axes."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from camera_bev import BevExtent, associate_camera_colors, render_bev, render_camera_bev


class CameraBevTests(unittest.TestCase):
    def setUp(self):
        self.camera = {"model": "pinhole", "image_size": [100, 100],
                       "K": [[10., 0., 50.], [0., 10., 50.], [0., 0., 1.]],
                       "D": [0., 0., 0., 0., 0.]}
        self.external = {"direction": "lidar_to_camera", "units": "m",
                         "T_camera_from_lidar": np.eye(4).tolist()}
        self.sensor = np.eye(4)
        self.image = np.full((100, 100, 3), [13, 74, 199], dtype=np.uint8)

    def sensor_forward_camera(self):
        # os_sensor +X -> optical +Z, +Y -> optical -X, +Z -> optical -Y.
        transform = np.eye(4)
        transform[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
        self.external["T_camera_from_lidar"] = transform.tolist()

    def test_b_composition_maps_geometry_and_camera_color_in_original_order(self):
        self.sensor_forward_camera()
        self.sensor = np.diag([-1., -1., 1., 1.])
        self.sensor[2, 3] = .038195
        xyz = np.array([[3., 0., 0.], [-3., .3, .2], [0., 0., 0.], [np.nan, 0., 1.]])
        # Sensor point [3,-.3,.238195] maps to optical [.3,-.238195,3].
        self.image[49, 51] = [2, 90, 201]
        before_xyz, before_image = xyz.copy(), self.image.copy()
        returns = associate_camera_colors(self.image, xyz, self.camera, self.external, self.sensor)
        np.testing.assert_array_equal(returns.projection.point_indices, [1])
        np.testing.assert_array_equal(returns.projection.pixel_indices, [[51, 49]])
        np.testing.assert_allclose(returns.xyz_sensor[1], [3., -.3, .238195], atol=1e-12)
        np.testing.assert_array_equal(returns.colors_bgr[1], [2, 90, 201])
        np.testing.assert_array_equal(returns.colors_bgr[[0, 2, 3]], np.full((3, 3), 255))
        np.testing.assert_array_equal(returns.valid_mask, [True, True, False, False])
        np.testing.assert_array_equal(returns.colored_mask, [False, True, False, False])
        np.testing.assert_array_equal(returns.T_camera_from_lidar,
                                      np.asarray(self.external["T_camera_from_lidar"]) @ self.sensor)
        np.testing.assert_array_equal(xyz, before_xyz)
        np.testing.assert_array_equal(self.image, before_image)

    def test_only_nearest_measured_camera_return_gets_bgr_color(self):
        xyz = np.array([[0., 0., 6.], [0., 0., 2.], [0., 0., 2.], [0., 0., -2.],
                        [100., 0., 1.], [np.nan, 0., 1.], [0., 0., 0.], [.2, 0., 2.]])
        self.image[50, 50] = [3, 31, 231]
        self.image[50, 51] = [203, 17, 2]
        returns = associate_camera_colors(self.image, xyz, self.camera, self.external, self.sensor)
        np.testing.assert_array_equal(returns.projection.point_indices, [1, 7])
        np.testing.assert_array_equal(returns.colors_bgr[[1, 7]], [[3, 31, 231], [203, 17, 2]])
        np.testing.assert_array_equal(returns.colors_bgr[[0, 2, 3, 4]], np.full((4, 3), 255))
        self.assertEqual(returns.colored_count, 2)
        self.assertFalse(returns.valid_mask[5])
        self.assertFalse(returns.valid_mask[6])

    def test_folded_radial_ray_cannot_take_a_real_returns_color(self):
        self.camera["D"] = [-1., 0., 0., 0., 0.]
        xyz = np.array([[1., 0., 1.], [0., 0., 2.]])
        returns = associate_camera_colors(self.image, xyz, self.camera, self.external, self.sensor)
        np.testing.assert_array_equal(returns.colored_mask, [False, True])
        self.assertEqual(returns.projection.radial_domain_excluded_count, 1)
        np.testing.assert_array_equal(returns.colors_bgr[0], [255, 255, 255])

    def test_float_bounds_precede_nearest_pixel_rounding_and_upper_edge_clamp(self):
        xyz = np.array([[4.975, 0., 1.], [5., 0., 1.], [-5.001, 0., 1.]])
        # 99.75 rounds to 100 then clamps to99; 100 and -.01 are excluded.
        self.image[50, 99] = [1, 2, 233]
        returns = associate_camera_colors(self.image, xyz, self.camera, self.external, self.sensor)
        np.testing.assert_array_equal(returns.projection.pixel_indices, [[99, 50]])
        np.testing.assert_array_equal(returns.colored_mask, [True, False, False])
        np.testing.assert_array_equal(returns.colors_bgr[0], [1, 2, 233])

    def test_top_down_forward_is_up_and_sensor_left_is_left(self):
        self.sensor_forward_camera()
        xyz = np.array([[1.2, 1.2, .5], [3.2, -1.2, .5]])
        self.image[46, 40] = [22, 99, 240]
        self.image[48, 54] = [201, 8, 25]
        result = render_camera_bev(self.image, xyz, self.camera, self.external, self.sensor,
                                   BevExtent(0, 4, -2, 2, 1), draw_grid=False)
        by_index = dict(zip(result.point_indices, result.pixel_indices.tolist()))
        self.assertEqual(by_index, {0: [0, 2], 1: [3, 0]})
        np.testing.assert_array_equal(result.image[2, 0], [22, 99, 240])
        np.testing.assert_array_equal(result.image[0, 3], [201, 8, 25])
        self.assertEqual(result.image.shape, (4, 4, 3))
        self.assertEqual(result.colored_count, 2)
        self.assertEqual(np.count_nonzero(np.any(result.image != 0, axis=2)), 2)

    def test_half_open_metric_boundaries_and_invalid_points(self):
        xyz = np.array([[0., -2., 1.], [4., 0., 1.], [2., 2., 1.], [-.0001, 0., 1.],
                        [np.nextafter(4., 0), np.nextafter(2., 0), 1.], [np.nan, 0., 1.]])
        returns = associate_camera_colors(self.image, xyz, self.camera, self.external, self.sensor)
        result = render_bev(returns, BevExtent(0, 4, -2, 2, 1), draw_grid=False)
        self.assertEqual(set(result.point_indices), {0, 4})
        by_index = dict(zip(result.point_indices, result.pixel_indices.tolist()))
        self.assertEqual(by_index, {0: [3, 3], 4: [0, 0]})
        self.assertEqual(result.in_extent_count, 2)

    def test_higher_uncolored_return_does_not_inherit_lower_colored_geometry(self):
        self.sensor_forward_camera()
        xyz = np.array([[1.2, .2, .1], [1.3, .3, 100.], [1.3, .3, 100.]])
        returns = associate_camera_colors(self.image, xyz, self.camera, self.external, self.sensor)
        np.testing.assert_array_equal(returns.colored_mask, [True, False, False])
        result = render_bev(returns, BevExtent(0, 4, -2, 2, 1), draw_grid=False)
        np.testing.assert_array_equal(result.point_indices, [1])
        np.testing.assert_array_equal(result.image[2, 1], [255, 255, 255])
        self.assertEqual(result.in_extent_count, 3)
        self.assertEqual(result.colored_count, 0)

    def test_empty_cloud_and_reject_native_size_transform_or_extent_errors(self):
        empty = render_camera_bev(self.image, np.empty((0, 3)), self.camera, self.external,
                                  self.sensor, BevExtent(0, 4, -2, 2, 1), draw_grid=False)
        self.assertEqual(empty.point_indices.shape, (0,))
        self.assertEqual(empty.pixel_indices.shape, (0, 2))
        self.assertFalse(np.any(empty.image))
        with self.assertRaisesRegex(ValueError, "native dimensions"):
            associate_camera_colors(self.image[::2, ::2], np.array([[1., 0., 1.]]),
                                     self.camera, self.external, self.sensor)
        with self.assertRaisesRegex(ValueError, "proper rotation"):
            associate_camera_colors(self.image, np.array([[1., 0., 1.]]), self.camera,
                                     self.external, np.diag([-1., 1., 1., 1.]))
        for arguments in ({"x_min_m": 20}, {"resolution_m": 0}, {"y_min_m": np.nan},
                          {"resolution_m": 1e-9}, {"resolution_m": True}):
            with self.assertRaises(ValueError):
                BevExtent(**arguments)


if __name__ == "__main__":
    unittest.main()
