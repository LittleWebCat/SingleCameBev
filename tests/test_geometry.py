import unittest

from bev_depth.config import BevConfig, Extrinsics, Intrinsics
from bev_depth.geometry import ego_to_bev_pixel, pixel_depth_to_ego


class GeometryTest(unittest.TestCase):
    def test_center_pixel_projects_forward(self):
        intrinsics = Intrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
        extrinsics = Extrinsics(height_m=1.4, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0)
        x_m, z_m = pixel_depth_to_ego(640.0, 360.0, 20.0, intrinsics, extrinsics)
        self.assertAlmostEqual(x_m, 0.0, places=4)
        self.assertAlmostEqual(z_m, 20.0, places=4)

    def test_bev_pixel_clips_valid_point(self):
        bev = BevConfig(forward_range_m=50.0, lateral_range_m=20.0, pixels_per_meter=10)
        xy = ego_to_bev_pixel(0.0, 10.0, bev)
        self.assertEqual(xy, (100, 400))

    def test_bev_pixel_rejects_out_of_range(self):
        bev = BevConfig(forward_range_m=50.0, lateral_range_m=20.0, pixels_per_meter=10)
        self.assertIsNone(ego_to_bev_pixel(20.0, 10.0, bev))
        self.assertIsNone(ego_to_bev_pixel(0.0, 60.0, bev))


if __name__ == "__main__":
    unittest.main()
