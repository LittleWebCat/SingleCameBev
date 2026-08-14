import unittest

from bev_depth.config import BevConfig, Extrinsics, Intrinsics
from bev_depth.geometry import ego_to_bev_pixel, pixel_depth_to_ego, scale_intrinsics


class GeometryTest(unittest.TestCase):
    def test_center_pixel_projects_forward(self):
        intrinsics = Intrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
        extrinsics = Extrinsics(height_m=1.4, pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0)
        x_m, z_m = pixel_depth_to_ego(640.0, 360.0, 20.0, intrinsics, extrinsics)
        self.assertAlmostEqual(x_m, 0.0, places=4)
        self.assertAlmostEqual(z_m, 20.0, places=4)

    def test_camera_mount_offsets_are_applied_in_ego_frame(self):
        intrinsics = Intrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0)
        extrinsics = Extrinsics(
            height_m=1.4,
            pitch_deg=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            lateral_offset_m=-0.6,
            forward_offset_m=1.2,
        )
        x_m, z_m = pixel_depth_to_ego(640.0, 360.0, 20.0, intrinsics, extrinsics)
        self.assertAlmostEqual(x_m, -0.6, places=4)
        self.assertAlmostEqual(z_m, 21.2, places=4)

    def test_intrinsics_scale_to_actual_frame_dimensions(self):
        intrinsics = Intrinsics(
            fx=1200.0,
            fy=1200.0,
            cx=960.0,
            cy=540.0,
            calibration_width=1920,
            calibration_height=1080,
        )
        scaled = scale_intrinsics(intrinsics, 800, 600)
        self.assertAlmostEqual(scaled.fx, 500.0)
        self.assertAlmostEqual(scaled.fy, 666.6666667)
        self.assertAlmostEqual(scaled.cx, 400.0)
        self.assertAlmostEqual(scaled.cy, 300.0)

    def test_intrinsics_reject_principal_point_outside_calibration_image(self):
        intrinsics = Intrinsics(
            fx=500.0,
            fy=500.0,
            cx=960.0,
            cy=300.0,
            calibration_width=800,
            calibration_height=600,
        )
        with self.assertRaisesRegex(ValueError, "cx=.*outside"):
            scale_intrinsics(intrinsics, 800, 600)

    def test_bev_pixel_clips_valid_point(self):
        bev = BevConfig(forward_range_m=50.0, lateral_range_m=20.0, pixels_per_meter=10)
        xy = ego_to_bev_pixel(0.0, 10.0, bev)
        self.assertEqual(xy, (100, 400))

    def test_bev_pixel_rejects_out_of_range(self):
        bev = BevConfig(forward_range_m=50.0, lateral_range_m=20.0, pixels_per_meter=10)
        self.assertIsNone(ego_to_bev_pixel(20.0, 10.0, bev))
        self.assertIsNone(ego_to_bev_pixel(0.0, 60.0, bev))

    def test_bev_accepts_configured_rear_area(self):
        bev = BevConfig(
            forward_range_m=80.0,
            lateral_range_m=160.0,
            pixels_per_meter=5,
            rear_range_m=80.0,
        )
        self.assertEqual(ego_to_bev_pixel(0.0, 0.0, bev), (400, 400))
        self.assertEqual(ego_to_bev_pixel(-40.0, -40.0, bev), (200, 600))
        self.assertIsNone(ego_to_bev_pixel(0.0, -81.0, bev))


if __name__ == "__main__":
    unittest.main()
