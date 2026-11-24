import os
import tempfile
import time
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

from bev_depth.config import InputConfig, Intrinsics, load_config
from bev_depth.depth import TensorRTDA3Backend
from bev_depth.multicam_infer import (
    LatestFrameCapture,
    OpenCamera,
    _fit_tile,
    camera_control_args,
    compose_dashboard,
    fuse_detections,
    load_bev_view,
    load_camera_specs,
    parse_args,
    project_to_view,
    render_multicam_frame,
    should_update_depth,
    tracks_to_bev_objects,
)
from bev_depth.object_tracking import TrackOutput
from bev_depth.types import Detection, ProjectedDetection


def projected(name, confidence, x_m, z_m, camera):
    detection = Detection(name, confidence, (10.0, 10.0, 20.0, 20.0))
    return ProjectedDetection(
        detection, z_m, x_m, z_m, (10, 10), "depth_map", camera
    )


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.delivered = Event()
        self.released = Event()

    def read(self):
        if self.frames:
            frame = self.frames.pop(0)
            if not self.frames:
                self.delivered.set()
            return True, frame
        self.released.wait(1.0)
        return False, None

    def release(self):
        self.released.set()


class StreamingCapture:
    """Delivers a frame every few ms, like a live camera, and records misuse."""

    def __init__(self):
        self.released = False
        self.read_after_release = False
        self.reading = False
        self.released_during_read = False

    def read(self):
        self.reading = True
        if self.released:
            self.read_after_release = True
        time.sleep(0.005)
        self.reading = False
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def release(self):
        if self.reading:
            self.released_during_read = True
        self.released = True


class MultiCameraConfigTest(unittest.TestCase):
    def test_temporal_tracking_cli_defaults_to_validated_rate(self):
        args = parse_args(["--temporal-tracking"])

        self.assertTrue(args.temporal_tracking)
        self.assertEqual(args.tracker_hz, 10.0)

    def test_causal_bev_option_enables_display_and_tracking(self):
        args = parse_args(["--display-causal-bev"])

        self.assertTrue(args.display_causal_bev)
        self.assertTrue(args.display_bev)
        self.assertTrue(args.temporal_tracking)

        alias_args = parse_args(["--display-casual-bev"])
        self.assertTrue(alias_args.display_causal_bev)

    def test_live_sensor_overlay_cli(self):
        args = parse_args(["--display-bev", "--show-sensors"])

        self.assertTrue(args.show_sensors)
        self.assertEqual(args.gps_port, "auto")
        self.assertEqual(args.imu_port, "auto")
        self.assertEqual(args.gps_baud, 115200)
        self.assertEqual(args.imu_baud, 115200)

    def test_sensor_logging_does_not_require_a_display(self):
        args = parse_args(["--save-sensors"])

        self.assertTrue(args.save_sensors)

    def test_verified_camera_order_and_center_only_traffic_color(self):
        path = "configs/three_camera_bev.yaml"
        config = load_config(path)
        cameras = load_camera_specs(path, config)

        self.assertEqual([camera.name for camera in cameras], ["left", "middle", "right"])
        self.assertEqual(
            [camera.source for camera in cameras],
            [
                f"/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:{port}:1.0-video-index0"
                for port in ("4.1.2.4", "4.1.2.1", "4.1.4.3")
            ],
        )
        self.assertEqual(
            [camera.traffic_color for camera in cameras], [False, True, False]
        )
        self.assertEqual(
            [camera.extrinsics.yaw_deg for camera in cameras], [-60.0, 0.0, 60.0]
        )
        # One shared lens estimate and capture mode for the three BEV cameras.
        self.assertEqual([camera.intrinsics.fx for camera in cameras], [320.0] * 3)
        self.assertEqual([camera.intrinsics.cx for camera in cameras], [320.0] * 3)
        self.assertEqual(
            [camera.intrinsics.calibration_width for camera in cameras], [640] * 3
        )
        self.assertEqual(
            [(c.input.width, c.input.height, c.input.fps) for c in cameras],
            [(640, 480, 30)] * 3,
        )
        self.assertEqual({camera.input.fourcc for camera in cameras}, {"MJPG"})
        self.assertEqual(config.bev.lateral_range_m, 80.0)
        self.assertEqual(config.bev.forward_range_m, 80.0)
        self.assertEqual(config.bev.rear_range_m, 40.0)
        self.assertEqual(config.detector.path, "models/yolo_vehicle_fp16.engine")

    def test_fusion_keeps_highest_confidence_nearby_duplicate(self):
        detections = [
            projected("car", 0.7, 1.0, 12.0, "left"),
            projected("car", 0.9, 1.4, 12.2, "middle"),
            projected("person", 0.8, 1.2, 12.1, "right"),
        ]

        fused = fuse_detections(detections, merge_distance_m=1.5)

        self.assertEqual(len(fused), 2)
        cars = [item for item in fused if item.detection.class_name == "car"]
        self.assertEqual(len(cars), 1)
        self.assertEqual(cars[0].camera_name, "middle")

    def test_latest_frame_capture_overwrites_old_frames(self):
        old_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        newest_frame = np.full((2, 2, 3), 7, dtype=np.uint8)
        fake = FakeCapture([old_frame, newest_frame])
        camera = OpenCamera(SimpleNamespace(name="left"), fake)
        capture = LatestFrameCapture([camera])

        try:
            capture.start()
            self.assertTrue(fake.delivered.wait(1.0))
            frames, sequences, captured_monotonic_ns = capture.read_latest()
        finally:
            capture.stop()

        np.testing.assert_array_equal(frames["left"], newest_frame)
        self.assertEqual(sequences, {"left": 2})
        self.assertGreater(captured_monotonic_ns["left"], 0)

    def test_stop_never_releases_a_camera_mid_read(self):
        fakes = [StreamingCapture() for _ in range(3)]
        capture = LatestFrameCapture(
            [OpenCamera(SimpleNamespace(name=f"cam{i}"), fake) for i, fake in enumerate(fakes)]
        )
        capture.start()
        capture.read_latest()
        capture.stop()

        for fake in fakes:
            self.assertTrue(fake.released)
            self.assertFalse(fake.released_during_read)
            self.assertFalse(fake.read_after_release)

    def test_bev_only_render_skips_annotations_and_dashboard(self):
        path = "configs/three_camera_bev.yaml"
        config = load_config(path)
        specs = load_camera_specs(path, config)
        frames = {
            spec.name: np.zeros((2, 2, 3), dtype=np.uint8) for spec in specs
        }
        projected_by_camera = {spec.name: [] for spec in specs}
        rendered_bev = np.zeros((4, 4, 3), dtype=np.uint8)

        with (
            patch("bev_depth.multicam_infer.render_bev", return_value=rendered_bev),
            patch("bev_depth.multicam_infer.draw_annotated_frame") as annotate,
            patch("bev_depth.multicam_infer.compose_dashboard") as dashboard,
        ):
            bev, rendered_dashboard = render_multicam_frame(
                frames,
                projected_by_camera,
                [],
                specs,
                config,
                include_dashboard=False,
            )

        self.assertIs(bev, rendered_bev)
        self.assertIsNone(rendered_dashboard)
        annotate.assert_not_called()
        dashboard.assert_not_called()

    def test_depth_updates_are_staggered_without_changing_cadence(self):
        cadence = 2

        initial = [
            should_update_depth(index, 0, cadence, has_depth=False)
            for index in range(3)
        ]
        frame_one = [
            should_update_depth(index, 1, cadence, has_depth=True)
            for index in range(3)
        ]
        frame_two = [
            should_update_depth(index, 2, cadence, has_depth=True)
            for index in range(3)
        ]

        self.assertEqual(initial, [True, True, True])
        self.assertEqual(frame_one, [False, True, False])
        self.assertEqual(frame_two, [True, False, True])

    def test_da3_and_rate_cap_cli(self):
        args = parse_args(
            ["--display", "--show-depth", "--da3-engine", "da3.engine", "--max-fps", "8"]
        )
        self.assertEqual(args.da3_engine, "da3.engine")
        self.assertEqual(args.max_fps, 8.0)
        self.assertTrue(args.show_depth)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--show-depth"])  # the depth row lives in the --display dashboard
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--max-fps", "0"])

    def test_track_objects_flag(self):
        self.assertTrue(parse_args(["--track-objects"]).track_objects)
        self.assertFalse(parse_args([]).track_objects)

    def test_confirmed_tracks_become_labelled_bev_markers(self):
        config = load_config("configs/three_camera_bev.yaml")
        last = projected("car", 0.9, 1.0, 10.0, "left")
        tracks = [TrackOutput(7, "car", 1.2, 10.5, 0.0, 2.0, True, 5, last)]

        (marker,) = tracks_to_bev_objects(tracks, config)

        self.assertEqual(marker.track_id, 7)
        self.assertEqual((marker.x_m_lateral, marker.z_m_forward), (1.2, 10.5))
        self.assertEqual(marker.depth_source, "track_coasting")
        self.assertEqual(marker.camera_name, "left")
        self.assertIsNotNone(marker.bev_xy)

    def test_dashboard_with_depth_puts_bev_beside_the_camera_grid(self):
        path = "configs/three_camera_bev.yaml"
        config = load_config(path)
        specs = load_camera_specs(path, config)
        annotated = {spec.name: np.zeros((480, 640, 3), np.uint8) for spec in specs}
        depth = {spec.name: np.full((480, 640), 5.0, np.float32) for spec in specs}
        bev = np.zeros((600, 400, 3), np.uint8)

        with_depth = compose_dashboard(annotated, specs, bev, depth)

        # Camera row + depth row (300 px each), BEV beside them at the same height.
        self.assertEqual(with_depth.shape[0], 600)
        self.assertEqual(with_depth.shape[1], 3 * 400 + 400)

    def test_front_cameras_use_the_drivers_view_layout(self):
        path = "configs/three_camera_bev.yaml"
        config = load_config(path)
        specs = load_camera_specs(path, config)
        annotated = {
            "left": np.full((720, 1280, 3), 50, np.uint8),
            "middle": np.full((768, 1024, 3), 150, np.uint8),
            "right": np.full((720, 1280, 3), 250, np.uint8),
        }
        bev = np.full((600, 400, 3), 245, np.uint8)

        dashboard = compose_dashboard(annotated, specs, bev)

        # Middle 800x600 over left/right 400x225 = 800x825; BEV scaled to 550x825.
        self.assertEqual(dashboard.shape, (825, 1350, 3))
        self.assertEqual(int(dashboard[300, 400, 0]), 150)   # middle camera, top
        self.assertEqual(int(dashboard[700, 200, 0]), 50)    # left camera, bottom-left
        self.assertEqual(int(dashboard[700, 600, 0]), 250)   # right camera, bottom-right
        self.assertEqual(int(dashboard[500, 1100, 0]), 245)  # BEV on the right


class BevViewTest(unittest.TestCase):
    def test_view_only_changes_what_is_drawn(self):
        path = "configs/three_camera_bev.yaml"
        config = load_config(path)
        view = load_bev_view(path, config.bev)

        # The tracked/matrix area stays 80 m ahead, 40 m behind, 80 m wide.
        self.assertEqual(
            (config.bev.forward_range_m, config.bev.rear_range_m, config.bev.lateral_range_m),
            (80.0, 40.0, 80.0),
        )
        self.assertEqual(
            (view.forward_range_m, view.rear_range_m, view.lateral_range_m, view.pixels_per_meter),
            (40.0, 20.0, 40.0, 10),
        )

        inside = projected("car", 0.9, 5.0, 30.0, "middle")     # 5 m right, 30 m ahead
        far = projected("car", 0.9, 0.0, 60.0, "middle")        # beyond 40 m ahead
        wide = projected("person", 0.9, -25.0, 10.0, "left")    # beyond 20 m left
        shown = project_to_view([inside, far, wide], view)

        # 400 x 600 px image: x = (5 + 20) * 10, y = (40 - 30) * 10.
        self.assertEqual(shown[0].bev_xy, (250, 100))
        self.assertIsNone(shown[1].bev_xy)
        self.assertIsNone(shown[2].bev_xy)

    def test_missing_view_block_shows_the_whole_bev(self):
        path = "configs/thor_1080p.yaml"
        config = load_config(path)
        self.assertEqual(load_bev_view(path, config.bev), config.bev)


class FiveCameraTest(unittest.TestCase):
    def setUp(self):
        self.path = "configs/five_camera_bev.yaml"
        self.config = load_config(self.path)
        self.specs = load_camera_specs(self.path, self.config)

    def test_config_adds_both_back_cameras(self):
        self.assertEqual(
            [s.name for s in self.specs], ["left", "middle", "right", "left_back", "right_back"]
        )
        self.assertEqual(
            [s.extrinsics.yaw_deg for s in self.specs], [-90.0, 0.0, 90.0, -120.0, 120.0]
        )
        self.assertEqual([s.traffic_color for s in self.specs], [False, True, False, False, False])
        self.assertEqual(len({s.source for s in self.specs}), 5)

    def test_dashboard_grid_with_bev_on_the_right(self):
        annotated = {s.name: np.zeros((720, 1280, 3), np.uint8) for s in self.specs}
        depth = {s.name: np.full((720, 1280), 5.0, np.float32) for s in self.specs}
        bev = np.zeros((600, 400, 3), np.uint8)

        # 5 cameras: 2 rows of 400x300 tiles, BEV 400 px wide at 600 px high.
        self.assertEqual(compose_dashboard(annotated, self.specs, bev).shape, (600, 1600, 3))
        # With depth: 4 rows shrink to 300x225 tiles to fit 900 px; BEV 600x900.
        self.assertEqual(
            compose_dashboard(annotated, self.specs, bev, depth).shape, (900, 1500, 3)
        )


class DashboardTileTest(unittest.TestCase):
    def test_widescreen_frames_are_letterboxed_not_stretched(self):
        frame = np.full((720, 1280, 3), 255, np.uint8)
        tile = _fit_tile(frame)

        self.assertEqual(tile.shape, (300, 400, 3))
        # 16:9 into 4:3 -> 225 image rows centered, black bars above and below.
        self.assertEqual(int(tile[:37].max()), 0)
        self.assertEqual(int(tile[-37:].max()), 0)
        self.assertEqual(int(tile[150].min()), 255)


class CameraExposureTest(unittest.TestCase):
    def mode(self, **fields):
        base = dict(width=640, height=480, fps=30, fourcc="MJPG", camera_index_backend="v4l2")
        return InputConfig(**{**base, **fields})

    def test_fixed_exposure_switches_to_manual_before_setting_the_time(self):
        args = camera_control_args(self.mode(exposure=300, gain=40, power_line_hz=60))
        self.assertEqual(
            args,
            ["power_line_frequency=2", "auto_exposure=1", "exposure_time_absolute=300", "gain=40"],
        )

    def test_auto_exposure_is_selected_explicitly(self):
        # Cameras keep the last exposure mode another program gave them.
        self.assertEqual(camera_control_args(self.mode()), ["auto_exposure=3"])

    def test_bad_power_line_frequency_is_rejected(self):
        with self.assertRaises(ValueError):
            camera_control_args(self.mode(power_line_hz=55))

    def test_config_exposure_and_per_camera_override(self):
        path = "configs/three_camera_bev.yaml"
        config = load_config(path)
        self.assertEqual(
            (config.input.exposure, config.input.gain, config.input.power_line_hz), (None, 40, 60)
        )
        raw = yaml.safe_load(open(path))
        raw["cameras"][1]["input"] = {"exposure": "auto", "gain": 10}
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            yaml.safe_dump(raw, handle)
        try:
            specs = load_camera_specs(handle.name, load_config(handle.name))
        finally:
            os.unlink(handle.name)
        self.assertEqual([s.input.exposure for s in specs], [None, None, None])
        self.assertEqual([s.input.gain for s in specs], [40, 10, 40])
        self.assertEqual([s.input.power_line_hz for s in specs], [60, 60, 60])


class DA3FocalTest(unittest.TestCase):
    def test_focal_is_scaled_to_the_network_input(self):
        backend = SimpleNamespace(net_width=504, net_height=378)
        narrow = Intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0,
                            calibration_width=640, calibration_height=480)
        focal = TensorRTDA3Backend.focal_at_network_size(backend, narrow)
        self.assertAlmostEqual(focal, 500.0 * 504 / 640)


if __name__ == "__main__":
    unittest.main()
