from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import (
    AppConfig,
    BevConfig,
    Extrinsics,
    InputConfig,
    Intrinsics,
    load_config,
    optional_int,
)
from .depth import create_depth_backend, sample_detection_depth
from .detector import DetectorBackend, create_detector
from .geometry import ego_to_bev_pixel, pixel_depth_to_ego, scale_intrinsics
from .io import ensure_output_dirs, save_bgr_image, write_jsonl_record
from .object_tracking import BEVObjectTracker, CameraTrackIds, TrackOutput
from .profiler import Profiler
from .render import draw_annotated_frame, render_bev
from .sensors import (
    LiveSensorMonitor,
    draw_live_sensor_panel,
    write_matrix_sensor_record,
)
from .semantic_grid import render_semantic_grid, write_semantic_metadata
from .temporal_tracking import (
    DEFAULT_TRACKER_DIR,
    TemporalBEVTracker,
)
from .types import Detection, ProjectedDetection


@dataclass(frozen=True)
class CameraSpec:
    name: str
    source: int | str
    traffic_color: bool
    extrinsics: Extrinsics
    intrinsics: Intrinsics
    input: InputConfig


@dataclass
class OpenCamera:
    spec: CameraSpec
    capture: cv2.VideoCapture


@dataclass
class _LatestFrameState:
    frame: np.ndarray | None = None
    sequence: int = 0
    captured_monotonic_ns: int = 0
    error: RuntimeError | None = None


class LatestFrameCapture:
    """Continuously drain cameras and expose only each camera's newest frame."""

    def __init__(self, cameras: list[OpenCamera]) -> None:
        self.cameras = cameras
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._states = {
            camera.spec.name: _LatestFrameState() for camera in self.cameras
        }
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("Camera capture threads have already been started")
        for camera in self.cameras:
            thread = threading.Thread(
                target=self._capture_loop,
                args=(camera,),
                name=f"capture-{camera.spec.name}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def _capture_loop(self, camera: OpenCamera) -> None:
        name = camera.spec.name
        while not self._stop.is_set():
            ok, frame = camera.capture.read()
            if self._stop.is_set():
                break
            if not ok or frame is None:
                with self._condition:
                    self._states[name].error = RuntimeError(
                        f"Failed to read a frame from {name}"
                    )
                    self._stop.set()
                    self._condition.notify_all()
                break
            with self._condition:
                state = self._states[name]
                state.frame = frame
                state.sequence += 1
                state.captured_monotonic_ns = time.monotonic_ns()
                self._condition.notify_all()

    def read_latest(
        self,
        previous_sequences: dict[str, int] | None = None,
        timeout_s: float = 5.0,
    ) -> tuple[dict[str, np.ndarray], dict[str, int], dict[str, int]]:
        """Return a fresh frame from every camera, dropping overwritten frames."""

        previous = previous_sequences or {}
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                errors = [state.error for state in self._states.values() if state.error]
                if errors:
                    raise errors[0]
                ready = all(
                    state.frame is not None
                    and state.sequence > previous.get(name, 0)
                    for name, state in self._states.items()
                )
                if ready:
                    frames = {
                        name: state.frame
                        for name, state in self._states.items()
                        if state.frame is not None
                    }
                    sequences = {
                        name: state.sequence for name, state in self._states.items()
                    }
                    captured_monotonic_ns = {
                        name: state.captured_monotonic_ns
                        for name, state in self._states.items()
                    }
                    return frames, sequences, captured_monotonic_ns
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    missing = [
                        name
                        for name, state in self._states.items()
                        if state.frame is None
                        or state.sequence <= previous.get(name, 0)
                    ]
                    raise TimeoutError(
                        "Timed out waiting for fresh camera frames: "
                        + ", ".join(missing)
                    )
                self._condition.wait(remaining)

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        # Join before releasing: releasing a VideoCapture while its thread is
        # still inside read() frees the buffers under that read and crashes the
        # process with a segfault. A streaming camera returns within one frame.
        for thread in self._threads:
            thread.join(timeout=2.0)
        stuck = {
            camera.spec.name
            for camera, thread in zip(self.cameras, self._threads)
            if thread.is_alive()
        }
        for camera in self.cameras:
            if camera.spec.name in stuck:
                # Blocked in read() on a camera that stopped sending; leave it
                # for process exit instead of freeing it under the read.
                print(f"warning: {camera.spec.name} capture did not stop; not releasing it")
                continue
            camera.capture.release()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse object detection and monocular depth from multiple cameras into one BEV"
    )
    parser.add_argument("--config", default="configs/three_camera_bev.yaml")
    parser.add_argument("--save-dir", default="runs/three_camera_bev")
    parser.add_argument("--display", action="store_true", help="Show camera row and fused BEV")
    parser.add_argument("--display-bev", action="store_true", help="Show only the fused BEV")
    parser.add_argument(
        "--skip-missing-cameras",
        action="store_true",
        help="Run with the configured cameras that are connected; warn about missing device paths",
    )
    parser.add_argument(
        "--display-cameras",
        action="store_true",
        help="Show only the camera grid with detections (no fused BEV panel)",
    )
    parser.add_argument(
        "--display-causal-bev",
        "--display-casual-bev",
        dest="display_causal_bev",
        action="store_true",
        help="Show the temporal tracker's raw-vs-causal BEV viewer",
    )
    parser.add_argument(
        "--show-sensors",
        action="store_true",
        help="Overlay live GPS and IMU values in the displayed dashboard/BEV",
    )
    parser.add_argument(
        "--save-sensors",
        action="store_true",
        help="Save frame-aligned GPS/IMU data beside the BEV matrices",
    )
    parser.add_argument(
        "--gps-port",
        default="auto",
        help="GPS serial port or 'auto' (default: Septentrio USB1/ttyACM0)",
    )
    parser.add_argument(
        "--imu-port",
        default="auto",
        help="IMU serial port or 'auto' (default: Xsens/ttyUSB0)",
    )
    parser.add_argument("--gps-baud", type=int, default=115200)
    parser.add_argument("--imu-baud", type=int, default=115200)
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--batch-detector-model",
        default=None,
        help="Fixed-batch detector engine for all configured cameras",
    )
    parser.add_argument(
        "--temporal-tracking",
        action="store_true",
        help="Run the causal Kalman tracker on the fused semantic BEV",
    )
    parser.add_argument(
        "--tracker-dir",
        type=Path,
        default=DEFAULT_TRACKER_DIR,
        help="Directory containing temporal_matrix_cleaner.py and target_speed.py",
    )
    parser.add_argument(
        "--tracker-hz",
        type=float,
        default=10.0,
        help="Maximum temporal-tracker update rate (default: validated 10 Hz)",
    )
    parser.add_argument(
        "--da3-engine",
        default=None,
        help="Use this Depth Anything 3 Metric TensorRT engine instead of the config's depth model",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=None,
        help="Cap the processing rate (lowers power draw on a weak supply)",
    )
    parser.add_argument(
        "--show-depth",
        action="store_true",
        help="Add each camera's depth map to the --display dashboard",
    )
    parser.add_argument(
        "--track-objects",
        action="store_true",
        help="Persistent object IDs: ByteTrack per camera plus a global BEV Kalman tracker",
    )
    args = parser.parse_args(argv)
    if args.max_fps is not None and args.max_fps <= 0.0:
        parser.error("--max-fps must be positive")
    if args.display_cameras:
        args.display = True
    if args.show_depth and not args.display:
        parser.error("--show-depth requires --display")
    if args.display_causal_bev:
        if args.display:
            parser.error("--display-causal-bev cannot be combined with --display")
        args.display_bev = True
        args.temporal_tracking = True
    if args.max_frames is not None and args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    if args.tracker_hz <= 0.0:
        parser.error("--tracker-hz must be positive")
    if args.show_sensors and not (args.display or args.display_bev):
        parser.error("--show-sensors requires --display or --display-bev")
    return args


def _source_value(value: object) -> int | str:
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text


def load_camera_specs(config_path: str | Path, base: AppConfig) -> list[CameraSpec]:
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    cameras = raw.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("Multi-camera config requires a non-empty 'cameras' list")

    specs: list[CameraSpec] = []
    names: set[str] = set()
    sources: set[int | str] = set()
    for index, item in enumerate(cameras):
        if not isinstance(item, dict):
            raise ValueError(f"cameras[{index}] must be a mapping")
        name = str(item.get("name", f"camera{index}")).strip().lower()
        source = _source_value(item.get("source", index))
        if not name or name in names:
            raise ValueError(f"Camera names must be non-empty and unique: {name!r}")
        if source in sources:
            raise ValueError(f"Camera sources must be unique: {source!r}")
        names.add(name)
        sources.add(source)

        pose = item.get("extrinsics", {}) or {}
        # Optional per-camera lens; unset fields fall back to camera.intrinsics.
        lens = item.get("intrinsics", {}) or {}
        intrinsics = replace(
            base.intrinsics,
            **{
                key: type(getattr(base.intrinsics, key))(lens[key])
                for key in ("fx", "fy", "cx", "cy", "calibration_width", "calibration_height")
                if key in lens
            },
        )
        # Optional per-camera capture mode; unset fields fall back to input.
        mode = item.get("input", {}) or {}
        capture_mode = replace(
            base.input,
            **{
                key: type(getattr(base.input, key))(mode[key])
                for key in ("width", "height", "fps", "fourcc")
                if key in mode
            },
            **{
                field: optional_int(mode[key])
                for key, field in (
                    ("exposure", "exposure"),
                    ("gain", "gain"),
                    ("power_line_frequency", "power_line_hz"),
                )
                if key in mode
            },
        )
        specs.append(
            CameraSpec(
                name=name,
                source=source,
                traffic_color=bool(item.get("traffic_color", False)),
                extrinsics=replace(
                    base.extrinsics,
                    height_m=float(pose.get("height_m", base.extrinsics.height_m)),
                    pitch_deg=float(pose.get("pitch_deg", base.extrinsics.pitch_deg)),
                    yaw_deg=float(pose.get("yaw_deg", base.extrinsics.yaw_deg)),
                    roll_deg=float(pose.get("roll_deg", base.extrinsics.roll_deg)),
                    lateral_offset_m=float(
                        pose.get("lateral_offset_m", base.extrinsics.lateral_offset_m)
                    ),
                    forward_offset_m=float(
                        pose.get("forward_offset_m", base.extrinsics.forward_offset_m)
                    ),
                ),
                intrinsics=intrinsics,
                input=capture_mode,
            )
        )
    if any(spec.traffic_color for spec in specs) and base.traffic_detector is None:
        raise ValueError("A traffic_detector model is required when traffic_color is enabled")
    return specs


def _is_camera_source(source: int | str) -> bool:
    return isinstance(source, int) or str(source).startswith(("/dev/video", "/dev/v4l/"))


_POWER_LINE_MENU = {0: 0, 50: 1, 60: 2}  # UVC power_line_frequency menu values


def camera_control_args(mode: InputConfig) -> list[str]:
    """v4l2-ctl controls for exposure, gain and anti-flicker, in the order UVC needs.

    Auto exposure is selected explicitly when no fixed exposure is set, because
    a camera keeps the last mode any program gave it until it is unplugged.
    The exposure time is only accepted after switching to manual mode.
    """
    controls = []
    if mode.power_line_hz is not None:
        if mode.power_line_hz not in _POWER_LINE_MENU:
            raise ValueError("power_line_frequency must be 0 (off), 50 or 60")
        controls.append(f"power_line_frequency={_POWER_LINE_MENU[mode.power_line_hz]}")
    if mode.exposure is None:
        controls.append("auto_exposure=3")
    else:
        controls += ["auto_exposure=1", f"exposure_time_absolute={mode.exposure}"]
    if mode.gain is not None:
        controls.append(f"gain={mode.gain}")
    return controls


def apply_camera_controls(source: int | str, mode: InputConfig) -> None:
    device = f"/dev/video{source}" if isinstance(source, int) else str(source)
    if shutil.which("v4l2-ctl") is None:
        print("warning: v4l2-ctl not found; exposure, gain and anti-flicker are not applied")
        return
    for control in camera_control_args(mode):
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, f"--set-ctrl={control}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            print(f"warning: {device}: could not set {control}: {result.stderr.strip()}")


def open_cameras(specs: list[CameraSpec], config: AppConfig) -> list[OpenCamera]:
    opened: list[OpenCamera] = []
    try:
        for spec in specs:
            backend = cv2.CAP_V4L2 if _is_camera_source(spec.source) else cv2.CAP_ANY
            capture = cv2.VideoCapture(spec.source, backend)
            if not capture.isOpened():
                raise RuntimeError(f"Failed to open {spec.name} camera: {spec.source}")
            if _is_camera_source(spec.source):
                capture.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(*spec.input.fourcc[:4]),
                )
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, spec.input.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.input.height)
                capture.set(cv2.CAP_PROP_FPS, spec.input.fps)
                # With a single driver buffer the camera has nowhere to put the
                # next frame while this one is read, so it drops every other
                # frame (20 -> 10 FPS). The capture threads drain continuously,
                # so extra buffers do not add lag.
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 4)
                apply_camera_controls(spec.source, spec.input)
            opened.append(OpenCamera(spec, capture))
            actual_fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc = "".join(
                chr((actual_fourcc_value >> (8 * offset)) & 0xFF) for offset in range(4)
            )
            print(
                f"{spec.name}: source={spec.source} "
                f"{int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                f"{capture.get(cv2.CAP_PROP_FPS):.1f} FPS {actual_fourcc!r} "
                f"yaw={spec.extrinsics.yaw_deg:+.1f}deg "
                f"traffic_color={'on' if spec.traffic_color else 'off'} "
                f"exposure="
                + ("auto" if spec.input.exposure is None else f"{spec.input.exposure / 10:g}ms")
                + ("" if spec.input.gain is None else f" gain={spec.input.gain}")
            )
    except Exception:
        for camera in opened:
            camera.capture.release()
        raise
    return opened


def project_detections(
    detections,
    depth_map: np.ndarray,
    frame: np.ndarray,
    spec: CameraSpec,
    config: AppConfig,
    depth_source: str,
) -> list[ProjectedDetection]:
    height, width = frame.shape[:2]
    intrinsics = scale_intrinsics(spec.intrinsics, width, height)
    projected: list[ProjectedDetection] = []
    for detection in detections:
        depth_m, sample_source = sample_detection_depth(depth_map, detection, config.depth)
        if depth_m is None:
            projected.append(
                ProjectedDetection(
                    detection, None, None, None, None, sample_source, spec.name
                )
            )
            continue
        x1, _, x2, y2 = detection.bbox_xyxy
        x_m, z_m = pixel_depth_to_ego(
            (x1 + x2) * 0.5,
            y2,
            depth_m,
            intrinsics,
            spec.extrinsics,
        )
        projected.append(
            ProjectedDetection(
                detection=detection,
                depth_m=depth_m,
                x_m_lateral=x_m,
                z_m_forward=z_m,
                bev_xy=ego_to_bev_pixel(x_m, z_m, config.bev),
                depth_source=depth_source if sample_source == "depth_map" else sample_source,
                camera_name=spec.name,
            )
        )
    return projected


def should_update_depth(
    camera_index: int,
    frame_id: int,
    cadence: int,
    has_depth: bool,
) -> bool:
    """Spread camera depth updates across cadence phases after initialization."""

    if not has_depth:
        return True
    return frame_id % cadence == camera_index % cadence


def tracks_to_bev_objects(
    tracks: list[TrackOutput], config: AppConfig
) -> list[ProjectedDetection]:
    """Confirmed tracks as BEV markers at their filtered positions, labelled by ID."""
    objects = []
    for track in tracks:
        last = track.detection
        objects.append(
            ProjectedDetection(
                detection=Detection(
                    track.class_name,
                    last.detection.confidence if last else 0.0,
                    last.detection.bbox_xyxy if last else (0.0, 0.0, 0.0, 0.0),
                ),
                depth_m=math.hypot(track.x_m_lateral, track.z_m_forward),
                x_m_lateral=track.x_m_lateral,
                z_m_forward=track.z_m_forward,
                bev_xy=ego_to_bev_pixel(track.x_m_lateral, track.z_m_forward, config.bev),
                depth_source="track_coasting" if track.coasting else "track",
                camera_name=last.camera_name if last else None,
                track_id=track.track_id,
            )
        )
    return objects


def _ego_motion(
    sensor_monitor: LiveSensorMonitor | None, max_age_s: float = 0.5
) -> tuple[float | None, float | None]:
    """Ego speed (m/s) and yaw rate (rad/s) from live GPS and IMU, if both are fresh."""
    if sensor_monitor is None:
        return None, None
    snapshot = sensor_monitor.snapshot()
    gps, imu = snapshot.gps, snapshot.imu
    now_ns = time.monotonic_ns()
    fresh = all(
        stamp is not None and (now_ns - stamp) / 1e9 <= max_age_s
        for stamp in (gps.updated_monotonic_ns, imu.updated_monotonic_ns)
    )
    if not fresh or gps.speed_kph is None or imu.angular_rate_rps is None:
        return None, None
    return gps.speed_kph / 3.6, float(imu.angular_rate_rps[2])


def fuse_detections(
    detections: list[ProjectedDetection], merge_distance_m: float = 1.5
) -> list[ProjectedDetection]:
    """Suppress lower-confidence same-class observations at nearly the same BEV point."""
    kept: list[ProjectedDetection] = []
    ordered = sorted(detections, key=lambda item: item.detection.confidence, reverse=True)
    for candidate in ordered:
        if candidate.x_m_lateral is None or candidate.z_m_forward is None:
            kept.append(candidate)
            continue
        duplicate = False
        for existing in kept:
            if existing.detection.class_name != candidate.detection.class_name:
                continue
            if existing.x_m_lateral is None or existing.z_m_forward is None:
                continue
            distance = math.hypot(
                existing.x_m_lateral - candidate.x_m_lateral,
                existing.z_m_forward - candidate.z_m_forward,
            )
            if distance <= merge_distance_m:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _fit_tile(image: np.ndarray, width: int = 400, height: int = 300) -> np.ndarray:
    """Scale into a width x height tile without distortion, padding with black."""
    scale = min(width / image.shape[1], height / image.shape[0])
    w = max(1, int(round(image.shape[1] * scale)))
    h = max(1, int(round(image.shape[0] * scale)))
    tile = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = (height - h) // 2, (width - w) // 2
    tile[y : y + h, x : x + w] = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return tile


def _camera_tile(
    frame: np.ndarray, name: str, traffic_color: bool, width: int = 400, height: int = 300
) -> np.ndarray:
    tile = _fit_tile(frame, width, height)
    # One-line label on a translucent strip, so it covers little of the image.
    strip_h = 30 if height >= 400 else 24
    strip = tile[:strip_h]
    strip[:] = (strip * 0.35).astype(np.uint8)
    title = name.upper().replace("_", " ")
    scale = 0.65 if height >= 400 else 0.5
    cv2.putText(
        tile, title, (10, strip_h - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2,
        cv2.LINE_AA,
    )
    (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    mode = "objects + depth + traffic color" if traffic_color else "objects + depth"
    cv2.putText(
        tile, mode, (22 + title_w, strip_h - 8), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.72,
        (120, 240, 120), 1, cv2.LINE_AA,
    )
    cv2.rectangle(tile, (0, 0), (width - 1, height - 1), (70, 70, 70), 1)
    return tile


def _depth_tiles(
    depth_m: dict[str, np.ndarray], specs: list[CameraSpec], width: int = 400, height: int = 300
) -> list[np.ndarray]:
    """Depth tiles on one shared meter scale (bright = near) so cameras compare directly."""
    values = np.concatenate([depth_m[spec.name].ravel() for spec in specs])
    values = values[np.isfinite(values) & (values > 0.0)]
    max_m = float(np.clip(np.percentile(values, 95), 2.0, 80.0)) if values.size else 80.0
    tiles = []
    for spec in specs:
        depth = np.nan_to_num(depth_m[spec.name], nan=max_m)
        scaled = (255.0 * (1.0 - np.clip(depth, 0.0, max_m) / max_m)).astype(np.uint8)
        tile = _fit_tile(cv2.applyColorMap(scaled, cv2.COLORMAP_INFERNO), width, height)
        cv2.rectangle(tile, (0, 0), (width, 26), (0, 0, 0), -1)
        cv2.putText(
            tile,
            f"{spec.name.upper()} depth  median {np.median(depth):.1f} m  (0-{max_m:.0f} m)",
            (10, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    return tiles


DASHBOARD_COLUMNS = 3
DASHBOARD_MAX_HEIGHT = 900  # px; tiles shrink so the window fits a 1080p screen
DASHBOARD_MIN_BEV_HEIGHT = 600


def _grid(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    """Tiles row by row, padding the last row with black tiles."""
    blank = np.zeros_like(tiles[0])
    padded = tiles + [blank] * (-len(tiles) % columns)
    rows = [np.hstack(padded[i : i + columns]) for i in range(0, len(padded), columns)]
    return np.vstack(rows)


def compose_dashboard(
    annotated: dict[str, np.ndarray],
    specs: list[CameraSpec],
    bev: np.ndarray,
    depth_m: dict[str, np.ndarray] | None = None,
    with_bev: bool = True,
) -> np.ndarray:
    """Camera views on the left, BEV on the right (cameras only when with_bev is False).

    The three front cameras (left/middle/right) get a driver's-view layout:
    the narrow front camera large on top, left and right below it. Other
    setups, and --show-depth, use a 3-column grid (depth grid below).
    """
    names = {s.name for s in specs}
    if depth_m is None and names == {"left", "middle", "right"} and with_bev:
        return _front_dashboard(annotated, {s.name: s for s in specs}, bev)
    columns = min(DASHBOARD_COLUMNS, len(specs))
    camera_rows = math.ceil(len(specs) / columns)
    rows = camera_rows * (2 if depth_m is not None else 1)
    tile_h = min(300, DASHBOARD_MAX_HEIGHT // rows)
    tile_w = tile_h * 4 // 3
    blocks = [
        _grid(
            [
                _camera_tile(annotated[s.name], s.name, s.traffic_color, tile_w, tile_h)
                for s in specs
            ],
            columns,
        )
    ]
    if depth_m is not None:
        blocks.append(_grid(_depth_tiles(depth_m, specs, tile_w, tile_h), columns))
    grid = np.vstack(blocks)
    if not with_bev:
        return grid

    height = max(grid.shape[0], DASHBOARD_MIN_BEV_HEIGHT)
    if grid.shape[0] < height:
        grid = np.vstack((grid, np.zeros((height - grid.shape[0], grid.shape[1], 3), np.uint8)))
    return np.hstack((grid, _bev_panel(bev, height)))


def _front_dashboard(
    annotated: dict[str, np.ndarray], specs: dict[str, CameraSpec], bev: np.ndarray
) -> np.ndarray:
    """Narrow front 800x600 on top, left/right 400x225 below it: 1350 x 825 with the BEV.

    Tile shapes match the cameras (4:3 middle, 16:9 corners), so nothing is
    letterboxed.
    """
    def tile(name: str, width: int, height: int) -> np.ndarray:
        return _camera_tile(annotated[name], name, specs[name].traffic_color, width, height)

    views = np.vstack(
        (tile("middle", 800, 600), np.hstack((tile("left", 400, 225), tile("right", 400, 225))))
    )
    return np.hstack((views, _bev_panel(bev, views.shape[0])))


def _bev_panel(bev: np.ndarray, height: int) -> np.ndarray:
    width = max(1, int(round(bev.shape[1] * height / bev.shape[0])))
    interpolation = cv2.INTER_AREA if height < bev.shape[0] else cv2.INTER_LINEAR
    panel = cv2.resize(bev, (width, height), interpolation=interpolation)
    strip = panel[:34]
    strip[:] = (strip * 0.25).astype(np.uint8)
    cv2.putText(
        panel, "FUSED BEV", (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2,
        cv2.LINE_AA,
    )
    return panel


WINDOW_ORIGIN = (90, 60)  # px from the screen's top-left, clear of the dock


def _show(name: str, image: np.ndarray, created: set[str]) -> None:
    """imshow, creating the window once: no toolbar, resizable, placed on screen."""
    if name not in created:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(name, image.shape[1], image.shape[0])
        cv2.moveWindow(name, *WINDOW_ORIGIN)
        created.add(name)
    cv2.imshow(name, image)


def load_bev_view(config_path: str | Path, base: BevConfig) -> BevConfig:
    """The optional bev_view block: the area the BEV window shows.

    Detection, the semantic matrix and tracking keep the full bev area; this
    only changes what is drawn. Unset fields fall back to bev.
    """
    raw = yaml.safe_load(Path(config_path).read_text()) or {}
    view = raw.get("bev_view") or {}
    return replace(
        base,
        **{
            key: type(getattr(base, key))(view[key])
            for key in ("forward_range_m", "rear_range_m", "lateral_range_m", "pixels_per_meter")
            if key in view
        },
    )


def project_to_view(
    detections: list[ProjectedDetection], view: BevConfig
) -> list[ProjectedDetection]:
    return [
        replace(item, bev_xy=ego_to_bev_pixel(item.x_m_lateral, item.z_m_forward, view))
        if item.x_m_lateral is not None and item.z_m_forward is not None
        else item
        for item in detections
    ]


def render_multicam_frame(
    frames: dict[str, np.ndarray],
    projected_by_camera: dict[str, list[ProjectedDetection]],
    fused: list[ProjectedDetection],
    specs: list[CameraSpec],
    config: AppConfig,
    include_dashboard: bool,
    depth_maps: dict[str, np.ndarray] | None = None,
    view: BevConfig | None = None,
    with_bev: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    # Draw only the view area (default: the whole BEV). Positions are stored as
    # pixels of the full BEV, so recompute them for the view; objects outside
    # it get no pixel and are not drawn.
    view = view or config.bev
    shown = project_to_view(fused, view)
    bev = render_bev(view, shown)
    if not include_dashboard:
        return bev, None
    annotated = {
        spec.name: draw_annotated_frame(
            frames[spec.name], projected_by_camera[spec.name]
        )
        for spec in specs
    }
    depth_m = None
    if depth_maps is not None:
        depth_m = {
            name: depth * config.depth.metric_scale + config.depth.metric_bias
            for name, depth in depth_maps.items()
        }
    return bev, compose_dashboard(annotated, specs, bev, depth_m, with_bev)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    specs = load_camera_specs(args.config, config)
    bev_view = load_bev_view(args.config, config.bev)
    if args.skip_missing_cameras:
        missing = [s for s in specs if isinstance(s.source, str) and s.source.startswith("/dev/")
                   and not Path(s.source).exists()]
        for spec in missing:
            print(f"warning: skipping {spec.name}: {spec.source} is not connected")
        specs = [s for s in specs if s not in missing]
    if not specs:
        raise ValueError("No cameras to open (none listed, or none connected)")

    # Persistent object IDs (--track-objects): ByteTrack per camera + BEV tracker.
    object_tracker = BEVObjectTracker() if args.track_objects else None
    camera_trackers = (
        {spec.name: CameraTrackIds() for spec in specs} if args.track_objects else {}
    )

    temporal_tracker: TemporalBEVTracker | None = None
    if args.temporal_tracking:
        temporal_tracker = TemporalBEVTracker(
            config.bev,
            tracker_dir=args.tracker_dir,
            update_hz=args.tracker_hz,
        )
        print(
            f"Temporal tracking enabled: {args.tracker_hz:g} Hz, "
            f"source={temporal_tracker.tracker_dir}"
        )

    if args.da3_engine:
        # DA3 output is already in meters once the backend applies each camera's
        # focal length, so the DA2 metric_scale correction must not be applied.
        config = replace(
            config,
            depth=replace(
                config.depth,
                backend="tensorrt_da3",
                path=args.da3_engine,
                metric_scale=1.0,
                metric_bias=0.0,
            ),
        )

    batch_detector = args.batch_detector_model is not None
    if batch_detector:
        config = replace(
            config,
            detector=replace(config.detector, path=args.batch_detector_model),
        )
    print(
        "Loading vehicle detector "
        f"({'batch' if batch_detector else 'sequential'} mode)..."
    )
    detector: DetectorBackend = create_detector(config.detector)
    traffic_detector: DetectorBackend | None = None
    if any(spec.traffic_color for spec in specs):
        print("Loading center-only traffic-light color detector...")
        traffic_detector = create_detector(config.traffic_detector)  # type: ignore[arg-type]
    print("Loading shared metric-depth model...")
    depth_backend = create_depth_backend(config.depth)

    cameras = open_cameras(specs, config)
    capture = LatestFrameCapture(cameras)
    sensor_monitor = (
        LiveSensorMonitor(
            gps_port=args.gps_port,
            imu_port=args.imu_port,
            gps_baud=args.gps_baud,
            imu_baud=args.imu_baud,
        )
        if args.show_sensors or args.save_sensors
        else None
    )
    show_sensor_overlay = args.show_sensors
    processed = 0
    started = time.perf_counter()
    try:
        capture.start()
        if sensor_monitor is not None:
            sensor_monitor.start()
            if args.show_sensors:
                print("Live GPS + IMU overlay enabled; press i to hide/show it")
            if args.save_sensors:
                print("Frame-aligned GPS + IMU matrix logging enabled")
        dirs = ensure_output_dirs(args.save_dir)
        dashboard_dir = dirs["root"] / "dashboard"
        dashboard_dir.mkdir(parents=True, exist_ok=True)
        matrix_dir = dirs["root"] / "matrix"
        save_raw_matrices = config.runtime.save_matrices or args.save_sensors
        if save_raw_matrices:
            matrix_dir.mkdir(parents=True, exist_ok=True)
            write_semantic_metadata(matrix_dir / "metadata.json", config.bev)
        tracked_matrix_dir = dirs["root"] / "matrix_tracked"
        if temporal_tracker is not None:
            tracked_matrix_dir.mkdir(parents=True, exist_ok=True)
            write_semantic_metadata(tracked_matrix_dir / "metadata.json", config.bev)
        latest_depth: dict[str, np.ndarray] = {}
        profiler = Profiler()
        max_frames = (
            args.max_frames
            if args.max_frames is not None
            else config.runtime.max_frames
        )
        jsonl_path = dirs["root"] / "detections.jsonl"
        sensor_jsonl_path = dirs["root"] / "sensors" / "matrix_sensors.jsonl"
        if args.save_sensors:
            sensor_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        save_images = config.runtime.save_images and not args.no_save_images
        include_dashboard = args.display or save_images
        frame_sequences: dict[str, int] = {}

        with ExitStack() as output_stack:
            jsonl = output_stack.enter_context(
                jsonl_path.open("w", encoding="utf-8")
            )
            sensor_jsonl = (
                output_stack.enter_context(
                    sensor_jsonl_path.open("w", encoding="utf-8")
                )
                if args.save_sensors
                else None
            )
            next_frame_at = 0.0
            shown_windows: set[str] = set()
            while not max_frames or processed < max_frames:
                with profiler.stage("capture"):
                    frames, frame_sequences, camera_capture_ns = capture.read_latest(
                        frame_sequences
                    )

                all_projected: list[ProjectedDetection] = []
                projected_by_camera: dict[str, list[ProjectedDetection]] = {}
                ordered_frames = [frames[spec.name] for spec in specs]
                if batch_detector:
                    with profiler.stage("detect_batch"):
                        batched_detections = detector.infer_batch(ordered_frames)
                else:
                    batched_detections = []
                    for spec, frame in zip(specs, ordered_frames):
                        with profiler.stage(f"detect_{spec.name}"):
                            batched_detections.append(detector.infer(frame))
                if len(batched_detections) != len(specs):
                    raise RuntimeError(
                        f"Detector returned {len(batched_detections)} batch items "
                        f"for {len(specs)} cameras"
                    )

                for camera_index, (spec, detections) in enumerate(
                    zip(specs, batched_detections)
                ):
                    frame = frames[spec.name]
                    if spec.traffic_color and traffic_detector is not None:
                        with profiler.stage("traffic_color_middle"):
                            detections.extend(traffic_detector.infer(frame))

                    run_depth = should_update_depth(
                        camera_index,
                        processed,
                        config.depth.cadence,
                        spec.name in latest_depth,
                    )
                    if run_depth:
                        with profiler.stage(f"depth_{spec.name}"):
                            if getattr(depth_backend, "uses_intrinsics", False):
                                latest_depth[spec.name] = depth_backend.infer(
                                    frame, intrinsics=spec.intrinsics
                                )
                            else:
                                latest_depth[spec.name] = depth_backend.infer(frame)
                        depth_source = "depth_map"
                    else:
                        depth_source = "reused_depth_map"

                    camera_projected = project_detections(
                        detections,
                        latest_depth[spec.name],
                        frame,
                        spec,
                        config,
                        depth_source,
                    )
                    if object_tracker is not None:
                        # project_detections returns one item per detection, in order.
                        with profiler.stage(f"bytetrack_{spec.name}"):
                            local_ids = camera_trackers[spec.name].assign(detections)
                        camera_projected = [
                            replace(item, camera_track_id=local_id)
                            for item, local_id in zip(camera_projected, local_ids)
                        ]
                    all_projected.extend(camera_projected)
                    projected_by_camera[spec.name] = camera_projected

                fused = fuse_detections(all_projected)
                bev_objects = fused
                tracks: list[TrackOutput] = []
                if object_tracker is not None:
                    with profiler.stage("object_tracking"):
                        ego_speed, ego_yaw_rate = _ego_motion(sensor_monitor)
                        tracks, track_ids = object_tracker.step(
                            fused,
                            np.mean(list(camera_capture_ns.values())) / 1e9,
                            ego_speed_mps=ego_speed,
                            ego_yaw_rate_rps=ego_yaw_rate,
                        )
                        global_id = {id(item): tid for item, tid in zip(fused, track_ids)}
                        fused = [
                            replace(item, track_id=tid) for item, tid in zip(fused, track_ids)
                        ]
                        projected_by_camera = {
                            name: [replace(p, track_id=global_id.get(id(p))) for p in items]
                            for name, items in projected_by_camera.items()
                        }
                        bev_objects = tracks_to_bev_objects(tracks, config)
                matrix_name = f"{processed:06d}.npy"
                semantic_matrix: np.ndarray | None = None
                if save_raw_matrices or temporal_tracker is not None:
                    with profiler.stage("matrix"):
                        semantic_matrix = render_semantic_grid(config.bev, fused)
                matrix_wall_time_ns = time.time_ns()
                matrix_monotonic_ns = time.monotonic_ns()
                sensor_snapshot = (
                    sensor_monitor.snapshot() if sensor_monitor is not None else None
                )
                if save_raw_matrices:
                    if semantic_matrix is None:
                        raise RuntimeError("Semantic matrix generation failed")
                    with profiler.stage("save_matrix"):
                        np.save(matrix_dir / matrix_name, semantic_matrix)

                tracker_result = None
                if temporal_tracker is not None:
                    if semantic_matrix is None:
                        raise RuntimeError("Temporal tracking requires a semantic matrix")
                    with profiler.stage("temporal_tracking"):
                        tracker_result = temporal_tracker.maybe_update(semantic_matrix)
                    if tracker_result is not None:
                        with profiler.stage("save_tracked_matrix"):
                            np.save(
                                tracked_matrix_dir / matrix_name,
                                tracker_result.cleaned_matrix,
                            )

                if sensor_jsonl is not None:
                    if sensor_snapshot is None:
                        raise RuntimeError("Sensor logging requires a sensor snapshot")
                    with profiler.stage("save_matrix_sensors"):
                        write_matrix_sensor_record(
                            sensor_jsonl,
                            frame_id=processed,
                            raw_matrix_file=f"matrix/{matrix_name}",
                            tracked_matrix_file=(
                                f"matrix_tracked/{matrix_name}"
                                if tracker_result is not None
                                else None
                            ),
                            wall_time_ns=matrix_wall_time_ns,
                            monotonic_ns=matrix_monotonic_ns,
                            camera_capture_monotonic_ns=camera_capture_ns,
                            snapshot=sensor_snapshot,
                        )

                latest_tracking = (
                    temporal_tracker.last_result
                    if temporal_tracker is not None
                    else None
                )
                with profiler.stage("render"):
                    bev, dashboard = render_multicam_frame(
                        frames,
                        projected_by_camera,
                        bev_objects,
                        specs,
                        config,
                        include_dashboard,
                        depth_maps=latest_depth if args.show_depth else None,
                        view=bev_view,
                        with_bev=not args.display_cameras,
                    )

                tracker_view = None
                if temporal_tracker is not None and latest_tracking is not None:
                    with profiler.stage("tracker_view"):
                        tracker_view = temporal_tracker.render_viewer()

                write_jsonl_record(
                    jsonl,
                    processed,
                    time.perf_counter() - started,
                    fused,
                    wall_time_ns=matrix_wall_time_ns,
                    monotonic_ns=matrix_monotonic_ns,
                    tracks=[
                        {
                            "track_id": t.track_id,
                            "class_name": t.class_name,
                            "x_m_lateral": round(t.x_m_lateral, 3),
                            "z_m_forward": round(t.z_m_forward, 3),
                            "vx_mps": round(t.vx_mps, 3),
                            "vz_mps": round(t.vz_mps, 3),
                            "coasting": t.coasting,
                            "hits": t.hits,
                        }
                        for t in tracks
                    ]
                    if object_tracker is not None
                    else None,
                )

                if save_images:
                    if dashboard is None:
                        raise RuntimeError("Dashboard rendering is required when saving images")
                    stem = f"{processed:06d}.jpg"
                    save_bgr_image(dirs["bev"] / stem, bev)
                    save_bgr_image(dashboard_dir / stem, dashboard)

                if args.display:
                    if dashboard is None:
                        raise RuntimeError("Dashboard rendering is required for --display")
                    display_frame = dashboard
                    if sensor_monitor is not None and show_sensor_overlay:
                        display_frame = draw_live_sensor_panel(
                            dashboard.copy(), sensor_monitor.snapshot()
                        )
                    _show(
                        "Camera detection" if args.display_cameras else "Camera detection + depth + fused BEV",
                        display_frame,
                        shown_windows,
                    )
                elif args.display_bev:
                    if args.temporal_tracking:
                        if tracker_view is None:
                            raise RuntimeError("Temporal tracker view is unavailable")
                        display_frame = tracker_view
                        window_name = "Live Causal BEV"
                    else:
                        display_frame = bev
                        window_name = "Three-camera fused BEV"
                    if sensor_monitor is not None and show_sensor_overlay:
                        display_frame = draw_live_sensor_panel(
                            display_frame.copy(), sensor_monitor.snapshot()
                        )
                    _show(window_name, display_frame, shown_windows)
                if args.display or args.display_bev:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("r") and temporal_tracker is not None:
                        temporal_tracker.reset()
                        print("temporal tracker reset")
                    if key == ord("i") and sensor_monitor is not None:
                        show_sensor_overlay = not show_sensor_overlay
                        print(
                            "live GPS + IMU overlay "
                            + ("shown" if show_sensor_overlay else "hidden")
                        )

                processed += 1
                if args.max_fps is not None:
                    # Pace from the previous frame, never from the start time, so a
                    # slow frame (e.g. the first one) is not followed by an uncapped
                    # catch-up burst.
                    now = time.perf_counter()
                    next_frame_at = max(next_frame_at + 1.0 / args.max_fps, now)
                    time.sleep(next_frame_at - now)
                if processed == 1 or processed % 20 == 0:
                    elapsed = time.perf_counter() - started
                    status = (
                        f"frame={processed} fused_objects={len(fused)} "
                        f"processing_fps={processed / elapsed:.1f}"
                    )
                    if object_tracker is not None:
                        status += f" tracks={len(tracks)}"
                    if latest_tracking is not None:
                        status += (
                            f" tracked_objects={len(latest_tracking.objects)}"
                            f" target_speed={latest_tracking.cleaned_target_speed_mph:.1f}mph"
                            f" tracker_ms={latest_tracking.latency_ms:.2f}"
                        )
                    print(status)
    except KeyboardInterrupt:
        # Stopped from outside (e.g. scripts/record_all_cameras_perf.sh); still report.
        print("Interrupted; stopping")
    finally:
        capture.stop()
        if sensor_monitor is not None:
            sensor_monitor.stop()
        if args.display or args.display_bev:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print(f"Finished: frames={processed} elapsed={elapsed:.2f}s fps={processed / elapsed:.1f}")
    if args.profile:
        print(profiler.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
