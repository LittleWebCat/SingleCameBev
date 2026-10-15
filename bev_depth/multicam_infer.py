from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import AppConfig, Extrinsics, load_config
from .depth import create_depth_backend, sample_detection_depth
from .detector import DetectorBackend, create_detector
from .geometry import ego_to_bev_pixel, pixel_depth_to_ego, scale_intrinsics
from .io import ensure_output_dirs, save_bgr_image, write_jsonl_record
from .profiler import Profiler
from .render import draw_annotated_frame, render_bev
from .semantic_grid import render_semantic_grid, write_semantic_metadata
from .types import ProjectedDetection


@dataclass(frozen=True)
class CameraSpec:
    name: str
    source: int | str
    traffic_color: bool
    extrinsics: Extrinsics


@dataclass
class OpenCamera:
    spec: CameraSpec
    capture: cv2.VideoCapture


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse object detection and monocular depth from multiple cameras into one BEV"
    )
    parser.add_argument("--config", default="configs/three_camera_bev.yaml")
    parser.add_argument("--save-dir", default="runs/three_camera_bev")
    parser.add_argument("--display", action="store_true", help="Show camera row and fused BEV")
    parser.add_argument("--display-bev", action="store_true", help="Show only the fused BEV")
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--batch-detector-model",
        default=None,
        help="Fixed-batch detector engine for all configured cameras",
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
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
            )
        )
    if any(spec.traffic_color for spec in specs) and base.traffic_detector is None:
        raise ValueError("A traffic_detector model is required when traffic_color is enabled")
    return specs


def _is_camera_source(source: int | str) -> bool:
    return isinstance(source, int) or str(source).startswith("/dev/video")


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
                    cv2.VideoWriter_fourcc(*config.input.fourcc[:4]),
                )
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.input.width)
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.input.height)
                capture.set(cv2.CAP_PROP_FPS, config.input.fps)
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
                f"traffic_color={'on' if spec.traffic_color else 'off'}"
            )
    except Exception:
        for camera in opened:
            camera.capture.release()
        raise
    return opened


def capture_synchronized(cameras: list[OpenCamera]) -> dict[str, np.ndarray]:
    for camera in cameras:
        if not camera.capture.grab():
            raise RuntimeError(f"Failed to grab a frame from {camera.spec.name}")
    frames: dict[str, np.ndarray] = {}
    for camera in cameras:
        ok, frame = camera.capture.retrieve()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to retrieve a frame from {camera.spec.name}")
        frames[camera.spec.name] = frame
    return frames


def project_detections(
    detections,
    depth_map: np.ndarray,
    frame: np.ndarray,
    spec: CameraSpec,
    config: AppConfig,
    depth_source: str,
) -> list[ProjectedDetection]:
    height, width = frame.shape[:2]
    intrinsics = scale_intrinsics(config.intrinsics, width, height)
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


def draw_camera_poses(
    bev_image: np.ndarray, specs: list[CameraSpec], config: AppConfig
) -> np.ndarray:
    result = bev_image.copy()
    colors = {
        "left": (255, 100, 40),
        "middle": (60, 220, 60),
        "right": (180, 80, 255),
    }
    scale = config.bev.pixels_per_meter
    for spec in specs:
        origin = ego_to_bev_pixel(
            spec.extrinsics.lateral_offset_m,
            max(0.0, spec.extrinsics.forward_offset_m),
            config.bev,
        )
        if origin is None:
            continue
        yaw = math.radians(spec.extrinsics.yaw_deg)
        endpoint = (
            int(round(origin[0] + math.sin(yaw) * 3.0 * scale)),
            int(round(origin[1] - math.cos(yaw) * 3.0 * scale)),
        )
        color = colors.get(spec.name, (255, 255, 0))
        cv2.arrowedLine(result, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=0.25)
        label_at = (max(2, origin[0] - 28), max(18, origin[1] - 6))
        cv2.putText(
            result,
            spec.name.upper(),
            label_at,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return result


def _camera_tile(frame: np.ndarray, name: str, traffic_color: bool) -> np.ndarray:
    tile = cv2.resize(frame, (400, 300), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (400, 48), (0, 0, 0), -1)
    mode = "objects + depth + traffic color" if traffic_color else "objects + depth"
    cv2.putText(
        tile,
        name.upper(),
        (12, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        tile,
        mode,
        (12, 41),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (120, 240, 120),
        1,
        cv2.LINE_AA,
    )
    return tile


def compose_dashboard(
    annotated: dict[str, np.ndarray], specs: list[CameraSpec], bev: np.ndarray
) -> np.ndarray:
    top = np.hstack(
        [_camera_tile(annotated[spec.name], spec.name, spec.traffic_color) for spec in specs]
    )
    bottom = np.zeros((720, top.shape[1], 3), dtype=np.uint8)
    scale = min(bottom.shape[1] / bev.shape[1], bottom.shape[0] / bev.shape[0])
    bev_width = max(1, int(round(bev.shape[1] * scale)))
    bev_height = max(1, int(round(bev.shape[0] * scale)))
    bev_scaled = cv2.resize(bev, (bev_width, bev_height), interpolation=cv2.INTER_AREA)
    offset_x = (bottom.shape[1] - bev_width) // 2
    offset_y = (bottom.shape[0] - bev_height) // 2
    bottom[offset_y : offset_y + bev_height, offset_x : offset_x + bev_width] = bev_scaled
    cv2.putText(
        bottom,
        "FUSED BEV",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack((top, bottom))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    specs = load_camera_specs(args.config, config)
    if len(specs) != 3:
        raise ValueError(f"This dashboard requires exactly 3 cameras; configured {len(specs)}")

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
    dirs = ensure_output_dirs(args.save_dir)
    dashboard_dir = dirs["root"] / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    matrix_dir = dirs["root"] / "matrix"
    if config.runtime.save_matrices:
        matrix_dir.mkdir(parents=True, exist_ok=True)
        write_semantic_metadata(matrix_dir / "metadata.json", config.bev)
    latest_depth: dict[str, np.ndarray] = {}
    profiler = Profiler()
    started = time.perf_counter()
    processed = 0
    max_frames = args.max_frames if args.max_frames is not None else config.runtime.max_frames
    jsonl_path = dirs["root"] / "detections.jsonl"

    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl:
            while not max_frames or processed < max_frames:
                with profiler.stage("capture"):
                    frames = capture_synchronized(cameras)

                all_projected: list[ProjectedDetection] = []
                annotated: dict[str, np.ndarray] = {}
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

                for spec, detections in zip(specs, batched_detections):
                    frame = frames[spec.name]
                    if spec.traffic_color and traffic_detector is not None:
                        with profiler.stage("traffic_color_middle"):
                            detections.extend(traffic_detector.infer(frame))

                    run_depth = (
                        spec.name not in latest_depth or processed % config.depth.cadence == 0
                    )
                    if run_depth:
                        with profiler.stage(f"depth_{spec.name}"):
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
                    all_projected.extend(camera_projected)
                    annotated[spec.name] = draw_annotated_frame(frame, camera_projected)

                fused = fuse_detections(all_projected)
                if config.runtime.save_matrices:
                    with profiler.stage("matrix"):
                        semantic_matrix = render_semantic_grid(config.bev, fused)
                        np.save(matrix_dir / f"{processed:06d}.npy", semantic_matrix)
                with profiler.stage("render"):
                    bev = draw_camera_poses(render_bev(config.bev, fused), specs, config)
                    dashboard = compose_dashboard(annotated, specs, bev)

                frame_wall_time_ns = time.time_ns()
                write_jsonl_record(
                    jsonl,
                    processed,
                    time.perf_counter() - started,
                    fused,
                    wall_time_ns=frame_wall_time_ns,
                    monotonic_ns=time.monotonic_ns(),
                )

                save_images = config.runtime.save_images and not args.no_save_images
                if save_images:
                    stem = f"{processed:06d}.jpg"
                    save_bgr_image(dirs["bev"] / stem, bev)
                    save_bgr_image(dashboard_dir / stem, dashboard)

                if args.display:
                    cv2.imshow("Three-camera detection + depth + fused BEV", dashboard)
                elif args.display_bev:
                    cv2.imshow("Three-camera fused BEV", bev)
                if args.display or args.display_bev:
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

                processed += 1
                if processed == 1 or processed % 20 == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"frame={processed} fused_objects={len(fused)} "
                        f"processing_fps={processed / elapsed:.1f}"
                    )
    finally:
        for camera in cameras:
            camera.capture.release()
        if args.display or args.display_bev:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print(f"Finished: frames={processed} elapsed={elapsed:.2f}s fps={processed / elapsed:.1f}")
    if args.profile:
        print(profiler.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
