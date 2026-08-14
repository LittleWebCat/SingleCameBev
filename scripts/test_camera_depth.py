#!/usr/bin/env python3
"""Run a standalone live-depth preview on the USB camera.

Example:
    .venv/bin/python scripts/test_camera_depth.py --display --no-save --max-frames 0

Press ``q`` or ``Esc`` to close the live preview. The saved video places the
camera image on the left and a colorized metric-depth estimate on the right.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bev_depth.depth import TensorRTDepthBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show live metric-depth estimates from a camera."
    )
    parser.add_argument(
        "--model",
        default="models/depth_anything_v2_small_fp16.engine",
        help="Depth Anything TensorRT engine path",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Camera index or video path (default: 0, which maps to /dev/video0)",
    )
    parser.add_argument("--width", type=int, default=1920, help="Camera capture width")
    parser.add_argument("--height", type=int, default=1200, help="Camera capture height")
    parser.add_argument("--fps", type=float, default=90.0, help="Requested camera FPS")
    parser.add_argument("--fourcc", default="MJPG", help="Camera FOURCC")
    parser.add_argument(
        "--rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="Clockwise frame rotation before depth inference",
    )
    parser.add_argument("--input-size", type=int, default=518, help="Depth model input size")
    parser.add_argument(
        "--preview-width",
        type=int,
        default=960,
        help="Width of each half of the RGB/depth preview",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Stop after this many frames; 0 runs until interrupted",
    )
    parser.add_argument(
        "--output",
        default="runs/depth_camera_test.mp4",
        help="Side-by-side annotated MP4 output path",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save preview video")
    parser.add_argument("--display", action="store_true", help="Show the live depth preview")
    parser.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="Print a status line every N frames",
    )
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    if args.input_size <= 0 or args.preview_width <= 0 or args.log_every <= 0:
        parser.error("--input-size, --preview-width, and --log-every must be positive")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    return args


def source_value(raw_source: str) -> int | str:
    return int(raw_source) if raw_source.isdigit() else raw_source


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    rotations = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return cv2.rotate(frame, rotations[degrees]) if degrees else frame


def open_source(args: argparse.Namespace) -> cv2.VideoCapture:
    source = source_value(args.source)
    is_camera = isinstance(source, int) or str(source).startswith("/dev/video")
    backend = cv2.CAP_V4L2 if is_camera else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source: {args.source}")

    if is_camera:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc[:4]))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
    return capture


def valid_depth_values(depth_map: np.ndarray) -> np.ndarray:
    return depth_map[np.isfinite(depth_map) & (depth_map > 0.0) & (depth_map < 1.0e4)]


def colorize_depth(depth_map: np.ndarray) -> tuple[np.ndarray, float, float]:
    valid = valid_depth_values(depth_map)
    if valid.size == 0:
        return np.zeros((*depth_map.shape, 3), dtype=np.uint8), float("nan"), float("nan")

    near_m, far_m = (float(value) for value in np.percentile(valid, [2.0, 98.0]))
    if far_m <= near_m:
        far_m = near_m + 1.0
    scaled = np.clip((depth_map - near_m) / (far_m - near_m), 0.0, 1.0)
    scaled_u8 = np.nan_to_num(scaled * 255.0, nan=0.0).astype(np.uint8)
    colored = cv2.applyColorMap(scaled_u8, cv2.COLORMAP_TURBO)
    colored[~np.isfinite(depth_map) | (depth_map <= 0.0)] = 0
    return colored, near_m, far_m


def center_depth_m(depth_map: np.ndarray) -> float:
    height, width = depth_map.shape[:2]
    radius = max(2, min(height, width) // 100)
    center_y, center_x = height // 2, width // 2
    patch = depth_map[
        max(0, center_y - radius) : min(height, center_y + radius + 1),
        max(0, center_x - radius) : min(width, center_x + radius + 1),
    ]
    valid = valid_depth_values(patch)
    return float(np.median(valid)) if valid.size else float("nan")


def add_text_banner(
    frame: np.ndarray,
    title: str,
    details: str,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (0, 0, 0), -1)
    cv2.putText(
        frame,
        title,
        (16, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        details,
        (16, 59),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def build_preview(
    frame: np.ndarray,
    depth_color: np.ndarray,
    near_m: float,
    far_m: float,
    center_m: float,
    depth_ms: float,
    processing_fps: float,
    preview_width: int,
) -> np.ndarray:
    preview_height = max(1, round(frame.shape[0] * preview_width / frame.shape[1]))
    size = (preview_width, preview_height)
    camera_preview = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    depth_preview = cv2.resize(depth_color, size, interpolation=cv2.INTER_AREA)

    add_text_banner(
        camera_preview,
        "CAMERA",
        f"depth inference {depth_ms:.1f} ms | processing {processing_fps:.1f} FPS",
    )
    center_text = f"center {center_m:.2f} m" if np.isfinite(center_m) else "center unavailable"
    range_text = (
        f"near {near_m:.2f} m | far {far_m:.2f} m"
        if np.isfinite(near_m) and np.isfinite(far_m)
        else "no valid depth"
    )
    add_text_banner(depth_preview, "DEPTH  (blue=near, red=far)", f"{range_text} | {center_text}")

    center = (preview_width // 2, preview_height // 2)
    cv2.drawMarker(
        depth_preview,
        center,
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=22,
        thickness=2,
    )
    return np.ascontiguousarray(np.hstack((camera_preview, depth_preview)))


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Depth model not found: {model_path}")

    print(f"Loading depth model: {args.model}")
    depth_backend = TensorRTDepthBackend(str(model_path), args.input_size)
    capture = open_source(args)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    actual_fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    print(
        f"Capture: {args.source} {actual_width}x{actual_height} "
        f"{actual_fps:.2f} FPS {actual_fourcc!r}; rotation={args.rotate}"
    )

    writer = None
    processed = 0
    depth_times_ms: list[float] = []
    started = time.perf_counter()
    try:
        while args.max_frames == 0 or processed < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                if processed == 0:
                    raise RuntimeError(f"Opened source but failed to read a frame: {args.source}")
                print("Camera stream ended.")
                break

            frame = rotate_frame(frame, args.rotate)
            depth_started = time.perf_counter()
            depth_map = depth_backend.infer(frame)
            depth_ms = (time.perf_counter() - depth_started) * 1000.0
            depth_times_ms.append(depth_ms)

            depth_color, near_m, far_m = colorize_depth(depth_map)
            center_m = center_depth_m(depth_map)
            processed += 1
            elapsed = time.perf_counter() - started
            processing_fps = processed / elapsed if elapsed > 0.0 else 0.0
            preview = build_preview(
                frame,
                depth_color,
                near_m,
                far_m,
                center_m,
                depth_ms,
                processing_fps,
                args.preview_width,
            )

            if not args.no_save:
                if writer is None:
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_height, output_width = preview.shape[:2]
                    output_fps = min(actual_fps, 30.0) if actual_fps > 0.0 else 30.0
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (output_width, output_height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to create output video: {output_path}")
                    print(f"Saving depth preview: {output_path} at {output_fps:.2f} FPS")
                writer.write(preview)

            if args.display:
                cv2.imshow("Camera depth test", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if processed == 1 or processed % args.log_every == 0:
                print(
                    f"frame={processed} depth={depth_ms:.1f}ms "
                    f"processing_fps={processing_fps:.1f} range={near_m:.2f}-{far_m:.2f}m "
                    f"center={center_m:.2f}m"
                )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    average_fps = processed / elapsed if elapsed > 0.0 else 0.0
    average_depth_ms = float(np.mean(depth_times_ms)) if depth_times_ms else float("nan")
    print(
        f"Finished: frames={processed}, elapsed={elapsed:.2f}s, "
        f"average={average_fps:.2f} FPS, depth_average={average_depth_ms:.2f}ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
