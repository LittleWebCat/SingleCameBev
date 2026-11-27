#!/usr/bin/env python3
"""Run a standalone Ultralytics YOLO test on the USB camera.

Example:
    .venv/bin/python scripts/test_camera_yolo.py --display

Press ``q`` or ``Esc`` to stop the display. The default model is the new
traffic-light detector in ``untested_model/best.engine``; pass ``--model`` to
test a different Ultralytics ``.pt``, ``.onnx``, or ``.engine`` model.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bev_depth.detector import COCO_CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test an Ultralytics YOLO model on a live camera stream."
    )
    parser.add_argument(
        "--model",
        default="untested_model/best.engine",
        help="Ultralytics model path (.pt, .onnx, or .engine)",
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
        help="Clockwise frame rotation before detection (default: 0)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--device",
        default=None,
        help="Ultralytics device, such as 0 or cpu (default: auto)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Stop after this many frames; 0 runs until interrupted",
    )
    parser.add_argument(
        "--output",
        default="runs/yolo_camera_test.mp4",
        help="Annotated MP4 output path",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated video")
    parser.add_argument("--display", action="store_true", help="Show the annotated live view")
    parser.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="Print a status line every N frames",
    )
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("--width, --height, and --fps must be positive")
    if args.imgsz <= 0 or args.max_frames < 0 or args.log_every <= 0:
        parser.error("--imgsz and --log-every must be positive; --max-frames cannot be negative")
    if not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0:
        parser.error("--conf and --iou must be between 0 and 1")
    return args


def source_value(raw_source: str) -> int | str:
    return int(raw_source) if raw_source.isdigit() else raw_source


def rotate_frame(frame, degrees: int):
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


def resolved_class_name(names: dict[int, str], class_id: int) -> str:
    name = str(names.get(class_id, class_id))
    if name == f"class{class_id}" and class_id < len(COCO_CLASS_NAMES):
        return COCO_CLASS_NAMES[class_id]
    return name


def model_classes_summary(names: dict[int, str], limit: int = 12) -> str:
    class_names = [resolved_class_name(names, class_id) for class_id in sorted(names)]
    if len(class_names) <= limit:
        return ", ".join(class_names)
    shown = ", ".join(class_names[:limit])
    return f"{len(class_names)} reported ({shown}, ...); model metadata may be incomplete"


def detection_summary(result) -> str:
    counts: Counter[str] = Counter()
    names = result.names
    for class_id in result.boxes.cls.tolist():
        numeric_id = int(class_id)
        counts[resolved_class_name(names, numeric_id)] += 1
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items())) or "none"


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    is_alias = model_path.parent == Path(".") and model_path.suffix == ".pt"
    if not is_alias and not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Run this script with the repo virtual environment."
        ) from exc

    print(f"Loading model: {args.model}")
    model = YOLO(args.model, task="detect")

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
    total_detections = 0
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
            inference_started = time.perf_counter()
            predict_options = {
                "source": frame,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "iou": args.iou,
                "verbose": False,
            }
            if args.device is not None:
                predict_options["device"] = args.device
            result = model.predict(**predict_options)[0]
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            if processed == 0:
                print(f"Model classes: {model_classes_summary(result.names)}")

            for class_id in {int(value) for value in result.boxes.cls.tolist()}:
                result.names[class_id] = resolved_class_name(result.names, class_id)
            annotated = result.plot()
            processed += 1
            frame_detections = len(result.boxes)
            total_detections += frame_detections
            elapsed = time.perf_counter() - started
            processing_fps = processed / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                annotated,
                f"YOLO {inference_ms:.1f} ms | {processing_fps:.1f} FPS | det {frame_detections}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if not args.no_save:
                if writer is None:
                    output_path = Path(args.output)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_height, output_width = annotated.shape[:2]
                    output_fps = min(actual_fps, 30.0) if actual_fps > 0 else 30.0
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        output_fps,
                        (output_width, output_height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(f"Failed to create output video: {output_path}")
                    print(f"Saving annotated video: {output_path} at {output_fps:.2f} FPS")
                writer.write(annotated)

            if args.display:
                cv2.imshow("YOLO camera test", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            if processed == 1 or processed % args.log_every == 0:
                print(
                    f"frame={processed} inference={inference_ms:.1f}ms "
                    f"processing_fps={processing_fps:.1f} detections=[{detection_summary(result)}]"
                )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    average_fps = processed / elapsed if elapsed > 0 else 0.0
    print(
        f"Finished: frames={processed}, detections={total_detections}, "
        f"elapsed={elapsed:.2f}s, average={average_fps:.2f} FPS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
