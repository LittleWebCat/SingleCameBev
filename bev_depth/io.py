from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from .config import InputConfig
from .types import ProjectedDetection


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_frames(
    source: str, input_config: InputConfig | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    path = Path(source)
    if source.isdigit():
        yield from _iter_capture(int(source), input_config)
    elif path.is_dir():
        frame_id = 0
        for image_path in sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            yield frame_id, float(frame_id), frame
            frame_id += 1
    elif path.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {source}")
        yield 0, 0.0, frame
    else:
        yield from _iter_capture(str(path), input_config)


def _iter_capture(
    source: int | str, input_config: InputConfig | None = None
) -> Iterator[tuple[int, float, np.ndarray]]:
    is_camera = isinstance(source, int)
    api_preference = cv2.CAP_ANY
    if (
        is_camera
        and input_config is not None
        and input_config.camera_index_backend.lower() == "v4l2"
    ):
        api_preference = cv2.CAP_V4L2
    cap = cv2.VideoCapture(source, api_preference)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source}")
    if is_camera and input_config is not None:
        if input_config.fourcc:
            fourcc = cv2.VideoWriter_fourcc(*input_config.fourcc[:4])
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        if input_config.width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, input_config.width)
        if input_config.height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, input_config.height)
        if input_config.fps > 0:
            cap.set(cv2.CAP_PROP_FPS, input_config.fps)
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_id / fps if fps > 0.0 else frame_id
            yield frame_id, timestamp, frame
            frame_id += 1
    finally:
        cap.release()


def ensure_output_dirs(save_dir: str | Path) -> dict[str, Path]:
    root = Path(save_dir)
    dirs = {
        "root": root,
        "annotated": root / "annotated",
        "depth": root / "depth",
        "bev": root / "bev",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def write_jsonl_record(
    handle,
    frame_id: int,
    timestamp: float,
    detections: list[ProjectedDetection],
    wall_time_ns: int | None = None,
    monotonic_ns: int | None = None,
    sensor_session_id: str | None = None,
    sensor_elapsed_ns: int | None = None,
) -> None:
    record = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "wall_time_ns": wall_time_ns,
        "monotonic_ns": monotonic_ns,
        "sensor_session_id": sensor_session_id,
        "sensor_elapsed_ns": sensor_elapsed_ns,
        "objects": [
            {
                "class_name": item.detection.class_name,
                "confidence": item.detection.confidence,
                "bbox_xyxy": list(item.detection.bbox_xyxy),
                "depth_m": item.depth_m,
                "x_m_lateral": item.x_m_lateral,
                "z_m_forward": item.z_m_forward,
                "bev_xy": list(item.bev_xy) if item.bev_xy is not None else None,
                "depth_source": item.depth_source,
                "camera_name": item.camera_name,
            }
            for item in detections
        ],
    }
    handle.write(json.dumps(record, separators=(",", ":")) + "\n")


def save_bgr_image(path: str | Path, image_bgr: np.ndarray) -> None:
    rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
    Image.fromarray(rgb).save(path)
