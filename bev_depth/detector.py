from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .config import DetectorConfig
from .types import Detection


class DetectorBackend(Protocol):
    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        ...


class SyntheticDetectorBackend:
    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        return [
            Detection(
                class_name="car",
                confidence=0.99,
                bbox_xyxy=(0.42 * w, 0.45 * h, 0.58 * w, 0.78 * h),
            )
        ]


class UltralyticsDetectorBackend:
    def __init__(self, config: DetectorConfig) -> None:
        if not config.path:
            raise FileNotFoundError(f"Detector model not found: {config.path}")
        path = Path(config.path)
        is_model_alias = path.parent == Path(".") and path.suffix == ".pt"
        if not is_model_alias and not path.exists():
            raise FileNotFoundError(f"Detector model not found: {config.path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics detector backend requires `pip install ultralytics`. "
                "Use a .engine model path on Thor for TensorRT inference."
            ) from exc
        self.model = YOLO(config.path)
        self.config = config
        self.allowed = set(config.classes)

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        try:
            from PIL import Image
            import torch
        except ImportError as exc:
            raise RuntimeError("Ultralytics tensor path requires Pillow and torch.") from exc
        original_h, original_w = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1]
        image = Image.fromarray(frame_rgb).resize(
            (self.config.input_size, self.config.input_size)
        )
        tensor = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        tensor = tensor.view(self.config.input_size, self.config.input_size, 3)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        results = self.model.predict(
            source=tensor,
            imgsz=self.config.input_size,
            conf=self.config.confidence,
            iou=self.config.iou,
            verbose=False,
        )
        if not results:
            return []
        result = results[0]
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(names.get(class_id, class_id))
            if self.allowed and class_name not in self.allowed:
                continue
            raw_xyxy = [float(v) for v in box.xyxy[0].tolist()]
            scale_x = original_w / float(self.config.input_size)
            scale_y = original_h / float(self.config.input_size)
            xyxy = (
                raw_xyxy[0] * scale_x,
                raw_xyxy[1] * scale_y,
                raw_xyxy[2] * scale_x,
                raw_xyxy[3] * scale_y,
            )
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                    bbox_xyxy=xyxy,
                )
            )
        return detections


def create_detector(config: DetectorConfig) -> DetectorBackend:
    backend = config.backend.lower()
    if backend == "synthetic":
        return SyntheticDetectorBackend()
    if backend == "ultralytics":
        return UltralyticsDetectorBackend(config)
    raise ValueError(f"Unsupported detector backend: {config.backend}")
