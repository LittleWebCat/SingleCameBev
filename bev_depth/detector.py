from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from .config import DetectorConfig
from .types import Detection

COCO_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)


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
        results = self.model.predict(
            source=frame_bgr,
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
            if class_name == f"class{class_id}" and class_id < len(COCO_CLASS_NAMES):
                class_name = COCO_CLASS_NAMES[class_id]
            if self.allowed and class_name not in self.allowed:
                continue
            raw_xyxy = [float(v) for v in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                    bbox_xyxy=tuple(raw_xyxy),
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
