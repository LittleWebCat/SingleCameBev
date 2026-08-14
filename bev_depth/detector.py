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

    def infer_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
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

    def infer_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        return [self.infer(frame) for frame in frames_bgr]


class CompositeDetectorBackend:
    def __init__(self, backends: tuple[DetectorBackend, ...]) -> None:
        self.backends = backends

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        detections: list[Detection] = []
        for backend in self.backends:
            detections.extend(backend.infer(frame_bgr))
        return detections

    def infer_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        combined = [[] for _ in frames_bgr]
        for backend in self.backends:
            batch_method = getattr(backend, "infer_batch", None)
            batch = (
                batch_method(frames_bgr)
                if callable(batch_method)
                else [backend.infer(frame) for frame in frames_bgr]
            )
            if len(batch) != len(frames_bgr):
                raise RuntimeError(
                    f"Detector returned {len(batch)} batch items for {len(frames_bgr)} frames"
                )
            for output, detections in zip(combined, batch):
                output.extend(detections)
        return combined


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
        self.model = YOLO(config.path, task="detect")
        self.config = config
        self.allowed = set(config.classes)

    def _convert_result(self, result) -> list[Detection]:
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

    def infer_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        if not frames_bgr:
            return []
        results = self.model.predict(
            source=frames_bgr,
            imgsz=self.config.input_size,
            conf=self.config.confidence,
            iou=self.config.iou,
            verbose=False,
        )
        if len(results) != len(frames_bgr):
            raise RuntimeError(
                f"Detector returned {len(results)} batch items for {len(frames_bgr)} frames"
            )
        return [self._convert_result(result) for result in results]

    def infer(self, frame_bgr: np.ndarray) -> list[Detection]:
        return self.infer_batch([frame_bgr])[0]


def _create_single_detector(config: DetectorConfig) -> DetectorBackend:
    backend = config.backend.lower()
    if backend == "synthetic":
        return SyntheticDetectorBackend()
    if backend == "ultralytics":
        return UltralyticsDetectorBackend(config)
    raise ValueError(f"Unsupported detector backend: {config.backend}")


def create_detector(
    config: DetectorConfig, traffic_config: DetectorConfig | None = None
) -> DetectorBackend:
    primary = _create_single_detector(config)
    if traffic_config is None:
        return primary
    return CompositeDetectorBackend((primary, _create_single_detector(traffic_config)))
