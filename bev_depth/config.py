from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class InputConfig:
    width: int
    height: int
    fps: int
    fourcc: str
    camera_index_backend: str


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class Extrinsics:
    height_m: float
    pitch_deg: float
    yaw_deg: float
    roll_deg: float


@dataclass(frozen=True)
class DetectorConfig:
    backend: str
    path: str
    input_size: int
    confidence: float
    iou: float
    classes: tuple[str, ...]


@dataclass(frozen=True)
class DepthConfig:
    backend: str
    path: str
    input_size: int
    cadence: int
    metric_scale: float
    metric_bias: float
    inverse_depth: bool
    constant_depth_m: float | None = None


@dataclass(frozen=True)
class BevConfig:
    forward_range_m: float
    lateral_range_m: float
    pixels_per_meter: int


@dataclass(frozen=True)
class RuntimeConfig:
    target_fps: int
    save_images: bool
    max_frames: int


@dataclass(frozen=True)
class AppConfig:
    input: InputConfig
    detector: DetectorConfig
    depth: DepthConfig
    intrinsics: Intrinsics
    extrinsics: Extrinsics
    bev: BevConfig
    runtime: RuntimeConfig


def _get(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    value = mapping.get(key, default)
    if value is None:
        return default
    return value


def load_config(path: str | Path) -> AppConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    input_config = data.get("input", {})
    models = data.get("models", {})
    detector = models.get("detector", {})
    depth = models.get("depth", {})
    camera = data.get("camera", {})
    intrinsics = camera.get("intrinsics", {})
    extrinsics = camera.get("extrinsics", {})
    bev = data.get("bev", {})
    runtime = data.get("runtime", {})

    return AppConfig(
        input=InputConfig(
            width=int(_get(input_config, "width", 1600)),
            height=int(_get(input_config, "height", 1200)),
            fps=int(_get(input_config, "fps", 15)),
            fourcc=str(_get(input_config, "fourcc", "MJPG")),
            camera_index_backend=str(_get(input_config, "camera_index_backend", "auto")),
        ),
        detector=DetectorConfig(
            backend=str(_get(detector, "backend", "ultralytics")),
            path=str(_get(detector, "path", "")),
            input_size=int(_get(detector, "input_size", 640)),
            confidence=float(_get(detector, "confidence", 0.35)),
            iou=float(_get(detector, "iou", 0.5)),
            classes=tuple(str(name) for name in _get(detector, "classes", [])),
        ),
        depth=DepthConfig(
            backend=str(_get(depth, "backend", "tensorrt")),
            path=str(_get(depth, "path", "")),
            input_size=int(_get(depth, "input_size", 518)),
            cadence=max(1, int(_get(depth, "cadence", 2))),
            metric_scale=float(_get(depth, "metric_scale", 1.0)),
            metric_bias=float(_get(depth, "metric_bias", 0.0)),
            inverse_depth=bool(_get(depth, "inverse_depth", False)),
            constant_depth_m=(
                float(depth["constant_depth_m"]) if "constant_depth_m" in depth else None
            ),
        ),
        intrinsics=Intrinsics(
            fx=float(_get(intrinsics, "fx", 1200.0)),
            fy=float(_get(intrinsics, "fy", 1200.0)),
            cx=float(_get(intrinsics, "cx", 960.0)),
            cy=float(_get(intrinsics, "cy", 540.0)),
        ),
        extrinsics=Extrinsics(
            height_m=float(_get(extrinsics, "height_m", 1.45)),
            pitch_deg=float(_get(extrinsics, "pitch_deg", 0.0)),
            yaw_deg=float(_get(extrinsics, "yaw_deg", 0.0)),
            roll_deg=float(_get(extrinsics, "roll_deg", 0.0)),
        ),
        bev=BevConfig(
            forward_range_m=float(_get(bev, "forward_range_m", 80.0)),
            lateral_range_m=float(_get(bev, "lateral_range_m", 20.0)),
            pixels_per_meter=int(_get(bev, "pixels_per_meter", 10)),
        ),
        runtime=RuntimeConfig(
            target_fps=int(_get(runtime, "target_fps", 30)),
            save_images=bool(_get(runtime, "save_images", True)),
            max_frames=int(_get(runtime, "max_frames", 0)),
        ),
    )
