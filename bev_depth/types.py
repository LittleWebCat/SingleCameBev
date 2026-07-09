from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ProjectedDetection:
    detection: Detection
    depth_m: float | None
    x_m_lateral: float | None
    z_m_forward: float | None
    bev_xy: tuple[int, int] | None
    depth_source: str
