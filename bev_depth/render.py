from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .config import BevConfig
from .types import ProjectedDetection


def _bgr_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    return Image.fromarray(rgb)


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


def draw_annotated_frame(
    frame_bgr: np.ndarray, detections: list[ProjectedDetection]
) -> np.ndarray:
    image = _bgr_to_pil(frame_bgr)
    draw = ImageDraw.Draw(image)
    for item in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in item.detection.bbox_xyxy)
        draw.rectangle((x1, y1, x2, y2), outline=(40, 220, 40), width=2)
        if item.depth_m is None:
            label = f"{item.detection.class_name} {item.detection.confidence:.2f} depth?"
        else:
            label = (
                f"{item.detection.class_name} {item.detection.confidence:.2f} "
                f"{item.z_m_forward:.1f}m"
            )
        label_y = max(2, y1 - 16)
        draw.rectangle((x1, label_y, x1 + max(80, 7 * len(label)), label_y + 14), fill=(0, 0, 0))
        draw.text((x1 + 2, label_y), label, fill=(40, 220, 40))
    return _pil_to_bgr(image)


def render_depth(depth_map: np.ndarray) -> np.ndarray:
    finite = depth_map[(depth_map > -1.0e6) & (depth_map < 1.0e6)]
    if finite.size == 0:
        scaled = np.zeros(depth_map.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [2, 98])
        if high <= low:
            high = low + 1.0
        scaled_float = (depth_map - low) * 255.0 / (high - low)
        scaled_float[scaled_float < 0] = 0
        scaled_float[scaled_float > 255] = 255
        scaled = scaled_float.astype(np.uint8)

    red = scaled
    green = np.full_like(scaled, 80)
    blue = (255 - scaled).astype(np.uint8)
    rgb = np.empty((scaled.shape[0], scaled.shape[1], 3), dtype=np.uint8)
    rgb[:, :, 0] = red
    rgb[:, :, 1] = green
    rgb[:, :, 2] = blue
    return np.ascontiguousarray(rgb[:, :, ::-1])


def render_bev(bev: BevConfig, detections: list[ProjectedDetection]) -> np.ndarray:
    width = int(round(bev.lateral_range_m * bev.pixels_per_meter))
    height = int(round(bev.forward_range_m * bev.pixels_per_meter))
    image = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    center_x = width // 2

    for meters in range(0, int(bev.forward_range_m) + 1, 10):
        y = int(round(height - meters * bev.pixels_per_meter))
        draw.line((0, y, width, y), fill=(220, 220, 220), width=1)
        draw.text((5, max(2, y - 14)), f"{meters}m", fill=(90, 90, 90))
    draw.line((center_x, 0, center_x, height), fill=(130, 130, 130), width=1)
    draw.rectangle((center_x - 8, height - 18, center_x + 8, height - 2), fill=(30, 30, 30))

    for item in detections:
        if item.bev_xy is None:
            continue
        x, y = item.bev_xy
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(255, 80, 0))
        label = item.detection.class_name
        if item.z_m_forward is not None:
            label += f" {item.z_m_forward:.1f}m"
        draw.text((min(x + 8, width - 80), max(2, y - 14)), label, fill=(255, 80, 0))
    return _pil_to_bgr(image)
