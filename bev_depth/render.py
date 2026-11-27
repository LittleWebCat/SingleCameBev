from __future__ import annotations

import cv2
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


# One color per class (RGB), shared by the camera boxes and the BEV markers.
CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "car": (30, 120, 255),
    "truck": (140, 70, 220),
    "bus": (255, 150, 0),
    "person": (230, 40, 160),
    "bicycle": (0, 175, 175),
    "motorcycle": (150, 100, 40),
    "stop sign": (210, 35, 35),
    "traffic light": (235, 190, 35),
    "red": (225, 45, 45),
    "yellow": (235, 190, 35),
    "green": (45, 190, 75),
}
UNKNOWN_COLOR_RGB = (120, 120, 120)
LEGEND_CLASSES = ("car", "truck", "bus", "person", "bicycle", "motorcycle")


def class_color_rgb(class_name: str) -> tuple[int, int, int]:
    return CLASS_COLORS_RGB.get(class_name, UNKNOWN_COLOR_RGB)


def class_color_bgr(class_name: str) -> tuple[int, int, int]:
    r, g, b = class_color_rgb(class_name)
    return (b, g, r)


def draw_annotated_frame(
    frame_bgr: np.ndarray, detections: list[ProjectedDetection]
) -> np.ndarray:
    """Boxes colored by class; the only text is the track ID when tracking is on."""
    result = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 3
    for item in detections:
        color = class_color_bgr(item.detection.class_name)
        x1, y1, x2, y2 = (int(round(v)) for v in item.detection.bbox_xyxy)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        if item.track_id is None:
            continue

        label = f"#{item.track_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        label_x = max(0, min(x1, result.shape[1] - text_width - 12))
        label_y = max(text_height + 8, y1 - 8)
        cv2.rectangle(
            result,
            (label_x, label_y - text_height - 8),
            (label_x + text_width + 12, label_y + baseline + 6),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            result, label, (label_x + 6, label_y), font, font_scale, color, font_thickness,
            cv2.LINE_AA,
        )
    return result


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


def _draw_bev_marker(draw: ImageDraw.ImageDraw, x: int, y: int, class_name: str) -> tuple[int, int, int]:
    if class_name == "stop sign":
        points = [
            (x - 3, y - 8),
            (x + 3, y - 8),
            (x + 8, y - 3),
            (x + 8, y + 3),
            (x + 3, y + 8),
            (x - 3, y + 8),
            (x - 8, y + 3),
            (x - 8, y - 3),
        ]
        draw.polygon(points, fill=(210, 35, 35), outline=(255, 255, 255), width=2)
        return (180, 20, 20)

    if class_name in {"traffic light", "red", "yellow", "green"}:
        draw.rounded_rectangle(
            (x - 6, y - 11, x + 6, y + 11),
            radius=3,
            fill=(25, 25, 25),
            outline=(90, 90, 90),
            width=1,
        )
        active = class_name if class_name != "traffic light" else None
        red = (225, 45, 45) if active in {None, "red"} else (75, 25, 25)
        yellow = (235, 190, 35) if active in {None, "yellow"} else (70, 60, 20)
        green = (45, 190, 75) if active in {None, "green"} else (20, 65, 30)
        draw.ellipse((x - 2, y - 8, x + 2, y - 4), fill=red)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=yellow)
        draw.ellipse((x - 2, y + 4, x + 2, y + 8), fill=green)
        return {
            "red": (180, 20, 20),
            "yellow": (180, 130, 20),
            "green": (25, 110, 45),
        }.get(class_name, (25, 110, 45))

    color = class_color_rgb(class_name)
    draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline=(255, 255, 255), width=2)
    return color


def _draw_legend(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    """Class colors in the bottom-right corner (distance labels are on the left)."""
    row_h, box_w = 15, 92
    top = height - row_h * len(LEGEND_CLASSES) - 10
    left = width - box_w - 6
    draw.rectangle((left, top - 4, width - 6, height - 6), fill=(255, 255, 255), outline=(200, 200, 200))
    for i, name in enumerate(LEGEND_CLASSES):
        y = top + i * row_h
        draw.ellipse((left + 6, y + 2, left + 16, y + 12), fill=class_color_rgb(name))
        draw.text((left + 22, y + 1), name, fill=(60, 60, 60))


def render_bev(bev: BevConfig, detections: list[ProjectedDetection]) -> np.ndarray:
    width = int(round(bev.lateral_range_m * bev.pixels_per_meter))
    height = int(
        round((bev.forward_range_m + bev.rear_range_m) * bev.pixels_per_meter)
    )
    image = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    center_x = width // 2

    rear_limit = int(bev.rear_range_m)
    forward_limit = int(bev.forward_range_m)
    first_grid_m = -((rear_limit // 10) * 10)
    for meters in range(first_grid_m, forward_limit + 1, 10):
        y = int(round((bev.forward_range_m - meters) * bev.pixels_per_meter))
        draw.line((0, y, width, y), fill=(220, 220, 220), width=1)
        distance_label = f"{meters}m" if meters >= 0 else f"rear {abs(meters)}m"
        draw.text((5, max(2, y - 14)), distance_label, fill=(90, 90, 90))
    draw.line((center_x, 0, center_x, height), fill=(130, 130, 130), width=1)
    ego_y = int(round(bev.forward_range_m * bev.pixels_per_meter))
    vehicle_top = max(0, ego_y - 12)
    vehicle_bottom = min(height - 1, ego_y + (10 if bev.rear_range_m else -2))
    draw.rectangle(
        (center_x - 8, vehicle_top, center_x + 8, vehicle_bottom),
        fill=(30, 30, 30),
    )

    for item in detections:
        if item.bev_xy is None:
            continue
        x, y = item.bev_xy
        label_color = _draw_bev_marker(draw, x, y, item.detection.class_name)
        # Color already says the class; only the track ID is written.
        if item.track_id is not None:
            draw.text((min(x + 10, width - 40), max(2, y - 14)), f"#{item.track_id}", fill=label_color)
    _draw_legend(draw, width, height)
    return _pil_to_bgr(image)
