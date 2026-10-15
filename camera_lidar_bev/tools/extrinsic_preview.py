"""ROS-free capture guidance in image pixels and the LiDAR's own XYZ frame.

The range view uses spherical coordinates, not a camera projection or a fitted
camera--LiDAR transform. A clicked return is only a user's target hint.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass
class StabilityResult:
    elapsed_s: float
    ready: bool
    reset: bool


class CornerStability:
    """Require a time interval with little corner motion in the camera only."""

    def __init__(self, settle_seconds=2.0, max_rms_motion_px=2.0, max_gap_s=0.6):
        for name, value in (("settle_seconds", settle_seconds),
                            ("max_rms_motion_px", max_rms_motion_px),
                            ("max_gap_s", max_gap_s)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.settle_seconds = float(settle_seconds)
        self.max_rms_motion_px = float(max_rms_motion_px)
        self.max_gap_s = float(max_gap_s)
        self._baseline = None
        self._started = None
        self._last = None

    def clear(self):
        self._baseline = self._started = self._last = None

    def observe(self, corners, now):
        if not math.isfinite(now):
            raise ValueError("Observation time must be finite")
        points = None if corners is None else np.asarray(corners, dtype=float).reshape(-1, 2)
        if points is None or not len(points) or not np.isfinite(points).all():
            had_baseline = self._baseline is not None
            self.clear()
            return StabilityResult(0.0, False, had_baseline)
        reset = (self._baseline is None or points.shape != self._baseline.shape
                 or now < self._last or now - self._last > self.max_gap_s)
        if not reset:
            rms = min(float(np.sqrt(np.mean(np.sum((points - previous) ** 2, axis=1))))
                      for previous in (self._baseline, self._baseline[::-1]))
            reset = rms > self.max_rms_motion_px
        if reset:
            self._baseline = points.copy()
            self._started = now
        self._last = now
        elapsed = max(0.0, now - self._started)
        return StabilityResult(elapsed, elapsed >= self.settle_seconds, reset)


@dataclass
class RangeRaster:
    image: np.ndarray
    ranges_m: np.ndarray
    point_indices: np.ndarray
    xyz: np.ndarray
    plot_x: int
    plot_y: int
    plot_width: int
    plot_height: int
    azimuth_center_deg: float
    azimuth_span_deg: float
    elevation_center_deg: float
    elevation_span_deg: float
    elevation_extent_mode: str = "full"
    point_radius_px: int = 0

    def pixel_for_direction(self, azimuth_deg, elevation_deg):
        delta = (azimuth_deg - self.azimuth_center_deg + 180.0) % 360.0 - 180.0
        x = self.plot_x + (delta / self.azimuth_span_deg + 0.5) * self.plot_width
        y = self.plot_y + (0.5 - (elevation_deg - self.elevation_center_deg)
                          / self.elevation_span_deg) * (self.plot_height - 1)
        if not (self.plot_x <= x < self.plot_x + self.plot_width
                and self.plot_y <= y < self.plot_y + self.plot_height):
            return None
        return float(x), float(y)

    def sample(self, x, y, radius_px=3):
        """Nearest drawn pixel return; no automatic board classification."""
        px, py = int(round(x - self.plot_x)), int(round(y - self.plot_y))
        if not (0 <= px < self.plot_width and 0 <= py < self.plot_height):
            return None
        x0, x1 = max(0, px - radius_px), min(self.plot_width, px + radius_px + 1)
        y0, y1 = max(0, py - radius_px), min(self.plot_height, py + radius_px + 1)
        patch = self.point_indices[y0:y1, x0:x1]
        rows, cols = np.nonzero(patch >= 0)
        if not len(rows):
            return None
        nearest = np.argmin((rows + y0 - py) ** 2 + (cols + x0 - px) ** 2)
        row, col = int(rows[nearest] + y0), int(cols[nearest] + x0)
        index = int(self.point_indices[row, col])
        point = self.xyz[index].astype(float)
        distance = float(np.linalg.norm(point))
        return {
            "xyz_m": point.tolist(),
            "range_m": distance,
            "azimuth_deg": float(np.degrees(np.arctan2(point[1], point[0]))),
            "elevation_deg": float(np.degrees(np.arctan2(point[2], np.hypot(point[0], point[1])))),
            "decoded_xyz_index": index,
            "selection_source": "user_clicked_return_not_automatic_board_detection",
        }

    def sample_direction(self, azimuth_deg, elevation_deg, tolerance_deg=0.8):
        pixel = self.pixel_for_direction(azimuth_deg, elevation_deg)
        if pixel is None:
            return None
        sample = self.sample(*pixel)
        if sample is None:
            return None
        azimuth_error = (sample["azimuth_deg"] - azimuth_deg + 180.0) % 360.0 - 180.0
        # Account for spherical convergence at high elevations.
        error = math.hypot(azimuth_error * math.cos(math.radians(elevation_deg)),
                           sample["elevation_deg"] - elevation_deg)
        return sample if error <= tolerance_deg else None


def _elevation_limits(elevation):
    """Include every measured direction, with padding and a minimum 5-degree span."""
    if not len(elevation):
        return -90.0, 90.0
    lower, upper = float(np.min(elevation)), float(np.max(elevation))
    padding = max(2.0, (upper - lower) * 0.08)
    lower, upper = max(-90.0, lower - padding), min(90.0, upper + padding)
    if upper - lower < 5.0:
        center = float(np.clip((upper + lower) / 2.0, -87.5, 87.5))
        lower, upper = center - 2.5, center + 2.5
    return lower, upper


def _return_footprints(ranges, indices, radius_px):
    """Draw small measured-return markers; every pixel retains an original index.

    Marker overlap keeps the nearer original return. This never interpolates XYZ
    or produces a new point, and gaps wider than a marker stay empty.
    """
    if radius_px == 0:
        return ranges, indices
    height, width = ranges.shape
    enlarged_ranges, enlarged_indices = ranges.copy(), indices.copy()
    for dy in range(-radius_px, radius_px + 1):
        for dx in range(-radius_px, radius_px + 1):
            if dx == 0 and dy == 0:
                continue
            source_y = slice(max(0, -dy), min(height, height - dy))
            source_x = slice(max(0, -dx), min(width, width - dx))
            target_y = slice(max(0, dy), min(height, height + dy))
            target_x = slice(max(0, dx), min(width, width + dx))
            source_ranges, source_indices = ranges[source_y, source_x], indices[source_y, source_x]
            target_ranges = enlarged_ranges[target_y, target_x]
            target_indices = enlarged_indices[target_y, target_x]
            closer = ((source_indices >= 0)
                      & ((source_ranges < target_ranges)
                         | ((source_ranges == target_ranges) & (source_indices < target_indices))))
            target_ranges[closer] = source_ranges[closer]
            target_indices[closer] = source_indices[closer]
    return enlarged_ranges, enlarged_indices


def render_range_view(xyz, width=640, height=480, color_max_m=15.0,
                      zoom_direction=None, elevation_limits_deg=None, point_radius_px=0,
                      color_mode="white"):
    """Render indexed measured returns; automatically fit their vertical extent.

    ``elevation_limits_deg=None`` includes all valid measured elevations plus
    padding. An explicit (-90, 90) selects the full sphere. White measured points
    on black are the default. Optional three-pixel markers retain original XYZ.
    """
    points = np.asarray(xyz)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "fi":
        raise ValueError("xyz must be a numeric N x 3 array")
    if type(width) is not int or type(height) is not int or width < 320 or height < 240:
        raise ValueError("Range pane dimensions must be integers at least 320 x 240")
    if isinstance(color_max_m, bool) or not math.isfinite(color_max_m) or color_max_m <= 0:
        raise ValueError("color_max_m must be finite and positive")
    if type(point_radius_px) is not int or not 0 <= point_radius_px <= 1:
        raise ValueError("point_radius_px must be 0 or 1")
    if color_mode not in ("white", "range"):
        raise ValueError("color_mode must be 'white' or 'range'")
    if elevation_limits_deg is not None:
        if (len(elevation_limits_deg) != 2 or not np.isfinite(elevation_limits_deg).all()
                or not -90 <= elevation_limits_deg[0] < elevation_limits_deg[1] <= 90):
            raise ValueError("elevation_limits_deg must be increasing limits within [-90, 90]")
    azimuth_center, elevation_center, azimuth_span, elevation_span = 0.0, 0.0, 360.0, 180.0
    elevation_extent_mode = "auto"
    if zoom_direction is not None:
        if len(zoom_direction) != 2 or not np.isfinite(zoom_direction).all():
            raise ValueError("zoom_direction must contain finite azimuth and elevation")
        azimuth_center = (float(zoom_direction[0]) + 180.0) % 360.0 - 180.0
        elevation_center = float(np.clip(zoom_direction[1], -70.0, 70.0))
        azimuth_span, elevation_span = 60.0, 40.0
        elevation_extent_mode = "zoom"
    plot_x, plot_y, plot_width, plot_height = 34, 34, width - 52, height - 102
    finite = np.isfinite(points).all(axis=1)
    # Avoid feeding infinities through trigonometry and retain original indices.
    valid_indices = np.flatnonzero(finite)
    good = points[valid_indices].astype(np.float32, copy=False)
    ranges = np.sqrt(np.sum(good * good, axis=1))
    positive = np.isfinite(ranges) & (ranges > 0.0)
    valid_indices, good, ranges = valid_indices[positive], good[positive], ranges[positive]
    azimuth = np.degrees(np.arctan2(good[:, 1], good[:, 0]))
    elevation = np.degrees(np.arctan2(good[:, 2], np.hypot(good[:, 0], good[:, 1])))
    if zoom_direction is None:
        lower, upper = (_elevation_limits(elevation) if elevation_limits_deg is None
                        else tuple(map(float, elevation_limits_deg)))
        elevation_center, elevation_span = (lower + upper) / 2, upper - lower
        if elevation_limits_deg is not None:
            elevation_extent_mode = "fixed"
    delta = (azimuth - azimuth_center + 180.0) % 360.0 - 180.0
    x_float = (delta / azimuth_span + 0.5) * plot_width
    y_float = (0.5 - (elevation - elevation_center) / elevation_span) * (plot_height - 1)
    inside = ((x_float >= 0) & (x_float < plot_width)
              & (y_float >= 0) & (y_float < plot_height))
    valid_indices, ranges = valid_indices[inside], ranges[inside]
    columns = x_float[inside].astype(np.int32)
    rows = y_float[inside].astype(np.int32)
    cells = rows * plot_width + columns
    minimum = np.full(plot_width * plot_height, np.inf, dtype=np.float32)
    np.minimum.at(minimum, cells, ranges)
    winners = ranges == minimum[cells]
    # Pick one deterministic original XYZ index when equal-range returns collide.
    missing = np.iinfo(np.int32).max
    indices = np.full(plot_width * plot_height, missing, dtype=np.int32)
    np.minimum.at(indices, cells[winners], valid_indices[winners].astype(np.int32))
    indices[indices == missing] = -1
    minimum, indices = _return_footprints(minimum.reshape(plot_height, plot_width),
                                          indices.reshape(plot_height, plot_width), point_radius_px)
    if color_mode == "white":
        colors = np.zeros((plot_height, plot_width, 3), dtype=np.uint8)
        colors[np.isfinite(minimum)] = (255, 255, 255)
    else:
        values = np.clip(np.where(np.isfinite(minimum), minimum, 0.0) / color_max_m, 0, 1)
        colors = cv2.applyColorMap((values.reshape(plot_height, plot_width) * 255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
        colors[~np.isfinite(minimum.reshape(plot_height, plot_width))] = (0, 0, 0)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[plot_y:plot_y + plot_height, plot_x:plot_x + plot_width] = colors
    appearance = "white measured points" if color_mode == "white" else "range (m)"
    scope = "LiDAR 360 deg" if zoom_direction is None else "LiDAR ZOOM"
    extent = "fitted elevation" if elevation_extent_mode == "auto" else "elevation deg"
    label = f"{scope} | {appearance} | {extent}"
    cv2.putText(image, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
                (240, 240, 240), 1, cv2.LINE_AA)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = int(plot_x + fraction * (plot_width - 1))
        angle = azimuth_center + (fraction - 0.5) * azimuth_span
        angle = (angle + 180.0) % 360.0 - 180.0
        cv2.line(image, (x, plot_y + plot_height), (x, plot_y + plot_height + 4), (170, 170, 170), 1)
        cv2.putText(image, f"{angle:.0f}", (max(0, x - 15), plot_y + plot_height + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1)
    for fraction in (0.0, 0.5, 1.0):
        y = int(plot_y + fraction * (plot_height - 1))
        angle = elevation_center + (0.5 - fraction) * elevation_span
        cv2.putText(image, f"{angle:+.0f}", (0, min(height - 1, y + 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (190, 190, 190), 1)
    marker_note = f"{2 * point_radius_px + 1} px measured-return markers"
    cv2.putText(image, f"Azimuth / elevation deg; {marker_note}", (plot_x, height - 41),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (170, 170, 170), 1)
    if color_mode == "range":
        gradient = np.tile(np.linspace(0, 255, plot_width, dtype=np.uint8), (10, 1))
        image[height - 28:height - 18, plot_x:plot_x + plot_width] = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
        cv2.putText(image, "0 m", (plot_x, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
        cv2.putText(image, f"{color_max_m:g}+ m", (width - 90, height - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)
    else:
        cv2.putText(image, "Click a measured point for XYZ direction and range (m)", (plot_x, height - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    return RangeRaster(image, minimum.reshape(plot_height, plot_width),
                       indices.reshape(plot_height, plot_width), points,
                       plot_x, plot_y, plot_width, plot_height,
                       azimuth_center, azimuth_span, elevation_center, elevation_span,
                       elevation_extent_mode, point_radius_px)
