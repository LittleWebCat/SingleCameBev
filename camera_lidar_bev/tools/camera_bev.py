"""ROS-free reference for camera-colored measured LiDAR returns.

The imported external transform is interpreted using the user's provisional
B choice: camera-from-os_sensor. Driver os_sensor-from-os_lidar is required;
geometry and camera projection therefore use the same explicit composition.
This module does not estimate occupancy, fill camera surfaces, accumulate
scans, publish TF, or qualify the supplied calibration's accuracy.
"""

from dataclasses import dataclass
import math
import numbers

import cv2
import numpy as np

from import_camera_lidar_calibration import validate_camera
from import_projection import ProjectedCloud, _transform, project_cloud


@dataclass(frozen=True)
class BevExtent:
    """Half-open measured-return extent in os_sensor meters; X up, Y left.

    The last cell may be shorter when an extent is not a resolution multiple.
    These visualization cells are distinct from the project's base_link
    occupancy-grid design contract.
    """

    x_min_m: float = 0.0
    x_max_m: float = 20.0
    y_min_m: float = -10.0
    y_max_m: float = 10.0
    resolution_m: float = 0.05

    def __post_init__(self):
        for name in ("x_min_m", "x_max_m", "y_min_m", "y_max_m", "resolution_m"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, numbers.Real)
                    or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m:
            raise ValueError("BEV maximum coordinates must exceed minimum coordinates")
        if self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if self.width * self.height > 16_000_000:
            raise ValueError("BEV extent exceeds 16 million visualization cells")

    @property
    def width(self):
        return math.ceil((self.y_max_m - self.y_min_m) / self.resolution_m)

    @property
    def height(self):
        return math.ceil((self.x_max_m - self.x_min_m) / self.resolution_m)


@dataclass
class ColoredReturns:
    """Full original-order arrays, including invalid/uncolored source returns.

    Colors use OpenCV BGR byte order. Only the nearest optical-depth measured
    return at each rounded camera pixel receives a camera color. This sparse
    z-buffer cannot establish visibility against unmeasured surfaces.
    """

    xyz_sensor: np.ndarray
    colors_bgr: np.ndarray
    colored_mask: np.ndarray
    valid_mask: np.ndarray
    projection: ProjectedCloud
    T_camera_from_lidar: np.ndarray

    @property
    def colored_count(self):
        return int(np.count_nonzero(self.colored_mask))


@dataclass
class BevPreview:
    """Native metric raster and original indices of the drawn returns."""

    image: np.ndarray
    point_indices: np.ndarray
    pixel_indices: np.ndarray
    colored_count: int
    in_extent_count: int
    extent: BevExtent
    returns: ColoredReturns


def associate_camera_colors(image_bgr, xyz_lidar, intrinsics, extrinsics,
                            T_sensor_from_lidar):
    """Associate native raw-image colors with the B candidate's LiDAR returns.

    No image resizing or undistortion occurs. ``project_cloud`` supplies its
    central radial-branch guard, floating bounds check, nearest-pixel rounding
    (floor(pixel + .5), then upper-edge clamp), and nearest optical-depth
    z-buffer. Equal-depth collisions retain the first original point index.
    Points behind/outside the camera or hidden by another measured return
    remain white; invalid source returns remain in arrays with valid_mask=False.
    """
    camera = validate_camera(intrinsics)
    width, height = camera["image_size"]
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.shape != (height, width, 3):
        raise ValueError(f"Raw BGR image must match native dimensions {width} x {height}")
    if (not isinstance(extrinsics, dict) or extrinsics.get("direction") != "lidar_to_camera"
            or extrinsics.get("units") != "m"):
        raise ValueError("Extrinsics must explicitly map lidar_to_camera in meters")
    external = _transform(extrinsics.get("T_camera_from_lidar"), "T_camera_from_os_sensor")
    sensor_from_lidar = _transform(T_sensor_from_lidar, "T_sensor_from_lidar")
    camera_from_lidar = external @ sensor_from_lidar
    projection = project_cloud(xyz_lidar, camera["K"], camera["D"],
                               camera_from_lidar, (width, height))
    points = np.asarray(xyz_lidar)
    with np.errstate(invalid="ignore", over="ignore"):
        xyz_sensor = points.astype(np.float64) @ sensor_from_lidar[:3, :3].T + sensor_from_lidar[:3, 3]
    valid = (np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1)
             & np.isfinite(xyz_sensor).all(axis=1))
    colors = np.full((len(points), 3), 255, dtype=np.uint8)
    colored = np.zeros(len(points), dtype=bool)
    selected = projection.point_indices
    pixels = projection.pixel_indices
    colors[selected] = image[pixels[:, 1], pixels[:, 0]]
    colored[selected] = True
    return ColoredReturns(xyz_sensor, colors, colored, valid, projection, camera_from_lidar)


def _draw_metric_grid(image, extent):
    """Dark two-meter reference lines, drawn before measured-return markers."""
    for axis in ("x", "y"):
        minimum, maximum = getattr(extent, f"{axis}_min_m"), getattr(extent, f"{axis}_max_m")
        for coordinate in np.arange(math.ceil(minimum / 2) * 2, maximum, 2.0):
            cell = int(math.floor((coordinate - minimum) / extent.resolution_m))
            color = (44, 44, 44) if coordinate == 0 else (22, 22, 22)
            if axis == "x":
                row = extent.height - 1 - cell
                if 0 <= row < extent.height:
                    image[row, :] = color
            else:
                column = extent.width - 1 - cell
                if 0 <= column < extent.width:
                    image[:, column] = color


def render_bev(returns, extent=BevExtent(), draw_grid=True, marker_radius_px=0):
    """Rasterize a single measured scan with +X up and +Y left in os_sensor.

    Where returns share a BEV cell, draw the largest sensor Z, then earliest
    original index. A higher uncolored return stays white even when a lower
    return was colored. Black background indicates absence of a drawn return,
    not observed free space. Positive marker radius only enlarges display dots.
    """
    if not isinstance(returns, ColoredReturns) or not isinstance(extent, BevExtent):
        raise TypeError("render_bev requires ColoredReturns and BevExtent")
    if type(marker_radius_px) is not int or not 0 <= marker_radius_px <= 3:
        raise ValueError("marker_radius_px must be an integer from 0 to 3")
    points = returns.xyz_sensor
    inside = (returns.valid_mask & (points[:, 0] >= extent.x_min_m)
              & (points[:, 0] < extent.x_max_m) & (points[:, 1] >= extent.y_min_m)
              & (points[:, 1] < extent.y_max_m))
    indices = np.flatnonzero(inside)
    selected_points = points[indices]
    forward = np.floor((selected_points[:, 0] - extent.x_min_m) / extent.resolution_m).astype(np.int64)
    lateral = np.floor((selected_points[:, 1] - extent.y_min_m) / extent.resolution_m).astype(np.int64)
    # Bounds are checked in metric coordinates first. Clamp only final-cell
    # floating roundoff, matching the half-open contract even nextafter(max).
    forward, lateral = np.minimum(forward, extent.height - 1), np.minimum(lateral, extent.width - 1)
    rows, columns = extent.height - 1 - forward, extent.width - 1 - lateral
    keys = rows * extent.width + columns
    order = np.lexsort((indices, -selected_points[:, 2], keys))
    unique = np.r_[True, np.diff(keys[order]) != 0] if len(order) else np.empty(0, dtype=bool)
    chosen = order[unique]
    chosen_indices = indices[chosen]
    pixels = np.column_stack((columns[chosen], rows[chosen]))
    image = np.zeros((extent.height, extent.width, 3), dtype=np.uint8)
    if draw_grid:
        _draw_metric_grid(image, extent)
    if marker_radius_px:
        for index, pixel in zip(chosen_indices, pixels):
            cv2.circle(image, tuple(int(value) for value in pixel), marker_radius_px,
                       tuple(int(value) for value in returns.colors_bgr[index]), -1)
    else:
        image[pixels[:, 1], pixels[:, 0]] = returns.colors_bgr[chosen_indices]
    return BevPreview(image, chosen_indices, pixels,
                      int(np.count_nonzero(returns.colored_mask[chosen_indices])),
                      len(indices), extent, returns)


def render_camera_bev(image_bgr, xyz_lidar, intrinsics, extrinsics, T_sensor_from_lidar,
                      extent=BevExtent(), draw_grid=True, marker_radius_px=0):
    """Convenience composition of native camera association and metric raster."""
    returns = associate_camera_colors(image_bgr, xyz_lidar, intrinsics, extrinsics,
                                      T_sensor_from_lidar)
    return render_bev(returns, extent, draw_grid, marker_radius_px)
