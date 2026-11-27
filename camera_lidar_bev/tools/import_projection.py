"""ROS-free raw-image projections for candidate calibration source frames.

This is a visual diagnostic: it neither chooses a frame nor publishes a transform.
The supplied pinhole K/D always refer to the image's exact native dimensions.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from import_camera_lidar_calibration import validate_camera


def _numeric_array(value, shape, name):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in "fi" or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite numeric array of shape {shape}")
    return result.astype(np.float64, copy=False)


def _transform(value, name):
    result = _numeric_array(value, (4, 4), name)
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0)):
        raise ValueError(f"{name} must be a rigid homogeneous transform with a proper rotation")
    return result


@dataclass
class ProjectedCloud:
    """One nearest-depth measured return per drawn pixel, with original indices."""

    pixels_px: np.ndarray
    pixel_indices: np.ndarray
    point_indices: np.ndarray
    camera_depth_m: np.ndarray
    image_size: tuple
    input_count: int
    valid_count: int
    front_count: int
    in_frame_count: int
    radial_domain_excluded_count: int = 0

    @property
    def projected_count(self):
        return len(self.point_indices)


def radial_domain_radius_squared(distortion):
    """First positive zero of radial scale or radial-map derivative, in r².

    For plumb_bob, r_d = r * (1 + k1*r² + k2*r⁴ + k3*r⁶). Its radial
    derivative is 1 + 3*k1*r² + 5*k2*r⁴ + 7*k3*r⁶. The central branch ends
    at the first positive root of either polynomial; farther rays can fold
    back into valid image pixels. This guard derives from the supplied lens
    model, not a guessed FoV. It does not validate tangential distortion or
    establish the lens's independently calibrated angular coverage.
    """
    values = _numeric_array(distortion, (5,), "D")
    k1, k2, _, _, k3 = values
    limits = []
    for coefficients in ([k3, k2, k1, 1.0], [7*k3, 5*k2, 3*k1, 1.0]):
        # np.roots drops exact leading zeros, including the undistorted model.
        for root in np.roots(coefficients):
            if root.real > 0 and abs(root.imag) <= 1e-7 * max(1.0, abs(root.real)):
                limits.append(float(root.real))
    return min(limits) if limits else math.inf


def project_cloud(xyz, K, D, T_camera_from_source, image_size):
    """Transform source XYZ in meters, then distort into the raw camera image.

    Invalid/zero source returns and points with optical Z <= 0 are discarded.
    Rays outside the radial model's central increasing branch are discarded
    before distortion, preventing folded rays from stealing real pixels.
    Pixel collisions keep the smallest optical depth; ties keep the earliest
    original index. Floating projections are checked against the image before
    rounding to their nearest pixel. No input arrays are changed.
    """
    points = np.asarray(xyz)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in "fi":
        raise ValueError("xyz must be a numeric N x 3 array")
    if (not isinstance(image_size, (tuple, list)) or len(image_size) != 2
            or any(type(value) is not int or value <= 0 for value in image_size)):
        raise ValueError("image_size must contain positive integer native width and height")
    width, height = image_size
    matrix = _numeric_array(K, (3, 3), "K")
    distortion = _numeric_array(D, (5,), "D")
    if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]],
                               [0, 0, 0, 0, 1], atol=1e-12, rtol=0)):
        raise ValueError("K must have positive focal lengths and standard zero-skew pinhole form")
    transform = _transform(T_camera_from_source, "T_camera_from_source")
    indices = np.flatnonzero(np.isfinite(points).all(axis=1) & np.any(points != 0, axis=1))
    valid_count = len(indices)
    camera = points[indices].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
    front = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)
    indices, camera = indices[front], camera[front]
    front_count = len(indices)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized = camera[:, :2] / camera[:, 2, None]
        radii_squared = np.sum(normalized * normalized, axis=1)
    central_branch = (np.isfinite(radii_squared)
                      & (radii_squared < radial_domain_radius_squared(distortion)))
    radial_domain_excluded_count = int(np.count_nonzero(~central_branch))
    indices, camera = indices[central_branch], camera[central_branch]
    pixels = np.empty((0, 2), dtype=np.float64)
    if len(indices):
        pixels = cv2.projectPoints(camera, np.zeros(3), np.zeros(3), matrix, distortion)[0].reshape(-1, 2)
    inside = (np.isfinite(pixels).all(axis=1) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    pixels, camera, indices = pixels[inside], camera[inside], indices[inside]
    in_frame_count = len(indices)
    raster = np.floor(pixels + 0.5).astype(np.int64)
    # The last half-pixel at the right/bottom edge belongs to the last pixel.
    raster = np.minimum(raster, [width - 1, height - 1])
    keys = raster[:, 1] * width + raster[:, 0]
    order = np.lexsort((indices, camera[:, 2], keys))
    unique = np.r_[True, np.diff(keys[order]) != 0] if len(order) else np.empty(0, dtype=bool)
    selected = order[unique]
    return ProjectedCloud(pixels[selected], raster[selected], indices[selected], camera[selected, 2],
                          (width, height), len(points), valid_count, front_count, in_frame_count,
                          radial_domain_excluded_count)


@dataclass
class HypothesisPreview:
    image: np.ndarray
    projections: dict
    hypotheses: dict
    view: str = "compare"


def _overlay(image, projection, max_display_points):
    result = image.copy()
    raster = projection.pixel_indices
    if len(raster) > max_display_points:
        raster = raster[np.linspace(0, len(raster) - 1, max_display_points, dtype=np.int64)]
    # Black outline and a single white center: display markers only, no colored depth map.
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            x, y = raster[:, 0] + dx, raster[:, 1] + dy
            inside = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
            result[y[inside], x[inside]] = 0
    result[raster[:, 1], raster[:, 0]] = 255
    return result, len(raster)


def render_calibration_preview(image, xyz, intrinsics, extrinsics, T_sensor_from_lidar,
                               max_display_points=15000, view="compare"):
    """Render both candidates or B chosen by the user for provisional viewing.

    ``T_sensor_from_lidar`` must be provided from verified driver TF/metadata in
    meters. Choosing B for viewing records a visual preference, not measured
    accuracy. Its camera-from-os_lidar matrix composes external camera-from-
    os_sensor with observed os_sensor-from-os_lidar, without modifying either.
    Temporal pairing and sensor-motion checks are the caller's responsibility.
    """
    camera = validate_camera(intrinsics)
    width, height = camera["image_size"]
    raw = np.asarray(image)
    if raw.dtype != np.uint8 or raw.shape != (height, width, 3):
        raise ValueError(f"Raw BGR image must match native calibration dimensions {width} x {height}")
    if type(max_display_points) is not int or max_display_points <= 0:
        raise ValueError("max_display_points must be a positive integer")
    if view not in ("compare", "b"):
        raise ValueError("view must be compare or b")
    if (not isinstance(extrinsics, dict) or extrinsics.get("direction") != "lidar_to_camera"
            or extrinsics.get("units") != "m"):
        raise ValueError("Extrinsics must explicitly map lidar_to_camera in meters")
    external = _transform(extrinsics.get("T_camera_from_lidar"), "T_camera_from_lidar")
    sensor_from_lidar = _transform(T_sensor_from_lidar, "T_sensor_from_lidar")
    hypotheses = {"direct_os_lidar": external, "via_os_sensor": external @ sensor_from_lidar}
    labels = {"direct_os_lidar": "A: external source = os_lidar (UNCONFIRMED)",
              "via_os_sensor": "B: external source = os_sensor (UNCONFIRMED)"}
    if view == "b":
        hypotheses = {"via_os_sensor": hypotheses["via_os_sensor"]}
        labels["via_os_sensor"] = "B: source = os_sensor (USER VISUAL CHOICE)"
    panes, projections = [], {}
    for name, transform in hypotheses.items():
        projection = project_cloud(xyz, camera["K"], camera["D"], transform, (width, height))
        projections[name] = projection
        overlay, drawn_count = _overlay(raw, projection, max_display_points)
        pane = np.zeros((height + 60, width, 3), dtype=np.uint8)
        pane[60:] = overlay
        cv2.putText(pane, "CANDIDATE " + labels[name], (10, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(pane, f"Raw distorted {width}x{height}; {projection.projected_count} pixels; "
                    f"{drawn_count} drawn; " + ("accuracy provisional" if view == "b" else "frame not selected"), (10, 47),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        panes.append(pane)
    return HypothesisPreview(np.hstack(panes), projections, hypotheses, view)


def render_two_hypotheses(image, xyz, intrinsics, extrinsics, T_sensor_from_lidar,
                          max_display_points=15000):
    """Backward-compatible comparison of both unconfirmed source frames."""
    return render_calibration_preview(image, xyz, intrinsics, extrinsics, T_sensor_from_lidar,
                                      max_display_points, view="compare")
