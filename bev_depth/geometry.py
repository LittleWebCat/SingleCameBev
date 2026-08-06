from __future__ import annotations

import math

import numpy as np

from .config import BevConfig, Extrinsics, Intrinsics


def scale_intrinsics(
    intrinsics: Intrinsics, frame_width: int, frame_height: int
) -> Intrinsics:
    """Scale calibrated pixel intrinsics to the dimensions of an input frame."""
    calibration_width = intrinsics.calibration_width or frame_width
    calibration_height = intrinsics.calibration_height or frame_height
    if calibration_width <= 0 or calibration_height <= 0:
        raise ValueError("intrinsics calibration dimensions must be positive")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if intrinsics.fx <= 0.0 or intrinsics.fy <= 0.0:
        raise ValueError("intrinsics focal lengths must be positive")
    if not (0.0 <= intrinsics.cx < calibration_width):
        raise ValueError(
            f"intrinsics cx={intrinsics.cx} is outside calibration width "
            f"{calibration_width}"
        )
    if not (0.0 <= intrinsics.cy < calibration_height):
        raise ValueError(
            f"intrinsics cy={intrinsics.cy} is outside calibration height "
            f"{calibration_height}"
        )

    scale_x = frame_width / calibration_width
    scale_y = frame_height / calibration_height
    return Intrinsics(
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=intrinsics.cx * scale_x,
        cy=intrinsics.cy * scale_y,
        calibration_width=frame_width,
        calibration_height=frame_height,
    )


def rotation_matrix(extrinsics: Extrinsics) -> np.ndarray:
    pitch = math.radians(extrinsics.pitch_deg)
    yaw = math.radians(extrinsics.yaw_deg)
    roll = math.radians(extrinsics.roll_deg)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float32,
    )
    ry = np.array(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rz = np.array(
        [
            [math.cos(roll), -math.sin(roll), 0.0],
            [math.sin(roll), math.cos(roll), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return rz @ ry @ rx


def pixel_depth_to_camera(
    u: float, v: float, depth_m: float, intrinsics: Intrinsics
) -> np.ndarray:
    x = (u - intrinsics.cx) * depth_m / intrinsics.fx
    y = (v - intrinsics.cy) * depth_m / intrinsics.fy
    z = depth_m
    return np.array([x, y, z], dtype=np.float32)


def camera_to_ego(point_camera: np.ndarray, extrinsics: Extrinsics) -> np.ndarray:
    point = rotation_matrix(extrinsics) @ point_camera
    point[1] += extrinsics.height_m
    return point


def pixel_depth_to_ego(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: Intrinsics,
    extrinsics: Extrinsics,
) -> tuple[float, float]:
    point_camera = pixel_depth_to_camera(u, v, depth_m, intrinsics)
    point_ego = camera_to_ego(point_camera, extrinsics)
    return float(point_ego[0]), float(point_ego[2])


def ego_to_bev_pixel(
    x_m_lateral: float, z_m_forward: float, bev: BevConfig
) -> tuple[int, int] | None:
    if z_m_forward < 0.0 or z_m_forward > bev.forward_range_m:
        return None
    half_width = bev.lateral_range_m / 2.0
    if x_m_lateral < -half_width or x_m_lateral > half_width:
        return None

    width_px = int(round(bev.lateral_range_m * bev.pixels_per_meter))
    height_px = int(round(bev.forward_range_m * bev.pixels_per_meter))
    x_px = int(round((x_m_lateral + half_width) * bev.pixels_per_meter))
    y_px = int(round(height_px - z_m_forward * bev.pixels_per_meter))
    x_px = int(np.clip(x_px, 0, width_px - 1))
    y_px = int(np.clip(y_px, 0, height_px - 1))
    return x_px, y_px
