"""ROS-free loading and rendering for an unvalidated intrinsic preview."""

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


def _positive_integer(document, name):
    value = document.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _matrix(document, name, rows, columns):
    value = document.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain rows, cols, and data")
    if (type(value.get("rows")) is not int or type(value.get("cols")) is not int
            or value["rows"] != rows or value["cols"] != columns):
        raise ValueError(f"{name} must have dimensions {rows}x{columns}")
    data = value.get("data")
    if not isinstance(data, list) or len(data) != rows * columns:
        raise ValueError(f"{name} must contain {rows * columns} coefficients")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in data):
        raise ValueError(f"{name} coefficients must be finite numbers")
    return np.asarray(data, dtype=np.float64).reshape(rows, columns)


@dataclass
class CandidatePreview:
    width: int
    height: int
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray

    def undistort(self, frame):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3):
            raise ValueError("Preview input must be a uint8 BGR image")
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Calibration is {self.width}x{self.height}, but image is "
                f"{frame.shape[1]}x{frame.shape[0]}; do not rescale the calibration implicitly")
        return cv2.remap(frame, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT)

    def comparison(self, frame):
        """Keep native image sizes and K, with a separate status band above."""
        corrected = self.undistort(frame)
        band_height = 36
        preview = np.zeros((self.height + band_height, 2 * self.width, 3), dtype=np.uint8)
        preview[band_height:, :self.width] = frame
        preview[band_height:, self.width:] = corrected
        for label, offset in (("RAW", 0), ("UNDISTORTED - CANDIDATE (unvalidated)", self.width)):
            cv2.putText(preview, label, (offset + 10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return preview


def load_calibration(path):
    """Read the original-size ROS plumb_bob model; do not use P to crop it."""
    try:
        document = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"Cannot read calibration {path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("Calibration YAML must contain a mapping")
    width = _positive_integer(document, "image_width")
    height = _positive_integer(document, "image_height")
    if document.get("distortion_model") != "plumb_bob":
        raise ValueError("Candidate preview requires distortion_model: plumb_bob")
    matrix = _matrix(document, "camera_matrix", 3, 3)
    distortion = _matrix(document, "distortion_coefficients", 1, 5)
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive")
    if (not np.allclose(matrix[2], [0.0, 0.0, 1.0], rtol=0, atol=1e-9)
            or abs(matrix[0, 1]) > 1e-9 or abs(matrix[1, 0]) > 1e-9):
        raise ValueError("camera_matrix must use the pinhole form [fx,0,cx; 0,fy,cy; 0,0,1]")
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix, distortion, None, matrix, (width, height), cv2.CV_32FC1)
    return CandidatePreview(width, height, matrix, distortion, map_x, map_y)
