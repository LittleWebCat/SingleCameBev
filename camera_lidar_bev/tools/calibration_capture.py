"""Save checkerboard observations without estimating camera calibration."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import numbers
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np
import yaml


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class BoardConfig:
    inner_corners_columns: int
    inner_corners_rows: int
    square_size_mm: float

    def __post_init__(self):
        _positive_integer(self.inner_corners_columns, "inner_corners_columns")
        _positive_integer(self.inner_corners_rows, "inner_corners_rows")
        size = self.square_size_mm
        if (isinstance(size, bool) or not isinstance(size, numbers.Real)
                or not math.isfinite(size) or size <= 0):
            raise ValueError("square_size_mm must be finite and positive")

    @property
    def pattern(self):
        return (int(self.inner_corners_columns), int(self.inner_corners_rows))

    @classmethod
    def from_yaml(cls, path):
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict) or data.get("pattern", "checkerboard") != "checkerboard":
            raise ValueError("Expected a checkerboard configuration mapping")
        columns = _positive_integer(data.get("squares_columns"), "squares_columns")
        rows = _positive_integer(data.get("squares_rows"), "squares_rows")
        inner_columns = _positive_integer(data.get("inner_corners_columns", columns - 1),
                                          "inner_corners_columns")
        inner_rows = _positive_integer(data.get("inner_corners_rows", rows - 1), "inner_corners_rows")
        if (inner_columns, inner_rows) != (columns - 1, rows - 1):
            raise ValueError("Inner corner dimensions must be one less than square dimensions")
        return cls(inner_columns, inner_rows, data.get("square_size_mm"))

    def as_dict(self):
        return {
            "pattern": "checkerboard",
            "squares_columns": self.pattern[0] + 1,
            "squares_rows": self.pattern[1] + 1,
            "inner_corners_columns": self.pattern[0],
            "inner_corners_rows": self.pattern[1],
            "square_size_mm": float(self.square_size_mm),
            "square_size_source": "user_provided",
        }


@dataclass(frozen=True)
class CaptureResult:
    accepted: bool
    reason: str
    count: int
    path: Path | None = None


class CaptureSession:
    """A new, append-only dataset with pose diversity feedback, not calibration validation."""

    def __init__(self, output_dir, board, topic, frame_id=None, duplicate_pose_threshold=0.02,
                 min_board_span_px=0.0):
        if not isinstance(board, BoardConfig):
            raise TypeError("board must be a BoardConfig")
        if not topic or not isinstance(topic, str):
            raise ValueError("topic must be a nonempty string")
        if frame_id is not None and (not isinstance(frame_id, str) or not frame_id):
            raise ValueError("frame_id must be a nonempty string or None")
        if (isinstance(duplicate_pose_threshold, bool)
                or not math.isfinite(duplicate_pose_threshold) or duplicate_pose_threshold <= 0):
            raise ValueError("duplicate_pose_threshold must be finite and positive")
        if (isinstance(min_board_span_px, bool) or not isinstance(min_board_span_px, numbers.Real)
                or not math.isfinite(min_board_span_px) or min_board_span_px < 0):
            raise ValueError("min_board_span_px must be finite and nonnegative")
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() or self.output_dir.is_symlink():
            raise FileExistsError(f"Refusing to reuse existing dataset path: {self.output_dir}")
        self.board = board
        self._frame_id = frame_id
        self._dimensions = None
        self._poses = []
        self._stamps = set()
        self._threshold = float(duplicate_pose_threshold)
        self._min_board_span_px = float(min_board_span_px)
        self._manifest = {
            "schema_version": 1,
            "created_at_utc": self._utc_now(),
            "calibration_status": "unmeasured",
            "board": board.as_dict(),
            "source_topic": topic,
            "frame_id": frame_id,
            "camera_width_px": None,
            "camera_height_px": None,
            "duplicate_pose_threshold_normalized": self._threshold,
            "min_board_span_px": self._min_board_span_px,
            "board_size_note": "Minimum corner span is a capture guide, not intrinsic validation.",
            "pose_diversity_note": "Corner motion is a coverage heuristic, not intrinsic validation.",
            "saved_frames": [],
        }
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / "images").mkdir()
        self._write_manifest(self._manifest)

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    @property
    def count(self):
        return len(self._manifest["saved_frames"])

    def _write_manifest(self, manifest):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.output_dir,
                                             prefix=".session-", suffix=".json", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(manifest, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.output_dir / "session.json")
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def capture(self, frame, corners, header_metadata, receipt_age_s):
        """Capture raw BGR pixels; header has frame_id and stamp_sec/stamp_nanosec."""
        def reject(reason):
            return CaptureResult(False, reason, self.count)

        if (isinstance(receipt_age_s, bool) or not isinstance(receipt_age_s, numbers.Real)
                or not math.isfinite(receipt_age_s) or not 0 <= receipt_age_s <= 1.0):
            return reject("Frame is stale or its receipt age is invalid")
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) <= 0):
            return reject("Frame must be a nonempty CV_8UC3 BGR image")
        height, width = frame.shape[:2]
        if self._dimensions is not None and (width, height) != self._dimensions:
            return reject("Image dimensions changed during this session")
        try:
            points = np.asarray(corners, dtype=np.float64)
        except (TypeError, ValueError):
            return reject("Corner coordinates are invalid")
        expected_count = self.board.pattern[0] * self.board.pattern[1]
        if points.shape not in ((expected_count, 2), (expected_count, 1, 2)):
            return reject(f"Expected all {expected_count} checkerboard corners")
        points = points.reshape(expected_count, 2)
        if not np.isfinite(points).all():
            return reject("Corner coordinates must be finite")
        if ((points < 0).any() or (points[:, 0] >= width).any() or (points[:, 1] >= height).any()):
            return reject("Checkerboard corners must be inside the image")
        span_px = np.ptp(points, axis=0)
        if float(span_px.max()) < self._min_board_span_px:
            return reject(f"Move closer: inner-corner span {span_px.max():.1f} px "
                          f"is below {self._min_board_span_px:g} px")
        if not isinstance(header_metadata, dict):
            return reject("Image header metadata is missing")
        current_frame_id = header_metadata.get("frame_id")
        if not isinstance(current_frame_id, str) or not current_frame_id:
            return reject("Image frame_id is missing")
        if self._frame_id is not None and current_frame_id != self._frame_id:
            return reject("Image frame_id changed or differs from the expected frame")
        stamp_data = header_metadata.get("stamp", {})
        if not isinstance(stamp_data, dict):
            return reject("Image timestamp is invalid")
        sec = header_metadata.get("stamp_sec", stamp_data.get("sec"))
        nanos = header_metadata.get("stamp_nanosec", stamp_data.get("nanosec"))
        if (isinstance(sec, bool) or not isinstance(sec, numbers.Integral) or sec < 0
                or isinstance(nanos, bool) or not isinstance(nanos, numbers.Integral)
                or not 0 <= nanos < 1_000_000_000 or sec * 1_000_000_000 + nanos <= 0):
            return reject("Image timestamp must be nonzero and valid")
        stamp = (int(sec), int(nanos))
        if stamp in self._stamps:
            return reject("Image timestamp was already saved")
        header = dict(header_metadata)
        header.update(frame_id=current_frame_id, stamp_sec=stamp[0], stamp_nanosec=stamp[1])
        if "stamp" in header:
            header["stamp"] = {"sec": stamp[0], "nanosec": stamp[1]}
        try:
            json.dumps(header, allow_nan=False)
        except (TypeError, ValueError):
            return reject("Image header metadata must be finite and JSON serializable")
        normalized = points / np.array([width, height])
        for previous in self._poses:
            # Checkerboard detectors may reverse the correspondence order by 180 degrees.
            distance = min(float(np.sqrt(np.mean(np.sum((normalized - candidate) ** 2, axis=1))))
                           for candidate in (previous, previous[::-1]))
            if distance < self._threshold:
                return reject("Board pose is too similar to a saved view; move or tilt the board")
        pixels = frame.copy()
        gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
        hull = cv2.convexHull(points.astype(np.float32))
        metadata = {
            "image_file": f"images/frame_{self.count + 1:04d}.png",
            "saved_at_utc": self._utc_now(),
            "header": header,
            "receipt_age_s": float(receipt_age_s),
            "width_px": width, "height_px": height, "encoding": "bgr8",
            "corners_px": points.tolist(),
            "feedback": {
                "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                "inner_corner_hull_area_fraction": float(cv2.contourArea(hull) / (width * height)),
                "centroid_normalized": normalized.mean(axis=0).tolist(),
                "corner_span_normalized": np.ptp(normalized, axis=0).tolist(),
                "corner_span_px": span_px.tolist(),
            },
        }
        success, encoded = cv2.imencode(".png", pixels)
        if not success:
            return reject("Could not encode PNG image")
        image_path = self.output_dir / metadata["image_file"]
        with image_path.open("xb") as stream:
            stream.write(encoded.tobytes())
        updated = dict(self._manifest)
        updated.update(frame_id=current_frame_id, camera_width_px=width, camera_height_px=height)
        updated["saved_frames"] = self._manifest["saved_frames"] + [metadata]
        self._write_manifest(updated)
        self._manifest = updated
        self._frame_id = current_frame_id
        self._dimensions = (width, height)
        self._stamps.add(stamp)
        self._poses.append(normalized.copy())
        return CaptureResult(True, "Saved checkerboard observation", self.count, image_path)
