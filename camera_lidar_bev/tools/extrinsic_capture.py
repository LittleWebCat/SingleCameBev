"""ROS-free storage of stationary camera/LiDAR observations, without fitting extrinsics."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
import yaml

from calibration_capture import BoardConfig, CaptureResult


_POINT_TYPES = {1: ("i1", 1), 2: ("u1", 1), 3: ("i2", 2), 4: ("u2", 2),
                5: ("i4", 4), 6: ("u4", 4), 7: ("f4", 4), 8: ("f8", 8)}


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, minimum=0.0):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _header(metadata, name):
    if not isinstance(metadata, dict):
        raise ValueError(f"{name} header is missing")
    frame_id = metadata.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError(f"{name} frame_id is missing")
    nested = metadata.get("stamp", {})
    if not isinstance(nested, dict):
        raise ValueError(f"{name} timestamp is invalid")
    sec = _integer(metadata.get("stamp_sec", nested.get("sec")), f"{name} stamp_sec")
    nanos = _integer(metadata.get("stamp_nanosec", nested.get("nanosec")),
                     f"{name} stamp_nanosec")
    if nanos >= 1_000_000_000 or sec * 1_000_000_000 + nanos == 0:
        raise ValueError(f"{name} timestamp must be nonzero and valid")
    header = dict(metadata, frame_id=frame_id, stamp_sec=sec, stamp_nanosec=nanos)
    if "stamp" in header:
        header["stamp"] = {"sec": sec, "nanosec": nanos}
    json.dumps(header, allow_nan=False)
    return header, sec * 1_000_000_000 + nanos


@dataclass(frozen=True)
class PointCloudData:
    """Decoded scalar XYZ in original point order; invalid points remain present."""
    xyz: np.ndarray
    intensity: np.ndarray | None
    width: int
    height: int

    @property
    def organized_xyz(self):
        return self.xyz.reshape(self.height, self.width, 3)

    @property
    def valid_mask(self):
        return np.isfinite(self.xyz).all(axis=1) & np.any(self.xyz != 0, axis=1)


def decode_point_cloud(metadata, data):
    """Decode ROS PointCloud2 layout, including row padding and either byte order.

    ``metadata`` contains width, height, point_step, row_step, is_bigendian,
    is_dense and fields ({name, offset, datatype, count}). XYZ fields must be
    scalar FLOAT32 (7) or FLOAT64 (8). The untouched payload is saved separately.
    """
    if not isinstance(metadata, dict):
        raise ValueError("PointCloud2 metadata must be a mapping")
    width = _integer(metadata.get("width"), "cloud width", 1)
    height = _integer(metadata.get("height"), "cloud height", 1)
    point_step = _integer(metadata.get("point_step"), "point_step", 1)
    row_step = _integer(metadata.get("row_step"), "row_step", 1)
    if row_step < width * point_step:
        raise ValueError("row_step is smaller than the packed point row")
    if not isinstance(metadata.get("is_bigendian"), bool):
        raise ValueError("is_bigendian must be a boolean")
    if not isinstance(metadata.get("is_dense"), bool):
        raise ValueError("is_dense must be a boolean")
    try:
        payload = memoryview(data)
        if not payload.contiguous:
            raise ValueError("Cloud payload must be contiguous")
        payload = payload.cast("B")
    except (TypeError, ValueError) as error:
        raise ValueError("Cloud payload must be a contiguous bytes buffer") from error
    if payload.nbytes != row_step * height:
        raise ValueError("Cloud payload byte count does not match row_step * height")
    fields = metadata.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ValueError("PointCloud2 fields are missing")
    named_fields = {}
    for field in fields:
        if not isinstance(field, dict):
            raise ValueError("Each PointCloud2 field must be a mapping")
        name = field.get("name")
        if not isinstance(name, str) or not name or name in named_fields:
            raise ValueError("PointCloud2 field names must be nonempty and unique")
        offset = _integer(field.get("offset"), f"{name} offset")
        datatype = _integer(field.get("datatype"), f"{name} datatype", 1)
        count = _integer(field.get("count"), f"{name} count", 1)
        if datatype not in _POINT_TYPES:
            raise ValueError(f"Unknown PointCloud2 datatype for {name}")
        if offset + count * _POINT_TYPES[datatype][1] > point_step:
            raise ValueError(f"PointCloud2 field {name} extends beyond point_step")
        named_fields[name] = (offset, datatype, count)
    byte_order = ">" if metadata["is_bigendian"] else "<"

    def column(name, coordinate=False):
        if name not in named_fields:
            raise ValueError(f"PointCloud2 field {name} is missing")
        offset, datatype, count = named_fields[name]
        if count != 1 or (coordinate and datatype not in (7, 8)):
            raise ValueError(f"PointCloud2 field {name} must be a scalar "
                             + ("FLOAT32 or FLOAT64" if coordinate else "number"))
        values = np.ndarray((height, width), dtype=np.dtype(byte_order + _POINT_TYPES[datatype][0]),
                            buffer=payload, offset=offset, strides=(row_step, point_step))
        return values.reshape(-1).astype(np.float64)

    xyz = np.column_stack([column(axis, coordinate=True) for axis in ("x", "y", "z")])
    intensity = column("intensity") if "intensity" in named_fields else None
    return PointCloudData(xyz, intensity, width, height)


def _calibration(path):
    path = Path(path)
    content = path.read_bytes()
    document = yaml.safe_load(content)
    if not isinstance(document, dict):
        raise ValueError("Camera calibration YAML must be a mapping")
    expected = {"width": _integer(document.get("image_width"), "image_width", 1),
                "height": _integer(document.get("image_height"), "image_height", 1),
                "distortion_model": document.get("distortion_model")}
    d_lengths = {"plumb_bob": 5, "rational_polynomial": 8, "equidistant": 4}
    if expected["distortion_model"] not in d_lengths:
        raise ValueError("Unsupported camera distortion_model")
    matrices = {"k": ("camera_matrix", 3, 3),
                "d": ("distortion_coefficients", 1, d_lengths[expected["distortion_model"]]),
                "r": ("rectification_matrix", 3, 3),
                "p": ("projection_matrix", 3, 4)}
    for key, (name, rows, cols) in matrices.items():
        matrix = document.get(name)
        if (not isinstance(matrix, dict) or matrix.get("rows") != rows
                or matrix.get("cols") != cols):
            raise ValueError(f"Invalid {name} shape in camera calibration")
        values = np.asarray(matrix.get("data"), dtype=np.float64)
        if values.shape != (rows * cols,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {name} data in camera calibration")
        expected[key] = values.tolist()
    k = np.asarray(expected["k"]).reshape(3, 3)
    if (k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1])
            or not np.allclose([k[0, 1], k[1, 0]], [0, 0])):
        raise ValueError("Camera matrix is not a valid pinhole matrix")
    return path, content, expected


class ExtrinsicCaptureSession:
    """Append stationary observations to a new dataset; extrinsics stay unmeasured."""

    def __init__(self, output_dir, board, calibration_file,
                 image_topic="/camera/front/image_raw",
                 camera_info_topic="/camera/front/camera_info", cloud_topic="/points",
                 duplicate_pose_threshold=0.02, min_board_span_px=140.0):
        if not isinstance(board, BoardConfig):
            raise TypeError("board must be a BoardConfig")
        for name, topic in (("image_topic", image_topic), ("camera_info_topic", camera_info_topic),
                            ("cloud_topic", cloud_topic)):
            if not isinstance(topic, str) or not topic:
                raise ValueError(f"{name} must be a nonempty string")
        threshold = _number(duplicate_pose_threshold, "duplicate_pose_threshold")
        if threshold == 0:
            raise ValueError("duplicate_pose_threshold must be positive")
        minimum_span = _number(min_board_span_px, "min_board_span_px")
        calibration_path, calibration_bytes, expected = _calibration(calibration_file)
        self.output_dir = Path(output_dir)
        if self.output_dir.exists() or self.output_dir.is_symlink():
            raise FileExistsError(f"Refusing to reuse existing dataset path: {self.output_dir}")
        self.board = board
        self.expected_camera_info = expected
        self._threshold = threshold
        self._minimum_span = minimum_span
        self._poses = []
        self._image_stamps = set()
        self._cloud_stamps = set()
        self._image_frame = None
        self._cloud_frame = None
        self._manifest = {
            "schema_version": 1, "created_at_utc": self._utc_now(),
            "dataset_type": "stationary_camera_lidar_checkerboard_observations",
            "calibration_status": "unmeasured", "board": board.as_dict(),
            "source_topics": {"image": image_topic, "camera_info": camera_info_topic,
                              "cloud": cloud_topic},
            "camera_calibration": {"source_file": str(calibration_path.resolve()),
                                   "copied_file": "camera_calibration.yaml",
                                   "sha256": hashlib.sha256(calibration_bytes).hexdigest(),
                                   "qualification": "provisional_intrinsics",
                                   "camera_info_reference": expected},
            "pairing": {"maximum_absolute_header_difference_s": 0.15,
                        "maximum_receipt_age_s": 1.0,
                        "qualification": "header_pairing_heuristic_only",
                        "note": "Camera host receipt timestamps and LiDAR header timestamps "
                                "do not establish exposure synchronization. Keep the board stationary."},
            "min_board_span_px": minimum_span,
            "duplicate_pose_threshold_normalized": threshold,
            "geometry_note": "Corner size and motion are capture guides, not extrinsic validation.",
            "image_frame_id": None, "cloud_frame_id": None, "saved_pairs": [],
        }
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / "observations").mkdir()
        (self.output_dir / "camera_calibration.yaml").write_bytes(calibration_bytes)
        self._write_manifest(self._manifest)

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    @property
    def count(self):
        return len(self._manifest["saved_pairs"])

    @staticmethod
    def _write_json(path, document):
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

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

    def capture(self, frame, corners, image_header, camera_info_metadata, cloud_metadata,
                cloud_data, image_age_s, cloud_age_s, board_stationary_confirmed=False,
                board_pose=None, lidar_target_hint=None):
        """Save raw pixels and the full cloud after explicit stationary-board confirmation.

        CameraInfo uses header,width,height,distortion_model,k,d,r,p. Cloud metadata
        uses the layout accepted by decode_point_cloud plus header. An optional
        board_pose dict is provenance only and cannot establish sensor extrinsics.
        """
        def reject(reason):
            return CaptureResult(False, reason, self.count)

        if board_stationary_confirmed is not True:
            return reject("Hold the board still and explicitly confirm it is stationary before saving")
        try:
            if _number(image_age_s, "image receipt age") > 1.0:
                return reject("Image is stale")
            if _number(cloud_age_s, "cloud receipt age") > 1.0:
                return reject("LiDAR cloud is stale")
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.ndim != 3 or frame.shape[2] != 3):
                return reject("Frame must be a CV_8UC3 BGR image")
            height, width = frame.shape[:2]
            if (width, height) != (self.expected_camera_info["width"],
                                   self.expected_camera_info["height"]):
                return reject("Image dimensions do not match the referenced camera calibration")
            image_header, image_stamp = _header(image_header, "Image")
            if not isinstance(camera_info_metadata, dict):
                return reject("CameraInfo metadata is missing")
            info_header, info_stamp = _header(camera_info_metadata.get("header"), "CameraInfo")
            if image_stamp != info_stamp or image_header["frame_id"] != info_header["frame_id"]:
                return reject("Image and CameraInfo headers must match exactly")
            for key in ("width", "height"):
                _integer(camera_info_metadata.get(key), f"CameraInfo {key}", 1)
            for key, expected in self.expected_camera_info.items():
                actual = camera_info_metadata.get(key)
                if key in ("k", "d", "r", "p"):
                    values = np.asarray(actual, dtype=np.float64)
                    if (values.shape != (len(expected),) or not np.isfinite(values).all()
                            or not np.allclose(values, expected, rtol=1e-12, atol=1e-9)):
                        return reject(f"CameraInfo {key.upper()} differs from the referenced calibration")
                elif actual != expected or isinstance(actual, bool):
                    return reject(f"CameraInfo {key} differs from the referenced calibration")
            json.dumps(camera_info_metadata, allow_nan=False)
            if not isinstance(cloud_metadata, dict):
                return reject("PointCloud2 metadata is missing")
            cloud_header, cloud_stamp = _header(cloud_metadata.get("header"), "LiDAR")
            if abs(image_stamp - cloud_stamp) > 150_000_000:
                return reject("Image and LiDAR headers differ by more than the 150 ms pairing guide")
            if self._image_frame is not None and image_header["frame_id"] != self._image_frame:
                return reject("Image frame_id changed during the session")
            if self._cloud_frame is not None and cloud_header["frame_id"] != self._cloud_frame:
                return reject("LiDAR frame_id changed during the session")
            if image_stamp in self._image_stamps or cloud_stamp in self._cloud_stamps:
                return reject("Image or cloud timestamp was already saved")
            points = np.asarray(corners, dtype=np.float64)
            expected_count = self.board.pattern[0] * self.board.pattern[1]
            if points.shape not in ((expected_count, 2), (expected_count, 1, 2)):
                return reject(f"Expected all {expected_count} checkerboard corners")
            points = points.reshape(expected_count, 2)
            if not np.isfinite(points).all():
                return reject("Checkerboard corners must be finite")
            if ((points < 0).any() or (points[:, 0] >= width).any()
                    or (points[:, 1] >= height).any()):
                return reject("Checkerboard corners must be inside the image")
            span = np.ptp(points, axis=0)
            if span.max() < self._minimum_span:
                return reject(f"Move closer: inner-corner span {span.max():.1f} px "
                              f"is below {self._minimum_span:g} px")
            normalized = points / np.array([width, height])
            for previous in self._poses:
                distance = min(float(np.sqrt(np.mean(np.sum((normalized - pose) ** 2, axis=1))))
                               for pose in (previous, previous[::-1]))
                if distance < self._threshold:
                    return reject("Board pose is too similar to a saved view; move or tilt the board")
            decoded = decode_point_cloud(cloud_metadata, cloud_data)
            valid = decoded.valid_mask
            if not valid.any():
                return reject("LiDAR cloud contains no finite nonzero XYZ points")
            json.dumps(cloud_metadata, allow_nan=False)
            if lidar_target_hint is not None:
                if not isinstance(lidar_target_hint, dict):
                    return reject("LiDAR target hint must be a mapping")
                target = np.asarray(lidar_target_hint.get("xyz_m"), dtype=np.float64)
                if target.shape != (3,) or not np.isfinite(target).all() or not np.any(target):
                    return reject("LiDAR target hint XYZ must be three finite nonzero coordinates")
                target_range = _number(lidar_target_hint.get("range_m"), "target range")
                if target_range == 0 or not math.isclose(target_range, np.linalg.norm(target),
                                                        rel_tol=1e-6, abs_tol=1e-6):
                    return reject("LiDAR target range must match the positive XYZ range")
                for name, limit in (("azimuth_deg", 180), ("elevation_deg", 90)):
                    if name in lidar_target_hint:
                        angle = lidar_target_hint[name]
                        if (isinstance(angle, bool) or not isinstance(angle, numbers.Real)
                                or not math.isfinite(angle) or abs(angle) > limit):
                            return reject(f"LiDAR target hint {name} is invalid")
                target_azimuth = math.degrees(math.atan2(target[1], target[0]))
                target_elevation = math.degrees(math.atan2(target[2], np.linalg.norm(target[:2])))
                if ("azimuth_deg" in lidar_target_hint and abs(
                        (lidar_target_hint["azimuth_deg"] - target_azimuth + 180) % 360 - 180) > 1e-3):
                    return reject("LiDAR target hint azimuth does not match XYZ")
                if ("elevation_deg" in lidar_target_hint and abs(
                        lidar_target_hint["elevation_deg"] - target_elevation) > 1e-3):
                    return reject("LiDAR target hint elevation does not match XYZ")
                if "header" in lidar_target_hint:
                    hint_header, hint_stamp = _header(lidar_target_hint["header"], "LiDAR target")
                    if hint_stamp != cloud_stamp or hint_header["frame_id"] != cloud_header["frame_id"]:
                        return reject("LiDAR target hint header must match the saved cloud")
                if "point_index" in lidar_target_hint:
                    index = _integer(lidar_target_hint["point_index"], "target point_index")
                    if index >= decoded.xyz.shape[0] or not np.allclose(
                            decoded.xyz[index], target, rtol=1e-6, atol=1e-6):
                        return reject("LiDAR target hint point does not match its saved cloud index")
                json.dumps(lidar_target_hint, allow_nan=False)
                lidar_target_hint = dict(lidar_target_hint, source="user_selected_return",
                                         board_return_verified=False,
                                         qualification="user_target_hint_only")
            if board_pose is not None:
                if not isinstance(board_pose, dict):
                    return reject("Optional board pose must be a mapping")
                for key in ("rvec", "tvec"):
                    value = np.asarray(board_pose.get(key), dtype=np.float64)
                    if value.shape not in ((3,), (3, 1)) or not np.isfinite(value).all():
                        return reject(f"Optional board pose {key} must contain three finite values")
                json.dumps(board_pose, allow_nan=False)
        except (TypeError, ValueError, OverflowError) as error:
            return reject(str(error))

        success, encoded = cv2.imencode(".png", frame)
        if not success:
            return reject("Could not encode raw PNG image")
        pair_number = self.count + 1
        relative = Path("observations") / f"pose_{pair_number:04d}"
        final_dir = self.output_dir / relative
        if final_dir.exists() or final_dir.is_symlink():
            return reject("Observation path already exists; refusing to overwrite it")
        entry = {"pair_number": pair_number, "saved_at_utc": self._utc_now(),
                 "image_file": str(relative / "image.png"),
                 "point_cloud_file": str(relative / "cloud.bin"),
                 "point_cloud_metadata_file": str(relative / "cloud.json"),
                 "point_cloud_preview_file": str(relative / "preview.npz"),
                 "image_header": image_header, "camera_info": camera_info_metadata,
                 "cloud_header": cloud_header, "image_receipt_age_s": float(image_age_s),
                 "cloud_receipt_age_s": float(cloud_age_s),
                 "absolute_header_difference_s": abs(image_stamp - cloud_stamp) / 1e9,
                 "board_stationary_user_confirmed": True, "corners_px": points.tolist(),
                 "lidar_target_hint": lidar_target_hint,
                 "board_pose_camera_from_board": board_pose,
                 "board_pose_note": "Optional camera board pose is not camera/LiDAR extrinsics.",
                 "feedback": {"centroid_normalized": normalized.mean(axis=0).tolist(),
                              "corner_span_px": span.tolist(),
                              "total_cloud_points": int(decoded.xyz.shape[0]),
                              "finite_nonzero_cloud_points": int(valid.sum())}}
        indices = np.flatnonzero(valid)
        indices = indices[np.linspace(0, len(indices) - 1, min(4096, len(indices)), dtype=int)]
        preview = {"xyz": decoded.xyz[indices].astype(np.float32), "point_indices": indices}
        if decoded.intensity is not None:
            preview["intensity"] = decoded.intensity[indices]
        payload = memoryview(cloud_data).cast("B")
        cloud_document = dict(cloud_metadata)
        cloud_document.update(header=cloud_header, payload_file="cloud.bin",
                              payload_bytes=payload.nbytes,
                              payload_sha256=hashlib.sha256(payload).hexdigest(),
                              xyz_units="m", payload_preserved_without_filtering=True)
        updated = dict(self._manifest, image_frame_id=image_header["frame_id"],
                       cloud_frame_id=cloud_header["frame_id"],
                       saved_pairs=self._manifest["saved_pairs"] + [entry])
        staging_dir = Path(tempfile.mkdtemp(prefix=f".pose-{pair_number:04d}-", dir=self.output_dir))
        committed = False
        try:
            (staging_dir / "image.png").write_bytes(encoded.tobytes())
            with (staging_dir / "cloud.bin").open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            self._write_json(staging_dir / "cloud.json", cloud_document)
            self._write_json(staging_dir / "observation.json", entry)
            np.savez_compressed(staging_dir / "preview.npz", **preview)
            os.rename(staging_dir, final_dir)
            committed = True
            self._write_manifest(updated)
        except Exception:
            shutil.rmtree(final_dir if committed else staging_dir, ignore_errors=True)
            raise
        self._manifest = updated
        self._image_frame = image_header["frame_id"]
        self._cloud_frame = cloud_header["frame_id"]
        self._image_stamps.add(image_stamp)
        self._cloud_stamps.add(cloud_stamp)
        self._poses.append(normalized.copy())
        return CaptureResult(True, "Saved stationary camera/LiDAR observation", self.count,
                             final_dir / "image.png")
