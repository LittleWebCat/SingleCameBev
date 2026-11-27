#!/usr/bin/env python3
"""Validate and preserve external camera/LiDAR calibration as offline candidates.

This importer never resizes intrinsics, changes sensor settings, publishes TF,
or treats fitting residuals as independent calibration validation.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml


class ImportError(ValueError):
    """A calibration document is inconsistent or has unsupported semantics."""


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ImportError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ImportError(f"{name} must be at least {minimum}")
    return float(value)


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ImportError(f"{name} must be an integer of at least {minimum}")
    return value


def _array(value, shape, name):
    # JSON booleans and numeric strings must not silently become numbers.
    def visit(item):
        if isinstance(item, list):
            for child in item:
                visit(child)
        else:
            _number(item, name)

    visit(value)
    try:
        result = np.array(value, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ImportError(f"{name} must have shape {shape}") from error
    if result.shape != shape:
        raise ImportError(f"{name} must have shape {shape}, got {result.shape}")
    return result


def validate_camera(document):
    if not isinstance(document, dict) or document.get("model") != "pinhole":
        raise ImportError("Camera model must be pinhole")
    size = document.get("image_size")
    if not isinstance(size, list) or len(size) != 2:
        raise ImportError("image_size must be [width, height]")
    width, height = [_integer(value, "image_size") for value in size]
    matrix = _array(document.get("K"), (3, 3), "K")
    distortion = _array(document.get("D"), (5,), "D")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ImportError("K focal lengths must be positive")
    if not np.allclose(matrix[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]],
                       [0, 0, 0, 0, 1], rtol=0, atol=1e-12):
        raise ImportError("K must use standard pinhole form with zero skew")
    if not 0 <= matrix[0, 2] < width or not 0 <= matrix[1, 2] < height:
        raise ImportError("K principal point must lie inside the native image")
    if "rms_px" in document:
        _number(document["rms_px"], "camera rms_px", minimum=0)
    images, per_view = document.get("images"), document.get("per_view_rms_px")
    if images is not None and (not isinstance(images, list) or
                              any(not isinstance(name, str) or not name for name in images)):
        raise ImportError("camera images must contain nonempty file paths")
    if per_view is not None:
        if not isinstance(per_view, list):
            raise ImportError("per_view_rms_px must be a list")
        for value in per_view:
            _number(value, "per_view_rms_px", minimum=0)
        if images is not None and len(images) != len(per_view):
            raise ImportError("camera images and per_view_rms_px counts differ")
    return {"image_size": (width, height), "K": matrix, "D": distortion}


def _quaternion_xyzw(rotation):
    """Convert a verified rotation, including rotations close to 180 degrees."""
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [(rotation[2, 1] - rotation[1, 2]) / scale,
                      (rotation[0, 2] - rotation[2, 0]) / scale,
                      (rotation[1, 0] - rotation[0, 1]) / scale, scale / 4.0]
    else:
        axis = int(np.argmax(np.diag(rotation)))
        j, k = (axis + 1) % 3, (axis + 2) % 3
        scale = math.sqrt(1.0 + rotation[axis, axis] - rotation[j, j] - rotation[k, k]) * 2.0
        quaternion = [0.0] * 4
        quaternion[axis] = scale / 4.0
        quaternion[j] = (rotation[j, axis] + rotation[axis, j]) / scale
        quaternion[k] = (rotation[k, axis] + rotation[axis, k]) / scale
        quaternion[3] = (rotation[k, j] - rotation[j, k]) / scale
    result = np.array(quaternion, dtype=np.float64)
    result /= np.linalg.norm(result)
    if result[3] < 0:
        result = -result
    return result.tolist()


def validate_documents(intrinsics, extrinsics):
    camera = validate_camera(intrinsics)
    if not isinstance(extrinsics, dict):
        raise ImportError("Extrinsics must be a JSON object")
    if extrinsics.get("direction") != "lidar_to_camera" or extrinsics.get("units") != "m":
        raise ImportError("Extrinsics require direction lidar_to_camera and units m")
    embedded = validate_camera(extrinsics.get("camera"))
    if (camera["image_size"] != embedded["image_size"] or
            not np.array_equal(camera["K"], embedded["K"]) or
            not np.array_equal(camera["D"], embedded["D"])):
        raise ImportError("Embedded camera K/D/image_size differs from the intrinsics file")
    rotation = _array(extrinsics.get("R"), (3, 3), "R")
    translation = _array(extrinsics.get("t"), (3,), "t")
    determinant = float(np.linalg.det(rotation))
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    if orthogonality_error > 1e-8 or abs(determinant - 1) > 1e-8:
        raise ImportError("R must be a proper orthonormal rotation (SO(3))")
    forward = _array(extrinsics.get("T_camera_from_lidar"), (4, 4), "T_camera_from_lidar")
    inverse = _array(extrinsics.get("T_lidar_from_camera"), (4, 4), "T_lidar_from_camera")
    expected_forward = np.eye(4)
    expected_forward[:3, :3], expected_forward[:3, 3] = rotation, translation
    expected_inverse = np.eye(4)
    expected_inverse[:3, :3], expected_inverse[:3, 3] = rotation.T, -rotation.T @ translation
    if not np.allclose(forward, expected_forward, rtol=0, atol=1e-8):
        raise ImportError("T_camera_from_lidar disagrees with R/t or homogeneous form")
    if not np.allclose(inverse, expected_inverse, rtol=0, atol=1e-8):
        raise ImportError("T_lidar_from_camera is not the inverse of T_camera_from_lidar")
    count = _integer(extrinsics.get("point_count"), "point_count", minimum=4)
    errors = _array(extrinsics.get("point_errors_px"), (count,), "point_errors_px")
    if (errors < 0).any():
        raise ImportError("point_errors_px cannot be negative")
    inliers = extrinsics.get("inlier_indices")
    if not isinstance(inliers, list) or len(inliers) < 4:
        raise ImportError("inlier_indices must include at least four point indices")
    for index in inliers:
        _integer(index, "inlier index", minimum=0)
    if len(set(inliers)) != len(inliers) or max(inliers) >= count:
        raise ImportError("inlier_indices must be unique and within point_count")
    all_rms = float(np.sqrt(np.mean(errors ** 2)))
    inlier_rms = float(np.sqrt(np.mean(errors[inliers] ** 2)))
    for field, computed in (("all_points_rms_px", all_rms), ("inlier_rms_px", inlier_rms)):
        reported = _number(extrinsics.get(field), field, minimum=0)
        if not math.isclose(reported, computed, rel_tol=1e-8, abs_tol=1e-8):
            raise ImportError(f"{field} disagrees with point_errors_px: {reported} versus {computed}")
    threshold = _number(extrinsics.get("ransac_threshold_px"), "ransac_threshold_px", minimum=0)
    if threshold == 0:
        raise ImportError("ransac_threshold_px must be positive")
    pair_files = extrinsics.get("pair_files")
    if not isinstance(pair_files, list) or not pair_files or any(
            not isinstance(name, str) or not name for name in pair_files):
        raise ImportError("pair_files must contain nonempty file paths")
    duplicates = {name: occurrences for name, occurrences in Counter(pair_files).items() if occurrences > 1}
    diagnostics = {
        "rotation_determinant": determinant,
        "rotation_orthogonality_max_abs_error": orthogonality_error,
        "inverse_product_max_abs_error": float(np.max(np.abs(forward @ inverse - np.eye(4)))),
        "point_count": count, "inlier_count": len(inliers),
        "inlier_fraction": len(inliers) / count,
        "all_points_rms_px_recomputed": all_rms,
        "inlier_rms_px_recomputed": inlier_rms,
        "maximum_point_error_px": float(errors.max()),
        "ransac_threshold_px": threshold,
        "maximum_final_inlier_error_px": float(errors[inliers].max()),
        "pair_file_count": len(pair_files), "unique_pair_file_count": len(set(pair_files)),
        "duplicate_pair_files": duplicates,
        "residual_interpretation": "Fitting residuals from supplied correspondences; not held-out validation.",
    }
    return camera, forward, inverse, diagnostics


def _ros_matrix(array):
    return {"rows": array.shape[0], "cols": array.shape[1], "data": array.reshape(-1).tolist()}


def candidate_documents(intrinsics, extrinsics, lidar_frame="UNKNOWN"):
    if not isinstance(lidar_frame, str) or not lidar_frame.strip() or any(char.isspace() for char in lidar_frame):
        raise ImportError("lidar_frame must be a nonempty frame label without whitespace")
    camera, forward, inverse, diagnostics = validate_documents(intrinsics, extrinsics)
    width, height = camera["image_size"]
    projection = np.column_stack([camera["K"], np.zeros(3)])
    camera_yaml = {
        "image_width": width, "image_height": height, "camera_name": "front_camera",
        "distortion_model": "plumb_bob", "camera_matrix": _ros_matrix(camera["K"]),
        "distortion_coefficients": _ros_matrix(camera["D"].reshape(1, 5)),
        "rectification_matrix": _ros_matrix(np.eye(3)), "projection_matrix": _ros_matrix(projection),
        "calibrated": False, "calibration_status": "external_candidate_unvalidated",
    }
    transform_yaml = {
        "schema_version": 1, "calibrated": False,
        "calibration_status": "external_candidate_unvalidated", "units": "m",
        "direction": "lidar_to_camera", "point_mapping": "p_camera = R * p_lidar + t",
        "frame_mapping_status": "source_unconfirmed" if lidar_frame == "UNKNOWN" else "source_user_supplied",
        "target_frame_mapping_note": "camera_front_optical_frame is the project integration label; the source JSON does not name ROS frames.",
        "transforms": {"T_camera_from_lidar": {
            "calibrated": False, "source_frame": lidar_frame,
            "target_frame": "camera_front_optical_frame", "translation_m": forward[:3, 3].tolist(),
            "quaternion_xyzw": _quaternion_xyzw(forward[:3, :3]), "matrix": forward.tolist(),
        }},
        "T_lidar_from_camera": inverse.tolist(),
    }
    warnings = [
        "No independent reprojection validation or correspondence geometry was supplied.",
        "Intrinsic camera and LiDAR extrinsic errors are training residuals, not held-out accuracy measurements.",
        "Native image size must match capture; intrinsics have not been resized, cropped or applied.",
        "Source JSON does not identify LiDAR coordinate frame or sensor serial numbers.",
        "Rigid sensor mounting and matching sensor identities must be confirmed separately.",
    ]
    if lidar_frame == "UNKNOWN":
        warnings.append("LiDAR source frame remains UNKNOWN; do not publish this transform as ROS TF.")
    if diagnostics["duplicate_pair_files"]:
        warnings.append("Repeated pair_files entries may weight a pose more than once; data was preserved without refitting.")
    if diagnostics["maximum_final_inlier_error_px"] > diagnostics["ransac_threshold_px"]:
        warnings.append("Some final residuals for recorded RANSAC inliers exceed the RANSAC threshold after refinement.")
    report = {
        "schema_version": 1, "calibration_status": "external_candidate_unvalidated",
        "validation_passed": False, "fusion_ready": False, "hardware_changed": False,
        "intrinsics_rescaled": False, "tf_published": False,
        "native_image_width_px": width, "native_image_height_px": height,
        "lidar_source_frame": lidar_frame, "target_frame": "camera_front_optical_frame",
        "same_sensor_current_mounting_user_confirmed": None,
        "camera_reported_training_rms_px": intrinsics.get("rms_px"),
        "camera_reported_view_count": len(intrinsics.get("images", [])),
        "board": intrinsics.get("board"), "extrinsics": diagnostics, "warnings": warnings,
    }
    return camera_yaml, transform_yaml, report


def _json_object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ImportError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(path):
    raw = Path(path).read_bytes()
    try:
        parsed = json.loads(raw, object_pairs_hook=_json_object_pairs,
                            parse_constant=lambda value: (_ for _ in ()).throw(ImportError(f"Non-finite JSON value: {value}")))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ImportError(f"Invalid JSON in {path}: {error}") from error
    return raw, parsed


def import_calibration(intrinsics_path, extrinsics_path, output_dir, lidar_frame="UNKNOWN"):
    intrinsics_raw, intrinsics = _read_json(intrinsics_path)
    extrinsics_raw, extrinsics = _read_json(extrinsics_path)
    camera_yaml, transform_yaml, report = candidate_documents(intrinsics, extrinsics, lidar_frame)
    output = Path(output_dir)
    if output.exists():
        raise ImportError(f"Output directory already exists; refusing to overwrite: {output}")
    width, height = report["native_image_width_px"], report["native_image_height_px"]
    camera_name = f"front_camera_{width}x{height}.candidate.yaml"
    report.update({
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {
            "intrinsics": {"path": str(Path(intrinsics_path).resolve()), "sha256": hashlib.sha256(intrinsics_raw).hexdigest(), "preserved_copy": "front.json"},
            "extrinsics": {"path": str(Path(extrinsics_path).resolve()), "sha256": hashlib.sha256(extrinsics_raw).hexdigest(), "preserved_copy": "front_external.json"},
        },
        "candidate_camera_file": camera_name, "candidate_transform_file": "T_candidate.yaml",
    })
    # Serialize before creating anything, so validation failures leave no outputs.
    documents = {
        "front.json": intrinsics_raw, "front_external.json": extrinsics_raw,
        camera_name: yaml.safe_dump(camera_yaml, sort_keys=False).encode(),
        "T_candidate.yaml": yaml.safe_dump(transform_yaml, sort_keys=False).encode(),
        "import_report.json": (json.dumps(report, indent=2, allow_nan=False) + "\n").encode(),
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, contents in documents.items():
        with (output / name).open("xb") as stream:
            stream.write(contents)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path, required=True)
    parser.add_argument("--extrinsics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing directories are refused")
    parser.add_argument("--lidar-frame", default="UNKNOWN", help="Source coordinate frame only when confirmed for this calibration")
    arguments = parser.parse_args(argv)
    try:
        report = import_calibration(arguments.intrinsics, arguments.extrinsics,
                                    arguments.output_dir, arguments.lidar_frame)
    except (ImportError, OSError) as error:
        print(f"Import failed: {error}", file=sys.stderr)
        return 1
    print(f"Imported candidates into {arguments.output_dir}; calibration remains unvalidated.")
    print(f"Native size: {report['native_image_width_px']}x{report['native_image_height_px']}; "
          f"{report['extrinsics']['inlier_count']}/{report['extrinsics']['point_count']} inliers; "
          f"all-point training RMS {report['extrinsics']['all_points_rms_px_recomputed']:.3f} px.")
    for warning in report["warnings"]:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
