#!/usr/bin/env python3
"""Fit a candidate monocular camera model from saved checkerboard images.

This is offline: no ROS, hardware capture, clock alignment or sensor extrinsics.
Held-out board poses are estimated independently with fixed training intrinsics.
Their reprojection residual measures model consistency, not intrinsic accuracy.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
import numbers
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml


class CalibrationError(ValueError):
    """Invalid observations or an unusable numerical camera model."""


@dataclass(frozen=True)
class BoardSpec:
    columns: int
    rows: int
    square_size_mm: float

    def __post_init__(self):
        for name in ("columns", "rows"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 2:
                raise CalibrationError(f"{name} must be an integer of at least 2 inner corners")
        if (isinstance(self.square_size_mm, bool)
                or not isinstance(self.square_size_mm, numbers.Real)
                or not math.isfinite(self.square_size_mm) or self.square_size_mm <= 0):
            raise CalibrationError("square_size_mm must be finite and positive")

    @property
    def pattern(self):
        return (int(self.columns), int(self.rows))

    def object_points(self):
        points = np.zeros((self.columns * self.rows, 3), dtype=np.float32)
        points[:, :2] = np.mgrid[:self.columns, :self.rows].T.reshape(-1, 2)
        points *= self.square_size_mm / 1000.0
        return points

    def as_dict(self):
        return {"pattern": "checkerboard", "inner_corners_columns": self.columns,
                "inner_corners_rows": self.rows, "squares_columns": self.columns + 1,
                "squares_rows": self.rows + 1, "square_size_mm": float(self.square_size_mm)}


@dataclass(frozen=True)
class View:
    name: str
    image_size: tuple
    corners: np.ndarray
    sha256: str | None = None
    detector: str | None = None


def validate_views(views, board):
    if not views:
        raise CalibrationError("No checkerboard observations supplied")
    size = views[0].image_size
    if (len(size) != 2 or any(isinstance(x, bool) or not isinstance(x, numbers.Integral)
                              or x <= 0 for x in size)):
        raise CalibrationError("Image size must contain positive integer width and height")
    names, digests, observations = set(), set(), []
    for view in views:
        if view.image_size != size:
            raise CalibrationError(f"Mixed image dimensions: {view.name}")
        if view.name in names or (view.sha256 is not None and view.sha256 in digests):
            raise CalibrationError(f"Duplicate image/file: {view.name}")
        names.add(view.name)
        if view.sha256 is not None:
            digests.add(view.sha256)
        points = np.asarray(view.corners)
        count = board.columns * board.rows
        if points.shape not in ((count, 2), (count, 1, 2)):
            raise CalibrationError(f"Expected {count} checkerboard corners in {view.name}")
        points = np.asarray(points, dtype=np.float64).reshape(count, 2)
        if not np.isfinite(points).all():
            raise CalibrationError(f"Non-finite corners in {view.name}")
        if (points < 0).any() or (points >= np.asarray(size)).any():
            raise CalibrationError(f"Corners fall outside image in {view.name}")
        hull = cv2.convexHull(points.astype(np.float32))
        if cv2.contourArea(hull) <= 1.0:
            raise CalibrationError(f"Degenerate checkerboard geometry in {view.name}")
        for previous in observations:
            if np.allclose(points, previous, atol=1e-5, rtol=0) or np.allclose(
                    points, previous[::-1], atol=1e-5, rtol=0):
                raise CalibrationError(f"Duplicate checkerboard observation: {view.name}")
        observations.append(points)
    return tuple(int(x) for x in size)


def split_views(views, validation_fraction=0.2, min_views=15):
    """Stable, evenly distributed held-out indices, selected before fitting."""
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise CalibrationError("validation_fraction must be between zero and one")
    if isinstance(min_views, bool) or not isinstance(min_views, numbers.Integral) or min_views < 6:
        raise CalibrationError("min_views must be at least 6 training observations")
    ordered = sorted(views, key=lambda view: view.name)
    count = max(3, math.ceil(len(ordered) * validation_fraction))
    if len(ordered) - count < min_views:
        raise CalibrationError(f"Need at least {min_views} training views and {count} held-out views; "
                               f"only {len(ordered)} views available")
    indices = {int((i + 0.5) * len(ordered) / count) for i in range(count)}
    training = [view for i, view in enumerate(ordered) if i not in indices]
    validation = [view for i, view in enumerate(ordered) if i in indices]
    return training, validation


def _homography_constraints(views, board, image_size):
    """Zhang constraint conditioning, normalized for the image dimensions."""
    planar = board.object_points()[:, :2].astype(np.float64)
    planar /= board.square_size_mm / 1000.0
    rows = []

    def product(h, i, j):
        return np.array([h[0, i] * h[0, j], h[0, i] * h[1, j] + h[1, i] * h[0, j],
                         h[1, i] * h[1, j], h[2, i] * h[0, j] + h[0, i] * h[2, j],
                         h[2, i] * h[1, j] + h[1, i] * h[2, j], h[2, i] * h[2, j]])

    for view in views:
        normalized = np.asarray(view.corners).reshape(-1, 2) / np.asarray(image_size)
        h, _ = cv2.findHomography(planar, normalized, method=0)
        if h is None or not np.isfinite(h).all():
            raise CalibrationError(f"Cannot estimate checkerboard homography: {view.name}")
        for row in (product(h, 0, 1), product(h, 0, 0) - product(h, 1, 1)):
            norm = np.linalg.norm(row)
            if norm <= np.finfo(float).eps:
                raise CalibrationError(f"Degenerate checkerboard homography: {view.name}")
            rows.append(row / norm)
    singular = np.linalg.svd(rows, compute_uv=False)
    ratio = float(singular[4] / singular[0])
    if ratio < 1e-8:
        raise CalibrationError("Training views do not constrain intrinsics; tilt the board in different directions")
    return {"singular_values": singular.tolist(), "fifth_to_first_singular_value_ratio": ratio,
            "note": "Conditioning diagnostic from planar homographies; distorted/noisy corners affect this metric."}


def validate_camera_model(camera_matrix, distortion, image_size):
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if camera_matrix.shape != (3, 3) or distortion.size != 5:
        raise CalibrationError("Expected a 3x3 pinhole matrix and five plumb_bob coefficients")
    if not np.isfinite(camera_matrix).all() or not np.isfinite(distortion).all():
        raise CalibrationError("Camera fit produced non-finite parameters")
    if camera_matrix[0, 0] <= 0 or camera_matrix[1, 1] <= 0:
        raise CalibrationError("Camera fit produced nonpositive focal lengths")
    if not np.allclose(camera_matrix[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1]):
        raise CalibrationError("Camera fit is not a standard ROS pinhole matrix")
    if not 0 <= camera_matrix[0, 2] < image_size[0] or not 0 <= camera_matrix[1, 2] < image_size[1]:
        raise CalibrationError("Fitted principal point lies outside the image; inspect dataset conditioning")


def fit_camera(views, board):
    image_size = validate_views(views, board)
    if len(views) < 6:
        raise CalibrationError("At least six training observations are required")
    conditioning = _homography_constraints(views, board, image_size)
    object_points = [board.object_points() for _ in views]
    image_points = [np.asarray(view.corners, dtype=np.float32).reshape(-1, 1, 2) for view in views]
    try:
        values = cv2.calibrateCameraExtended(
            object_points, image_points, image_size, None, None, flags=0,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-10))
    except cv2.error as error:
        raise CalibrationError(f"OpenCV camera fit failed: {error}") from error
    rms, matrix, distortion, rotations, translations, intrinsic_stddev, extrinsic_stddev, per_view = values
    validate_camera_model(matrix, distortion, image_size)
    if not math.isfinite(rms) or not np.isfinite(per_view).all():
        raise CalibrationError("Camera fit produced invalid reprojection errors")
    return {"camera_matrix": matrix, "distortion": distortion.reshape(5), "rms_px": float(rms),
            "rvecs": rotations, "tvecs": translations, "per_view_rms_px": per_view.reshape(-1),
            "intrinsic_stddev": intrinsic_stddev.reshape(-1),
            "extrinsic_stddev": extrinsic_stddev.reshape(-1), "image_size": image_size,
            "homography_conditioning": conditioning}


def evaluate_held_out(views, board, camera_matrix, distortion):
    validate_views(views, board)
    validate_camera_model(camera_matrix, distortion, views[0].image_size)
    object_points = board.object_points()
    records = []
    sum_squared, count = 0.0, 0
    for view in views:
        corners = np.asarray(view.corners, dtype=np.float64).reshape(-1, 2)
        try:
            success, rotation, translation = cv2.solvePnP(
                object_points, corners, camera_matrix, distortion, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error as error:
            raise CalibrationError(f"Held-out pose estimation failed: {view.name}: {error}") from error
        if not success or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
            raise CalibrationError(f"Held-out pose estimation failed: {view.name}")
        rotation_matrix = cv2.Rodrigues(rotation)[0]
        depths = (object_points @ rotation_matrix.T + translation.reshape(1, 3))[:, 2]
        if (depths <= 0).any():
            raise CalibrationError(f"Held-out pose places board behind camera: {view.name}")
        projected = cv2.projectPoints(object_points, rotation, translation, camera_matrix, distortion)[0]
        squared = np.sum((projected.reshape(-1, 2) - corners) ** 2, axis=1)
        if not np.isfinite(squared).all():
            raise CalibrationError(f"Invalid held-out projection: {view.name}")
        rms = float(np.sqrt(squared.mean()))
        sum_squared += float(squared.sum())
        count += len(squared)
        records.append({"image": view.name, "rms_px": rms,
                        "max_corner_error_px": float(np.sqrt(squared.max())),
                        "board_pose_rvec": rotation.reshape(3).tolist(),
                        "board_pose_translation_m": translation.reshape(3).tolist()})
    return {"rms_px": math.sqrt(sum_squared / count),
            "max_view_rms_px": max(record["rms_px"] for record in records), "views": records,
            "method": "Each held-out board pose is fitted independently; intrinsics/distortion stay fixed.",
            "limitation": "Pose fitting can absorb intrinsic errors. This is a consistency check, not independent camera accuracy validation."}


def coverage_metrics(views, board):
    size = validate_views(views, board)
    all_points, centers, areas, spacings, per_view = [], [], [], [], []
    for view in views:
        points = np.asarray(view.corners, dtype=np.float64).reshape(-1, 2)
        normalized = points / np.asarray(size)
        area = float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))) / np.prod(size))
        grid = points.reshape(board.rows, board.columns, 2)
        lengths = np.concatenate([np.linalg.norm(np.diff(grid, axis=1), axis=2).reshape(-1),
                                  np.linalg.norm(np.diff(grid, axis=0), axis=2).reshape(-1)])
        spacing = float(np.median(lengths))
        all_points.append(normalized)
        centers.append(normalized.mean(axis=0))
        areas.append(area)
        spacings.append(spacing)
        per_view.append({"image": view.name, "inner_corner_hull_area_fraction": area,
                         "median_adjacent_corner_spacing_px": spacing,
                         "centroid_normalized": normalized.mean(axis=0).tolist()})
    points = np.concatenate(all_points)
    bins = {(min(3, int(x * 4)), min(2, int(y * 3))) for x, y in points}
    return {"views": per_view, "inner_corner_hull_area_fraction": {
                "minimum": min(areas), "median": float(np.median(areas)), "maximum": max(areas)},
            "median_adjacent_corner_spacing_px": float(np.median(spacings)),
            "centroid_span_normalized": np.ptp(centers, axis=0).tolist(),
            "all_corner_hull_area_fraction": float(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))),
            "occupied_corner_bins_4_by_3": len(bins), "corner_bin_coverage_fraction": len(bins) / 12.0,
            "note": "Coverage summarizes inner corners, not full printed board size or ground-truth accuracy."}


def _pose_diversity(fit):
    normals = np.array([cv2.Rodrigues(rvec)[0][:, 2] for rvec in fit["rvecs"]])
    # A checkerboard's two physical face normals are equivalent for this metric.
    angles = np.rad2deg(np.arccos(np.clip(np.abs(normals @ normals.T), 0, 1)))
    tilts = np.rad2deg(np.arccos(np.clip(np.abs(normals[:, 2]), 0, 1)))
    return {"normal_angle_span_deg": float(angles.max()), "median_tilt_deg": float(np.median(tilts)),
            "maximum_tilt_deg": float(tilts.max()),
            "note": "Pose diversity uses the candidate model and is not an independent measurement."}


def radial_domain_diagnostic(camera_matrix, distortion, image_size):
    """Find exact radial-polynomial extrema on a conservative nominal image domain.

    D uses OpenCV order k1,k2,p1,p2,k3. A positive radial scale and derivative
    are necessary for an orientation-preserving radial mapping. No arbitrary
    threshold on coefficient magnitude is used. This does not certify the full
    tangential-distortion mapping or actual coverage beyond observed corners.
    """
    validate_camera_model(camera_matrix, distortion, image_size)
    matrix = np.asarray(camera_matrix)
    k1, k2, _, _, k3 = np.asarray(distortion).reshape(5)
    corners = np.array([[0, 0], [image_size[0] - 1, 0], [0, image_size[1] - 1],
                        [image_size[0] - 1, image_size[1] - 1]], dtype=float)
    normalized = (corners - matrix[:2, 2]) / matrix[[0, 1], [0, 1]]
    maximum_squared_radius = float(np.sum(normalized ** 2, axis=1).max())

    def extrema(coefficients):
        polynomial = np.polynomial.Polynomial(coefficients)
        candidates = [0.0, maximum_squared_radius]
        for root in polynomial.deriv().roots():
            if abs(root.imag) < 1e-10 and 0 < root.real < maximum_squared_radius:
                candidates.append(float(root.real))
        values = [float(polynomial(x)) for x in candidates]
        index = int(np.argmin(values))
        zero_radii = [float(math.sqrt(root.real)) for root in polynomial.roots()
                      if abs(root.imag) < 1e-10 and 0 < root.real <= maximum_squared_radius]
        return {"minimum": values[index], "minimum_at_radius": math.sqrt(candidates[index]),
                "zero_radii_within_domain": sorted(zero_radii)}

    scale = extrema([1, k1, k2, k3])
    derivative = extrema([1, 3 * k1, 5 * k2, 7 * k3])
    reasons = []
    if scale["minimum"] <= 0:
        reasons.append("Radial distortion scale becomes nonpositive inside the nominal image domain.")
    if derivative["minimum"] <= 0:
        reasons.append("Radial mapping derivative becomes nonpositive inside the nominal image domain.")
    return {"radial_mapping_valid_in_nominal_domain": not reasons,
            "nominal_domain_maximum_radius": math.sqrt(maximum_squared_radius),
            "radial_scale": scale, "radial_mapping_derivative": derivative,
            "rejection_reasons": reasons,
            "domain_note": "Maximum corner radius from K-normalized image pixels; conservative radial disk enclosing the operating image rectangle.",
            "method": "Analytic extrema of scale 1+k1*r^2+k2*r^4+k3*r^6 and derivative 1+3*k1*r^2+5*k2*r^4+7*k3*r^6.",
            "limitation": "A passing radial check does not establish intrinsic accuracy or certify tangential distortion invertibility."}


def subset_stability(training, board, reference):
    """Refit three overlapping subsets using training data only; no held-out leakage."""
    records = []
    matrix = reference["camera_matrix"]
    for omitted_residue in range(3):
        subset = [view for index, view in enumerate(training) if index % 3 != omitted_residue]
        record = {"excluded_index_modulo_3": omitted_residue, "training_images": [v.name for v in subset]}
        try:
            fitted = fit_camera(subset, board)
            record.update({"result": "fitted", "rms_px": fitted["rms_px"],
                           "camera_matrix": fitted["camera_matrix"].tolist(),
                           "distortion": fitted["distortion"].tolist(),
                           "focal_relative_change": ((fitted["camera_matrix"][[0, 1], [0, 1]]
                               - matrix[[0, 1], [0, 1]]) / matrix[[0, 1], [0, 1]]).tolist(),
                           "principal_point_change_px": (fitted["camera_matrix"][:2, 2] - matrix[:2, 2]).tolist()})
        except CalibrationError as error:
            record.update(result="failed", error=str(error))
        records.append(record)
    areas = [cv2.contourArea(cv2.convexHull(np.asarray(view.corners, dtype=np.float32).reshape(-1, 2)))
             for view in training]
    dominant_index = int(np.argmax(areas))
    dominant = {"excluded_image": training[dominant_index].name,
                "excluded_inner_corner_hull_area_fraction": float(areas[dominant_index] / np.prod(reference["image_size"])),
                "training_images": [view.name for i, view in enumerate(training) if i != dominant_index]}
    try:
        fitted = fit_camera([view for i, view in enumerate(training) if i != dominant_index], board)
        dominant.update({"result": "fitted", "rms_px": fitted["rms_px"],
                         "camera_matrix": fitted["camera_matrix"].tolist(), "distortion": fitted["distortion"].tolist(),
                         "focal_relative_change": ((fitted["camera_matrix"][[0, 1], [0, 1]]
                             - matrix[[0, 1], [0, 1]]) / matrix[[0, 1], [0, 1]]).tolist(),
                         "principal_point_change_px": (fitted["camera_matrix"][:2, 2] - matrix[:2, 2]).tolist()})
    except CalibrationError as error:
        dominant.update(result="failed", error=str(error))
    fitted_records = [record for record in records if record["result"] == "fitted"]
    return {"subsets": records, "successful_subsets": len(fitted_records),
            "maximum_absolute_focal_relative_change": max(
                (max(abs(x) for x in record["focal_relative_change"]) for record in fitted_records), default=None),
            "maximum_absolute_principal_point_change_px": max(
                (max(abs(x) for x in record["principal_point_change_px"]) for record in fitted_records), default=None),
            "largest_board_view_removed": dominant,
            "note": "Sensitivity diagnostic using training data only; held-out images are never included in any refit."}


def camera_info_yaml(fit, camera_name="front_camera_candidate"):
    matrix = fit["camera_matrix"]
    validate_camera_model(matrix, fit["distortion"], fit["image_size"])
    radial = radial_domain_diagnostic(matrix, fit["distortion"], fit["image_size"])
    if radial["rejection_reasons"]:
        raise CalibrationError("Refusing loadable CameraInfo for invalid radial model: " + "; ".join(radial["rejection_reasons"]))
    projection = np.zeros((3, 4), dtype=np.float64)
    projection[:, :3] = matrix

    def data(values):
        return {"rows": values.shape[0], "cols": values.shape[1], "data": values.reshape(-1).tolist()}

    return {"image_width": fit["image_size"][0], "image_height": fit["image_size"][1],
            "camera_name": camera_name, "camera_matrix": data(matrix), "distortion_model": "plumb_bob",
            "distortion_coefficients": data(fit["distortion"].reshape(1, 5)),
            "rectification_matrix": data(np.eye(3)), "projection_matrix": data(projection)}


def calibrate_views(training, validation, board, *, include_stability=True, max_validation_rms_px=None):
    validate_views(training + validation, board)
    if len(validation) < 3:
        raise CalibrationError("At least three independent held-out images are required")
    fit = fit_camera(training, board)
    radial = radial_domain_diagnostic(fit["camera_matrix"], fit["distortion"], fit["image_size"])
    rejection_reasons = list(radial["rejection_reasons"])
    try:
        held_out = evaluate_held_out(validation, board, fit["camera_matrix"], fit["distortion"])
        held_out["evaluation_status"] = "evaluated"
    except CalibrationError as error:
        # Retain fitted parameters even when an unusable model cannot project held-out poses.
        rejection_reasons.append(f"Held-out projection evaluation failed: {error}")
        held_out = {"evaluation_status": "failed", "error": str(error), "rms_px": None,
                    "max_view_rms_px": None, "views": []}
    if max_validation_rms_px is not None:
        if not math.isfinite(max_validation_rms_px) or max_validation_rms_px <= 0:
            raise CalibrationError("max_validation_rms_px must be finite and positive")
        held_out["explicit_rms_limit_px"] = float(max_validation_rms_px)
        held_out["residual_screen_passed"] = (held_out["max_view_rms_px"] is not None
                                                and held_out["max_view_rms_px"] <= max_validation_rms_px)
    else:
        held_out["explicit_rms_limit_px"] = None
        held_out["residual_screen_passed"] = None
    coverage = coverage_metrics(training, board)
    poses = _pose_diversity(fit)
    stability = subset_stability(training, board, fit) if include_stability else None
    warnings = []
    if coverage["inner_corner_hull_area_fraction"]["median"] < 0.02:
        warnings.append("Most boards occupy little image area; capture closer views with clearly resolved corners.")
    if coverage["median_adjacent_corner_spacing_px"] < 15:
        warnings.append("Typical adjacent corners are less than 15 pixels apart; fine corner errors strongly affect this fit.")
    if coverage["corner_bin_coverage_fraction"] < 0.75:
        warnings.append("Inner corners do not cover most image regions; include large boards near image edges.")
    if poses["normal_angle_span_deg"] < 15:
        warnings.append("Candidate board orientations have limited tilt diversity; capture different horizontal and vertical tilts.")
    if fit["homography_conditioning"]["fifth_to_first_singular_value_ratio"] < 1e-3:
        warnings.append("Planar homography constraints are poorly conditioned.")
    stddev = fit["intrinsic_stddev"]
    if not np.isfinite(stddev).all():
        warnings.append("OpenCV intrinsic uncertainty estimates contain non-finite values.")
    elif np.any(stddev[:2] / fit["camera_matrix"][[0, 1], [0, 1]] > 0.05):
        warnings.append("Estimated focal-length standard deviations exceed 5 percent of the fitted focal lengths.")
    if stability is not None:
        if stability["successful_subsets"] != 3:
            warnings.append("Some training-subset fits failed; candidate intrinsics are sensitive to view selection.")
        if ((stability["maximum_absolute_focal_relative_change"] or 0) > 0.05
                or (stability["maximum_absolute_principal_point_change_px"] or 0) > 10):
            warnings.append("Training subsets substantially change intrinsics; collect stronger pose/scale coverage.")
        dominant = stability["largest_board_view_removed"]
        if dominant["result"] == "failed" or max(abs(x) for x in dominant["focal_relative_change"]) > 0.05:
            warnings.append("Removing the largest board view destabilizes the fit; collect more close and tilted observations.")
    parameters = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tau_x", "tau_y"]
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "opencv_version": cv2.__version__, "result": "candidate_intrinsics_fitted" if not rejection_reasons else "rejected_camera_model",
              "calibration_status": "candidate_unvalidated" if not rejection_reasons else "rejected_camera_model",
              "validation_passed": False, "model_rejection_reasons": rejection_reasons,
              "candidate_yaml_permitted": not rejection_reasons,
              "fusion_ready": False, "board": board.as_dict(),
              "board_geometry_assumption": "The target must be flat and rigid, with accurately equal square spacing; image residuals cannot independently verify the physical board geometry.",
              "image_width": fit["image_size"][0], "image_height": fit["image_size"][1],
              "split": {"training_images": [view.name for view in training],
                        "held_out_images": [view.name for view in validation],
                        "training_count": len(training), "held_out_count": len(validation),
                        "held_out_used_for_intrinsic_fit": False},
              "camera_matrix": fit["camera_matrix"].tolist(), "distortion_model": "plumb_bob",
              "distortion_coefficients": fit["distortion"].tolist(),
              "fit": {"rms_px": fit["rms_px"], "flags": 0,
                      "per_view": [{"image": view.name, "rms_px": float(rms)}
                                   for view, rms in zip(training, fit["per_view_rms_px"])],
                      "intrinsic_standard_deviation": {name: float(value) if math.isfinite(value) else None
                                                       for name, value in zip(parameters, stddev)},
                      "extrinsic_standard_deviation": [float(x) if math.isfinite(x) else None
                                                       for x in fit["extrinsic_stddev"]],
                      "extrinsic_standard_deviation_note": "Six uncertainty values per training board pose, in training-image order; these are board poses, not measured sensor extrinsics.",
                      "uncertainty_note": "OpenCV local fit uncertainty, not a guarantee of accuracy; unfitted parameter standard deviations are zero."},
              "held_out": held_out, "training_coverage": coverage, "training_pose_diversity": poses,
              "homography_conditioning": fit["homography_conditioning"], "training_subset_stability": stability,
              "radial_domain_diagnostic": radial,
              "capture_quality_warnings": warnings,
              "quality_policy_note": "Coverage, tilt, spacing and stability warning limits are heuristics, not universal acceptance thresholds.",
              "rectification_note": "Identity R and P=[K|0] describe an unrectified monocular camera; they are not measured sensor extrinsics.",
              "remaining_validation": ["Independent camera accuracy/undistortion validation and operating-resolution/lens-setting checks.",
                                       "Camera-LiDAR and vehicle extrinsics are unmeasured.",
                                       "Exposure timing and cross-sensor clock alignment are unvalidated."],
              "application_status": "not_applied_to_camera_bridge"}
    return fit, report


def _detect_corners(image, board):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, board.pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            return corners, "findChessboardCornersSB"
    found, corners = cv2.findChessboardCorners(
        gray, board.pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4))
        return corners, "findChessboardCorners+cornerSubPix"
    return None, None


def load_image_views(source, board):
    source_path = Path(source).expanduser()
    manifest_path = (source_path / "session.json") if source_path.is_dir() else source_path
    manifest = None
    if manifest_path.is_file() and manifest_path.name == "session.json":
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            raise CalibrationError("Capture session manifest must be a mapping")
        measured_board = manifest.get("board", {})
        if (measured_board.get("pattern") != "checkerboard"
                or measured_board.get("inner_corners_columns") != board.columns
                or measured_board.get("inner_corners_rows") != board.rows
                or measured_board.get("squares_columns") != board.columns + 1
                or measured_board.get("squares_rows") != board.rows + 1
                or measured_board.get("square_size_mm") != board.square_size_mm):
            raise CalibrationError("Explicit board dimensions/spacing disagree with session.json")
        root = manifest_path.resolve().parent
        frames = manifest.get("saved_frames")
        if not isinstance(frames, list) or not frames:
            raise CalibrationError("Capture manifest has no saved_frames")
        paths = []
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("image_file"), str):
                raise CalibrationError("Invalid captured image_file in session.json")
            path = (root / frame["image_file"]).resolve()
            if not path.is_relative_to(root):
                raise CalibrationError("Captured image_file escapes the dataset directory")
            paths.append(path)
    elif source_path.is_dir():
        extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        paths = [path.resolve() for path in source_path.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
    else:
        paths = [Path(path).resolve() for path in glob.glob(str(source_path), recursive=True)]
    if not paths:
        raise CalibrationError(f"No saved image files found: {source}")
    if len(set(paths)) != len(paths):
        raise CalibrationError("Repeated image files in dataset")
    views, rejected, fingerprints = [], [], set()
    dimensions = None
    for path in sorted(paths):
        if not path.is_file():
            raise CalibrationError(f"Saved image file is missing: {path}")
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        if fingerprint in fingerprints:
            raise CalibrationError(f"Duplicate image contents: {path}")
        fingerprints.add(fingerprint)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise CalibrationError(f"Cannot read image: {path}")
        size = (image.shape[1], image.shape[0])
        if dimensions is not None and size != dimensions:
            raise CalibrationError(f"Mixed image dimensions: {path}")
        dimensions = size
        if manifest and size != (manifest.get("camera_width_px"), manifest.get("camera_height_px")):
            raise CalibrationError(f"Image dimensions disagree with capture manifest: {path}")
        corners, detector = _detect_corners(image, board)
        if corners is None:
            rejected.append({"image": str(path), "reason": "checkerboard_not_detected"})
        else:
            views.append(View(str(path), size, corners, fingerprint, detector))
    if views:
        validate_views(views, board)
    return views, {"source": str(source), "source_image_count": len(paths),
                   "detected_view_count": len(views), "rejected_images": rejected,
                   "detected_images": [{"image": view.name, "sha256": view.sha256,
                                        "detector": view.detector} for view in views],
                   "corners_source": "re-detected from saved raw image pixels; saved manifest corners not used",
                   "capture_manifest": str(manifest_path.resolve()) if manifest else None,
                   "source_topic": manifest.get("source_topic") if manifest else None,
                   "frame_id": manifest.get("frame_id") if manifest else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Capture dataset, image directory or quoted file glob")
    parser.add_argument("--columns", required=True, type=int, help="Number of INNER corners per row")
    parser.add_argument("--rows", required=True, type=int, help="Number of INNER corners per column")
    parser.add_argument("--square-size-mm", required=True, type=float, help="Measured/user-provided square edge length")
    parser.add_argument("--output", required=True, type=Path, help="New candidate ROS CameraInfo YAML")
    parser.add_argument("--report", type=Path, help="New JSON report (default: output stem + .report.json)")
    parser.add_argument("--validation-images", help="Separate held-out dataset/directory/glob")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--min-views", type=int, default=15, help="Minimum number of training views (default: 15)")
    parser.add_argument("--max-validation-rms-px", type=float, help="Optional per-held-out-view RMS screening limit; does not validate intrinsics")
    parser.add_argument("--skip-subset-stability", action="store_true", help="Skip the three training-only subset refits")
    parser.add_argument("--force", action="store_true", help="Explicitly replace existing candidate/report files")
    args = parser.parse_args(argv)
    # Keep the requested leaf path: rejected forced refits may unlink this file
    # or symlink, and must never delete a symlink's underlying target instead.
    output = args.output.expanduser().absolute()
    report_path = args.report.expanduser().resolve() if args.report else output.with_suffix(".report.json")
    try:
        if output.resolve() == report_path.resolve():
            raise CalibrationError("Candidate YAML and JSON report need different paths")
        for path in (output, report_path):
            if not args.force and (path.exists() or path.is_symlink()):
                raise CalibrationError(f"Refusing to overwrite existing output: {path}; use --force explicitly")
        board = BoardSpec(args.columns, args.rows, args.square_size_mm)
        observations, source_report = load_image_views(args.images, board)
        if args.validation_images:
            training = sorted(observations, key=lambda view: view.name)
            validation, validation_source = load_image_views(args.validation_images, board)
            validation = sorted(validation, key=lambda view: view.name)
            if args.min_views < 6 or len(training) < args.min_views:
                raise CalibrationError(f"Need at least {max(6, args.min_views)} training views")
            source_report["validation_source"] = validation_source
        else:
            training, validation = split_views(observations, args.validation_fraction, args.min_views)
        fitted, report = calibrate_views(training, validation, board,
                                        include_stability=not args.skip_subset_stability,
                                        max_validation_rms_px=args.max_validation_rms_px)
        report["input_dataset"] = source_report
        report["split"]["method"] = ("explicit separate validation source" if args.validation_images else
                                      "lexicographic filenames; evenly distributed deterministic held-out indices before fitting")
        report["candidate_yaml"] = str(output) if report["candidate_yaml_permitted"] else None
        report["requested_candidate_yaml"] = str(output)
        report["report_json"] = str(report_path)
        candidate = yaml.safe_dump(camera_info_yaml(fitted), sort_keys=False) if report["candidate_yaml_permitted"] else None
        if candidate is not None:
            candidate = ("# CANDIDATE INTRINSICS: unvalidated and not applied to the camera bridge.\n"
                         "# Identity rectification is monocular CameraInfo bookkeeping, not sensor extrinsics.\n" + candidate)
        for path in ([output, report_path] if candidate is not None else [report_path]):
            path.parent.mkdir(parents=True, exist_ok=True)
        # Render both outputs before writing; strict JSON forbids silent NaN values.
        report_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        with report_path.open("w" if args.force else "x", encoding="utf-8") as stream:
            stream.write(report_text)
        if candidate is None:
            # Retain numerical fit evidence in JSON while withholding a loadable YAML.
            # --force explicitly replaces this output. Remove an older candidate
            # only after saving rejection evidence, so it cannot appear current.
            if args.force and (output.is_file() or output.is_symlink()):
                output.unlink()
            print(f"Rejected camera model; JSON report saved: {report_path}", file=sys.stderr)
            for reason in report["model_rejection_reasons"]:
                print(reason, file=sys.stderr)
            return 2
        with output.open("w" if args.force else "x", encoding="utf-8") as stream:
            stream.write(candidate)
        print(f"Candidate saved: {output}\nReport saved: {report_path}\n"
              f"Training {len(training)}, held-out {len(validation)}; "
              f"fit RMS {fitted['rms_px']:.3f} px; held-out RMS {report['held_out']['rms_px']:.3f} px.\n"
              "Status: candidate_unvalidated; not applied to camera bridge.")
        for warning in report["capture_quality_warnings"]:
            print(f"Capture quality: {warning}")
    except (CalibrationError, OSError, cv2.error, json.JSONDecodeError, TypeError) as error:
        print(f"Calibration failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
