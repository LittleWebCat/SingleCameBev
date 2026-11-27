#!/usr/bin/python3
"""Anchor a saved local LiDAR–IMU map to recorded GPS without changing its scale.

Offline only: indexed rosbag2 GPS reads do not create ROS nodes or access sensors.
The original map/trajectory are never modified. GPS fixes are matched to source
pose timestamps; the fit is a proper planar rotation plus a constant translation.
GPS covariance and residuals are receiver/fit diagnostics, not surveyed accuracy.

Coordinate equations: ESA Navipedia, Ellipsoidal and Cartesian Coordinates
Conversion and Transformations between ECEF and ENU coordinates. WGS84 uses
a = 6378137 m and inverse flattening = 298.257223563.
"""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

# Small matrices do not benefit from a large pool of busy BLAS threads.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np


GPS_TOPIC = '/sensing/gnss/fix'
POSE_COLUMNS = ('stamp_sec', 'x_m', 'y_m', 'z_m', 'qx', 'qy', 'qz', 'qw')
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
REFERENCES = [
    'https://gssc.esa.int/navipedia/index.php/Ellipsoidal_and_Cartesian_Coordinates_Conversion',
    'https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates',
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def finite_array(value, shape_tail, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != shape_tail or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite N x {shape_tail} array')
    return array


def geodetic_to_ecef(geographic):
    """Latitude/longitude in degrees, ellipsoidal altitude in metres -> ECEF m."""
    geographic = finite_array(geographic, 3, 'geographic coordinates')
    if np.any(np.abs(geographic[:, 0]) > 90) or np.any(np.abs(geographic[:, 1]) > 180):
        raise ValueError('latitude or longitude is outside the WGS84 geographic range')
    latitude, longitude = np.radians(geographic[:, :2]).T
    altitude = geographic[:, 2]
    eccentricity_sq = WGS84_F * (2 - WGS84_F)
    radius = WGS84_A / np.sqrt(1 - eccentricity_sq * np.sin(latitude)**2)
    return np.column_stack(((radius + altitude) * np.cos(latitude) * np.cos(longitude),
                            (radius + altitude) * np.cos(latitude) * np.sin(longitude),
                            (radius * (1 - eccentricity_sq) + altitude) * np.sin(latitude)))


def geographic_to_enu(geographic, origin):
    """Use a right-handed tangent frame with +X east, +Y north and +Z up."""
    geographic = finite_array(geographic, 3, 'geographic coordinates')
    origin = finite_array(np.asarray(origin).reshape(1, 3), 3, 'ENU origin')[0]
    latitude, longitude = np.radians(origin[:2])
    sp, cp = math.sin(latitude), math.cos(latitude)
    sl, cl = math.sin(longitude), math.cos(longitude)
    rotation = np.array([[-sl, cl, 0], [-sp * cl, -sp * sl, cp],
                         [cp * cl, cp * sl, sp]])
    return (geodetic_to_ecef(geographic) - geodetic_to_ecef(origin[None, :])) @ rotation.T


def load_trajectory(path):
    """Reject corrupt/rewound poses, drop exact duplicate timestamps, normalize q."""
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(POSE_COLUMNS):
            raise ValueError(f'unsupported trajectory columns in {path}')
        rows = np.array([[float(row[key]) for key in POSE_COLUMNS] for row in reader],
                        dtype=np.float64)
    rows = finite_array(rows, 8, 'trajectory')
    if len(rows) < 20 or np.any(rows[:, 0] <= 0):
        raise ValueError('at least 20 timestamped poses are required')
    intervals = np.diff(rows[:, 0])
    if np.any(intervals < 0):
        raise ValueError('trajectory timestamps go backwards')
    keep = np.r_[True, intervals > 0]
    duplicates = int((~keep).sum())
    rows = rows[keep]
    norms = np.linalg.norm(rows[:, 4:8], axis=1)
    if np.any(norms < 1e-9):
        raise ValueError('trajectory contains a zero quaternion')
    rows[:, 4:8] /= norms[:, None]
    return rows, {'poses': len(rows), 'duplicate_pose_timestamps_removed': duplicates}


def usable_fix(message):
    """Return stamp/geographic/covariance for a valid fix, or None for no fix."""
    stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
    geographic = np.array([message.latitude, message.longitude, message.altitude], dtype=float)
    if (int(message.status.status) < 0 or stamp <= 0 or not math.isfinite(stamp)
            or not np.isfinite(geographic).all() or abs(geographic[0]) > 90
            or abs(geographic[1]) > 180):
        return None
    # Unknown covariance gets an explicit nominal weight, not invented accuracy.
    covariance = np.asarray(message.position_covariance, dtype=float).reshape(3, 3)
    known = int(message.position_covariance_type) > 0
    known = bool(known and np.isfinite(covariance).all()
                 and np.allclose(covariance, covariance.T, atol=1e-6)
                 and np.linalg.eigvalsh(covariance).min() >= -1e-6)
    sigma = float(np.sqrt(max(np.linalg.eigvalsh(covariance[:2, :2]).max(), 0))) if known else None
    variance = float(np.trace(covariance[:2, :2]) / 2) if known else 9.0
    return stamp, geographic, max(.25, min(100., variance)), sigma, known


def read_gps(bag, topic=GPS_TOPIC):
    """Read the indexed fix topic only; keep source headers rather than bag receipt."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import NavSatFix
    import yaml

    metadata_path = Path(bag) / 'metadata.yaml'
    metadata = yaml.safe_load(metadata_path.read_text())['rosbag2_bagfile_information']
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=metadata['storage_identifier']),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
    if topics.get(topic) != 'sensor_msgs/msg/NavSatFix':
        raise ValueError(f'the recording does not contain NavSatFix topic {topic}')
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    rows, sigmas, known_flags = [], [], []
    counts, modes = Counter(), Counter()
    previous_stamp = -math.inf
    while reader.has_next():
        _, payload, _ = reader.read_next()
        message = deserialize_message(payload, NavSatFix)
        counts['read'] += 1
        observation = usable_fix(message)
        if observation is None:
            counts['invalid_fix_removed'] += 1
            continue
        stamp, geographic, variance, sigma, known = observation
        if stamp <= previous_stamp:
            counts['duplicate_or_rewound_fix_removed'] += 1
            continue
        previous_stamp = stamp
        rows.append([stamp, *geographic, variance])
        sigmas.append(sigma if known else math.nan)
        known_flags.append(known)
        modes[str(int(message.status.status))] += 1
    if len(rows) < 20:
        raise ValueError('the recording has fewer than 20 usable unique GPS fixes')
    counts['usable'] = len(rows)
    counts['known_covariance'] = sum(known_flags)
    statistics = {'counts': dict(counts), 'navsat_status_counts': dict(modes),
                  'header_stamp_convention': 'recorded host ROS receipt; physical clock alignment unverified',
                  'topic': topic}
    return np.asarray(rows), np.asarray(sigmas), np.asarray(known_flags), statistics


def match_timestamps(trajectory, gps, maximum_pose_gap_s=.25):
    """Interpolate local XYZ at exact GPS headers without crossing long pose gaps."""
    trajectory = finite_array(trajectory, 8, 'trajectory')
    gps = finite_array(gps, 5, 'GPS fixes')
    if not math.isfinite(maximum_pose_gap_s) or maximum_pose_gap_s <= 0:
        raise ValueError('maximum pose gap must be finite and positive')
    if np.any(np.diff(trajectory[:, 0]) <= 0) or np.any(np.diff(gps[:, 0]) <= 0):
        raise ValueError('matching requires strictly increasing source timestamps')
    pose_stamps = trajectory[:, 0]
    inside = (gps[:, 0] >= pose_stamps[0]) & (gps[:, 0] <= pose_stamps[-1])
    candidates = np.flatnonzero(inside)
    stamps = gps[candidates, 0]
    upper = np.searchsorted(pose_stamps, stamps, side='left')
    upper = np.minimum(upper, len(pose_stamps) - 1)
    exact = pose_stamps[upper] == stamps
    lower = np.maximum(upper - 1, 0)
    gaps = pose_stamps[upper] - pose_stamps[lower]
    valid = exact | ((upper > 0) & (gaps <= maximum_pose_gap_s))
    candidates, upper, lower, exact, gaps = (item[valid] for item in
                                            (candidates, upper, lower, exact, gaps))
    if len(candidates) < 20:
        raise ValueError('fewer than 20 GPS fixes overlap adequately sampled map poses')
    fractions = np.zeros(len(candidates))
    interpolate = ~exact
    fractions[interpolate] = ((gps[candidates, 0] - pose_stamps[lower])[interpolate]
                              / gaps[interpolate])
    xyz = trajectory[upper, 1:4].copy()
    xyz[interpolate] = (trajectory[lower[interpolate], 1:4] * (1 - fractions[interpolate, None])
                        + trajectory[upper[interpolate], 1:4] * fractions[interpolate, None])
    statistics = {'matched_fixes': len(candidates),
                  'outside_trajectory_time_range': int((~inside).sum()),
                  'pose_gap_rejections': int((~valid).sum()),
                  'maximum_interpolation_bracket_s': float(gaps[~exact].max()) if (~exact).any() else 0.,
                  'maximum_allowed_pose_gap_s': maximum_pose_gap_s,
                  'timestamp_offset_applied_s': 0.,
                  'interpolation': 'linear local XYZ at recorded GPS header stamps'}
    return candidates, xyz, statistics


def weighted_planar_fit(local, enu, weights, proper=True):
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    local_center, enu_center = weights @ local, weights @ enu
    cross = (local - local_center).T @ (weights[:, None] * (enu - enu_center))
    left, _, right_t = np.linalg.svd(cross)
    correction = np.eye(2)
    if proper and np.linalg.det(right_t.T @ left.T) < 0:
        correction[-1, -1] = -1
    rotation = right_t.T @ correction @ left.T
    translation = enu_center - rotation @ local_center
    return rotation, translation


def fit_planar_rigid(local_xyz, gps_enu, variances):
    """Robust yaw + XYZ translation, with scale fixed to 1 and no roll/pitch fit."""
    local_xyz = finite_array(local_xyz, 3, 'matched local positions')
    gps_enu = finite_array(gps_enu, 3, 'matched GPS ENU positions')
    variances = np.asarray(variances, dtype=float)
    if (len(local_xyz) != len(gps_enu) or len(local_xyz) < 20
            or variances.shape != (len(local_xyz),) or not np.isfinite(variances).all()
            or np.any(variances <= 0)):
        raise ValueError('fit needs at least 20 equal-length finite positions and positive variances')
    local, enu = local_xyz[:, :2], gps_enu[:, :2]
    # Percentile spread avoids a single bad GPS fix satisfying the movement gate.
    spreads = [float(np.linalg.norm(np.percentile(x, 95, axis=0) - np.percentile(x, 5, axis=0)))
               for x in (local, enu)]
    if min(spreads) < 10:
        raise ValueError('less than 10 m of route extent: heading alignment is not observable reliably')
    base_weights = 1 / np.clip(variances, .25, 100)
    weights = base_weights.copy()
    for _ in range(30):
        rotation, translation = weighted_planar_fit(local, enu, weights)
        residuals = np.linalg.norm(local @ rotation.T + translation - enu, axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        huber = max(1., median + 1.5 * 1.4826 * mad)
        updated = base_weights * np.minimum(1., huber / np.maximum(residuals, 1e-9))
        if np.max(np.abs(updated / updated.sum() - weights / weights.sum())) < 1e-8:
            weights = updated
            break
        weights = updated
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    cutoff = max(3., median + 4 * 1.4826 * mad)
    inliers = residuals <= cutoff
    if inliers.sum() < max(20, len(inliers) // 2):
        raise ValueError('too few consistent matched positions for a GPS anchor')
    rotation, translation = weighted_planar_fit(local[inliers], enu[inliers], weights[inliers])
    reflected, reflected_translation = weighted_planar_fit(local[inliers], enu[inliers],
                                                            weights[inliers], proper=False)
    residuals = np.linalg.norm(local @ rotation.T + translation - enu, axis=1)
    eigenvalues = np.linalg.eigvalsh(np.cov(local[inliers].T))
    shape_ratio = float(eigenvalues[0] / max(eigenvalues[1], 1e-9))
    reflected_rmse = float(np.sqrt(np.mean(np.sum((local[inliers] @ reflected.T
                                    + reflected_translation - enu[inliers])**2, axis=1))))
    proper_rmse = float(np.sqrt(np.mean(residuals[inliers]**2)))
    if (np.linalg.det(reflected) < 0 and shape_ratio > .005
            and proper_rmse > max(3., 2 * reflected_rmse)):
        raise ValueError('matched routes favor a reflection; check source coordinate conventions')
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    # Only a constant vertical offset: GPS noise cannot reshape the LIO geometry.
    vertical_offset = float(np.median(gps_enu[inliers, 2] - local_xyz[inliers, 2]))
    transform = np.eye(4)
    transform[:2, :2], transform[:3, 3] = rotation, [*translation, vertical_offset]
    transformed = local_xyz @ transform[:3, :3].T + transform[:3, 3]
    vertical_residuals = transformed[:, 2] - gps_enu[:, 2]
    metrics = {'matched_positions': len(local), 'inliers': int(inliers.sum()),
               'outliers': int((~inliers).sum()), 'outlier_cutoff_m': cutoff,
               'horizontal_residual_rmse_m': proper_rmse,
               'horizontal_residual_median_m': float(np.median(residuals[inliers])),
               'horizontal_residual_p95_m': float(np.percentile(residuals[inliers], 95)),
               'horizontal_residual_max_inlier_m': float(residuals[inliers].max()),
               'horizontal_residual_rmse_all_m': float(np.sqrt(np.mean(residuals**2))),
               'vertical_residual_median_absolute_m': float(np.median(np.abs(vertical_residuals[inliers]))),
               'local_route_extent_m': spreads[0], 'gps_route_extent_m': spreads[1],
               'local_route_planar_shape_eigenvalue_ratio': shape_ratio,
               'proper_rotation_determinant': float(np.linalg.det(rotation)),
               'scale': 1., 'roll_pitch_fitted': False,
               'method': 'covariance-weighted proper planar Procrustes, Huber IRLS and residual rejection',
               'residuals_are_accuracy_truth': False}
    return transform, inliers, residuals, metrics


def transform_trajectory(trajectory, transform):
    """Apply the same rigid transform to XYZ and orientation, retaining stamp values."""
    output = np.asarray(trajectory, dtype=float).copy()
    output[:, 1:4] = output[:, 1:4] @ transform[:3, :3].T + transform[:3, 3]
    yaw = math.atan2(transform[1, 0], transform[0, 0])
    sine, cosine = math.sin(yaw / 2), math.cos(yaw / 2)
    x, y, z, w = output[:, 4:8].copy().T
    # q_enu_from_map * q_map_from_sensor; order is essential.
    output[:, 4:8] = np.column_stack((cosine * x - sine * y, cosine * y + sine * x,
                                     cosine * z + sine * w, cosine * w - sine * z))
    return output


def write_csv(path, columns, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(['' if isinstance(value, float) and math.isnan(value)
                             else format(float(value), '.17g') for value in row])


def transform_pcd(source, destination, transform, chunk_points=65536):
    """Transform binary float32 XYZ/intensity in bounded chunks; preserve intensity."""
    if not isinstance(chunk_points, int) or chunk_points < 1:
        raise ValueError('PCD chunk size must be positive')
    source_digest, output_digest = hashlib.sha256(), hashlib.sha256()
    with Path(source).open('rb') as incoming:
        header, lines = {}, []
        while True:
            line = incoming.readline()
            if not line or sum(map(len, lines)) > 65536:
                raise ValueError('missing or oversized PCD header')
            source_digest.update(line)
            lines.append(line)
            words = line.decode('ascii').split()
            if words and not words[0].startswith('#'):
                header[words[0]] = words[1:]
            if words and words[0] == 'DATA':
                break
        if (header.get('FIELDS') != ['x', 'y', 'z', 'intensity']
                or header.get('SIZE') != ['4'] * 4 or header.get('TYPE') != ['F'] * 4
                or header.get('COUNT') != ['1'] * 4 or header.get('DATA') != ['binary']):
            raise ValueError('expected saved binary float32 XYZ/intensity PCD')
        count = int(header['POINTS'][0])
        if count < 1 or int(header['WIDTH'][0]) * int(header['HEIGHT'][0]) != count:
            raise ValueError('invalid PCD dimensions')
        if Path(source).stat().st_size - incoming.tell() != count * 16:
            raise ValueError('PCD payload length differs from the declared point count')
        if header.get('VIEWPOINT') != ['0', '0', '0', '1', '0', '0', '0']:
            raise ValueError('expected identity local-map PCD viewpoint')
        yaw = math.atan2(transform[1, 0], transform[0, 0])
        viewpoint = [*transform[:3, 3], math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
        with Path(destination).open('xb') as outgoing:
            for line in lines:
                if line.startswith(b'VIEWPOINT '):
                    line = ('VIEWPOINT ' + ' '.join(format(x, '.17g') for x in viewpoint) + '\n').encode()
                outgoing.write(line)
                output_digest.update(line)
            for start in range(0, count, chunk_points):
                payload = incoming.read(min(chunk_points, count - start) * 16)
                source_digest.update(payload)
                points = np.frombuffer(payload, dtype='<f4').reshape(-1, 4).copy()
                if not np.isfinite(points[:, :3]).all():
                    raise ValueError('saved PCD contains nonfinite geometry')
                points[:, :3] = points[:, :3].astype(np.float64) @ transform[:3, :3].T + transform[:3, 3]
                payload = points.tobytes()
                outgoing.write(payload)
                output_digest.update(payload)
    return {'points': count, 'source_sha256': source_digest.hexdigest(),
            'aligned_sha256': output_digest.hexdigest(), 'chunk_points': chunk_points}


def make_plot(path, gps_stamps, gps_enu, aligned_xyz, inliers, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, (route, residual) = plt.subplots(1, 2, figsize=(14, 6),
                                            gridspec_kw={'width_ratios': [1.5, 1]})
    route.plot(gps_enu[:, 0], gps_enu[:, 1], color='#3984c6', linewidth=1.5, label='Recorded GPS')
    route.plot(aligned_xyz[:, 0], aligned_xyz[:, 1], color='#ee823d', linewidth=1.2,
               label='Aligned LiDAR–IMU trajectory')
    if (~inliers).any():
        route.scatter(gps_enu[~inliers, 0], gps_enu[~inliers, 1], c='#cb3d51', s=15,
                      label='Excluded GPS fit pairs')
    route.scatter(*aligned_xyz[0, :2], c='#229960', s=40, label='Start')
    route.set_aspect('equal', adjustable='datalim')
    route.set_xlabel('East (m)')
    route.set_ylabel('North (m)')
    route.legend(fontsize=9)
    errors = np.linalg.norm(aligned_xyz[:, :2] - gps_enu[:, :2], axis=1)
    residual.plot(gps_stamps - gps_stamps[0], errors, color='#7556a3', linewidth=1)
    residual.axhline(metrics['outlier_cutoff_m'], color='#cb3d51', linestyle='--',
                    label='Outlier cutoff')
    residual.set_xlabel('Source time since first match (s)')
    residual.set_ylabel('Horizontal GPS–LIO residual (m)')
    residual.legend(fontsize=9)
    for axis in (route, residual):
        axis.grid(alpha=.25)
    figure.suptitle('GPS anchor of local LiDAR–IMU map — scale preserved\n'
                   f'Inlier fit RMSE {metrics["horizontal_residual_rmse_m"]:.2f} m; '
                   'GPS is not ground truth; antenna lever arm and physical clock alignment unverified')
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def align_map(map_directory, bag=None, output=None, maximum_pose_gap_s=.25):
    map_directory = Path(map_directory).expanduser().resolve()
    manifest_path = map_directory / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    source_bag = Path(manifest['bag']).expanduser().resolve()
    bag = Path(bag).expanduser().resolve() if bag is not None else source_bag
    output = Path(output or map_directory / 'gps_alignment').expanduser().resolve()
    if output.exists():
        raise ValueError(f'output already exists; choose a new directory: {output}')
    if output == map_directory or map_directory.is_relative_to(output):
        raise ValueError('alignment output must not replace the source map directory')
    if bag != source_bag:
        raise ValueError('selected GPS bag differs from the map run_manifest source bag')
    mapping_report_path = map_directory / 'mapping_report.json'
    mapping_report = json.loads(mapping_report_path.read_text())
    if (mapping_report.get('map_frame') != 'map'
            or mapping_report.get('pose_child_frame') != 'os_sensor'):
        raise ValueError('mapping_report must describe map-frame poses of os_sensor')
    if mapping_report.get('failed_input') is not False:
        raise ValueError('mapping_report does not confirm valid mapping inputs (failed_input must be false)')
    if not (bag / 'metadata.yaml').is_file():
        raise ValueError(f'bag metadata does not exist: {bag}')
    trajectory_path, pcd_path = map_directory / 'trajectory.csv', map_directory / 'pointcloud_map.pcd'
    trajectory, trajectory_statistics = load_trajectory(trajectory_path)
    gps, sigmas, known_covariances, gps_statistics = read_gps(bag)
    indices, local_positions, matching_statistics = match_timestamps(trajectory, gps,
                                                                    maximum_pose_gap_s)
    matched_gps = gps[indices]
    origin = matched_gps[0, 1:4].copy()
    gps_enu = geographic_to_enu(matched_gps[:, 1:4], origin)
    transform, inliers, residuals, metrics = fit_planar_rigid(local_positions, gps_enu,
                                                           matched_gps[:, 4])
    aligned_trajectory = transform_trajectory(trajectory, transform)
    matched_aligned = local_positions @ transform[:3, :3].T + transform[:3, 3]
    sigma_values = sigmas[indices][known_covariances[indices]]
    if len(sigma_values):
        metrics['receiver_reported_horizontal_sigma_median_m'] = float(np.median(sigma_values))
        metrics['receiver_reported_horizontal_sigma_max_m'] = float(np.max(sigma_values))
    metrics['gps_status_class'] = {'0': 'standalone fix', '1': 'SBAS fix', '2': 'GBAS fix'}
    # Exclusive output creation makes reruns explicit, including failed partial runs.
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / '.incomplete'
    incomplete.write_text('Alignment artifacts are incomplete until this marker is removed.\n')
    pcd_statistics = transform_pcd(pcd_path, output / 'pointcloud_map_enu.pcd', transform)
    write_csv(output / 'trajectory_enu.csv', POSE_COLUMNS, aligned_trajectory)
    pair_columns = ('stamp_sec', 'map_x_m', 'map_y_m', 'map_z_m',
                    'gps_east_m', 'gps_north_m', 'gps_up_m',
                    'aligned_east_m', 'aligned_north_m', 'aligned_up_m',
                    'horizontal_residual_m', 'fit_inlier', 'reported_horizontal_sigma_m')
    pairs = np.column_stack((matched_gps[:, 0], local_positions, gps_enu, matched_aligned,
                             residuals, inliers, sigmas[indices]))
    write_csv(output / 'matched_gps_lio.csv', pair_columns, pairs)
    make_plot(output / 'gps_alignment_preview.png', matched_gps[:, 0], gps_enu,
              matched_aligned, inliers, metrics)
    result = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': {'bag': str(bag), 'map_dir': str(map_directory),
                   'trajectory': str(trajectory_path), 'pointcloud': str(pcd_path),
                   'trajectory_sha256': sha256_file(trajectory_path),
                   'pointcloud_sha256': pcd_statistics['source_sha256'],
                   'run_manifest_sha256': sha256_file(manifest_path),
                   'mapping_report_sha256': sha256_file(mapping_report_path),
                   'bag_metadata_sha256': sha256_file(bag / 'metadata.yaml')},
        'coordinate_conventions': {
            'source_frame': 'map', 'pose_child_frame': 'os_sensor', 'target_frame': 'enu',
            'target_axes': {'x': 'east', 'y': 'north', 'z': 'up'},
            'units': 'metres', 'datum': 'WGS84', 'altitude': 'ellipsoidal',
            'origin_definition': 'first usable GPS fix matched to saved trajectory',
            'matrix_operation': 'p_enu = T_enu_from_map @ [p_map_x,p_map_y,p_map_z,1]',
            'orientation_operation': 'q_enu_from_sensor = q_enu_from_map * q_map_from_sensor',
            'pcd_viewpoint': 'transformed identity local-map viewpoint'},
        'origin_wgs84': {'latitude_deg': float(origin[0]), 'longitude_deg': float(origin[1]),
                         'altitude_m': float(origin[2])},
        'transform': {'T_enu_from_map': transform.tolist(),
                      'rotation_yaw_rad': math.atan2(transform[1, 0], transform[0, 0]),
                      'translation_m': transform[:3, 3].tolist(), 'scale': 1.,
                      'roll_pitch_fitted': False},
        'trajectory': trajectory_statistics, 'gps': gps_statistics,
        'timestamp_matching': matching_statistics, 'fit': metrics, 'pointcloud': pcd_statistics,
        'limitations': {'antenna_lever_arm_measured': False,
                       'antenna_lever_arm_applied': False,
                       'constant_lever_arm_offset_can_bias_translation': True,
                       'turning_antenna_offset_can_contribute_residuals': True,
                       'physical_clock_alignment_verified': False,
                       'receiver_covariance_is_accuracy_truth': False,
                       'GPS_or_survey_ground_truth_used': False,
                       'gps_fused_by_LIO_estimator': False,
                       'loop_closure': False,
                       'vertical_anchor': 'robust constant offset; original relative Z geometry preserved'},
        'artifacts': {'aligned_pointcloud': 'pointcloud_map_enu.pcd',
                      'aligned_trajectory': 'trajectory_enu.csv',
                      'matched_pairs': 'matched_gps_lio.csv', 'preview': 'gps_alignment_preview.png'},
        'references': REFERENCES,
    }
    with (output / 'alignment.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    incomplete.unlink()
    return output, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('map_directory', type=Path)
    parser.add_argument('--bag', type=Path, help='defaults to the saved run_manifest.json bag')
    parser.add_argument('--output', type=Path, help='new directory; defaults to map_directory/gps_alignment')
    parser.add_argument('--maximum-pose-gap', type=float, default=.25, metavar='SECONDS')
    args = parser.parse_args(argv)
    try:
        output, report = align_map(args.map_directory, args.bag, args.output,
                                   args.maximum_pose_gap)
    except Exception as error:
        print(f'GPS alignment failed: {error}', file=sys.stderr)
        return 1
    print(f'GPS alignment saved: {output}')
    print(f'{report["fit"]["inliers"]}/{report["fit"]["matched_positions"]} fit pairs; '
          f'horizontal residual RMSE {report["fit"]["horizontal_residual_rmse_m"]:.3f} m; '
          f'scale 1; {report["pointcloud"]["points"]:,} map points')
    print('Residuals compare GPS with LIO; they do not measure true map accuracy.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
