#!/usr/bin/python3
"""Experimental live scan localization against a frozen, GPS-aligned 3D map.

This estimator never loads a saved trajectory. GPS gives only a coarse position;
unknown sensor yaw is searched against 3D geometry. The map stays immutable.
Latest-scan processing, source scan-end stamps, registration rejection and explicit
lost states keep a slow/failed result from masquerading as a current location.

Dependency: system NumPy/SciPy; SciPy cKDTree reference:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
"""
import argparse
from collections import deque
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import threading
import time

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from align_local_map_gps import geographic_to_enu, usable_fix
from saved_map_source import read_pcd_xyz, scan_end_nanoseconds


@dataclass
class LocalizerConfig:
    map_voxel_m: float = .6
    scan_voxel_m: float = .65
    maximum_scan_points: int = 3500
    maximum_target_points: int = 100000
    map_crop_radius_m: float = 65.
    map_crop_rebuild_m: float = 10.
    minimum_range_m: float = 3.
    maximum_range_m: float = 55.
    maximum_iterations: int = 18
    correspondence_m: float = 1.2
    minimum_fitness: float = .45
    maximum_rmse_m: float = .7
    maximum_gps_offset_m: float = 15.
    maximum_speed_mps: float = 35.
    maximum_yaw_rate_rps: float = 1.
    maximum_tracking_gap_s: float = 2.
    initialization_yaw_step_degrees: float = 15.
    initialization_score_margin: float = .015
    max_rate_hz: float = 5.


def voxel_points(points, size, limit=None):
    points = np.asarray(points, float)
    points = points[np.isfinite(points).all(axis=1)]
    if size <= 0 or not math.isfinite(size):
        raise ValueError('voxel size must be positive and finite')
    if not len(points):
        return np.empty((0, 3), float)
    # A representative original point preserves actual surfaces, unlike centroids
    # across ground/wall intersections. Fixed sampling makes timing repeatable.
    cells = np.floor(points/size).astype(np.int64)
    cells -= cells.min(axis=0)
    shape = cells.max(axis=0)+1
    # Integer voxel keys avoid sorting three structured fields per raw return.
    # Fall back to tuples rather than overflowing for unusually vast maps.
    if math.prod(int(value) for value in shape) < np.iinfo(np.int64).max:
        keys = (cells[:, 0]*shape[1]+cells[:, 1])*shape[2]+cells[:, 2]
        _, indices = np.unique(keys, return_index=True)
    else:
        _, indices = np.unique(cells, axis=0, return_index=True)
    points = points[np.sort(indices)]
    if limit and len(points) > limit:
        points = points[np.linspace(0, len(points)-1, limit, dtype=int)]
    return np.ascontiguousarray(points, dtype=float)


def validate_transform(value):
    transform = np.asarray(value, float)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(transform[:3, :3].T@transform[:3, :3], np.eye(3), atol=1e-5)
            or not math.isclose(np.linalg.det(transform[:3, :3]), 1., abs_tol=1e-5)):
        raise ValueError('transform must be finite, rigid and right-handed')
    return transform.copy()


def sensor_from_metadata(metadata):
    value = metadata.get('lidar_intrinsics', {}).get('lidar_to_sensor_transform')
    if value is None:
        value = metadata.get('lidar_to_sensor_transform')
    if value is None:
        raise ValueError('Ouster factory lidar_to_sensor_transform is required')
    transform = np.asarray(value, float).reshape(4, 4)
    transform[:3, 3] *= .001  # Ouster metadata factory translation is in millimetres.
    return validate_transform(transform)


def read_sensor_scan(message, sensor_from_lidar, gyro_sensor=None, velocity_sensor=None):
    """Decode original padded/endian PointCloud2 and native nanosecond t.

    Constant-rate gyro/velocity compensation is deliberately a bounded prototype,
    not a full IMU preintegration/deskew estimator. No synthetic scan stamp is used.
    """
    end_ns = scan_end_nanoseconds(message)
    fields = {field.name: field for field in message.fields}
    ring = fields.get('ring')
    if ring is None or ring.datatype != 4 or ring.count != 1 or ring.offset+2 > message.point_step:
        raise ValueError('original Ouster uint16 ring field is required')
    values = []
    for name in ('x', 'y', 'z'):
        field = fields.get(name)
        if field is None or field.datatype != 7 or field.count != 1 or field.offset+4 > message.point_step:
            raise ValueError('original float32 XYZ fields are required')
        values.append(np.ndarray((message.height, message.width),
                                 dtype='>f4' if message.is_bigendian else '<f4',
                                 buffer=message.data, offset=field.offset,
                                 strides=(message.row_step, message.point_step)).reshape(-1))
    points = np.column_stack(values).astype(float)
    if message.header.frame_id == 'os_lidar':
        transform = validate_transform(sensor_from_lidar)
        points = points@transform[:3, :3].T+transform[:3, 3]
    elif message.header.frame_id != 'os_sensor':
        raise ValueError('expected original os_lidar or os_sensor scan')
    if gyro_sensor is not None or velocity_sensor is not None:
        field = fields['t']
        offsets = np.ndarray((message.height, message.width),
                             dtype='>u4' if message.is_bigendian else '<u4',
                             buffer=message.data, offset=field.offset,
                             strides=(message.row_step, message.point_step)).reshape(-1)
        dt = (int(offsets.max())-offsets.astype(float))*1e-9
        if gyro_sensor is not None:
            gyro = np.asarray(gyro_sensor, float)
            if gyro.shape != (3,) or not np.isfinite(gyro).all():
                raise ValueError('gyro must be a finite sensor-axis angular rate')
            # First-order exp(-omega*dt), with a second order term. <=100 ms here.
            rot = -dt[:, None]*gyro
            cross = np.cross(rot, points)
            points += cross+.5*np.cross(rot, cross)
        if velocity_sensor is not None:
            velocity = np.asarray(velocity_sensor, float)
            if velocity.shape != (3,) or not np.isfinite(velocity).all():
                raise ValueError('velocity must be finite and use sensor axes')
            points -= dt[:, None]*velocity
    return points, end_ns*1e-9


def rigid_fit(source, target, weights=None):
    if len(source) < 6:
        raise ValueError('rigid alignment needs at least six correspondences')
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, float)
    weights = weights/weights.sum()
    a, b = weights@source, weights@target
    left, _, right_t = np.linalg.svd((source-a).T@(weights[:, None]*(target-b)))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(right_t.T@left.T)
    rotation = right_t.T@correction@left.T
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, b-rotation@a
    return transform


def registration_metrics(source, tree, transform, threshold):
    transformed = source@transform[:3, :3].T+transform[:3, 3]
    distances, _ = tree.query(transformed, workers=1)
    valid = distances < threshold
    rmse = float(np.sqrt(np.mean(distances[valid]**2))) if valid.any() else math.inf
    # Structural points discriminate yaw on a flat road; a ground-only match is
    # insufficient to claim a localized position/heading.
    structural = source[:, 2] > -1.25
    structural_fitness = float(np.mean(valid[structural])) if structural.any() else 0.
    return {'fitness': float(np.mean(valid)), 'rmse_m': rmse,
            'structural_fitness': structural_fitness,
            'matched_points': int(valid.sum()),
            'structural_points': int(structural.sum())}


def coarse_icp(source, tree, target, initial, iterations=12, distance=3.):
    transform = initial.copy()
    for _ in range(iterations):
        points = source@transform[:3, :3].T+transform[:3, 3]
        distances, indices = tree.query(points, workers=1)
        keep = distances < distance
        if np.count_nonzero(keep) < 100:
            break
        threshold = np.quantile(distances[keep], .8)
        keep &= distances <= threshold
        delta = rigid_fit(points[keep], target[indices[keep]])
        transform = delta@transform
        if np.linalg.norm(delta[:3, 3]) < .015 and Rotation.from_matrix(delta[:3, :3]).magnitude() < .001:
            break
    return transform


def target_normals(tree, target, query=None):
    _, indices = tree.query(target if query is None else query, k=10, workers=1)
    neighbors = target[indices]
    centered = neighbors-neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    planar = eigenvalues[:, 0] < .4*np.maximum(eigenvalues[:, 1], 1e-6)
    return normals, planar


def fine_icp(source, tree, target, normals, planar, initial, config):
    transform = initial.copy()
    eigenvalues = np.zeros(6)
    for iteration in range(config.maximum_iterations):
        points = source@transform[:3, :3].T+transform[:3, 3]
        distance, indices = tree.query(points, workers=1)
        keep = (distance < max(config.correspondence_m, 2.-iteration*.1)) & planar[indices]
        if np.count_nonzero(keep) < 120:
            break
        p, q, n = points[keep], target[indices[keep]], normals[indices[keep]]
        residual = np.einsum('ij,ij->i', n, p-q)
        robust = np.minimum(1., .35/np.maximum(np.abs(residual), 1e-6))
        robust *= np.where(np.abs(n[:, 2]) > .9, .25, 1.)
        jacobian = np.column_stack((n, np.cross(p-transform[:3, 3], n)))
        weighted = jacobian*np.sqrt(robust[:, None])
        eigenvalues = np.linalg.eigvalsh(weighted.T@weighted)
        if eigenvalues[0] < 1e-5 or eigenvalues[-1]/eigenvalues[0] > 1e7:
            break
        step = np.linalg.lstsq(weighted, -residual*np.sqrt(robust), rcond=None)[0]
        scale = min(1., .8/max(np.linalg.norm(step[:3]), 1e-9),
                    .08/max(np.linalg.norm(step[3:]), 1e-9))
        step *= scale
        transform[:3, :3] = Rotation.from_rotvec(step[3:]).as_matrix()@transform[:3, :3]
        transform[:3, 3] += step[:3]
        if np.linalg.norm(step[:3]) < .01 and np.linalg.norm(step[3:]) < .001:
            break
    return transform, eigenvalues, iteration+1


def gate_registration(metrics, transform, prior, stamp_gap_s, config, initializing=False):
    """Acceptance is independent of a claimed covariance or prior saved pose."""
    if not np.isfinite(transform).all() or not math.isfinite(metrics['rmse_m']):
        return 'nonfinite_registration'
    if metrics['matched_points'] < 120 or metrics['structural_points'] < 100:
        return 'insufficient_structure'
    if metrics['fitness'] < config.minimum_fitness or metrics['structural_fitness'] < .35:
        return 'low_overlap'
    if metrics['rmse_m'] > config.maximum_rmse_m:
        return 'high_registration_residual'
    if prior is not None and not initializing:
        distance = float(np.linalg.norm(transform[:3, 3]-prior[:3, 3]))
        angle = Rotation.from_matrix(transform[:3, :3]@prior[:3, :3].T).magnitude()
        if distance > max(1., config.maximum_speed_mps*stamp_gap_s):
            return 'implausible_translation'
        if angle > max(.15, config.maximum_yaw_rate_rps*stamp_gap_s):
            return 'implausible_rotation'
    return None


def freshness_reason(now_sec, source_sec, receipt_age_sec, maximum_age_sec=1.):
    if not all(math.isfinite(value) for value in (now_sec, source_sec, receipt_age_sec)):
        return 'invalid_source_time'
    age = now_sec-source_sec
    if age < -.15:
        return 'future_scan'
    if age > maximum_age_sec or receipt_age_sec > 2.:
        return 'stale_scan'
    return None


def write_ready_file(path, payload):
    """Publish a complete readiness record, atomically and without overwriting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.localizer_ready_', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        # The hard link appears as a fully written file; unlike replace(), link
        # fails if an old readiness marker or another writer already exists.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class SourceClockGuard:
    """An out-of-order message cannot impersonate a recorded /clock rewind."""
    def __init__(self, allow_reset=False):
        self.allow_reset = bool(allow_reset)
        self.last_clock_ns = self.last_source_ns = None
        self.generation = self.rewinds = 0

    def observe_clock(self, now_ns):
        rewind = self.last_clock_ns is not None and now_ns < self.last_clock_ns-50_000_000
        self.last_clock_ns = int(now_ns)
        if rewind:
            self.rewinds += 1
            if self.allow_reset:
                self.generation += 1
                self.last_source_ns = None
        return rewind

    def accept_source(self, source_ns):
        if self.last_source_ns is not None and source_ns <= self.last_source_ns:
            return False
        self.last_source_ns = int(source_ns)
        return True


class MapLocalizer:
    """Offline-friendly estimator; accepts sensor-axis XYZ and independent GPS ENU."""
    def __init__(self, map_points, config=None):
        self.config = config or LocalizerConfig()
        started = time.perf_counter()
        self.map_points = voxel_points(map_points, self.config.map_voxel_m)
        if len(self.map_points) < 300:
            raise ValueError('the saved map has too few finite occupied voxels')
        self.map_xy = cKDTree(self.map_points[:, :2])
        # Surface normals belong to the immutable map, so compute them once.
        # Chunking bounds neighbor/covariance scratch memory. Normal estimation
        # must not create a hundreds-of-ms pause whenever the vehicle moves10m.
        normals_started = time.perf_counter()
        global_tree = cKDTree(self.map_points)
        self.map_normals = np.empty((len(self.map_points), 3), np.float32)
        self.map_planar = np.empty(len(self.map_points), bool)
        for first in range(0, len(self.map_points), 32768):
            stop = min(first+32768, len(self.map_points))
            normals, planar = target_normals(global_tree, self.map_points, self.map_points[first:stop])
            self.map_normals[first:stop], self.map_planar[first:stop] = normals, planar
        self.normal_preparation_s = time.perf_counter()-normals_started
        self.target_center = None
        self.target = self.tree = self.normals = self.planar = None
        self.transform = None
        self.last_stamp = None
        self.velocity = np.zeros(3)
        self.last_gps = None
        self.map_preparation_s = time.perf_counter()-started
        self.rejections = 0

    @classmethod
    def from_directory(cls, directory, alignment_directory=None, config=None):
        directory = Path(directory)
        alignment_directory = Path(alignment_directory or directory/'gps_alignment')
        alignment = json.loads((alignment_directory/'alignment.json').read_text())
        enu_from_map = validate_transform(alignment['transform']['T_enu_from_map'])
        # Only the immutable map is loaded. No trajectory provenance/pose lookup.
        points = read_pcd_xyz(directory/'pointcloud_map.pcd')
        points = points@enu_from_map[:3, :3].T+enu_from_map[:3, 3]
        instance = cls(points, config)
        origin = alignment['origin_wgs84']
        instance.origin_wgs84 = np.array([origin['latitude_deg'], origin['longitude_deg'], origin['altitude_m']])
        instance.alignment = alignment
        return instance

    def reset(self):
        self.transform = None
        self.last_stamp = None
        self.velocity[:] = 0
        self.last_gps = None

    def crop_target(self, center):
        if self.target_center is not None and np.linalg.norm(center[:2]-self.target_center[:2]) < self.config.map_crop_rebuild_m:
            return
        indices = np.asarray(self.map_xy.query_ball_point(center[:2], self.config.map_crop_radius_m), dtype=np.int64)
        if len(indices) < 300:
            raise ValueError('GPS/predicted position is outside usable saved-map coverage')
        if len(indices) > self.config.maximum_target_points:
            indices = indices[np.linspace(0, len(indices)-1, self.config.maximum_target_points, dtype=int)]
        self.target = np.ascontiguousarray(self.map_points[indices])
        self.tree = cKDTree(self.target)
        self.normals, self.planar = self.map_normals[indices], self.map_planar[indices]
        self.target_center = np.asarray(center).copy()

    def prepare_scan(self, points):
        points = np.asarray(points, float)
        keep = np.isfinite(points).all(axis=1)
        ranges = np.linalg.norm(points, axis=1)
        keep &= (ranges >= self.config.minimum_range_m) & (ranges <= self.config.maximum_range_m)
        # Crop upper sky and low far-below-ground returns; actual geometry remains3D.
        keep &= (points[:, 2] > -3.5) & (points[:, 2] < 8.)
        return voxel_points(points[keep], self.config.scan_voxel_m, self.config.maximum_scan_points)

    def localize(self, points, stamp, gps_enu=None, gyro_sensor=None, initial_yaw=None):
        started = time.perf_counter()
        result = {'accepted': False, 'status': 'waiting', 'stamp_sec': float(stamp),
                  'transform': None, 'map_frame': 'enu', 'child_frame': 'os_sensor',
                  'backend': 'bounded_robust_3d_icp', 'saved_trajectory_used': False}
        if not math.isfinite(stamp) or stamp <= 0:
            result.update(status='invalid_scan_stamp', elapsed_ms=0.)
            return result
        if self.last_stamp is not None and stamp <= self.last_stamp:
            result.update(status='source_clock_rewind', elapsed_ms=0.)
            return result
        gap = stamp-self.last_stamp if self.last_stamp is not None else 0.
        if gap > self.config.maximum_tracking_gap_s:
            self.reset()
        initializing = self.transform is None
        gps = np.asarray(gps_enu, float) if gps_enu is not None else None
        if gps is not None and (gps.shape != (3,) or not np.isfinite(gps).all()):
            gps = None
        if initializing and gps is None:
            result.update(status='waiting_for_fresh_gps', elapsed_ms=0.)
            return result
        prepared = time.perf_counter()
        source = self.prepare_scan(points)
        result['preprocess_ms'] = (time.perf_counter()-prepared)*1000
        result['scan_points'] = len(source)
        if len(source) < 300:
            result.update(status='insufficient_scan_points', elapsed_ms=(time.perf_counter()-started)*1000)
            return result
        prior = None if initializing else self.transform.copy()
        tracking_guess = None
        if not initializing:
            tracking_guess = prior.copy()
            tracking_guess[:3, 3] += self.velocity*gap
            # A cold bootstrap can finish after several scans have passed. GPS
            # displacement supplies a catch-up position seed; ICP still must
            # independently pass every geometric/motion gate before publication.
            if gps is not None and self.last_gps is not None and gap > .5:
                displacement = gps[:2]-self.last_gps[:2]
                if np.linalg.norm(displacement) <= self.config.maximum_speed_mps*gap+3.:
                    tracking_guess[:2, 3] = prior[:2, 3]+displacement
                    result['gps_displacement_seed_m'] = float(np.linalg.norm(displacement))
            if gyro_sensor is not None:
                gyro = np.asarray(gyro_sensor, float)
                if gyro.shape == (3,) and np.isfinite(gyro).all():
                    tracking_guess[:3, :3] = tracking_guess[:3, :3]@Rotation.from_rotvec(gyro*gap).as_matrix()
        center = gps if initializing else tracking_guess[:3, 3]
        cropped = time.perf_counter()
        try:
            self.crop_target(center)
        except ValueError as error:
            result.update(status='outside_map', detail=str(error), elapsed_ms=(time.perf_counter()-started)*1000)
            return result
        result['map_crop_ms'] = (time.perf_counter()-cropped)*1000
        result['target_points'] = len(self.target)
        registration_started = time.perf_counter()
        if initializing:
            bootstrap_started = time.perf_counter()
            yaws = ([float(initial_yaw)] if initial_yaw is not None else
                    np.radians(np.arange(-180, 180, self.config.initialization_yaw_step_degrees)))
            coarse_source = source[::max(1, len(source)//1100)]
            candidates = []
            for yaw in yaws:
                guess = np.eye(4)
                guess[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
                guess[:3, 3] = gps
                transform = coarse_icp(coarse_source, self.tree, self.target, guess)
                metrics = registration_metrics(coarse_source, self.tree, transform, self.config.correspondence_m)
                score = metrics['structural_fitness']-.35*metrics['rmse_m']
                candidates.append((score, transform))
            candidates.sort(key=lambda item: item[0], reverse=True)
            result['bootstrap_ms'] = (time.perf_counter()-bootstrap_started)*1000
            fine_started = time.perf_counter()
            refined = []
            for _, guess in candidates[:4]:
                transform, information, iterations = fine_icp(source, self.tree, self.target, self.normals, self.planar, guess, self.config)
                metrics = registration_metrics(source, self.tree, transform, self.config.correspondence_m)
                score = metrics['structural_fitness']-.35*metrics['rmse_m']
                refined.append((score, transform, metrics, information, iterations))
            refined.sort(key=lambda item: item[0], reverse=True)
            result['fine_icp_ms'] = (time.perf_counter()-fine_started)*1000
            score, transform, metrics, information, iterations = refined[0]
            distinct = [item for item in refined[1:]
                        if Rotation.from_matrix(item[1][:3, :3]@transform[:3, :3].T).magnitude() > .3]
            result['bootstrap_hypotheses'] = len(yaws)
            result['bootstrap_score_margin'] = score-distinct[0][0] if distinct else None
            if initial_yaw is None and distinct and score-distinct[0][0] < self.config.initialization_score_margin:
                result.update(metrics, status='ambiguous_initial_heading', elapsed_ms=(time.perf_counter()-started)*1000)
                return result
        else:
            result['bootstrap_ms'] = 0.
            fine_started = time.perf_counter()
            transform, information, iterations = fine_icp(source, self.tree, self.target, self.normals, self.planar, tracking_guess, self.config)
            metrics = registration_metrics(source, self.tree, transform, self.config.correspondence_m)
            result['fine_icp_ms'] = (time.perf_counter()-fine_started)*1000
        reason = gate_registration(metrics, transform, prior, gap, self.config, initializing)
        if not reason and gps is not None and np.linalg.norm(transform[:2, 3]-gps[:2]) > self.config.maximum_gps_offset_m:
            reason = 'gps_prior_disagreement'
        if not reason and (information[0] < 1e-5 or information[-1]/max(information[0], 1e-12) > 1e7):
            reason = 'unobservable_registration'
        result.update(metrics, iterations=iterations, registration_ms=(time.perf_counter()-registration_started)*1000,
                      elapsed_ms=(time.perf_counter()-started)*1000,
                      initializing=initializing,
                      gps_horizontal_offset_m=float(np.linalg.norm(transform[:2, 3]-gps[:2])) if gps is not None else None)
        result['information_condition'] = float(information[-1]/max(information[0], 1e-12))
        if reason:
            self.rejections += 1
            result['status'] = reason
            return result
        if prior is not None and gap > 0:
            measured_velocity = (transform[:3, 3]-prior[:3, 3])/gap
            self.velocity = .6*measured_velocity+.4*self.velocity
        self.transform, self.last_stamp = transform.copy(), float(stamp)
        self.last_gps = gps.copy() if gps is not None else None
        result.update(accepted=True, status='localized', transform=transform)
        # This is a nominal display covariance, not validated accuracy. It
        # prevents zero covariance claims; map/GPS/systematicerrors remainoutside.
        result['covariance'] = np.diag([max(.15, metrics['rmse_m'])**2]*3+[.05**2]*3)
        return result


def run_ros(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from rclpy.clock import Clock, ClockType
    from sensor_msgs.msg import PointCloud2, NavSatFix, Imu
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String

    class LocalizationNode(Node):
        def __init__(self):
            initialization_started = time.perf_counter()
            super().__init__('saved_map_lidar_localizer')
            self.localizer = MapLocalizer.from_directory(args.map, args.alignment,
                            LocalizerConfig(max_rate_hz=args.max_rate))
            metadata = json.loads(Path(args.metadata).read_text())
            self.sensor_from_lidar = sensor_from_metadata(metadata)
            self.sensor_from_imu_rotation = np.eye(3)
            # Ouster exposes imu_to_sensor_transform in millimetres as well.
            imu = metadata.get('imu_intrinsics', {}).get('imu_to_sensor_transform')
            if imu is None:
                raise ValueError('factory imu_to_sensor_transform is required')
            self.sensor_from_imu_rotation = validate_transform(
                np.asarray(imu, float).reshape(4, 4)*np.array([[1,1,1,.001]]*3+[[1,1,1,1]]))[:3, :3]
            self.lock, self.event, self.stop_event = threading.Lock(), threading.Event(), threading.Event()
            self.latest = None
            self.gps_fixes = deque(maxlen=300)
            self.imus = deque(maxlen=1000)
            self.received = self.overwritten = self.processed = self.accepted = 0
            self.generation = 0
            self.clock_guard = SourceClockGuard(args.reset_on_rewind and
                                               bool(self.get_parameter('use_sim_time').value))
            self.clock_rewinds = 0
            self.create_timer(.05, self.observe_clock, clock=Clock(clock_type=ClockType.STEADY_TIME))
            reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
            best_effort = QoSProfile(depth=100, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.odom_publisher = self.create_publisher(Odometry, '/camera_lidar_bev/localization/odometry', 5)
            self.status_publisher = self.create_publisher(String, '/camera_lidar_bev/localization/status', 5)
            self.create_subscription(PointCloud2, args.points_topic, self.receive_scan, reliable)
            self.create_subscription(NavSatFix, args.gps_topic, self.receive_gps, 20)
            self.create_subscription(Imu, args.imu_topic, self.receive_imu, best_effort)
            self.results = []
            self.worker = threading.Thread(target=self.work, daemon=False)
            self.worker.start()
            self.get_logger().info(f'Frozen map prepared: {len(self.localizer.map_points)} voxels; '
                                   f'{self.localizer.map_preparation_s:.2f}s '
                                   f'(normals {self.localizer.normal_preparation_s:.2f}s); '
                                   'yaw from geometry, no saved poses')
            if args.ready_file:
                try:
                    write_ready_file(args.ready_file, {
                        'ready': True, 'status': 'ready', 'pid': os.getpid(),
                        'backend': 'bounded_robust_3d_icp',
                        'map_directory': str(Path(args.map).resolve()),
                        'map_voxels': len(self.localizer.map_points),
                        'startup_s': time.perf_counter()-initialization_started,
                        'map_preparation_s': self.localizer.map_preparation_s,
                        'normal_preparation_s': self.localizer.normal_preparation_s,
                        'subscriptions_created': True, 'worker_started': True})
                except OSError:
                    self.stop_event.set()
                    self.event.set()
                    self.worker.join()
                    self.destroy_node()
                    raise

        def receive_scan(self, message):
            self.observe_clock()
            start_ns = int(message.header.stamp.sec)*10**9+int(message.header.stamp.nanosec)
            with self.lock:
                if not self.clock_guard.accept_source(start_ns):
                    self.publish_status({'status': 'out_of_order_scan', 'accepted': False})
                    return
                self.received += 1
                if self.latest is not None:
                    self.overwritten += 1
                self.latest = (message, self.generation, time.monotonic())
            self.event.set()

        def observe_clock(self):
            with self.lock:
                now = self.get_clock().now().nanoseconds
                previous_generation = self.clock_guard.generation
                rewind = self.clock_guard.observe_clock(now)
                if rewind:
                    self.clock_rewinds = self.clock_guard.rewinds
                    if self.clock_guard.generation != previous_generation:
                        self.generation = self.clock_guard.generation
                        self.latest = None
                        self.gps_fixes.clear()
                        self.imus.clear()

        def receive_gps(self, message):
            self.observe_clock()
            fix = usable_fix(message)
            if fix is None:
                return
            stamp, geographic, _, _, _ = fix
            xyz = geographic_to_enu(geographic[None, :], self.localizer.origin_wgs84)[0]
            with self.lock:
                self.gps_fixes.append((stamp, xyz, time.monotonic()))

        def receive_imu(self, message):
            self.observe_clock()
            if message.header.frame_id != 'os_imu':
                return
            stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
            gyro = self.sensor_from_imu_rotation@np.array([message.angular_velocity.x,
                message.angular_velocity.y, message.angular_velocity.z])
            if np.isfinite(gyro).all():
                with self.lock:
                    self.imus.append((stamp, gyro))

        def publish_status(self, result):
            serial = {key: value for key, value in result.items() if key not in ('transform', 'covariance')}
            for key, value in tuple(serial.items()):
                if isinstance(value, float) and not math.isfinite(value):
                    serial[key] = None
            serial.update(received=self.received, processed=self.processed, accepted_count=self.accepted,
                          overwritten_scans=self.overwritten,
                          observed_ros_clock_rewinds=self.clock_rewinds,
                          imu_aiding='constant-rate gyro rotation and constant-velocity deskew; no accelerometer integration',
                          covariance_is_accuracy_truth=False,
                          initialization='GPS position plus geometry yaw hypotheses')
            serial['state'] = 'tracking' if serial.get('accepted') else 'lost'
            serial['compute_ms'] = serial.get('elapsed_ms', 0.)+serial.get('decode_ms', 0.)
            if 'stamp_sec' in serial:
                serial.setdefault('stamp_ns', round(serial['stamp_sec']*10**9))
            message = String()
            message.data = json.dumps(serial, allow_nan=False)
            self.status_publisher.publish(message)
            self.results.append(serial)

        def work(self):
            last_generation, last_processing = -1, 0.
            while not self.stop_event.is_set():
                self.event.wait(.2)
                remaining = 1/args.max_rate-(time.monotonic()-last_processing)
                if remaining > 0 and self.stop_event.wait(remaining):
                    break
                with self.lock:
                    item, self.latest = self.latest, None
                    self.event.clear()
                    gps_fixes, imus = list(self.gps_fixes), list(self.imus)
                if item is None:
                    continue
                message, generation, received_wall = item
                if generation != last_generation:
                    self.localizer.reset()
                    last_generation = generation
                last_processing = time.monotonic()
                try:
                    stamp_ns = scan_end_nanoseconds(message)
                    stamp = stamp_ns*1e-9
                    now = self.get_clock().now().nanoseconds*1e-9
                    receipt_age = time.monotonic()-received_wall
                    timing_failure = freshness_reason(now, stamp, receipt_age, args.maximum_age)
                    if timing_failure:
                        self.publish_status({'status': timing_failure, 'accepted': False, 'stamp_sec': stamp,
                                             'source_age_s': now-stamp, 'receipt_age_s': receipt_age})
                        continue
                    gyro_samples = [value for t, value in imus if stamp-.15 <= t <= stamp+.03]
                    gyro = np.mean(gyro_samples, axis=0) if gyro_samples else None
                    propagation_samples = [value for t, value in imus
                                           if self.localizer.last_stamp is not None
                                           and self.localizer.last_stamp-.01 <= t <= stamp+.03]
                    propagation_gyro = np.mean(propagation_samples, axis=0) if propagation_samples else gyro
                    velocity_sensor = (self.localizer.transform[:3, :3].T@self.localizer.velocity
                                       if self.localizer.transform is not None else None)
                    decoded = time.perf_counter()
                    points, stamp = read_sensor_scan(message, self.sensor_from_lidar, gyro, velocity_sensor)
                    decode_ms = (time.perf_counter()-decoded)*1000
                    # Associate source times rather than rejecting an otherwise
                    # usable past fix because a newer fix arrived during work.
                    candidates = [fix for fix in gps_fixes
                                  if -.15 <= stamp-fix[0] <= 1. and time.monotonic()-fix[2] <= 2.]
                    past = [fix for fix in candidates if fix[0] <= stamp]
                    gps = max(past, key=lambda fix: fix[0]) if past else (
                        min(candidates, key=lambda fix: abs(fix[0]-stamp)) if candidates else None)
                    position = gps[1] if gps else None
                    result = self.localizer.localize(points, stamp, position, propagation_gyro,
                                                    None if args.initial_yaw is None else math.radians(args.initial_yaw))
                    result['stamp_ns'] = stamp_ns
                    result['matched_gps_age_s'] = stamp-gps[0] if gps else None
                    result['decode_ms'] = decode_ms
                    self.processed += 1
                    age = self.get_clock().now().nanoseconds*1e-9-stamp
                    result['source_age_s'] = age
                    result['receipt_age_s'] = time.monotonic()-received_wall
                    self.observe_clock()
                    with self.lock:
                        same_generation = generation == self.generation
                    timing_failure = freshness_reason(self.get_clock().now().nanoseconds*1e-9,
                                                       stamp, result['receipt_age_s'], args.maximum_age)
                    if not same_generation or timing_failure:
                        result['accepted'] = False
                        result['status'] = 'rewound_during_registration' if not same_generation else timing_failure+'_registration'
                    if result['accepted']:
                        transform = result['transform']
                        odometry = Odometry()
                        odometry.header.stamp.sec, odometry.header.stamp.nanosec = divmod(stamp_ns, 10**9)
                        odometry.header.frame_id, odometry.child_frame_id = 'enu', 'os_sensor'
                        odometry.pose.pose.position.x, odometry.pose.pose.position.y, odometry.pose.pose.position.z = map(float, transform[:3, 3])
                        q = Rotation.from_matrix(transform[:3, :3]).as_quat()
                        odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z, odometry.pose.pose.orientation.w = map(float, q)
                        odometry.pose.covariance = result['covariance'].reshape(-1).tolist()
                        self.odom_publisher.publish(odometry)
                        self.accepted += 1
                    self.publish_status(result)
                except (ValueError, TypeError, np.linalg.LinAlgError) as error:
                    self.publish_status({'status': 'invalid_input', 'accepted': False, 'detail': str(error)})

        def close(self):
            self.stop_event.set()
            self.event.set()
            self.worker.join(timeout=15.)
            if args.report:
                path = Path(args.report)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'backend': 'bounded_robust_3d_icp',
                    'config': asdict(self.localizer.config), 'map_preparation_s': self.localizer.map_preparation_s,
                    'normal_preparation_s': self.localizer.normal_preparation_s,
                    'samples': self.results, 'saved_trajectory_used': False,
                    'accuracy_independently_validated': False}, indent=2)+'\n')

    # Drain the registration worker before destroying ROS publishers/context.
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: stopped.set())
    signal.signal(signal.SIGTERM, lambda signum, frame: stopped.set())
    rclpy.init(args=args.ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    executor = SingleThreadedExecutor()
    try:
        node = LocalizationNode()
        executor.add_node(node)
        while not stopped.is_set() and rclpy.ok():
            executor.spin_once(timeout_sec=.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, required=True)
    parser.add_argument('--alignment', type=Path)
    parser.add_argument('--metadata', type=Path, required=True,
                        help='captured/current factory Ouster metadata; sensor must retain mounting')
    parser.add_argument('--points-topic', default='/points')
    parser.add_argument('--gps-topic', default='/sensing/gnss/fix')
    parser.add_argument('--imu-topic', default='/imu')
    parser.add_argument('--max-rate', type=float, default=5.)
    parser.add_argument('--maximum-age', type=float, default=1.)
    parser.add_argument('--initial-yaw', type=float, help='measured ENU yaw of os_sensor +X, degrees; default geometry search')
    parser.add_argument('--reset-on-rewind', action='store_true', help='explicit recorded-loop reset; requires use_sim_time')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--ready-file', type=Path,
                        help='new readiness marker path; atomically created after map/subscriptions/worker setup')
    args, args.ros_args = parser.parse_known_args()
    if args.ready_file and args.ready_file.exists():
        parser.error('ready file already exists; choose a new path')
    if (args.max_rate <= 0 or not math.isfinite(args.max_rate) or args.maximum_age <= 0
            or not math.isfinite(args.maximum_age)
            or (args.initial_yaw is not None and not math.isfinite(args.initial_yaw))):
        parser.error('rate and maximum age must be finite and positive')
    run_ros(args)


if __name__ == '__main__':
    main()
