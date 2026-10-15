"""Frozen map geometry driven exclusively by fresh ENU localization messages.

This module has no ROS dependency and never parses recorded trajectory poses.
The recorded trajectory may be hashed to verify map/alignment provenance. Its
positions are never used as a fallback when the live estimator is unavailable.
"""
from bisect import bisect_left
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import numpy as np

from saved_map_source import (SpatialMap, quaternion_rotation, read_pcd_xyz,
                              slerp, validate_source)


def rigid_transform(value, label='pose'):
    matrix = np.asarray(value, float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
            not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-8) or
            not np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-6) or
            not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
        raise ValueError(f'{label} must be a finite rigid transform')
    return matrix


def unit_quaternion(value):
    quaternion = np.asarray(value, float)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError('localization quaternion is invalid')
    with np.errstate(over='ignore'):
        norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm < 1e-8:
        raise ValueError('localization quaternion is invalid')
    return quaternion/norm


def rotation_quaternion(rotation):
    """Stable matrix to XYZW conversion, including rotations near 180 degrees."""
    r = np.asarray(rotation, float)
    candidates = np.array([1+r[0, 0]-r[1, 1]-r[2, 2],
                           1-r[0, 0]+r[1, 1]-r[2, 2],
                           1-r[0, 0]-r[1, 1]+r[2, 2],
                           1+np.trace(r)])
    index = int(np.argmax(candidates))
    q = np.zeros(4)
    q[index] = .5*math.sqrt(max(0., candidates[index]))
    denominator = 4*q[index]
    if index == 0:
        q[1], q[2], q[3] = ((r[0, 1]+r[1, 0])/denominator,
                            (r[0, 2]+r[2, 0])/denominator,
                            (r[2, 1]-r[1, 2])/denominator)
    elif index == 1:
        q[0], q[2], q[3] = ((r[0, 1]+r[1, 0])/denominator,
                            (r[1, 2]+r[2, 1])/denominator,
                            (r[0, 2]-r[2, 0])/denominator)
    elif index == 2:
        q[0], q[1], q[3] = ((r[0, 2]+r[2, 0])/denominator,
                            (r[1, 2]+r[2, 1])/denominator,
                            (r[1, 0]-r[0, 1])/denominator)
    else:
        q[:3] = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0],
                          r[1, 0]-r[0, 1]])/denominator
    return q/np.linalg.norm(q)


def transform_boxes(boxes, enu_from_sensor):
    """Copy detector boxes into ENU, retaining their complete 3D orientation.

    Source centers are in os_sensor and size_lwh_m follows each box's own axes.
    Quaternion orientation takes precedence over the legacy sensor yaw. Eight
    ENU corners retain roll/pitch; callers may flatten only for BEV rendering.
    """
    transform = rigid_transform(enu_from_sensor, 'ENU localization')
    rotation, translation = transform[:3, :3], transform[:3, 3]
    result = []
    signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    for source in boxes:
        center = np.asarray(source['os_sensor_xyz_m'], float)
        size = np.asarray(source['size_lwh_m'], float)
        if (center.shape != (3,) or size.shape != (3,) or
                not np.isfinite(center).all() or not np.isfinite(size).all() or
                np.any(size <= 0)):
            raise ValueError('detector box needs a finite center and positive L/W/H')
        if 'quaternion_xyzw' in source:
            quaternion = unit_quaternion(source['quaternion_xyzw'])
        else:
            yaw = float(source['yaw_rad'])
            if not math.isfinite(yaw):
                raise ValueError('detector box yaw is invalid')
            quaternion = np.array([0., 0., math.sin(yaw/2), math.cos(yaw/2)])
        box_rotation = quaternion_rotation(quaternion)
        world_rotation = rotation@box_rotation
        world_center = rotation@center+translation
        corners = (signs*size*.5)@world_rotation.T+world_center
        box = dict(source)
        box.update(world_frame='enu', world_xyz_m=world_center.tolist(),
                   world_quaternion_xyzw=rotation_quaternion(world_rotation).tolist(),
                   world_yaw_rad=math.atan2(world_rotation[1, 0], world_rotation[0, 0]),
                   corners_world_m=corners.tolist())
        result.append(box)
    return result


class LiveMapSource:
    """Bounded localization history with exact/short-bracket scan-end pairing.

    source_now_ns uses the caller's ROS clock (including simulated replay time).
    Wall times use time.monotonic(), independent of the source clock. The
    default source clock is wall UTC for standalone live use.
    """
    def __init__(self, directory, alignment_directory=None, *, max_pose_records=4000,
                 max_interval_ns=150_000_000, max_source_age_s=1.,
                 max_receive_age_s=2., max_future_s=.15, max_overview_points=350000):
        directory = Path(directory).expanduser().resolve()
        manifest = json.loads((directory/'run_manifest.json').read_text())
        self.directory, self.manifest, self.report = validate_source(directory, manifest.get('bag', ''))
        self.source_bag = Path(self.manifest['bag']).expanduser().resolve()
        self.alignment_directory = Path(alignment_directory or directory/'gps_alignment').expanduser().resolve()
        self.enu_from_map, self.alignment = self._load_alignment()
        self.map_from_enu = np.linalg.inv(self.enu_from_map)
        xyz = read_pcd_xyz(directory/'pointcloud_map.pcd')
        if len(xyz) != self.report.get('map_point_count'):
            raise ValueError('frozen PCD count differs from mapping report')
        self.spatial = SpatialMap(xyz)
        self.point_count = len(xyz)
        if (max_pose_records < 2 or max_interval_ns <= 0 or max_source_age_s <= 0 or
                max_receive_age_s <= 0 or max_future_s < 0 or max_overview_points < 1):
            raise ValueError('live map limits must be positive, with at least two pose records')
        self.max_pose_records = int(max_pose_records)
        self.max_interval_ns = int(max_interval_ns)
        self.max_source_age_ns = int(max_source_age_s*1e9)
        self.max_receive_age_s = float(max_receive_age_s)
        self.max_future_ns = int(max_future_s*1e9)
        indices = self.spatial.indices[::max(1, math.ceil(len(self.spatial.indices)/max_overview_points))]
        self.world_map_preview_xyz = (xyz[indices]@self.enu_from_map[:3, :3].T+
                                      self.enu_from_map[:3, 3]).astype(np.float32)
        self._records = deque(maxlen=self.max_pose_records)
        self._lock = threading.RLock()
        self._clock_ns = None
        self.invalid_reason = 'waiting for fresh ENU localization'
        self.epoch = 0

    def _load_alignment(self):
        path = self.alignment_directory/'alignment.json'
        if (path.parent/'.incomplete').exists():
            raise ValueError('GPS alignment is incomplete')
        data = json.loads(path.read_text())
        source = data.get('source', {})
        source_bag = source.get('bag') or data.get('source_bag')
        source_map = source.get('map_dir') or data.get('source_map_dir')
        if (not source_bag or Path(source_bag).expanduser().resolve() != self.source_bag or
                not source_map or Path(source_map).expanduser().resolve() != self.directory):
            raise ValueError('GPS alignment does not belong to this frozen map/source bag')
        convention = data.get('coordinate_conventions', {})
        for name, expected in (('source_frame', 'map'), ('pose_child_frame', 'os_sensor'),
                               ('target_frame', 'enu')):
            if name in convention and convention[name] != expected:
                raise ValueError('GPS alignment requires map -> ENU and os_sensor poses')
        for name, source_path in (('trajectory', self.directory/'trajectory.csv'),
                                  ('pointcloud', self.directory/'pointcloud_map.pcd'),
                                  ('run_manifest', self.directory/'run_manifest.json'),
                                  ('bag_metadata', self.source_bag/'metadata.yaml')):
            expected = source.get(name+'_sha256')
            if expected is not None:
                digest = hashlib.sha256()
                with source_path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024*1024), b''):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    raise ValueError(f'GPS alignment source {name} checksum changed; regenerate alignment')
        return rigid_transform(data.get('transform', data)['T_enu_from_map'], 'map -> ENU alignment'), data

    @property
    def pose_count(self):
        with self._lock:
            return len(self._records)

    def invalidate(self, reason='localization unavailable'):
        with self._lock:
            self._records.clear()
            self.invalid_reason = str(reason)
            self.epoch += 1

    def observe_clock(self, now_ns, reset_on_rewind=False):
        """Clear all live history on a rewind; report if explicitly permitted.

        A rejected rewind also clears poses, because preserving records across
        a changed time domain would make earlier live data look current.
        """
        now_ns = int(now_ns)
        with self._lock:
            rewind = self._clock_ns is not None and now_ns < self._clock_ns-50_000_000
            self._clock_ns = now_ns
            if rewind:
                self.invalidate('source clock rewound; waiting for fresh localization')
                return bool(reset_on_rewind), self.invalid_reason
        return True, 'source clock monotonic'

    def _fresh(self, record, source_now_ns, wall_now):
        age_ns = source_now_ns-record['stamp_ns']
        if age_ns < -self.max_future_ns:
            return False, 'localization pose is ahead of the source clock'
        if age_ns > self.max_source_age_ns:
            return False, 'localization pose source timestamp is stale'
        age_wall = wall_now-record['received_wall']
        if age_wall < 0 or age_wall > self.max_receive_age_s:
            return False, 'localization pose wall receipt is stale'
        return True, 'localization pose is fresh'

    def add_pose(self, stamp_ns, position, quaternion_xyzw, *, parent_frame='enu',
                 child_frame='os_sensor', source_now_ns=None, received_wall=None, wall_now=None):
        """Accept a finite, fresh ENU -> os_sensor pose; reject older arrivals."""
        source_now_ns = time.time_ns() if source_now_ns is None else int(source_now_ns)
        received_wall = time.monotonic() if received_wall is None else float(received_wall)
        wall_now = time.monotonic() if wall_now is None else float(wall_now)
        if parent_frame != 'enu' or child_frame != 'os_sensor':
            return False, 'live localization requires enu -> os_sensor frames'
        try:
            stamp_ns = int(stamp_ns)
            position = np.asarray(position, float)
            if stamp_ns <= 0 or position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError('invalid localization timestamp/position')
            quaternion = unit_quaternion(quaternion_xyzw)
            if not math.isfinite(received_wall) or not math.isfinite(wall_now):
                raise ValueError('invalid localization wall receipt')
        except (TypeError, ValueError, OverflowError) as error:
            return False, str(error)
        record = dict(stamp_ns=stamp_ns, position=position.copy(), quaternion=quaternion.copy(),
                      received_wall=received_wall)
        # Checking at receipt avoids accepting poses that were already stale
        # before the consumer ever saw them. Future queries check again.
        fresh, reason = self._fresh(record, source_now_ns, wall_now)
        if not fresh:
            return False, reason
        with self._lock:
            if self._records and stamp_ns <= self._records[-1]['stamp_ns']:
                return False, 'duplicate or out-of-order localization pose'
            self._records.append(record)
            self.invalid_reason = ''
        return True, 'fresh localization pose accepted'

    def pose_at(self, stamp_ns, source_now_ns=None, wall_now=None):
        """Pair a native scan-end using <=1 ms tolerance or <=150 ms bracket."""
        stamp_ns = int(stamp_ns)
        source_now_ns = time.time_ns() if source_now_ns is None else int(source_now_ns)
        wall_now = time.monotonic() if wall_now is None else float(wall_now)
        if stamp_ns <= 0 or not math.isfinite(wall_now):
            return None, 'invalid source scan timestamp or wall clock'
        age_ns = source_now_ns-stamp_ns
        if age_ns < -self.max_future_ns or age_ns > self.max_source_age_ns:
            return None, 'source scan-end is outside the freshness window'
        with self._lock:
            records = list(self._records)
            epoch = self.epoch
            missing_reason = self.invalid_reason
        if not records:
            return None, missing_reason or 'waiting for fresh ENU localization'
        # A newer rejected/non-fresh pose cannot cause fallback to an older
        # recorded trajectory. All candidate records originated live here.
        fresh, reason = self._fresh(records[-1], source_now_ns, wall_now)
        if not fresh:
            return None, reason
        stamps = [record['stamp_ns'] for record in records]
        index = bisect_left(stamps, stamp_ns)
        candidates = [i for i in (index-1, index) if 0 <= i < len(records)]
        closest = min(candidates, key=lambda i: abs(stamps[i]-stamp_ns))
        delta = stamps[closest]-stamp_ns
        if abs(delta) <= 1_000_000:
            selected = [records[closest]]
            position, quaternion = selected[0]['position'], selected[0]['quaternion']
            pose_stamp, method, bracket = stamps[closest], 'live localization sample', 0
        elif index == 0 or index == len(records):
            return None, 'scan-end has no live pose bracket; no extrapolation'
        else:
            bracket = stamps[index]-stamps[index-1]
            if bracket > self.max_interval_ns:
                return None, f'live pose gap {bracket/1e6:.1f} ms exceeds {self.max_interval_ns/1e6:.1f} ms'
            selected = records[index-1:index+1]
            fraction = (stamp_ns-stamps[index-1])/bracket
            position = selected[0]['position']+fraction*(selected[1]['position']-selected[0]['position'])
            quaternion = slerp(selected[0]['quaternion'], selected[1]['quaternion'], fraction)
            pose_stamp, method, delta = stamp_ns, 'interpolated live localization poses', 0
        for record in selected:
            fresh, reason = self._fresh(record, source_now_ns, wall_now)
            if not fresh:
                return None, reason
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = quaternion_rotation(quaternion), position
        return dict(transform=transform, query_stamp_ns=stamp_ns, pose_stamp_ns=pose_stamp,
                    pose_delta_ms=delta/1e6, method=method, bracket_ms=bracket/1e6,
                    frame_id='enu', child_frame_id='os_sensor', epoch=epoch,
                    received_wall=min(record['received_wall'] for record in selected)), 'fresh scan-end paired'

    def mapping_for_pose(self, pose, forward_m, lateral_m, max_points, path=None):
        """Crop the original map before decimation, using the live 6DOF pose."""
        if forward_m <= 0 or lateral_m <= 0 or max_points < 1:
            raise ValueError('map crop and point limits must be positive')
        enu_pose = rigid_transform(pose['transform'], 'ENU localization')
        with self._lock:
            if pose.get('epoch', self.epoch) != self.epoch:
                raise ValueError('live pose was invalidated before map rendering')
            if path is None:
                path = [record['position'] for record in self._records
                        if record['stamp_ns'] <= pose['query_stamp_ns']]
        world_path = np.asarray(path, float).reshape(-1, 3)
        if not np.isfinite(world_path).all():
            raise ValueError('live localization path contains nonfinite positions')
        position, rotation = enu_pose[:3, 3], enu_pose[:3, :3]
        if not len(world_path) or not np.array_equal(world_path[-1], position):
            world_path = np.vstack((world_path, position))
        world_path = world_path[-self.max_pose_records:]
        map_pose = self.map_from_enu@enu_pose
        points, count = self.spatial.local_points(map_pose, forward_m, lateral_m, int(max_points))
        with self._lock:
            if pose.get('epoch', self.epoch) != self.epoch:
                raise ValueError('live pose was invalidated during map rendering')
        return dict(points=points, path=(world_path-position)@rotation, world_path=world_path,
                    world_position=position.copy(), world_yaw_rad=math.atan2(rotation[1, 0], rotation[0, 0]),
                    world_map_preview_xyz=self.world_map_preview_xyz,
                    overview_points=self.world_map_preview_xyz, source='live localization on saved map',
                    georeferenced=True, visible_point_count=count, pose_method=pose.get('method', 'live localization'))


FrozenLiveMap = LiveMapSource
