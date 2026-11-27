"""Read a saved map only for replay of its original, timestamped recording.

No ROS node, clock, transform publisher or localization estimator lives here.
Source scan-end time selects a recorded map -> os_sensor pose. Spatial cropping
precedes display decimation so a large saved route keeps useful local detail.
"""
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def stamp_nanoseconds(value):
    try:
        number = Decimal(str(value)) * Decimal(1_000_000_000)
        if not number.is_finite() or number <= 0:
            raise ValueError('trajectory timestamp must be positive and finite')
        return int(number.to_integral_value())
    except InvalidOperation as error:
        raise ValueError('trajectory timestamp is invalid') from error


def quaternion_rotation(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('saved trajectory quaternion is invalid')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def slerp(first, second, fraction):
    """Shortest quaternion arc, including equivalent q/-q representations."""
    a, b = np.asarray(first, float), np.asarray(second, float)
    a, b = a/np.linalg.norm(a), b/np.linalg.norm(b)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, -1., 1.))
    if dot > .9995:
        result = a+fraction*(b-a)
        return result/np.linalg.norm(result)
    angle = math.acos(dot)
    return (math.sin((1-fraction)*angle)*a+math.sin(fraction*angle)*b)/math.sin(angle)


def read_trajectory(path):
    stamps, positions, quaternions = [], [], []
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        expected = ['stamp_sec', 'x_m', 'y_m', 'z_m', 'qx', 'qy', 'qz', 'qw']
        if reader.fieldnames != expected:
            raise ValueError('saved trajectory has unsupported columns')
        for row in reader:
            stamp = stamp_nanoseconds(row['stamp_sec'])
            position = [float(row[key]) for key in ('x_m', 'y_m', 'z_m')]
            quaternion = [float(row[key]) for key in ('qx', 'qy', 'qz', 'qw')]
            if not np.isfinite(position).all():
                raise ValueError('saved trajectory contains a nonfinite position')
            quaternion_rotation(quaternion)
            if stamps and stamp <= stamps[-1]:
                raise ValueError('saved trajectory timestamps must increase strictly')
            stamps.append(stamp)
            positions.append(position)
            quaternions.append(quaternion)
    if not stamps:
        raise ValueError('saved trajectory is empty')
    return (np.asarray(stamps, np.int64), np.asarray(positions, float),
            np.asarray(quaternions, float))


def scan_end_nanoseconds(message):
    """Use the Ouster's native uint32 t offsets, with row padding/endian support."""
    if message.header.frame_id not in ('os_lidar', 'os_sensor'):
        raise ValueError('saved map requires an os_lidar/os_sensor source scan')
    fields = {field.name: field for field in message.fields}
    field = fields.get('t')
    if field is None or field.datatype != 6 or field.count != 1:
        raise ValueError('saved map requires the original uint32 t nanosecond field')
    if (message.width < 1 or message.height < 1 or message.point_step <= 0 or
            field.offset < 0 or field.offset+4 > message.point_step or
            message.row_step < message.width*message.point_step or
            len(message.data) < message.row_step*message.height):
        raise ValueError('source scan timing layout is incomplete')
    offsets = np.ndarray((message.height, message.width),
                         dtype='>u4' if message.is_bigendian else '<u4',
                         buffer=message.data, offset=field.offset,
                         strides=(message.row_step, message.point_step))
    last_offset = int(offsets.max())
    # Current capture modes have a 100 ms or 50 ms revolution. Accept slower
    # native modes too, but reject incompatible units or a corrupted offset.
    if not 0 < last_offset <= 250_000_000:
        raise ValueError('source scan t offsets do not describe a valid revolution')
    start = int(message.header.stamp.sec)*1_000_000_000+int(message.header.stamp.nanosec)
    if start <= 0:
        raise ValueError('source scan timestamp is invalid')
    return start+last_offset


def validate_source(map_directory, source_bag):
    directory, bag = Path(map_directory).expanduser().resolve(), Path(source_bag).expanduser().resolve()
    manifest = json.loads((directory/'run_manifest.json').read_text())
    report = json.loads((directory/'mapping_report.json').read_text())
    if not manifest.get('bag') or Path(manifest['bag']).expanduser().resolve() != bag:
        raise ValueError('saved map belongs to a different recording; use its original source bag')
    if float(manifest.get('timestamp_offset_seconds', 0.)) != 0.:
        raise ValueError('saved-map replay requires unchanged original trajectory timestamps')
    if (report.get('map_frame') != 'map' or report.get('pose_child_frame') != 'os_sensor'
            or report.get('failed_input') is not False):
        raise ValueError('saved map must have valid map -> os_sensor trajectory provenance')
    if not (directory/'pointcloud_map.pcd').is_file() or not (directory/'trajectory.csv').is_file():
        raise ValueError('saved map is missing its point cloud or trajectory')
    return directory, manifest, report


def read_pcd_xyz(path):
    """Memory-map the adapter's unchanged binary XYZ/intensity PCD."""
    path = Path(path)
    with path.open('rb') as stream:
        header = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError('saved PCD has no DATA header')
            words = line.decode('ascii').strip().split()
            if not words or words[0].startswith('#'):
                continue
            header[words[0]] = words[1:]
            if words[0] == 'DATA':
                break
        offset = stream.tell()
    if (header.get('FIELDS') != ['x', 'y', 'z', 'intensity'] or
            header.get('SIZE') != ['4']*4 or header.get('TYPE') != ['F']*4 or
            header.get('COUNT', ['1']*4) != ['1']*4 or header.get('DATA') != ['binary']):
        raise ValueError('saved map must use binary float32 XYZ/intensity PCD')
    count = int(header['POINTS'][0])
    if count <= 0 or path.stat().st_size != offset+count*16:
        raise ValueError('saved PCD point count/payload is inconsistent')
    return np.memmap(path, mode='r', dtype='<f4', offset=offset, shape=(count, 4))[:, :3]


class SpatialMap:
    """Index all finite saved points; limit only the requested local display."""
    def __init__(self, xyz, cell_m=10.):
        self.xyz, self.cell_m = np.asarray(xyz), float(cell_m)
        valid = np.flatnonzero(np.isfinite(self.xyz).all(axis=1))
        if not len(valid):
            raise ValueError('saved map has no finite points')
        cells = np.floor(self.xyz[valid, :2]/self.cell_m).astype(np.int64)
        order = np.lexsort((cells[:, 1], cells[:, 0]))
        sorted_cells = cells[order]
        self.indices = valid[order]
        changes = np.r_[0, np.flatnonzero(np.any(sorted_cells[1:] != sorted_cells[:-1], axis=1))+1,
                        len(sorted_cells)]
        self.cells = {tuple(sorted_cells[start]): (int(start), int(stop))
                      for start, stop in zip(changes[:-1], changes[1:])}
        self.z_bounds = (float(self.xyz[valid, 2].min()), float(self.xyz[valid, 2].max()))

    def local_points(self, transform, forward_m, lateral_m, max_points):
        rotation, translation = transform[:3, :3], transform[:3, 3]
        # Bound map XY from sensor XY and every map Z. This also handles roll
        # and pitch rather than assuming an exactly level vehicle.
        if abs(np.linalg.det(rotation[:2, :2])) < 1e-5:
            selected = self.indices
        else:
            corners = np.array([[x, y, z] for x in (0., forward_m)
                                for y in (-lateral_m, lateral_m) for z in self.z_bounds])
            world_xy = ((corners[:, :2]-(corners[:, 2:3]-translation[2])*rotation[2, :2])
                        @ np.linalg.inv(rotation[:2, :2])+translation[:2])
            lower = np.floor(world_xy.min(axis=0)/self.cell_m).astype(int)
            upper = np.floor(world_xy.max(axis=0)/self.cell_m).astype(int)
            ranges = [value for key, value in self.cells.items()
                      if lower[0] <= key[0] <= upper[0] and lower[1] <= key[1] <= upper[1]]
            if not ranges:
                return np.empty((0, 3), np.float32), 0
            selected = np.concatenate([self.indices[start:stop] for start, stop in ranges])
        points = (self.xyz[selected]-translation)@rotation
        keep = ((points[:, 0] >= 0) & (points[:, 0] <= forward_m) &
                (np.abs(points[:, 1]) <= lateral_m))
        points = points[keep]
        visible_count = len(points)
        points = points[::max(1, math.ceil(visible_count/max_points))]
        return points.astype(np.float32), visible_count


class SavedMapSource:
    def __init__(self, directory, source_bag, alignment_directory=None):
        self.directory, self.manifest, self.report = validate_source(directory, source_bag)
        self.source_bag = Path(source_bag).expanduser().resolve()
        self.stamps, self.positions, self.quaternions = read_trajectory(self.directory/'trajectory.csv')
        xyz = read_pcd_xyz(self.directory/'pointcloud_map.pcd')
        if len(xyz) != self.report.get('map_point_count'):
            raise ValueError('saved PCD count differs from mapping report')
        self.spatial = SpatialMap(xyz)
        self.point_count = len(xyz)
        self.enu_from_map = None
        self.alignment = None
        if alignment_directory:
            self.load_alignment(alignment_directory)

    def load_alignment(self, directory):
        path = Path(directory).expanduser().resolve()/'alignment.json'
        if (path.parent/'.incomplete').exists():
            raise ValueError('GPS alignment is incomplete; wait for the alignment export to finish')
        data = json.loads(path.read_text())
        source = data.get('source', {})
        source_bag = source.get('bag') or data.get('source_bag')
        source_map = source.get('map_dir') or data.get('source_map_dir')
        if (not source_bag or Path(source_bag).resolve() != self.source_bag or
                not source_map or Path(source_map).resolve() != self.directory):
            raise ValueError('GPS alignment does not belong to this saved map/source bag')
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
        transform_data = data.get('transform', data)
        matrix = np.asarray(transform_data['T_enu_from_map'], float)
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all() or
                not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8) or
                not np.allclose(matrix[:3, :3].T@matrix[:3, :3], np.eye(3), atol=1e-6) or
                not np.isclose(np.linalg.det(matrix[:3, :3]), 1., atol=1e-6)):
            raise ValueError('GPS alignment must be a finite rigid map -> ENU transform')
        self.enu_from_map, self.alignment = matrix, data

    def pose_at(self, stamp_ns, max_interval_ns=150_000_000):
        """Use an exact/roundoff-near sample or short bracket; never extrapolate."""
        index = int(np.searchsorted(self.stamps, stamp_ns))
        candidates = [i for i in (index-1, index) if 0 <= i < len(self.stamps)]
        closest = min(candidates, key=lambda i: abs(int(self.stamps[i])-stamp_ns))
        delta = int(self.stamps[closest])-stamp_ns
        # Native preprocessing can omit a final return; sub-ms differences are
        # reported explicitly and remain much smaller than the source pair gate.
        if abs(delta) <= 1_000_000:
            position, quaternion = self.positions[closest], self.quaternions[closest]
            pose_stamp, method, bracket = int(self.stamps[closest]), 'recorded sample', 0
        elif index == 0 or index == len(self.stamps):
            return None, 'source scan is outside the saved trajectory; no extrapolation'
        else:
            left, right = int(self.stamps[index-1]), int(self.stamps[index])
            bracket = right-left
            if bracket > max_interval_ns:
                return None, f'saved trajectory gap {bracket/1e6:.1f} ms exceeds 150 ms'
            fraction = (stamp_ns-left)/bracket
            position = self.positions[index-1]+fraction*(self.positions[index]-self.positions[index-1])
            quaternion = slerp(self.quaternions[index-1], self.quaternions[index], fraction)
            pose_stamp, method, delta = stamp_ns, 'interpolated recorded poses', 0
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = quaternion_rotation(quaternion), position
        return {'transform': transform, 'query_stamp_ns': int(stamp_ns),
                'pose_stamp_ns': pose_stamp, 'pose_delta_ms': delta/1e6,
                'method': method, 'bracket_ms': bracket/1e6}, 'source scan-end paired'

    def mapping_for_pose(self, pose, forward_m, lateral_m, max_points):
        transform, stamp = pose['transform'], pose['query_stamp_ns']
        path = self.positions[self.stamps <= stamp]
        path = np.vstack((path, transform[:3, 3]))
        points, visible_count = self.spatial.local_points(transform, forward_m, lateral_m, max_points)
        mapping = {'points': points, 'path': (path-transform[:3, 3])@transform[:3, :3],
                   'world_path': path, 'world_position': transform[:3, 3],
                   'world_yaw_rad': math.atan2(transform[1, 0], transform[0, 0]),
                   'source': 'saved map', 'georeferenced': self.enu_from_map is not None,
                   'visible_point_count': visible_count}
        if self.enu_from_map is not None:
            enu_pose = self.enu_from_map@transform
            mapping.update(world_path=path@self.enu_from_map[:3, :3].T+self.enu_from_map[:3, 3],
                           world_position=enu_pose[:3, 3],
                           world_yaw_rad=math.atan2(enu_pose[1, 0], enu_pose[0, 0]))
        return mapping
