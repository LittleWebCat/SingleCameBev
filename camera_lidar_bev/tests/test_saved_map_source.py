"""Saved replay geometry, source provenance and native scan timing regressions."""
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from saved_map_source import (SavedMapSource, SpatialMap, quaternion_rotation,
                              scan_end_nanoseconds, stamp_nanoseconds, validate_source)
from combined_bev_display import CombinedBevRenderer


class SavedMapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.bag = self.directory/'bag'
        self.bag.mkdir()
        self.map = self.directory/'map'
        self.map.mkdir()
        self.xyz = np.array([[2., 0., 1.], [10., 2., 0.], [18., -1., 3.]], np.float32)
        payload = np.column_stack((self.xyz, np.zeros(3, np.float32))).astype('<f4').tobytes()
        (self.map/'pointcloud_map.pcd').write_bytes(
            b'FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
            b'WIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA binary\n'+payload)
        (self.map/'run_manifest.json').write_text(json.dumps({'bag': str(self.bag),
                                                            'timestamp_offset_seconds': 0.}))
        (self.map/'mapping_report.json').write_text(json.dumps({'map_frame': 'map',
            'pose_child_frame': 'os_sensor', 'failed_input': False, 'map_point_count': 3}))
        self.write_trajectory(['100.000000001', '100.100000001', '100.200000001'])

    def tearDown(self):
        self.temporary.cleanup()

    def write_trajectory(self, stamps):
        with (self.map/'trajectory.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['stamp_sec', 'x_m', 'y_m', 'z_m', 'qx', 'qy', 'qz', 'qw'])
            for i, stamp in enumerate(stamps):
                yaw = i*math.pi/2
                # Deliberately alternate equivalent quaternion signs.
                sign = -1 if i % 2 else 1
                writer.writerow([stamp, i*2., 0., 0., 0., 0.,
                                 sign*math.sin(yaw/2), sign*math.cos(yaw/2)])

    def test_decimal_timestamp_and_native_padded_scan_end(self):
        self.assertEqual(stamp_nanoseconds('1791670489.809906960'), 1791670489809906960)
        for bigendian in (False, True):
            payload = bytearray(32)
            for offset, stamp in ((4, 0), (12, 20_000_000), (20, 50_000_000), (28, 99_900_000)):
                struct.pack_into('>I' if bigendian else '<I', payload, offset, stamp)
            scan = SimpleNamespace(header=SimpleNamespace(frame_id='os_lidar',
                         stamp=SimpleNamespace(sec=100, nanosec=950_000_000)),
                         fields=[SimpleNamespace(name='t', datatype=6, count=1, offset=4)],
                         width=2, height=2, point_step=8, row_step=16,
                         data=payload, is_bigendian=bigendian)
            self.assertEqual(scan_end_nanoseconds(scan), 101_049_900_000)
            scan.fields[0].datatype = 7
            with self.assertRaisesRegex(ValueError, 'uint32'):
                scan_end_nanoseconds(scan)

    def test_interpolation_uses_short_quaternion_arc_and_never_extrapolates(self):
        source = SavedMapSource(self.map, self.bag)
        pose, reason = source.pose_at(100_050_000_001)
        self.assertIsNotNone(pose)
        np.testing.assert_allclose(pose['transform'][:3, 3], [1., 0., 0.])
        self.assertAlmostEqual(math.atan2(pose['transform'][1, 0], pose['transform'][0, 0]), math.pi/4)
        self.assertEqual(pose['query_stamp_ns'], 100_050_000_001)
        self.assertEqual(pose['method'], 'interpolated recorded poses')
        self.assertIsNone(source.pose_at(99_990_000_001)[0])
        self.assertIsNone(source.pose_at(100_210_000_001)[0])
        self.write_trajectory(['100.000000001', '100.200000001'])
        source = SavedMapSource(self.map, self.bag)
        self.assertIsNone(source.pose_at(100_100_000_001)[0])

    def test_source_bag_and_pose_frame_are_enforced(self):
        with self.assertRaisesRegex(ValueError, 'different recording'):
            validate_source(self.map, self.directory/'different_bag')
        report = json.loads((self.map/'mapping_report.json').read_text())
        report['pose_child_frame'] = 'lio_imu'
        (self.map/'mapping_report.json').write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'os_sensor'):
            validate_source(self.map, self.bag)

    def test_enu_anchor_preserves_sensor_geometry_and_path(self):
        source = SavedMapSource(self.map, self.bag)
        pose, _ = source.pose_at(100_050_000_001)
        local = source.mapping_for_pose(pose, 80., 30., 100)
        alignment = self.map/'gps_alignment'
        alignment.mkdir()
        matrix = np.eye(4)
        matrix[:3, :3] = quaternion_rotation([0., 0., math.sin(.6), math.cos(.6)])
        matrix[:3, 3] = [123., -456., 2.]
        path = alignment/'alignment.json'
        path.write_text(json.dumps({'source': {'bag': str(self.bag), 'map_dir': str(self.map)},
                                   'transform': {'T_enu_from_map': matrix.tolist()}}))
        source.load_alignment(alignment)
        enu = source.mapping_for_pose(pose, 80., 30., 100)
        np.testing.assert_array_equal(enu['points'], local['points'])
        np.testing.assert_array_equal(enu['path'], local['path'])
        np.testing.assert_allclose(enu['world_path'], local['world_path']@matrix[:3, :3].T+matrix[:3, 3])
        self.assertEqual(len(enu['world_path']), 2)  # only past sample plus queried endpoint
        matrix[:3, :3] *= 2
        path.write_text(json.dumps({'source': {'bag': str(self.bag), 'map_dir': str(self.map)},
                                   'transform': {'T_enu_from_map': matrix.tolist()}}))
        with self.assertRaisesRegex(ValueError, 'rigid'):
            source.load_alignment(alignment)

    def test_crop_precedes_display_sampling_and_handles_tilt(self):
        inside = np.column_stack((np.linspace(1, 79, 300), np.zeros((300, 2))))
        outside = np.column_stack((np.linspace(1000, 1200, 3000), np.zeros((3000, 2))))
        spatial = SpatialMap(np.vstack((outside, inside)))
        points, count = spatial.local_points(np.eye(4), 80., 30., 100)
        self.assertEqual(count, 300)
        self.assertEqual(len(points), 100)
        rng = np.random.default_rng(3)
        xyz = rng.uniform([-150, -150, -15], [150, 150, 100], (5000, 3))
        transform = np.eye(4)
        transform[:3, :3] = quaternion_rotation([.1, -.08, .3, .9])
        transform[:3, 3] = [7., -12., 3.]
        points, count = SpatialMap(xyz).local_points(transform, 80., 30., 10000)
        all_local = (xyz-transform[:3, 3])@transform[:3, :3]
        keep = ((all_local[:, 0] >= 0) & (all_local[:, 0] <= 80.) & (np.abs(all_local[:, 1]) <= 30.))
        self.assertEqual(count, int(keep.sum()))
        np.testing.assert_allclose(np.sort(points[:, 0]), np.sort(all_local[keep, 0]), atol=1e-5)

    def test_alignment_rejects_incomplete_or_modified_source_artifacts(self):
        source = SavedMapSource(self.map, self.bag)
        alignment = self.map/'gps_alignment'
        alignment.mkdir()
        trajectory = self.map/'trajectory.csv'
        pointcloud = self.map/'pointcloud_map.pcd'
        data = {'source': {'bag': str(self.bag), 'map_dir': str(self.map),
                          'trajectory_sha256': hashlib.sha256(trajectory.read_bytes()).hexdigest(),
                          'pointcloud_sha256': hashlib.sha256(pointcloud.read_bytes()).hexdigest()},
                'transform': {'T_enu_from_map': np.eye(4).tolist()}}
        (alignment/'alignment.json').write_text(json.dumps(data))
        marker = alignment/'.incomplete'
        marker.touch()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            source.load_alignment(alignment)
        marker.unlink()
        source.load_alignment(alignment)
        original = trajectory.read_bytes()
        trajectory.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError, 'trajectory checksum changed'):
            source.load_alignment(alignment)
        trajectory.write_bytes(original)
        pointcloud.write_bytes(pointcloud.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError, 'pointcloud checksum changed'):
            source.load_alignment(alignment)

    def test_map_toggle_keeps_camera_pixels_unchanged(self):
        calibration = {'K': [100., 0., 160., 0., 100., 120., 0., 0., 1.],
                       'D': [0.]*5, 'T': np.eye(4).ravel().tolist()}
        renderer = CombinedBevRenderer(calibration)
        image = np.full((240, 320, 3), [20, 40, 60], np.uint8)
        mapping = {'points': self.xyz, 'path': np.array([[0., 0., 0.], [10., 0., 0.]])}
        sample = {'camera_2d_boxes': [], 'lidar_boxes': [], 'associations': []}
        on = renderer.render(image, sample, mapping=mapping, layers={'mapping': True})
        off = renderer.render(image, sample, mapping=mapping, layers={'mapping': False})
        np.testing.assert_array_equal(on[168:408, 22:342], off[168:408, 22:342])
        self.assertFalse(np.array_equal(on[168:408, 368:608], off[168:408, 368:608]))


if __name__ == '__main__':
    unittest.main()
