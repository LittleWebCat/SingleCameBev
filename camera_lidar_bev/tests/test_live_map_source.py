"""Fresh live localization, native scan pairing and complete box geometry."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from live_map_source import LiveMapSource, rotation_quaternion, transform_boxes
from saved_map_source import quaternion_rotation


class LiveMapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.map = self.directory/'map'
        self.bag = self.directory/'old_bag'
        self.map.mkdir()
        self.bag.mkdir()
        self.xyz = np.array([[2., 0., 1.], [10., 2., 0.], [18., -1., 3.],
                             [1000., 30., 0.]], np.float32)
        payload = np.column_stack((self.xyz, np.zeros(4, np.float32))).astype('<f4').tobytes()
        (self.map/'pointcloud_map.pcd').write_bytes(
            b'FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n'
            b'WIDTH 4\nHEIGHT 1\nPOINTS 4\nDATA binary\n'+payload)
        (self.map/'run_manifest.json').write_text(json.dumps({'bag': str(self.bag),
                                                            'timestamp_offset_seconds': 0.}))
        (self.map/'mapping_report.json').write_text(json.dumps({'map_frame': 'map',
            'pose_child_frame': 'os_sensor', 'failed_input': False, 'map_point_count': 4}))
        # Deliberately impossible CSV: hashing it is allowed; reading pose rows
        # would fail. The live mode must never acquire positions from this file.
        self.trajectory = self.map/'trajectory.csv'
        self.trajectory.write_bytes(b'not,a,trajectory\nINVALID\n')
        self.alignment = self.map/'gps_alignment'
        self.alignment.mkdir()
        self.enu_from_map = np.eye(4)
        self.enu_from_map[:3, :3] = quaternion_rotation([.1, -.08, .3, .9])
        self.enu_from_map[:3, 3] = [123., -456., 2.]
        self.write_alignment()
        self.wall = time.monotonic()

    def tearDown(self):
        self.temporary.cleanup()

    def write_alignment(self):
        data = {'source': {'bag': str(self.bag), 'map_dir': str(self.map),
                'trajectory_sha256': hashlib.sha256(self.trajectory.read_bytes()).hexdigest(),
                'pointcloud_sha256': hashlib.sha256((self.map/'pointcloud_map.pcd').read_bytes()).hexdigest()},
                'coordinate_conventions': {'source_frame': 'map', 'target_frame': 'enu',
                                           'pose_child_frame': 'os_sensor'},
                'transform': {'T_enu_from_map': self.enu_from_map.tolist()}}
        (self.alignment/'alignment.json').write_text(json.dumps(data))

    def add(self, source, stamp, position, quaternion=(0., 0., 0., 1.), **extra):
        return source.add_pose(stamp, position, quaternion, source_now_ns=stamp,
                               received_wall=self.wall, wall_now=self.wall, **extra)

    def test_map_load_never_reads_recorded_pose_values_and_crop_precedes_sampling(self):
        source = LiveMapSource(self.map, max_overview_points=2)
        self.assertEqual(source.pose_count, 0)
        self.assertEqual(source.point_count, 4)
        self.assertLessEqual(len(source.world_map_preview_xyz), 2)
        self.assertIsNone(source.pose_at(100_000_000_000, 100_000_000_000, self.wall)[0])
        sensor_pose = self.enu_from_map.copy()
        self.add(source, 100_000_000_000, sensor_pose[:3, 3], rotation_quaternion(sensor_pose[:3, :3]))
        pose, _ = source.pose_at(100_000_000_000, 100_000_000_000, self.wall)
        mapping = source.mapping_for_pose(pose, 80., 30., 2)
        self.assertEqual(mapping['visible_point_count'], 3)
        self.assertEqual(len(mapping['points']), 2)
        # Spatial cell ordering may differ from native PCD ordering; every
        # sampled return still comes from one of the three local points.
        for point in mapping['points']:
            self.assertLess(np.linalg.norm(self.xyz[:3]-point, axis=1).min(), 1e-5)
        np.testing.assert_allclose(mapping['world_position'], sensor_pose[:3, 3])
        self.assertTrue(mapping['georeferenced'])

    def test_pair_exact_integer_stamp_and_short_quaternion_bracket(self):
        source = LiveMapSource(self.map)
        start = 1_791_670_489_809_906_960
        self.add(source, start, [0., 0., 0.])
        self.add(source, start+100_000_000, [2., 0., 0.], [0., 0., -math.sqrt(.5), -math.sqrt(.5)])
        pose, _ = source.pose_at(start, start+100_000_000, self.wall)
        self.assertEqual(pose['pose_stamp_ns'], start)
        self.assertEqual(pose['pose_delta_ms'], 0.)
        pose, _ = source.pose_at(start+50_000_000, start+100_000_000, self.wall)
        np.testing.assert_allclose(pose['transform'][:3, 3], [1., 0., 0.])
        self.assertAlmostEqual(math.atan2(pose['transform'][1, 0], pose['transform'][0, 0]), math.pi/4)
        self.assertEqual(pose['bracket_ms'], 100.)
        near, _ = source.pose_at(start+100_500_000, start+100_500_000, self.wall)
        self.assertEqual(near['pose_delta_ms'], -.5)
        self.assertIsNone(source.pose_at(start-2_000_000, start, self.wall)[0])
        self.assertIsNone(source.pose_at(start+102_000_000, start+102_000_000, self.wall)[0])
        self.add(source, start+300_000_000, [5., 0., 0.])
        self.assertIsNone(source.pose_at(start+200_000_000, start+300_000_000, self.wall)[0])

    def test_freshness_checks_source_clock_and_paused_clock_wall_receipt(self):
        source = LiveMapSource(self.map)
        stamp = 100_000_000_000
        self.add(source, stamp, [1., 2., 3.])
        self.assertIsNotNone(source.pose_at(stamp, stamp, self.wall+1.)[0])
        self.assertIsNone(source.pose_at(stamp, stamp, self.wall+2.01)[0])
        self.assertIsNone(source.pose_at(stamp, stamp+1_000_000_001, self.wall)[0])
        self.assertIsNone(source.pose_at(stamp, stamp-150_000_001, self.wall)[0])
        accepted, _ = source.add_pose(stamp+100_000_000, [0., 0., 0.], [0., 0., 0., 1.],
            source_now_ns=stamp+2_000_000_000, received_wall=self.wall, wall_now=self.wall)
        self.assertFalse(accepted)
        accepted, _ = source.add_pose(stamp+200_000_000, [0., 0., 0.], [0., 0., 0., 1.],
            source_now_ns=stamp+200_000_000, received_wall=self.wall, wall_now=self.wall+3.)
        self.assertFalse(accepted)

    def test_frame_invalid_rotation_and_out_of_order_rejection_preserve_good_records(self):
        source = LiveMapSource(self.map)
        stamp = 100_000_000_000
        self.assertTrue(self.add(source, stamp, [1., 2., 3.])[0])
        for extra in ({'parent_frame': 'map'}, {'child_frame': 'os_lidar'}):
            self.assertFalse(self.add(source, stamp+100_000_000, [0., 0., 0.], **extra)[0])
        self.assertFalse(self.add(source, stamp+100_000_000, [0., 0., 0.], [0., 0., 0., 0.])[0])
        self.assertFalse(self.add(source, stamp+100_000_000, [0., 0., 0.], [0., float('nan'), 0., 1.])[0])
        self.assertFalse(self.add(source, stamp+100_000_000, [0., 0., 0.], [0., 1e308, 0., 1.])[0])
        self.assertFalse(self.add(source, stamp-1, [9., 9., 9.])[0])
        self.assertEqual(source.pose_count, 1)
        pose, _ = source.pose_at(stamp, stamp, self.wall)
        np.testing.assert_array_equal(pose['transform'][:3, 3], [1., 2., 3.])

    def test_bounded_live_history_invalidation_and_explicit_clock_rewind(self):
        source = LiveMapSource(self.map, max_pose_records=3)
        for index in range(10):
            self.add(source, 100_000_000_000+index*100_000_000, [index, 0., 0.])
        self.assertEqual(source.pose_count, 3)
        stamp = 100_900_000_000
        pose, _ = source.pose_at(stamp, stamp, self.wall)
        mapping = source.mapping_for_pose(pose, 80., 30., 100)
        self.assertEqual(len(mapping['world_path']), 3)
        self.assertTrue(source.observe_clock(stamp, reset_on_rewind=True)[0])
        self.assertTrue(source.observe_clock(stamp-1_000_000_000, reset_on_rewind=True)[0])
        self.assertEqual(source.pose_count, 0)
        self.assertIsNone(source.pose_at(stamp, stamp, self.wall)[0])
        with self.assertRaisesRegex(ValueError, 'invalidated'):
            source.mapping_for_pose(pose, 80., 30., 100)
        self.add(source, stamp, [10., 0., 0.])
        source.invalidate('estimator lost tracking')
        self.assertIn('lost tracking', source.pose_at(stamp, stamp, self.wall)[1])
        self.assertFalse(source.observe_clock(stamp-2_000_000_000, reset_on_rewind=False)[0])

    def test_box_world_corners_preserve_size_roll_pitch_and_source_schema(self):
        transform = np.eye(4)
        transform[:3, :3] = quaternion_rotation([.3, -.1, .5, .8])
        transform[:3, 3] = [7., 9., -2.]
        quaternion = np.array([-.1, .2, -.05, .9])
        source = {'class': 'car', 'score': .91, 'os_sensor_xyz_m': [10., -2., 1.],
                  'size_lwh_m': [4., 2., 1.5], 'yaw_rad': 99., 'quaternion_xyzw': quaternion.tolist()}
        result = transform_boxes([source], transform)[0]
        expected_center = transform[:3, :3]@np.array(source['os_sensor_xyz_m'])+transform[:3, 3]
        np.testing.assert_allclose(result['world_xyz_m'], expected_center)
        expected_rotation = transform[:3, :3]@quaternion_rotation(quaternion)
        np.testing.assert_allclose(quaternion_rotation(result['world_quaternion_xyzw']), expected_rotation, atol=1e-12)
        corners = np.asarray(result['corners_world_m'])
        np.testing.assert_allclose(corners.mean(axis=0), expected_center)
        self.assertAlmostEqual(np.linalg.norm(corners[1]-corners[0]), 4.)
        self.assertAlmostEqual(np.linalg.norm(corners[2]-corners[1]), 2.)
        self.assertAlmostEqual(np.linalg.norm(corners[4]-corners[0]), 1.5)
        self.assertEqual(result['class'], 'car')
        self.assertEqual(result['os_sensor_xyz_m'], source['os_sensor_xyz_m'])
        self.assertNotIn('world_xyz_m', source)
        for axis in np.eye(3):
            rotation = quaternion_rotation([*axis, 0.])
            np.testing.assert_allclose(quaternion_rotation(rotation_quaternion(rotation)), rotation, atol=1e-12)

    def test_alignment_changed_provenance_and_invalid_transform_rejected(self):
        marker = self.alignment/'.incomplete'
        marker.touch()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            LiveMapSource(self.map)
        marker.unlink()
        self.trajectory.write_bytes(b'changed source')
        with self.assertRaisesRegex(ValueError, 'trajectory checksum changed'):
            LiveMapSource(self.map)
        self.write_alignment()
        self.enu_from_map[0, 0] *= 2.
        self.write_alignment()
        with self.assertRaisesRegex(ValueError, 'rigid'):
            LiveMapSource(self.map)


if __name__ == '__main__':
    unittest.main()
