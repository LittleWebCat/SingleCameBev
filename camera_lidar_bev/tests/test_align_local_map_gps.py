"""Frame direction, source-time pairing and rigid-map preservation without ROS nodes."""
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import align_local_map_gps as alignment


def curved_route(count=200):
    angle = np.linspace(0, 2.1 * math.pi, count)
    return np.column_stack((60 * np.cos(angle), 35 * np.sin(angle), np.linspace(-2, 3, count)))


def known_transform(yaw=.8, translation=(21., -31., 4.)):
    transform = np.eye(4)
    transform[:2, :2] = [[math.cos(yaw), -math.sin(yaw)],
                        [math.sin(yaw), math.cos(yaw)]]
    transform[:3, 3] = translation
    return transform


def poses(stamps):
    rows = np.zeros((len(stamps), 8))
    rows[:, 0] = stamps
    rows[:, 1] = np.asarray(stamps) * 2
    rows[:, 2] = np.asarray(stamps) * -3
    rows[:, 7] = 1
    return rows


def gps_rows(stamps):
    rows = np.zeros((len(stamps), 5))
    rows[:, 0] = stamps
    rows[:, 1:4] = [42., -89., 230.]
    rows[:, 4] = 4
    return rows


class GeographicFrameTests(unittest.TestCase):
    def test_equator_reference_and_local_axes_have_correct_direction_and_units(self):
        np.testing.assert_allclose(alignment.geodetic_to_ecef([[0, 0, 0]])[0],
                                   [6378137., 0, 0], atol=1e-9)
        origin = [0., 0., 100.]
        east_angle = math.degrees(1 / (6378137. + 100.))
        north_angle = math.degrees(1 / (6378137. * (1 - alignment.WGS84_F)**2 + 100.))
        points = [[0, east_angle, 100], [north_angle, 0, 100], [0, 0, 101]]
        np.testing.assert_allclose(alignment.geographic_to_enu(points, origin),
                                   np.eye(3), atol=1e-6)

    def test_antimeridian_crossing_is_short_and_nonfinite_geography_is_rejected(self):
        points = alignment.geographic_to_enu([[0., -179.9999, 0]], [0., 179.9999, 0])
        self.assertGreater(points[0, 0], 20.)
        self.assertLess(points[0, 0], 23.)
        for bad in ([[math.nan, 0, 0]], [[91., 0, 0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                alignment.geodetic_to_ecef(bad)


class SourceTimestampTests(unittest.TestCase):
    def test_saved_trajectory_drops_duplicate_stamps_but_rejects_corrupt_poses(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trajectory.csv'
            rows = poses(np.arange(100., 125.))
            rows[:, 7] = 2.  # Normalize orientation without changing translation.
            rows = np.insert(rows, 5, rows[5], axis=0)
            alignment.write_csv(path, alignment.POSE_COLUMNS, rows)
            cleaned, report = alignment.load_trajectory(path)
            self.assertEqual(report['duplicate_pose_timestamps_removed'], 1)
            self.assertEqual(len(cleaned), 25)
            np.testing.assert_allclose(cleaned[:, 7], 1.)
            path.unlink()
            rows[8, 1] = math.nan
            alignment.write_csv(path, alignment.POSE_COLUMNS, rows)
            with self.assertRaises(ValueError):
                alignment.load_trajectory(path)

    def test_pairs_interpolate_headers_and_never_bridge_missing_pose_intervals(self):
        trajectory = poses([100., 100.1, 100.2, 101., 101.1, 101.2])
        stamps = np.r_[np.linspace(100., 100.2, 25), 100.6, 101., 101.15, 99., 102.]
        gps = gps_rows(np.sort(stamps))
        selected, xyz, report = alignment.match_timestamps(trajectory, gps, .25)
        self.assertNotIn(100.6, gps[selected, 0])
        self.assertNotIn(99., gps[selected, 0])
        self.assertNotIn(102., gps[selected, 0])
        np.testing.assert_allclose(xyz[:, 0], gps[selected, 0] * 2)
        np.testing.assert_allclose(xyz[:, 1], gps[selected, 0] * -3)
        self.assertEqual(report['pose_gap_rejections'], 1)
        self.assertEqual(report['outside_trajectory_time_range'], 2)
        self.assertEqual(report['timestamp_offset_applied_s'], 0.)

    def test_duplicate_and_nonmonotonic_pair_inputs_are_rejected(self):
        for stamps in ([100., 100.], [100., 99.]):
            with self.subTest(stamps=stamps), self.assertRaisesRegex(ValueError, 'strictly increasing'):
                alignment.match_timestamps(poses(stamps), gps_rows([100., 101.]))

    def test_bad_fix_and_unknown_covariance_are_not_accepted_as_accuracy(self):
        message = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=100, nanosec=1)),
                                  status=SimpleNamespace(status=1), latitude=42., longitude=-89.,
                                  altitude=230., position_covariance=[0.] * 9,
                                  position_covariance_type=0)
        result = alignment.usable_fix(message)
        self.assertFalse(result[-1])
        self.assertIsNone(result[-2])
        self.assertEqual(result[2], 9.)
        message.status.status = -1
        self.assertIsNone(alignment.usable_fix(message))
        message.status.status = 0
        message.latitude = math.nan
        self.assertIsNone(alignment.usable_fix(message))


class RigidMapTests(unittest.TestCase):
    def test_different_bag_or_invalid_mapping_report_is_rejected_before_gps_read(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            bag = directory / 'source_bag'
            bag.mkdir()
            (bag / 'metadata.yaml').write_text('unused fixture metadata')
            (directory / 'run_manifest.json').write_text(json.dumps({'bag': str(bag)}))
            valid_report = {'map_frame': 'map', 'pose_child_frame': 'os_sensor',
                            'failed_input': False}
            report_path = directory / 'mapping_report.json'
            report_path.write_text(json.dumps(valid_report))
            with patch.object(alignment, 'read_gps') as gps_reader:
                with self.assertRaisesRegex(ValueError, 'source bag'):
                    alignment.align_map(directory, bag=directory / 'different_bag')
                for key, value in (('map_frame', 'odom'), ('pose_child_frame', 'lio_imu'),
                                   ('failed_input', True), ('failed_input', None)):
                    report_path.write_text(json.dumps(dict(valid_report, **{key: value})))
                    with self.subTest(key=key, value=value), self.assertRaisesRegex(
                            ValueError, 'mapping_report'):
                        alignment.align_map(directory)
                gps_reader.assert_not_called()
            self.assertFalse((directory / 'gps_alignment').exists())

    def test_existing_output_is_refused_before_any_gps_read_or_map_write(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / 'run_manifest.json').write_text('{"bag":"/no/such/bag"}')
            output = directory / 'gps_alignment'
            output.mkdir()
            sentinel = output / 'alignment.json'
            sentinel.write_text('existing result')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                alignment.align_map(directory)
            self.assertEqual(sentinel.read_text(), 'existing result')

    def test_recovers_direction_and_translation_without_scale_or_z_distortion(self):
        local = curved_route()
        expected = known_transform()
        gps = local @ expected[:3, :3].T + expected[:3, 3]
        transform, inliers, _, report = alignment.fit_planar_rigid(local, gps, np.full(len(local), 4.))
        np.testing.assert_allclose(transform, expected, atol=1e-10)
        self.assertTrue(inliers.all())
        self.assertEqual(report['scale'], 1.)
        self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.)
        np.testing.assert_allclose(np.diff((local @ transform[:3, :3].T)[:, 2]),
                                   np.diff(local[:, 2]), atol=1e-10)

    def test_sparse_large_gps_outliers_do_not_move_the_map(self):
        local = curved_route()
        expected = known_transform()
        gps = local @ expected[:3, :3].T + expected[:3, 3]
        bad = np.arange(0, len(local), 17)
        gps[bad, :2] += [150., -120.]
        transform, inliers, _, report = alignment.fit_planar_rigid(local, gps, np.full(len(local), 4.))
        np.testing.assert_allclose(transform, expected, atol=1e-7)
        self.assertFalse(inliers[bad].any())
        self.assertEqual(report['outliers'], len(bad))

    def test_reflection_and_stationary_trajectory_fail_instead_of_inventing_heading(self):
        local = curved_route()
        reflected = local * [-1., 1., 1.]
        with self.assertRaisesRegex(ValueError, 'reflection'):
            alignment.fit_planar_rigid(local, reflected, np.ones(len(local)))
        quiet = np.zeros((25, 3))
        with self.assertRaisesRegex(ValueError, '10 m'):
            alignment.fit_planar_rigid(quiet, quiet, np.ones(25))

    def test_orientation_is_left_multiplied_and_source_timestamps_are_preserved(self):
        local = poses(np.arange(100., 103.))
        # Nonzero roll makes multiplication order observable.
        local[:, 4] = math.sin(.3 / 2)
        local[:, 7] = math.cos(.3 / 2)
        transform = known_transform(math.pi / 2)
        result = alignment.transform_trajectory(local, transform)
        np.testing.assert_array_equal(result[:, 0], local[:, 0])
        np.testing.assert_allclose(result[:, 4:8], np.tile([
            math.sin(.3 / 2) / math.sqrt(2), math.sin(.3 / 2) / math.sqrt(2),
            math.cos(.3 / 2) / math.sqrt(2), math.cos(.3 / 2) / math.sqrt(2)], (3, 1)))

    def test_binary_pcd_chunks_preserve_point_count_intensity_and_original_file(self):
        with tempfile.TemporaryDirectory() as folder:
            source, destination = Path(folder) / 'original.pcd', Path(folder) / 'aligned.pcd'
            original = np.array([[1., 2., 3., 17.], [4., 5., 6., 29.], [-3., 2., 1., 31.]], dtype='<f4')
            header = ('VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
                      'COUNT 1 1 1 1\nWIDTH 3\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
                      'POINTS 3\nDATA binary\n').encode()
            payload = header + original.tobytes()
            source.write_bytes(payload)
            transform = known_transform(math.pi / 2)
            report = alignment.transform_pcd(source, destination, transform, chunk_points=2)
            self.assertEqual(source.read_bytes(), payload)
            aligned = np.frombuffer(destination.read_bytes().split(b'DATA binary\n', 1)[1],
                                    dtype='<f4').reshape(-1, 4)
            np.testing.assert_allclose(aligned[:, :3], original[:, :3] @ transform[:3, :3].T
                                       + transform[:3, 3], atol=1e-6)
            np.testing.assert_array_equal(aligned[:, 3], original[:, 3])
            self.assertEqual(report['points'], 3)
            with self.assertRaises(FileExistsError):
                alignment.transform_pcd(source, destination, transform)


if __name__ == '__main__':
    unittest.main()
