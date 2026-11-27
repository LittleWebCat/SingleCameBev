"""Independent registration, failure-state and factory-frame checks, without ROS."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from lidar_map_localizer import (LocalizerConfig, MapLocalizer, gate_registration,
                                 freshness_reason, rigid_fit, sensor_from_metadata, SourceClockGuard)


def asymmetric_room():
    rng = np.random.default_rng(781)
    ground = np.column_stack((rng.uniform(-18, 23, 2500), rng.uniform(-21, 17, 2500),
                              np.full(2500, -1.8)))
    walls = []
    for x, y, axis in ((-13., 0, 0), (0, 17., 1), (23., 0, 0)):
        points = np.column_stack((rng.uniform(-18, 23, 1500), rng.uniform(-21, 17, 1500),
                                  rng.uniform(-1.5, 5.5, 1500)))
        points[:, axis] = x if axis == 0 else y
        walls.append(points)
    # Off-center asymmetric structure removes the square-room heading ambiguity.
    structure = rng.uniform([-7, -9, -.5], [-3, -6, 2.5], (1000, 3))
    return np.concatenate([ground, *walls, structure])


class RegistrationTests(unittest.TestCase):
    def test_rigid_fit_recovers_full_rotation_and_translation_without_reflection(self):
        rng = np.random.default_rng(35)
        source = rng.normal(size=(70, 3))
        rotation = Rotation.from_euler('xyz', [.2, -.15, .8]).as_matrix()
        target = source@rotation.T+[4, -6, 1]
        fitted = rigid_fit(source, target)
        np.testing.assert_allclose(fitted[:3, :3], rotation, atol=1e-10)
        np.testing.assert_allclose(fitted[:3, 3], [4, -6, 1], atol=1e-10)
        self.assertAlmostEqual(np.linalg.det(fitted[:3, :3]), 1.)

    def test_unknown_heading_registration_uses_geometry_and_independent_position_seed(self):
        target = asymmetric_room()
        truth = np.eye(4)
        truth[:3, :3] = Rotation.from_euler('z', .73).as_matrix()
        truth[:3, 3] = [2, -3, .25]
        source = (target-truth[:3, 3])@truth[:3, :3]
        config = LocalizerConfig(map_voxel_m=.35, scan_voxel_m=.35,
                                 initialization_yaw_step_degrees=30., maximum_scan_points=4000,
                                 map_crop_radius_m=50., maximum_iterations=22)
        estimator = MapLocalizer(target, config)
        result = estimator.localize(source, 100., np.array([2.8, -3.6, .4]))
        self.assertTrue(result['accepted'], result)
        np.testing.assert_allclose(result['transform'][:3, 3], truth[:3, 3], atol=.12)
        self.assertLess(Rotation.from_matrix(result['transform'][:3, :3]@truth[:3, :3].T).magnitude(), .03)
        self.assertFalse(result['saved_trajectory_used'])

    def test_missing_gps_rewind_and_outside_map_are_explicit_failures(self):
        estimator = MapLocalizer(asymmetric_room(), LocalizerConfig(map_crop_radius_m=45.))
        scan = asymmetric_room()
        self.assertEqual(estimator.localize(scan, 100.)['status'], 'waiting_for_fresh_gps')
        self.assertEqual(estimator.localize(scan, 101., np.array([500, 500, 0]))['status'], 'outside_map')
        estimator.last_stamp = 110.
        self.assertEqual(estimator.localize(scan, 105., np.zeros(3))['status'], 'source_clock_rewind')

    def test_gps_displacement_seeds_large_motion_after_bootstrap_but_geometry_refines_it(self):
        target = asymmetric_room()
        config = LocalizerConfig(map_voxel_m=.35, scan_voxel_m=.35,
                                 maximum_scan_points=4000, map_crop_radius_m=50.)
        estimator = MapLocalizer(target, config)
        first = estimator.localize(target, 100., np.array([.8, -.6, .15]), initial_yaw=0.)
        self.assertTrue(first['accepted'], first)
        # Simulate skipping one second of scans while cold bootstrap runs. No
        # saved poses or measured velocity are supplied to the new estimator.
        moved = target-np.array([8., 0, 0])
        second = estimator.localize(moved, 101.3, np.array([8.9, -.6, .15]))
        self.assertTrue(second['accepted'], second)
        self.assertGreater(second['gps_displacement_seed_m'], 7.)
        np.testing.assert_allclose(second['transform'][:3, 3], [8., 0, 0], atol=.15)

    def test_stale_future_and_old_wall_receipt_are_rejected(self):
        self.assertEqual(freshness_reason(100., 101., .1), 'future_scan')
        self.assertEqual(freshness_reason(100., 98., .1), 'stale_scan')
        self.assertEqual(freshness_reason(100., 100., 3.), 'stale_scan')
        self.assertIsNone(freshness_reason(100., 100.1, .1))  # native scan end can lead clock

    def test_out_of_order_cloud_with_forward_clock_does_not_reset_recorded_loop(self):
        guard = SourceClockGuard(allow_reset=True)
        guard.observe_clock(100_000_000_000)
        self.assertTrue(guard.accept_source(99_900_000_000))
        guard.observe_clock(100_100_000_000)
        self.assertFalse(guard.accept_source(99_800_000_000))
        self.assertEqual(guard.generation, 0)
        self.assertEqual(guard.rewinds, 0)
        # A confirmed ROS-clock rewind allows the actual next loop to bootstrap.
        guard.observe_clock(80_000_000_000)
        self.assertEqual(guard.generation, 1)
        self.assertTrue(guard.accept_source(79_900_000_000))

    def test_low_overlap_flat_ground_and_implausible_jump_cannot_publish_pose(self):
        config = LocalizerConfig()
        transform, prior = np.eye(4), np.eye(4)
        metrics = dict(fitness=.9, rmse_m=.1, structural_fitness=.9,
                       matched_points=1000, structural_points=0)
        self.assertEqual(gate_registration(metrics, transform, prior, .1, config), 'insufficient_structure')
        metrics['structural_points'] = 800
        metrics['fitness'] = .1
        self.assertEqual(gate_registration(metrics, transform, prior, .1, config), 'low_overlap')
        metrics['fitness'] = .9
        transform[0, 3] = 15
        self.assertEqual(gate_registration(metrics, transform, prior, .1, config), 'implausible_translation')

    def test_ouster_factory_sensor_rotation_and_mm_translation(self):
        value = [-1,0,0,0, 0,-1,0,0, 0,0,1,38.195, 0,0,0,1]
        transform = sensor_from_metadata({'lidar_intrinsics': {'lidar_to_sensor_transform': value}})
        np.testing.assert_allclose(transform[:3, :3], np.diag([-1,-1,1]))
        np.testing.assert_allclose(transform[:3, 3], [0,0,.038195])


if __name__ == '__main__':
    unittest.main()
