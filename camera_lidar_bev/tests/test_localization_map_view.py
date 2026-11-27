"""Independent geometry, invalid-fix and age-gate checks for the map view."""
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from localization_map_view import (LocalizationMapRenderer, ellipse_geometry, freshness,
                                    gps_observation, lidar_observation, display_number)


def gps(latitude=0., longitude=0., altitude=0., covariance=None, known=True, status=1):
    return NS(header=NS(stamp=NS(sec=100, nanosec=0)), latitude=latitude,
              longitude=longitude, altitude=altitude, status=NS(status=status),
              position_covariance=(np.eye(3) if covariance is None else covariance).ravel(),
              position_covariance_type=3 if known else 0)


def pose(frame='local_enu', child='os_sensor', position=(1., 2., 3.), yaw=0.):
    return NS(header=NS(stamp=NS(sec=100, nanosec=0), frame_id=frame), child_frame_id=child,
              pose=NS(pose=NS(position=NS(x=position[0], y=position[1], z=position[2]),
                             orientation=NS(x=0., y=0., z=math.sin(yaw/2), w=math.cos(yaw/2)))))


class MapViewTests(unittest.TestCase):
    def test_numeric_diagnostics_are_readable_and_reject_nonfinite_values(self):
        self.assertEqual(display_number(132.293564, 1), '132.3')
        self.assertEqual(display_number(.999714, 3), '1.000')
        for value in (None, math.nan, math.inf, 'invalid'):
            self.assertEqual(display_number(value, 1), 'unknown')

    def test_gps_axes_and_no_invented_heading(self):
        origin = np.array([0., 0., 0.])
        east, _ = gps_observation(gps(longitude=.0001), origin)
        north, _ = gps_observation(gps(latitude=.0001), origin)
        self.assertAlmostEqual(east['position'][0], 11.131949079, places=5)
        self.assertAlmostEqual(north['position'][1], 11.05742758, places=5)
        self.assertAlmostEqual(east['position'][1], 0., places=6)
        self.assertAlmostEqual(north['position'][0], 0., places=6)
        self.assertIsNone(east['heading'])
        self.assertEqual(east['stamp_ns'], 100_000_000_000)
        message = gps()
        message.header.stamp.sec, message.header.stamp.nanosec = 1791670489, 809906960
        self.assertEqual(gps_observation(message, origin)[0]['stamp_ns'], 1791670489809906960)

    def test_no_fix_and_unknown_covariance_do_not_invent_uncertainty(self):
        origin = np.zeros(3)
        self.assertIsNone(gps_observation(gps(status=-1), origin)[0])
        self.assertIsNone(gps_observation(gps(latitude=91.), origin)[0])
        self.assertIsNone(gps_observation(gps(altitude=math.nan), origin)[0])
        observation, _ = gps_observation(gps(known=False), origin)
        self.assertIsNone(observation['covariance'])
        covariance = np.diag([-1., 1., 1.])
        self.assertIsNone(gps_observation(gps(covariance=covariance), origin)[0]['covariance'])

    def test_rotated_covariance_95_percent_ellipse(self):
        angle = .63
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        covariance = rotation@np.diag([9., 1.])@rotation.T
        axes, direction = ellipse_geometry(covariance)
        np.testing.assert_allclose(axes, np.sqrt(5.991464547)*np.array([3., 1.]))
        self.assertAlmostEqual(abs(math.cos(direction-angle)), 1., places=6)
        with self.assertRaises(ValueError):
            ellipse_geometry(np.diag([-1., 1.]))

    def test_paused_replay_and_future_or_stale_source_are_hidden(self):
        observation = {'stamp_ns': 100_000_000_000, 'received_wall': 7.}
        self.assertTrue(freshness(observation, 100_100_000_000, 7.1)[0])
        self.assertFalse(freshness(observation, 100_100_000_000, 9.1)[0])
        self.assertFalse(freshness(observation, 101_100_000_000, 7.1)[0])
        self.assertFalse(freshness(observation, 99_900_000_000, 7.1)[0])
        self.assertFalse(freshness(observation, 0, 7.1)[0])
        self.assertFalse(freshness(None, 100_100_000_000, 7.1)[0])

    def test_local_map_pose_to_enu_preserves_size_and_heading(self):
        transform = np.eye(4)
        transform[:2, :2] = [[0., -1.], [1., 0.]]
        transform[:3, 3] = [20., 30., 4.]
        observation, _ = lidar_observation(pose(frame='map', yaw=.2), transform)
        np.testing.assert_allclose(observation['position'], [18., 31., 7.])
        self.assertAlmostEqual(observation['heading'], math.pi/2+.2)
        self.assertIsNone(lidar_observation(pose(frame='unexpected'), transform)[0])
        self.assertIsNone(lidar_observation(pose(child='base_link'), transform)[0])

    def test_north_is_up_and_east_is_right_on_equal_scale(self):
        renderer = LocalizationMapRenderer(np.array([[-20., -10., 0.], [20., 10., 0.]]))
        center, east, north = renderer.pixels([[0., 0.], [10., 0.], [0., 10.]])
        self.assertGreater(east[0], center[0])
        self.assertLess(north[1], center[1])
        self.assertEqual(east[0]-center[0], center[1]-north[1])
        self.assertEqual(renderer.render().shape, (900, 1400, 3))


if __name__ == '__main__':
    unittest.main()
