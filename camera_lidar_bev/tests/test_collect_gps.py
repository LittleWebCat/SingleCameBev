"""GPS collection gate and command regressions; no ROS nodes or sensor access."""
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import collect_camera_lidar as collector


def fix(status=0, latitude=42., longitude=-89., altitude=250., stamp=99):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id='gnss_antenna',
                               stamp=SimpleNamespace(sec=stamp, nanosec=0)),
        status=SimpleNamespace(status=status, service=1),
        latitude=latitude, longitude=longitude, altitude=altitude,
        position_covariance=[1., 0., 0., 0., 1., 0., 0., 0., 4.],
        position_covariance_type=3)


class GpsFixGateTests(unittest.TestCase):
    def observe(self, message, **options):
        return collector.gps_observation(message, 100_000_000_000, 9., 10., **options)

    def test_fresh_usable_fix_retains_covariance_frame_and_receiver_mode(self):
        status = {'solution_mode': 6, 'mode_name': 'SBAS', 'gnss_week': 2400, 'tow_ms': 1000}
        observation = self.observe(fix(), receiver_status=status, status_age_s=.2)
        self.assertTrue(observation['valid_fix'])
        self.assertEqual(observation['frame_id'], 'gnss_antenna')
        self.assertEqual(observation['position_covariance_m2'][8], 4.)
        self.assertEqual(observation['receiver_status'], status)
        self.assertEqual(observation['timestamp_source'], 'host_ros_receipt')
        self.assertFalse(observation['antenna_lever_arm_measured'])

    def test_no_fix_requires_explicit_override_and_has_standard_json(self):
        message = fix(status=-1, latitude=math.nan, longitude=math.nan, altitude=math.nan)
        with self.assertRaisesRegex(ValueError, 'no usable position fix'):
            self.observe(message)
        result = self.observe(message, allow_no_fix=True)
        self.assertFalse(result['valid_fix'])
        self.assertIsNone(result['latitude_deg'])
        json.dumps(result, allow_nan=False)

    def test_invalid_coordinates_are_not_accepted_as_a_valid_fix(self):
        for message in (fix(latitude=91), fix(longitude=-181), fix(altitude=math.nan)):
            with self.subTest(message=message), self.assertRaises(ValueError):
                self.observe(message)

    def test_diagnostic_override_never_accepts_stale_or_future_headers(self):
        for stamp in (0, 97, 101):
            with self.subTest(stamp=stamp), self.assertRaisesRegex(ValueError, 'stale'):
                self.observe(fix(stamp=stamp), allow_no_fix=True)
        with self.assertRaisesRegex(ValueError, 'stale'):
            collector.gps_observation(fix(), 100_000_000_000, 7., 10., allow_no_fix=True)


class GpsCaptureCommands(unittest.TestCase):
    def dry_run(self, *options):
        with tempfile.TemporaryDirectory(prefix='bev_gps_collect_test_') as folder:
            output = Path(folder)/'capture'
            result = subprocess.run(
                [sys.executable, str(ROOT/'tools/collect_camera_lidar.py'),
                 '--output', str(output), '--dry-run', *options],
                capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(Path(str(output)+'.capture').exists())
            self.assertFalse(Path(str(output)+'.session.json').exists())
            line = next(line for line in result.stdout.splitlines()
                        if 'record_sensor_session.py' in line)
            command = shlex.split(line)
            return result.stdout, command[command.index('--topics')+1:]

    def test_gps_dry_run_starts_bridge_and_records_all_selected_topics(self):
        text, topics = self.dry_run('--lidar-mode', '2048x10', '--gps')
        self.assertIn('septentrio_gps.py --config', text)
        self.assertEqual(topics, collector.capture_topics(True))
        self.assertTrue(all(topic in topics for topic in collector.GPS_TOPICS))

    def test_legacy_dry_run_keeps_original_topics_and_no_gps_bridge(self):
        original = list(collector.DEFAULT_TOPICS)
        collector.capture_topics(True)
        text, topics = self.dry_run()
        self.assertEqual(topics, original)
        self.assertEqual(collector.DEFAULT_TOPICS, original)
        self.assertNotIn('septentrio_gps.py', text)

    def test_no_fix_override_is_rejected_without_gps(self):
        result = subprocess.run(
            [sys.executable, str(ROOT/'tools/collect_camera_lidar.py'),
             '--dry-run', '--allow-no-gps-fix'], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 2)
        self.assertIn('requires --gps', result.stderr)


if __name__ == '__main__':
    unittest.main()
