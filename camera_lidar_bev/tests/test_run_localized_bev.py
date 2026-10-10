"""Focused ownership/startup checks without ROS, sensors, inference or GUI."""

import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import run_localized_bev as runner
import start_bev


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='bev_localized_runner_')
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.map = self.folder/'map'
        self.map.mkdir()
        self.bag = self.folder/'map_capture_bag'
        self.output = self.folder/'session'
        (self.map/'pointcloud_map.pcd').write_bytes(b'fixture; no map geometry is loaded')
        (self.map/'run_manifest.json').write_text(json.dumps({'bag': str(self.bag)}))
        (self.map/'mapping_report.json').write_text(json.dumps({
            'failed_input': False, 'map_frame': 'map', 'pose_child_frame': 'os_sensor'}))
        alignment = self.map/'gps_alignment'
        alignment.mkdir()
        (alignment/'alignment.json').write_text(json.dumps({'source': {'map_dir': str(self.map)}}))
        capture = Path(str(self.bag)+'.capture')
        capture.mkdir()
        (capture/'ouster_metadata.json').write_text(json.dumps({
            'lidar_intrinsics': {'lidar_to_sensor_transform': [0]*16},
            'imu_intrinsics': {'imu_to_sensor_transform': [0]*16}}))

    def options(self, *extras):
        return ['--map', str(self.map), '--output', str(self.output), *extras]

    def test_live_plan_has_native_10hz_lidar_and_no_recording_or_road_model(self):
        args = runner.parse_args(self.options(), ROOT)
        provenance = runner.prepare_inputs(args, ROOT)
        plan = runner.commands(args, ROOT)
        self.assertEqual(args.domain, 1)
        self.assertIn('--live', plan['localization'])
        self.assertIn('--start-sensors', plan['localization'])
        self.assertIn('--no-display', plan['localization'])
        self.assertNotIn('--bag', plan['localization'])
        self.assertEqual(plan['combined_view'][plan['combined_view'].index('--lidar-mode')+1], '2048x10')
        self.assertIn('--localize-map', plan['combined_view'])
        self.assertIn('--no-road-lanes', plan['combined_view'])
        self.assertFalse(provenance['saved_trajectory_used_for_localization'])
        self.assertEqual(args.max_rate, 10.)
        self.assertEqual(args.range_m, 80.)
        self.assertFalse(self.output.exists())

    def test_dry_run_writes_nothing_and_never_probes_hardware_or_models(self):
        with patch.object(runner.subprocess, 'Popen') as popen, \
                patch.object(runner, 'matching_processes') as matches, \
                patch.object(runner, 'preflight') as preflight, patch('sys.stdout', io.StringIO()):
            result = runner.main(self.options('--dry-run'))
        self.assertEqual(result, 0)
        popen.assert_not_called()
        matches.assert_not_called()
        preflight.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_session_is_never_overwritten(self):
        self.output.mkdir()
        marker = self.output/'keep'
        marker.write_text('existing')
        args = runner.parse_args(self.options('--dry-run'), ROOT)
        with self.assertRaises(FileExistsError):
            runner.prepare_inputs(args, ROOT)
        self.assertEqual(marker.read_text(), 'existing')

    def test_incomplete_alignment_is_refused_before_startup(self):
        (self.map/'gps_alignment/.incomplete').touch()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            runner.prepare_inputs(runner.parse_args(self.options(), ROOT), ROOT)

    def test_ready_marker_must_belong_to_the_owned_controller(self):
        ready = self.folder/'ready.json'
        ready.write_text(json.dumps({'ready': True, 'pid': os.getpid()}))
        owned = SimpleNamespace(pid=os.getppid(), poll=lambda: None)
        self.assertTrue(runner.wait_localizer_ready(ready, owned, lambda: False)['ready'])
        unrelated = SimpleNamespace(pid=os.getppid()+100000, poll=lambda: None)
        with self.assertRaisesRegex(RuntimeError, 'different controller'):
            runner.wait_localizer_ready(ready, unrelated, lambda: False)

    def test_unready_or_exited_controller_does_not_allow_viewer_start(self):
        owned = SimpleNamespace(pid=999999, poll=lambda: None)
        with patch.object(runner.time, 'monotonic', side_effect=[0., 2.]), \
                self.assertRaisesRegex(RuntimeError, 'did not become ready'):
            runner.wait_localizer_ready(self.folder/'missing', owned, lambda: False, timeout=1.)
        exited = SimpleNamespace(pid=999999, poll=lambda: 1, returncode=1)
        with self.assertRaisesRegex(RuntimeError, 'controller exited'):
            runner.wait_localizer_ready(self.folder/'missing', exited, lambda: False)

    def test_viewer_close_stops_only_the_two_owned_controllers(self):
        localizer = SimpleNamespace(pid=100, poll=lambda: None, returncode=None)
        viewer = SimpleNamespace(pid=101, poll=lambda: 0, returncode=0)
        with patch.object(runner, 'preflight') as preflight, \
                patch.object(runner, 'matching_processes', return_value=[]), \
                patch.object(runner, 'lock_session', return_value=Mock()), \
                patch.object(runner, 'wait_localizer_ready', return_value={'ready': True}) as ready, \
                patch.object(runner.subprocess, 'Popen', side_effect=[localizer, viewer]) as popen, \
                patch.object(runner, 'stop_owned') as stop, \
                patch.object(runner.signal, 'signal'), patch('sys.stdout', io.StringIO()):
            result = runner.main(self.options())
        self.assertEqual(result, 0)
        self.assertEqual(popen.call_count, 2)
        ready.assert_called_once()
        preflight.assert_called_once_with(ROOT, combined_view=True, road_lanes=False)
        stop.assert_called_once_with([('localization', localizer), ('combined_view', viewer)], grace_s=35.)
        report = json.loads((self.output/'session_report.json').read_text())
        self.assertEqual(report['outcome'], 'viewer_closed')
        self.assertFalse(report['reused_processes_stopped'])
        self.assertFalse(report['recorder_started'])

    def test_failed_readiness_stops_localization_without_starting_display(self):
        owned = SimpleNamespace(pid=100, poll=lambda: None, returncode=None)
        with patch.object(runner, 'preflight'), \
                patch.object(runner, 'matching_processes', return_value=[]), \
                patch.object(runner, 'lock_session', return_value=Mock()), \
                patch.object(runner, 'wait_localizer_ready', side_effect=RuntimeError('not ready')), \
                patch.object(runner.subprocess, 'Popen', return_value=owned) as popen, \
                patch.object(runner, 'stop_owned') as stop, patch.object(runner.signal, 'signal'), \
                patch('sys.stdout', io.StringIO()), patch('sys.stderr', io.StringIO()):
            result = runner.main(self.options())
        self.assertEqual(result, 1)
        popen.assert_called_once()
        stop.assert_called_once_with([('localization', owned)], grace_s=35.)

    def test_start_bev_reuses_a_starting_driver_and_omits_road_inference(self):
        def matches(name, domain):
            return [123] if name == 'start_lidar.py' else []
        with patch.object(start_bev, 'preflight'), \
                patch.object(start_bev, 'matching_processes', side_effect=matches), \
                patch.object(start_bev.subprocess, 'Popen') as popen, \
                patch('sys.stdout', io.StringIO()) as output:
            code = start_bev.main(['--combined-view', '--localize-map', str(self.map),
                                   '--lidar-mode', '2048x10', '--no-road-lanes', '--dry-run'])
        self.assertEqual(code, 0)
        popen.assert_not_called()
        self.assertIn('Reuse os_driver PID 123', output.getvalue())
        self.assertNotIn('Would start os_driver', output.getvalue())
        self.assertNotIn('road_lane_detector.py', output.getvalue())
        self.assertIn('--localize-map', output.getvalue())


if __name__ == '__main__':
    unittest.main()
