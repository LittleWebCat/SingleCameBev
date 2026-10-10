"""Controller checks run without ROS nodes, GUI, playback or hardware access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import run_localization_compare as controller


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='bev_localization_controller_')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.map = self.folder/'map'
        self.map.mkdir()
        self.bag = self.folder/'bag'
        self.bag.mkdir()
        self.output = self.folder/'output'
        (self.map/'pointcloud_map.pcd').write_bytes(b'fixture; controller does not load geometry')
        (self.map/'run_manifest.json').write_text(json.dumps({'bag': str(self.bag)}))
        (self.map/'mapping_report.json').write_text(json.dumps({
            'failed_input': False, 'map_frame': 'map', 'pose_child_frame': 'os_sensor'}))
        alignment = self.map/'gps_alignment'
        alignment.mkdir()
        (alignment/'alignment.json').write_text(json.dumps({'source': {'map_dir': str(self.map)}}))
        self.capture = Path(str(self.bag)+'.capture')
        self.capture.mkdir()
        (self.capture/'ouster_metadata.json').write_text(json.dumps({
            'lidar_intrinsics': {'lidar_to_sensor_transform': [0]*16},
            'imu_intrinsics': {'imu_to_sensor_transform': [0]*16}}))
        self.write_topics(controller.REQUIRED_TYPES)

    def write_topics(self, topics):
        metadata = {'rosbag2_bagfile_information': {
            'duration': {'nanoseconds': 60_000_000_000},
            'topics_with_message_count': [{'topic_metadata': {'name': name, 'type': kind},
                                          'message_count': 10} for name, kind in topics.items()]}}
        (self.bag/'metadata.yaml').write_text(yaml.safe_dump(metadata))

    def args(self, *options):
        return controller.parse_args(['--map', str(self.map), '--bag', str(self.bag),
                                      '--output', str(self.output), *options], ROOT)

    def test_replay_is_one_pass_isolated_and_has_gps_imu_and_latched_qos(self):
        args = self.args('--start-offset', '10', '--no-display')
        provenance = controller.prepare_inputs(args, ROOT)
        plan = controller.commands(args, ROOT)
        self.assertEqual(args.domain, 92)
        self.assertNotIn('--loop', plan['bag player'])
        self.assertEqual(plan['bag player'][plan['bag player'].index('--topics')+1:], controller.TOPICS)
        self.assertIn('--qos-profile-overrides-path', plan['bag player'])
        self.assertIn('static context', plan)
        self.assertIn('use_sim_time:=true', plan['localizer'])
        self.assertIn('--no-display', plan['map view'])
        self.assertTrue(provenance['same_recording_as_map'])
        self.assertFalse(provenance['saved_trajectory_used_for_localization'])
        self.assertFalse(provenance['accuracy_independently_validated'])
        self.assertNotIn('GPS bridge', plan)
        self.assertNotIn('LiDAR driver', plan)

    def test_rejects_bag_without_gps_instead_of_silent_partial_comparison(self):
        self.write_topics({name: kind for name, kind in controller.REQUIRED_TYPES.items()
                           if name != '/sensing/gnss/fix'})
        with self.assertRaisesRegex(ValueError, '/sensing/gnss/fix'):
            controller.prepare_inputs(self.args(), ROOT)

    def test_live_default_reuses_streams_without_hardware_start(self):
        args = controller.parse_args(['--live', '--map', str(self.map), '--output', str(self.output)])
        controller.prepare_inputs(args, ROOT)
        plan = controller.commands(args, ROOT)
        self.assertEqual(args.domain, 1)
        self.assertNotIn('bag player', plan)
        self.assertNotIn('GPS bridge', plan)
        self.assertNotIn('LiDAR driver', plan)
        self.assertIn('use_sim_time:=false', plan['localizer'])

    def test_hardware_opt_in_cannot_be_used_with_bag(self):
        with self.assertRaises(SystemExit) as error:
            self.args('--start-sensors')
        self.assertEqual(error.exception.code, 2)

    def test_existing_session_is_not_overwritten_even_for_dry_run(self):
        self.output.mkdir()
        marker = self.output/'keep.txt'
        marker.write_text('existing result')
        with self.assertRaises(FileExistsError):
            controller.prepare_inputs(self.args('--dry-run'), ROOT)
        self.assertEqual(marker.read_text(), 'existing result')

    def test_dry_run_creates_nothing_and_never_checks_or_starts_sensors(self):
        with patch.object(controller.subprocess, 'Popen') as popen, \
                patch.object(controller, 'matching_processes') as matches:
            code = controller.main(['--live', '--start-sensors', '--map', str(self.map),
                                    '--output', str(self.output), '--dry-run'])
        self.assertEqual(code, 0)
        popen.assert_not_called()
        matches.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_a_different_source_bag_is_allowed_and_marked_out_of_sample(self):
        manifest = self.map/'run_manifest.json'
        manifest.write_text(json.dumps({'bag': str(self.folder/'original_map_bag')}))
        provenance = controller.prepare_inputs(self.args(), ROOT)
        self.assertFalse(provenance['same_recording_as_map'])
        self.assertFalse(provenance['accuracy_independently_validated'])

    def test_seek_outside_bag_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            controller.prepare_inputs(self.args('--start-offset', '60'), ROOT)

    def test_unready_worker_times_out_and_does_not_allow_playback(self):
        ready = self.folder/'localizer_ready.json'
        process = SimpleNamespace(pid=123)
        with patch.object(controller.time, 'monotonic', side_effect=[0., 1., 2.]), \
                patch.object(controller.time, 'sleep'), \
                self.assertRaisesRegex(RuntimeError, 'did not become ready'):
            controller.wait_for_ready(ready, process, lambda: None, lambda: False, timeout=2.)
        ready.write_text(json.dumps({'ready': True, 'pid': 456}))
        with self.assertRaisesRegex(RuntimeError, 'different process'):
            controller.wait_for_ready(ready, process, lambda: None, lambda: False)


if __name__ == '__main__':
    unittest.main()
