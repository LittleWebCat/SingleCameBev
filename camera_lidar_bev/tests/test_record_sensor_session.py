"""Exercise recorder shutdown/reporting with a fake CLI; no ROS or sensors."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FAKE_ROS = '''#!/usr/bin/python3
import os, signal, sys, time
from pathlib import Path
bag = Path(sys.argv[sys.argv.index("--output")+1])
bag.mkdir()
def stop(signum, frame):
    (bag / "metadata.yaml").write_text("fake_recorder_fixture: true\\n")
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
if os.environ.get("FAKE_EARLY_EXIT"):
    sys.exit(3)
while True:
    time.sleep(.01)
'''


class RecorderLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        fake = self.folder/'ros2'
        fake.write_text(FAKE_ROS)
        fake.chmod(0o700)
        self.environment = dict(os.environ, PATH=str(self.folder)+os.pathsep+os.environ.get('PATH',''), AMENT_PREFIX_PATH='')

    def tearDown(self):
        self.temp.cleanup()

    def command(self, output, *extra):
        return [sys.executable,str(ROOT/'tools/record_sensor_session.py'),
                '--output',str(output),'--duration','.25',*extra]

    def test_graceful_bounded_stop_finalizes_metadata_and_records_status(self):
        output = self.folder/'complete'
        result = subprocess.run(self.command(output),env=self.environment,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr+result.stdout)
        report = json.loads(Path(str(output)+'.session.json').read_text())
        self.assertEqual(report['status'],'completed')
        self.assertEqual(report['stop_method'],'sigint')
        self.assertTrue(report['bag_metadata_present'])
        self.assertEqual(report['recorder_returncode'],0)

    def test_disk_floor_stops_cleanly_and_reports_partial_capture(self):
        output = self.folder/'disk_floor'
        result = subprocess.run(self.command(output,'--min-free-gib','1000000'),env=self.environment,
                                capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0)
        report = json.loads(Path(str(output)+'.session.json').read_text())
        self.assertEqual(report['status'],'disk_space_low')
        self.assertIn(report['stop_method'],('sigint','already_exited'))

    def test_early_failure_is_not_a_completed_sensor_recording(self):
        output = self.folder/'early'
        environment = dict(self.environment,FAKE_EARLY_EXIT='1')
        result = subprocess.run(self.command(output),env=environment,capture_output=True,text=True,timeout=10)
        self.assertNotEqual(result.returncode,0)
        report = json.loads(Path(str(output)+'.session.json').read_text())
        self.assertEqual(report['status'],'recorder_exited_early')
        self.assertFalse(report['bag_metadata_present'])

    def test_dry_run_preserves_all_paths_and_reports_mcap_compression(self):
        marker = self.folder/'ament/share/ament_index/resource_index/packages/rosbag2_storage_mcap'
        marker.parent.mkdir(parents=True)
        marker.touch()
        environment = dict(self.environment,AMENT_PREFIX_PATH=str(self.folder/'ament'))
        output = self.folder/'dry'
        result = subprocess.run(self.command(output,'--dry-run','--storage-preset-profile','zstd_fast'),
                                env=environment,capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('zstd_fast',result.stdout)
        self.assertFalse(output.exists())
        self.assertFalse(Path(str(output)+'.session.json').exists())

    def test_existing_bag_and_report_are_never_overwritten(self):
        output = self.folder/'existing'
        output.mkdir()
        sentinel = output/'keep'
        sentinel.write_text('existing sensor data')
        result = subprocess.run(self.command(output),env=self.environment,capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,2)
        self.assertEqual(sentinel.read_text(),'existing sensor data')


if __name__=='__main__':
    unittest.main()
