"""Keep recorded loss evidence separate from inferred cadence gaps."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from check_camera_lidar_bag import recorder_cache_losses


class RecorderCacheLossTests(unittest.TestCase):
    def test_repeated_split_reports_are_cumulative_not_additive(self):
        block='[WARN] Cache buffers lost messages per topic: \n\t/points: 16\n\t/camera/front/image_raw: 23\nTotal lost: 39\n'
        result=recorder_cache_losses(block*11)
        self.assertEqual(result['cumulative_reports_seen'],11)
        self.assertEqual(result['reported_total_max'],39)
        self.assertEqual(result['per_topic_cumulative_max'],{'/points':16,'/camera/front/image_raw':23})
        self.assertTrue(result['has_confirmed_recorder_cache_loss'])
        self.assertFalse(result['is_sensor_drop_count'])

    def test_new_losses_raise_the_cumulative_maximum(self):
        text=('Cache buffers lost messages per topic:\n /points: 2\nTotal lost: 2\n'
              'Cache buffers lost messages per topic:\n /points: 5\n /imu: 3\nTotal lost: 8\n')
        result=recorder_cache_losses(text)
        self.assertEqual(result['reported_total_max'],8)
        self.assertEqual(result['per_topic_cumulative_max'],{'/points':5,'/imu':3})

    def test_empty_log_does_not_invent_loss(self):
        result=recorder_cache_losses('[INFO] Recording stopped\n')
        self.assertFalse(result['has_confirmed_recorder_cache_loss'])
        self.assertEqual(result['cumulative_reports_seen'],0)


if __name__=='__main__':
    unittest.main()
