import io
import json
import unittest

from bev_depth.io import write_jsonl_record
from bev_depth.types import Detection, ProjectedDetection


class JsonlTest(unittest.TestCase):
    def test_record_schema(self):
        handle = io.StringIO()
        detection = Detection("car", 0.95, (1.0, 2.0, 3.0, 4.0))
        projected = ProjectedDetection(detection, 12.0, 0.5, 12.0, (42, 100), "depth_map")
        write_jsonl_record(
            handle,
            7,
            0.25,
            [projected],
            wall_time_ns=123456789,
            monotonic_ns=987654321,
            sensor_session_id="session-1",
            sensor_elapsed_ns=1234,
        )
        record = json.loads(handle.getvalue())
        self.assertEqual(record["frame_id"], 7)
        self.assertEqual(record["wall_time_ns"], 123456789)
        self.assertEqual(record["monotonic_ns"], 987654321)
        self.assertEqual(record["sensor_session_id"], "session-1")
        self.assertEqual(record["sensor_elapsed_ns"], 1234)
        self.assertEqual(record["objects"][0]["class_name"], "car")
        self.assertEqual(record["objects"][0]["bev_xy"], [42, 100])
        self.assertEqual(record["objects"][0]["depth_source"], "depth_map")


if __name__ == "__main__":
    unittest.main()
