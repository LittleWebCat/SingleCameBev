import base64
import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from bev_depth.sensors import (
    RecorderSnapshot,
    RecordingSession,
    SensorRecordControls,
    SerialRecorder,
)


class FakeSerial:
    def __init__(self, payload: bytes, line_mode: bool):
        self.payload = payload
        self.line_mode = line_mode
        self.closed = False
        self.writes = []
        self.reset_count = 0

    @property
    def in_waiting(self):
        return len(self.payload)

    def readline(self):
        return self._next_payload()

    def read(self, size):
        del size
        return self._next_payload()

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)

    def flush(self):
        pass

    def reset_input_buffer(self):
        self.reset_count += 1

    def close(self):
        self.closed = True

    def _next_payload(self):
        if self.payload:
            payload, self.payload = self.payload, b""
            return payload
        time.sleep(0.002)
        return b""


def wait_for_record(recorder):
    deadline = time.monotonic() + 1.0
    while recorder.snapshot().records == 0 and time.monotonic() < deadline:
        time.sleep(0.005)


class SerialRecorderTest(unittest.TestCase):
    def test_records_timestamped_gps_nmea(self):
        device = FakeSerial(b"$GPGGA,example*00\r\n", line_mode=True)
        session = RecordingSession("aligned-session", 100, 200)
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SerialRecorder(
                "gps",
                temp_dir,
                port="/dev/fake-gps",
                serial_factory=lambda **kwargs: device,
            )
            self.assertTrue(recorder.start(session))
            wait_for_record(recorder)
            recorder.stop()
            snapshot = recorder.snapshot()
            record = json.loads(snapshot.output_path.read_text())

        self.assertEqual(record["nmea"], "$GPGGA,example*00")
        self.assertEqual(record["session_id"], "aligned-session")
        self.assertEqual(record["session_start_wall_time_ns"], 100)
        self.assertEqual(record["session_start_monotonic_ns"], 200)
        self.assertEqual(record["elapsed_ns"], record["monotonic_ns"] - 200)
        self.assertGreater(record["wall_time_ns"], 0)
        self.assertGreater(record["monotonic_ns"], 0)
        self.assertEqual(snapshot.bytes_written, 19)
        self.assertTrue(device.closed)
        self.assertEqual(device.reset_count, 1)

    def test_preserves_binary_imu_chunks_as_base64(self):
        payload = b"\xfa\xff\x36\x04\x01\x02\x03\x04"
        device = FakeSerial(payload, line_mode=False)
        session = RecordingSession("aligned-session", 100, 200)
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = SerialRecorder(
                "imu",
                temp_dir,
                port="/dev/fake-imu",
                serial_factory=lambda **kwargs: device,
            )
            self.assertTrue(recorder.start(session))
            wait_for_record(recorder)
            recorder.stop()
            record = json.loads(recorder.snapshot().output_path.read_text())

        self.assertEqual(base64.b64decode(record["xbus"]), payload)
        self.assertEqual(record["encoding"], "base64")
        self.assertEqual(record["session_id"], "aligned-session")


class FakeRecorder:
    def __init__(self, kind, output_dir, should_start=True):
        self.kind = kind
        self.output_dir = Path(output_dir)
        self.should_start = should_start
        self.state = "idle"
        self.start_sessions = []
        self.stop_count = 0
        self.output_path = None
        self.error = None

    def snapshot(self):
        return RecorderSnapshot(self.state, None, 3, 30, self.output_path, self.error)

    def start(self, session):
        self.start_sessions.append(session)
        if not self.should_start:
            self.state = "error"
            self.error = f"{self.kind} unavailable"
            return False
        self.state = "recording"
        self.output_path = self.output_dir / f"{self.kind}_{session.session_id}.jsonl"
        return True

    def stop(self):
        self.stop_count += 1
        if self.state == "recording":
            self.state = "idle"


class SensorControlsTest(unittest.TestCase):
    def test_one_button_starts_and_stops_one_aligned_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gps = FakeRecorder("gps", temp_dir)
            imu = FakeRecorder("imu", temp_dir)
            controls = SensorRecordControls(gps, imu)
            frame = np.zeros((800, 400, 3), dtype=np.uint8)
            controls.draw(frame)

            controls.mouse_callback(cv2.EVENT_LBUTTONUP, 20, 20, 0, None)
            session = controls.active_session()
            self.assertIsNotNone(session)
            self.assertIs(gps.start_sessions[0], session)
            self.assertIs(imu.start_sessions[0], session)

            controls.mouse_callback(cv2.EVENT_LBUTTONUP, 20, 20, 0, None)
            manifest_path = Path(temp_dir) / f"session_{session.session_id}.json"
            manifest = json.loads(manifest_path.read_text())

        self.assertIsNone(controls.active_session())
        self.assertEqual(gps.stop_count, 1)
        self.assertEqual(imu.stop_count, 1)
        self.assertEqual(manifest["session_id"], session.session_id)
        self.assertEqual(manifest["gps_log"], f"gps_{session.session_id}.jsonl")
        self.assertEqual(manifest["imu_log"], f"imu_{session.session_id}.jsonl")

    def test_start_is_atomic_when_one_sensor_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            gps = FakeRecorder("gps", temp_dir)
            imu = FakeRecorder("imu", temp_dir, should_start=False)
            controls = SensorRecordControls(gps, imu)

            self.assertFalse(controls.start_session())

        self.assertIsNone(controls.active_session())
        self.assertEqual(gps.state, "idle")
        self.assertIn("imu unavailable", controls.last_message)


if __name__ == "__main__":
    unittest.main()
