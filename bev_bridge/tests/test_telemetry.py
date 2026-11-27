"""Response-monitor and 0x741 telemetry tests. No ROS, no hardware -- runs on any laptop.

    python3 -m unittest discover -s bev_bridge/tests -p 'test_telemetry.py' -v

tests/test_bridge.py imports the rclpy node, so it only runs on the Jetson. These cover the
parts that decide whether the car gets stopped, so they must be runnable where the code is
edited. The firmware line shape is dbw_esp_full's 0x741 echo via adapter_esp:

    T,<ms>,<angle_deg>,<mph>,<engaged>,<setpoint_deg>,<blinkL>,<blinkR>
"""
import contextlib
import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge_core import ResponseMonitor
from dbw_serial import DBW


class FakeSerial:
    """Feeds bytes to DBW.drain() the way a real port does -- including split lines."""
    def __init__(self, chunks=()):
        self.chunks = list(chunks)

    def read(self, _n):
        return self.chunks.pop(0) if self.chunks else b''


def link(chunks=()):
    """A DBW with its transport replaced; __init__ is skipped (it needs a real device)."""
    dbw = object.__new__(DBW)
    dbw.ser, dbw.failed, dbw._rx = FakeSerial(chunks), False, ''
    dbw.meas_deg = dbw.meas_mph = dbw.fw_sp_deg = dbw.meas_mono = None
    dbw.engaged, dbw.telem_lines, dbw.telem_bad = False, 0, 0
    return dbw


class TelemetryParsing(unittest.TestCase):
    def test_parses_a_firmware_line(self):
        dbw = link([b'T,1234,-12.50,6.20,1,-13.00,0,0\n'])
        dbw.poll(100.)
        self.assertAlmostEqual(dbw.meas_deg, -12.5)
        self.assertAlmostEqual(dbw.meas_mph, 6.2)
        self.assertAlmostEqual(dbw.fw_sp_deg, -13.0)
        self.assertTrue(dbw.engaged)
        self.assertEqual(dbw.meas_age(100.), 0.)

    def test_split_line_is_held_until_complete(self):
        dbw = link([b'T,1234,-12.50,6.2', b'0,1,-13.00,0,0\n'])
        dbw.poll(100.)
        self.assertIsNone(dbw.meas_deg)          # half a line must never be parsed
        dbw.poll(101.)
        self.assertAlmostEqual(dbw.meas_mph, 6.2)

    def test_newest_line_wins(self):
        dbw = link([b'T,1,0.00,0.00,1,0.00,0,0\nT,2,45.00,9.00,1,44.00,0,0\n'])
        dbw.poll(100.)
        self.assertAlmostEqual(dbw.meas_mph, 9.0)
        self.assertEqual(dbw.telem_lines, 2)

    def test_boot_garbage_and_other_lines_ignored(self):
        dbw = link([b'ADAPTER: ready. 0x740 v2 @ 50Hz.\n\x00\xff junk\n'])
        dbw.poll(100.)
        self.assertIsNone(dbw.meas_mph)
        self.assertEqual(dbw.telem_lines, 0)

    def test_implausible_values_are_rejected_not_believed(self):
        # A corrupt line must not be able to fake a safe reading or trigger a false trip.
        dbw = link([b'T,1,9999.00,500.00,1,0.00,0,0\nT,2,abc,1.0,1,0,0,0\nT,3,0.0,1.0,7,0,0,0\n'])
        dbw.poll(100.)
        self.assertIsNone(dbw.meas_mph)
        self.assertEqual(dbw.telem_bad, 3)

    def test_age_grows_and_starts_infinite(self):
        dbw = link()
        self.assertEqual(dbw.meas_age(100.), float('inf'))
        dbw.ser.chunks.append(b'T,1,0.00,0.00,1,0.00,0,0\n')
        dbw.poll(100.)
        self.assertAlmostEqual(dbw.meas_age(100.6), .6)


class Trips(unittest.TestCase):
    def monitor(self, **kw):
        kw.setdefault('max_speed_mph', 6.)
        return ResponseMonitor(**kw)

    def test_nothing_trips_before_the_first_command(self):
        m = self.monitor()
        self.assertIsNone(m.update(100., float('inf'), None, None, 0., 0., None))

    def test_silent_board_trips(self):
        m = self.monitor()
        self.assertIsNone(m.update(100., float('inf'), None, None, 0., 0., 100.))
        self.assertEqual(m.update(101.5, float('inf'), None, None, 0., 0., 100.),
                         'telemetry_never_arrived')

    def test_telemetry_lost_after_it_was_working(self):
        m = self.monitor()
        m.update(100., .05, 3., 0., 3., 0., 100.)
        self.assertEqual(m.update(102., 1.6, 3., 0., 3., 0., 100.), 'telemetry_lost')

    def test_overspeed_trips_at_once(self):
        # The October run: 17.6 mph measured against a 7 mph command. Firmware has no overspeed trip.
        m = self.monitor()
        self.assertEqual(m.update(100., .05, 17.6, 0., 6., 0., 99.), 'overspeed')

    def test_speed_commanded_but_car_stationary(self):
        # The other October symptom: 10+ s stationary while 5-7 mph was commanded (stale 0x284).
        m = self.monitor(no_motion_grace=3.)
        for t in (100., 101., 102.):
            self.assertIsNone(m.update(t, .05, 0.0, 0., 6., 0., 99.))
        self.assertEqual(m.update(103.5, .05, 0.0, 0., 6., 0., 99.), 'speed_hold_not_responding')

    def test_motion_clears_the_no_motion_timer(self):
        m = self.monitor(no_motion_grace=3.)
        m.update(100., .05, 0.0, 0., 6., 0., 99.)
        m.update(101., .05, 4.0, 0., 6., 0., 99.)        # car moved -- timer must restart
        self.assertIsNone(m.update(103.5, .05, 0.0, 0., 6., 0., 99.))

    def test_no_motion_not_armed_when_we_command_a_stop(self):
        m = self.monitor(no_motion_grace=1.)
        for t in (100., 102., 104.):
            self.assertIsNone(m.update(t, .05, 0.0, 0., 0., 0., 99.))

    def test_steering_not_tracking(self):
        m = self.monitor(steer_tol_deg=60., steer_grace=2.)
        self.assertIsNone(m.update(100., .05, 0., 0., 0., 200., 99.))
        self.assertEqual(m.update(102.5, .05, 0., 0., 0., 200., 99.), 'steering_not_tracking')

    def test_steering_inside_tolerance_never_trips(self):
        m = self.monitor(steer_tol_deg=60.)
        for t in (100., 102., 104., 106.):
            self.assertIsNone(m.update(t, .05, 0., 185., 0., 200., 99.))

    def test_stale_telemetry_suspends_the_value_trips(self):
        # A stale reading must not be used as evidence either way.
        m = self.monitor()
        m.update(100., .05, 3., 0., 3., 0., 100.)
        self.assertIsNone(m.update(100.5, .9, 0.0, 0., 6., 0., 100.))

    def test_rejects_nonsense_thresholds(self):
        with self.assertRaises(ValueError):
            ResponseMonitor(6., no_motion_grace=0.)
        with self.assertRaises(ValueError):
            ResponseMonitor(-1.)


if __name__ == '__main__':
    unittest.main()


class MilaRunPreflight(unittest.TestCase):
    """mila_run's checks, exercised without ROS: it imports rclpy and pyserial lazily."""
    def setUp(self):
        import mila_run
        self.m = mila_run

    @contextlib.contextmanager
    def fake_board(self, banner):
        """A pty that prints like a board. The writer is joined BEFORE the fds close: a daemon
        thread still writing to a closed fd will hit the number after it is reassigned to the
        next test's pty, and silently feed it this banner."""
        import pty
        master, slave = pty.openpty()
        stop = threading.Event()
        def talk():
            while not stop.is_set():
                try:
                    os.write(master, banner)
                except OSError:
                    return
                time.sleep(.05)
        writer = threading.Thread(target=talk)
        writer.start()
        try:
            yield os.ttyname(slave)
        finally:
            stop.set()
            writer.join(timeout=2)
            os.close(master)
            os.close(slave)

    def test_recognises_the_adapter(self):
        with self.fake_board(b'ADAPTER fresh steer=+12.0 mph=+0.00 cnt=7\n') as name:
            ok, detail = self.m.preflight_serial(name, seconds=.5)
        self.assertTrue(ok, detail)
        self.assertIn('ADAPTER', detail)

    def test_rejects_the_dbw_board(self):
        with self.fake_board(b'DBW-FULL: ready (steer angle + speed hold).\n') as name:
            ok, detail = self.m.preflight_serial(name, seconds=.5)
        self.assertFalse(ok)
        self.assertIn('DBW BOARD', detail)

    def test_rejects_a_silent_port(self):
        import pty
        master, slave = pty.openpty()
        try:
            ok, detail = self.m.preflight_serial(os.ttyname(slave), seconds=.4)
        finally:
            os.close(master); os.close(slave)
        self.assertFalse(ok)
        self.assertIn('silent', detail)

    def test_rejects_a_missing_device(self):
        ok, detail = self.m.preflight_serial('/dev/does-not-exist')
        self.assertFalse(ok)
        self.assertIn('does not exist', detail)

    def good_status(self, **changes):
        fields = dict(frame='enu', pose_reference='rear_axle', source_mode='live_localization',
                      diagnostic_only=True, tracking_valid=True, targets_valid=True,
                      vehicle_transform_verified=True, state='tracking', source_stamp_ns=10**10,
                      steering_rad=.1, target_speed_mps=1.)
        fields.update(changes)
        return json.dumps(fields)

    def test_accepts_a_live_target(self):
        ok, why = self.m.describe_status(self.good_status())
        self.assertTrue(ok, why)

    def test_names_a_replay_run(self):
        ok, why = self.m.describe_status(self.good_status(source_mode='recorded'))
        self.assertFalse(ok)
        self.assertIn('source_mode', why)

    def test_names_every_failed_flag(self):
        ok, why = self.m.describe_status(self.good_status(tracking_valid=False, targets_valid=None))
        self.assertFalse(ok)
        self.assertIn('tracking_valid', why)
        self.assertIn('targets_valid', why)

    def test_survives_garbage(self):
        for payload in ('not json', '[]', '{"state":'):
            ok, _ = self.m.describe_status(payload)
            self.assertFalse(ok)


class SpeedPassthrough(unittest.TestCase):
    """The bridge follows the planner's target unless an operator cap is given."""
    def speed(self, target_mps, cap):
        import mila_run  # noqa: F401  (keeps the import path consistent with the other tests)
        MPS_TO_MPH = 2.2369363
        WIRE_MAX_MPH = 10.
        speed = target_mps*MPS_TO_MPH
        if cap is not None:
            speed = min(cap, speed)
        return min(speed, WIRE_MAX_MPH)

    def test_uncapped_follows_the_planner(self):
        self.assertAlmostEqual(self.speed(1.0, None), 2.2369363)
        self.assertAlmostEqual(self.speed(2.0, None), 4.4738726)

    def test_planner_ceiling_lands_on_ten(self):
        # pure_pursuit.yaml preview_speed_mps 4.4704 is exactly 10 mph, so faithful
        # pass-through never asks the wire for more than it accepts.
        self.assertAlmostEqual(self.speed(4.4704, None), 10.0, places=4)

    def test_operator_cap_still_binds_when_given(self):
        self.assertAlmostEqual(self.speed(4.4704, 6.), 6.)
        self.assertAlmostEqual(self.speed(1.0, 6.), 2.2369363)   # cap above target changes nothing

    def test_wire_bound_is_the_backstop(self):
        # Even a planner asking for more than 10 mph cannot reach the wire check as a rejection.
        self.assertAlmostEqual(self.speed(20.0, None), 10.0)
