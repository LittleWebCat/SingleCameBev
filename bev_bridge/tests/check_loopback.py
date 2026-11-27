#!/usr/bin/env python3
"""End-to-end loopback: real pyserial DBW <-> a simulated dbw_esp_full, over a real pty.

No ROS, no car, no ESP. Exercises the whole wire path the bridge actually uses -- the outgoing
'steer speed pedal mode' line, the incoming 0x741 'T,...' echo, and the ResponseMonitor trips --
so the October failure modes can be reproduced on a laptop before anyone sits in the Leaf.

    python3 bev_bridge/tests/check_loopback.py            # all scenarios
    python3 bev_bridge/tests/check_loopback.py --verbose  # print every exchange

Scenarios, each a known firmware/vehicle behaviour:
    healthy          car follows the speed command            -> no trip
    stale_wheelspeed 0x284 stale: firmware rests the pedals   -> speed_hold_not_responding
    runaway          car accelerates past the cap             -> overspeed
    silent           board stops echoing 0x741                -> telemetry_lost
    stuck_wheel      measured wheel never leaves centre       -> steering_not_tracking
"""
import argparse
import os
import pty
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bridge_core import ResponseMonitor
from dbw_serial import DBW

TICK = .05            # 20 Hz, the firmware's telemetry cadence
CAP_MPH = 6.


class FakeEsp:
    """The car + dbw_esp_full, as seen through the adapter's USB line."""
    def __init__(self, fd, mode):
        self.fd, self.mode = fd, mode
        self.rx = b''
        self.cmd_steer = self.cmd_speed = 0.
        self.meas_deg = self.meas_mph = 0.
        self.ms = 0

    def step(self):
        try:                                            # drain whatever the bridge wrote
            self.rx += os.read(self.fd, 4096)
        except BlockingIOError:
            pass
        while b'\n' in self.rx:
            line, _, self.rx = self.rx.partition(b'\n')
            parts = line.decode('ascii', 'replace').split()
            if len(parts) == 4:
                try:
                    self.cmd_steer, self.cmd_speed = float(parts[0]), float(parts[1])
                except ValueError:
                    pass
        # vehicle response
        if self.mode == 'stale_wheelspeed':
            self.meas_mph = 0.                          # pedals rest: 0x284 older than 300 ms
        elif self.mode == 'runaway':
            self.meas_mph = min(20., self.meas_mph+.6)  # integrator wind-up, no firmware trip
        else:
            self.meas_mph += (self.cmd_speed-self.meas_mph)*.25
        if self.mode != 'stuck_wheel':
            self.meas_deg += (self.cmd_steer-self.meas_deg)*.3
        self.ms += int(TICK*1000)
        if self.mode == 'silent' and self.ms > 1000:
            return                                      # board stops talking
        line = f'T,{self.ms},{self.meas_deg:.2f},{self.meas_mph:.2f},1,{self.cmd_steer:.2f},0,0\n'
        try:
            os.write(self.fd, line.encode('ascii'))
        except OSError:
            pass


def run(mode, seconds=8., verbose=False):
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    esp = FakeEsp(master, mode)
    dbw = DBW(os.ttyname(slave), settle=0.)
    monitor = ResponseMonitor(CAP_MPH, telemetry_timeout=1., no_motion_grace=3.,
                              steer_tol_deg=60., steer_grace=2.)
    first_cmd = None
    trip = None
    t0 = time.monotonic()
    try:
        while time.monotonic()-t0 < seconds:
            now = time.monotonic()
            steer, speed = 120., CAP_MPH          # a steady demand, as the tracker would make
            if dbw.command(steer, speed_mph=speed, mode=0):
                first_cmd = now if first_cmd is None else first_cmd
            esp.step()
            dbw.poll(now)
            trip = monitor.update(now, dbw.meas_age(now), dbw.meas_mph, dbw.meas_deg,
                                  speed, steer, first_cmd)
            if verbose:
                print(f'  t={now-t0:5.2f} cmd={speed:.1f}mph/{steer:.0f}deg '
                      f'meas={dbw.meas_mph}mph/{dbw.meas_deg}deg age={dbw.meas_age(now):.2f} trip={trip}')
            if trip:
                break
            time.sleep(TICK)
    finally:
        dbw.close()
        os.close(master)
        os.close(slave)
    return trip, round(time.monotonic()-t0, 2)


EXPECTED = dict(healthy=None, stale_wheelspeed='speed_hold_not_responding',
                runaway='overspeed', silent='telemetry_lost', stuck_wheel='steering_not_tracking')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--only', choices=sorted(EXPECTED))
    args = parser.parse_args()
    scenarios = [args.only] if args.only else list(EXPECTED)
    failures = 0
    for mode in scenarios:
        want = EXPECTED[mode]
        if args.verbose:
            print(f'--- {mode} (expect {want})')
        got, elapsed = run(mode, verbose=args.verbose)
        ok = got == want
        failures += not ok
        print(f'{"PASS" if ok else "FAIL"}  {mode:17s} expected={want!s:27s} got={got!s:27s} ({elapsed}s)')
    print('\nloopback:', 'OK' if not failures else f'{failures} FAILED')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
