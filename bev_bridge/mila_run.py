#!/usr/bin/env python3
"""The MiLa side, in one command: preflight, bridge, live status, engage.

Hanchen runs his usual BEV stack and changes nothing. It publishes
`/camera_lidar_bev/path_tracking/status`. This script does everything on our side:

  1. SERIAL PREFLIGHT  -- resolve the device in hardware.json and confirm it is the ADAPTER.
     The adapter and the DBW board are both CP2102 serial 0001, so their /dev/serial/by-id
     alias follows whichever enumerated last; opening the wrong one commands nothing. Read-only,
     and DTR/RTS are held low so the check does not reset the board.
  2. TOPIC PREFLIGHT   -- watch his status topic and report rate plus every field the bridge
     gates on, so a rejected run is diagnosed before the car is involved, not after.
  3. BRIDGE           -- start bev_dbw_bridge.py as a child process (dry-run unless --actuate).
     It is NOT run_all.sh: that would launch a second copy of his stack.
  4. STATUS           -- one live line: gate state, commanded steer/speed, and the 0x741
     telemetry the board echoes back (measured angle, measured mph, engaged, age).
  5. ENGAGE           -- only with --engage, only after a typed confirmation, and the heartbeat
     stops the instant this script does.

    python3 bev_bridge/mila_run.py --preflight-only     # checks only, no ROS, no bridge
    python3 bev_bridge/mila_run.py                      # preflight + dry-run bridge + status
    python3 bev_bridge/mila_run.py --actuate --engage   # the real thing

This script NEVER engages on its own. --engage still requires typing the confirmation, and
Ctrl-C stops the heartbeat, after which the bridge stop-latches and the firmware's 500 ms
command watchdog drops the board to safe. The hardware e-stop / JD-VCC cut is the real stop
at any point, and nothing here replaces it.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIRM = 'drive'


# ---------------------------------------------------------------- 1. serial preflight

def read_port(device, seconds=3.5):
    """Read the device without writing to it and without resetting it."""
    import serial
    port = serial.Serial(port=None, baudrate=115200, timeout=.2)
    port.dtr = False                 # the ESP resets on DTR/RTS; this check must not reboot it
    port.rts = False
    port.port = str(device)
    port.open()
    try:
        port.reset_input_buffer()    # only FRESH output counts: stale buffered bytes would let a
        time.sleep(.1)               # silent or wrong board pass on what someone else left behind
        deadline = time.time()+seconds
        data = b''
        while time.time() < deadline:
            data += port.read(4096)
        return data.decode('ascii', 'replace')
    finally:
        port.close()


def preflight_serial(device, seconds=3.5):
    """(ok, detail). Looks for the adapter's banner or its periodic status line."""
    if not device:
        return False, 'no device configured'
    real = os.path.realpath(device)
    if not os.path.exists(device):
        return False, f'{device} does not exist'
    try:
        text = read_port(device, seconds)
    except Exception as error:
        return False, f'could not read {device}: {error}'
    where = f'{device}' + (f' -> {os.path.basename(real)}' if real != str(device) else '')
    if 'ADAPTER' in text:
        line = next((l for l in text.splitlines() if 'ADAPTER' in l), '').strip()
        return True, f'{where}: {line[:90]}'
    if 'DBW-FULL' in text or 'DBW ' in text:
        return False, (f'{where} is the DBW BOARD, not the adapter. The by-id alias followed the '
                       'wrong CP2102. Replug so the adapter enumerates last, then re-run.')
    if not text.strip():
        return False, f'{where}: silent. Is it powered and on this port?'
    sample = ' '.join(text.split())[:70]
    return False, f'{where}: unrecognised output, not the adapter firmware -- "{sample}"'


def load_hardware(path, actuate):
    if not actuate:
        return None, None
    if not path.exists():
        return None, f'{path} not found'
    try:
        hardware = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return None, f'{path}: {error}'
    return hardware, None


# ---------------------------------------------------------------- 2. topic preflight

GATED = (('source_mode', 'live_localization'), ('frame', 'enu'), ('pose_reference', 'rear_axle'))
FLAGS = ('tracking_valid', 'targets_valid', 'vehicle_transform_verified', 'diagnostic_only')


def describe_status(payload):
    """Why the bridge would accept or reject one of his status messages."""
    try:
        fields = json.loads(payload)
        if not isinstance(fields, dict):
            return False, 'status is not a JSON object'
    except ValueError as error:
        return False, f'unparsable status JSON: {error}'
    problems = [f'{k}={fields.get(k)!r} (want {v!r})' for k, v in GATED if fields.get(k) != v]
    problems += [f'{k} is {fields.get(k)!r}' for k in FLAGS if fields.get(k) is not True]
    if fields.get('state') not in ('tracking', 'endpoint'):
        problems.append(f'state={fields.get("state")!r}')
    if type(fields.get('source_stamp_ns')) is not int:
        problems.append('source_stamp_ns missing')
    if problems:
        return False, '; '.join(problems)
    return True, (f'state={fields.get("state")} steering={fields.get("steering_rad")} '
                  f'target={fields.get("target_speed_mps")} m/s')


# ---------------------------------------------------------------- 3-5. ROS side

class Side:
    """Watches his status topic and our bridge status; owns the engage heartbeat."""
    def __init__(self, args):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from std_msgs.msg import String, Bool
        self.a = args
        self.rclpy = rclpy
        rclpy.init(args=[])
        self.node = Node('mila_run')
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.src_count = 0
        self.src_first = None
        self.src_last_reason = 'no message yet'
        self.src_ok = False
        self.bridge_status = None
        self.engaging = False
        self.node.create_subscription(String, args.topic, self._on_source, qos)
        self.node.create_subscription(String, args.status_topic, self._on_status, 1)
        self.engage_pub = self.node.create_publisher(Bool, args.engage_topic, 10)
        self._Bool = Bool
        self._stop = threading.Event()
        self._spin = threading.Thread(target=self._spin_forever, daemon=True)
        self._spin.start()

    def _spin_forever(self):
        while not self._stop.is_set() and self.rclpy.ok():
            self.rclpy.spin_once(self.node, timeout_sec=.1)

    def _on_source(self, message):
        self.src_count += 1
        self.src_first = self.src_first or time.monotonic()
        self.src_ok, self.src_last_reason = describe_status(message.data)

    def _on_status(self, message):
        try:
            self.bridge_status = json.loads(message.data)
        except ValueError:
            pass

    def wait_for_source(self, seconds):
        print(f'  watching {self.a.topic} for {seconds:.0f}s ...')
        deadline = time.time()+seconds
        while time.time() < deadline:
            time.sleep(.2)
        if not self.src_count:
            return False, f'no messages on {self.a.topic} -- is his stack running?'
        span = max(1e-6, time.monotonic()-self.src_first)
        rate = self.src_count/span
        if not self.src_ok:
            return False, f'{self.src_count} msgs at {rate:.1f} Hz, but REJECTED: {self.src_last_reason}'
        return True, f'{self.src_count} msgs at {rate:.1f} Hz, accepted -- {self.src_last_reason}'

    def heartbeat(self):
        """Explicit release, then True at 10 Hz until stopped. Never called without confirmation."""
        self.engage_pub.publish(self._Bool(data=False))
        time.sleep(.2)
        self.engaging = True
        while not self._stop.is_set():
            self.engage_pub.publish(self._Bool(data=True))
            time.sleep(.1)

    def line(self):
        status = self.bridge_status or {}
        telemetry = status.get('telemetry') or {}
        state = status.get('state', '-')
        if status.get('stop_latched'):
            cause = (status.get('stop_reason') or {}).get('cause', '?')
            state = f'STOP LATCHED: {cause}'
        out = status.get('last_output') or {}
        age = telemetry.get('age_s')
        tele = ('telemetry: none' if age is None else
                f'meas {telemetry.get("meas_deg")}deg {telemetry.get("meas_mph")}mph '
                f'eng={1 if telemetry.get("firmware_engaged") else 0} age={age:.2f}s')
        return (f'[{"ENGAGING" if self.engaging else "idle":8s}] {state:38.38s} | '
                f'cmd {out.get("deg", 0):+7.1f}deg {out.get("speed_mph", 0):5.2f}mph | {tele}')

    def close(self):
        self._stop.set()
        try:
            self.engage_pub.publish(self._Bool(data=False))     # explicit release on the way out
        except Exception:
            pass
        time.sleep(.2)
        try:
            self.node.destroy_node()
            if self.rclpy.ok():
                self.rclpy.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------- bridge child

def bridge_command(args):
    cmd = [sys.executable, str(ROOT/'bev_dbw_bridge.py'),
           '--topic', args.topic, '--status-topic', args.status_topic,
           '--engage-topic', args.engage_topic]
    if args.max_speed_mph is not None:
        cmd += ['--max-speed-mph', str(args.max_speed_mph)]
    if args.actuate:
        cmd += ['--actuate', '--hardware-config', str(args.hardware_config)]
    else:
        cmd += ['--dry-run']
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--actuate', action='store_true', help='open the adapter (default: dry run)')
    parser.add_argument('--engage', action='store_true', help='offer the engage heartbeat after preflight')
    parser.add_argument('--hardware-config', type=Path, default=ROOT/'hardware.json')
    parser.add_argument('--device', help='override the device to preflight')
    parser.add_argument('--topic', default='/camera_lidar_bev/path_tracking/status')
    parser.add_argument('--engage-topic', default='/bev_dbw/engage')
    parser.add_argument('--status-topic', default='/bev_dbw/bridge_status')
    parser.add_argument('--max-speed-mph', type=float, default=None,
                        help='optional operator speed cap in mph (max 10); omitted = follow the planner')
    parser.add_argument('--preflight-only', action='store_true', help='checks only; no ROS, no bridge')
    parser.add_argument('--source-seconds', type=float, default=5., help='how long to watch his topic')
    parser.add_argument('--serial-seconds', type=float, default=3.5)
    args = parser.parse_args()

    print('== 1. serial preflight ==')
    hardware, error = load_hardware(args.hardware_config, args.actuate)
    if error:
        print(f'  FAIL  {error}')
        return 1
    device = args.device or (hardware or {}).get('device')
    if args.actuate and args.device and hardware and args.device != hardware.get('device'):
        print('  FAIL  --device differs from hardware.json; the bridge would reject it too')
        return 1
    serial_ok = True
    if device:
        serial_ok, detail = preflight_serial(device, args.serial_seconds)
        print(f'  {"OK  " if serial_ok else "FAIL"}  {detail}')
        if not serial_ok and args.actuate:
            return 1
    else:
        print('  skip  dry run with no device configured')

    if args.preflight_only:
        print('\npreflight only; nothing started.')
        return 0 if serial_ok else 1      # nonzero so this is usable as a scripted gate

    print('\n== 2. his path-tracking topic ==')
    try:
        side = Side(args)
    except ImportError as error:
        print(f'  FAIL  rclpy unavailable ({error}). Source the ROS environment first.')
        return 1
    bridge = None
    rc = 0
    try:
        ok, detail = side.wait_for_source(args.source_seconds)
        print(f'  {"OK  " if ok else "FAIL"}  {detail}')
        if not ok:
            print('\nNot starting the bridge. Fix his side first; nothing here was engaged.')
            return 1

        print(f'\n== 3. bridge ({"HARDWARE" if args.actuate else "dry run"}) ==')
        bridge = subprocess.Popen(bridge_command(args))
        time.sleep(2.5)
        if bridge.poll() is not None:
            print(f'  FAIL  bridge exited immediately (code {bridge.returncode})')
            return 1
        print(f'  started pid {bridge.pid}')

        if args.engage:
            print(f'\n== 4. engage ==\n  The car will move. E-stop in hand. Wheels up for a first run.')
            print(f'  Type "{CONFIRM}" to start the heartbeat, anything else to stay idle.')
            if input('  > ').strip() != CONFIRM:
                print('  staying idle; bridge keeps running, nothing engaged.')
            else:
                threading.Thread(target=side.heartbeat, daemon=True).start()
                print('  heartbeat running. Ctrl-C stops it and releases.')
        else:
            print('\n== 4. engage ==\n  --engage not given: watching only, nothing will move.')

        print('\n== status (Ctrl-C to stop) ==')
        while bridge.poll() is None:
            print('  '+side.line(), end='\r', flush=True)
            time.sleep(.25)
        print(f'\nbridge exited (code {bridge.returncode})')
        rc = bridge.returncode or 0
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        side._stop.set()
        side.close()
        if bridge is not None and bridge.poll() is None:
            bridge.terminate()                       # its own shutdown() brakes, then releases
            try:
                bridge.wait(timeout=10)
            except subprocess.TimeoutExpired:
                bridge.kill()
        print('stopped: heartbeat released, bridge down. '
              'Firmware watchdog has the board in safe.')
    return rc


if __name__ == '__main__':
    sys.exit(main())
