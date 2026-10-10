#!/usr/bin/python3
"""Replay a saved camera/LiDAR recording beside its combined detector BEV."""

import argparse
import fcntl
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be finite and greater than zero')
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('must be finite and at least zero')
    return number


def positive_int(value):
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError('must be a positive integer') from error
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def stop_owned(children):
    """Signal only the process groups this controller created."""
    for signum, timeout in ((signal.SIGINT, 4), (signal.SIGTERM, 2),
                            (signal.SIGKILL, 1)):
        for name, child in reversed(children):
            try:
                if name in ('local mapping', 'combined display') and signum == signal.SIGINT and child.poll() is None:
                    # ros2 launch forwards this to the estimator and collector.
                    # Sending it to the group too interrupts their cleanup twice.
                    child.send_signal(signum)
                else:
                    os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        for _, child in reversed(children):
            try:
                child.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        if all(child.poll() is not None for _, child in children):
            break


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', nargs='?', type=Path,
                        default=None,
                        help='recording directory (default: collected outdoor recording)')
    parser.add_argument('--rate', type=positive_float, default=0.5,
                        help='playback speed; default 0.5 (half speed)')
    parser.add_argument('--no-loop', action='store_true',
                        help='play once and stop the viewer when playback ends')
    parser.add_argument('--no-road-lanes', action='store_true',
                        help='replay object boxes without starting the road/lane segmentation model')
    parser.add_argument('--max-points', type=positive_int, default=30000,
                        help='display point limit (default 30000); use 300000 to retain a dense 10 Hz scan')
    parser.add_argument('--range-m', type=positive_float, default=80.0,
                        help='forward BEV display range in meters (default 80)')
    map_mode = parser.add_mutually_exclusive_group()
    map_mode.add_argument('--mapping', action='store_true',
                          help='add LiDAR–IMU odometry/local map; forces one-pass replay and saves maps/local_UTC')
    map_mode.add_argument('--saved-map', type=Path,
                          help='display saved map with its original recorded trajectory; no new odometry')
    parser.add_argument('--gps-alignment', type=Path,
                        help='GPS alignment directory for --saved-map (default: saved map/gps_alignment when present)')
    parser.add_argument('--max-map-points', type=positive_int, default=100000,
                        help='local saved-map display limit after spatial cropping (default 100000)')
    parser.add_argument('--mapping-output', type=Path,
                        help='new map output directory; requires --mapping')
    parser.add_argument('--start-offset', type=nonnegative_float, default=0.0,
                        help='start this many recorded seconds into the bag')
    parser.add_argument('--domain', type=int,
                        default=os.environ.get('ROS_DOMAIN_ID', '89'),
                        help='isolated ROS domain (wrapper default: 89)')
    args = parser.parse_args()
    if not 0 <= args.domain <= 231:
        parser.error('ROS domain must be between 0 and 231')
    if args.mapping_output and not args.mapping:
        parser.error('--mapping-output requires --mapping')
    if args.gps_alignment and not args.saved_map:
        parser.error('--gps-alignment requires --saved-map')
    if args.mapping:
        args.no_loop = True
    if args.saved_map:
        import json
        args.saved_map = args.saved_map.expanduser().resolve()
        try:
            source_bag = json.loads((args.saved_map/'run_manifest.json').read_text())['bag']
            bag = (args.bag or Path(source_bag)).expanduser().resolve()
            from saved_map_source import validate_source
            validate_source(args.saved_map, bag)
        except (OSError, KeyError, TypeError, ValueError) as error:
            parser.error(f'saved-map provenance check failed: {error}')
        default_alignment = args.saved_map/'gps_alignment'
        if args.gps_alignment is None and (default_alignment/'alignment.json').is_file():
            args.gps_alignment = default_alignment
        if args.gps_alignment:
            args.gps_alignment = args.gps_alignment.expanduser().resolve()
            if not (args.gps_alignment/'alignment.json').is_file():
                parser.error(f'GPS alignment is missing: {args.gps_alignment}/alignment.json')
    else:
        bag = (args.bag or root/'bags/camera_lidar_20261009T192214_724569Z').expanduser().resolve()
    if not bag.exists():
        parser.error(f'recording does not exist: {bag}')
    try:
        receive_limit = int(Path('/proc/sys/net/core/rmem_max').read_text().strip())
    except (OSError, ValueError) as exc:
        parser.error(f'cannot read DDS receive-buffer limit: {exc}')
    if receive_limit < 16 * 1024**2:
        parser.error('DDS needs a 16 MiB receive-buffer limit. '
                     'See docs/combined_bev_display.md for the host setup command.')

    # Keep the inode in place after exiting: unlinking a flock file introduces
    # a race in which two controllers can lock different inodes for one domain.
    lock_path = Path('/tmp')/f'camera_lidar_bev_replay_{os.getuid()}_{args.domain}.lock'
    lock = lock_path.open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        owner = lock.read().strip() or 'unknown'
        lock.close()
        parser.error(f'a replay controller already owns domain {args.domain} '
                     f'(PID {owner}); stop it with Ctrl+C before starting another')
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()

    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = str(args.domain)
    launch = ['ros2', 'launch', 'bev_bringup', 'combined_bev_display.launch.py',
              'use_sim_time:=true', f'max_points:={args.max_points}', f'range_m:={args.range_m}']
    if args.saved_map:
        launch += [f'saved_map:={args.saved_map}', f'replay_bag:={bag}',
                   f'max_map_points:={args.max_map_points}']
        if args.gps_alignment:
            launch.append(f'gps_alignment:={args.gps_alignment}')
    if args.no_road_lanes:
        launch.append('road_lanes:=false')
    captured = Path(str(bag)+'.capture')
    for key, filename in [('intrinsics_file', 'front_camera_1024x768.candidate.yaml'),
                          ('extrinsics_file', 'T_B_os_sensor.candidate.yaml')]:
        if (captured/filename).is_file():
            launch.append(f'{key}:={captured/filename}')
    player = ['ros2', 'bag', 'play', str(bag), '--clock', '30', '--rate', str(args.rate),
              '--start-offset', str(args.start_offset), '--disable-keyboard-controls',
              '--read-ahead-queue-size', '50']
    if not args.no_loop:
        player.append('--loop')
    player += ['--topics', '/camera/front/image_raw', '/camera/front/camera_info',
               '/points', '/tf_static', '/metadata']
    if args.mapping:
        player.append('/imu')
        player += ['--qos-profile-overrides-path', str(root/'config/recording/raw_sensor_qos.yaml')]
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(f'Replay: {bag}\nROS domain: {args.domain}; speed: {args.rate:g}x; '
          f'loop: {not args.no_loop}\nCtrl+C stops this replay and its detectors. '
          'No physical sensors are started.', flush=True)
    if args.saved_map:
        print(f'Saved source map: {args.saved_map}\nGPS alignment: {args.gps_alignment or "none"}\n'
              'Map poses use native scan-end timestamps from this original recording; '
              'this mode does not estimate localization against another recording.', flush=True)
    children = []
    mapping_output = None
    try:
        from mapping_session import start_static_context
        context = start_static_context(children, root, env, bag, args.start_offset)
        if args.mapping:
            from mapping_session import start_mapping
            mapping_output, mapper = start_mapping(children, root, env, bag, args.mapping_output)
        print(shlex.join(launch), flush=True)
        display = subprocess.Popen(launch, cwd=root, env=env,
                                   stdin=subprocess.DEVNULL, start_new_session=True)
        children.append(('combined display', display))
        # Give subscribers a short head start without scanning the recording.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not stopped:
            if display.poll() is not None:
                print('Combined display launch exited before playback started.',
                      file=sys.stderr, flush=True)
                return display.returncode or 1
            if args.mapping and mapper.poll() is not None:
                raise RuntimeError(f'mapping launch exited; inspect {mapping_output}/mapping.log')
            if context is not None and context.poll() is not None:
                raise RuntimeError('recorded static context helper exited before playback')
            time.sleep(0.1)
        if stopped:
            return 0
        print(shlex.join(player), flush=True)
        playback = subprocess.Popen(player, cwd=root, env=env,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        children.append(('bag playback', playback))
        while not stopped:
            for name, child in children:
                code = child.poll()
                if code is not None:
                    print(f'{name} exited with status {code}; stopping owned processes.',
                          flush=True)
                    if name == 'bag playback' and args.mapping and code == 0:
                        # Let the last registered scan reach the map collector.
                        deadline = time.monotonic()+2
                        while time.monotonic() < deadline and not stopped:
                            time.sleep(0.1)
                    return code if code >= 0 else 1
            time.sleep(0.2)
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Replay startup failed: {exc}. Run tools/replay_combined_bev.sh '
              'to source the ROS environments.', file=sys.stderr, flush=True)
        return 1
    finally:
        stop_owned(children)
        for _, child in children:
            if hasattr(child, 'mapping_lock'):
                child.mapping_lock.close()
        if mapping_output:
            from mapping_session import mapping_result
            mapping_result(mapping_output)
        lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
