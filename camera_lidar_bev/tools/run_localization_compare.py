#!/usr/bin/python3
"""Compare GPS position and scan localization on a saved map, live or from a bag."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import yaml

from replay_combined_bev import positive_float, nonnegative_float
from start_bev import matching_processes, stop_owned


TOPICS = ['/points', '/imu', '/sensing/gnss/fix', '/metadata', '/tf_static']
REQUIRED_TYPES = {
    '/points': 'sensor_msgs/msg/PointCloud2', '/imu': 'sensor_msgs/msg/Imu',
    '/sensing/gnss/fix': 'sensor_msgs/msg/NavSatFix',
    '/metadata': 'std_msgs/msg/String', '/tf_static': 'tf2_msgs/msg/TFMessage',
}


def parse_args(argv=None, root=None):
    root = root or Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, default=root/'maps/local_gps_recording_002',
                        help='frozen local map directory')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bag', type=Path, help='one-pass recorded sensors; starts no hardware')
    source.add_argument('--live', action='store_true', help='use already-running GPS, /points and /imu')
    parser.add_argument('--alignment', type=Path, help='default MAP/gps_alignment')
    parser.add_argument('--metadata', type=Path, help='factory Ouster JSON; defaults to selected bag capture')
    parser.add_argument('--rate', type=positive_float, default=1., help='bag playback speed; default 1x')
    parser.add_argument('--start-offset', type=nonnegative_float, default=0.)
    parser.add_argument('--duration', type=positive_float, help='maximum active wall time in seconds')
    parser.add_argument('--output', type=Path, help='new session folder; default data/localization_compare_UTC')
    parser.add_argument('--domain', type=int, help='default 92 for replay, 1 for live')
    parser.add_argument('--max-rate', type=positive_float, default=5., help='localizer maximum wall updates/second')
    parser.add_argument('--no-display', action='store_true', help='save reports and snapshot without opening a window')
    parser.add_argument('--start-sensors', action='store_true',
                        help='with --live: start missing passive GPS bridge and 2048x10 LiDAR; reuse existing drivers')
    parser.add_argument('--dry-run', action='store_true', help='print commands; no files, nodes or sensors started')
    args = parser.parse_args(argv)
    if args.domain is None:
        args.domain = 1 if args.live else 92
    if not 0 <= args.domain <= 231:
        parser.error('--domain must be between 0 and 231')
    if args.live and args.start_offset:
        parser.error('--start-offset requires --bag')
    if args.live and args.rate != 1.:
        parser.error('--rate applies only to --bag')
    if args.start_sensors and not args.live:
        parser.error('--start-sensors requires --live')
    return args


def prepare_inputs(args, root):
    """Check small metadata files; never inspect original trajectory samples."""
    args.map = args.map.expanduser().resolve()
    for name in ('pointcloud_map.pcd', 'run_manifest.json', 'mapping_report.json'):
        if not (args.map/name).is_file():
            raise ValueError(f'saved map is missing {name}: {args.map}')
    manifest = json.loads((args.map/'run_manifest.json').read_text())
    mapping = json.loads((args.map/'mapping_report.json').read_text())
    if (mapping.get('failed_input') is not False or mapping.get('map_frame') != 'map'
            or mapping.get('pose_child_frame') != 'os_sensor'):
        raise ValueError('saved map must have valid map/os_sensor provenance')
    args.alignment = (args.alignment or args.map/'gps_alignment').expanduser().resolve()
    if (args.alignment/'.incomplete').exists():
        raise ValueError('GPS alignment is incomplete')
    alignment = json.loads((args.alignment/'alignment.json').read_text())
    if Path(alignment['source']['map_dir']).expanduser().resolve() != args.map:
        raise ValueError('GPS alignment belongs to a different map')
    source_bag = Path(manifest['bag']).expanduser().resolve()
    if args.bag is not None:
        args.bag = args.bag.expanduser().resolve()
        metadata_file = args.bag/'metadata.yaml'
        if not metadata_file.is_file():
            raise ValueError(f'bag metadata is missing: {args.bag}')
        info = yaml.safe_load(metadata_file.read_text())['rosbag2_bagfile_information']
        topics = {item['topic_metadata']['name']: item for item in info['topics_with_message_count']}
        for topic, expected in REQUIRED_TYPES.items():
            item = topics.get(topic)
            if item is None or item['message_count'] < 1 or item['topic_metadata']['type'] != expected:
                raise ValueError(f'bag needs recorded {topic} ({expected}) for this comparison')
        duration = int(info['duration']['nanoseconds'])/1e9
        if args.start_offset >= duration:
            raise ValueError(f'start offset is outside the {duration:g}-second bag')
    metadata_bag = args.bag or source_bag
    args.metadata = (args.metadata or Path(str(metadata_bag)+'.capture')/'ouster_metadata.json').expanduser().resolve()
    factory = json.loads(args.metadata.read_text())
    for group, key in (('lidar_intrinsics', 'lidar_to_sensor_transform'),
                       ('imu_intrinsics', 'imu_to_sensor_transform')):
        if len(factory[group][key]) != 16:
            raise ValueError(f'factory metadata needs {group}/{key}')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    args.output = (args.output or root/'data'/f'localization_compare_{timestamp}').expanduser().absolute()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f'refusing to overwrite session: {args.output}')
    return {
        'map_source_bag': str(source_bag),
        'same_recording_as_map': bool(args.bag and args.bag == source_bag),
        'saved_trajectory_used_for_localization': False,
        'accuracy_independently_validated': False,
        'gps_is_initial_position_prior': True,
        'antenna_lever_arm_measured': False,
    }


def commands(args, root):
    simulation = f'use_sim_time:={str(not args.live).lower()}'
    result = {
        'localizer': ['/usr/bin/python3', str(root/'tools/lidar_map_localizer.py'),
                      '--map', str(args.map), '--alignment', str(args.alignment),
                      '--metadata', str(args.metadata), '--report', str(args.output/'lidar_localization.json'),
                      '--ready-file', str(args.output/'localizer_ready.json'),
                      '--max-rate', str(args.max_rate), '--ros-args', '-p', simulation],
        'map view': ['/usr/bin/python3', str(root/'tools/localization_map_view.py'),
                     '--alignment-dir', str(args.alignment),
                     '--report', str(args.output/'viewer_report.json'),
                     '--snapshot', str(args.output/'latest.png')],
    }
    if args.no_display:
        result['map view'].append('--no-display')
    result['map view'] += ['--ros-args', '-p', simulation]
    if args.bag is not None:
        if args.start_offset:
            result['static context'] = ['/usr/bin/python3', str(root/'tools/replay_static_context.py'),
                                        '--bag', str(args.bag), '--start-offset', str(args.start_offset)]
        result['bag player'] = ['ros2', 'bag', 'play', str(args.bag), '--clock', '100',
                                '--rate', str(args.rate), '--start-offset', str(args.start_offset),
                                '--disable-keyboard-controls', '--read-ahead-queue-size', '20',
                                '--qos-profile-overrides-path', str(root/'config/recording/raw_sensor_qos.yaml'),
                                '--topics', *TOPICS]
    if args.start_sensors:
        result['GPS bridge'] = ['/usr/bin/python3', str(root/'tools/septentrio_gps.py'),
                                '--config', str(root/'config/sensors/septentrio_gps.yaml')]
        result['LiDAR driver'] = ['/usr/bin/python3', str(root/'tools/start_lidar.py'),
                                  '--params-file', str(root/'config/sensors/ouster_2048x10.yaml')]
    return result


def lock_domain(domain, replay):
    locks = []
    kinds = ('replay', 'localization') if replay else ('localization',)
    try:
        for kind in kinds:
            path = Path(f'/tmp/camera_lidar_bev_{kind}_{os.getuid()}_{domain}.lock')
            lock = path.open('a+')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                raise RuntimeError(f'another {kind} controller already owns ROS domain {domain}')
            lock.seek(0)
            lock.truncate()
            lock.write(str(os.getpid()))
            lock.flush()
            locks.append(lock)
    except BaseException:
        for lock in locks:
            lock.close()
        raise
    return locks


def wait_for_ready(path, process, check_running, is_stopped, timeout=30.):
    """Delay playback until this worker has actually built its map index."""
    deadline = time.monotonic()+timeout
    while not is_stopped():
        exited = check_running()
        if exited:
            raise RuntimeError(f'{exited[0]} exited with {exited[1]} during startup; inspect its session log')
        if path.is_file():
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or value.get('ready') is not True:
                raise RuntimeError('localizer readiness file does not confirm a prepared worker')
            if value.get('pid', process.pid) != process.pid:
                raise RuntimeError('localizer readiness file belongs to a different process')
            return True
        if time.monotonic() >= deadline:
            raise RuntimeError(f'localizer did not become ready within {timeout:g}s; inspect localizer.log')
        time.sleep(.1)
    return False


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    args = parse_args(argv, root)
    try:
        provenance = prepare_inputs(args, root)
        plan = commands(args, root)
        if args.dry_run:
            print(f'ROS domain: {args.domain}; output: {args.output}')
            for name, command in plan.items():
                print(f'{name}: {shlex.join(command)}')
            print('Dry run: no output, nodes or sensors started.')
            return 0
        if int(Path('/proc/sys/net/core/rmem_max').read_text()) < 16*1024**2:
            raise RuntimeError('DDS needs a 16 MiB receive buffer; run sudo bash tools/setup_dds_transport.sh')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f'Localization startup failed: {error}', file=sys.stderr)
        return 1

    children, locks, reused = [], [], {}
    stopped = False
    outcome, code = 'starting', 0
    started = time.monotonic()

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = str(args.domain)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'

    def start(name):
        with (args.output/f'{name.replace(" ", "_")}.log').open('x') as log:
            process = subprocess.Popen(plan[name], cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append((name, process))
        return process

    def check_running():
        for name, process in children:
            if process.poll() is not None:
                return name, process.returncode
        return None

    try:
        locks = lock_domain(args.domain, args.bag is not None)
        if args.start_sensors:
            # Probe both before starting either: another ROS domain can still own
            # the same physical sensor. Reused drivers are never owned or stopped.
            reused['GPS bridge'] = matching_processes('septentrio_gps.py', str(args.domain))
            driver = matching_processes('os_driver', str(args.domain))
            helper = matching_processes('start_lidar.py', str(args.domain))
            reused['LiDAR driver'] = driver or helper
        args.output.mkdir(parents=True)
        manifest = {'started_utc': datetime.now(timezone.utc).isoformat(), 'map': str(args.map),
                    'alignment': str(args.alignment), 'metadata': str(args.metadata),
                    'bag': str(args.bag) if args.bag else None, 'live': args.live,
                    'rate': args.rate, 'start_offset_s': args.start_offset, 'domain': args.domain,
                    'requested_wall_duration_s': args.duration, 'commands': plan,
                    'sensor_processes_reused': reused, **provenance}
        (args.output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'GPS / LiDAR map comparison: {args.output}\nROS domain: {args.domain}. '
              'Press q in the viewer or Ctrl+C to stop owned processes.', flush=True)
        if provenance['same_recording_as_map']:
            print('This bag built the reference map; results measure replay performance, not new-drive accuracy.', flush=True)
        for name in ('GPS bridge', 'LiDAR driver'):
            if name in plan and not reused.get(name):
                start(name)
            elif reused.get(name):
                print(f'Reusing {name} PID(s): {reused[name]}; these processes stay running after exit.', flush=True)
        if 'static context' in plan:
            start('static context')
        localizer = start('localizer')
        start('map view')
        print('Preparing the frozen map index before localization starts...', flush=True)
        wait_for_ready(args.output/'localizer_ready.json', localizer, check_running, lambda: stopped)
        if not stopped and 'bag player' in plan:
            start('bag player')
        active_started = time.monotonic()
        outcome = 'running'
        while not stopped:
            if args.duration and time.monotonic()-active_started >= args.duration:
                outcome = 'duration_reached'
                break
            exited = check_running()
            if exited:
                name, child_code = exited
                code = child_code if child_code >= 0 else 1
                if name == 'bag player' and code == 0:
                    outcome = 'playback_complete'
                    # Allow the latest queued registration to reach the viewer.
                    deadline = time.monotonic()+2.
                    while time.monotonic() < deadline and not stopped:
                        time.sleep(.1)
                elif name == 'map view' and code == 0:
                    outcome = 'viewer_closed'
                else:
                    outcome = 'child_exited'
                    code = code or 1
                    print(f'{name} exited with {child_code}; inspect session logs.', file=sys.stderr, flush=True)
                break
            time.sleep(.1)
        if stopped:
            outcome = 'interrupted'
    except (OSError, ValueError, RuntimeError) as error:
        outcome, code = 'failed', 1
        print(f'Localization comparison failed: {error}', file=sys.stderr, flush=True)
    finally:
        # A second Ctrl+C must not interrupt a worker join or report write.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        stop_owned(children)
        if args.output.is_dir() and (args.output/'run_manifest.json').is_file():
            summary = {'outcome': outcome, 'exit_code': code,
                       'wall_duration_s': time.monotonic()-started,
                       'owned_process_exit_codes': {name: process.poll() for name, process in children},
                       'reused_sensors_stopped': False,
                       'localizer_report_present': (args.output/'lidar_localization.json').is_file(),
                       'viewer_report_present': (args.output/'viewer_report.json').is_file(),
                       'snapshot_present': (args.output/'latest.png').is_file(), **provenance}
            (args.output/'session_report.json').write_text(json.dumps(summary, indent=2)+'\n')
            print(f'Reports and latest map view: {args.output}', flush=True)
        for lock in locks:
            lock.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
