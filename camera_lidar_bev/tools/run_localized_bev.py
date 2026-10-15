#!/usr/bin/python3
"""Run camera/LiDAR BEV with independent live localization on a frozen map."""

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

import run_localization_compare as comparison
from replay_combined_bev import positive_float
from start_bev import matching_processes, preflight, stop_owned


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def parse_args(argv=None, root=None):
    root = root or Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=Path, default=root/'maps/local_gps_recording_002')
    parser.add_argument('--gps-alignment', type=Path, help='default MAP/gps_alignment')
    parser.add_argument('--metadata', type=Path, help='default factory metadata from the map capture')
    parser.add_argument('--domain', type=int, default=int(os.environ.get('ROS_DOMAIN_ID', '1')))
    # Above the 10 Hz scan rate so the cap never delays or overwrites a scan (~70 ms compute).
    parser.add_argument('--max-rate', type=positive_float, default=20., help='localization update cap in Hz')
    parser.add_argument('--range-m', type=positive_float, default=80.)
    parser.add_argument('--max-points', type=positive_int, default=30000,
                        help='current LiDAR point limit for drawing')
    parser.add_argument('--max-map-points', type=positive_int, default=100000,
                        help='local frozen-map crop limit for drawing')
    parser.add_argument('--road-lanes', action='store_true', help='also start image road/lane inference')
    parser.add_argument('--pure-pursuit', action='store_true', help='show reference path and diagnostic tracking targets')
    parser.add_argument('--tracking-config', type=Path, help='Pure Pursuit YAML; requires --pure-pursuit')
    parser.add_argument('--duration', type=positive_float, help='maximum wall seconds after the viewer starts')
    parser.add_argument('--output', type=Path, help='new session folder; default data/localized_bev_UTC')
    parser.add_argument('--dry-run', action='store_true', help='print commands without starting or writing anything')
    args = parser.parse_args(argv)
    if not 0 <= args.domain <= 231:
        parser.error('--domain must be between 0 and 231')
    if args.range_m > 200.:
        parser.error('--range-m must be at most 200')
    if args.tracking_config and not args.pure_pursuit:
        parser.error('--tracking-config requires --pure-pursuit')
    return args


def prepare_inputs(args, root):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    args.output = (args.output or root/'data'/f'localized_bev_{stamp}').expanduser().absolute()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f'refusing to overwrite session: {args.output}')
    # Reuse the comparison controller's small-file provenance and factory checks.
    # No recorded trajectory is loaded or supplied to live localization.
    options = ['--live', '--map', str(args.map), '--output', str(args.output/'localization'),
               '--domain', str(args.domain)]
    if args.gps_alignment:
        options += ['--alignment', str(args.gps_alignment)]
    if args.metadata:
        options += ['--metadata', str(args.metadata)]
    inputs = comparison.parse_args(options, root)
    provenance = comparison.prepare_inputs(inputs, root)
    args.map, args.gps_alignment, args.metadata = inputs.map, inputs.alignment, inputs.metadata
    if args.pure_pursuit:
        from saved_map_source import read_trajectory
        read_trajectory(args.gps_alignment/'trajectory_enu.csv')
        args.tracking_config = (args.tracking_config or root/'config/tracking/pure_pursuit.yaml').expanduser().resolve()
        if not args.tracking_config.is_file():
            raise FileNotFoundError(f'missing tracking configuration: {args.tracking_config}')
    return provenance


def commands(args, root):
    localize = ['bash', str(root/'tools/run_localization_compare.sh'), '--live', '--start-sensors',
                '--no-display', '--map', str(args.map), '--alignment', str(args.gps_alignment),
                '--metadata', str(args.metadata), '--domain', str(args.domain),
                '--max-rate', str(args.max_rate), '--output', str(args.output/'localization')]
    view = ['bash', str(root/'tools/run_bev.sh'), '--combined-view', '--localize-map', str(args.map),
            '--gps-alignment', str(args.gps_alignment), '--lidar-mode', '2048x10',
            '--range-m', str(args.range_m), '--max-points', str(args.max_points),
            '--max-map-points', str(args.max_map_points)]
    if not args.road_lanes:
        view.append('--no-road-lanes')
    if args.pure_pursuit:
        view += ['--pure-pursuit', '--tracking-config', str(args.tracking_config)]
    return {'localization': localize, 'combined_view': view}


def lock_session(domain):
    path = Path(f'/tmp/camera_lidar_bev_localized_view_{os.getuid()}_{domain}.lock')
    handle = path.open('a+')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f'another localized BEV controller owns ROS domain {domain}')
    return handle


def wait_localizer_ready(path, controller, is_stopped, timeout=40.):
    """Accept readiness only from this controller's live worker."""
    deadline = time.monotonic()+timeout
    while not is_stopped():
        if controller.poll() is not None:
            raise RuntimeError(f'localization controller exited with {controller.returncode}; '
                               'inspect localization.log and localization/*.log')
        if path.is_file():
            ready = json.loads(path.read_text())
            if not isinstance(ready, dict) or ready.get('ready') is not True:
                raise RuntimeError('localizer readiness marker is invalid')
            pid = ready.get('pid')
            if not isinstance(pid, int) or pid <= 0:
                raise RuntimeError('localizer readiness marker has no worker PID')
            try:
                state = (Path('/proc')/str(pid)/'status').read_text()
                parent = next(int(line.split(':', 1)[1]) for line in state.splitlines()
                              if line.startswith('PPid:'))
            except (OSError, ValueError, StopIteration) as error:
                raise RuntimeError('prepared localizer worker is no longer running') from error
            if parent != controller.pid:
                raise RuntimeError('localizer readiness belongs to a different controller')
            return ready
        if time.monotonic() >= deadline:
            raise RuntimeError('localizer did not become ready; inspect localization/localizer.log')
        time.sleep(.1)
    return None


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    try:
        args = parse_args(argv, root)
        provenance = prepare_inputs(args, root)
        plan = commands(args, root)
        if args.dry_run:
            print(f'ROS domain: {args.domain}; output: {args.output}')
            for name, command in plan.items():
                print(f'{name}: {shlex.join(command)}')
            print('Wait for the owned localizer map index, then open the combined viewer. '
                  'New LiDAR drivers use 2048x10; existing drivers are reused.')
            print('Dry run: no output, sensors, models, localization or recorder started.')
            return 0
        preflight(root, combined_view=True, road_lanes=args.road_lanes)
        # These viewers/localizers cannot be silently reused with a different map.
        for executable in ('lidar_map_localizer.py', 'combined_bev_live.py'):
            if matching_processes(executable, str(args.domain)):
                raise RuntimeError(f'{executable} already runs; close its controller first')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f'Localized BEV startup stopped: {error}', file=sys.stderr)
        return 1

    children, session_lock = [], None
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
        with (args.output/f'{name}.log').open('x') as log:
            process = subprocess.Popen(plan[name], cwd=root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append((name, process))
        return process

    try:
        session_lock = lock_session(args.domain)
        args.output.mkdir(parents=True)
        manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), map=str(args.map),
                        gps_alignment=str(args.gps_alignment), metadata=str(args.metadata),
                        live=True, domain=args.domain, lidar_mode_for_new_driver='2048x10',
                        requested_wall_duration_s=args.duration, max_localization_rate_hz=args.max_rate,
                        range_m=args.range_m, max_points=args.max_points, max_map_points=args.max_map_points,
                        road_lanes=args.road_lanes, commands=plan, recorder_started=False, **provenance)
        (args.output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        print(f'Live camera/LiDAR BEV on saved map: {args.output}\n'
              'Preparing the localization index; missing GPS, LiDAR and camera drivers will start. '
              'Press q in the combined window or Ctrl+C to stop owned processes.', flush=True)
        controller = start('localization')
        ready = wait_localizer_ready(args.output/'localization/localizer_ready.json',
                                      controller, lambda: stopped)
        if ready is not None and not stopped:
            start('combined_view')
        active_started = time.monotonic()
        outcome = 'running'
        while not stopped:
            if args.duration and time.monotonic()-active_started >= args.duration:
                outcome = 'duration_reached'
                break
            exited = next(((name, process) for name, process in children if process.poll() is not None), None)
            if exited:
                name, process = exited
                if name == 'combined_view' and process.returncode == 0:
                    outcome = 'viewer_closed'
                else:
                    outcome, code = 'child_exited', 1
                    print(f'{name} exited with {process.returncode}; inspect {args.output/name}.log',
                          file=sys.stderr, flush=True)
                break
            time.sleep(.1)
        if stopped:
            outcome = 'interrupted'
    except (OSError, ValueError, RuntimeError) as error:
        outcome, code = 'failed', 1
        print(f'Localized BEV failed: {error}', file=sys.stderr, flush=True)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # The child controllers handle their own workers/drivers. Their reused
        # sensor and inference processes are deliberately outside our ownership.
        # Give the nested controllers longer than their own 15-second worker
        # grace period so their report writes and driver cleanup can finish.
        stop_owned(children, grace_s=35.)
        if (args.output/'run_manifest.json').is_file():
            report = dict(outcome=outcome, exit_code=code, wall_duration_s=time.monotonic()-started,
                          owned_controller_exit_codes={name: p.poll() for name, p in children},
                          reused_processes_stopped=False, recorder_started=False,
                          localizer_report_present=(args.output/'localization/lidar_localization.json').is_file(),
                          **provenance)
            (args.output/'session_report.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'Session logs and localization reports: {args.output}', flush=True)
        if session_lock is not None:
            session_lock.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
