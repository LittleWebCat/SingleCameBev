#!/usr/bin/python3
"""Build a local LiDAR–IMU map from a bag or already-running Ouster driver."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from mapping_session import start_mapping, mapping_result, start_static_context
from replay_combined_bev import positive_float, positive_int, nonnegative_float, stop_owned


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bag', type=Path, help='recorded bag directory; one pass, no physical sensors')
    source.add_argument('--live', action='store_true', help='use existing /points and /imu; does not start drivers')
    parser.add_argument('--output', type=Path, help='new output directory; default maps/local_UTC')
    parser.add_argument('--rate', type=positive_float, default=1.0)
    parser.add_argument('--start-offset', type=nonnegative_float, default=0.0)
    parser.add_argument('--duration', type=positive_float, help='maximum wall-clock runtime in seconds')
    parser.add_argument('--domain', type=int, default=os.environ.get('ROS_DOMAIN_ID', '90'))
    parser.add_argument('--max-map-voxels', type=positive_int,
                        help='per-run map capacity; default 500000')
    parser.add_argument('--map-publish-period', type=positive_float,
                        help='seconds between map publications; default 2')
    args = parser.parse_args()
    if not 0 <= args.domain <= 231:
        parser.error('domain must be in [0, 231]')
    bag = args.bag.expanduser().resolve() if args.bag else None
    if bag and not (bag/'metadata.yaml').is_file():
        parser.error(f'bag metadata is missing: {bag}')
    if args.live and args.start_offset:
        parser.error('--start-offset is only for bags')
    if int(Path('/proc/sys/net/core/rmem_max').read_text()) < 16*1024**2:
        parser.error('DDS needs a 16 MiB receive limit; run sudo bash tools/setup_dds_transport.sh')
    env = os.environ.copy()
    env['ROS_DOMAIN_ID'] = str(args.domain)
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    children, output = [], None
    success = True
    try:
        context = start_static_context(children, root, env, bag, args.start_offset) if bag else None
        output, mapper = start_mapping(
            children, root, env, bag, args.output, not args.live,
            max_map_voxels=args.max_map_voxels,
            map_publish_period=args.map_publish_period,
            latest_scan_only=False if bag else None)
        deadline = time.monotonic()+3
        while time.monotonic() < deadline and not stopped:
            if mapper.poll() is not None:
                raise RuntimeError(f'mapping launch exited; inspect {output}/mapping.log')
            if context is not None and context.poll() is not None:
                raise RuntimeError('recorded static context helper exited before playback')
            time.sleep(0.1)
        if stopped:
            return 130
        if bag:
            command = ['ros2', 'bag', 'play', str(bag), '--clock', '100', '--rate', str(args.rate),
                       '--start-offset', str(args.start_offset), '--disable-keyboard-controls',
                       '--read-ahead-queue-size', '20', '--qos-profile-overrides-path',
                       str(root/'config/recording/raw_sensor_qos.yaml'),
                       '--topics', '/points', '/imu', '/metadata', '/tf_static']
            player = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                      start_new_session=True)
            children.append(('mapping bag player', player))
        started = time.monotonic()
        print(f'ROS domain: {args.domain}. Ctrl+C stops this run and saves its map.', flush=True)
        while not stopped and (not args.duration or time.monotonic()-started < args.duration):
            if mapper.poll() is not None:
                raise RuntimeError(f'mapping launch exited; inspect {output}/mapping.log')
            if context is not None and context.poll() is not None:
                raise RuntimeError('recorded static context helper exited')
            if bag and player.poll() is not None:
                success = player.returncode == 0
                # Drain the final scan/IMU callbacks before saving.
                deadline = time.monotonic()+2
                while time.monotonic() < deadline and not stopped:
                    time.sleep(0.1)
                break
            time.sleep(0.1)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Mapping failed: {error}', file=sys.stderr, flush=True)
        success = False
    finally:
        stop_owned(children)
        for _, child in children:
            if hasattr(child, 'mapping_lock'):
                child.mapping_lock.close()
        if output:
            success = mapping_result(output) and success
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
