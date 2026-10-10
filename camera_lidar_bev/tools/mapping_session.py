"""Prepare one local mapping run from the selected Ouster factory metadata."""
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import yaml


def prepare_mapping(root, bag=None, output=None, *, max_map_voxels=None,
                    map_publish_period=None, latest_scan_only=None):
    if max_map_voxels is not None and (
            not isinstance(max_map_voxels, int) or isinstance(max_map_voxels, bool)
            or max_map_voxels <= 0):
        raise ValueError('max_map_voxels must be a positive integer')
    if map_publish_period is not None and (
            not math.isfinite(map_publish_period) or map_publish_period <= 0):
        raise ValueError('map_publish_period must be finite and positive')
    if latest_scan_only is not None and not isinstance(latest_scan_only, bool):
        raise ValueError('latest_scan_only must be a boolean')
    root = Path(root)
    if output is None:
        name = datetime.now(timezone.utc).strftime('local_%Y%m%dT%H%M%S_%fZ')
        output = root/'maps'/name
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite mapping run: {output}')
    config = yaml.safe_load((root/'config/mapping/fast_lio_ouster.yaml').read_text())
    params = config['/**']['ros__parameters']
    if latest_scan_only is not None:
        params['common']['latest_scan_only'] = latest_scan_only
    adapter = {}
    metadata_path = Path(str(bag)+'.capture')/'ouster_metadata.json' if bag else None
    if metadata_path is not None and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        lidar = np.array(metadata['lidar_intrinsics']['lidar_to_sensor_transform'], dtype=float).reshape(4, 4)
        imu = np.array(metadata['imu_intrinsics']['imu_to_sensor_transform'], dtype=float).reshape(4, 4)
        lidar[:3, 3] /= 1000
        imu[:3, 3] /= 1000
        imu_from_lidar = np.linalg.inv(imu)@lidar
        imu_from_sensor = np.linalg.inv(imu)
        if not np.allclose(imu_from_sensor[:3, :3], np.eye(3), atol=1e-6):
            raise ValueError('adapter currently supports the Ouster identity IMU-to-sensor rotation')
        adapter = {
            'imu_from_sensor_translation': imu_from_sensor[:3, 3].tolist(),
            'imu_from_lidar_translation': imu_from_lidar[:3, 3].tolist(),
            'imu_from_lidar_rotation': imu_from_lidar[:3, :3].ravel().tolist(),
        }
        params['mapping']['extrinsic_T'] = adapter['imu_from_lidar_translation']
        params['mapping']['extrinsic_R'] = adapter['imu_from_lidar_rotation']
        fmt = metadata['lidar_data_format']
        params['preprocess']['scan_line'] = int(fmt['pixels_per_column'])
        mode = metadata['config_params']['lidar_mode']
        params['preprocess']['scan_rate'] = int(mode.split('x')[1])
    if max_map_voxels is not None:
        adapter['max_map_voxels'] = max_map_voxels
    if map_publish_period is not None:
        adapter['map_publish_period'] = float(map_publish_period)
    output.mkdir(parents=True)
    (output/'fast_lio_runtime.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (output/'adapter_runtime.yaml').write_text(yaml.safe_dump(
        {'mapping_adapter': {'ros__parameters': adapter}}, sort_keys=False))
    manifest = {'bag': str(bag) if bag else None, 'output_directory': str(output),
                'factory_metadata': str(metadata_path) if metadata_path and metadata_path.is_file() else None,
                'backend': 'FAST-LIO2', 'georeferenced': False, 'loop_closure': False,
                'timestamp_offset_seconds': 0.0, 'clock_alignment': 'unverified host-derived timestamps',
                'source_revision': (root/'src/fast_lio/UPSTREAM_COMMIT').read_text().strip()}
    (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return output


def mapping_launch(output, use_sim_time=True):
    return ['ros2', 'launch', 'bev_bringup', 'lidar_imu_mapping.launch.py',
            f'use_sim_time:={str(use_sim_time).lower()}', f'output_directory:={output}',
            f'config_file:={output}/fast_lio_runtime.yaml',
            f'adapter_config:={output}/adapter_runtime.yaml']


def start_static_context(children, root, env, bag, start_offset):
    """Keep the recording's earlier latched context available after seeking."""
    if start_offset <= 0:
        return None
    process = subprocess.Popen([
        '/usr/bin/python3', str(Path(root)/'tools/replay_static_context.py'),
        '--bag', str(bag), '--start-offset', str(start_offset)],
        cwd=root, env=env, stdin=subprocess.DEVNULL, start_new_session=True)
    children.append(('recorded static context', process))
    return process


def start_mapping(children, root, env, bag=None, output=None, use_sim_time=True, *,
                  max_map_voxels=None, map_publish_period=None, latest_scan_only=None):
    from ament_index_python.packages import get_package_prefix, PackageNotFoundError
    try:
        for package in ('fast_lio', 'bev_mapping'):
            get_package_prefix(package)
    except PackageNotFoundError as error:
        raise RuntimeError('Build fast_lio, bev_mapping and bev_bringup first; '
                           'see docs/lidar_imu_mapping.md') from error
    domain = env.get('ROS_DOMAIN_ID', '0')
    lock = Path(f'/tmp/camera_lidar_bev_mapping_{os.getuid()}_{domain}.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError(f'another mapping controller already owns domain {domain}')
    try:
        output = prepare_mapping(
            root, bag, output, max_map_voxels=max_map_voxels,
            map_publish_period=map_publish_period, latest_scan_only=latest_scan_only)
        with (output/'mapping.log').open('x') as log:
            process = subprocess.Popen(mapping_launch(output, use_sim_time), cwd=root, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
    except BaseException:
        lock.close()
        raise
    children.append(('local mapping', process))
    process.mapping_lock = lock
    print(f'Local mapping: {output}\nMapper log: {output}/mapping.log\n'
          'Keep the rig still during initialization; this run has no GPS alignment or loop closure.',
          flush=True)
    return output, process


def mapping_result(output):
    report_file = Path(output)/'mapping_report.json'
    if not report_file.is_file():
        print(f'Mapper did not finish saving; inspect {output}/mapping.log', flush=True)
        return False
    try:
        report = json.loads(report_file.read_text())
        count, poses = report['map_point_count'], report['counters']['poses']
    except (OSError, ValueError, KeyError) as error:
        print(f'Cannot read mapping result: {error}; inspect {output}/mapping.log', flush=True)
        return False
    print(f'Map output: {output}/pointcloud_map.pcd ({count} points)\n'
          f'Trajectory: {output}/trajectory.csv ({poses} poses)', flush=True)
    if not report['map_point_count']:
        print('No map points were produced. Check input topics and allow a quiet initialization interval.', flush=True)
    return bool(report['map_point_count']) and not report['failed_input']
