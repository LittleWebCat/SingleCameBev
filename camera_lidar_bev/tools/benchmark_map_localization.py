#!/usr/bin/python3
"""Compare GPS conversion and independent scan registration on sampled route sections.

The saved trajectory is used only after registration, as a correlated comparison
reference. It is never passed to the localizer or used to choose its initial yaw.
The map-building recording is an optimistic test, not independent accuracy truth.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from align_local_map_gps import geographic_to_enu, load_trajectory, read_gps
from lidar_map_localizer import MapLocalizer, read_sensor_scan, sensor_from_metadata


def statistics(values):
    values = np.asarray(values, float)
    return None if not len(values) else {
        'count': len(values), 'median': float(np.median(values)),
        'p95': float(np.percentile(values, 95)), 'maximum': float(values.max())}


def select_scans(bag, windows, count, step):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2, Imu
    metadata = yaml.safe_load((bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    beginning = int(metadata['starting_time']['nanoseconds_since_epoch'])
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=metadata['storage_identifier']),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/points', '/imu']))
    groups = []
    for window in windows:
        reader.seek(beginning+int((window-.3)*1e9))
        imu, scans, previous = [], [], -np.inf
        limit = beginning+int((window+count*step+1)*1e9)
        while reader.has_next():
            topic, payload, receipt = reader.read_next()
            if receipt > limit:
                break
            if topic == '/imu':
                message = deserialize_message(payload, Imu)
                stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
                gyro = [message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z]
                imu.append((stamp, gyro))
            elif receipt >= beginning+int(window*1e9) and len(scans) < count:
                message = deserialize_message(payload, PointCloud2)
                stamp = message.header.stamp.sec+message.header.stamp.nanosec*1e-9
                if stamp-previous >= step-.03:
                    scans.append(message)
                    previous = stamp
        if not scans:
            raise ValueError(f'no point clouds near recording offset {window}s')
        groups.append((window, scans, np.asarray([t for t,_ in imu]), np.asarray([g for _,g in imu])))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True, type=Path)
    parser.add_argument('--bag', required=True, type=Path)
    parser.add_argument('--windows', default='40,75,100,140,175')
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--step', type=float, default=.2)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.map, args.bag = args.map.resolve(), args.bag.resolve()
    windows = [float(value) for value in args.windows.split(',')]
    if args.samples < 2 or not .05 <= args.step <= 1. or any(value < .3 for value in windows):
        parser.error('samples >=2, step .05..1 and window offsets >=.3 are required')
    if args.output.exists():
        parser.error('choose a new output directory')
    metadata = json.loads((Path(str(args.bag)+'.capture')/'ouster_metadata.json').read_text())
    factory = sensor_from_metadata(metadata)
    imu_rotation = np.asarray(metadata['imu_intrinsics']['imu_to_sensor_transform'], float).reshape(4,4)[:3,:3]
    prepared = time.perf_counter()
    localizer = MapLocalizer.from_directory(args.map)
    map_startup_ms = (time.perf_counter()-prepared)*1000
    gps, sigmas, known, gps_statistics = read_gps(args.bag)
    gps_enu = geographic_to_enu(gps[:,1:4], localizer.origin_wgs84)
    gps_ms = []
    for row in gps:
        before = time.perf_counter()
        geographic_to_enu(row[None,1:4], localizer.origin_wgs84)
        gps_ms.append((time.perf_counter()-before)*1000)
    groups = select_scans(args.bag, windows, args.samples, args.step)
    # Reference loaded separately only for report metrics, never initialization.
    reference, _ = load_trajectory(args.map/'gps_alignment/trajectory_enu.csv')
    samples = []
    for window, scans, imu_stamps, gyros in groups:
        localizer.reset()
        for cloud in scans:
            source = cloud.header.stamp.sec+cloud.header.stamp.nanosec*1e-9
            selected = (imu_stamps >= source-.05) & (imu_stamps <= source+.12)
            gyro = imu_rotation@gyros[selected].mean(axis=0) if selected.any() else None
            velocity = (localizer.transform[:3,:3].T@localizer.velocity
                        if localizer.transform is not None else None)
            before = time.perf_counter()
            points, stamp = read_sensor_scan(cloud, factory, gyro, velocity)
            decode_ms = (time.perf_counter()-before)*1000
            index = int(np.searchsorted(gps[:,0], stamp, side='right')-1)
            prior = gps_enu[index] if index >=0 and stamp-gps[index,0] <=1. else None
            result = localizer.localize(points, stamp, prior, gyro)
            row = {key:value for key,value in result.items() if key not in ('transform','covariance')}
            row.update(window_offset_s=window, decode_ms=decode_ms,
                       total_compute_ms=decode_ms+result.get('elapsed_ms',0))
            upper = int(np.searchsorted(reference[:,0], stamp))
            if 0 < upper < len(reference) and reference[upper,0]-reference[upper-1,0] <=.25:
                fraction=(stamp-reference[upper-1,0])/(reference[upper,0]-reference[upper-1,0])
                ref_position=reference[upper-1,1:4]+fraction*(reference[upper,1:4]-reference[upper-1,1:4])
                row['gps_vs_correlated_lio_horizontal_m']=float(np.linalg.norm(prior[:2]-ref_position[:2])) if prior is not None else None
                row['reference_enu_xyz_m']=ref_position.tolist()
                if result.get('accepted'):
                    transform=result['transform']
                    row['estimated_enu_xyz_m']=transform[:3,3].tolist()
                    row['lidar_vs_correlated_lio_horizontal_m']=float(np.linalg.norm(transform[:2,3]-ref_position[:2]))
                    quaternion=reference[upper-1,4:8]
                    ref_rotation=Rotation.from_quat(quaternion).as_matrix()
                    row['yaw_vs_correlated_lio_deg']=float(np.degrees(np.arctan2(
                        np.sin(np.arctan2(transform[1,0],transform[0,0])-np.arctan2(ref_rotation[1,0],ref_rotation[0,0])),
                        np.cos(np.arctan2(transform[1,0],transform[0,0])-np.arctan2(ref_rotation[1,0],ref_rotation[0,0])))))
            for key,value in list(row.items()):
                if isinstance(value,float) and not np.isfinite(value):
                    row[key]=None
            samples.append(row)
        print(json.dumps({'offset':window,'accepted':sum(r.get('accepted',False) for r in samples if r['window_offset_s']==window),
                          'samples':len(scans)}),flush=True)
    source_bag=Path(json.loads((args.map/'run_manifest.json').read_text())['bag']).resolve()
    steady=[row for row in samples if row.get('initializing') is False]
    bootstrap=[row for row in samples if row.get('initializing') is True]
    report={'map':str(args.map),'bag':str(args.bag),'same_recording_as_map':args.bag==source_bag,
            'accuracy_independently_validated':False,'reference':'correlated mapping LIO, evaluated after registration only',
            'saved_trajectory_used_for_localization':False,'initial_heading':'24 geometry hypotheses; no saved yaw',
            'timing_excludes':'MCAP reads, ROS deserialization/DDS, GUI and sensor acquisition',
            'map_startup_ms':map_startup_ms,'map_voxels':len(localizer.map_points),
            'gps_coordinate_conversion_ms':statistics(gps_ms),'gps_input':gps_statistics,
            'receiver_reported_horizontal_sigma_m':statistics(sigmas[known]),
            'lidar_steady_total_compute_ms':statistics([row['total_compute_ms'] for row in steady]),
            'lidar_bootstrap_total_compute_ms':statistics([row['total_compute_ms'] for row in bootstrap]),
            'accepted':sum(row.get('accepted',False) for row in samples),'total':len(samples),
            'states':dict(Counter(row['status'] for row in samples)),
            'gps_vs_correlated_lio_horizontal_m':statistics([row['gps_vs_correlated_lio_horizontal_m'] for row in samples if row.get('gps_vs_correlated_lio_horizontal_m') is not None]),
            'lidar_vs_correlated_lio_horizontal_m':statistics([row['lidar_vs_correlated_lio_horizontal_m'] for row in samples if row.get('lidar_vs_correlated_lio_horizontal_m') is not None]),
            'samples':samples}
    args.output.mkdir(parents=True)
    (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for row in samples:
        stamp=row['stamp_sec']-gps[0,0]
        axes[0].scatter(stamp,row['total_compute_ms'],s=12,c='#e87722' if row.get('initializing') else '#287abb')
        if row.get('gps_vs_correlated_lio_horizontal_m') is not None:
            axes[1].scatter(stamp,row['gps_vs_correlated_lio_horizontal_m'],s=12,c='#287abb')
        if row.get('lidar_vs_correlated_lio_horizontal_m') is not None:
            axes[1].scatter(stamp,row['lidar_vs_correlated_lio_horizontal_m'],s=12,c='#e87722')
    axes[0].set(title='LiDAR compute: orange bootstrap, blue tracking',ylabel='Milliseconds',yscale='log',xlabel='Recorded source offset (s)')
    axes[1].set(title='GPS blue / LiDAR orange vs correlated LIO',ylabel='Horizontal difference (m)',xlabel='Recorded source offset (s)')
    axes[2].plot(reference[:,1],reference[:,2],c='#8d8d8d',label='Map-building LIO reference')
    selected_gps=[row for row in samples if row.get('accepted') and row.get('estimated_enu_xyz_m')]
    if selected_gps:
        estimated=np.array([row['estimated_enu_xyz_m'] for row in selected_gps])
        axes[2].scatter(estimated[:,0],estimated[:,1],s=13,c='#e87722',label='Independent ICP estimates')
    axes[2].set(aspect='equal',xlabel='East (m)',ylabel='North (m)',title='Sampled localization route')
    axes[2].legend(fontsize=8)
    for axis in axes: axis.grid(alpha=.25)
    fig.suptitle('GPS vs frozen-map LiDAR localization — same-recording performance test; not independent accuracy truth')
    fig.tight_layout()
    fig.savefig(args.output/'comparison.png',dpi=140)
    plt.close(fig)
    print(json.dumps({key:value for key,value in report.items() if key not in ('samples','gps_input')},indent=2),flush=True)


if __name__=='__main__':
    main()
