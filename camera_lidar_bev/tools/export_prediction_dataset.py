#!/usr/bin/env python3
"""Export a recorded drive (tools/route_planner.py bag) to a ROS-free prediction dataset (.npz).

Run once on the car's computer (needs ROS 2 to read the bag); the .npz then works on any PC with
numpy + scipy via tools/eval_prediction.py --dataset. Each /camera_lidar_bev/lidar_detections_3d scan is
paired with the localization pose of the same scan (stamps within 30 ms), and its detections of
trackable classes (car, truck, bus, trailer, construction_vehicle, bicycle, motorcycle, person) within
--max-range-m are stored in the map frame (enu) with their scores; filter by score when loading.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/export_prediction_dataset.py --bag data/drive_20261124T213905Z/bag \
      [--out datasets/prediction/drive_20261124T213905Z.npz]
"""
import argparse
from datetime import datetime, timezone
import math
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from prediction_dataset import GROUP_OF, save_dataset
from prediction_models import quat_matrix, wrap

ROOT = Path(__file__).resolve().parents[1]
ODOM, DETS = '/camera_lidar_bev/localization/odometry', '/camera_lidar_bev/lidar_detections_3d'


def read_bag(bag, min_score=0.2, max_range_m=80.0):
    """(frame times, ego [x, y, z, yaw], detections [(frame, class, xyz, size, yaw, score)])."""
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    r.set_filter(rosbag2_py.StorageFilter(topics=[ODOM, DETS]))
    poses, scans = {}, []
    while r.has_next():
        topic, data, _ = r.read_next()
        m = deserialize_message(data, get_message(types[topic]))
        stamp = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
        if topic == ODOM:
            if m.child_frame_id == 'os_sensor':
                p = m.pose.pose.position
                poses[stamp] = (quat_matrix(m.pose.pose.orientation), np.array([p.x, p.y, p.z]))
        else:
            scans.append((stamp, m))
    keys = np.array(sorted(poses))
    frame_t, ego, dets = [], [], []
    for stamp, m in sorted(scans, key=lambda s: s[0]):
        i = int(np.clip(np.searchsorted(keys, stamp), 1, len(keys) - 1))
        k = keys[i] if abs(keys[i] - stamp) < abs(keys[i - 1] - stamp) else keys[i - 1]
        if abs(k - stamp) >= 30_000_000:
            continue  # no localization for this scan
        rot, pos = poses[k]
        ego_yaw = math.atan2(rot[1, 0], rot[0, 0])
        f = len(frame_t)
        frame_t.append(stamp * 1e-9)
        ego.append((pos[0], pos[1], pos[2], ego_yaw))
        for d in m.detections:
            if not d.results:
                continue
            cls, score = d.results[0].hypothesis.class_id, d.results[0].hypothesis.score
            c = d.bbox.center.position
            if cls not in GROUP_OF or score < min_score or math.hypot(c.x, c.y) > max_range_m:
                continue
            q = d.bbox.center.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            dets.append((f, cls, rot @ np.array([c.x, c.y, c.z]) + pos, (d.bbox.size.x, d.bbox.size.y, d.bbox.size.z),
                         wrap(yaw + ego_yaw), score))
    return frame_t, ego, dets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--out', type=Path, help='default: datasets/prediction/<drive>.npz')
    parser.add_argument('--min-score', type=float, default=0.2, help='lowest score kept; filter higher when loading')
    parser.add_argument('--max-range-m', type=float, default=80.0)
    args = parser.parse_args()
    frame_t, ego, dets = read_bag(args.bag, args.min_score, args.max_range_m)
    t = np.genfromtxt(args.map / 'gps_alignment/trajectory_enu.csv', delimiter=',', names=True)
    out = args.out or ROOT / 'datasets/prediction' / f'{args.bag.parent.name}.npz'
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {'source_bag': str(args.bag), 'map': args.map.name, 'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'min_score': args.min_score, 'max_range_m': args.max_range_m, 'frames': len(frame_t), 'detections': len(dets),
            'minutes': round((frame_t[-1] - frame_t[0]) / 60, 1) if frame_t else 0,
            'detector': 'CenterPoint-PointPillar nuScenes, 3 sweeps (config/perception/lidar_centerpoint_detector.yaml)',
            'frame': 'enu of the GPS-aligned saved map; ego_xyz_yaw is the LiDAR (os_sensor) pose'}
    save_dataset(out, frame_t, ego, dets, np.c_[t['x_m'], t['y_m'], t['z_m']], meta)
    print(f'{len(frame_t)} frames, {len(dets)} detections over {meta["minutes"]} min -> {out} '
          f'({out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
