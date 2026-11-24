#!/usr/bin/env python3
"""Extract recorded Ouster scans plus saved-trajectory poses for offline multi-sweep tests.

Writes OUTPUT/scan_NNNNN.bin (float32 [N,4]: os_sensor x, y, z, reflectivity/255, decoded and
self-masked like the live detector node) and OUTPUT/scans.csv with each scan's start/end time
and the os_sensor pose in the map at scan end, interpolated from a FAST-LIO trajectory.csv
the same way the combined viewer does (exact sample or a short bracket, no extrapolation).

Run with the system Python and ROS Jazzy sourced (rosbag2_py), for example:
  /usr/bin/python3 tools/extract_lidar_sweeps.py bags/camera_lidar_20261010T221427_088256Z \
      maps/local_gps_recording_002/trajectory.csv OUTPUT --start 20 --end 80
"""
import argparse
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2

SELF_MASK = ((-2.131, 2.350), (-0.905, 0.885), (-1.8, 0.0))  # os_sensor, as the detector config


def slerp(a, b, f):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + f * (b - a)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - f) * th) * a + np.sin(f * th) * b) / np.sin(th)


def pose_at(stamps, pos, quat, t_ns, max_gap_ns=150_000_000):
    i = int(np.searchsorted(stamps, t_ns))
    cands = [k for k in (i - 1, i) if 0 <= k < len(stamps)]
    k = min(cands, key=lambda k: abs(int(stamps[k]) - t_ns))
    if abs(int(stamps[k]) - t_ns) <= 1_000_000:
        return pos[k], quat[k]
    if i == 0 or i == len(stamps) or stamps[i] - stamps[i - 1] > max_gap_ns:
        return None
    f = (t_ns - stamps[i - 1]) / (stamps[i] - stamps[i - 1])
    return pos[i - 1] + f * (pos[i] - pos[i - 1]), slerp(quat[i - 1], quat[i], f)


def decode(msg):
    fields = {f.name: f for f in msg.fields}
    raw = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.row_step)[:, :msg.width * msg.point_step]
    raw = raw.reshape(-1, msg.point_step)
    get = lambda name, dt: raw[:, fields[name].offset:fields[name].offset + np.dtype(dt).itemsize].copy().view(dt)[:, 0]
    x, y, z = (get(n, '<f4') for n in 'xyz')
    refl = get('reflectivity', '<u2').astype(np.float32)
    t_off = get('t', '<u4')
    if msg.header.frame_id == 'os_lidar':  # driver's fixed os_sensor <- os_lidar transform
        x, y, z = -x, -y, z + 0.038195
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (refl <= 255) & (x * x + y * y + z * z >= 0.25)
    (x0, x1), (y0, y1), (z0, z1) = SELF_MASK
    ok &= ~((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1))
    pts = np.stack([x[ok], y[ok], z[ok], refl[ok] / 255.0], 1).astype(np.float32)
    start = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    return pts, start, start + int(t_off.max())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('bag', type=Path)
    p.add_argument('trajectory', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--start', type=float, default=0.0, help='seconds after the first scan')
    p.add_argument('--end', type=float, default=1e9, help='seconds after the first scan')
    a = p.parse_args()
    traj = np.genfromtxt(a.trajectory, delimiter=',', names=True)
    stamps = np.round(traj['stamp_sec'] * 1e9).astype(np.int64)
    pos = np.stack([traj['x_m'], traj['y_m'], traj['z_m']], 1)
    quat = np.stack([traj['qx'], traj['qy'], traj['qz'], traj['qw']], 1)
    a.output.mkdir(parents=True, exist_ok=False)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(a.bag), storage_id='mcap'),
                rosbag2_py.ConverterOptions('cdr', 'cdr'))
    reader.set_filter(rosbag2_py.StorageFilter(topics=['/points']))
    first = None
    rows, kept, posed = [], 0, 0
    while reader.has_next():
        _, data, _ = reader.read_next()
        msg = deserialize_message(data, PointCloud2)
        pts, start, end = decode(msg)
        first = first if first is not None else start
        rel = (start - first) / 1e9
        if rel < a.start:
            continue
        if rel > a.end:
            break
        pose = pose_at(stamps, pos, quat, end)
        name = f'scan_{kept:05d}.bin'
        pts.tofile(a.output / name)
        (px, py, pz), (qx, qy, qz, qw) = pose if pose is not None else ((0, 0, 0), (0, 0, 0, 1))
        rows.append(f'{kept},{start},{end},{px:.6f},{py:.6f},{pz:.6f},{qx:.9f},{qy:.9f},{qz:.9f},{qw:.9f},'
                    f'{int(pose is not None)},{name}')
        kept += 1
        posed += pose is not None
    (a.output / 'scans.csv').write_text('index,start_ns,end_ns,x,y,z,qx,qy,qz,qw,pose_ok,file\n' + '\n'.join(rows) + '\n')
    print(f'{kept} scans ({posed} with poses) -> {a.output}')


if __name__ == '__main__':
    main()
