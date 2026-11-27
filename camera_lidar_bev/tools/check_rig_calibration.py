#!/usr/bin/env python3
"""Check rig camera/LiDAR calibration on the live sensors: overlay + edge-alignment score.

Grabs one live /points scan (ROS, current ROS_DOMAIN_ID) and one frame per camera at its
calibrated resolution (cameras must be free), projects the scan with the imported calibration,
and scores how well LiDAR depth edges land on image edges. The same score is computed for the
alternative interpretation that the extrinsics' LiDAR frame were os_lidar (180 deg yaw), and for
a deliberately mis-rotated extrinsic (+3 deg yaw), so the number has a reference. Writes
OUTPUT/<camera>_overlay.jpg and OUTPUT/report.json.

  /usr/bin/python3 tools/check_rig_calibration.py config/calibration/rig_20261124 OUTPUT [--cameras ...]
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import yaml

SENSOR_FROM_LIDAR = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0.038195], [0, 0, 0, 1]], float)


def grab_scan():
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    rclpy.init()
    node = rclpy.create_node('rig_calibration_check')
    got = {}
    node.create_subscription(PointCloud2, '/points', lambda m: got.setdefault('m', m), qos_profile_sensor_data)
    end = time.monotonic() + 15
    while 'm' not in got and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    if 'm' not in got:
        raise SystemExit('no /points scan received')
    m = got['m']
    f = {x.name: x for x in m.fields}
    raw = np.frombuffer(bytes(m.data), np.uint8).reshape(-1, m.point_step)
    xyz = np.stack([raw[:, f[k].offset:f[k].offset + 4].copy().view('<f4')[:, 0] for k in 'xyz'], 1).astype(float)
    xyz = xyz[np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 1.0)]
    if m.header.frame_id == 'os_lidar':
        xyz = xyz @ SENSOR_FROM_LIDAR[:3, :3].T + SENSOR_FROM_LIDAR[:3, 3]
    return xyz  # os_sensor


def grab_frame(device, w, h):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    frame = None
    for _ in range(15):  # let auto exposure settle
        ok, img = cap.read()
        if ok:
            frame = img
    cap.release()
    if frame is None or frame.shape[1] != w or frame.shape[0] != h:
        raise RuntimeError(f'could not read {w}x{h} from {device}')
    return frame


def project(xyz, cam_from_sensor, k, d, w, h):
    pc = xyz @ cam_from_sensor[:3, :3].T + cam_from_sensor[:3, 3]
    front = pc[:, 2] > 0.5
    pc = pc[front]
    uv, _ = cv2.projectPoints(pc.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, d)
    uv = uv.reshape(-1, 2)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[inside], pc[inside, 2]


def edge_score(img, uv, depth):
    """Mean distance (px) from LiDAR depth-discontinuity points to the nearest image edge."""
    if len(uv) < 200:
        return None
    w, h = img.shape[1], img.shape[0]
    grid = np.full((h // 4 + 1, w // 4 + 1), np.inf)
    ui, vi = (uv[:, 0] // 4).astype(int), (uv[:, 1] // 4).astype(int)
    np.minimum.at(grid, (vi, ui), depth)
    near = grid.copy()
    near[~np.isfinite(near)] = 0
    gy = np.abs(np.diff(near, axis=0, prepend=near[:1]))
    gx = np.abs(np.diff(near, axis=1, prepend=near[:, :1]))
    jump = ((gx > 0.5) | (gy > 0.5)) & np.isfinite(grid) & (grid > 0)
    ys, xs = np.nonzero(jump)
    if len(xs) < 30:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    px = np.clip(xs * 4 + 2, 0, w - 1), np.clip(ys * 4 + 2, 0, h - 1)
    scale = w / 1024.0  # report in 1024-px-wide units so cameras compare
    return float(np.median(dist[px[1], px[0]]) / scale), int(len(xs))


def yaw(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('calibration', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--cameras', nargs='*')
    a = p.parse_args()
    manifest = json.loads((a.calibration / 'manifest.json').read_text())
    a.output.mkdir(parents=True, exist_ok=True)
    xyz = grab_scan()
    report = {'points': int(len(xyz)), 'cameras': {}}
    for name, c in manifest['cameras'].items():
        if a.cameras and name not in a.cameras:
            continue
        intr = yaml.safe_load((a.calibration / c['intrinsics']).read_text())
        k = np.array(intr['camera_matrix']['data'], float).reshape(3, 3)
        d = np.array(intr['distortion_coefficients']['data'], float)
        w, h = intr['image_width'], intr['image_height']
        ext = yaml.safe_load((a.calibration / c['extrinsics']).read_text())
        t = np.array(ext['transforms']['T_camera_from_os_sensor']['matrix'], float)
        try:
            img = grab_frame(c['device_by_path'], w, h)
        except RuntimeError as e:
            report['cameras'][name] = {'error': str(e)}
            print(f'{name:13s} skipped: {e}')
            continue
        scores = {}
        for label, transform in (('calibration (os_sensor)', t), ('if frame were os_lidar', t @ SENSOR_FROM_LIDAR),
                                 ('calibration +3 deg yaw', t @ yaw(3))):
            uv, depth = project(xyz, transform, k, d, w, h)
            scores[label] = edge_score(img, uv, depth)
        uv, depth = project(xyz, t, k, d, w, h)
        overlay = img.copy()
        lo, hi = np.percentile(depth, [3, 97])  # scene depth range, so object outlines change colour
        colors = cv2.applyColorMap(np.clip(255 - (depth - lo) / max(hi - lo, 0.5) * 255, 0, 255).astype(np.uint8).reshape(-1, 1),
                                   cv2.COLORMAP_JET).reshape(-1, 3)
        for (u, v), col in zip(uv.astype(int)[::2], colors[::2]):
            cv2.circle(overlay, (int(u), int(v)), max(1, w // 800), tuple(int(x) for x in col), -1)
        cv2.putText(overlay, f'{name} {w}x{h}  LiDAR depth {lo:.1f} m (red) to {hi:.1f} m (blue)', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, w / 1600, (255, 255, 255), 2)
        out = a.output / f'{name}_overlay.jpg'
        cv2.imwrite(str(out), cv2.resize(overlay, (1280, int(1280 * h / w))), [cv2.IMWRITE_JPEG_QUALITY, 85])
        report['cameras'][name] = {'points_in_image': int(len(uv)), 'overlay': out.name,
                                   'edge_distance_px_per_1024': {k2: (v[0] if v else None) for k2, v in scores.items()},
                                   'depth_edge_samples': (scores['calibration (os_sensor)'] or (None, 0))[1]}
        s = report['cameras'][name]['edge_distance_px_per_1024']
        fmt = lambda v: f'{v:5.1f}' if v is not None else '  n/a'
        print(f'{name:13s} {len(uv):6d} pts in image | edge distance (lower = better): calibration {fmt(s["calibration (os_sensor)"])}'
              f'  +3deg {fmt(s["calibration +3 deg yaw"])}  os_lidar {fmt(s["if frame were os_lidar"])}')
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
