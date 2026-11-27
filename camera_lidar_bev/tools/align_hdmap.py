#!/usr/bin/env python3
"""Align a labelled HD map (its own LiDAR-odometry frame, no GPS) to this project's GPS-aligned saved map (enu).

1. Coarse: the HD map drive's LiDAR positions (poses.csv) against our mapped route (gps_alignment/trajectory_enu.csv),
   both LiDAR positions on the same loop: 2-D ICP from 72 starting headings, then 3-D ICP (the HD frame is the sensor at
   its first scan, so it is tilted).
2. Fine: the HD map's paint (painted lane lines, crosswalk edges, stop lines, densified every 0.2 m) against paint in our
   map (road-level points brighter than the road around them), 6-DoF trimmed ICP with shrinking match distance.
3. Checks: paint and route residuals, our kept stop signs against the HD stop lines, and a top-down overlay image.
Writes MAP/hdmap/: alignment.json (T_enu_from_hdmap and the checks), hdmap3d_enu.json (every HD map layer in enu, same
schema as hdmap3d.json) and alignment_overlay.png. The HD map files themselves are not changed.

  /usr/bin/python3 tools/align_hdmap.py --hdmap ../hdmap_20261125_125612_20261125_1820 [--map maps/local_gps_recording_002]
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import cv2
import numpy as np
from scipy.spatial import cKDTree

from saved_map_source import read_pcd_xyz

ROOT = Path(__file__).resolve().parents[1]
PAINT_LAYERS = ('lane_line', 'crosswalk', 'stop_line')


def kabsch(a, b, planar=False):
    """Rigid T (4x4) minimising |T a - b| for matched rows; planar keeps rotation about z only."""
    ca, cb = a.mean(0), b.mean(0)
    if planar:
        h = (a[:, :2] - ca[:2]).T @ (b[:, :2] - cb[:2])
        u, _, vt = np.linalg.svd(h)
        r2 = vt.T @ np.diag([1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
        r = np.eye(3)
        r[:2, :2] = r2
    else:
        h = (a - ca).T @ (b - cb)
        u, _, vt = np.linalg.svd(h)
        r = vt.T @ np.diag([1, 1, np.sign(np.linalg.det(vt.T @ u.T))]) @ u.T
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, cb - r @ ca
    return t


def apply(t, p):
    return p @ t[:3, :3].T + t[:3, 3]


def icp(src, dst, t, dists, planar=False, iters=20, trim=0.8):
    tree = cKDTree(dst[:, :2] if planar else dst)
    for d in dists:
        for _ in range(iters):
            s = apply(t, src)
            q = s[:, :2] if planar else s
            dist, idx = tree.query(q, distance_upper_bound=d)
            ok = np.isfinite(dist)
            if ok.sum() < 20:
                break
            keep = ok & (dist <= np.quantile(dist[ok], trim))
            target = dst[idx[keep]].copy()
            if planar:
                target[:, 2] = s[keep, 2]
            t = kabsch(s[keep], target, planar) @ t
    s = apply(t, src)
    dist, _ = tree.query(s[:, :2] if planar else s)
    return t, dist


def densify(points, closed=False, step=0.2):
    p = np.asarray(points, float)
    if closed:
        p = np.vstack([p, p[:1]])
    out = []
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    out.append(p[-1:])
    return np.vstack(out)


def hd_paint(layers):
    pts = []
    for name in PAINT_LAYERS:
        for e in layers[name]:
            if name == 'lane_line' and e.get('mark_type') == 'NONE':
                continue
            pts.append(densify(e['points'], closed=bool(e.get('closed'))))
    return np.vstack(pts)


def map_paint(xyz, inten, route_xy, near_m=14.0, cell=1.0):
    """Road-level points of our map that are bright relative to the road in their 1 m cell."""
    tree = cKDTree(route_xy)
    near = tree.query(xyz[:, :2], distance_upper_bound=near_m)[0] < near_m
    p, i = xyz[near], inten[near]
    ij = np.floor(p[:, :2] / cell).astype(np.int64)
    key = ij[:, 0] * 1_000_003 + ij[:, 1]
    order = np.lexsort((p[:, 2], key))
    p, i, key = p[order], i[order], key[order]
    starts = np.r_[0, np.nonzero(np.diff(key))[0] + 1]
    counts = np.diff(np.r_[starts, len(key)])
    ground = np.repeat(p[starts + (counts * 0.1).astype(int), 2], counts)  # cell 10th-percentile height
    road = np.abs(p[:, 2] - ground) < 0.15
    med = np.zeros(len(p))
    for s, c in zip(starts, counts):
        r = road[s:s + c]
        med[s:s + c] = np.median(i[s:s + c][r]) if r.any() else np.inf
    paint = road & (i > 1.8 * med) & (i > med + 10)
    return p[paint]


def transform_layers(layers, t):
    out = copy.deepcopy(layers)
    for elements in out.values():
        for e in elements:
            e['points'] = np.round(apply(t, np.asarray(e['points'], float)), 3).tolist()
            for k in ('edge1', 'edge2'):
                if isinstance(e.get(k), list) and e[k] and isinstance(e[k][0], list):
                    e[k] = np.round(apply(t, np.asarray(e[k], float)), 3).tolist()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hdmap', type=Path, required=True)
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    args = parser.parse_args()
    hd = json.loads((args.hdmap / 'hdmap3d.json').read_text())
    poses = np.genfromtxt(args.hdmap / 'poses.csv', delimiter=',', names=True)
    hd_traj = np.c_[poses['T03'], poses['T13'], poses['T23']]
    r = np.genfromtxt(args.map / 'gps_alignment/trajectory_enu.csv', delimiter=',', names=True)
    route = np.c_[r['x_m'], r['y_m'], r['z_m']]

    # 1. Coarse: trajectories, many starting headings.
    best = None
    for yaw in np.radians(np.arange(0, 360, 5)):
        t0 = np.eye(4)
        c, s = math.cos(yaw), math.sin(yaw)
        t0[:2, :2] = [[c, -s], [s, c]]
        t0[:3, 3] = route.mean(0) - apply(t0, hd_traj.mean(0, keepdims=True))[0]
        t, d = icp(hd_traj, route, t0, (30.0, 10.0, 3.0), planar=True, iters=15)
        if best is None or np.median(d) < best[1]:
            best = (t, float(np.median(d)))
    t, _ = icp(hd_traj, route, best[0], (3.0, 1.5), planar=False)
    _, d_route = icp(hd_traj, route, t, (), planar=False)  # residuals only
    coarse = t.copy()

    # 2. Fine: paint against paint.
    xyz = np.asarray(read_pcd_xyz(args.map / 'gps_alignment/pointcloud_map_enu.pcd'), np.float64)
    src = args.map / 'gps_alignment/pointcloud_map_enu.pcd'
    inten = np.memmap(src, mode='r', dtype='<f4', offset=src.stat().st_size - len(xyz) * 16, shape=(len(xyz), 4))[:, 3].astype(np.float64)
    ours = map_paint(xyz, inten, route[:, :2])
    theirs = hd_paint(hd['layers'])
    t, d_paint = icp(theirs, ours, t, (1.5, 0.8, 0.4, 0.25), planar=False, trim=0.7)
    _, d_route_final = icp(hd_traj, route, t, (), planar=False)

    # 3. Checks.
    rot = t[:3, :3]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, rot[2, 2]))))
    yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    stops = []
    reviewed = args.map / 'landmarks/landmarks_reviewed.json'
    if reviewed.exists():
        hd_stops = [apply(t, np.asarray(e['points'], float)) for e in hd['layers']['stop_line']]
        for l in json.loads(reviewed.read_text())['landmarks']:
            if l.get('keep') and l['type'] == 'stop_sign':
                p = np.asarray(l['position_enu_m'][:2])
                dist = [float(np.min(np.linalg.norm(densify(s)[:, :2] - p, axis=1))) for s in hd_stops]
                k = int(np.argmin(dist))
                entry = {'stop_sign': l['id'], 'nearest_hd_stop_line': hd['layers']['stop_line'][k]['id'], 'sign_to_line_m': round(dist[k], 2)}
                if l.get('stop_line_estimated'):
                    est = np.asarray(l['stop_line_estimated']['endpoints_enu_m'], float).mean(0)[:2]
                    entry['our_estimated_line_to_hd_line_m'] = round(float(np.min(np.linalg.norm(densify(hd_stops[k])[:, :2] - est, axis=1))), 2)
                stops.append(entry)
    q = lambda d: {'median_m': round(float(np.median(d)), 3), 'p90_m': round(float(np.percentile(d, 90)), 3)}  # noqa: E731
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'hdmap': str(args.hdmap.resolve()), 'hdmap_version': hd.get('version'), 'hdmap_recording': hd.get('recording'),
        'map': args.map.name, 'T_enu_from_hdmap': np.round(t, 6).tolist(), 'T_coarse_from_trajectories': np.round(coarse, 6).tolist(),
        'yaw_deg': round(yaw, 3), 'tilt_deg': round(tilt, 3), 'translation_m': np.round(t[:3, 3], 3).tolist(),
        'route_residual_after_coarse': q(d_route), 'route_residual_final': q(d_route_final),
        'paint_points': {'hdmap': int(len(theirs)), 'our_map': int(len(ours))},
        'paint_residual_final': q(d_paint), 'paint_within_0.3m': round(float(np.mean(d_paint < 0.3)), 3),
        'stop_signs_vs_hd_stop_lines': stops,
        'note': 'route residuals include that the two drives used different lane positions; paint residuals are the alignment check'}

    out = args.map / 'hdmap'
    out.mkdir(exist_ok=True)
    (out / 'alignment.json').write_text(json.dumps(report, indent=1) + '\n')
    enu = dict(hd, frame={'name': 'enu', 'units': 'm', 'description': f'GPS-aligned enu of {args.map.name}, via T_enu_from_hdmap'},
               layers=transform_layers(hd['layers'], t), alignment='alignment.json')
    (out / 'hdmap3d_enu.json').write_text(json.dumps(enu) + '\n')

    # Overlay: our map paint (grey) and road-level points, HD layers coloured, route green.
    lo, cell = route[:, :2].min(0) - 30, 0.25
    w, h = ((route[:, :2].max(0) + 30 - lo) / cell).astype(int) + 1
    img = np.full((h, w, 3), 12, np.uint8)
    ij = ((ours[:, :2] - lo) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < w) & (ij[:, 1] < h)
    img[h - 1 - ij[ok, 1], ij[ok, 0]] = (150, 150, 150)
    colours = {'curb': (60, 60, 255), 'lane_line': (0, 220, 255), 'crosswalk': (255, 0, 255), 'stop_line': (0, 0, 255),
               'lane_centerline': (80, 200, 80), 'intersection': (255, 255, 0)}
    for name, col in colours.items():
        for e in enu['layers'][name]:
            p = np.asarray(e['points'])[:, :2]
            if e.get('closed'):
                p = np.vstack([p, p[:1]])
            uv = ((p - lo) / cell).astype(int)
            cv2.polylines(img, [np.c_[uv[:, 0], h - 1 - uv[:, 1]].reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(out / 'alignment_overlay.png'), img)
    print(json.dumps({k: report[k] for k in ('yaw_deg', 'tilt_deg', 'translation_m', 'route_residual_after_coarse', 'route_residual_final',
                                             'paint_points', 'paint_residual_final', 'paint_within_0.3m', 'stop_signs_vs_hd_stop_lines')}, indent=1))
    print(f'wrote {out}/alignment.json, hdmap3d_enu.json, alignment_overlay.png')


if __name__ == '__main__':
    main()
