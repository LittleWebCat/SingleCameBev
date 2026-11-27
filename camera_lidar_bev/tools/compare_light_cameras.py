#!/usr/bin/env python3
"""Which front camera to use for traffic-light colour: narrow, wide or both (ROS-free; numpy + opencv + matplotlib).

1. Measured (front_wide, data/traffic_light_eval_*/frames.jsonl from tools/traffic_light_camera_view.py): how often a
   mapped light in the field of view was detected, against how tall it appears at the model input (the state model
   runs at 640 px, so the 1920 px wide image is shrunk 3x). The apparent height is k / range with k from the matched
   detection boxes (box height x range), i.e. the physical signal-head height.
2. Geometry (rig calibration, config/calibration/<rig>): each signal's kept lights projected into both cameras from the
   car's pose along the mapped route, front bumper 0-120 m before the stop line (planner_core.load_signals): in
   the image or not (lens distortion included, a 3 % edge margin), and the apparent height at the model input.
3. Predicted per-frame detection along each approach: the measured rate at that apparent height (the narrow camera's
   larger image of a light treated like a closer light for the wide camera; capped at the best measured rate). A
   signal counts as seen when any of its lights is; 'both' assumes the two cameras miss independently.
It reports, per signal and camera, the farthest distance from the line from which detection stays >= 50 % per frame
up to the line, against the distance the planner needs at 10 / 20 / 25 mph (comfortable stop at 1.5 m/s^2 plus
0.5 s to accept a colour). Not modelled: occlusion, exposure / glare (the narrow camera saturated in sun on manual
exposure 300), and colour errors. Writes evaluation/light_camera_choice.json and .png.

  /usr/bin/python3 tools/compare_light_cameras.py [--eval data/traffic_light_eval_<UTC>] [--rig config/calibration/rig_20261125]
  # after a drive with both cameras (tools/run_light_camera_test.sh): measured side by side, and 'both'
  /usr/bin/python3 tools/compare_light_cameras.py --eval data/traffic_light_eval_<UTC>_front_wide --narrow-eval data/traffic_light_eval_<UTC>_front_narrow
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from planner_core import PlannerConfig, load_signals
from route_geometry import Route

ROOT = Path(__file__).resolve().parents[1]
MODEL_PX = 640
CAMERAS = {'wide': 'front_wide_1920x1080', 'narrow': 'front_narrow_1024x768'}
SPEEDS = {'10 mph': 4.47, '20 mph': 8.94, '25 mph': 11.18}


def load_camera(rig, stem):
    intr = yaml.safe_load((rig / f'{stem}.yaml').read_text())
    name = stem.split('_1')[0].rsplit('_', 0)[0]
    extr = yaml.safe_load((rig / f'{name}_T_os_sensor.yaml').read_text())
    k = np.array(intr['camera_matrix']['data'], float).reshape(3, 3)
    d = np.array(intr['distortion_coefficients']['data'], float)
    t = np.array(extr['transforms']['T_camera_from_os_sensor']['matrix'], float)
    w, h = intr['image_width'], intr['image_height']
    return {'name': name, 'k': k, 'd': d, 'T': t, 'w': w, 'h': h, 'scale': MODEL_PX / max(w, h)}


def measured_curve(eval_dir, fx_eval):
    """(apparent heights at the model input in px, detection rate, count) for the wide camera, plus k (m)."""
    rows, prods = [], []
    for line in (eval_dir / 'frames.jsonl').open():
        f = json.loads(line)
        if not f.get('localized'):
            continue
        for m in f.get('mapped') or []:
            rows.append((m['distance_m'], bool(m['detected'])))
            if m['detected']:
                u, v = m['uv']
                near = [b['box'] for b in f.get('detections') or []
                        if b['box'][0] - 25 <= u <= b['box'][2] + 25 and b['box'][1] - 25 <= v <= b['box'][3] + 25]
                if near:
                    b = min(near, key=lambda b: abs((b[0] + b[2]) / 2 - u) + abs((b[1] + b[3]) / 2 - v))
                    prods.append((b[3] - b[1]) * m['distance_m'] / fx_eval)
    k = float(np.median(prods))  # physical height of the detected box (m)
    dist = np.array([r[0] for r in rows])
    det = np.array([r[1] for r in rows])
    edges = np.array([0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 120])
    centres, rates, counts = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (dist >= a) & (dist < b)
        if sel.sum() >= 20:
            centres.append(k * fx_eval * (MODEL_PX / 1920) / ((a + b) / 2))
            rates.append(float(det[sel].mean()))
            counts.append(int(sel.sum()))
    order = np.argsort(centres)
    return np.array(centres)[order], np.array(rates)[order], np.array(counts)[order], k, len(prods)


def measured_pair(wide_dir, narrow_dir, max_dt_ns=60_000_000):
    """Measured detection of mapped lights by distance for each camera and for both together (same light, frames
    within 60 ms; a light out of one camera's view counts as missed by that camera)."""
    def load(d):
        rows = {}
        for line in (d / 'frames.jsonl').open():
            f = json.loads(line)
            if f.get('localized'):
                for m in f.get('mapped') or []:
                    rows.setdefault(m['id'], []).append((f['stamp_ns'], m['distance_m'], bool(m['detected'])))
        return {k: sorted(v) for k, v in rows.items()}
    w, n = load(wide_dir), load(narrow_dir)
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 120)]
    out = {}
    for name, src, other in (('wide', w, n), ('narrow', n, w), ('both', w, n)):
        res = {f'{a}-{b} m': [0, 0] for a, b in bins}
        lights = set(src) | set(other) if name == 'both' else set(src)
        for lid in lights:
            if name != 'both':
                for _, dist, det in src.get(lid, []):
                    for a, b in bins:
                        if a <= dist < b:
                            res[f'{a}-{b} m'][0] += 1
                            res[f'{a}-{b} m'][1] += det
                continue
            a_rows, b_rows = src.get(lid, []), other.get(lid, [])  # union over time: each frame of either camera
            b_t = np.array([r[0] for r in b_rows]) if b_rows else np.array([])
            a_t = np.array([r[0] for r in a_rows]) if a_rows else np.array([])
            for rows_x, t_y, rows_y in ((a_rows, b_t, b_rows), (b_rows, a_t, a_rows)):
                for t, dist, det in rows_x:
                    j = int(np.argmin(np.abs(t_y - t))) if len(t_y) else -1
                    det_y = j >= 0 and abs(t_y[j] - t) <= max_dt_ns and rows_y[j][2]
                    for a, b in bins:
                        if a <= dist < b:
                            res[f'{a}-{b} m'][0] += 1
                            res[f'{a}-{b} m'][1] += bool(det or det_y)
        out[name] = {k: {'frames': v[0], 'rate': round(v[1] / v[0], 3) if v[0] else None} for k, v in res.items()}
    return out


def rate_at(h_px, curve):
    hs, rs = curve
    if h_px <= hs[0]:
        return float(rs[0] * h_px / hs[0])
    # monotone in size: running maximum so a larger light is never predicted worse than a smaller one
    return float(np.interp(h_px, hs, np.maximum.accumulate(rs)))


def project(cam, p_sensor):
    """(in image, pixel (u, v), range along the optical axis)."""
    p = cam['T'][:3, :3] @ p_sensor + cam['T'][:3, 3]
    if p[2] < 1.0:
        return False, None, p[2]
    u0, v0 = cam['k'][0, 0] * p[0] / p[2] + cam['k'][0, 2], cam['k'][1, 1] * p[1] / p[2] + cam['k'][1, 2]
    if not (-0.2 * cam['w'] < u0 < 1.2 * cam['w'] and -0.2 * cam['h'] < v0 < 1.2 * cam['h']):
        return False, None, p[2]
    uv, _ = cv2.projectPoints(p.reshape(1, 1, 3), np.zeros(3), np.zeros(3), cam['k'], cam['d'])
    u, v = uv.reshape(2)
    m = 0.03
    return bool(m * cam['w'] <= u <= (1 - m) * cam['w'] and m * cam['h'] <= v <= (1 - m) * cam['h']), (float(u), float(v)), p[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--eval', type=Path, help='default: newest data/traffic_light_eval_*')
    parser.add_argument('--rig', type=Path, default=ROOT / 'config/calibration/rig_20261125')
    parser.add_argument('--narrow-eval', type=Path, help='measured narrow-camera run (tools/run_light_camera_test.sh) to compare with --eval')
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    args = parser.parse_args()
    eval_dir = args.eval or sorted((ROOT / 'data').glob('traffic_light_eval_*'))[-1]
    cams = {k: load_camera(args.rig, v) for k, v in CAMERAS.items()}
    hs, rs, counts, k_m, n_boxes = measured_curve(eval_dir, cams['wide']['k'][0, 0])
    curve = (hs, rs)
    route = Route(args.map / 'gps_alignment/trajectory_enu.csv')
    cfg = PlannerConfig()
    signals = load_signals(route, args.map, cfg)
    lights = {l['id']: l for l in json.loads((args.map / 'landmarks/landmarks_reviewed.json').read_text())['landmarks']}
    need = {k: v * v / (2 * cfg.comfort_decel_mps2) + 0.5 * v for k, v in SPEEDS.items()}
    report = {'eval': str(eval_dir), 'rig': str(args.rig), 'signal_head_height_m': round(k_m, 2), 'matched_boxes': n_boxes,
              'measured_wide': [{'height_px_at_model': round(float(h), 1), 'rate': round(float(r), 3), 'n': int(n)}
                                for h, r, n in zip(hs, rs, counts)],
              'cameras': {k: {'fx': round(float(c['k'][0, 0]), 1), 'width': c['w'],
                              'hfov_deg': round(math.degrees(2 * math.atan(c['w'] / 2 / c['k'][0, 0])), 1),
                              'vfov_deg': round(math.degrees(2 * math.atan(c['h'] / 2 / c['k'][1, 1])), 1),
                              'fx_at_model_input': round(float(c['k'][0, 0] * c['scale']), 1)} for k, c in cams.items()},
              'needed_decision_distance_m': {k: round(v, 1) for k, v in need.items()}, 'signals': {}}
    d_line = np.arange(0.0, 120.01, 1.0)  # front bumper distance to the stop line
    traces = {}
    for sig in signals:
        per = {'wide': [], 'narrow': []}
        for dl in d_line:
            s = sig['line_s'] - cfg.lidar_to_front_m - dl
            x, y, z, h, _ = (float(q) for q in route.at(s))
            c, sn = math.cos(h), math.sin(h)
            for cam_name, cam in cams.items():
                miss = 1.0
                for lid in sig['lights']:
                    lx, ly, lz = lights[lid]['position_enu_m']
                    dx, dy = lx - x, ly - y
                    p_sensor = np.array([c * dx + sn * dy, -sn * dx + c * dy, lz - z])
                    ok, _, rng = project(cam, p_sensor)
                    if ok:
                        h_px = k_m * cam['k'][1, 1] * cam['scale'] / rng
                        miss *= 1.0 - rate_at(h_px, curve)
                per[cam_name].append(1.0 - miss)
        per = {k: np.array(v) for k, v in per.items()}
        per['both'] = 1.0 - (1.0 - per['wide']) * (1.0 - per['narrow'])
        traces[sig['id']] = per
        entry = {'lights': sig['lights'], 'line_route_s': round(sig['line_s'], 1)}
        for cam_name, p in per.items():
            ok = p >= 0.5
            reach = 0.0
            for i, dl in enumerate(d_line):  # farthest distance with >= 50 % all the way to the line
                if not ok[: i + 1].all():
                    break
                reach = dl
            entry[cam_name] = {'reliable_from_m': round(float(reach), 0),
                               'rate_at_m': {f'{int(dl)}': round(float(p[int(dl)]), 2) for dl in (0, 5, 10, 20, 30, 40, 50, 60, 80, 100)},
                               'enough_for': [k for k, v in need.items() if reach >= v]}
        report['signals'][sig['id']] = entry
    if args.narrow_eval:
        report['measured_pair'] = measured_pair(eval_dir, args.narrow_eval)
        print('measured on the drive (mapped light in view, detected):')
        for cam_name, rows in report['measured_pair'].items():
            print(f'   {cam_name:6s}', {k: (v['rate'], v['frames']) for k, v in rows.items()})
    out = ROOT / 'evaluation/light_camera_choice'
    out.with_suffix('.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: report[k] for k in ('signal_head_height_m', 'matched_boxes', 'cameras', 'needed_decision_distance_m')}, indent=1))
    print('measured wide:', report['measured_wide'])
    for sid, e in report['signals'].items():
        print(f"signal {sid} (lights {e['lights']}):", {c: (e[c]['reliable_from_m'], e[c]['enough_for']) for c in ('wide', 'narrow', 'both')})
        for c in ('wide', 'narrow', 'both'):
            print(f'   {c:6s}', e[c]['rate_at_m'])
    plot(traces, d_line, need, report, out.with_suffix('.png'))
    print(f'wrote {out}.json and .png')


def plot(traces, d_line, need, report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(traces), figsize=(5.2 * len(traces), 4.2), sharey=True)
    colours = {'wide': '#d9822b', 'narrow': '#2b7bd9', 'both': '#444444'}
    for ax, (sid, per) in zip(np.atleast_1d(axes), traces.items()):
        for cam_name, p in per.items():
            ax.plot(d_line, p, color=colours[cam_name], lw=2.2 if cam_name != 'both' else 1.2,
                    ls='-' if cam_name != 'both' else '--', label=cam_name)
        for (label, dist), ls in zip(need.items(), (':', '-.', '--')):
            ax.axvline(dist, color='0.6', ls=ls, lw=1)
            ax.text(dist + 0.8, 0.03, f'needed at {label}', rotation=90, fontsize=7, color='0.4')
        ax.axhline(0.5, color='0.8', lw=0.8)
        ax.set_title(f"signal {sid} (lights {', '.join(map(str, report['signals'][sid]['lights']))})", fontsize=10)
        ax.set_xlabel('front bumper to stop line (m)')
        ax.set_xlim(0, 120)
        ax.set_ylim(0, 1.02)
    np.atleast_1d(axes)[0].set_ylabel('predicted detection per frame')
    np.atleast_1d(axes)[0].legend(loc='upper right', fontsize=8)
    fig.suptitle('Traffic-light detection along each approach: wide vs narrow camera (geometry + measured wide-camera size curve)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == '__main__':
    main()
