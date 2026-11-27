#!/usr/bin/env python3
"""Compare centerpoint_sweep_eval outputs (e.g. 1 vs 3 vs 5 sweeps) without ground truth.

Boxes are moved into the map frame with each scan's saved-trajectory pose, then judged by
temporal consistency: a real object reappears at about the same map position in the next
scan, a false alarm tends to flicker. Reports confidence, persistence, isolated boxes,
scan-to-scan jitter of matched boxes, and timing.

  python3 tools/analyze_sweep_eval.py SCANS_DIR eval_s1.jsonl eval_s3.jsonl eval_s5.jsonl
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

# Current per-class thresholds (config/perception/lidar_centerpoint_detector.yaml).
THRESHOLDS = dict(car=.30, truck=.40, bus=.45, trailer=.50, construction_vehicle=.50, person=.30,
                  motorcycle=.35, bicycle=.35, barrier=.50, traffic_cone=.40)
GROUP = dict(car='vehicle', truck='vehicle', bus='vehicle', trailer='vehicle', construction_vehicle='vehicle',
             motorcycle='two-wheeler', bicycle='two-wheeler', person='person', barrier='static', traffic_cone='static')
GATE_M = dict(vehicle=2.0, **{'two-wheeler': 1.5}, person=1.0, static=1.0)  # match radius between scans (10 Hz)


def rot(q):
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def load(path, poses, threshold=None):
    frames = []
    for line in Path(path).read_text().splitlines():
        f = json.loads(line)
        pose = poses.get(f['i'])
        boxes = []
        for cls, score, x, y, z, l, w, h, yaw in f['boxes']:
            if score < (threshold if threshold is not None else THRESHOLDS[cls]):
                continue
            item = dict(cls=cls, score=score, dist=math.hypot(x, y))
            if pose:
                (t, r) = pose
                item['wx'] = r[0][0] * x + r[0][1] * y + r[0][2] * z + t[0]
                item['wy'] = r[1][0] * x + r[1][1] * y + r[1][2] * z + t[1]
                item['wyaw'] = yaw + math.atan2(r[1][0], r[0][0])
            boxes.append(item)
        frames.append(dict(i=f['i'], posed=pose is not None, boxes=boxes, ms=f['detector_ms'],
                           pre=f['pre_ms'], net=f['net_ms'], points=f['points'], pillars=f['pillars']))
    return frames


def nearest(box, others):
    best = None
    for o in others:
        if GROUP[o['cls']] != GROUP[box['cls']] or 'wx' not in o:
            continue
        d = math.hypot(o['wx'] - box['wx'], o['wy'] - box['wy'])
        if d <= GATE_M[GROUP[box['cls']]] and (best is None or d < best[0]):
            best = (d, o)
    return best


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float('nan')


def analyse(frames):
    out = {}
    posed = [k for k in range(1, len(frames) - 1) if frames[k]['posed'] and frames[k - 1]['posed'] and frames[k + 1]['posed']]
    boxes = [b for k in posed for b in frames[k]['boxes']]
    out['boxes_per_scan'] = len(boxes) / max(1, len(posed))
    persistent = isolated = 0
    jitter, yaw_jit = [], []
    for k in posed:
        for b in frames[k]['boxes']:
            prev, nxt = nearest(b, frames[k - 1]['boxes']), nearest(b, frames[k + 1]['boxes'])
            persistent += prev is not None and nxt is not None
            isolated += prev is None and nxt is None
            if prev and GROUP[b['cls']] == 'vehicle':
                jitter.append(prev[0])
                dy = abs((b['wyaw'] - prev[1]['wyaw'] + math.pi / 2) % math.pi - math.pi / 2)  # heading mod 180
                yaw_jit.append(math.degrees(dy))
    out['persistent_frac'] = persistent / max(1, len(boxes))
    out['isolated_frac'] = isolated / max(1, len(boxes))
    out['vehicle_jitter_m_median'] = statistics.median(jitter) if jitter else float('nan')
    out['vehicle_yaw_jitter_deg_median'] = statistics.median(yaw_jit) if yaw_jit else float('nan')
    by_cls = {}
    for b in boxes:
        by_cls.setdefault(b['cls'], []).append(b['score'])
    out['per_class'] = {c: (len(v) / max(1, len(posed)), statistics.median(v)) for c, v in by_cls.items()}
    vehicles = [b for b in boxes if GROUP[b['cls']] == 'vehicle']
    out['vehicle_score_median'] = statistics.median(b['score'] for b in vehicles) if vehicles else float('nan')
    out['far_vehicles_per_scan'] = sum(1 for b in vehicles if b['dist'] > 30) / max(1, len(posed))
    out['ms_median'] = statistics.median(f['ms'] for f in frames)
    out['ms_p95'] = pct([f['ms'] for f in frames], .95)
    out['pre_ms'] = statistics.median(f['pre'] for f in frames)
    out['net_ms'] = statistics.median(f['net'] for f in frames)
    out['points'] = statistics.median(f['points'] for f in frames)
    out['pillars'] = statistics.median(f['pillars'] for f in frames)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('scans_dir', type=Path)
    p.add_argument('evals', nargs='+', type=Path)
    a = p.parse_args()
    poses = {}
    with open(a.scans_dir / 'scans.csv') as f:
        for r in csv.DictReader(f):
            if r['pose_ok'] == '1':
                poses[int(r['index'])] = ([float(r['x']), float(r['y']), float(r['z'])],
                                          rot([float(r[k]) for k in ('qx', 'qy', 'qz', 'qw')]))
    for label, threshold in (('per-class thresholds (current config)', None), ('all classes at 0.30', 0.3),
                             ('all classes at 0.50', 0.5)):
        results = {e.stem: analyse(load(e, poses, threshold)) for e in a.evals}
        names = list(results)
        print(f'\n=== {label} ===')
        rows = [('boxes per scan', 'boxes_per_scan', '{:.2f}'), ('persistent (in prev AND next scan)', 'persistent_frac', '{:.1%}'),
                ('isolated (in neither)', 'isolated_frac', '{:.1%}'), ('vehicle score median', 'vehicle_score_median', '{:.3f}'),
                ('vehicles beyond 30 m per scan', 'far_vehicles_per_scan', '{:.2f}'),
                ('vehicle position jitter, median (m)', 'vehicle_jitter_m_median', '{:.3f}'),
                ('vehicle heading jitter, median (deg)', 'vehicle_yaw_jitter_deg_median', '{:.2f}')]
        if threshold is None:
            rows += [('detector ms median', 'ms_median', '{:.1f}'), ('detector ms p95', 'ms_p95', '{:.1f}'),
                     ('  pillar step ms', 'pre_ms', '{:.1f}'), ('  GPU ms', 'net_ms', '{:.1f}'),
                     ('points per input', 'points', '{:,.0f}'), ('pillars', 'pillars', '{:,.0f}')]
        print(f'{"":40s}' + ''.join(f'{n:>14s}' for n in names))
        for title, key, fmt in rows:
            print(f'{title:40s}' + ''.join(f'{fmt.format(results[n][key]):>14s}' for n in names))
        classes = sorted({c for r in results.values() for c in r['per_class']})
        print('per class: boxes/scan (median score)')
        for c in classes:
            cells = [results[n]['per_class'].get(c) for n in names]
            print(f'  {c:38s}' + ''.join(f'{f"{v[0]:.2f} ({v[1]:.2f})" if v else "-":>14s}' for v in cells))


if __name__ == '__main__':
    main()
