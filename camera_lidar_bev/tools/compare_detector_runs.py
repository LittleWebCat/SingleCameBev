#!/usr/bin/env python3
"""Detailed comparison of LiDAR detector runs on the same recorded scans (no ground truth).

Extends tools/analyze_sweep_eval.py. Each run is a centerpoint_sweep_eval-format .jsonl (boxes in os_sensor)
for the scans in SCANS_DIR (tools/extract_lidar_sweeps.py; poses from scans.csv). With the live per-class
thresholds it reports, per run and object group:
  * boxes per scan by distance (0-20, 20-35, 35-55 m);
  * persistence (box also in the previous and next scan, at about the same map position) and flicker;
  * cross-run support: share of a run's boxes with a box of the other run within the group's gate in the
    same scan (any class / same group);
  * motion on the map: for boxes chained through >= 10 consecutive scans, how far they move (static
    objects such as posts stay put; people and vehicles usually move);
  * score quartiles and timing.
Writes JSON (--out) for reports; prints a short summary.

  /usr/bin/python3 tools/compare_detector_runs.py data/sweeps_20261010 cp=data/transfusion_eval/centerpoint_s3.jsonl \\
      tf=data/transfusion_eval/transfusion_openpcdet_s3_v400k.jsonl --out evaluation/detector_comparison_001.json
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

THRESHOLDS = dict(car=.30, truck=.40, bus=.45, trailer=.50, construction_vehicle=.50, person=.30,
                  motorcycle=.35, bicycle=.35, barrier=.50, traffic_cone=.40)
GROUP = dict(car='vehicle', truck='vehicle', bus='vehicle', trailer='vehicle', construction_vehicle='vehicle',
             motorcycle='two-wheeler', bicycle='two-wheeler', person='person', barrier='static', traffic_cone='static')
GATE = {'vehicle': 2.0, 'two-wheeler': 1.5, 'person': 1.0, 'static': 1.0}
RANGES = ((0, 20), (20, 35), (35, 55))


def quat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load(path, poses):
    frames = {}
    for line in open(path):
        f = json.loads(line)
        if f['i'] not in poses:
            continue
        r, t = poses[f['i']]
        boxes = []
        for cls, score, x, y, z, l, w, h, yaw in f['boxes']:
            if score < THRESHOLDS[cls]:
                continue
            m = r @ np.array([x, y, z]) + t
            boxes.append(dict(cls=cls, group=GROUP[cls], score=score, rng=math.hypot(x, y), map=m[:2]))
        frames[f['i']] = dict(boxes=boxes, ms=f['detector_ms'], pre=f['pre_ms'], net=f['net_ms'])
    return frames


def near(b, boxes, same_group=True):
    g = GATE[b['group']]
    return any((not same_group or o['group'] == b['group']) and np.linalg.norm(o['map'] - b['map']) < g for o in boxes)


def chains(frames, group):
    """Greedy scan-to-scan chains of same-group boxes; returns lists of map positions."""
    out, active = [], []
    for i in sorted(frames):
        boxes = [b for b in frames[i]['boxes'] if b['group'] == group]
        used, nxt = set(), []
        for ch in active:
            best, bd = None, GATE[group]
            for k, b in enumerate(boxes):
                d = np.linalg.norm(b['map'] - ch[-1])
                if k not in used and d < bd:
                    best, bd = k, d
            if best is None:
                out.append(ch)
            else:
                used.add(best)
                ch.append(boxes[best]['map'])
                nxt.append(ch)
        nxt += [[b['map']] for k, b in enumerate(boxes) if k not in used]
        active = nxt
    return out + active


def analyse(name, frames, other):
    idx = sorted(frames)
    res = {'scans': len(idx), 'groups': {}}
    ms = [frames[i]['ms'] for i in idx]
    res['timing_ms'] = {'median': float(np.median(ms)), 'p95': float(np.percentile(ms, 95)),
                        'pre_median': float(np.median([frames[i]['pre'] for i in idx])),
                        'net_median': float(np.median([frames[i]['net'] for i in idx]))}
    for group in GATE:
        boxes = [(k, b) for k, i in enumerate(idx) for b in frames[i]['boxes'] if b['group'] == group]
        if not boxes:
            res['groups'][group] = None
            continue
        g = {'per_scan': len(boxes) / len(idx)}
        g['per_scan_by_range'] = {f'{lo}-{hi} m': sum(lo <= b['rng'] < hi for _, b in boxes) / len(idx) for lo, hi in RANGES}
        pers = iso = inner = 0
        for k, b in boxes:
            if 0 < k < len(idx) - 1:
                inner += 1
                p, n = near(b, frames[idx[k - 1]]['boxes']), near(b, frames[idx[k + 1]]['boxes'])
                pers += p and n
                iso += not p and not n
        g['persistent'] = pers / inner if inner else None
        g['isolated'] = iso / inner if inner else None
        oth = [(k, b) for k, b in boxes if idx[k] in other]
        g['supported_by_other_same_group'] = sum(near(b, other[idx[k]]['boxes']) for k, b in oth) / len(oth) if oth else None
        g['supported_by_other_any_class'] = sum(near(b, other[idx[k]]['boxes'], False) for k, b in oth) / len(oth) if oth else None
        sc = [b['score'] for _, b in boxes]
        g['score_quartiles'] = [float(v) for v in np.percentile(sc, [25, 50, 75])]
        long = [c for c in chains(frames, group) if len(c) >= 10]
        travel = [float(np.linalg.norm(c[-1] - c[0])) for c in long]
        g['chains_10plus'] = len(long)
        g['chain_travel_m_median'] = float(np.median(travel)) if travel else None
        g['chains_static_share'] = sum(t < 0.5 for t in travel) / len(travel) if travel else None
        res['groups'][group] = g
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('scans_dir', type=Path)
    p.add_argument('runs', nargs='+', help='name=path.jsonl; the first run is the reference for cross-support')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    poses = {}
    with open(a.scans_dir / 'scans.csv') as f:
        for r in csv.DictReader(f):
            if r['pose_ok'] == '1':
                poses[int(r['index'])] = (quat([float(r[k]) for k in ('qx', 'qy', 'qz', 'qw')]),
                                          np.array([float(r['x']), float(r['y']), float(r['z'])]))
    runs = {n: load(path, poses) for n, path in (s.split('=', 1) for s in a.runs)}
    names = list(runs)
    report = {}
    for n in names:
        other = runs[names[1]] if n == names[0] else runs[names[0]]
        report[n] = analyse(n, runs[n], other)
    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + '\n')
    for n, r in report.items():
        print(f"== {n}: {r['timing_ms']['median']:.0f} ms median")
        for group, g in r['groups'].items():
            if g:
                print(f"  {group:12s} {g['per_scan']:5.2f}/scan  persistent {g['persistent']:.0%}  isolated {g['isolated']:.0%}  "
                      f"confirmed by other {g['supported_by_other_same_group']:.0%}  static long chains {g['chains_static_share'] or 0:.0%} "
                      f"(n={g['chains_10plus']})  by range {', '.join(f'{k} {v:.2f}' for k, v in g['per_scan_by_range'].items())}")


if __name__ == '__main__':
    main()
