#!/usr/bin/env python3
"""Live camera vs LiDAR comparison of lane lines, road boundaries and lane centre (no map, no ground truth).

Subscribes to /camera_lidar_bev/lanes/<method> for the given methods (tools/lane_lidar.py,
tools/lane_camera.py), and every few seconds prints per method:
  * rate (Hz), latency (sensor stamp -> published, median / p95 ms) and compute time;
  * availability: share of results with an ego-lane left / right line, a left / right road boundary,
    a lane centre, and a two-sided (not one_sided) centre;
  * usable range: median farthest look-ahead of the ego left / right line;
and for each pair (first method = reference) results within --pair-s of each other in time:
  * agreement: median and p90 |difference| of the ego-lane left / right line, road boundaries and lane centre
    at 5 / 10 / 20 / 30 m ahead, and how often both have that feature.
Agreement is not accuracy: when the two disagree this does not say which one is right. Writes
data/lane_compare_<UTC>/results.jsonl (every result) and summary.json on exit.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/compare_lanes.py --methods lidar camera_front_narrow [camera_front_wide]
"""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from std_msgs.msg import String

import road_features as rf

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ('ego_left', 'ego_right', 'road_left', 'road_right', 'lane_centre')


def stats(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    a = np.abs(np.array(v))
    return {'n': len(v), 'median': round(float(np.median(a)), 3), 'p90': round(float(np.percentile(a, 90)), 3)}


def jitter(rs, x=10.0):
    """Median frame-to-frame change (m) of the ego left / right line at x ahead; lower is steadier."""
    steps = []
    for side in ('ego_left', 'ego_right'):
        prev = None
        for r in rs:
            y = rf.lateral_at(r[side], x) if r[side] else None
            if y is not None and prev is not None:
                steps.append(abs(y - prev))
            prev = y
    return round(float(np.median(steps)), 3) if steps else None


def summarize(results, pairs, ref, methods):
    out = {'methods': {}, 'agreement': {}}
    for m in methods:
        rs = list(results[m])
        if len(rs) < 2:
            out['methods'][m] = None
            continue
        t = np.array([r['_recv'] for r in rs])
        lat = np.array([r['latency_ms'] for r in rs])
        reach = defaultdict(list)
        for r in rs:
            for side in ('ego_left', 'ego_right'):
                if r[side]:
                    reach[side].append(r[side]['points'][-1][0])
        out['methods'][m] = {
            'hz': round((len(rs) - 1) / (t[-1] - t[0]), 1) if t[-1] > t[0] else None,
            'latency_ms': {'median': round(float(np.median(lat)), 1), 'p95': round(float(np.percentile(lat, 95)), 1)},
            'compute_ms': round(float(np.median([r.get('compute_ms', 0) for r in rs])), 1),
            'available': {f: round(float(np.mean([bool(r[f]) for r in rs])), 2) for f in FEATURES},
            'two_sided_centre': round(float(np.mean([bool(r['lane_centre']) and not r['one_sided'] for r in rs])), 2),
            'reach_m': {s: round(float(np.median(v)), 1) for s, v in reach.items()},
            'lane_width_m': round(float(np.median([r['lane_width_m'] for r in rs if r['lane_width_m']])), 2)
            if any(r['lane_width_m'] for r in rs) else None,
            'road_z': round(float(np.median([r['road_z'] for r in rs if r.get('road_z') is not None])), 2)
            if any(r.get('road_z') is not None for r in rs) else None,
            'jitter_10m': jitter(rs)}
    for m in methods:
        if m == ref:
            continue
        diffs = defaultdict(list)
        for d in pairs[m]:
            for f in FEATURES:
                for x, v in d[f].items():
                    diffs[(f, x)].append(v)
        out['agreement'][f'{m} vs {ref}'] = {
            f: {f'{x:g} m': {**(stats(diffs[(f, x)]) or {'n': 0}),
                             'both_have': round(len([v for v in diffs[(f, x)] if v is not None]) / max(1, len(diffs[(f, x)])), 2)}
                for x in rf.LOOK_AHEAD_M} for f in FEATURES}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--methods', nargs='+', default=['lidar', 'camera_front_narrow'])
    parser.add_argument('--pair-s', type=float, default=0.06, help='max time between paired results')
    parser.add_argument('--window', type=int, default=600, help='results per method kept for the live table')
    parser.add_argument('--every-s', type=float, default=5.0)
    args = parser.parse_args()
    out_dir = ROOT / 'data' / f"lane_compare_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True)
    log = (out_dir / 'results.jsonl').open('w')
    ref = args.methods[0]
    results = {m: deque(maxlen=args.window) for m in args.methods}
    everything = {m: [] for m in args.methods}  # for the final summary
    pairs = {m: deque(maxlen=args.window) for m in args.methods}
    all_pairs = {m: [] for m in args.methods}

    rclpy.init()
    node = rclpy.create_node('compare_lanes')

    def on_result(method, text):
        r = rf.from_message(text)
        r['_recv'] = time.monotonic()
        results[method].append(r)
        everything[method].append({k: r[k] for k in ('stamp', 'latency_ms', 'one_sided', 'lane_width_m', '_recv')} |
                                  {'compute_ms': r.get('compute_ms', 0), 'road_z': r.get('road_z'), **{f: r[f] for f in FEATURES}})
        log.write(text + '\n')
        if method == ref:
            return
        best = min(results[ref], key=lambda a: abs(a['stamp'] - r['stamp']), default=None)
        if best is not None and abs(best['stamp'] - r['stamp']) <= args.pair_s:
            d = rf.compare(best, r)
            pairs[method].append(d)
            all_pairs[method].append(d)

    for m in args.methods:
        node.create_subscription(String, f'/camera_lidar_bev/lanes/{m}', lambda msg, m=m: on_result(m, msg.data), 20)
    print(f'Comparing {args.methods} (reference {ref}); writing {out_dir}', flush=True)
    last = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - last < args.every_s:
                continue
            last = time.monotonic()
            s = summarize(results, pairs, ref, args.methods)
            print(f"\n--- last {args.window} results per method ---")
            for m, v in s['methods'].items():
                if v is None:
                    print(f'{m:22s} waiting for /camera_lidar_bev/lanes/{m}')
                    continue
                a = v['available']
                print(f"{m:22s} {v['hz']} Hz, latency {v['latency_ms']['median']} / {v['latency_ms']['p95']} ms, compute {v['compute_ms']} ms | "
                      f"ego L {a['ego_left']:.0%} R {a['ego_right']:.0%}, road L {a['road_left']:.0%} R {a['road_right']:.0%}, "
                      f"centre {a['lane_centre']:.0%} (two-sided {v['two_sided_centre']:.0%}), reach {v['reach_m']}, jitter {v['jitter_10m']} m, width {v['lane_width_m']}, road z {v['road_z']}")
            for name, feats in s['agreement'].items():
                cells = []
                for f in FEATURES:
                    c = feats[f]['10 m']
                    cells.append(f"{f} {c['median']:.2f}/{c['p90']:.2f} m ({c['both_have']:.0%})" if c['n'] else f'{f} -')
                print(f'{name} at 10 m (median/p90, both have): ' + ', '.join(cells))
    except KeyboardInterrupt:
        pass
    finally:
        log.close()
        full = {m: deque(v) for m, v in everything.items()}
        summary = summarize(full, all_pairs, ref, args.methods)
        summary.update(created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'), reference=ref,
                       note='agreement between methods, not accuracy against ground truth')
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
        print(f'wrote {out_dir}/summary.json', flush=True)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
