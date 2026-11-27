#!/usr/bin/env python3
"""Run the first and the current route planner on a recorded drive and compare their decisions (ROS-free).

Open loop: the car is where it really drove (dataset ego poses), the planners see the recorded CenterPoint
detections, and each plans every frame from the driven speed, as in shadow mode on the car. For each version:
which mode the plan is in (cruise / curve / stop_line / follow / yield), the front object it follows, crossing or
cutting-in objects it yields to, and the planned speed 1 s ahead against the speed actually driven. Writes
evaluation/planner_replay_<drive>.json and .png (speed timeline with lead gap and yields).

  python3 tools/compare_planner_replay.py [--dataset datasets/prediction/drive_20261126T013444Z.npz] [--from-min 13 --to-min 20]
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from planner_core import PlannerConfig, RoutePlanner, load_stops
from prediction_dataset import load_dataset
from route_geometry import Route

ROOT = Path(__file__).resolve().parents[1]


def driven_speed(t, xy, window_s=0.5):
    """Speed from positions over +-window_s/2 (the recorded LiDAR poses)."""
    v = np.zeros(len(t))
    j0 = np.searchsorted(t, t - window_s / 2)
    j1 = np.clip(np.searchsorted(t, t + window_s / 2), 0, len(t) - 1)
    ok = t[j1] - t[j0] > 0.1
    v[ok] = np.linalg.norm(xy[j1[ok]] - xy[j0[ok]], axis=1) / (t[j1[ok]] - t[j0[ok]])
    return v


def run(frames, ego, route, stops, cfg, v_drive):
    pl = RoutePlanner(route, stops, cfg)
    rows, s_prev = [], None
    for k, ((t, dets), (x, y, _, yaw)) in enumerate(zip(frames, ego)):
        pr = route.project(x, y, yaw, s_prev) or route.project(x, y, yaw)
        if pr is None:
            s_prev = None
            rows.append({'t': t, 'mode': 'off_route'})
            continue
        s_prev, lateral = pr[0], pr[1]
        v = float(v_drive[k])
        pl.update_stops(s_prev, v, t)
        objs = [(p[0], p[1], p[2], max(sz[:2]), min(sz[:2]), cls, 0.9, yw) for _, cls, p, sz, yw in dets]
        lead = pl.lead_vehicle(t, objs, s_prev, now=t, v_ego=v)
        p = pl.plan(s_prev, lateral, v, lead)
        mode = p['reason']
        near = pl.nearest_stop(s_prev)
        if near is not None and near['state'] == 'stopping':
            mode = 'stopped_at_stop_line'
        conflict = getattr(pl, 'conflict', None)
        if conflict:
            mode = 'yield'
        rows.append({'t': t, 'mode': mode, 'v_plan_1s': float(p['v'][10]), 'a_plan': float(p['ax'][0]),
                     'lead_gap': None if lead is None else float(lead[0]), 'lead_v': None if lead is None else float(lead[1]),
                     'yield': None if not conflict else conflict['class']})
    return rows


def summary(rows, v_drive, t):
    on = [r for r in rows if r['mode'] != 'off_route']
    modes = Counter(r['mode'] for r in rows)
    # planned speed 1 s ahead vs the speed driven 1 s later
    err = []
    for k, r in enumerate(rows):
        if 'v_plan_1s' in r:
            j = int(np.searchsorted(t, r['t'] + 1.0))
            if j < len(t):
                err.append(r['v_plan_1s'] - v_drive[j])
    err = np.array(err)
    yields, prev = Counter(), None
    for r in rows:  # yield events (a new yield after none)
        if r.get('yield') and not prev:
            yields[r['yield']] += 1
        prev = r.get('yield')
    return {'frames': len(rows), 'on_route': len(on), 'modes': dict(modes.most_common()),
            'frames_with_lead': sum(r.get('lead_gap') is not None for r in rows),
            'hard_brake_frames_planned_lt_-3': int(sum(r.get('a_plan', 0) < -3.0 for r in rows)),
            'emergency_frames_planned_lt_-4': int(sum(r.get('a_plan', 0) < -4.05 for r in rows)),
            'yield_events': dict(yields), 'yield_frames': sum(bool(r.get('yield')) for r in rows),
            'plan_minus_driven_speed_1s_mps': {'median': round(float(np.median(err)), 2),
                                               'p10': round(float(np.percentile(err, 10)), 2),
                                               'p90': round(float(np.percentile(err, 90)), 2)} if len(err) else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/prediction/drive_20261126T013444Z.npz')
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--from-min', type=float, default=13.0, help='chart window start (minutes into the drive)')
    parser.add_argument('--to-min', type=float, default=20.0)
    parser.add_argument('--max-speed-mps', type=float, default=PlannerConfig.max_speed_mps)
    args = parser.parse_args()
    frames, _, _, ego = load_dataset(args.dataset, 0.35)
    ego = np.asarray(ego)
    t = np.array([f[0] for f in frames])
    v_drive = driven_speed(t, ego[:, :2])
    route = Route(args.map / 'gps_alignment/trajectory_enu.csv')
    out = {'dataset': str(args.dataset), 'max_speed_mps': args.max_speed_mps, 'versions': {}}
    rows = {}
    for name, track in (('first', False), ('current', True)):
        cfg = PlannerConfig(track_objects=track, max_speed_mps=args.max_speed_mps)
        rows[name] = run(frames, ego, route, load_stops(route, args.map, cfg), cfg, v_drive)
        out['versions'][name] = summary(rows[name], v_drive, t)
        print(name, json.dumps(out['versions'][name]))
    stem = ROOT / 'evaluation' / f'planner_replay_{args.dataset.stem}'
    stem.with_suffix('.json').write_text(json.dumps(out, indent=1) + '\n')
    plot(rows, t, v_drive, args, stem.with_suffix('.png'))
    print(f'wrote {stem}.json and .png')


def plot(rows, t, v_drive, args, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m = (t - t[0]) / 60.0
    keep = (m >= args.from_min) & (m <= args.to_min)
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1]})
    ax = axes[0]
    ax.plot(m[keep], v_drive[keep], color='0.35', lw=1.6, label='driven')
    colours = {'first': '#d9822b', 'current': '#2b7bd9'}
    for name, rs in rows.items():
        v1 = np.array([r.get('v_plan_1s', np.nan) for r in rs])
        ax.plot(m[keep], v1[keep], color=colours[name], lw=1.1, label=f'{name} planner: planned speed in 1 s')
    ax.axhline(args.max_speed_mps, color='0.6', ls=':', lw=1, label='speed cap')
    ax.set_ylabel('speed (m/s)')
    ax.legend(loc='upper right', fontsize=8, ncol=4)
    ax.set_title(f'{args.dataset.stem}: planners replayed on the recorded drive (open loop, the car where it drove)')
    ax = axes[1]
    for name, rs in rows.items():
        g = np.array([np.nan if r.get('lead_gap') is None else r['lead_gap'] for r in rs])
        ax.plot(m[keep], g[keep], '.', ms=2, color=colours[name], label=f'{name}: gap to object it follows / yields to')
    ax.set_ylabel('gap (m)')
    ax.set_ylim(0, 60)
    ax.legend(loc='upper right', fontsize=8)
    ax = axes[2]
    names = ['cruise', 'curve', 'stop_line', 'stopped_at_stop_line', 'follow', 'yield', 'route_end', 'off_route']
    mode_c = {'cruise': '#7fbf7f', 'curve': '#b0b0b0', 'stop_line': '#e05050', 'stopped_at_stop_line': '#a01010',
              'follow': '#9467bd', 'yield': '#ff9f1c', 'route_end': '#444444', 'off_route': '#dddddd'}
    for y, (name, rs) in enumerate(rows.items()):
        md = np.array([r['mode'] for r in rs])
        for mode in names:
            sel = keep & (md == mode)
            ax.scatter(m[sel], np.full(sel.sum(), y), s=6, marker='s', color=mode_c[mode], label=mode if y == 1 else None)
    ax.set_yticks([0, 1], list(rows))
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel('minutes into the drive')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.45), ncol=8, fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


if __name__ == '__main__':
    main()
