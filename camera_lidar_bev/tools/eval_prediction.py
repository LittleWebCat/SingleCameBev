#!/usr/bin/env python3
"""Benchmark object trajectory prediction on a recorded drive. Runs on any PC with numpy + scipy (no ROS).

Replays a drive's CenterPoint detections (already in the map frame) from a dataset written by
tools/export_prediction_dataset.py (default: the 24-minute drive in datasets/prediction/), runs each
predictor in tools/prediction_models.py, and scores its predictions against what the objects really did:
ground truth is a reference constant-velocity tracker's raw detection positions over the following
seconds (the same reference for every predictor; a prediction is linked to the reference track whose
detection in that frame is within 1 m). A prediction is made for each confirmed track every --every-s.

Reported per object group, on the first half of the drive (dev, used for tuning) and the second half
(test, held out):
  * moving objects (moved >= 2 m in 2 s, pedestrians >= 1 m): median / mean final displacement error at
    1, 3 and 5 s (pedestrians 4 s), mean error over the horizon (ADE), split into straight and turning
    (heading changed > 30 degrees within 3 s), and how often motion was missed (predicted to stay);
  * parked / still objects (moved < 0.7 m in 3 s): how often they were predicted to move.
Errors include tracking, detection and localization noise; ground truth is only as good as the detections.

Your own predictor: write a file that calls prediction_models.register_model('mine', factory) and run
with --plugin that_file.py --models imm mine (see docs/prediction_development.md).

  pip install numpy scipy        # any PC; on the car /usr/bin/python3 already has them
  python3 tools/eval_prediction.py [--dataset datasets/prediction/<drive>.npz] [--models cv imm imm+map imm+hd]
  python3 tools/eval_prediction.py --plugin my_model.py --models imm mine --quick   # first 4 minutes only
  # straight from a bag on the car (needs ROS): --bag data/drive_<UTC>/bag
"""
import argparse
from collections import defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from prediction_dataset import GROUP_OF, load_dataset
from prediction_models import MODELS, RouteLanes, Tracker
from route_geometry import Route

ROOT = Path(__file__).resolve().parents[1]
HORIZON = {'vehicle': 5.0, 'cyclist': 5.0, 'pedestrian': 4.0}
MOVED = {'vehicle': 2.0, 'cyclist': 2.0, 'pedestrian': 1.0}  # m in 2 s


def frames_from_bag(bag, min_score):
    """Read a recorded drive directly (needs ROS 2 sourced); same frames as load_dataset."""
    from export_prediction_dataset import read_bag
    frame_t, _, dets = read_bag(bag, min_score)
    per = defaultdict(list)
    for f, cls, xyz, size, yaw, _ in dets:
        per[f].append((GROUP_OF[cls], cls, xyz, size, yaw))
    return [(t, per[i]) for i, t in enumerate(frame_t)]


def ground_truth(frames):
    """Reference tracks: {id: (times, xy)} of raw matched detections, and per-frame [(id, xy)]."""
    tracker, tracks, per_frame = Tracker('cv'), defaultdict(list), []
    for t, dets in frames:
        matched = tracker.step(t, dets)
        here = []
        for tr in matched:
            xy = np.asarray(tr.last_det[2][:2])
            tracks[tr.id].append((t, xy[0], xy[1]))
            here.append((tr.id, xy, tr.group))
        per_frame.append(here)
    return {k: (np.array([v[0] for v in vs]), np.array([v[1:] for v in vs])) for k, vs in tracks.items()}, per_frame


def gt_at(track, t, max_gap=0.25):
    ts, xy = track
    i = np.searchsorted(ts, t)
    if i == 0 or i >= len(ts) or ts[i] - ts[i - 1] > max_gap:
        return None
    a = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return xy[i - 1] * (1 - a) + xy[i] * a


def run(model, frames, gt, per_frame, route, every_s, hdmap=None):
    lanes = RouteLanes(route) if model.endswith('+map') else hdmap if model.endswith('+hd') else None
    tracker = Tracker(model.split('+')[0], lanes=lanes)
    last_pred, samples = {}, []
    for (t, dets), here in zip(frames, per_frame):
        for tr in tracker.step(t, dets):
            if tr.hits < tracker.confirm_hits or t - last_pred.get(tr.id, -1e9) < every_s:
                continue
            xy = np.asarray(tr.last_det[2][:2])
            ref = min(((np.linalg.norm(gxy - xy), gid) for gid, gxy, g in here if g == tr.group), default=(9.9, None))
            if ref[0] > 1.0:
                continue
            last_pred[tr.id] = t
            moving, path = tracker.predict_track(tr, HORIZON[tr.group], 0.5)
            samples.append((t, tr.group, ref[1], moving, np.array([(p[0], p[1], p[2]) for p in path])))
    return samples


def score(samples, gt, t_split):
    out = {}
    for half, keep in (('dev', lambda t: t < t_split), ('test', lambda t: t >= t_split)):
        res = defaultdict(lambda: defaultdict(list))
        for t, group, gid, moving, path in samples:
            if not keep(t):
                continue
            track = gt[gid]
            p0, p2, p3 = gt_at(track, t), gt_at(track, t + 2.0), gt_at(track, t + 3.0)
            if p0 is None:
                continue
            if p3 is not None and np.linalg.norm(p3 - p0) < 0.7:
                res[group]['still_predicted_moving'].append(float(moving))
                continue
            if p2 is None or np.linalg.norm(p2 - p0) < MOVED[group]:
                continue
            p1, q1 = gt_at(track, t + 1.0), gt_at(track, t + 0.5)
            turning = False
            if p3 is not None and p1 is not None and q1 is not None:
                h0 = math.atan2(*(q1 - p0)[::-1])
                h1 = math.atan2(*(p3 - p2)[::-1])
                turning = abs((h1 - h0 + math.pi) % (2 * math.pi) - math.pi) > math.radians(30)
            kind = 'turning' if turning else 'straight'
            res[group]['missed_motion'].append(float(not moving))
            errors = []
            for tau, px, py in path[1:]:
                g = gt_at(track, t + tau)
                if g is None:
                    break
                e = float(np.hypot(px - g[0], py - g[1]))
                errors.append(e)
                for k in (kind, 'all'):
                    if abs(tau - round(tau)) < 1e-6 and round(tau) in (1, 3, 4, 5):
                        res[group][f'{k}_fde_{round(tau)}s'].append(e)
            if errors:
                for k in (kind, 'all'):
                    res[group][f'{k}_ade'].append(float(np.mean(errors)))
        out[half] = {g: {k: {'n': len(v), 'median': round(float(np.median(v)), 2), 'mean': round(float(np.mean(v)), 2)}
                         for k, v in sorted(d.items()) if v} for g, d in res.items()}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'datasets/prediction/drive_20261124T213905Z.npz')
    parser.add_argument('--bag', type=Path, help='read a recorded drive directly instead (needs ROS 2)')
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002', help='route for --bag')
    parser.add_argument('--models', nargs='+', default=['cv', 'imm', 'imm+map', 'imm+hd'],
                        help="registered model names, '+map' adds lane following along the mapped route, '+hd' rule-based "
                             "prediction along the HD map lanes (tools/hdmap_lanes.py)")
    parser.add_argument('--hdmap', type=Path, default=ROOT / 'maps/local_gps_recording_002/hdmap/hdmap3d_enu.json',
                        help="aligned HD map for '+hd'")
    parser.add_argument('--plugin', type=Path, action='append', default=[],
                        help='python file that registers extra models with prediction_models.register_model')
    parser.add_argument('--min-score', type=float, default=0.35, help='detections below this are dropped (live: 0.35)')
    parser.add_argument('--quick', action='store_true', help='first 4 minutes only, for fast iteration')
    parser.add_argument('--every-s', type=float, default=0.5)
    parser.add_argument('--out', type=Path, help='default: evaluation/prediction_benchmark_<drive>.json')
    args = parser.parse_args()
    for plugin in args.plugin:
        spec = importlib.util.spec_from_file_location(plugin.stem, plugin)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
    unknown = [m for m in args.models if m.split('+')[0] not in MODELS]
    if unknown:
        parser.error(f'unknown model(s) {unknown}; registered: {sorted(MODELS)}')
    hdmap = None
    if any(m.endswith('+hd') for m in args.models):
        from hdmap_lanes import HDMapLanes
        hdmap = HDMapLanes(args.hdmap)
    t0 = time.time()
    if args.bag:
        frames, route = frames_from_bag(args.bag, args.min_score), Route(args.map / 'gps_alignment/trajectory_enu.csv')
        source = args.bag
    else:
        frames, route_xyz, _, _ = load_dataset(args.dataset, args.min_score)
        route, source = Route(route_xyz), args.dataset
    if args.quick:
        frames = [f for f in frames if f[0] - frames[0][0] <= 240.0]
    gt, per_frame = ground_truth(frames)
    t_split = (frames[0][0] + frames[-1][0]) / 2
    print(f'{len(frames)} frames over {(frames[-1][0] - frames[0][0]) / 60:.1f} min, {len(gt)} reference tracks '
          f'({time.time() - t0:.0f} s)', flush=True)
    report = {'source': str(source), 'min_score': args.min_score, 'quick': args.quick, 'frames': len(frames), 'minutes': round((frames[-1][0] - frames[0][0]) / 60, 1),
              'split': 'dev = first half of the drive, test = second half', 'models': {}}
    for model in args.models:
        t1 = time.time()
        samples = run(model, frames, gt, per_frame, route, args.every_s, hdmap)
        report['models'][model] = score(samples, gt, t_split)
        print(f'{model}: {len(samples)} predictions ({time.time() - t1:.0f} s)', flush=True)
    name = args.bag.parent.name if args.bag else args.dataset.stem
    out = args.out or ROOT / 'evaluation' / f'prediction_benchmark_{name}{"_quick" if args.quick else ""}.json'
    out.write_text(json.dumps(report, indent=1) + '\n')
    rows = [('vehicle', 'straight_fde_3s'), ('vehicle', 'turning_fde_3s'), ('vehicle', 'all_fde_5s'), ('vehicle', 'all_ade'),
            ('vehicle', 'missed_motion'), ('vehicle', 'still_predicted_moving'), ('cyclist', 'all_fde_3s'),
            ('pedestrian', 'all_fde_3s'), ('pedestrian', 'all_fde_4s'), ('pedestrian', 'missed_motion'),
            ('pedestrian', 'still_predicted_moving')]
    for half in ('dev', 'test'):
        print(f'\n{half}: median (mean) [n]; rates are fractions')
        print(f"{'':34s}" + ''.join(f'{m:>22s}' for m in args.models))
        for group, metric in rows:
            cells = []
            for m in args.models:
                v = report['models'][m][half].get(group, {}).get(metric)
                cells.append('-' if v is None else (f"{v['mean']:.2f} [{v['n']}]" if metric in ('missed_motion', 'still_predicted_moving')
                                                    else f"{v['median']:.2f} ({v['mean']:.2f}) [{v['n']}]"))
            print(f'{group + " " + metric:34s}' + ''.join(f'{c:>22s}' for c in cells))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
