#!/usr/bin/env python3
"""TransFusion-L / BEVFusion-L (LiDAR only) on recorded Ouster scans, for comparison with CenterPoint.

Both checkpoints are the same network (MIT's "BEVFusion LiDAR-only baseline is TransFusion-L"):
sparse voxel backbone (0.075 m, spconv) + BEV backbone + transformer detection head, nuScenes, 10 classes,
+-54 m. The network is built from the vendored OpenPCDet source (third_party/OpenPCDet; its compiled
training ops are not needed for inference and are stubbed). Weights:
  --weights openpcdet  models/lidar_objects_003/transfusion_lidar_nuscenes.pth  (OpenPCDet model zoo)
  --weights mit        models/lidar_objects_003/bevfusion_lidar_only_det.pth    (MIT BEVFusion release)
MIT stores voxels as (x, y, z) and its BEV map as (x, y); OpenPCDet uses (z, y, x) and (y, x). The MIT
weights are converted by reversing the spatial axes of every 3D kernel and transposing every 2D kernel;
its boxes use the older mmdet3d convention ((w, l, h), yaw' = -yaw - pi/2) and are converted back.

Input: a folder from tools/extract_lidar_sweeps.py (scans.csv + scan_NNNNN.bin, os_sensor, reflectivity/255).
Sweeps are merged exactly like the live detector / centerpoint_sweep_eval (newest scan's sensor frame,
time-lag feature, --sweeps / --max-span-s). Output: one JSON line per scan in the centerpoint_sweep_eval
format (boxes in os_sensor: class, score, x, y, z, length, width, height, yaw), so
tools/analyze_sweep_eval.py compares it with CenterPoint.

Needs the project venv with torch + spconv built for this GPU (see docs/lidar_transfusion.md):
  SPCONV_DISABLE_JIT=1 ../.venv/bin/python tools/transfusion_lidar.py data/sweeps_20261010 out.jsonl --weights openpcdet
"""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import sys
import time
import types

for _jit in ('SPCONV_DISABLE_JIT', 'CUMM_DISABLE_JIT'):  # use the prebuilt libraries (JIT rebuilds fail on CUDA 13)
    os.environ.setdefault(_jit, '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import torch
import yaml

if not hasattr(np, 'int'):  # OpenPCDet still uses the alias NumPy 1.24 removed
    np.int = int

ROOT = Path(__file__).resolve().parents[1]
OPENPCDET = ROOT / 'third_party/OpenPCDet'
WEIGHTS = {'openpcdet': ROOT / 'models/lidar_objects_003/transfusion_lidar_nuscenes.pth',
           'mit': ROOT / 'models/lidar_objects_003/bevfusion_lidar_only_det.pth'}
CLASS_NAMES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
               'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
OUR_NAMES = {'pedestrian': 'person'}  # nuScenes -> this project's vocabulary (as the live detector)
PC_RANGE = np.array([-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], np.float32)
VOXEL_SIZE = np.array([0.075, 0.075, 0.2], np.float32)


def import_openpcdet():
    """Import the OpenPCDet modules TransFusion-L needs without its compiled (training-only) ops."""
    def package(name, path):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m

    def stub(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m

    root = OPENPCDET / 'pcdet'
    package('pcdet', root)
    for sub in ('models', 'models/backbones_3d', 'models/backbones_3d/vfe', 'models/backbones_2d',
                'models/backbones_2d/map_to_bev', 'models/dense_heads', 'models/dense_heads/target_assigner',
                'models/model_utils', 'utils', 'ops', 'ops/iou3d_nms', 'ops/roiaware_pool3d'):
        package('pcdet.' + sub.replace('/', '.'), root / sub)
    stub('pcdet.version', __version__='vendored')
    stub('pcdet.ops.iou3d_nms.iou3d_nms_cuda')
    stub('pcdet.ops.iou3d_nms.iou3d_nms_utils')
    stub('pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils')
    stub('SharedArray')
    if 'numba' not in sys.modules:  # only used by training-target helpers
        def jit(*args, **kwargs):
            return args[0] if args and callable(args[0]) else (lambda f: f)
        stub('numba', jit=jit, njit=jit)
    from pcdet.models.backbones_3d.vfe.mean_vfe import MeanVFE
    from pcdet.models.backbones_3d.spconv_backbone import VoxelResBackBone8x
    from pcdet.models.backbones_2d.map_to_bev.height_compression import HeightCompression
    from pcdet.models.backbones_2d.base_bev_backbone import BaseBEVBackbone
    from pcdet.models.dense_heads.transfusion_head import TransFusionHead
    return MeanVFE, VoxelResBackBone8x, HeightCompression, BaseBEVBackbone, TransFusionHead


def to_easydict(d):
    from easydict import EasyDict
    return EasyDict(d)


class TransFusionL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        MeanVFE, VoxelResBackBone8x, HeightCompression, BaseBEVBackbone, TransFusionHead = import_openpcdet()
        cfg = to_easydict(yaml.safe_load((OPENPCDET / 'tools/cfgs/nuscenes_models/transfusion_lidar.yaml').read_text())['MODEL'])
        grid = np.round((PC_RANGE[3:] - PC_RANGE[:3]) / VOXEL_SIZE).astype(np.int64)
        self.vfe = MeanVFE(cfg.VFE, num_point_features=5)
        self.backbone_3d = VoxelResBackBone8x(cfg.BACKBONE_3D, input_channels=5, grid_size=grid)
        self.map_to_bev_module = HeightCompression(cfg.MAP_TO_BEV)
        self.backbone_2d = BaseBEVBackbone(cfg.BACKBONE_2D, input_channels=cfg.MAP_TO_BEV.NUM_BEV_FEATURES)
        self.dense_head = TransFusionHead(cfg.DENSE_HEAD, input_channels=self.backbone_2d.num_bev_features,
                                          num_class=len(CLASS_NAMES), class_names=CLASS_NAMES, grid_size=grid,
                                          point_cloud_range=PC_RANGE, voxel_size=VOXEL_SIZE,
                                          predict_boxes_when_training=False)

    def forward(self, batch):
        for m in (self.vfe, self.backbone_3d, self.map_to_bev_module, self.backbone_2d, self.dense_head):
            batch = m(batch)
        return batch['final_box_dicts'][0]


def mit_to_openpcdet(model_keys, mit_state):
    """MIT BEVFusion LiDAR-only weights -> OpenPCDet TransFusion-L names and axis order (same layer order)."""
    mit_keys = list(mit_state)
    if len(mit_keys) != len(model_keys):
        raise ValueError(f'{len(mit_keys)} MIT tensors vs {len(model_keys)} model tensors')
    out = {}
    for ours, theirs in zip(model_keys, mit_keys):
        w = mit_state[theirs]
        if ours.startswith('backbone_3d.') and w.dim() == 5:
            w = w.permute(2, 1, 0, 3, 4)  # kernel spatial axes x, y, z -> z, y, x
        elif (ours.startswith('backbone_2d.') or ours.startswith('dense_head.')) and w.dim() == 4:
            w = w.transpose(2, 3)  # BEV map (x, y) -> (y, x)
        out[ours] = w.contiguous()
    return out


def load(weights):
    model = TransFusionL()
    state = torch.load(WEIGHTS[weights], map_location='cpu', weights_only=True)
    keys = [k for k in model.state_dict()]
    if weights == 'openpcdet':
        state = {k: v for k, v in state['model_state'].items() if k != 'global_step'}
    else:
        state = mit_to_openpcdet(keys, state['state_dict'])
    for k, v in model.state_dict().items():
        w = state[k]
        if w.dim() == 5 and tuple(w.shape) != tuple(v.shape) and tuple(w.permute(4, 0, 1, 2, 3).shape) == tuple(v.shape):
            state[k] = w.permute(4, 0, 1, 2, 3).contiguous()  # (k, k, k, in, out) -> spconv 2 (out, k, k, k, in)
        if tuple(state[k].shape) != tuple(v.shape):
            raise ValueError(f'{k}: checkpoint {tuple(state[k].shape)} vs model {tuple(v.shape)}')
    model.load_state_dict(state, strict=True)
    return model.cuda().eval()


def quat_matrix(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Sweeps:
    """Same rule as bev_lidar_detection SweepAccumulator: newest frame, drop sweeps beyond count or span."""

    def __init__(self, count, span_s):
        self.count, self.span_ns, self.items = count, int(span_s * 1e9), deque()

    def add(self, points, end_ns, rot, t):
        if self.items and end_ns <= self.items[-1][1]:
            self.items.clear()
        self.items.append((points, end_ns, rot, t))
        while len(self.items) > self.count:
            self.items.popleft()
        while len(self.items) > 1 and end_ns - self.items[0][1] > self.span_ns:
            self.items.popleft()

    def build(self):
        pts, end_k, rk, tk = self.items[-1]
        out = []
        for p, end_j, rj, tj in self.items:
            xyz = p[:, :3] if p is pts else (p[:, :3] @ (rk.T @ rj).T + rk.T @ (tj - tk))
            age = np.full((len(p), 1), (end_k - end_j) * 1e-9, np.float32)
            out.append(np.hstack([xyz.astype(np.float32), p[:, 3:4], age]))
        return np.vstack(out), len(self.items)


def model_points(sensor_points):
    """os_sensor [x, y, z, refl/255, lag] -> nuScenes LIDAR_TOP [x right, y fwd, z, intensity 0..255, lag]."""
    p = sensor_points
    out = np.stack([-p[:, 1], p[:, 0], p[:, 2], p[:, 3] * 255.0, p[:, 4]], 1).astype(np.float32)
    keep = np.all((out[:, :3] >= PC_RANGE[:3]) & (out[:, :3] < PC_RANGE[3:]), 1)
    return out[keep]


def boxes_to_sensor(boxes):
    """Model [x, y, z, dx, dy, dz, yaw, ...] -> os_sensor (as tools/centerpoint_pillar.model_boxes_to_ouster)."""
    out = boxes[:, :7].copy()
    out[:, 0], out[:, 1] = boxes[:, 1], -boxes[:, 0]
    out[:, 6] = np.remainder(boxes[:, 6] - math.pi / 2 + math.pi, 2 * math.pi) - math.pi
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scans', type=Path, help='folder from tools/extract_lidar_sweeps.py')
    parser.add_argument('out', type=Path, help='output .jsonl (centerpoint_sweep_eval format)')
    parser.add_argument('--weights', choices=sorted(WEIGHTS), default='openpcdet')
    parser.add_argument('--sweeps', type=int, default=3)
    parser.add_argument('--max-span-s', type=float, default=0.35)
    parser.add_argument('--score', type=float, default=0.1, help='keep boxes above this; thresholds come in the analysis')
    parser.add_argument('--limit', type=int, default=0, help='first N scans only')
    parser.add_argument('--max-voxels', type=int, default=160000,
                        help='nuScenes default 160000; dense Ouster sweeps fill it, raise it to keep all points')
    args = parser.parse_args()

    from spconv.pytorch.utils import PointToVoxel
    model = load(args.weights)
    voxelize = PointToVoxel(vsize_xyz=VOXEL_SIZE.tolist(), coors_range_xyz=PC_RANGE.tolist(), num_point_features=5,
                            max_num_voxels=args.max_voxels, max_num_points_per_voxel=10, device=torch.device('cuda'))
    rows = [line.split(',') for line in (args.scans / 'scans.csv').read_text().splitlines()[1:]]
    if args.limit:
        rows = rows[:args.limit]
    sweeps, n = Sweeps(args.sweeps, args.max_span_s), 0
    with args.out.open('w') as out, torch.no_grad():
        for c in rows:
            pose_ok = c[10] == '1'
            end_ns = int(c[2])
            raw = np.fromfile(args.scans / c[11].strip(), np.float32).reshape(-1, 4)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if args.sweeps == 1 or not pose_ok:
                if not pose_ok:
                    sweeps.items.clear()
                merged, used = np.hstack([raw, np.zeros((len(raw), 1), np.float32)]), 1
            else:
                sweeps.add(raw, end_ns, quat_matrix([float(v) for v in c[6:10]]), np.array([float(v) for v in c[3:6]]))
                merged, used = sweeps.build()
            pts = torch.from_numpy(model_points(merged)).cuda()
            voxels, coords, num = voxelize(pts)
            coords = torch.cat([torch.zeros_like(coords[:, :1]), coords], 1).int()
            batch = {'batch_size': 1, 'voxels': voxels, 'voxel_num_points': num, 'voxel_coords': coords}
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            res = model(batch)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            boxes = res['pred_boxes'].cpu().numpy()
            if args.weights == 'mit':  # mmdet3d < 1.0 boxes: (w, l, h) and yaw' = -yaw - pi/2
                boxes[:, [3, 4]] = boxes[:, [4, 3]]
                boxes[:, 6] = -boxes[:, 6] - math.pi / 2
            scores, labels = res['pred_scores'].cpu().numpy(), res['pred_labels'].cpu().numpy() - 1
            keep = scores >= args.score
            sensor = boxes_to_sensor(boxes[keep]) if keep.any() else np.zeros((0, 7))
            names = [OUR_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i]) for i in labels[keep]]
            t3 = time.perf_counter()
            out.write(json.dumps({
                'i': int(c[0]), 'end_ns': end_ns, 'sweeps_used': used, 'pose_ok': pose_ok, 'points': int(len(pts)),
                'pillars': int(len(voxels)), 'total_ms': (t3 - t0) * 1e3, 'detector_ms': (t3 - t0) * 1e3,
                'pre_ms': (t1 - t0) * 1e3, 'net_ms': (t2 - t1) * 1e3, 'post_ms': (t3 - t2) * 1e3,
                'boxes': [[nm, round(float(s), 4), *[round(float(v), 4) for v in b]] for nm, s, b in zip(names, scores[keep], sensor)],
            }) + '\n')
            n += 1
            if n % 200 == 0:
                print(f'{n} scans', file=sys.stderr, flush=True)
    print(f'done: {n} scans, {args.sweeps} sweep(s), weights {args.weights} -> {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
