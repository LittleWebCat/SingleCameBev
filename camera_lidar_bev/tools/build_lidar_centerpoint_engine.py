#!/usr/bin/env python3
"""Export OpenPCDet CenterPoint-PointPillar (nuScenes) for the native runtime.

Writes the BatchNorm-folded pillar feature network (pfn_weights.bin, run by the
CUDA VFE kernel) and the backbone + CenterHead as ONNX and a TensorRT FP16 engine.
Checks the folded PFN, ONNX Runtime and TensorRT against the PyTorch reference on
a recorded Ouster scan before writing provenance.json.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import centerpoint_pillar as cp  # noqa: E402

CHECKPOINT_URL = 'https://drive.google.com/file/d/1UvGm6mROMyJzeSRu7OD1leU_YWoAZG7v'
CHECKPOINT_SHA256 = '955a3e38868b81f6ae74f09f84a774ef002d03484c6a8e1194b147069c0a6c2a'
OUTPUTS = ('heatmap', 'reg')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ouster_reference_points(npz_path, z_offset):
    """Model-frame [N,5] points from a raw os_lidar scan, matching the ROS node's decode."""
    d = np.load(npz_path)
    xyz = np.stack([-d['x'], -d['y'], d['z'] + 0.038195], 1)
    refl = d['reflectivity'].astype(np.float32)
    ok = np.isfinite(xyz).all(1) & ((xyz ** 2).sum(1) >= 0.25) & (refl <= 255)
    ok &= ~((xyz[:, 0] >= -2.131) & (xyz[:, 0] <= 2.350) & (xyz[:, 1] >= -0.905) & (xyz[:, 1] <= 0.885)
            & (xyz[:, 2] >= -1.8) & (xyz[:, 2] <= 0.0))
    pts = np.concatenate([cp.ouster_to_model(xyz[ok], z_offset), refl[ok, None],
                          np.zeros((ok.sum(), 1), np.float32)], 1)
    return pts.astype(np.float32)


def folded_pfn_canvas(pts, w):
    """NumPy re-implementation of the CUDA VFE kernel math from pfn_weights.bin."""
    o = 0
    def take(n, shape):
        nonlocal o
        a = w[o:o + n].reshape(shape); o += n; return a
    W1, s1, t1 = take(32 * 11, (32, 11)), take(32, 32), take(32, 32)
    W2, s2, t2 = take(64 * 64, (64, 64)), take(64, 64), take(64, 64)
    cx = np.floor((pts[:, 0] - cp.PC_RANGE[0]) / cp.VOXEL_SIZE[0]).astype(np.int64)
    cy = np.floor((pts[:, 1] - cp.PC_RANGE[1]) / cp.VOXEL_SIZE[1]).astype(np.int64)
    keep = (cx >= 0) & (cx < cp.GRID_X) & (cy >= 0) & (cy < cp.GRID_Y)
    pts, cx, cy = pts[keep], cx[keep], cy[keep]
    unq, inv, cnt = np.unique(cy * cp.GRID_X + cx, return_inverse=True, return_counts=True)
    mean = np.zeros((len(unq), 3)); np.add.at(mean, inv, pts[:, :3]); mean /= cnt[:, None]
    f = np.concatenate([pts, pts[:, :3] - mean[inv],
                        np.stack([pts[:, 0] - (cx * cp.VOXEL_SIZE[0] + cp.VOXEL_SIZE[0] / 2 + cp.PC_RANGE[0]),
                                  pts[:, 1] - (cy * cp.VOXEL_SIZE[1] + cp.VOXEL_SIZE[1] / 2 + cp.PC_RANGE[1]),
                                  pts[:, 2] - (cp.VOXEL_SIZE[2] / 2 + cp.PC_RANGE[2])], 1)], 1)
    h1 = np.maximum(f @ W1.T * s1 + t1, 0)
    m1 = np.zeros((len(unq), 32)); np.maximum.at(m1, inv, h1)
    h2 = np.maximum(np.concatenate([h1, m1[inv]], 1) @ W2.T * s2 + t2, 0)
    m2 = np.zeros((len(unq), 64)); np.maximum.at(m2, inv, h2)
    canvas = np.zeros((64, cp.GRID_Y * cp.GRID_X), np.float32); canvas[:, unq] = m2.T
    return canvas.reshape(1, 64, cp.GRID_Y, cp.GRID_X)


def peak_errors(ref_hm, ref_reg, hm, reg, threshold=0.3):
    """Max heatmap error everywhere; regression error only at reference peaks above threshold."""
    hm_err = float(np.abs(hm - ref_hm).max())
    cls, ys, xs = np.nonzero(ref_hm > threshold)
    head = np.searchsorted(np.cumsum([len(c) for c in cp.HEAD_CLASSES]), cls, side='right')
    reg_err = 0.0
    for h, y, x in zip(head, ys, xs):
        sl = slice(h * cp.REG_CHANNELS, (h + 1) * cp.REG_CHANNELS)
        reg_err = max(reg_err, float(np.abs(reg[sl, y, x] - ref_reg[sl, y, x]).max()))
    return hm_err, reg_err, len(cls)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=root / 'models/lidar_objects_002/cbgs_dyn_pp_centerpoint.pth')
    parser.add_argument('--output-directory', type=Path, default=root / 'models/lidar_objects_002')
    parser.add_argument('--reference-scan', type=Path,
                        default=root / 'recordings/lidar_detector_input_001/raw_points.npz')
    parser.add_argument('--z-offset', type=float, default=0.0,
                        help='added to os_sensor z to match the training sensor height')
    parser.add_argument('--trtexec', type=Path, default=Path('/usr/src/tensorrt/bin/trtexec'))
    args = parser.parse_args()

    out = args.output_directory.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        parser.error('Checkpoint hash mismatch')
    model = cp.CenterPointPillar.from_checkpoint(str(args.checkpoint))

    # Reference scan, saved for the C++ checks.
    pts = ouster_reference_points(args.reference_scan, args.z_offset)
    ref_dir = out / 'reference'
    ref_dir.mkdir(exist_ok=True)
    pts.tofile(ref_dir / 'ouster_points_model_frame_f32x5.bin')

    weights = model.pfn_weights()
    weights_path = out / 'pfn_weights.bin'
    weights.astype(np.float32).tofile(weights_path)
    with torch.no_grad():
        canvas = model.forward_dynamic(torch.from_numpy(pts))
        hm_ref, reg_ref = (t[0].numpy() for t in model.bev_heads(canvas))
    canvas = canvas.numpy()
    np.savez_compressed(ref_dir / 'pytorch_dynamic_outputs.npz', heatmap=hm_ref, reg=reg_ref)
    folded = folded_pfn_canvas(pts, weights)
    checks = {}
    canvas_err = float(np.abs(folded - canvas).max())

    onnx_path = out / 'centerpoint_pillar.onnx'

    class Export(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, spatial_features):
            return self.m.bev_heads(spatial_features)

    torch.onnx.export(Export(model), (torch.from_numpy(canvas),), str(onnx_path),
                      input_names=['spatial_features'], output_names=list(OUTPUTS), opset_version=17, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    hm_o, reg_o = (o[0] for o in sess.run(None, {'spatial_features': canvas}))
    checks['onnxruntime_vs_pytorch'] = peak_errors(hm_ref, reg_ref, hm_o, reg_o)

    # TensorRT FP16 build, then a parity run on the same canvas.
    engine = out / 'centerpoint_pillar_fp16.plan'
    build = [str(args.trtexec), f'--onnx={onnx_path}', '--fp16', f'--saveEngine={engine}',
             '--skipInference', '--memPoolSize=workspace:4096']
    start = time.monotonic()
    with (out / 'build_engine.log').open('w') as log:
        if subprocess.run(build, stdout=log, stderr=subprocess.STDOUT).returncode:
            raise SystemExit(f'Engine build failed; inspect {out / "build_engine.log"}')
    build_seconds = time.monotonic() - start

    export_json = ref_dir / 'tensorrt_outputs.json'
    canvas_bin = ref_dir / 'spatial_features.bin'  # 64 MB, removed after the check
    canvas.tofile(canvas_bin)
    run = [str(args.trtexec), f'--loadEngine={engine}', f'--loadInputs=spatial_features:{canvas_bin}',
           f'--exportOutput={export_json}', '--iterations=50', '--warmUp=500', '--avgRuns=50']
    timing = subprocess.run(run, capture_output=True, text=True)
    canvas_bin.unlink()
    if timing.returncode:
        raise SystemExit(timing.stdout[-3000:] + timing.stderr[-3000:])
    trt_out = {o['name']: np.asarray(o['values'], np.float32).reshape([int(v) for v in str(o['dimensions']).split('x')])
               for o in json.loads(export_json.read_text())}
    export_json.unlink()
    checks['tensorrt_fp16_vs_pytorch'] = peak_errors(hm_ref, reg_ref, trt_out['heatmap'][0], trt_out['reg'][0])
    gpu_ms = [line.split('GPU Compute Time:')[1].strip() for line in timing.stdout.splitlines()
              if 'GPU Compute Time: min' in line]  # per-run summary, not the 'Total GPU Compute Time' line

    # Decoded detections from the TensorRT output, in os_sensor.
    boxes, scores, labels = cp.decode(trt_out['heatmap'][0], trt_out['reg'][0])
    ref_boxes, ref_scores, ref_labels = cp.decode(hm_ref, reg_ref)
    sensor = cp.model_boxes_to_ouster(boxes, args.z_offset)
    keep = scores >= 0.3

    provenance = {
        'upstream_repo': 'https://github.com/open-mmlab/OpenPCDet',
        'upstream_config': 'tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml',
        'checkpoint_url': CHECKPOINT_URL, 'checkpoint_file': args.checkpoint.name,
        'checkpoint_sha256': CHECKPOINT_SHA256,
        'pretrained_dataset': 'nuScenes (10 sweeps, Velodyne HDL-32E)',
        'license_notes': 'OpenPCDet code Apache-2.0; weights trained on nuScenes, whose data terms are CC BY-NC-SA 4.0 (non-commercial).',
        'classes': cp.CLASS_NAMES, 'head_classes': cp.HEAD_CLASSES,
        'model_frame': 'nuScenes LIDAR_TOP: x right, y forward, z up; os_sensor -> model is (x,y,z) -> (-y, x, z + z_offset)',
        'z_offset_m': args.z_offset,
        'point_features': 'x, y, z, intensity = Ouster reflectivity 0..255, dt = 0 (single sweep)',
        'point_cloud_range': cp.PC_RANGE, 'voxel_size': cp.VOXEL_SIZE, 'grid': [cp.GRID_X, cp.GRID_Y],
        'feature_map_stride': cp.FEATURE_STRIDE,
        'vfe': 'DynamicPillarVFE (every point, no per-pillar cap) in the native CUDA kernel from pfn_weights.bin',
        'pfn_weights_file': weights_path.name, 'pfn_weights_sha256': sha256(weights_path),
        'pfn_weights_layout': 'float32 W1[32][11], scale1[32], shift1[32], W2[64][64], scale2[64], shift2[64] (BatchNorm folded)',
        'pfn_point_features': 'x,y,z,intensity,dt, x-mean,y-mean,z-mean, x-pillar_cx,y-pillar_cy,z-(z_min+4)',
        'folded_pfn_canvas_max_abs_err': canvas_err,
        'engine_input_contract': {
            'spatial_features': {'dtype': 'float32', 'shape': [1, 64, cp.GRID_Y, cp.GRID_X],
                                 'meaning': 'pillar features at [channel, y_idx, x_idx]; empty cells zero'}},
        'engine_output_contract': {
            'heatmap': {'dtype': 'float32', 'shape': [1, 10, 128, 128], 'meaning': 'sigmoid scores, CLASS_NAMES order'},
            'reg': {'dtype': 'float32', 'shape': [1, 60, 128, 128],
                    'meaning': 'per head: dx_off, dy_off, z, log(dx), log(dy), log(dz), cos, sin, vx, vy'}},
        'postprocess': {'score_thresh': cp.SCORE_THRESH, 'topk_per_head': cp.TOPK_PER_HEAD,
                        'post_center_limit': cp.POST_CENTER_LIMIT, 'nms': 'class-agnostic rotated BEV IoU per head',
                        'nms_iou': cp.NMS_IOU, 'nms_pre_max': cp.NMS_PRE_MAX, 'nms_post_max': cp.NMS_POST_MAX},
        'onnx_file': onnx_path.name, 'onnx_sha256': sha256(onnx_path),
        'engine_file': engine.name, 'engine_sha256': sha256(engine), 'engine_bytes': engine.stat().st_size,
        'engine_build_command': build, 'engine_build_seconds': round(build_seconds, 1),
        'engine_build_log': 'build_engine.log',
        'runtime': {'cuda': torch.version.cuda, 'tensorrt': '10.13.3.9', 'cuda_architecture': 'sm_110',
                    'precision': 'FP16', 'torch_export': torch.__version__},
        'reference_scan': str(args.reference_scan.relative_to(root)),
        'reference_scan_sha256': sha256(args.reference_scan),
        'reference_scan_stats': {'model_points': len(pts), 'pillars': int((np.abs(canvas[0]).sum(0) > 0).sum())},
        'parity_checks': {k: {'heatmap_max_abs_err': v[0], 'reg_max_abs_err_at_peaks': v[1], 'peaks_checked': v[2]}
                          for k, v in checks.items()},
        'reference_detections_ge_0_3': {'tensorrt': int(keep.sum()), 'pytorch_dynamic': int((ref_scores >= 0.3).sum())},
        'tensorrt_gpu_compute_time': gpu_ms[-1] if gpu_ms else None,
        'ouster_accuracy_validated': False, 'metric_size_accuracy_validated': False,
    }
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')

    print(f'folded PFN canvas vs PyTorch: {canvas_err:.2e}')
    for k, v in checks.items():
        print(f'{k:36s} heatmap {v[0]:.2e}  reg@peaks {v[1]:.2e}  ({v[2]} peaks)')
    print('TensorRT GPU compute:', provenance['tensorrt_gpu_compute_time'])
    print(f'Detections >= 0.3: TensorRT {keep.sum()}, PyTorch {int((ref_scores >= 0.3).sum())}')
    for b, s, lab in sorted(zip(sensor[keep], scores[keep], labels[keep]), key=lambda t: -t[1]):
        print(f'  {cp.CLASS_NAMES[lab]:>20s} {s:.2f}  x={b[0]:6.1f} y={b[1]:6.1f}  '
              f'L={b[3]:.1f} W={b[4]:.1f} H={b[5]:.1f} yaw={np.degrees(b[6]):6.1f}')


if __name__ == '__main__':
    main()
