#!/usr/bin/env python3
"""Render genuine SMOKE camera-only 3D predictions from a saved calibrated image."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from smoke_model import load_model, prepare_image
from build_camera_3d_engine import decoded_samples


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True,
                        help='Native camera YAML or KITTI 4x4 rectified projection text')
    parser.add_argument('--checkpoint', type=Path,
                        default=root/'models/camera_3d_001/model_final.pth')
    parser.add_argument('--engine', type=Path,
                        default=root/'models/camera_3d_001/smoke_fp16_raw.engine')
    parser.add_argument('--backend', choices=('torch', 'tensorrt'), default='tensorrt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_suffix('.json')
    if args.output.exists() or report_path.exists():
        parser.error('Output already exists; choose a new result path')
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error('Cannot read saved camera image')
    if args.calibration.suffix in ('.yaml', '.yml'):
        import yaml
        data = yaml.safe_load(args.calibration.read_text())
        if image.shape[:2] != (data['image_height'], data['image_width']):
            parser.error('Image dimensions differ from native calibration')
        intrinsic = np.asarray(data['camera_matrix']['data']).reshape(3, 3)
        distortion = np.asarray(data['distortion_coefficients']['data'])
    else:
        projection = np.loadtxt(args.calibration).reshape(4, 4)
        intrinsic, distortion = projection[:3, :3], np.zeros(5)
    image = cv2.undistort(image, intrinsic, distortion, None, intrinsic)
    prepared, affine = prepare_image(image)
    import torch
    input_tensor = torch.from_numpy(prepared).cuda()
    if args.backend == 'torch':
        if not args.checkpoint.is_file():
            parser.error('Pinned checkpoint is missing')
        model = load_model(args.checkpoint)
        with torch.inference_mode():
            tensors = model(input_tensor)
    else:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, '')
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
        if engine is None:
            raise RuntimeError('Cannot load SMOKE engine')
        context = engine.create_execution_context()
        tensors = [torch.empty((1, 3, 96, 320), device='cuda'),
                   torch.empty((1, 8, 96, 320), device='cuda')]
        bindings = {'image': input_tensor, 'heatmap': tensors[0], 'regression': tensors[1]}
        for name, tensor in bindings.items():
            if tuple(engine.get_tensor_shape(name)) != tuple(tensor.shape) or engine.get_tensor_dtype(name) != trt.float32:
                raise RuntimeError(f'Unexpected SMOKE engine tensor: {name}')
            if not context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f'Cannot assign SMOKE tensor: {name}')
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('SMOKE inference failed')
    torch.cuda.synchronize()
    heads = [tensor.detach().cpu().numpy() for tensor in tensors]
    if any(not np.isfinite(head).all() for head in heads):
        raise RuntimeError('Nonfinite SMOKE prediction')
    boxes = decoded_samples(*heads, affine, intrinsic)
    accepted = []
    names = ('car', 'bicycle', 'person')
    canvas = image.copy()
    for box in boxes:
        cls, center, dims, yaw = box['class_id'], box['xyz'], box['lhw'], box['yaw']
        if cls not in (0, 2) or not .5 <= center[2] <= 60:
            continue
        rotation = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                             [-np.sin(yaw), 0, np.cos(yaw)]])
        corners = []
        for i in range(8):
            local = np.array([(1 if i & 1 else -1)*dims[0]/2,
                              (1 if i & 2 else -1)*dims[1]/2,
                              (1 if i & 4 else -1)*dims[2]/2])
            corners.append(rotation @ local + center)
        corners = np.asarray(corners)
        if (corners[:, 2] <= .3).any():
            continue
        pixels = corners @ intrinsic.T
        pixels = pixels[:, :2]/pixels[:, 2:]
        if not np.isfinite(pixels).all():
            continue
        color = (255, 0, 255) if cls == 2 else (0, 165, 255)
        for i in range(8):
            for bit in (1, 2, 4):
                j = i ^ bit
                if j <= i:
                    continue
                a, b = tuple(np.clip(pixels[i], -10000, 10000).astype(int)), tuple(np.clip(pixels[j], -10000, 10000).astype(int))
                visible, a, b = cv2.clipLine((0, 0, canvas.shape[1], canvas.shape[0]), a, b)
                if visible:
                    cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
        x, y = pixels.min(axis=0).astype(int)
        text = f'{names[cls]} {box["score"]:.2f} | {dims[0]:.1f} x {dims[2]:.1f} x {dims[1]:.1f} m'
        cv2.putText(canvas, text, (int(np.clip(x, 0, canvas.shape[1]-1)),
                                  int(np.clip(y-7, 20, canvas.shape[0]-1))),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, color, 1, cv2.LINE_AA)
        accepted.append({'class': names[cls], 'confidence': box['score'],
                         'center_camera_xyz_m': center.tolist(), 'size_lhw_m': dims.tolist(),
                         'rotation_y_rad': yaw, 'projected_corners_uv': pixels.tolist()})
    result = cv2.copyMakeBorder(canvas, 0, 58, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(result, f'SAVED OUTDOOR EXAMPLE | Camera-only SMOKE 3D | {len(accepted)} predicted boxes',
                (12, canvas.shape[0]+22), cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, 'Labels: length x width x height. Model predictions; not measured dimensions.',
                (12, canvas.shape[0]+46), cv2.FONT_HERSHEY_SIMPLEX, .5, (180, 180, 180), 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise RuntimeError('Cannot save 3D result')
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'backend': args.backend,
              'model': 'SMOKE', 'camera_only': True, 'lidar_used': False, 'saved_frame': True,
              'source_image': str(args.image.resolve()),
              'source_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
              'intrinsic_matrix': intrinsic.tolist(), 'native_image_size': list(image.shape[:2][::-1]),
              'physical_accuracy_validated': False, 'detections': accepted}
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(args.output), 'boxes': len(accepted), 'backend': args.backend}))


if __name__ == '__main__':
    main()
