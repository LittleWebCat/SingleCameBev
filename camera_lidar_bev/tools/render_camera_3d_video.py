#!/usr/bin/env python3
"""Render camera-only SMOKE 3D predictions from a recorded camera stream.

Uses the existing TensorRT engine and production-compatible SMOKE decoder.
No model conversion, training, LiDAR processing, or accuracy validation runs.
Run with the project's CUDA-capable virtual environment and ROS environment.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from build_camera_3d_engine import decoded_samples
from smoke_model import prepare_image


class SmokeTensorRT:
    """Keep TensorRT execution context and CUDA allocations across frames."""

    def __init__(self, engine_path):
        import torch
        import tensorrt as trt

        self.torch = torch
        torch.set_num_threads(2)
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError('Cannot deserialize existing SMOKE engine')
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.current_stream()
        self.buffers = {
            'image': torch.empty((1, 3, 384, 1280), dtype=torch.float32, device='cuda'),
            'heatmap': torch.empty((1, 3, 96, 320), dtype=torch.float32, device='cuda'),
            'regression': torch.empty((1, 8, 96, 320), dtype=torch.float32, device='cuda'),
        }
        for name, tensor in self.buffers.items():
            if tuple(self.engine.get_tensor_shape(name)) != tuple(tensor.shape):
                raise RuntimeError(f'Unexpected existing engine shape for {name}')
            if self.engine.get_tensor_dtype(name) != trt.float32:
                raise RuntimeError(f'Unexpected existing engine dtype for {name}')
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f'Cannot bind existing engine tensor {name}')

    def infer(self, prepared):
        self.buffers['image'].copy_(self.torch.from_numpy(prepared))
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('SMOKE TensorRT inference failed')
        self.stream.synchronize()
        return [self.buffers[name].cpu().numpy()
                for name in ('heatmap', 'regression')]


def load_calibration(path):
    import yaml
    data = yaml.safe_load(path.read_text())
    size = (int(data['image_width']), int(data['image_height']))
    intrinsic = np.asarray(data['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(data['distortion_coefficients']['data'], dtype=np.float64)
    maps = cv2.initUndistortRectifyMap(intrinsic, distortion, None, intrinsic,
                                      size, cv2.CV_32FC1)
    return size, intrinsic, distortion, maps


def project_predictions(boxes, intrinsic, score_threshold, maximum_depth):
    accepted = []
    for box in boxes:
        cls = box['class_id']
        center, dims, yaw = box['xyz'], box['lhw'], box['yaw']
        if cls not in (0, 2) or box['score'] < score_threshold:
            continue
        if not .5 <= center[2] <= maximum_depth:
            continue
        rotation = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                             [-np.sin(yaw), 0, np.cos(yaw)]])
        corners = np.asarray([
            rotation @ np.array([(1 if i & 1 else -1)*dims[0]/2,
                                 (1 if i & 2 else -1)*dims[1]/2,
                                 (1 if i & 4 else -1)*dims[2]/2]) + center
            for i in range(8)])
        if not np.isfinite(corners).all() or (corners[:, 2] <= .3).any():
            continue
        pixels = corners @ intrinsic.T
        pixels = pixels[:, :2] / pixels[:, 2:]
        if not np.isfinite(pixels).all():
            continue
        accepted.append({
            'class': 'car' if cls == 0 else 'person',
            'confidence': float(box['score']),
            'center_camera_xyz_m': center.tolist(),
            'size_lhw_m': dims.tolist(),
            'size_lwh_m': dims[[0, 2, 1]].tolist(),
            'rotation_y_rad': float(yaw),
            'projected_corners_uv': pixels.tolist(),
        })
    return accepted


def label(canvas, text, origin, color, scale=.45):
    x, y = origin
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = min(max(4, x), max(4, canvas.shape[1] - width - 5))
    y = min(max(height + 5, y), canvas.shape[0] - baseline - 5)
    cv2.rectangle(canvas, (x - 3, y - height - 3),
                  (x + width + 3, y + baseline + 3), (18, 18, 18), -1)
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)


def overlay(rectified, boxes, metadata, fps, score_threshold):
    canvas = rectified.copy()
    for box in boxes:
        pixels = np.asarray(box['projected_corners_uv'])
        color = (255, 0, 255) if box['class'] == 'person' else (0, 165, 255)
        for i in range(8):
            for bit in (1, 2, 4):
                j = i ^ bit
                if j <= i:
                    continue
                a = tuple(np.clip(pixels[i], -10000, 10000).astype(int))
                b = tuple(np.clip(pixels[j], -10000, 10000).astype(int))
                visible, a, b = cv2.clipLine((0, 0, canvas.shape[1], canvas.shape[0]), a, b)
                if visible:
                    cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
        length, width, height = box['size_lwh_m']
        depth = box['center_camera_xyz_m'][2]
        text = (f'{box["class"]} {box["confidence"]:.2f} | '
                f'{length:.1f} x {width:.1f} x {height:.1f} m | Z {depth:.1f} m')
        x, y = pixels.min(axis=0).astype(int)
        label(canvas, text, (int(x), int(y - 7)), color)
    result = cv2.copyMakeBorder(canvas, 0, 80, 0, 0,
                               cv2.BORDER_CONSTANT, value=(14, 14, 14))
    offset = float(metadata.get('bag_offset_s', metadata['time_s']))
    minutes, seconds = divmod(offset, 60)
    held = ' | held sensor frame' if metadata.get('held') else ''
    lines = [
        (f'CAMERA-ONLY 3D | SMOKE | recording {int(minutes):02d}:{seconds:04.1f}'
         f' | {len(boxes)} predicted boxes{held}'),
        (f'Orange: car | Magenta: person | score >= {score_threshold:.2f}'
         ' | size: length x width x height (m)'),
        f'Model predictions | No LiDAR used | {fps:g} fps playback',
    ]
    for i, text in enumerate(lines):
        cv2.putText(result, text, (12, canvas.shape[0] + 22 + i*23),
                    cv2.FONT_HERSHEY_SIMPLEX, .48,
                    (240, 240, 240) if i == 0 else (185, 185, 185),
                    1, cv2.LINE_AA)
    return result


def contact_sheet(snapshots, best_frame):
    selected = [snapshots[key] for key in sorted(snapshots)]
    if best_frame is not None:
        selected.append(best_frame)
    if not selected:
        return None
    tile_width = 512
    tile_height = round(selected[0].shape[0]*tile_width/selected[0].shape[1])
    rows = (len(selected) + 2)//3
    sheet = np.full((rows*tile_height, 3*tile_width, 3), 14, dtype=np.uint8)
    for i, canvas in enumerate(selected):
        thumbnail = cv2.resize(canvas, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        y, x = (i//3)*tile_height, (i%3)*tile_width
        sheet[y:y+tile_height, x:x+tile_width] = thumbnail
    return sheet


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--fps', type=float, default=10.)
    parser.add_argument('--engine', type=Path,
                        default=root/'models/camera_3d_001/smoke_fp16_raw.engine')
    parser.add_argument('--calibration', type=Path, default=root /
                        'config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml')
    parser.add_argument('--score-threshold', type=float, default=.25)
    parser.add_argument('--maximum-depth', type=float, default=60.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.fps <= 0 or args.score_threshold < .25 or args.maximum_depth <= .5:
        parser.error('Need positive fps, score-threshold >= .25 and maximum-depth > .5')
    if args.output.suffix.lower() != '.mp4':
        parser.error('Output must be an MP4 path')
    paths = {
        'video': args.output,
        'summary': args.output.with_suffix('.json'),
        'detections': args.output.with_suffix('.detections.jsonl'),
        'preview': args.output.with_name(args.output.stem + '_preview.png'),
        'contact_sheet': args.output.with_name(args.output.stem + '_contact_sheet.png'),
    }
    if any(path.exists() for path in paths.values()):
        parser.error('Choose a new output filename; existing results are preserved')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    size, intrinsic, distortion, maps = load_calibration(args.calibration)
    print(json.dumps({'stage': 'loading_existing_engine', 'engine': str(args.engine)}), flush=True)
    runner = SmokeTensorRT(args.engine)
    print(json.dumps({'stage': 'engine_loaded', 'model': 'SMOKE',
                      'camera_only': True, 'playback_fps': args.fps}), flush=True)
    from extract_camera_video import iter_camera_frames, camera_bag_timeline
    timeline = camera_bag_timeline(args.bag)
    targets = np.linspace(0, max(0, int(timeline['duration_ns']/1e9*args.fps) - 1), 6).astype(int)
    targets = set(targets.tolist())
    snapshots, best_frame, best_rank = {}, None, (-1, -1.)
    best_metadata = None
    previous_stamp = None
    rectified, accepted = None, []
    frame_count, inferred_frames, held_frames = 0, 0, 0
    detections_per_class = Counter()
    frames_with_detections = 0
    processing_ms = []
    first_metadata, last_metadata = None, None
    encoder = None
    started = time.perf_counter()
    try:
        with paths['detections'].open('w') as records:
            for frame, metadata in iter_camera_frames(args.bag, fps=args.fps):
                if first_metadata is None:
                    first_metadata = dict(metadata)
                last_metadata = dict(metadata)
                if (frame.shape[1], frame.shape[0]) != size:
                    raise RuntimeError('Recorded camera dimensions differ from native calibration')
                stamp = metadata['source_stamp_ns']
                reused = previous_stamp == stamp
                began = time.perf_counter()
                if not reused:
                    rectified = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
                    prepared, affine = prepare_image(rectified)
                    heads = runner.infer(prepared)
                    accepted = project_predictions(decoded_samples(*heads, affine, intrinsic),
                                                    intrinsic, args.score_threshold,
                                                    args.maximum_depth)
                    previous_stamp = stamp
                    inferred_frames += 1
                canvas = overlay(rectified, accepted, metadata, args.fps, args.score_threshold)
                if not reused:
                    processing_ms.append((time.perf_counter() - began)*1000.)
                    detections_per_class.update(box['class'] for box in accepted)
                held_frames += int(bool(metadata.get('held', reused)))
                frames_with_detections += int(bool(accepted))
                records.write(json.dumps({**metadata, 'inference_reused': reused,
                                          'detections': accepted}, separators=(',', ':')) + '\n')
                if encoder is None:
                    command = [
                        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                        '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size',
                        f'{canvas.shape[1]}x{canvas.shape[0]}', '-framerate', str(args.fps),
                        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2',
                        '-movflags', '+faststart', str(args.output),
                    ]
                    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
                encoder.stdin.write(canvas.tobytes())
                index = int(metadata.get('frame_index', frame_count))
                if index in targets:
                    snapshots[index] = canvas.copy()
                rank = (len(accepted), max((box['confidence'] for box in accepted), default=0.))
                if rank > best_rank:
                    best_rank, best_frame = rank, canvas.copy()
                    best_metadata = dict(metadata)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(json.dumps({'stage': 'rendering', 'frames': frame_count,
                                      'unique_source_frames': inferred_frames,
                                      'recording_offset_s': metadata.get('bag_offset_s', metadata['time_s']),
                                      'predicted_boxes_current_frame': len(accepted),
                                      'elapsed_s': round(time.perf_counter() - started, 2)}), flush=True)
        if encoder is None:
            raise RuntimeError('Recording has no camera frames')
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError('FFmpeg could not encode the result video')
    except BaseException:
        if encoder is not None and encoder.poll() is None:
            if encoder.stdin is not None:
                encoder.stdin.close()
            encoder.wait()
        raise
    cv2.imwrite(str(paths['preview']), best_frame)
    sheet = contact_sheet(snapshots, best_frame)
    cv2.imwrite(str(paths['contact_sheet']), sheet)
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_bag': str(args.bag.resolve()),
        'model': 'SMOKE', 'backend': 'TensorRT',
        'engine': str(args.engine.resolve()), 'camera_only': True, 'lidar_used': False,
        'native_calibration': str(args.calibration.resolve()),
        'intrinsic_matrix': intrinsic.tolist(), 'distortion': distortion.tolist(),
        'rendered_image': 'Undistorted native image with unchanged intrinsic matrix',
        'classes': ['car', 'person'], 'score_threshold': args.score_threshold,
        'maximum_depth_m': args.maximum_depth,
        'box_sizes': 'Model predictions, not measured dimensions',
        'physical_accuracy_validated': False,
        'playback_fps': args.fps, 'output_frames': frame_count,
        'video_duration_s': frame_count/args.fps,
        'unique_inferred_source_frames': inferred_frames,
        'held_output_frames': held_frames,
        'output_frames_with_detections': frames_with_detections,
        'predictions_across_unique_source_frames': dict(detections_per_class),
        'most_boxes_in_a_frame': best_rank[0],
        'representative_frame_index': int(best_metadata['frame_index']),
        'representative_time_s': float(best_metadata['time_s']),
        'representative_recording_offset_s': float(best_metadata.get('bag_offset_s',
                                                                   best_metadata['time_s'])),
        'first_frame': first_metadata, 'last_frame': last_metadata,
        'processing_time_ms': {
            'median': float(np.median(processing_ms)),
            'p95': float(np.percentile(processing_ms, 95)),
            'scope': 'Unique-frame undistortion, preprocessing, TensorRT, decode and overlay; '
                     'excludes bag read/image decode and video encoding; not a repeated benchmark',
        },
        'total_wall_s': time.perf_counter() - started,
        'artifacts': {name: str(path.resolve()) for name, path in paths.items()},
    }
    paths['summary'].write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'stage': 'complete', **report}, indent=2), flush=True)


if __name__ == '__main__':
    main()
