#!/usr/bin/env python3
"""Export the pinned official SMOKE checkpoint and validate native TensorRT heads.

Run with the project's CUDA-capable virtualenv. No downloads or installations
are performed. The output directory must be new, preserving model versions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def decoded_samples(heatmap, regression, affine, intrinsic):
    """Decode reference peaks with official SMOKE's metric conventions."""
    import torch
    from torch.nn import functional as F
    hm = torch.from_numpy(heatmap)
    maxima = hm == F.max_pool2d(hm, 3, stride=1, padding=1)
    scores = np.where(maxima.numpy(), heatmap, 0.).reshape(-1)
    indices = np.argsort(scores)[-50:][::-1]
    indices = indices[scores[indices] >= .25]
    intrinsic_inv = np.linalg.inv(intrinsic)
    feature_affine = np.vstack((affine, [0., 0., 1.]))
    feature_affine[:2] /= 4.
    affine_inv = np.linalg.inv(feature_affine)
    dimensions = np.array([[3.88, 1.63, 1.53], [1.78, 1.70, .58],
                           [.88, 1.73, .67]], np.float64)
    samples = []
    for index in indices:
        cls, y, x = np.unravel_index(index, heatmap.shape[1:])
        r = regression[0, :, y, x].astype(np.float64)
        depth = r[0] * 16.32 + 28.01
        native_pixel = affine_inv @ [x + r[1], y + r[2], 1.]
        location = intrinsic_inv @ (native_pixel * depth)
        dims = np.exp(r[3:6]) * dimensions[cls]
        # Equivalent quadrant-aware implementation of the official atan rules.
        alpha = np.arctan2(r[6], r[7]) - np.pi / 2
        yaw = (alpha + np.arctan(location[0] / (location[2] + 1e-7)) + np.pi) % (2 * np.pi) - np.pi
        samples.append({'class_id': int(cls), 'index': [int(y), int(x)],
                        'score': float(scores[index]), 'xyz': location,
                        'lhw': dims, 'yaw': float(yaw)})
    return samples


def compare_heads(reference, actual, affine, intrinsic):
    if any(not np.isfinite(value).all() for value in actual):
        raise RuntimeError('TensorRT produced non-finite heads')
    errors = {}
    for name, ref, got in zip(('heatmap', 'regression'), reference, actual):
        difference = np.abs(ref - got)
        errors[name] = {'max_abs': float(difference.max()),
                        'mean_abs': float(difference.mean()),
                        'p99_abs': float(np.percentile(difference, 99))}
    samples = decoded_samples(*reference, affine, intrinsic)
    # Compare exactly the same reference peaks, avoiding confidence jitter or a
    # change in peak count being mistaken for a different predicted object.
    position_errors, dimension_errors, yaw_errors = [], [], []
    for sample in samples:
        cls = sample['class_id']
        y, x = sample['index']
        selected = np.zeros_like(actual[0])
        selected[0, cls, y, x] = 1.
        decoded = decoded_samples(selected, actual[1], affine, intrinsic)[0]
        position_errors.append(float(np.linalg.norm(sample['xyz'] - decoded['xyz'])))
        dimension_errors.append(float(np.max(np.abs(sample['lhw'] - decoded['lhw']))))
        delta = sample['yaw'] - decoded['yaw']
        yaw_errors.append(float(abs(np.arctan2(np.sin(delta), np.cos(delta)))))
    errors['decoded_reference_peaks'] = {
        'count': len(samples),
        'max_position_error_m': max(position_errors, default=0.),
        'max_dimension_error_m': max(dimension_errors, default=0.),
        'max_yaw_error_rad': max(yaw_errors, default=0.),
        'reference_classes': [sample['class_id'] for sample in samples]}
    limits = {'heatmap_max_abs': .035, 'position_m': .75,
              'dimension_m': .05, 'yaw_rad': .15}
    errors['limits'] = limits
    passed = (errors['heatmap']['max_abs'] <= limits['heatmap_max_abs'] and
              max(position_errors, default=0.) <= limits['position_m'] and
              max(dimension_errors, default=0.) <= limits['dimension_m'] and
              max(yaw_errors, default=0.) <= limits['yaw_rad'])
    errors['passed'] = passed
    return errors


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--reference-image', type=Path, required=True)
    parser.add_argument('--camera-calibration', type=Path, default=project /
                        'config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml',
                        help='native ROS camera YAML, or a KITTI demo flat 4x4 projection .txt')
    parser.add_argument('--fp32', action='store_true', help='disable internal FP16')
    parser.add_argument('--iterations', type=int, default=30)
    args = parser.parse_args()
    if args.output_directory.exists() or args.output_directory.is_symlink():
        parser.error('output-directory already exists; choose a new model version')
    if not args.checkpoint.is_file() or not args.reference_image.is_file():
        parser.error('checkpoint and reference-image must be existing local files')
    if not 1 <= args.iterations <= 1000:
        parser.error('iterations must be between 1 and 1000')

    import cv2
    import onnx
    import torch
    import torchvision
    import tensorrt as trt
    import yaml
    from smoke_model import load_model, prepare_image

    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; use the CUDA-capable project virtualenv')
    destination = args.output_directory.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = destination / 'model_final.pth'
    shutil.copyfile(args.checkpoint, checkpoint)
    source_hash = sha256(checkpoint)
    if source_hash != 'e2cb8a1643b37b275907867716fce1f682f113dc6dd2dda141ffa007023f9546':
        raise RuntimeError('Checkpoint SHA256 differs from the pinned official SMOKE model')
    image = cv2.imread(str(args.reference_image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Reference image could not be read')
    if args.camera_calibration.suffix == '.txt':
        projection = np.loadtxt(args.camera_calibration).reshape(4, 4)
        intrinsic, distortion = projection[:3, :3], np.zeros(5)
        calibration = {'image_height': image.shape[0], 'image_width': image.shape[1]}
    else:
        calibration = yaml.safe_load(args.camera_calibration.read_text())
        intrinsic = np.array(calibration['camera_matrix']['data']).reshape(3, 3)
        distortion = np.array(calibration['distortion_coefficients']['data'])
    if image.shape[:2] != (calibration['image_height'], calibration['image_width']):
        raise RuntimeError('Reference image dimensions must match native calibration')
    prepared, affine = prepare_image(cv2.undistort(image, intrinsic, distortion, None, intrinsic))
    model = load_model(checkpoint, device='cuda')
    image_cuda = torch.from_numpy(prepared).cuda()
    torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(3):
            reference_tensors = model(image_cuda)
        reference = [value.cpu().numpy() for value in reference_tensors]
    if any(not np.isfinite(value).all() for value in reference):
        raise RuntimeError('Official Torch reference produced non-finite heads')
    if not decoded_samples(*reference, affine, intrinsic):
        raise RuntimeError('Reference has no supported 3D detections at confidence 0.25; '
                           'use a calibrated KITTI example or a suitable image before engine build')
    print(json.dumps({'stage': 'strict_checkpoint_loaded', 'weights_sha256': source_hash,
                      'state_parameters': len(model.state_dict()),
                      'reference_head_shapes': [list(value.shape) for value in reference]}), flush=True)

    onnx_path = destination / 'smoke_raw.onnx'
    model.cpu()
    with torch.inference_mode():
        torch.onnx.export(model, torch.from_numpy(prepared), str(onnx_path),
                          input_names=['image'], output_names=['heatmap', 'regression'],
                          opset_version=17, dynamo=False, do_constant_folding=True,
                          custom_opsets={'trt.plugins': 1})
    del model, reference_tensors
    torch.cuda.empty_cache()
    exported = onnx.load(str(onnx_path))
    onnx.checker.check_model(exported)
    print(json.dumps({'stage': 'onnx_exported', 'nodes': len(exported.graph.node)}), flush=True)

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse(onnx_path.read_bytes()):
        raise RuntimeError('\n'.join(str(onnx_parser.get_error(i))
                                     for i in range(onnx_parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1024 * 1024 * 1024)
    if not args.fp32:
        config.set_flag(trt.BuilderFlag.FP16)
    print(json.dumps({'stage': 'building_engine', 'layers': network.num_layers,
                      'internal_fp16': not args.fp32}), flush=True)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT could not build the exported SMOKE model')
    candidate = destination / 'smoke_candidate_unvalidated.engine'
    candidate.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError('Fresh engine could not be deserialized')
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    expected = {'image': (1, 3, 384, 1280), 'heatmap': (1, 3, 96, 320),
                'regression': (1, 8, 96, 320)}
    if set(names) != set(expected):
        raise RuntimeError(f'Unexpected engine tensor names: {names}')
    bindings = []
    context = engine.create_execution_context()
    tensors = {'image': image_cuda}
    for name in names:
        shape = tuple(engine.get_tensor_shape(name))
        if shape != expected[name] or engine.get_tensor_dtype(name) != trt.float32:
            raise RuntimeError(f'Unexpected engine interface for {name}: {shape}')
        if name != 'image':
            tensors[name] = torch.empty(shape, dtype=torch.float32, device='cuda')
        if not context.set_tensor_address(name, tensors[name].data_ptr()):
            raise RuntimeError(f'Could not bind {name}')
        bindings.append({'name': name, 'mode': str(engine.get_tensor_mode(name)),
                         'shape': list(shape), 'dtype': 'FLOAT'})
    stream = torch.cuda.current_stream()

    def infer():
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT inference failed')

    for _ in range(5):
        infer()
    stream.synchronize()
    actual = [tensors[name].cpu().numpy() for name in ('heatmap', 'regression')]
    validation = compare_heads(reference, actual, affine, intrinsic)
    if not validation['passed']:
        (destination / 'failed_numeric_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
        raise RuntimeError(f'TensorRT numeric validation failed: {validation}')
    durations = []
    for _ in range(args.iterations):
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        infer()
        end_event.record(stream)
        end_event.synchronize()
        durations.append(start_event.elapsed_time(end_event))
    engine_path = destination / 'smoke_fp16_raw.engine'
    candidate.rename(engine_path)
    report = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_name': 'SMOKE', 'source_repository': 'https://github.com/lzccccc/SMOKE',
        'source_commit': 'bc5d2bba66e2d66fa56b7b599d55457cb1a05b33',
        'checkpoint_source_url': 'https://drive.google.com/file/d/11VK8_HfR7t0wm-6dCNP5KS3Vh-Qm686-/view',
        'source_license': 'MIT', 'weights_sha256': source_hash,
        'engine_file': engine_path.name, 'engine_sha256': sha256(engine_path),
        'onnx_file': onnx_path.name, 'onnx_sha256': sha256(onnx_path),
        'classes': ['Car', 'Cyclist', 'Pedestrian'], 'bindings': bindings,
        'torch_version': torch.__version__, 'torchvision_version': torchvision.__version__,
        'torch_cuda_version': torch.version.cuda, 'tensorrt_version': trt.__version__,
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_compute_capability': list(torch.cuda.get_device_capability(0)),
        'onnx_version': onnx.__version__, 'opencv_version': cv2.__version__,
        'export': {'opset': 17, 'internal_fp16': not args.fp32, 'batch': 1,
                   'deform_plugin': 'ModulatedDeformConv2d', 'deform_plugin_version': '2'},
        'preprocessing': {'input': 'undistorted native BGR', 'pixel_scale': 1 / 255,
                          'mean': [.485, .456, .406], 'std': [.229, .224, .225],
                          'uniform_scale': '1280/native_width',
                          'centered_vertical_crop': True, 'network_width': 1280, 'network_height': 384,
                          'reference_affine': affine.tolist()},
        'numeric_validation': validation,
        'reference_image_sha256': sha256(args.reference_image),
        'reference_image_dimensions': [image.shape[1], image.shape[0]],
        'benchmark': {'warmup': 5, 'iterations': args.iterations,
                      'inference_median_ms': float(np.median(durations)),
                      'inference_p95_ms': float(np.percentile(durations, 95)),
                      'inference_max_ms': float(np.max(durations)),
                      'scope': 'engine only; excludes capture, preprocessing, decoder, ROS and display'},
        'engine_build_and_validation_seconds': time.perf_counter() - started,
        'accuracy_on_live_camera_validated': False, 'speed_on_live_camera_measured': False}
    (destination / 'provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'engine': str(engine_path), 'provenance': str(destination / 'provenance.json'),
                      'validation': validation, 'benchmark': report['benchmark']}, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
