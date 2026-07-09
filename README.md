# Single-Camera BEV Depth

Inference-only prototype for one dashcam camera. It detects front objects, estimates depth, projects detections into ego coordinates, and writes annotated frames, depth previews, BEV images, and JSONL records.

Target deployment is NVIDIA Jetson AGX Thor / Jetson T5000 class hardware at 1080p around 30 FPS using TensorRT engines where available.

## Quick Start

```bash
python -m bev_depth.infer --source path/to/video.mp4 --config configs/thor_1080p.yaml --save-dir runs/demo --profile
```

The default config is set up for Thor-style deployment but uses placeholder model paths. Put your exported YOLO TensorRT engine and depth engine/checkpoint paths in `configs/thor_1080p.yaml`.

For local smoke tests without model files, use the synthetic backends:

```bash
python -m bev_depth.infer --source path/to/image.jpg --config configs/local_synthetic.yaml --save-dir runs/smoke
```

Use `--no-save-images` for JSON-only/headless profiling.

For a real local depth smoke test, use MiDaS:

```bash
python -m bev_depth.infer --source path/to/image.jpg --config configs/online_yolo_midas.yaml --save-dir runs/midas --profile
```

MiDaS produces relative depth. Tune `metric_scale`, `metric_bias`, and `inverse_depth` with measured distances before treating `depth_m` as real meters.

## Model Backends

- Detector: `ultralytics`, supports `.pt`, `.onnx`, and `.engine` paths through Ultralytics YOLO.
- Depth: `tensorrt`, `onnx`, `depth_anything_v2`, `midas`, or `constant`.
- TensorRT engines should be built on the target Thor device for best performance and calibration correctness.

## Outputs

- `annotated/`: input frames with detections and distance labels.
- `depth/`: depth visualization frames.
- `bev/`: front BEV grid frames.
- `detections.jsonl`: per-frame structured records.

## Performance Defaults

- 1080p video input.
- Detector input size: `640`.
- Depth input size: `518`.
- Detector every frame.
- Depth every 2 frames, reusing the latest depth map between depth frames.
- FP16 TensorRT baseline; INT8 is an optional deployment/export step.
