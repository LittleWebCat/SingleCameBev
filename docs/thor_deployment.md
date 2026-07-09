# NVIDIA Thor Deployment Notes

Target: Jetson AGX Thor / Jetson T5000 class device running JetPack 7.

## Runtime Strategy

- Build TensorRT engines on the Thor device, not on a separate GPU.
- Start with FP16 engines for both detector and depth.
- Move to INT8 only after collecting representative dashcam calibration frames.
- Keep detector input fixed at 640 for the first pass.
- Keep depth input fixed at 518 or lower and run depth every 2 frames.
- If the pipeline misses 30 FPS, reduce depth input first, then detector input, then increase depth cadence to 3.

## Detector Export

Use Ultralytics export on the Thor device:

```bash
python scripts/export_yolo_engine.py --weights yolo11n.pt --imgsz 640 --quantize 16
```

For INT8, provide a representative calibration dataset YAML:

```bash
python scripts/export_yolo_engine.py --weights yolo11n.pt --imgsz 640 --quantize 8 --data dashcam_calib.yaml
```

Place the resulting engine at the path configured in `configs/thor_1080p.yaml`, for example:

```text
models/yolo_vehicle_fp16.engine
```

## Depth Engine

The runtime includes a generic single-input/single-output TensorRT depth backend. Export Depth Anything V2-Small to ONNX/TensorRT using your preferred exporter, then configure:

```yaml
models:
  depth:
    backend: tensorrt
    path: models/depth_anything_v2_small_fp16.engine
    input_size: 518
    cadence: 2
```

The expected depth engine input is NCHW RGB normalized with ImageNet mean/std. The output should be a single dense depth map or relative-depth map that can be squeezed to HxW and resized to the source frame size.

## Calibration

Update `configs/thor_1080p.yaml` with real camera intrinsics and measured camera mounting:

- `camera.intrinsics.fx/fy/cx/cy`
- `camera.extrinsics.height_m`
- `camera.extrinsics.pitch_deg`
- `camera.extrinsics.yaw_deg`
- `camera.extrinsics.roll_deg`

If the depth model is relative rather than metric, tune:

- `models.depth.metric_scale`
- `models.depth.metric_bias`
- `models.depth.inverse_depth`

Use a few measured distances to fit scale/bias before trusting BEV range values.

For non-TensorRT local testing, `configs/online_yolo_midas.yaml` uses a real MiDaS relative-depth model. It is useful for checking the full pipeline with real depth structure, but it is not the recommended Thor runtime path.

## Run

```bash
python -m bev_depth.infer \
  --source /path/to/dashcam.mp4 \
  --config configs/thor_1080p.yaml \
  --save-dir runs/thor_1080p \
  --profile
```

Outputs are written to `annotated/`, `depth/`, `bev/`, and `detections.jsonl`.
