# Single-Camera BEV Depth

Single-camera object detection, metric depth, and front BEV projection for NVIDIA Jetson AGX Thor. The current workspace is configured for a USB camera on `/dev/video0`, TensorRT YOLO detection, TensorRT Depth Anything V2 metric depth, and a `40m x 80m` BEV grid rendered as `400x800` pixels.

The pipeline can process videos, image folders, single images, or a live camera. It writes annotated camera frames, depth previews, BEV frames, and per-frame JSONL records.

## Current Thor Setup

The main runtime config is `configs/thor_1080p.yaml`.

Current live-camera defaults:

- Camera node: `/dev/video0`
- Capture mode: `800x600 MJPG @ 30 FPS`
- Detector: `models/yolo_vehicle_fp16.engine`
- Depth: `models/depth_anything_v2_small_fp16.engine`
- Depth cadence: every 2 frames
- BEV range: `40m` lateral by `80m` forward
- BEV resolution: `10 px/m`, output `400x800`

The connected camera also supports `1600x1200 MJPG @ 15 FPS`. That mode gives more image detail but lower real-time FPS.

## Environment

Use the repo virtual environment:

```bash
cd /home/agxthor/SingleCameraBev
```

Most Thor TensorRT runs need the CUDA/TensorRT library path:

```bash
export PYTHONPATH=/home/agxthor/SingleCameraBev
export LD_LIBRARY_PATH=/home/agxthor/SingleCameraBev/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:/home/agxthor/SingleCameraBev/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/agxthor/SingleCameraBev/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:/home/agxthor/SingleCameraBev/.venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:/home/agxthor/SingleCameraBev/.venv/lib/python3.12/site-packages/nvidia/nvshmem/lib:/usr/local/cuda/lib64
```

Run tests:

```bash
.venv/bin/python -m unittest discover -v
```

## Camera Check

List cameras:

```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats-ext
```

This workspace detected:

- `/dev/video0`: HD USB Camera image stream
- `/dev/video1`: UVC metadata node

Capture a single frame:

```bash
ffmpeg -y -f v4l2 -input_format mjpeg -video_size 1600x1200 -framerate 15 \
  -i /dev/video0 -frames:v 1 runs/camera_check.jpg
```

## Live BEV, Highest FPS

For the fastest live BEV generation, use `800x600 MJPG @ 30 FPS`, save only BEV frames, and skip camera/depth image rendering:

```bash
.venv/bin/python -m bev_depth.infer \
  --source 0 \
  --config configs/thor_1080p.yaml \
  --save-dir runs/live_camera_bev \
  --save-bev-only \
  --profile
```

Measured on this Thor:

- No image saving: about `31 FPS`
- Saving BEV-only frames: about `28 FPS`

BEV frames are written to:

```text
runs/live_camera_bev/bev/
```

Build a BEV video from the saved frames:

```bash
ffmpeg -y -framerate 30 -i runs/live_camera_bev/bev/%06d.jpg \
  -c:v libx264 -pix_fmt yuv420p -crf 20 \
  runs/live_camera_bev/bev.mp4
```

## Live BEV Visualization

Use `--display-bev` to show only the BEV window in real time. This avoids the extra annotated-camera and depth preview windows.

```bash
DISPLAY=:1 .venv/bin/python -m bev_depth.infer \
  --source 0 \
  --config configs/thor_1080p.yaml \
  --save-dir runs/live_camera_bev_display \
  --no-save-images \
  --display-bev \
  --profile
```

Press `Esc` in the OpenCV window to stop.

## 1600x1200 Camera Mode

Use this when you want higher camera detail:

```bash
.venv/bin/python -m bev_depth.infer \
  --source 0 \
  --config configs/thor_1080p.yaml \
  --input-width 1600 \
  --input-height 1200 \
  --input-fps 15 \
  --input-fourcc MJPG \
  --save-dir runs/live_camera_1600_bev \
  --save-bev-only \
  --profile
```

Measured on this Thor at `1600x1200 MJPG @ 15 FPS`:

- No saving/display: `13.0 FPS` end-to-end
- Saving BEV-only frames: `12.7 FPS` end-to-end
- BEV-only live display: `9.1 FPS` end-to-end

The camera advertises only `15 FPS` at `1600x1200`, so this mode is camera-limited.

## Video Inference

Run the full pipeline on the sample video:

```bash
.venv/bin/python -m bev_depth.infer \
  --source /home/agxthor/SingleCameraBev/video.mkv \
  --config configs/thor_1080p.yaml \
  --save-dir runs/thor_yolo_depth_video \
  --profile
```

Outputs:

- `annotated/`: camera frames with object boxes and distance labels
- `depth/`: depth preview frames
- `bev/`: BEV frames, currently `400x800`
- `detections.jsonl`: structured per-frame detections, depth, ego coordinates, and BEV pixels

Encode individual videos:

```bash
ffmpeg -y -framerate 10 -i runs/thor_yolo_depth_video/annotated/%06d.jpg \
  -c:v libx264 -pix_fmt yuv420p -crf 20 \
  runs/thor_yolo_depth_video/annotated_depth.mp4

ffmpeg -y -framerate 10 -i runs/thor_yolo_depth_video/depth/%06d.jpg \
  -c:v libx264 -pix_fmt yuv420p -crf 20 \
  runs/thor_yolo_depth_video/depth_preview.mp4

ffmpeg -y -framerate 10 -i runs/thor_yolo_depth_video/bev/%06d.jpg \
  -c:v libx264 -pix_fmt yuv420p -crf 20 \
  runs/thor_yolo_depth_video/bev.mp4
```

Encode the combined camera/depth/BEV dashboard:

```bash
ffmpeg -y \
  -framerate 10 -i runs/thor_yolo_depth_video/annotated/%06d.jpg \
  -framerate 10 -i runs/thor_yolo_depth_video/depth/%06d.jpg \
  -framerate 10 -i runs/thor_yolo_depth_video/bev/%06d.jpg \
  -filter_complex "[0:v]scale=1440:1080[a];[1:v]scale=480:360[d];[2:v]scale=405:1080[b];color=c=black:s=1920x1080:r=10:d=300.5[base];[base][a]overlay=0:0:shortest=1[tmp1];[tmp1][d]overlay=1440:0:shortest=1[tmp2];[tmp2][b]overlay=1477:0:shortest=1[out]" \
  -map "[out]" -frames:v 3005 -c:v libx264 -pix_fmt yuv420p -crf 20 \
  runs/thor_yolo_depth_video/combined_annotated_depth_bev.mp4
```

## TensorRT Engines

Current engine paths:

```text
models/yolo_vehicle_fp16.engine
models/depth_anything_v2_small_fp16.engine
```

Detector engine benchmark on this Thor:

- `trtexec --loadEngine=models/yolo_vehicle_fp16.engine`
- Throughput: about `803 qps`
- Mean GPU compute: about `1.24 ms`

Depth engine benchmark on this Thor:

- `trtexec --loadEngine=models/depth_anything_v2_small_fp16.engine`
- Throughput: about `265 qps`
- Mean GPU compute: about `3.56 ms`

### Export YOLO Engine

The YOLO helper exports from an Ultralytics weight:

```bash
.venv/bin/python scripts/export_yolo_engine.py \
  --weights yolo11n.pt \
  --imgsz 640 \
  --quantize 16 \
  --no-simplify
```

Place or rename the result to:

```text
models/yolo_vehicle_fp16.engine
```

### Export Depth Anything V2 Metric Engine

This workspace uses Depth Anything V2 metric small, exported to ONNX and built with TensorRT.

Checkpoint:

```text
models/checkpoints/depth_anything_v2_metric_vkitti_vits.pth
```

ONNX:

```text
models/depth_anything_v2_metric_vkitti_vits_518.onnx
models/depth_anything_v2_metric_vkitti_vits_518.onnx.data
```

TensorRT engine:

```text
models/depth_anything_v2_small_fp16.engine
```

Exporter:

```bash
.venv/bin/python scripts/export_depth_anything_metric_onnx.py \
  --checkpoint models/checkpoints/depth_anything_v2_metric_vkitti_vits.pth \
  --encoder vits \
  --input-size 518 \
  --output models/depth_anything_v2_metric_vkitti_vits_518.onnx
```

Build TensorRT:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/depth_anything_v2_metric_vkitti_vits_518.onnx \
  --saveEngine=models/depth_anything_v2_small_fp16.engine \
  --fp16
```

## BEV Geometry

BEV output size comes from:

```text
width  = lateral_range_m * pixels_per_meter
height = forward_range_m * pixels_per_meter
```

Current config:

```yaml
bev:
  forward_range_m: 80.0
  lateral_range_m: 40.0
  pixels_per_meter: 10
```

So the BEV frame size is:

```text
400x800
```

If you already have a `detections.jsonl`, redraw BEV frames with a new range without rerunning detection/depth:

```bash
.venv/bin/python scripts/redraw_bev_from_jsonl.py \
  --jsonl runs/thor_yolo_depth_video/detections.jsonl \
  --out-dir runs/thor_yolo_depth_video/bev \
  --forward-range-m 80 \
  --lateral-range-m 40 \
  --pixels-per-meter 10
```

## CLI Options Added For Live Use

- `--max-frames N`: stop after N frames
- `--save-bev-only`: save only BEV images, not annotated/depth frames
- `--display-bev`: show only the live BEV window
- `--input-width N`: override camera capture width
- `--input-height N`: override camera capture height
- `--input-fps N`: override camera capture FPS
- `--input-fourcc MJPG`: override camera capture FOURCC
- `--no-save-images`: JSONL/profile only
- `--profile`: print stage timing and end-to-end FPS

## Calibration Notes

Update `configs/thor_1080p.yaml` for your final camera mount:

- `camera.intrinsics.fx`
- `camera.intrinsics.fy`
- `camera.intrinsics.cx`
- `camera.intrinsics.cy`
- `camera.extrinsics.height_m`
- `camera.extrinsics.pitch_deg`
- `camera.extrinsics.yaw_deg`
- `camera.extrinsics.roll_deg`

Depth Anything metric gives useful meter-scale depth, but final BEV accuracy still depends on camera intrinsics, pitch, mounting height, and real-world validation.
