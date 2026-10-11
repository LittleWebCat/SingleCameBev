# Single-Camera BEV Depth

Single-camera object detection, metric depth, and front BEV projection for NVIDIA Jetson AGX Thor. The current workspace is configured for a USB camera on `/dev/video0`, TensorRT YOLO detection, TensorRT Depth Anything V2 metric depth, and a `40m x 80m` BEV grid rendered as `400x800` pixels.

The pipeline can process videos, image folders, single images, or a live camera. It writes annotated camera frames, depth previews, BEV frames, and per-frame JSONL records.

The separate [ROS 2 camera + LiDAR BEV project](camera_lidar_bev/README.md) provides the current Ouster/front-camera preview: native C++/TensorRT detection, a flat 60 m BEV, camera-matched learned boxes and a labeled camera pane. Its README contains the launch command, display behavior, measured 20 Hz updates and remaining calibration checks.

## Camera + LiDAR results

![Camera and LiDAR recording replay with a saved-map BEV](camera_lidar_bev/docs/media/camera_bev_replay.gif)

Recorded camera/LiDAR replay with camera 2D labels, LiDAR 3D predictions and an 80 m flat BEV. The map overlay in this GIF uses the original recording's saved poses. These are prototype predictions.

The [visual results gallery](camera_lidar_bev/docs/results.md) also shows independent live localization, the GPS/LiDAR drive comparison and measured processing times. See the [camera + LiDAR project](camera_lidar_bev/README.md) for launch commands.

## Current Thor Setup

The main runtime config is `configs/thor_1080p.yaml`.

Current live-camera defaults:

- Camera node: `/dev/video0`
- Capture mode: `800x600 MJPG @ 30 FPS`
- Road-user detector: `models/yolo_vehicle_fp16.engine`
- Traffic-state detector: `models/traffic_light_state_fp32.engine`
- Detected classes: person, bicycle, car, motorcycle, bus, truck, stop sign, red, yellow, and green
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

## New Global-Shutter Camera BEV

`configs/new_camera_bev.yaml` combines the new `1920x1200 MJPG @ 90 FPS`
camera stream with:

- `models/yolo_vehicle_fp16.engine` for road-user/object detection
- `untested_model/best.engine` for red/yellow/green traffic-light state
- `models/depth_anything_v2_small_fp16.engine` for metric-depth estimation
- Front BEV projection with class, confidence, and estimated distance

Show the annotated camera, depth map, and BEV together without saving frames:

```bash
DISPLAY=:1 .venv/bin/python -m bev_depth.infer \
  --source 0 \
  --config configs/new_camera_bev.yaml \
  --save-dir runs/new_camera_bev_live \
  --no-save-images \
  --display \
  --profile
```

Show only the live BEV window:

```bash
DISPLAY=:1 .venv/bin/python -m bev_depth.infer \
  --source 0 \
  --config configs/new_camera_bev.yaml \
  --save-dir runs/new_camera_bev_live \
  --no-save-images \
  --display-bev \
  --profile
```

Press `Esc` to stop either live view. Remove `--no-save-images` to record the
annotated camera, depth, and BEV frames. The new camera configuration contains
estimated intrinsics and mounting values; calibrate the lens and measure the
final camera pose before relying on BEV positions or distances.

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

## Three-Camera Fused BEV

`configs/three_camera_bev.yaml` uses the verified physical layout:

- Left: `/dev/video0`, yaw `-90°`, object detection and depth
- Middle: `/dev/video4`, yaw `0°`, object detection, depth, and red/yellow/green classification
- Right: `/dev/video2`, yaw `+90°`, object detection and depth

All three cameras use `640x480 YUYV @ 20 FPS`. The device numbers identify the
current USB topology; verify them with `v4l2-ctl --list-devices` after moving a
camera to another hub port.

### Required local models

Model binaries are intentionally excluded from Git. A clean checkout needs
these files copied or generated locally before running the three-camera BEV:

```text
models/yolo_vehicle_fp16.engine
models/depth_anything_v2_small_fp16.engine
untested_model/best.engine
```

The last file is the Middle-camera red/yellow/green detector. Left and Right do
not invoke it.

### Three simultaneous UVC streams

These uncompressed camera streams require the Linux UVC bandwidth correction
quirk on the current host. Stop programs using the cameras, then reload the
driver:

```bash
sudo modprobe -r uvcvideo
sudo modprobe uvcvideo quirks=0x80
cat /sys/module/uvcvideo/parameters/quirks
```

The last command should print `128`. This setting is temporary and must be
reapplied after a reboot or UVC driver reload unless it is made persistent in a
system modprobe configuration. The correction applies to the verified YUYV
mode; it did not allow three simultaneous MJPG streams on the current USB 2.0
topology.

An experimental fixed-batch engine can be generated locally as
`models/yolo_vehicle_batch3_fp16.engine`. Because model binaries are not stored
in Git, build or copy that engine before running. To build it from local
`yolo11n.pt` weights:

```bash
.venv/bin/python scripts/export_yolo_engine.py \
  --weights yolo11n.pt --imgsz 640 --batch 3 --quantize 16 --no-simplify
mv yolo11n.engine models/yolo_vehicle_batch3_fp16.engine
```

Then run:

```bash
DISPLAY=:1 .venv/bin/python -m bev_depth.multicam_infer \
  --config configs/three_camera_bev.yaml \
  --batch-detector-model models/yolo_vehicle_batch3_fp16.engine \
  --save-dir runs/three_camera_bev_batch3 \
  --no-save-images --display-bev --profile
```

On the current hardware, batch-3 reduced vehicle inference from about 22 ms to
18 ms, but the measured 200-frame end-to-end rate was 7.4 FPS versus 8.5 FPS
sequentially because camera acquisition waited longer. Sequential mode remains
the default. Traffic-color inference is middle-camera-only in both modes.

Run the synchronized three-camera dashboard and fused BEV:

```bash
DISPLAY=:1 .venv/bin/python -m bev_depth.multicam_infer \
  --config configs/three_camera_bev.yaml \
  --save-dir runs/three_camera_bev_live \
  --no-save-images \
  --display \
  --profile
```

Use `--display-bev` instead of `--display` to show only the fused BEV. Press
`q` or `Esc` to stop. The current intrinsics, camera offsets, and yaw angles are
estimates; calibrate all three cameras before relying on fused positions or
depth values as ground truth.

Run only the live fused BEV while saving one semantic matrix per frame:

```bash
DISPLAY=:1 PYTHONPATH=. .venv/bin/python -m bev_depth.multicam_infer \
  --config configs/three_camera_bev.yaml \
  --save-dir runs/three_camera_bev_live \
  --no-save-images \
  --display-bev \
  --profile
```

`--no-save-images` disables annotated, depth, dashboard, and BEV image files;
it does not disable semantic matrices because `save_matrices: true` is set in
the three-camera config.

The fused map covers 40 m to either side, 80 m forward, and 40 m rearward
(80 m lateral by 120 m longitudinal). This retains the requested side/rear
range while keeping the longer forward range for the middle camera. The
directly rear-facing sector remains unseen because no rear camera is configured.

### Semantic matrix output

Every fused frame also writes a one-meter semantic matrix to `matrix/NNNNNN.npy`.
The physical grid is 80 cells wide by 120 cells longitudinal, represented in
NumPy as shape `(120, 80)`. Row 0 is 80 m forward; the last row is 40 m rear.
Column 0 is 40 m left; the last column is 40 m right. Empty cells are `0`.
Class IDs and axis metadata are stored in `matrix/metadata.json`.

| Value | Meaning |
| ---: | --- |
| `0` | empty |
| `1` | person |
| `2` | bicycle |
| `3` | car |
| `4` | motorcycle |
| `5` | bus |
| `6` | truck |
| `7` | stop sign |
| `8` | traffic light |
| `9` | red traffic light |
| `10` | yellow traffic light |
| `11` | green traffic light |
| `255` | unknown class |

If multiple detections occupy the same one-meter cell, the
highest-confidence detection supplies that cell's value. Load a saved matrix
with:

```python
import numpy as np

matrix = np.load("runs/three_camera_bev_live/matrix/000000.npy")
assert matrix.shape == (120, 80)
```

BEV marker legend:

- Stop sign: red octagon
- Traffic state: vertical black signal with only the detected red, yellow, or green lamp illuminated
- Other detected classes: orange circle

### GPS and IMU recording controls

The live BEV window includes one `RECORD GPS + IMU` control at the top. One
click starts both timestamped sensor logs; the next click stops both. If either
sensor is unavailable, neither recorder remains active.

The ports are detected automatically. The current hardware resolves to the
Septentrio `USB1` serial interface for GPS and `/dev/ttyUSB0` for the Xsens
MTi-30 once its kernel driver is loaded. Override detection when needed:

```bash
.venv/bin/python -m bev_depth.infer \
  --source 0 \
  --config configs/thor_1080p.yaml \
  --save-dir runs/live_camera_bev_display \
  --no-save-images \
  --display-bev \
  --gps-port /dev/ttyACM0 \
  --imu-port /dev/ttyUSB0 \
  --profile
```

Recordings are created only after their button is clicked:

```text
runs/live_camera_bev_display/sensors/gps_YYYYMMDD_HHMMSS_NNNNNNNNN.jsonl
runs/live_camera_bev_display/sensors/imu_YYYYMMDD_HHMMSS_NNNNNNNNN.jsonl
runs/live_camera_bev_display/sensors/session_YYYYMMDD_HHMMSS_NNNNNNNNN.json
```

GPS records contain the original NMEA sentence. For the connected Septentrio
receiver, the button temporarily enables `GGA+RMC` at 10 Hz on an otherwise
unused output stream and disables that stream when recording stops. IMU records
store lossless base64-encoded Xbus byte chunks for later decoding. Stale serial
input is cleared before the shared session begins. GPS and IMU records contain
the same `session_id`, `session_start_monotonic_ns`, and session-relative
`elapsed_ns`. While recording, camera frames in `detections.jsonl` contain the
matching `sensor_session_id` and `sensor_elapsed_ns`. The session manifest names
all three logs and records their common start, stop, and duration timestamps.

An orange `RETRY GPS + IMU` button means at least one port is absent or
inaccessible. Reconnect the device or correct its permissions/driver, then
click it again.

Serial access requires membership in the `dialout` group. After adding a user
to that group, log out and back in before launching the live BEV process.

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
models/traffic_light_state_fp32.engine
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

The Thor runtime presets use two detectors. The vehicle model detects people,
road vehicles, and stop signs; its generic COCO `traffic light` output is
filtered out. `models/traffic_light_state_fp32.engine` replaces that output with
`red`, `yellow`, or `green` detections. All detections use the same metric-depth
sampling and BEV projection, so their estimated depth is shown on annotated/BEV
frames and written to `detections.jsonl` as `depth_m`.

On a 300-frame unrestricted run of `video.mkv` on Thor, the two detector stages
averaged `13.90 ms`. The constant-depth pipeline achieved `47.9 FPS` end to end.
With TensorRT metric depth every other frame, detector time averaged `13.12 ms`,
depth time averaged `15.70 ms`, and the full pipeline achieved `36.2 FPS` without
image rendering or saving. After a 10-frame warmup, isolated application latency
averaged `5.03 ms` for the vehicle model, `3.74 ms` for the traffic-state model,
and `8.77 ms` combined (`10.35 ms` p95). USB-camera capture was unavailable for
this benchmark.

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
- `--gps-port PATH`: GPS port; default `auto`
- `--imu-port PATH`: IMU port; default `auto`
- `--gps-baud N`: GPS baud rate; default `115200`
- `--imu-baud N`: IMU baud rate; default `115200`
- `--no-save-images`: JSONL/profile only
- `--profile`: print stage timing and end-to-end FPS

## Calibration Notes

Update `configs/thor_1080p.yaml` for your final camera mount:

- `camera.intrinsics.fx`
- `camera.intrinsics.fy`
- `camera.intrinsics.cx`
- `camera.intrinsics.cy`
- `camera.intrinsics.calibration_width`
- `camera.intrinsics.calibration_height`
- `camera.extrinsics.height_m`
- `camera.extrinsics.pitch_deg`
- `camera.extrinsics.yaw_deg`
- `camera.extrinsics.roll_deg`

Depth Anything metric gives useful meter-scale depth, but final BEV accuracy
still depends on camera intrinsics, pitch, mounting height, and real-world
validation. Intrinsics are automatically scaled to the dimensions of each
decoded frame. The calibration width and height must describe the image
resolution at which `fx`, `fy`, `cx`, and `cy` were measured.
