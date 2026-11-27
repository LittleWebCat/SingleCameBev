# Camera road and lane masks

The combined display runs the pretrained [TwinLiteNet](https://github.com/chequanghuy/TwinLiteNet) camera model alongside the existing object detectors. It predicts road-area and lane-marking masks for the camera image. The BEV shows LiDAR points and object boxes. Road-to-BEV geometry, ground fitting, contour projection and curb candidates have been removed from this pipeline.

## Replay and live display

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
bash tools/replay_combined_bev.sh
```

The default outdoor recording plays at half speed and loops. Closing the window with `q` or Escape stops playback and the owned processors. For physical sensors, use `bash tools/run_bev.sh --combined-view` instead.

The new layers are:

| Color | Camera image |
| --- | --- |
| Green | Predicted road-area mask |
| Magenta | Lane-head predictions |
| Blue | Outer semantic road-mask edge |

Keys `6`, `7` and `8` toggle road masks, lane masks and semantic edges on the camera image. Keys `1`–`5` control the object/point layers. To omit segmentation processing entirely, use `bash tools/replay_combined_bev.sh --no-road-lanes`; the standalone launch also accepts `road_lanes:=false`.

On an unmarked street, the lane head can follow curb edges or other line-like features. Its pixels are predictions, not confirmation of painted lanes. Blue edges describe the semantic road-mask boundary in image pixels; they do not supply a measured road edge in meters.

## Model and source-headed outputs

The local TensorRT FP16 engine is `models/road_lanes_001/twinlitenet_fp16.engine`. [Provenance](../models/road_lanes_001/provenance.json) records the MIT upstream source, pinned commit, official BDD100K weights, hashes and tensor contract. The model has 439,633 parameters. Its official preprocessing directly resizes the raw camera image to RGB 640 × 360 and divides by 255, with no letterboxing, mean/std normalization or undistortion. Argmax foreground masks are resized back to the native image with nearest-neighbor interpolation.

| Topic | Message | Meaning |
| --- | --- | --- |
| `/camera_lidar_bev/road_mask` | `sensor_msgs/Image`, `mono8` | Native road foreground, 255; background, 0 |
| `/camera_lidar_bev/lane_mask` | `sensor_msgs/Image`, `mono8` | Native lane foreground, 255; background, 0 |
| `/camera_lidar_bev/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Latest segmentation processing time and counts |
| `/camera_lidar_bev/combined_view` | `sensor_msgs/Image`, `bgr8` | Combined image, BEV and object table |

Both masks retain the exact camera source header. The viewer attaches them only to that image. Missing or expired inputs clear their layers. The model loads once, uses reusable TensorRT/CUDA buffers, and processes at most 10 frames per wall-clock second. This cap does not assert measured full-view throughput.

## Removed BEV geometry

The earlier `tools/road_lane_geometry.py` experiment and its saved results remain available for reference. The combined renderer and preview helper no longer import or call it, and it is no longer installed with the display scripts. This removes the CPU road/BEV geometry stage previously observed at approximately 81 ms in a replay snapshot. That timing does not establish the updated pipeline's sustained FPS.

## Saved preview

The updated layout is saved in `data/combined_bev_no_road_geometry_001/combined.png`, rendered from the existing predictions and masks with no new inference. Its `result.json` reports `road_geometry_enabled: false`. The earlier result in `data/road_lane_preview_001/combined.png` retains the historical BEV geometry overlays. New previews display road/lane masks only on the camera image.

To generate another result from the ten previously extracted samples, choose an unused output directory:

```bash
../.venv/bin/python tools/preview_road_lanes.py --sample-id 007 \
  --output-dir data/NEW_road_lane_preview
```

`tools/build_road_lane_engine.py` reproduces the pinned local model export/build without changing the Python environment. The current engine was built once and exercised on a saved frame. One engine execution took about 7 ms; this excludes preprocessing, mask transfer, ROS and display, and is not an accuracy or sustained-throughput result.
