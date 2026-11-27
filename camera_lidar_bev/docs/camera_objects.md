# Camera object labels and observed 3D extents

The front camera now labels people and vehicles, and adds LiDAR-supported object extents to the stationary BEV. Camera and BEV overlays use the same colors: **magenta for people** and **orange for vehicles** (`car`, `motorcycle`, `bus`, `truck`). Bicycle labeling is available by adding `bicycle` to the detector's `classes` list and restarting the detector; its color is cyan. Keep the rig stationary; camera–LiDAR calibration B and sensor timing remain provisional.

## Run the live view

For camera labels on learned full-size LiDAR car/person boxes, use `bash tools/run_bev.sh --camera-lidar-boxes`. The [combined guide](camera_lidar_boxes.md) describes the flat BEV layout, separate camera/LiDAR scores and provisional projection association. The camera-assisted modes below enclose only observed LiDAR returns.

For flat vehicle boxes over the object points, run `bash tools/run_bev.sh --vehicle-bev-boxes`. The [vehicle box layout](../config/rviz/vehicle_bev_boxes.rviz) keeps the black, approximately 0–60 m forward view and shows orange rectangles and class/confidence labels for `car`, `motorcycle`, `bus` and `truck`. Person points remain magenta. Each rectangle is the axis-aligned XY footprint of accepted observed LiDAR returns, displayed at a fixed height of 0.15 m in `os_sensor`; it does not estimate complete vehicle shape or yaw. New results clear older rectangles, stale inputs clear the overlay, and markers expire after 0.5 seconds if updates stop.

For only vehicle and person points, run `bash tools/run_bev.sh --object-points-only`. The [points-only layout](../config/rviz/object_points_only.rviz) shows `/camera_lidar_bev/semantic_points` on a black background, with ground, occupancy, boxes and camera panels hidden. It retains the approximately 0–60 m forward view. The detector profile excludes bicycles; only accepted points associated with front-camera detections appear. Changes to detector classes take effect after restarting the detector.

For a single BEV window with ground/obstacle points and object boxes, run `bash tools/run_bev.sh --bev-only`. This enables camera labeling for association and uses [bev_only.rviz](../config/rviz/bev_only.rviz), with the camera pane and extra grid window hidden. The default view centers 30 m ahead and shows approximately 0–60 m forward at its default window size; window resizing changes the visible extent. Press Tab in RViz to show the configuration docks when needed. The geometric occupancy raster still covers 40 m forward; this view shows the point and object outputs.

After building the project, run:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
bash tools/run_bev.sh --camera-labels
```

The wrapper reuses existing sensor and geometry nodes, starts missing detector/fusion nodes, and opens the object RViz layout plus the occupancy window. Close an existing RViz window before opening this layout. `--no-rviz` starts missing processors while retaining the current viewer; `--no-occupancy-view` skips the separate occupancy window. Ctrl+C stops only processes started by that wrapper.

The camera pane shows class names and confidence on native 1024 × 768 images. BEV shows accepted object points, outline boxes and labels over the ground/obstacle/unknown geometry. A camera detection can appear without a BEV box when visible LiDAR evidence is sparse, ambiguous or lacks supported ground. The existing occupancy colors and cell values retain their geometric meaning.

## Detector and saved engine

`camera_object_detector` runs YOLO11n COCO inference through native C++ and TensorRT FP16, with a fixed 640 × 640 input and batch size 1. It reverses letterboxing when publishing image coordinates. The enabled classes are `person`, `car`, `motorcycle`, `bus` and `truck`, with confidence ≥ 0.35 and NMS IoU 0.50. These are image object labels; road and pavement segmentation remains a future milestone. See the [Ultralytics detection task](https://docs.ultralytics.com/tasks/detect/) for the image-box/class/confidence model contract.

The saved 7.8 MiB engine is loaded at startup from `models/camera_objects_001/yolo11n_fp16_raw.engine`. Its first GPU compilation took about eight minutes as a one-time preparation step. [Model provenance](../models/camera_objects_001/provenance.json) records the copied local weights, hashes, export settings, class mapping and installed TensorRT version; the source model and legacy pipeline were preserved. Startup checks the engine hash against that record.

The detector uses a latest-image mailbox and a 20 Hz publication cap. Fusion also caps publication at 20 Hz. Report camera-label rate, accepted BEV output rate, processing time and source age separately; configured limits and GPU inference time do not establish full-pipeline throughput.

The project DDS profile uses `AllowMulticast=spdp`: multicast discovery with unicast image/cloud data. This prevents bulk local traffic using multicast on the physical LiDAR Ethernet. See [Cyclone DDS multicast configuration](https://cyclonedds.io/docs/cyclonedds/latest/config/multicasting.html). Camera display images and the fusion cloud subscriptions use a latest-frame best-effort policy; detection messages and occupancy remain reliable.

To build another engine version using the existing project model environment:

```bash
/home/agxthor/SingleCameraBev/.venv/bin/python tools/build_camera_object_engine.py \
  --weights /home/agxthor/SingleCameraBev/yolo11n.pt \
  --output-directory models/camera_objects_002
```

The builder requires local weights and a new output directory. It preserves prior exports and records hashes and class names. Use the new raw engine through the `engine_file` launch argument after validation; the convenience wrapper currently pins version 001.

## Association and box meaning

`camera_object_bev` joins the full RGB precursor cloud and classified cloud with **exactly matching LiDAR headers**, then matches the nearest fresh camera detection image to each distinct fresh scan using bounded timestamp caches. Detection and CameraInfo headers must match exactly. The stationary profile allows at most 0.15 seconds between camera and LiDAR headers and expires inputs after 0.75 seconds. A new empty camera result clears older object evidence immediately.

Projection uses the native raw-image K/D model and `T_camera_from_os_sensor`. The full geometry supplies the nearest-optical-depth visibility buffer, including vehicle/mount returns removed from classification. Original `source_index` values join visible returns to supported obstacle labels. Foreground components use a 0.45 m Euclidean adjacency radius and require at least eight returns and six distinct native pixels. Sparse, oversized and nearby competing components are rejected.

The box encloses **observed LiDAR returns** in `os_sensor`, with axis-aligned dimensions and identity orientation. A visible vehicle face can therefore have almost zero depth. Class confidence comes from the camera; complete object size, heading, calibrated 3D confidence and tracking are not estimated. Semantic colors apply only to accepted visible returns. Geometry outside camera coverage and free/occupied/unknown occupancy are preserved. Each new result replaces its markers, and stale inputs clear the overlays.

## Outputs and checks

| Topic | Message | Meaning |
| --- | --- | --- |
| `/camera_lidar_bev/camera_labeled` | `sensor_msgs/Image` | Native camera image with class boxes |
| `/camera_lidar_bev/detections_2d` | `vision_msgs/Detection2DArray` | Camera class/confidence and native-image bounds |
| `/camera_lidar_bev/detections_3d` | `vision_msgs/Detection3DArray` | Observed extents stamped with the selected LiDAR scan |
| `/camera_lidar_bev/object_markers` | `visualization_msgs/MarkerArray` | BEV outlines and labels |
| `/camera_lidar_bev/vehicle_bev_markers` | `visualization_msgs/MarkerArray` | Flat observed vehicle footprints and class/confidence labels |
| `/camera_lidar_bev/semantic_points` | `sensor_msgs/PointCloud2` | Accepted returns with original IDs, COCO class IDs and camera confidence |
| `/camera_lidar_bev/diagnostics` | `diagnostic_msgs/DiagnosticArray` | Detector/fusion timings, source age, pairing delta and rejection counts |

The [vehicle footprint replay check](../evaluation/camera_object_bev_replay_005.json) passed all 14 phases, including exact rectangle bounds against accepted point coordinates, matching source headers, person/bicycle exclusion, and explicit clearing after a vehicle result becomes stale.

The three CTest targets cover RGB projection, object fusion and detector postprocessing. Fusion cases include masked-vehicle occlusion, foreground/background separation, competing components, source provenance and unsupported ground. Compact original source IDs use bounded direct indexing; wide sparse IDs retain the general lookup. Tests include nonzero ID offsets, UINT32 limits, missing IDs and duplicates. [The photographic detector replay check](../evaluation/camera_detector_replay_003.json) passed 10 source-header/image pairs and seven clear-output pairs, including BGR/RGB input, invalid layouts/frames, stale sources and recovery. [The isolated fusion check](../evaluation/camera_object_bev_replay_004.json) passed 12 scenarios covering delayed nearest-image matching, the empty-result barrier, cloud-header mismatch, occlusion and stale clearing. These software checks do not validate physical calibration or detection accuracy on the mounted camera.

The final [12-second live measurement](../evaluation/camera_objects_live_004.json) observed **20.0 FPS camera labels**, **19.8 Hz object/marker updates**, and **19.8 Hz occupancy**. Of 234 object updates, 32 contained supported observed extents; unsupported observations cleared the overlay. Sampled medians were **2.03 ms TensorRT inference**, **10.92 ms camera processing before publication**, **4.07 ms camera publication**, and **16.12 ms full object association**. Median camera-header age before publication was 43.8 ms. Timings are approximately one last-frame sample per 1 Hz diagnostic, not every inference, and the short check is not a sustained or worst-case performance qualification.

The [nonempty CPU benchmark](../evaluation/object_fusion_benchmark_002.json) used 129,793 recorded-background plus synthetic-object points, with three accepted candidates on all 30 runs. Fusion alone measured 6.59 ms median and 7.00 ms p95, a 54% improvement over the prior lookup. It excludes cloud decoding, ROS transport and GPU detection, and does not measure physical detection accuracy.

A separate [trained LiDAR car/person detector](lidar_vehicle_detection.md) is available through `bash tools/run_bev.sh --lidar-3d-boxes`. It predicts complete dimensions and yaw while keeping the flat BEV layout. Its KITTI model publishes `Car` and `Pedestrian` by default; `Cyclist` stays disabled. Ouster domain accuracy and predicted size remain unvalidated. This camera-assisted mode still reports observed extents. Validate physical alignment, vehicle-frame calibration and motion before tracking and moving BEV.
