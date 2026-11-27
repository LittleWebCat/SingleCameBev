# Learned LiDAR vehicle detection in flat BEV

PointPillars predicts car and person centers, full length/width/height, heading and confidence from the LiDAR scan. The display keeps the black, top-down BEV layout: orange indicates cars and magenta indicates people, with oriented rectangles and predicted dimensions in meters. It does not switch to a perspective 3D view. These learned dimensions differ from the camera association view's bounds around only observed LiDAR returns.

## Run

After building the project and preparing the saved engine:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
bash tools/run_bev.sh --lidar-3d-boxes
```

This mode starts or reuses the Ouster driver, starts the native C++/TensorRT `lidar_vehicle_detector`, and opens [lidar_vehicle_boxes.rviz](../config/rviz/lidar_vehicle_boxes.rviz). It needs no camera stream, camera detector, RGB projection or ground/occupancy processor. It shows approximately 0–60 m forward at the default window size; resizing the window changes the visible extent. Close an existing RViz window before opening the layout, or pass `--no-rviz` to start only missing processors. Ctrl+C stops processes started by the wrapper.

The wrapper checks the installed detector, saved model class mapping and engine hash, and the host DDS receive-buffer prerequisite before launch. The [detector profile](../config/perception/lidar_vehicle_detector.yaml) enables `classes: [car, person]`, with a confidence threshold of 0.35, a 20 Hz publication cap, a 60 m forward limit and a 0.5 s freshness limit. A 20 Hz cap is a target; GPU inference time, end-to-end processing time, fresh output rate and source age must be measured separately.

For camera class/confidence associations on these learned footprints, use `bash tools/run_bev.sh --camera-lidar-boxes`. The [combined guide](camera_lidar_boxes.md) explains its matching rules; it keeps the flat layout and preserves the LiDAR model's geometry and score.

## Verification and capacity

The benchmark and live reports below used the earlier car-only output profile. The updated [native detector replay](../evaluation/lidar_vehicle_detector_replay_004.json) passed 14/14 phases, including trained pedestrian inference on a KITTI reference scan. Pedestrian accuracy on this Ouster rig remains unvalidated.

The [recorded Ouster benchmark](../evaluation/lidar_pointpillars_ouster_benchmark_001.json) produced eight car predictions and measured 5.14 ms median detector processing, 6.68 ms p95 over 30 timed runs after five warmups. This excludes ROS decoding, ROI point selection and publication. The [isolated ROS replay](../evaluation/lidar_vehicle_detector_replay_003.json) passed 13 stages using the real captured scan and trained engine, including oriented box dimensions, exact footprint/point provenance, matching source headers and invalid/stale clearing. The [CUDA memory check](../evaluation/lidar_pointpillars_cuda_memcheck_001.log) reported zero errors.

The [10-second live check](../evaluation/lidar_vehicle_live_001.json) observed 20.0 Hz for detections, flat markers and vehicle points, with nonempty predictions on all 201 received detection updates. Ten once-per-second diagnostic samples measured 5.83 ms median detector processing, 11.89 ms median complete node processing (14.05 ms p95), and 145 ms median scan-header age after publication. This is a short observation, not sustained/worst-case performance or physical detection accuracy. Header age includes scan acquisition and transport; it is distinct from node processing time.

The pinned model accepts at most 10,000 occupied pillars, each holding up to 32 points. The captured benchmark scan occupied 10,865 pillars; 865 were truncated. Diagnostics report input, accepted and truncated pillar counts. This capacity and KITTI-to-Ouster differences can affect detection quality; the checks establish software behavior, not physical size accuracy.

## Coordinates and outputs

The detector consumes `/points`, uses the actual driver transform to express raw `os_lidar` points in `os_sensor`, and preserves the scan header on its results. It excludes the same measured vehicle/mount envelope used by the ground processor: `x = [-2.131, 2.350] m`, `y = [-0.905, 0.885] m`, `z = [-1.800, 0.000] m`. Mirrors remain outside that mask. New results replace old footprints; empty or stale inputs clear the display.

| Topic | Message | Meaning |
| --- | --- | --- |
| `/camera_lidar_bev/lidar_detections_3d` | `vision_msgs/Detection3DArray` | Learned car/person centers, dimensions, yaw and confidence in `os_sensor` |
| `/camera_lidar_bev/lidar_vehicle_markers` | `visualization_msgs/MarkerArray` | Flat oriented footprints and predicted length/width/height labels |
| `/camera_lidar_bev/lidar_vehicle_points` | `sensor_msgs/PointCloud2` | Orange car and magenta person returns inside accepted predicted boxes |
| `/camera_lidar_bev/diagnostics` | `diagnostic_msgs/DiagnosticArray` | `lidar_vehicle_detector` freshness, processing timings and counters |

The [launch file](../src/bev_bringup/launch/lidar_vehicle_detector.launch.py) also supports `engine_file:=/path/to/compatible.plan` and `rviz:=true` after sourcing the project overlay. The convenience wrapper pins `models/lidar_objects_001/pointpillar_fp16.plan`, built for the local TensorRT/CUDA environment. [Model provenance](../models/lidar_objects_001/provenance.json) records the upstream source, model classes, export hashes, build configuration and remaining validation status. No model is compiled during each live frame.

## Accuracy and class coverage

The [NVIDIA CUDA-PointPillars model](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars) is trained on KITTI with `Car`, `Pedestrian` and `Cyclist` classes. The default node profile publishes `Car` and `Pedestrian` as `car` and `person`; `Cyclist` stays disabled. It does not provide trained bus, truck or motorcycle classes, even though the separate camera detector can label those categories. The existing `lidar_vehicle_*` topic names are retained for compatibility and now carry enabled car/person outputs.

Full dimensions and heading are **predictions**, not physical measurements of hidden vehicle surfaces. The saved model was trained with KITTI LiDAR, and this rig uses an Ouster sensor. The initial input uses Ouster `reflectivity / 255` as the model's intensity feature; equivalence to the training sensor is unvalidated. Sensor geometry, point density, occlusion and scene differences can affect confidence, size, heading and missed detections. Validate the predicted boxes against measured vehicles and representative Ouster recordings before relying on their dimensions.

Keep the rig stationary for this preview. Vehicle-frame calibration, scan deskewing, moving-scene timing, tracking and sustained performance qualification remain subsequent steps. The camera-assisted observed-extents mode remains available with `--vehicle-bev-boxes`.
