# First steps

For a five-minute camera/LiDAR test session, follow [camera_lidar_recording.md](camera_lidar_recording.md) and run `bash tools/collect_camera_lidar.sh --duration 300 --scene 'multi_scene_test'` from the project directory after positioning the stationary rig. This starts only the required sensors and records their data; detectors and RViz are not needed. The guide explains storage, scene notes and bag validation.

The project now provides stationary camera-colored LiDAR, supported ground/obstacle classification, an observed free/occupied/unknown grid and camera object labels with LiDAR-supported observed extents. The steps below retain the input, recording and calibration workflow; the geometry live view starts with:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
bash tools/run_bev.sh
```

Keep the rig stationary. This command reuses existing project nodes, starts missing ones, and opens RViz plus a separate occupancy window. Green indicates observed ground, red obstacles, and gray unknown evidence. See [ground_bev.md](ground_bev.md) for prerequisites, viewer flags and interpretation. A measured body mask excludes the vehicle/mount below the LiDAR within `x = [-2.131, 2.350] m`, `y = [-0.905, 0.885] m`, `z = [-1.800, 0.000] m` in `os_sensor`; mirrors remain unmasked. Vehicle-frame calibration, known-distance physical checks and motion validation remain pending.

For camera labels and object boxes in BEV, close the existing RViz window and run:

```bash
bash tools/run_bev.sh --camera-labels
```

Native C++/TensorRT inference labels people in magenta and vehicles in orange. Bicycle labeling is available in the detector configuration. BEV boxes enclose supported visible LiDAR returns; full object dimensions, heading and tracking are not inferred by this camera association path. The saved YOLO11n FP16 engine is loaded at startup; its initial compilation is a one-time step. See [camera_objects.md](camera_objects.md) for model provenance, topics, timestamp matching and software checks. Keep the rig stationary with the provisional calibration.

To use trained LiDAR car/person detection and predicted dimensions while keeping a flat BEV view:

```bash
bash tools/run_bev.sh --lidar-3d-boxes
```

This mode runs without the camera processors and displays orange car and magenta person footprints with predicted length/width/height labels. The KITTI-trained model publishes `Car` and `Pedestrian`; cyclists stay disabled. Dimensions and accuracy on this Ouster rig remain unvalidated. See [lidar_vehicle_detection.md](lidar_vehicle_detection.md) for the saved engine, inputs and interpretation.

To add camera class/confidence associations to the learned LiDAR boxes, retain the flat view with:

```bash
bash tools/run_bev.sh --camera-lidar-boxes
```

The default display shows native car/person points as context and only camera-matched boxes, with the labeled camera image beside the flat BEV. Filtered camera-matched points remain available as an optional display. Matching does not alter LiDAR dimensions, yaw or confidence, and raw LiDAR predictions continue independently. This mode needs both saved engines and the provisional camera calibration. See [camera_lidar_boxes.md](camera_lidar_boxes.md) for timestamp matching and association limits.

For genuine camera-only 3D predictions, use `bash tools/run_bev.sh --camera-3d-boxes`. This separate mode runs the SMOKE model using the calibrated camera image and CameraInfo, with no LiDAR input. It shows the labeled image beside flat boxes in `camera_front_bev`, whose origin is the camera. The saved engine is required, and predicted depth and dimensions remain unvalidated on this rig; see [camera_3d_detection.md](camera_3d_detection.md).

## 1. Build and check the foundation

In a new terminal, use the installed ROS 2 Jazzy environment:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 tools/check_environment.py
colcon build --base-paths src --cmake-args \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release
colcon test --event-handlers console_direct+
colcon test-result --verbose
source install/setup.bash
```

The environment report identifies missing tooling without installing or upgrading packages. Use the installed JetPack-matched CUDA/TensorRT when inference is introduced. These foundation packages do not require neural-network engines.

Optional integration checks exercise the installed probe and camera bridge with synthetic inputs on localhost:

```bash
/usr/bin/python3 tools/smoke_test_sensor_probe.py
/usr/bin/python3 tools/smoke_test_camera_bridge.py
```

The probe check uses ROS domain 187 to check missing/stale streams, point-timing fields, malformed payloads, and invalid/future timestamps. The camera check validates synthetic publication and error handling. These checks start no hardware drivers and do not measure live-camera performance. Calibration readiness remains pending even with healthy synthetic inputs.

The grid library can also be built and tested without ROS:

```bash
cmake -S src/bev_core -B /tmp/camera_lidar_bev-core-build \
  -DBUILD_TESTING=ON -DBEV_CORE_USE_AMENT=OFF
cmake --build /tmp/camera_lidar_bev-core-build
ctest --test-dir /tmp/camera_lidar_bev-core-build --output-on-failure
```

## 2. Start the front camera and identify sensor topics

In a terminal with this project's overlay sourced, start the C++ bridge:

```bash
ros2 launch bev_bringup front_camera.launch.py
```

The default is the parent's front-mid device at `/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.1.2.2:1.0-video-index0`, requesting 640 × 480 MJPG at 30 frames/s. Confirm that this port corresponds to the intended front-facing camera; those settings are requests, not measured rates. To select a different device:

```bash
ros2 launch bev_bringup front_camera.launch.py device:=/dev/video0
```

It publishes `/camera/front/image_raw` (`bgr8`) and `/camera/front/camera_info` with matching headers. Stamps use host ROS time immediately after capture returns; hardware exposure time and camera–LiDAR synchronization remain unknown. Without a measured calibration YAML, `CameraInfo.k[0]` is zero and the bridge reports uncalibrated images. See [camera_bridge.md](camera_bridge.md) for calibration and failure diagnostics.

Use the project's explicit LiDAR launcher after sourcing the existing driver workspace:

```bash
source /home/agxthor/sensors_ws/install/setup.bash
/usr/bin/python3 tools/start_lidar.py
```

It uses this project's `original` point format, preserves per-point timing, and waits for configure/activate to complete. It refuses a duplicate driver. The original sensor workspace files remain separate. See [lidar_bringup.md](lidar_bringup.md) for temporary network settings, input inspection, bounded recording, and replay. Start motion hardware using its validated driver procedure.

```bash
source /home/agxthor/sensors_ws/install/setup.bash
ros2 topic list -t
ros2 topic info /points --verbose
ros2 topic hz /points
```

Stop the rate command with Ctrl+C. Inspect each enabled camera and IMU topic in the same way. Confirm frame IDs and timestamp conventions from actual message headers, and inspect cloud fields:

```bash
ros2 topic echo /points --once --field header
ros2 topic echo /points --once --field fields
```

For the front-camera pair:

```bash
ros2 topic hz /camera/front/image_raw
ros2 topic echo /camera/front/camera_info --once
```

Record observed rates under capture and bag-recording load; the bridge's nominal FPS setting is not sufficient evidence.

The existing `start_ouster.sh` starts the driver in the root namespace, producing `/points` and the Ouster's own `/imu`. The driver's standard `driver.launch.py` instead defaults to the `ouster` namespace, producing `/ouster/points`. Verify which procedure is active rather than starting an additional driver.

The earlier inspected parent Ouster configuration selected `2048x10`, `TIME_FROM_ROS_TIME`, and `xyzir`. That was a 10 Hz scan-mode setting, not a measured reception rate. `xyzir` omits the point time needed for deskewing. The saved `lidar_frame_id`/`imu_frame_id` keys are legacy names; they do not establish effective frames for the installed driver. The current project profile uses the user-selected `1024x20` mode and `original` format; verify active parameters and message headers after each restart.

The project [ouster_baseline.yaml](../config/sensors/ouster_baseline.yaml) uses supported frame keys and the timing-bearing `original` cloud. Its `t` field was observed as UINT32 with an approximately 100 ms span in 10 Hz scans. Driver source defines this field as relative nanoseconds. Shared-clock alignment, motion, and mounting calibration still require validation; field presence alone does not establish deskew readiness.

The parent camera program exposes OpenCV frames. The new bridge supplies the ROS Image/CameraInfo pair for the selected front camera. It does not supply a calibrated camera–`base_link` transform. GPS speed and an IMU alone do not supply the complete odometry trajectory required for deskewing; confirm wheel or LiDAR odometry and its frames.

## 3. Run the input probe

In a terminal with this project's overlay sourced:

```bash
ros2 launch bev_bringup sensor_probe.launch.py
```

Defaults are candidates: `/points`, `/camera/front/image_raw`, `/camera/front/camera_info`, `/sensing/imu/imu_data`, and `/lio_sam/mapping/odometry`. The custom `mila_xsens_driver` launch remaps the Xsens IMU to `/sensing/imu/imu_data`; the upstream Xsens launch may publish `/imu/data` instead. Override topics using the inventory found above:

```bash
ros2 launch bev_bringup sensor_probe.launch.py \
  lidar_topic:=/points \
  camera_topic:=/camera/front/image_raw \
  camera_info_topic:=/camera/front/camera_info \
  imu_topic:=/sensing/imu/imu_data \
  odom_topic:=/lio_sam/mapping/odometry
```

Read the output in another sourced terminal:

```bash
ros2 topic echo /camera_lidar_bev/diagnostics
```

The node is `/camera_lidar_bev/sensor_probe`; its executable is `bev_sensor_bridge/sensor_probe`. The launch also accepts `params_file`, `use_sim_time`, and `lidar_reliability` (default `reliable`). Use `imu_topic:=/imu` to inspect the Ouster's IMU. A missing-stream or stale-stream diagnostic identifies remaining bring-up work. The camera bridge publishes its own `camera_lidar_bev/front_camera_bridge` status on the same diagnostic topic; distinguish status names when both nodes are running. Header ages are meaningful only after clock domains are understood.

Earlier `2048x10` `original` clouds were 12 MiB each. Those recording checks fell to about 7 Hz with the extra point-cloud probe running; the recorder-only check captured approximately 10 Hz. Current `1024x20` clouds are 6 MiB each, with 131,072 returns. During recording, close redundant LiDAR subscribers and inspect saved counts. Keep the camera bridge and its viewer running. Historical evidence is in [lidar_bringup_validation.md](../evaluation/lidar_bringup_validation.md), and the current mode check is in [lidar_20hz_001.json](../evaluation/lidar_20hz_001.json).

## 4. Make the first recording

Once the desired streams are present, create a short static session containing a ground surface, objects at several distances, and an occluded area. Add a moving session once motion timing is available. Save bags outside the repository and complete [the recording report](../evaluation/first_recording.md).

For the verified default topic names, a recording command is:

```bash
mkdir -p /home/agxthor/bev_recordings
ros2 bag record -o /home/agxthor/bev_recordings/session_001 \
  /points \
  /camera/front/image_raw /camera/front/camera_info \
  /sensing/imu/imu_data /lio_sam/mapping/odometry \
  /tf /tf_static /camera_lidar_bev/diagnostics
```

Replace absent or differently named topics before recording. Stop with Ctrl+C, then inspect counts, duration, and types:

```bash
ros2 bag info /home/agxthor/bev_recordings/session_001
```

Preserve Ouster metadata, active parameter files, sensor serial numbers, and software versions alongside the bag. Check rates and recorder warnings under recording load; a bag containing only LiDAR does not satisfy the full camera + motion recording gate.

## 5. Verify replay

Replay in a separate ROS domain so live publishers do not mix with recorded messages. In both replay terminals, source ROS and this project's overlay, then set `ROS_DOMAIN_ID=42` (or another unused domain).

Terminal A:

```bash
export ROS_DOMAIN_ID=42
ros2 launch bev_bringup sensor_probe.launch.py use_sim_time:=true
```

Terminal B:

```bash
export ROS_DOMAIN_ID=42
ros2 bag play /home/agxthor/bev_recordings/session_001 --clock \
  --exclude-topics /camera_lidar_bev/diagnostics
```

Use the same topic overrides as the recording. Excluding recorded diagnostics lets the new probe report replay behavior without mixing its output with old reports. Check diagnostics, message counts, timestamp ordering, gaps, and replay behavior. Recording and replay provide evidence for timing analysis; they do not themselves synchronize sensors.

## 6. Check calibration and stationary LiDAR BEV

The current session uses the user's [imported native 1024 × 768 calibration](imported_calibration.md). Live camera matrices were verified. After the two-pane comparison, the user selected B (`os_sensor`) as the working source-frame hypothesis. A single B overlay supports further board-edge checks. The [camera-colored BEV diagnostic](camera_bev.md) now displays native image RGB on visible measured LiDAR returns through a C++ node, with replay and live transport checks. Accuracy remains provisional and no imported camera–LiDAR TF has been published. The project checkerboard evidence below describes the earlier 640 × 480 work.

The user's target is 6 × 5 total squares, 65 mm per square, giving 5 × 4 inner corners. Detection of all 20 corners was verified at 640 × 480. The first 25-view dataset was rejected. The second set of 25 closer views produced a candidate with 0.611-pixel held-out RMS and at most 2.09% training subset focal change. Following the user's straight-edge feedback, that candidate was applied provisionally; 59 matched image/CameraInfo pairs verified its matrices and dimensions. Numerical intrinsic accuracy remains unvalidated. See [the second review](../evaluation/front_camera_002_review.md) for that earlier application evidence. [Stationary board-pair capture](camera_lidar_capture.md) remains available if the imported extrinsics need to be remeasured; it requires a camera calibration matching the current image dimensions.

Fill [calibration.example.yaml](../config/calibration/calibration.example.yaml) with measured data. All calibration values begin unknown and `calibrated: false`. The probe does not load this file; it is a schema example for subsequent tools. Confirm intrinsics/distortion at the operating resolution, camera–LiDAR and sensor–`base_link` transforms, and time alignment on independent static and moving recordings.

The front-camera bridge accepts a separate standard ROS camera-calibration YAML through `calibration_file`; see [camera_bridge.md](camera_bridge.md). It does not load the multi-sensor example above. Use a calibration measured at the actual capture dimensions; the bridge does not scale intrinsics implicitly.

The existing camera config's roughly 65° FOV and the source plan's 43° FOV are estimates from different assumptions. Neither replaces calibration.

Stationary self-filtering, supported local ground, obstacle extraction and conservative free/occupied/unknown visibility are implemented in `bev_geometry`. The current map has 200 forward × 150 lateral cells at 0.20 m in `os_sensor`; no `base_link` transform is inferred from the body mask. [The geometry guide](ground_bev.md) describes the core regressions, isolated ROS checks and saved baseline. Verify the exclusion boundary and a known-distance physical obstacle before treating displayed geometry as measured accuracy.

Camera object labels, visible LiDAR associations and a pretrained LiDAR car/person detector are available as stationary previews. Next validate physical alignment, Ouster-domain detection and size accuracy, the sensor-to-vehicle transform, shared timing and an odometry trajectory before motion deskew, rolling-map motion and evidence expiration. Road segmentation and dynamic tracking remain future steps. The current LiDAR mode is `1024x20`; measure input reception, geometry, camera-label and object-output rates separately, together with source age.
