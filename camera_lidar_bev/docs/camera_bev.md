# Live camera-colored BEV

The front camera supplies RGB to measured LiDAR returns. The C++ `camera_rgb_cloud` node publishes `/camera_lidar_bev/points_rgb` in `os_sensor`; RViz displays that cloud from above beside the raw camera image. Camera-visible points use image colors. Other measured points remain white, including points outside the image and points hidden by a closer return at the same camera pixel.

This is a diagnostic using [the user-selected calibration B](imported_calibration.md#working-candidate-b). Alignment accuracy and exposure synchronization remain provisional. The view shows individual returns rather than a filled road image, semantic labels, free space, or a rolling occupancy map.

## Open the live view

Build the visualization and launch packages once after adding this component:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
colcon build --packages-select bev_visualization bev_bringup --symlink-install \
  --cmake-args -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPython3_EXECUTABLE=/usr/bin/python3
```

Keep the existing native **1024 × 768** camera bridge and single Ouster driver running. Source the environments and launch the color node plus viewer:

After reboot, restore the temporary network settings before using the large-cloud DDS profile:

```bash
sudo sysctl -w net.core.rmem_max=16777216 net.core.rmem_default=4194304 \
  net.ipv4.ipfrag_time=3 net.ipv4.ipfrag_high_thresh=134217728
```

The profile requires a 16 MiB receive buffer. The default Linux limit is too small and makes both the color node and RViz fail during ROS initialization. These settings reset on reboot. The camera and LiDAR processes also need to be started again after reboot; use [the native camera command](imported_calibration.md#native-resolution-camera-application) and [the LiDAR launcher](lidar_bringup.md#inspect-the-active-driver-first).

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=1 DISPLAY=:1 \
  CYCLONEDDS_URI=file:///home/agxthor/SingleCameraBev/camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml \
  ros2 launch bev_bringup camera_bev.launch.py
```

The launch uses the preserved native camera candidate and `T_B_os_sensor.candidate.yaml` from the installed package. `intrinsics_file` and `extrinsics_file` can select reviewed alternative files. Set `rviz:=false` to run only the color node. This launch consumes the already-running sensor topics.

In RViz, scroll to zoom and left-drag to rotate. Forward is initially up. The camera's field of view covers only part of the LiDAR scan, so much of the cloud can remain white. Geometry retains its measured height when viewed from above.

## Association and freshness

For B, the external matrix maps `os_sensor` into the camera optical frame. The node reads the actual driver `os_sensor` parent / `os_lidar` child transform and checks it against the candidate before using it. It does not publish an extrinsic TF. The raw camera projection applies native K/D and rejects invalid returns, nonpositive camera depth, and rays beyond the distortion model's first monotonic radial branch. Each projected camera pixel colors only its nearest-depth LiDAR return.

An image must have an exact-header CameraInfo with matching native dimensions and full K/D/R/P, compatible ROI/binning, and the expected optical frame. Image/cloud headers must be within 150 ms and receipts must be fresh. This is a stationary-scene pairing heuristic, without exposure synchronization. Old inputs are not repeatedly published as fresh colors. RViz shows only the newest scan (`Decay Time: 0`), and the node clears its output when inputs become stale.

Defaults and topics are in [camera_rgb_cloud.yaml](../config/perception/camera_rgb_cloud.yaml). The user selected **20 Hz with fewer horizontal points per scan**. The LiDAR now uses `1024x20`, and the color publication cap is 20 Hz. RViz renders at up to 30 FPS. Restart the launch after changing these settings; the node creates its processing timer at startup.

The earlier `2048x10` mode produced 262,144 returns per scan. The current `1024x20` mode produces 131,072 returns per scan, with the same 128 vertical channels. The source point-cloud layout is read from each message. Native camera K/D and the driver frame conversion remain unchanged.

## Validation

The pure projection/BEV helper provides an independent reference for the C++ output. The bounded replay verifier uses a recorded native image/cloud pair on private topics in a separate ROS domain. It checks original point indices, transformed XYZ, exact pixel RGB, hidden-point handling, rejection of mismatched CameraInfo, and input expiration. Replay agreement checks implementation; it does not establish physical calibration accuracy.

```bash
/usr/bin/python3 tools/check_camera_bev.py \
  --output evaluation/camera_bev_replay.json
```

Use a new output path and an unused ROS domain. Live observations and replay results are saved as evaluation evidence after running these checks.

The C++ projection regressions and all eight Python reference tests passed. [The isolated replay check](../evaluation/camera_bev_replay_001.json) verified eight complete output clouds against all 260,248 valid returns in the 262,144-return fixture, including 18,260 camera-colored returns per scan. Both mismatched-CameraInfo phases were received and rejected; valid inputs recovered and stale output cleared.

[The bounded live check](../evaluation/camera_bev_live_001.json) received 11 fresh clouds over four seconds, about 2.94 Hz. The last had 260,063 valid returns and 18,211 camera-colored returns, with verified white fallback and field layout. RViz displayed both camera and colored points with status OK; [the window snapshot](../evaluation/camera_bev_live_001.png) records that observation. These short checks do not establish sustained throughput or physical calibration accuracy.

After the user requested faster refresh, [the 20 Hz LiDAR check](../evaluation/lidar_20hz_001.json) received 81 distinct scan headers over four seconds, measuring 19.999 Hz. [The 20 Hz colored-view check](../evaluation/camera_bev_live_20hz_001.json) received 99 distinct output clouds over five seconds, measuring 20.000 Hz; node diagnostics independently reported 20.000 Hz input and output. The first colored scan had 130,052 valid returns and 9,111 camera-colored returns. Driver geometry and original per-point timing were preserved; the measured `t` range was 0–49,953,984 ns. These are bounded transport/visualization measurements.

RViz subsequently exited with `std::system_error: Invalid argument`; the sensor and colorizer continued running. The viewer was reopened separately and displayed status OK with latest-scan rendering; [the recovered window](../evaluation/camera_bev_live_20hz_001.png) records the 20 Hz setup. The cause of that viewer exit is unresolved. To reopen only the viewer while the colorizer is already running, use the same ROS/display environment and `rviz2 -d config/rviz/camera_bev.rviz`, avoiding another colorizer instance.
