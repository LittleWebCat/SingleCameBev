# Camera–LiDAR alignment: first stationary pair

This step checks that the same checkerboard can be observed by the camera and LiDAR. It saves raw observations for later alignment; it does not estimate a transform or publish a fused overlay.

The current session has moved to [imported native 1024 × 768 calibration and the user-selected B projection](imported_calibration.md). B uses the `os_sensor` source-frame hypothesis; accuracy remains provisional. The 640 × 480 capture commands below describe the earlier visibility attempt. To use this capture tool with the current camera, explicitly pass `--calibration-file config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml`; the tool does not rescale its 640 × 480 default calibration.

The first two capture windows were closed with zero saved pairs while troubleshooting board visibility. A live RViz 3D view now uses one-pixel white points on black, replacing overlapping 2 cm squares. Its fixed frame is the received `os_lidar`; the camera remains a separate Image display. [The visibility check](../evaluation/lidar_monochrome_preview_check.json) found complete scan coverage and no obvious XYZ units/layout fault. The board has not yet been identified in the cloud, so capture and fitting remain pending.

The user confirmed that both sensors are fixed in their operating positions and the printed checkerboard is flat on rigid backing. The target has 6 × 5 total squares, 5 × 4 inner corners, and 65 mm square edges. The provisional camera intrinsics are [loaded at 640 × 480](../evaluation/front_camera_002_review.md).

## First capture

Keep the vehicle and sensor mounts still. Place the board roughly 1–1.5 m in front of both sensors, with the printed face visible to the camera and LiDAR. Adjust its height until both sensors see it. Keep it separated from a wall or other large flat surface so the board returns can be identified.

Distance is a starting suggestion, not a requirement: sensors mounted far apart may need a more distant target to share visibility. The LiDAR's observed vertical coverage is approximately −21.45° to +21.11°. A board visible to a lower camera can lie below the scanner's rays. Start with the board center near scanner height, clear of the vehicle, and verify both views as its position changes. In a white geometry view, look for the board's flat surface and boundary; the printed checker pattern is not encoded by the constant point color. A board flush against a wall may merge geometrically with that wall.

For the live 3D inspection, source ROS and the overlay, then run:

```bash
ROS_DOMAIN_ID=1 DISPLAY=:1 \
  CYCLONEDDS_URI=file:///home/agxthor/SingleCameraBev/camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml \
  rviz2 -d /home/agxthor/SingleCameraBev/camera_lidar_bev/config/rviz/live_sensors.rviz
```

Drag in RViz to orbit the cloud and use the wheel to zoom. The viewing direction is a display choice, not a measured camera alignment. Close RViz before actual capture to avoid redundant large-cloud subscribers.

Open the first-pair capture window after sourcing ROS and this project's overlay:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=1 DISPLAY=:1 \
  CYCLONEDDS_URI=file:///home/agxthor/SingleCameraBev/camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml \
  /usr/bin/python3 tools/capture_extrinsics.py \
  --output data/calibration/camera_lidar_visibility_002 \
  --target-poses 1 --lidar-display white
```

This directory already exists once the window has been started. Choose a new name for any later session; the tool refuses existing paths.

The left pane is the raw camera image with detected corners. The right pane plots white LiDAR points on a black background, following the user's display preference. Its vertical extent fits all measured elevations with padding, so the points fill more of the pane. Click a point to read its distance. Optional `--lidar-display range` restores range colors. The two panes use different coordinate systems: the LiDAR view uses azimuth and elevation in its own frame, derived from XYZ rather than assuming scan columns are camera pixels or that staggered rows line up.

1. Keep all 20 corners visible and hold the board steady for at least two seconds. The window checks camera corner motion and displays **HOLD STILL**, **MOVE CLOSER**, or readiness feedback.
2. Locate the board's returns in the LiDAR pane. A small, deliberate board movement can help identify the corresponding patch; then stop moving. **Left-click** a board return to zoom and inspect its range. Click again in the zoomed view if needed. **Right-click** restores the full 360-degree view. White points show direction and do not encode depth by color; use the selected distance to help distinguish the board from background returns.
3. When the board is steady and the status shows **READY**, press **Space** or **S**. This confirms that the board and sensors stayed still and saves the displayed image/scan pair. Press **Q** or **Esc** after the first saved pair.

A click is a user-selected target hint, not automatic proof that the point lies on the board. Before using the data, review the saved cloud neighborhood and identify the actual board plane. A background wall can also form a good plane.

## Saved data and checks

The new directory contains a calibration YAML snapshot and `session.json`. The original `camera_lidar_visibility_001` session was closed with zero saved pairs and remains preserved. The white display uses a new session, `camera_lidar_visibility_002`. Each running first-pair session also has `lidar_metadata.json` for active Ouster metadata and `capture_setup.json` for the user's physical setup confirmations. Each observation contains raw `image.png`, untouched full `cloud.bin`, complete PointCloud2 layout in `cloud.json`, the image/CameraInfo/cloud headers, all 20 corner coordinates, and the selected LiDAR return's original point index and XYZ. A small `preview.npz` is for inspection; the full payload remains the source for geometry.

Capture requires matching image/CameraInfo headers and calibration values, fresh input, an image/scan header difference no greater than 150 ms, all detected corners, and a sufficiently large, steady board view. The GUI requires a current manually selected LiDAR return. The two-second camera stability and 140-pixel corner-span guides do not establish calibration accuracy or LiDAR-plane stability.

Host/header pairing does not establish exposure synchronization. Keeping the target and sensors stationary is necessary for these geometry observations. Extrinsics stay `unmeasured`; no identity transform is substituted for missing measurements.

## Current stream preflight

Before capture, the existing Ouster driver was found **unconfigured**. The sensor was reachable over Ethernet; configure/activate restored reception. The [preflight report](../evaluation/camera_lidar_capture_preflight.json) records fresh calibrated camera information and complete timing-bearing `original` clouds in `os_lidar`.

The current sensor metadata reports **1024 × 10**, with 128 × 1024 points, 48 bytes per point, and a 6 MiB payload. Twenty observed scan headers gave approximately 10 Hz. The ROS driver parameter still requests `2048x10`, so use the actual metadata and received layout for this session. The earlier 2048-column bring-up evidence remains a record of that earlier state. No scan-mode change was made for this capture.

The capture helper's 12 tests and preview/pair-selection helper's 11 tests passed. They cover raw-data preservation, byte order and row padding, metadata/header mismatches, stale/duplicate observations, selected-return indexing, spherical geometry, and camera stability. The actual received cloud decoded in about 2 ms and rendered in about 20 ms; the GUI refresh is capped at 5 Hz. These are short software measurements, not sustained pipeline performance.

## After reviewing the first pair

Once the board plane is identifiable in the first cloud, collect about 15 distinct stationary poses in a new directory, such as `camera_lidar_extrinsics_001`, with `--target-poses 15`. Tilt the board left/right and up/down by roughly 20–40 degrees, and vary position and distance while keeping both sensors' observations usable. Moving sideways with the same board orientation does not provide enough geometry by itself.

Later fitting will use the camera board planes and manually confirmed LiDAR board planes to estimate `T_camera_from_lidar`, mapping `p_camera = R * p_lidar + t`. Require well-constrained normals and translation, and reserve at least three distinct poses for independent projection/plane checks. The first saved pair alone cannot determine and validate this transform.

The source plan also references [Direct Visual LiDAR Calibration](https://github.com/koide3/direct_visual_lidar_calibration), a targetless alternative. It was not found installed locally. Its [Ouster collection protocol](https://koide3.github.io/direct_visual_lidar_calibration/collection/) differs from these fixed-sensor board snapshots; these datasets are not claimed to satisfy that protocol.
