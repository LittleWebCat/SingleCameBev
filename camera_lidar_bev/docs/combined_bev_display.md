# Combined camera image and BEV

The combined display keeps the camera image on the left and a flat 80 m BEV on the right, with lateral coverage of ±30 m and equal meter scaling on both axes. It draws the existing LiDAR points and projected LiDAR cuboids with YOLO camera 2D boxes. TwinLiteNet adds road/lane masks and semantic road-mask edges on the camera image; see [the road/lane guide](road_lane_detection.md). Road-to-BEV ground fitting, contour projection and curb processing have been removed. The BEV contains LiDAR points and 3D footprints, predicted dimensions, positions, yaw, confidence, and camera/LiDAR association information. Raw LiDAR predictions remain visible even when strict association produces no matched boxes.

Raw points can fill the full 80 m view. PointPillars has a trained input range ending at 69.12 m forward; the combined launch sets its box-center filter to the smaller of the display range and 69.12 m. This exposes available model outputs beyond the former 60 m filter without changing the model's point-input bounds. The standalone LiDAR/RViz modes retain their existing 60 m profile.

Camera 2D boxes alone have no metric position. Their label and confidence appear on a BEV object only when associated with an existing LiDAR 3D box. The legend identifies each source. Distances and dimensions are LiDAR model predictions, and a camera match is association evidence rather than verified object identity. Camera-only SMOKE 3D detection is archived and no longer runs, subscribes or renders in this display.

## Live display

The large-cloud DDS profile requires a host UDP receive limit of at least 16 MiB. Set up the host once before starting the display:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
sudo bash tools/setup_dds_transport.sh
```

The helper writes `/etc/sysctl.d/90-camera-lidar-bev.conf` and applies those settings immediately, so they also survive reboot. It retains existing larger receive-buffer and fragment-memory limits. `sudo` may ask for your password in the terminal; the BEV launcher does not invoke sudo itself.

After building the workspace, use the existing native sensor entry point:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
bash tools/run_bev.sh --combined-view
```

This starts or reuses the front camera and Ouster driver, then starts PointPillars, YOLO, TwinLiteNet and association nodes plus the combined display. It keeps the flat layout in one window. Press `q` or Escape to close the display; Ctrl+C stops processes owned by the wrapper. The view defaults to a 10 Hz ROS-clock timer with at most 30,000 displayed cloud points. This targets 10 FPS with the live clock or approximately 5 wall-clock FPS during half-speed replay; actual throughput also depends on data availability and rendering. The former road/BEV geometry stage no longer runs.

For a temporary fix that lasts only until reboot, use this command instead of the setup helper, then rerun the launcher:

```bash
sudo sysctl -w net.core.rmem_max=16777216 net.core.rmem_default=4194304 \
  net.ipv4.ipfrag_time=3 net.ipv4.ipfrag_high_thresh=134217728
```

If sensors already run and detector startup is managed separately, source ROS and launch the inference/display group:

```bash
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1
export DISPLAY=:1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$PWD/config/transport/cyclonedds_large_clouds.xml"
ros2 launch bev_bringup combined_bev_display.launch.py
```

The launch starts processors and the viewer; it expects `/points`, `/camera/front/image_raw`, `/camera/front/camera_info` and the Ouster frame transform to be available. If all processors already run, add `detectors:=false` to open only the display. Do not start a second copy of the detectors in the same ROS domain. Keys `6`–`8` toggle road, lane and semantic-edge overlays on the camera image; `--no-road-lanes` on the replay helper skips the segmentation processor.

## Recorded data

To replay the collected outdoor recording in the combined image/BEV window:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
bash tools/replay_combined_bev.sh
```

The helper starts the existing inference/display group and recorded-clock player together. It uses ROS domain **89**, plays at **half speed**, and loops the recording. The camera and LiDAR can remain disconnected. It starts no physical sensor drivers and creates no recording. The persistent DDS host setup above is also required for replay.

To view the new dense 10 Hz recording at its recorded speed with camera 2D detections, LiDAR points and LiDAR 3D boxes:

```bash
bash tools/replay_combined_bev.sh bags/camera_lidar_20261010T195520_925217Z \
  --rate 1 --no-road-lanes --max-points 300000 --range-m 80
```

This omits road/lane inference and raises the display point limit to retain the front-area points of each dense scan. The flat BEV covers 80 m forward; full-view FPS depends on processing and rendering. The point limit changes visualization only, not LiDAR detector input or model range. `--range-m 60` restores the smaller view.

Press **Ctrl+C in the replay terminal**, or close the replay window with `q` or Escape, to stop playback and its detectors. A second replay helper in the same domain is rejected rather than starting duplicate processors. Stop any separately started processors or player in domain 89 before running the helper. When available, the helper uses the calibration snapshots stored beside the selected recording.

Select another recording, change speed, or jump to a later scene:

```bash
bash tools/replay_combined_bev.sh bags/camera_lidar_20261009T192214_724569Z \
  --rate 1 --start-offset 180 --no-loop
```

`--start-offset` is measured in recorded seconds. `--no-loop` plays once and stops the owned display/processors at the end. Use `--domain 90` to select another isolated ROS domain. `--help` lists the options without starting playback or inference.

For manual two-terminal control, use the same sourced environment and isolated domain in both terminals:

```bash
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=89
export DISPLAY=:1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$PWD/config/transport/cyclonedds_large_clouds.xml"
```

Start the inference/display group with the recorded clock in the first terminal:

```bash
ros2 launch bev_bringup combined_bev_display.launch.py use_sim_time:=true
```

Then play the camera/LiDAR bag in the second terminal:

```bash
ros2 bag play bags/camera_lidar_20261009T192214_724569Z --clock --rate 0.5 --loop
```

The native detectors process replayed sensor data and the viewer uses source headers to select nearby inputs. The adapter uses the paired recorded CameraInfo, transforms cloud points with the recorded sensor transform, and clears stale or missing layers rather than keeping old detections on new images. Rendering rate is capped independently from replay and inference. No new recording or model build is required.

## Existing prediction preview

To inspect the ten cached comparison samples without running any models:

```bash
/usr/bin/python3 tools/combined_bev_display.py --sample-id 002 \
  --output data/NEW_combined_bev_preview.png
/usr/bin/python3 tools/combined_bev_display.py --display
```

These commands use the previously extracted samples and `evaluation/perception_speed_compare_001.json`. They show ten sampled frames, rather than every frame of the recording. The preview creates no new detector predictions.

Choose an unused output filename. In the cached object viewer, press the left/right arrow keys or `p`/`n` to change samples; its original cached results contain object detections. Historical camera-only 3D results are ignored. The object-details table is removed. Image and BEV labels retain frame-local IDs, which are not tracking IDs. A right-side panel compares desired curvature (yellow) and vehicle curvature estimated from localized rear-axle motion (cyan) on one 1/m axis, with target speed (m/s) below, over 10 seconds of source-clock time. Enable [Pure Pursuit](path_tracking.md#live-curvature-comparison) for live/replay targets; invalid or unconfigured values show unavailable and create gaps rather than zero values. Vehicle curvature is a trailing pose-based estimate, not CAN steering feedback. Live/replay also supplies road/lane masks; [the road/lane guide](road_lane_detection.md) provides a saved segmentation preview.

| Key | Layer |
| --- | --- |
| `1` | BEV cloud points |
| `2` | LiDAR projection on the image |
| `3` | Camera 2D detections |
| `4` | LiDAR 3D boxes |
| `5` | Object labels |
| `6` | Road mask on the camera image |
| `7` | Lane mask on the camera image |
| `8` | Semantic road-mask edges on the camera image |
| `9` | Map and trajectory from live localization, online mapping or the original saved recording |

## Display inputs

| Topic | Display layer |
| --- | --- |
| `/camera/front/image_raw` | Native camera image |
| `/camera/front/camera_info` | Image calibration |
| `/points` | Cloud background in BEV and projected returns on image |
| `/camera_lidar_bev/detections_2d` | YOLO image rectangles and labels |
| `/camera_lidar_bev/lidar_detections_3d` | PointPillars cuboids and BEV footprints |
| `/camera_lidar_bev/camera_lidar_detections_3d` | Existing camera-associated LiDAR subset |
| `/camera_lidar_bev/localization/odometry` | Independent live `enu` → `os_sensor` pose in localized mode |
| `/camera_lidar_bev/localization/detections_3d` | Source-paired camera-matched LiDAR boxes in ENU |
| `/camera_lidar_bev/road_mask` | Native camera road mask |
| `/camera_lidar_bev/lane_mask` | Native camera lane mask |

Launch arguments `view_fps`, `max_points`, `range_m`, `lateral_m`, `intrinsics_file`, `extrinsics_file`, `lidar_engine_file`, `camera_2d_engine_file` and `road_engine_file` select rendering and saved-model inputs. Engine paths default to the already prepared local TensorRT files. The LiDAR detector class profile includes cars and people for 3D prediction; the camera 2D detector also labels motorcycles, buses and trucks. This display adds no tracker or new detector classes.

## GPS-aligned saved map

### Live vehicle and objects on the saved map

Run the combined display with independent localization against the frozen map:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
bash tools/run_localized_bev.sh --map maps/local_gps_recording_002
```

This starts or reuses the passive GPS receiver, dense `2048x10` Ouster stream and front camera, then starts scan-to-map localization, YOLO, PointPillars and camera/LiDAR association. Road/lane inference is disabled by default in this mode. It creates no recording and does not change the saved map. Stay within its mapped coverage with the same sensor mounting.

The camera image and flat 80 m BEV retain their current layout. The saved point-cloud map appears behind current points, and the north-up inset shows the live sensor position, its trajectory and source-paired objects. Orange identifies raw LiDAR predictions; green identifies the camera-matched subset. Camera rectangles alone have no metric map location. Display points are sampled before coordinate conversion to reduce CPU work; detection and localization still receive the original scans. Status separates native input count from sampled valid display count. Press `9` to toggle the map overlay, and `q`/Escape or Ctrl+C to stop the session's owned processes. Reused sensor processes remain running.

LiDAR boxes retain their raw scan-start headers; localization publishes native scan-end poses. The viewer joins these using the original cloud's `t` offsets, an exact pose or short bounded interpolation, and no extrapolation. Stale or rejected localization clears the map placement and world boxes while current sensor detections can still appear in the local BEV. It never uses the original recording's trajectory to locate the live rig.

`/camera_lidar_bev/localization/detections_3d` publishes the fresh camera-matched LiDAR subset in `enu`, with transformed centers and full orientations, preserving predicted dimensions, classes and scores. `/camera_lidar_bev/combined_view/status` reports the source stamps and pose pairing. A bounded four-scan subscription queue retains a fresh source while localization completes; it does not extend freshness limits. This adds world placement, not persistent object tracking or measured absolute accuracy. The original-recording mode below continues to use `--saved-map`; the independent live mode uses `--localize-map` internally.

The [isolated ROS protocol check](../data/localized_bev_protocol_001/report.json) uses clearly marked synthetic inputs to exercise nonempty world publications, full 3D transforms, preserved prediction fields and clearing after localization stops. It starts no physical drivers and supplies no accuracy or speed measurement.

The [final coupled live check](../evaluation/localized_bev_live_001.json) observed 10 FPS and map availability in all 200 checked frames over 20 seconds after initialization. Localization processing was 78.9 ms median and 90.9 ms p95, with a 0.250 s median pose source age after registration. Camera-matched world boxes were observed in an earlier physical scene; the final scene had no accepted matches. These measurements describe this run, not absolute localization or detection accuracy. A [live preview](../data/localized_bev_live_005/check/combined_view.png) shows the completed layout.

Replay the completed map with camera detections, LiDAR points and LiDAR 3D boxes:

```bash
bash tools/replay_combined_bev.sh \
  --saved-map maps/local_gps_recording_002 \
  --no-road-lanes --max-points 300000 --range-m 80
```

The original source bag and its separate `gps_alignment/` output are selected automatically. The blue backdrop uses the saved local map; the inset and position status use east/north/up coordinates. The badge reads **MAP SAVED ON [9] | GPS ALIGNED** when a source-paired pose is available. Press `9` to toggle the map and route, and `q`/Escape to close the replay. This starts no physical sensors and runs no mapping estimator. The replay uses source trajectory poses from the same bag; it does not localize another recording or a live rig against this map.

Native scan-end timestamps select the recorded pose, with short bounded interpolation and no extrapolation. Existing exact camera labels and source freshness limits still apply. Spatial cropping happens before map sampling; `--max-map-points` changes only the backdrop display limit. `--gps-alignment PATH` selects another alignment for this same map. Incomplete exports, incompatible provenance and mismatched available checksums are rejected. `--saved-map` cannot be combined with `--mapping`.

`/camera_lidar_bev/combined_view/status` publishes JSON with image/cloud/detection stamps, pose-query details, accepted association count and saved-map readiness. See [the mapping guide](lidar_imu_mapping.md#gps-alignment-of-a-saved-map) for alignment outputs and its approximate SBAS GPS accuracy limits.

## Local map backdrop

Add `--mapping` to the replay helper to run FAST-LIO2 LiDAR–IMU odometry and accumulate a map behind the existing flat BEV:

```bash
bash tools/replay_combined_bev.sh bags/camera_lidar_20261010T195520_925217Z \
  --rate 1 --no-road-lanes --max-points 300000 --range-m 80 --mapping
```

This includes `/imu`, forces one-pass playback, and saves a new map/trajectory under `maps/local_<UTC timestamp>/`. `--mapping-output maps/NEW_map` selects another unused directory. The adapter waits for two seconds of low-motion IMU data; the beginning of this recording is moving, so `--start-offset 58` is a later initialization attempt, not a guaranteed stationary interval.

The viewer receives `/camera_lidar_bev/mapping/map`, `/camera_lidar_bev/mapping/path` and `/camera_lidar_bev/mapping/odometry`. It uses the source-paired `map → os_sensor` pose to express the accumulated map in the current sensor frame. Press `9` to toggle the backdrop. A missing/stale pose clears it.

The mapping estimator has no GNSS fusion, loop closure or HD-road-map annotations; moving objects can remain in its points. Finished outputs can receive the separate offline GPS anchor described above. The [live GPS and LiDAR localization view](lidar_imu_mapping.md#compare-live-gps-with-lidar-localization) matches new scans against the frozen map. `run_localized_bev.sh` supplies these live poses to this camera/BEV display; `--saved-map` continues to use recorded poses. Deskew and odometry still use unverified host-derived timing. See [local LiDAR–IMU mapping](lidar_imu_mapping.md) for build requirements, standalone/live commands, frames and saved outputs.
