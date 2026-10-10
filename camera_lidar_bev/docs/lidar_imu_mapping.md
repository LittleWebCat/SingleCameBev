# Local LiDAR–IMU mapping

FAST-LIO2 now estimates local motion and deskews Ouster scans using LiDAR points and the internal six-axis IMU. The mapping adapter accumulates registered points, exports a trajectory and supplies a local map backdrop for the existing flat BEV. Camera labels and LiDAR object detection remain separate.

The estimator produces a local point-cloud map. Its `map` frame has an arbitrary initial position and heading. Offline GPS alignment anchors a saved copy to local east/north/up coordinates with a geographic origin; it does not feed GPS into the estimator. An experimental live GPS/scan-matching comparison can now estimate new sensor poses against this frozen map. Loop closure and Lanelet2 lane topology remain unimplemented. Moving vehicles and people can leave points in the accumulated map. Lane/road annotations and independently validated localization are later steps toward an HD road map.

## Build

Use the existing ROS 2 Jazzy environment and system Python:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
colcon build --base-paths src --packages-select fast_lio bev_mapping bev_bringup \
  --symlink-install --cmake-args \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release
```

The wrappers source the installed overlay. Large clouds also require the existing host DDS setup, `sudo bash tools/setup_dds_transport.sh`, if it has not already been applied.

The pinned FAST-LIO backend in `src/fast_lio` is GPL-2.0 and runs as a separate executable. Its license and upstream revision remain with the vendored source. The adapter provides input checks, frame conversion and export; it does not replace the estimator.

## Map the saved 10 Hz recording

This command maps the new 10 Hz GPS recording once, starts no physical sensors and saves a new output directory:

```bash
bash tools/run_mapping.sh \
  --bag bags/camera_lidar_20261010T221427_088256Z \
  --domain 90 --rate 0.5 \
  --max-map-voxels 8000000 --map-publish-period 600
```

Standalone mapping uses `ROS_DOMAIN_ID` if set, otherwise domain **90**, recorded-clock playback at **1×**, and a new `maps/local_<UTC timestamp>/` folder. Add `--domain 90` to explicitly isolate a bag replay. Use `--output maps/NEW_local_map` to select a new folder; existing output directories are refused. Ctrl+C stops the processes owned by this invocation and asks the adapter to save its map. `--duration 60` bounds the wall-clock run time.

Standalone bag runs disable the backend's latest-scan-only policy in their generated runtime configuration. Live and combined runs keep their existing policy. `--max-map-voxels` sets this run's map capacity; the first pass of this recording reached four million voxels, so the example allows eight million. `--map-publish-period 600` avoids large intermediate map publications during this short offline run; the final map and trajectory are still saved on exit. Captured factory transforms remain in the generated configuration. GPS is recorded in this bag but is not replayed or fused by this mapping command; the resulting map uses local coordinates.

The adapter waits for a **two-second low-motion IMU interval** before forwarding inputs. Early moving scans are skipped. The new GPS recording has a suitable quiet interval at its beginning, so it needs no offset. The older `camera_lidar_20261010T195520_925217Z` recording began with movement/vibration; a sampled interval near 60 seconds was quieter. If that older bag is still available, initialization can be tried there:

```bash
bash tools/run_mapping.sh \
  --bag bags/camera_lidar_20261010T195520_925217Z \
  --start-offset 58
```

That offset is a useful starting point, not proof of a stationary interval. Initialization still has to pass the gate. Keep a live rig still until initialization finishes and the estimator begins publishing poses. Low IMU variation alone cannot distinguish rest from constant-speed motion.

## GPS alignment of a saved map

The new map's alignment is saved in `maps/local_gps_recording_002/gps_alignment/`. The original local PCD and trajectory are unchanged. `alignment.json` contains the WGS84 origin, `T_enu_from_map`, source checksums, timing conventions and fit statistics. The separate `pointcloud_map_enu.pcd` and `trajectory_enu.csv` use +X east, +Y north and +Z up; `gps_alignment_preview.png` compares the aligned route with recorded GPS.

For a new saved map that has GPS in its original recording:

```bash
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source install/setup.bash
/usr/bin/python3 tools/align_local_map_gps.py maps/YOUR_LOCAL_MAP
```

The source bag is read from `run_manifest.json`. The tool reads only GPS fixes, matches their source headers to local trajectory timestamps and fits a covariance-weighted planar rigid transform. It preserves scale, relative elevation and all point-cloud returns. It applies a constant vertical offset, removes invalid fixes, avoids interpolation across long pose gaps and refuses an existing output directory. `--output maps/YOUR_LOCAL_MAP/NEW_alignment` selects another unused destination. It requires sufficient movement to estimate heading and rejects incompatible source/frame provenance.

This recording provided 1,860 matched positions and a horizontal fit RMS residual of **0.90 m**. This measures GPS–LiDAR/IMU agreement, not true map accuracy. Most fixes used SBAS; antenna position relative to the LiDAR and physical clock alignment remain unverified. The geographic anchor is approximate, and the estimator still has no GPS fusion or loop closure. Coordinate conversion follows [ESA's ECEF/ENU equations](https://gssc.esa.int/navipedia/index.php/Transformations_between_ECEF_and_ENU_coordinates).

## Replay the saved map with camera and BEV

This uses the finished map and its recorded trajectory without running FAST-LIO again:

```bash
bash tools/replay_combined_bev.sh \
  --saved-map maps/local_gps_recording_002 \
  --no-road-lanes --max-points 300000 --range-m 80
```

The helper selects the original bag from the map manifest and finds `gps_alignment/` automatically. `--gps-alignment PATH` selects another alignment belonging to the same map. The window retains the camera image, current LiDAR returns, YOLO 2D detections and LiDAR 3D boxes. The blue saved map sits behind current BEV points; the inset shows the recorded route in east/north coordinates. Press `9` to hide/show the map and route, or `q`/Escape to stop this replay and its detectors. Playback loops in domain 89 at half speed by default; add `--no-loop` for one pass.

Saved poses are selected using the paired LiDAR scan's original header plus its native per-point end offset. Interpolation is restricted to short pose brackets, with no extrapolation. The camera/cloud freshness and 150 ms pairing limits remain in force. The map is cropped to the visible region before display sampling; `--max-map-points` changes this display limit without changing the saved map. GPS anchoring changes world-coordinate status and the route inset; sensor-relative map, box and camera geometry stays consistent.

This mode only replays the original recording. It rejects a different source bag and verifies available alignment checksums. It does not estimate the current rig's position against a saved map. `--saved-map` and online `--mapping` are mutually exclusive. JSON frame, pose, map and association status is available on `/camera_lidar_bev/combined_view/status`.

## Compare live GPS with LiDAR localization

The comparison has its own north-up map window. Orange shows the GPS antenna position and an ellipse derived from reported covariance. Cyan shows the independently matched LiDAR sensor pose and heading. The header distinguishes **LIVE** sensor data from **REPLAY**. Source/receipt freshness and registration rejection clear expired or invalid dots; paths remain as history. GPS has no heading arrow at rest.

Run both methods using the connected sensors:

```bash
bash tools/run_localization_compare.sh --live --start-sensors --max-rate 10
```

This starts missing passive GPS and `2048x10` Ouster streams in domain 1. Matching existing drivers are reused and left running on exit; newly started drivers are owned by the controller. It does not start cameras, object detectors or a recording. Press `q`/Escape in the map window or Ctrl+C in its terminal to stop owned processes. If streams already run, omit `--start-sensors`. The default localization cap is 5 Hz; `--max-rate 10` selects the cap used in the stationary live test. Keep the same sensor mounting and stay within mapped coverage for an initial drive.

Compare a saved GPS/LiDAR recording at recorded speed:

```bash
bash tools/run_localization_compare.sh \
  --bag bags/camera_lidar_20261010T221427_088256Z \
  --start-offset 40 --duration 30
```

Replay defaults to isolated domain 92 and one pass. The map index and normal cache are prepared before playback begins. `--map PATH`, `--alignment PATH`, `--metadata PATH`, `--output NEW_DIRECTORY`, `--no-display` and `--max-rate` select the reference, factory geometry, reports and display. For another recording, the same frozen map is used; the estimator does not read source trajectory poses. A different recording must include original points, IMU, GPS fixes, metadata and static transforms. Live metadata defaults to the map's original sensor calibration; supply current factory metadata if the sensor differs.

The prototype converts native Ouster returns into `os_sensor`, uses original scan-end timestamps, and matches a bounded scan against the immutable 3D map. GPS supplies a coarse position; 24 geometric yaw hypotheses initialize heading. Approximate gyro/velocity compensation supports tracking. Overlap, structural support, residual, observability, motion and freshness gates reject unsupported matches. GPS displacement can aid catch-up after initialization, but no GPS-only fallback is published as a LiDAR pose. Cached map normals reduce recurring crop costs. Covariance on the pose is nominal display uncertainty, not measured accuracy.

| Test | GPS | LiDAR scan matching |
| --- | --- | --- |
| Stationary live output | 10 Hz | 10 Hz |
| Median processing | 0.62 ms per fix callback | 76.6 ms per accepted scan |
| Processing p95 | 0.70 ms | 82.9 ms |
| Reported/test quantity | Receiver covariance | 0.253 m median scan residual |

The stationary live test accepted 188 of 190 attempts; the two exceptions occurred during startup. Header conventions differ: GPS headers use receiver-stream receipt time, while LiDAR poses use scan end. In this test the LiDAR pose's median header age at publication was 0.277 s. Physical acquisition clocks and the antenna lever arm remain unverified. Covariance, scan residual and GPS–LiDAR separation do not establish absolute position accuracy.

The subsequent [new-drive session](../data/localization_compare_live_002/lidar_localization.json) accepted 3,117 poses over 312.9 seconds at 9.96 Hz. Its only two rejected attempts occurred during startup; tracking then continued with a maximum 0.200 s update gap. Median processing was 77.1 ms, p95 87.1 ms, and median pose source age after processing was 0.247 s. The [map snapshot](../data/localization_compare_live_002/latest.png) shows the live route. These logs establish timing and tracking continuity, but do not provide an independent accuracy reference or serialized pose coordinates for measuring drift.

The sampled replay benchmark accepted all 60 scans across five route sections. After normal caching, median scan processing including decoding was 69.1 ms, p95 96.4 ms, initialization about 1.05 s, and one-time map preparation about 4.24 s. This recording also built the reference map, so its reference agreement is optimistic. Reproduce the sampled benchmark in a new output directory:

```bash
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source install/setup.bash
/usr/bin/python3 tools/benchmark_map_localization.py \
  --map maps/local_gps_recording_002 \
  --bag bags/camera_lidar_20261010T221427_088256Z \
  --output evaluation/NEW_localization_performance
```

Evidence: [live performance](../evaluation/localization_live_performance_001.json), [live performance chart](../evaluation/localization_live_performance_001.png), [sampled comparison](../evaluation/localization_performance_002/comparison.json), and [sampled comparison plot](../evaluation/localization_performance_002/comparison.png). Each controller session under `data/localization_compare_<UTC>/` saves `lidar_localization.json`, `viewer_report.json`, `session_report.json`, `latest.png` and owned process logs.

| Topic | Meaning |
| --- | --- |
| `/camera_lidar_bev/localization/odometry` | Accepted, source-stamped `enu` → `os_sensor` pose |
| `/camera_lidar_bev/localization/status` | Compute stages, acceptance/rejection and counters |
| `/camera_lidar_bev/localization/view_status` | GPS/pose freshness and viewer timings |
| `/camera_lidar_bev/localization/map_image` | Optional north-up image from the viewer |

This is a CPU NumPy/SciPy registration prototype. It does not add loop closure, a guarantee of global convergence or LiDAR accuracy validation on a new drive. Its live poses can now drive the combined camera/BEV display; the existing `--saved-map` replay continues to use recorded poses.

## Live localization with camera and object BEV

```bash
bash tools/run_localized_bev.sh --map maps/local_gps_recording_002
```

This starts missing GPS, `2048x10` LiDAR and camera streams, the independent scan matcher, YOLO, PointPillars and camera association. The flat 80 m BEV shows the frozen map behind live points. A north-up inset places the sensor and source-paired objects in ENU. The fresh camera-matched LiDAR subset is published on `/camera_lidar_bev/localization/detections_3d`; camera-only rectangles have no metric map position. Road/lane inference is disabled by default to reduce concurrent work.

Native scan-end timestamps join detector source clouds to live poses, with bounded interpolation and no extrapolation. Invalid or stale localization clears world placement. The map stays immutable and recorded trajectory poses are never used in this live mode. Press `9` for the map layer, and `q`/Escape or Ctrl+C to stop owned processes; reused sensor drivers remain running. This does not create a recording or add persistent object tracking. See [the combined display guide](combined_bev_display.md#live-vehicle-and-objects-on-the-saved-map) for display and topic details.

The [coupled live check](../evaluation/localized_bev_live_001.json) with camera and detectors running maintained 10 FPS and a ready map in all 200 checked frames over 20 seconds after initialization. Localization processing was 78.9 ms median, 90.9 ms p95; median pose source age after registration was 0.250 s. Display-only point sampling and bounded scan buffering keep visualization current without reducing detector or localizer input scans. These results establish runtime behavior, not absolute accuracy.

## Build a map beside the camera/BEV replay

The combined replay can run mapping together with camera 2D detection and LiDAR 3D detection:

```bash
bash tools/replay_combined_bev.sh \
  bags/camera_lidar_20261010T195520_925217Z \
  --rate 1 --no-road-lanes --max-points 300000 --range-m 80 \
  --mapping
```

`--mapping` includes the recorded IMU and forces **one-pass playback**. It uses the combined helper's domain, **89** by default. Add `--mapping-output maps/NEW_combined_map` to select a new output directory, or `--start-offset 58` for the later initialization attempt. Close the viewer with `q` or Escape, or use Ctrl+C, to stop the run and save the current map.

Press **`9`** to toggle the accumulated local map and path behind the current BEV points and boxes. The viewer converts map coordinates through the matching `map → os_sensor` pose. A missing/stale pose clears that overlay; it does not place map points directly into the sensor frame. The displayed 80 m range crops the local backdrop, while the mapper retains the route accumulated during this run.

## Live mapping

With the Ouster driver already publishing `/points`, `/imu` and metadata in domain 1:

```bash
ROS_DOMAIN_ID=1 bash tools/run_mapping.sh --live
```

This helper starts the mapper only. It does not start sensor drivers or the camera viewer. A combined viewer already running in domain 1 can receive its map and odometry. Keep the rig still for initialization before moving. The live profile uses the factory transforms of the project's Ouster; metadata from another sensor must agree with the supplied configuration.

## Outputs

Each successful run contains:

| File | Content |
| --- | --- |
| `pointcloud_map.pcd` | Binary XYZ/intensity points in the local `map` frame |
| `trajectory.csv` | Source timestamp, position and quaternion for `os_sensor` in `map` |
| `mapping_report.json` | Map size, pose/scan counts, initialization and failure state |
| `run_manifest.json` | Input bag, backend revision and calibration/timing provenance |
| `fast_lio_runtime.yaml`, `adapter_runtime.yaml` | Effective parameters derived from captured factory metadata |
| `mapping.log` | Estimator and adapter output |

The exported map keeps one observed point per **0.25 m voxel**, up to **500,000 voxels** by default. `--max-map-voxels` changes this capacity for a standalone run. When full, existing cells remain and new cells stop being added. This bounds memory; it does not remove moving objects or provide free-space evidence. A run with no registered points writes a report and trajectory but no PCD. Inspect `failed_input`, `map_point_count`, `voxel_limit_reached` and the counters in the report before using a map.

Render a saved map and trajectory as a flat PNG:

```bash
/usr/bin/python3 tools/preview_local_map.py maps/YOUR_local_map
```

This writes `map_preview.png` beside the PCD. The new GPS recording was processed in full in `maps/local_gps_recording_002/`: **4,128,074 map points**, **1,855 poses** and no voxel-capacity limit reached. This pass used playback at 0.55×, eight million allowed voxels and a 600-second map publication period. The adapter received all 1,889 scans and 18,885 IMU messages; it skipped 20 startup scans and five without sufficient IMU coverage. The exported cloud includes 1,855 registered scans. These counters describe delivered mapping inputs and outputs; they do not establish accuracy or real-time throughput.

The earlier short run in `maps/local_mapping_check_002/` produced 195,620 map points and 87 sensor poses.

| Public topic | Contract |
| --- | --- |
| `/camera_lidar_bev/mapping/odometry` | `nav_msgs/Odometry`, parent `map`, child `os_sensor` |
| `/camera_lidar_bev/mapping/map` | Accumulated `PointCloud2` in `map`, reliable and transient-local |
| `/camera_lidar_bev/mapping/path` | Sensor trajectory in `map`, reliable and transient-local |
| `/camera_lidar_bev/mapping/diagnostics` | Initialization, input and map-capacity status |
| `/camera_lidar_bev/mapping/save_map` | `std_srvs/Trigger` service to save the current map |

The adapter owns dynamic TF `map → os_sensor`; the backend's TF publisher is disabled. The driver supplies the existing factory `os_sensor → os_lidar` and `os_sensor → os_imu` transforms. This does not establish a vehicle `base_link` calibration. Pose covariance is moved to the sensor origin; twist uncertainty remains conservative.

## Timing and calibration limits

The input is the original cloud in `os_lidar`, with `t` as UINT32 nanoseconds from scan start and `ring` as UINT16. `/imu` uses `os_imu`, acceleration in m/s² and angular velocity in rad/s. Its orientation is unavailable. The adapter preserves headers and per-point times, rejects clock rewind, and forwards scans only when IMU coverage spans their point times. Use a fresh mapping run rather than looping a bag.

Factory transforms from the selected bag are converted from millimeters to meters. For this Ouster, `T_imu_from_lidar` has rotation `diag(-1, -1, 1)` and translation `[0.002441, 0.009725, 0.030662] m`. Estimation of these extrinsics is disabled. The sensor-frame pose uses the matching IMU-to-sensor lever arm.

Current sensor headers are host-derived, and their physical acquisition alignment remains unverified. The profile sets no guessed clock correction and leaves FAST-LIO's automatic time-offset heuristic disabled. Deskew and trajectory accuracy therefore remain provisional. The new GPS recording contains Ouster IMU and GNSS positions. Offline GPS alignment supplies a geographic anchor, but this mapper does not fuse those GNSS positions and has no Xsens orientation or independent trajectory reference. Saved maps are not surveyed HD maps or verified global localization results.
