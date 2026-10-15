# Recorded-route tracking with Pure Pursuit

The ROS 2/C++ `bev_path_tracking` package compares a source-paired pose with the saved ENU route and calculates diagnostic Pure Pursuit targets. The camera and flat BEV layout is retained. A gold line shows the reference horizon, a gold circle marks the lookahead, and a cyan circle marks the projected reference point. The banner reports lateral/heading errors and available steering/speed targets.

The object-details table is removed. A right-side panel compares **desired curvature (yellow)** with **vehicle curvature estimated from localized motion (cyan)** on the same 1/m axis, and plots target speed in m/s below it. Both charts cover the last **10 seconds of source-clock time**. Positive curvature turns left; negative turns right. Repeated source samples are not duplicated, unavailable values make independent gaps, and replay rewind clears both histories. With the rear-axle transform unset, readouts show unavailable rather than zero.

This is a route follower/monitor, not an obstacle-avoidance planner. It publishes JSON diagnostics on `/camera_lidar_bev/path_tracking/status`, with no vehicle-command publisher. Detections do not change the tracking targets. Speed and steering limits in the configuration are preview values rather than measured actuator limits.

## Build

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
source install/setup.bash
colcon build --base-paths src --packages-select bev_path_tracking bev_bringup \
  --symlink-install --cmake-args \
  -DPython3_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Release
```

The existing sensor and detector packages must already be built for the combined view.

## Saved recording

```bash
bash tools/replay_combined_bev.sh \
  --saved-map maps/local_gps_recording_002 \
  --pure-pursuit --no-road-lanes --range-m 80
```

This starts no physical sensors. Reference positions come from `gps_alignment/trajectory_enu.csv`; displayed poses come from the same recording's saved trajectory, paired at native scan end. It checks software geometry and the overlay, not independent localization accuracy or closed-loop control. Replay clock rewind resets progress and clears old results.

## New drive

```bash
bash tools/run_localized_bev.sh \
  --map maps/local_gps_recording_002 --pure-pursuit
```

This starts or reuses the existing live sensors, detectors and independent map localizer. It matches against the saved route by location, heading and progress rather than by original route timestamps. Manually driving the route supplies a new tracking observation. Press `q` or Ctrl+C to stop owned processes.

Add `--tracking-config /path/to/your_tracking.yaml` to either command for another configuration. The default is [pure_pursuit.yaml](../config/tracking/pure_pursuit.yaml). The reference is selected from the map's GPS alignment and must share the current pose's ENU coordinate system.

## Live curvature comparison

For a new, manually driven pass over the saved route, close the saved replay or previous localized viewer, then run:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
ROS_DOMAIN_ID=1 bash tools/run_localized_bev.sh \
  --map maps/local_gps_recording_002 --pure-pursuit --range-m 80
```

The live wrapper starts or reuses the sensor streams and estimates new poses against the frozen map. Drive on the mapped route to acquire a compatible reference segment. The chart shows desired Pure Pursuit curvature, estimated vehicle curvature and `vehicle - desired` at the same source timestamp. The desired value is the geometric Pure Pursuit request before preview steering angle/rate limits; it is not an actuator command. The target-speed chart is a preview recommendation, not measured vehicle speed. This remains a diagnostic display during manual driving.

No CAN steering or wheel-speed measurement is available. Vehicle curvature is estimated using the **rear-axle** poses after applying the configured sensor lever arm: `sum(wrapped heading changes) / sum(horizontal travel)`. Under the planar forward-motion/no-slip assumption, this is windowed yaw rate divided by path speed. It uses only current and previous poses, independent of the reference and desired curvature. The estimator defaults to a trailing **0.5 s** window, requiring at least 80% of that duration, **0.25 m** travel, and both current and window speed at least **0.5 m/s**. A stop, reverse motion, pose gap/reset or angular jump over 2 rad/s clears the motion history. Warm-up, low speed, stale or invalid tracking produces gaps rather than zero curvature.

The window reduces noise and introduces lag; localization latency and heading noise also affect the estimate. This is not steering feedback, ground truth, or an independent controller-performance measurement. The plotted comparison is not time-shifted to compensate for vehicle response or filter delay. Configure `motion_curvature_window_s`, `motion_curvature_min_speed_mps` and `motion_curvature_min_distance_m` in the tracking YAML. The diagnostic JSON adds `vehicle_curvature_valid`, nullable `vehicle_curvature_per_m`, `motion_speed_mps`, `estimated_yaw_rate_rad_s` and `motion_window_span_s`, with `vehicle_curvature_source: localization_pose_estimate`. Historical replay uses recorded poses and must not be interpreted as a new live test.

## Vehicle transform

The vehicle is a **2025 Nissan LEAF**. Earlier measurements: body length **4.481 m**, width **1.791 m** excluding mirrors, wheelbase approximately **2.70 m**, LiDAR-to-front **2.35 m**, and LiDAR-to-left/right **0.885/0.905 m**. These do not directly determine the rear-axle center.

The user reports the LiDAR cable connector faces the rear bumper. For an upright Ouster, `os_sensor` +X points away from the connector, so the nominal sensor yaw relative to the car is **0 rad**. This establishes the mounting direction; exact angular alignment has not been measured. The `os_lidar` +X axis points toward the connector and must not be substituted for `os_sensor`. See [Ouster's coordinate-frame guide](https://static.ouster.dev/sensor-docs/image_route1/image_route2/sensor_data/sensor-data.html).

The user subsequently measured the LiDAR **1.25 m ahead of the rear-axle centerline**. This replaces the earlier 1.345 m estimate from regional body dimensions. The default configuration now uses `lidar_forward_from_rear_axle_m: 1.25`, `lidar_left_from_rear_axle_m: 0.01` and `sensor_yaw_in_vehicle_rad: 0.0`, with `vehicle_transform_verified: true` to enable diagnostic targets. The 0.01 m left offset is inferred as `(0.905 - 0.885) / 2`, assuming symmetric body sides and alignment of the body and axle centerlines. These tape measurements and mounting direction establish a nominal planar transform, not a surveyed extrinsic calibration.

Curvature, steering and preview target speed become available after consecutive valid poses establish speed. If `vehicle_transform_verified` is set to `false`, errors and reference geometry remain in `os_sensor`, while `steering_rad`, `curvature_per_m` and `target_speed_mps` are `null` and `targets_valid` is `false`.

For vehicle-based targets, set measured `lidar_forward_from_rear_axle_m`, `lidar_left_from_rear_axle_m` and `sensor_yaw_in_vehicle_rad`, then set `vehicle_transform_verified: true`. Positions use forward/left signs; yaw is the sensor +X heading relative to vehicle +X, positive counterclockwise. Both reference and current poses receive the same planar rear-axle conversion. Vertical lever-arm and roll/pitch effects are not modeled. Steering is the desired road-wheel angle, not steering-wheel angle.

## Algorithm and input handling

Recorded vertices are preserved, near-duplicates removed, and targets interpolated by arc length. Reference gaps over **0.5 s** split the route; lookahead does not bridge splits. Defaults require a compatible section within **5 m / 60 degrees**. Once acquired, matching stays near previous progress in the same continuous section to avoid jumping at crossings; restart/reset to acquire another section.

Lookahead is `clamp(3 m + 1.5 s × estimated speed, 3 m, 12 m)`. Curvature is `2 × target_left / target_distance²`; steering is `atan(wheelbase × curvature)`, bounded by preview angle/rate limits. The configured tracker preview speed ceiling is **4.4704 m/s (10 mph, about 16.09 km/h)**. The optional DBW bridge separately defaults to a lower **6 mph** cap. Preview speed decreases with curvature and remaining distance and reaches zero at the section endpoint. Calibrated targets support forward motion only.

Speed is filtered from consecutive source poses; unset localizer `Odometry.twist` is not used. Localization noise/latency and unverified acquisition clocks affect these diagnostics. Measured vehicle speed/steering feedback and delay compensation remain future control work.

In live localization (`start_bev.py --pure-pursuit`), the tracker's `/camera_lidar_bev/path_tracking/source_pose` input is remapped to `/camera_lidar_bev/localization/odometry`, so it receives each accepted, fresh localizer pose directly (about 213 ms after scan end) rather than the viewer's copy of the same pose up to one render period later. In replay, the viewer publishes `source_pose` only for a valid, source-paired ENU sensor pose. The tracker rejects invalid frames/quaternions, out-of-order data and pose jumps. Source or wall age over **0.5 s** clears results, including when replay is paused. Overlay results must be fresh and within **150 ms** of the displayed scan-end pose.

## Checks

```bash
ctest --test-dir build/bev_path_tracking --output-on-failure
/usr/bin/python3 -m unittest discover -s tests -p 'test_path_tracking_overlay.py'
```

C++ checks cover steering sign, transform gating, curves, endpoints, gaps, crossings and kinematic bicycle convergence. Python checks cover ENU overlay transforms and stale/invalid clearing. To run the isolated ROS check with recorded poses and a synthetic calibrated fixture:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ROS_DOMAIN_ID=94 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  /usr/bin/python3 tests/check_path_tracking_ros.py \
  --output evaluation/NEW_path_tracking_software.json
```

It starts only the diagnostic tracker and refuses an existing report. Its synthetic calibration does not configure the physical rig. Recorded-pose/reference agreement is a software check; physical tracking accuracy remains unmeasured. The [first software report](../evaluation/path_tracking_software_001.json) passed 98 checks and measured median core calculation time of 0.169 ms, p95 0.179 ms, over 30 recorded poses. These times exclude localization, inference, DDS and rendering.

After configuring the user's 1.25 m forward measurement, the [LEAF configuration check](../evaluation/path_tracking_leaf_measured_001.json) passed 128 checks, including 10 recorded poses with the configured rear-axle conversion and finite preview targets after speed initialization. It also retained the unconfigured-transform, stale-input and synthetic steering-sign checks. No physical sensors were started or vehicle commands published; this verifies software behavior against the saved route, not independent physical tracking accuracy.

The [curvature comparison check](../evaluation/path_tracking_curvature_compare_001.json) passed 163 ROS checks, including independent desired/observed signals in a synthetic straight-motion case with a lateral route offset and unavailable observed curvature at a stop. C++ checks also cover left/right circular motion, yaw wrap, the sensor lever arm, window warm-up, stationary heading noise and reset/gap handling. A short saved replay produced both chart lines using recorded poses. A new physical drive has not yet been measured.
