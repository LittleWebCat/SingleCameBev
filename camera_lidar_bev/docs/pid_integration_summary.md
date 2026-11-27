# BEV and Pure Pursuit signals for PID integration

Prepared **2026-10-14** for the current camera/LiDAR BEV project and 2025 Nissan LEAF mounting configuration.

The project provides diagnostic tracking targets, localized motion estimates, path errors and perception. It currently publishes **no vehicle commands**. A connection to an existing PID still needs a defined input/output contract and the vehicle feedback appropriate to that controller.

An optional sibling [DBW bridge](../../bev_bridge/README.md) now consumes these diagnostics. Its corrected implementation defaults to dry-run, requires fresh live ENU rear-axle targets and explicit release/engagement, and latches stop/transport faults. Hardware mode requires an explicit configuration: either documented firmware verification or the operator-managed external-safety arrangement requested by the user. The external profile keeps unknown firmware checks marked unverified. The bridge defaults to a 6 mph speed cap; current validation allows an explicit cap up to 10 mph. The current external profile requests +/-300 steering-wheel degrees / 80 degrees/s, exceeding its mode's +/-90-degree validation limit and blocking hardware startup. Firmware source was not found locally; no moving-vehicle control has been performed by the bridge tests. A historical real-adapter standby check observed valid live GPS/localization/tracking with zero command events. The tracker itself still publishes diagnostics only. The run-all entry point is `bash bev_bridge/run_all.sh --dry-run` from the repository root; hardware startup, the configuration blocker and explicit engagement are described in the bridge README.

## Main interface

| Item | Current implementation |
|---|---|
| Topic | `/camera_lidar_bev/path_tracking/status` |
| ROS message | `std_msgs/msg/String`; parse `data` as JSON |
| Publisher | C++ `bev_path_tracking/path_tracker` |
| Fresh-update rate | Approximately 10 Hz when the live viewer and localization supply poses; this is a target, not a guaranteed rate |
| Publisher QoS | Reliable, volatile, keep-last depth 10 |
| Live ROS domain | `1` by default |
| Runtime | ROS 2 Jazzy; current launch uses Cyclone DDS |
| Purpose | Diagnostic inputs for inspection/comparison; `diagnostic_only` is always `true` |

The node publishes results upon receiving source poses. A separate 100 ms wall timer publishes waiting/stale diagnostics when appropriate. **Topic message frequency can include invalid messages and does not prove fresh tracking updates.**

## Targets, estimates and errors

| JSON field | Meaning | Unit / convention |
|---|---|---|
| `curvature_per_m` | Geometric desired Pure Pursuit curvature, before preview steering limits | 1/m; positive left |
| `vehicle_curvature_per_m` | Observed rear-axle curvature estimated from localization poses | 1/m; positive left |
| `steering_rad` | Desired equivalent road-wheel steering angle after preview angle/rate limiting | rad; positive left; not steering-wheel angle |
| `target_speed_mps` | Preview speed recommendation based on curvature and route endpoint | m/s; reaches zero at the endpoint |
| `estimated_speed_mps` | Filtered speed from consecutive rear-axle poses | m/s; not wheel-speed feedback |
| `motion_speed_mps` | Average horizontal rear-axle path speed over the curvature window | m/s |
| `estimated_yaw_rate_rad_s` | Average wrapped heading change over the same window | rad/s; positive counterclockwise |
| `lateral_error_m` | Signed displacement from the matched reference segment | m; positive left of the reference direction |
| `heading_error_rad` | Vehicle heading minus reference heading | rad; wrapped to the signed pi interval |
| `progress_m` | Arc-length progress along the reference | m |
| `remaining_m` | Distance remaining in the current continuous route section | m |
| `lookahead_m` | Arc-length lookahead used to select the target | m |
| `steering_saturated` | Geometric steering request exceeded the preview angle limit | Boolean; does not flag rate limiting |
| `processing_ms` | Tracker calculation duration | ms; excludes localization, transport and rendering |

`curvature_per_m` and `steering_rad` can describe different curvature after limits: `tan(steering_rad) / wheelbase_m` is the equivalent curvature of the limited preview angle. Neither field is an actuator command.

## Geometry, source and validity

| JSON field | Meaning |
|---|---|
| `frame` | `enu`: world X east, Y north, Z up |
| `pose_reference` | `rear_axle` when the mounting transform is enabled; otherwise `os_sensor` |
| `ego_xyz` | Current configured pose position in ENU meters; XYZ array |
| `projection_xyz` | Matched reference projection in ENU meters; XYZ array |
| `lookahead_xyz` | Selected lookahead position in ENU meters; XYZ array |
| `reference_xyz` | Reference horizon in ENU meters; array of XYZ points, approximately up to 80 m ahead |
| `source_stamp_ns` | Source-pose timestamp as an integer number of nanoseconds; preserve 64-bit/integer precision |
| `source_mode` | `live_localization`, `saved_recording_pose` or `viewer_paired_pose`; configured by the launcher |
| `diagnostic_only` | Always `true`; the message is not an actuation authorization |
| `tracking_valid` | Route match and reference geometry are valid |
| `targets_valid` | Desired curvature, steering and preview speed are available |
| `vehicle_curvature_valid` | Observed vehicle-curvature estimate is available independently of desired targets |
| `vehicle_transform_verified` | Configured transform-enable flag; does not certify surveyed mounting accuracy |
| `vehicle_curvature_source` | `localization_pose_estimate` |
| `motion_curvature_window_s` | Configured trailing estimation window; default 0.5 s |
| `motion_window_span_s` | Actual duration covered by the valid motion estimate |
| `state` | State/rejection reason described below |

Unavailable target/error/motion values are `null`. Geometry is unavailable when tracking is invalid; reference horizons can be empty. Fields such as `progress_m`, `remaining_m` and `lookahead_m` may contain default zeros in an invalid result, so check flags before using any numbers.

Typical states include `waiting`, `waiting_for_speed`, `tracking`, `endpoint`, `sensor_path_only_vehicle_transform_unset`, `lookahead_behind_vehicle`, `off_route_or_heading`, `localization_stale`, `invalid_pose`, `invalid_pose_frame_or_stamp`, `out_of_order_pose`, `pose_jump` and `reverse_motion_unsupported`.

## Timing and estimation limits

- Fresh pose input is rejected when its source age exceeds **0.5 s** or it is more than **50 ms** ahead of the ROS clock. The tracker also expires data after **0.5 s of wall receipt age**, including a paused replay clock. These are current diagnostic thresholds, not validated controller latency limits.
- Consumers need their own source-age and monotonic receipt-age checks. Reject invalid flags even if their timestamp repeats. Do not integrate PID error again merely because the same source timestamp was received; do not treat a missing or `null` value as zero. A rewind/restart requires resetting temporal state.
- The live wrapper uses the live ROS clock; saved replay uses the recorded `/clock`. `source_mode` is descriptive metadata, not proof that a physical sensor is running. Keep replay and a vehicle control session separate.
- Vehicle curvature uses `sum(wrapped heading changes) / sum(horizontal travel)` from configured rear-axle poses. Under planar forward motion with negligible sideslip, this estimates yaw rate divided by path speed.
- The default window is **0.5 s**, with at least **80%** of its duration, **0.25 m** travel and both current and window speed at least **0.5 m/s**. Stops, reverse motion, resets/gaps or angular jumps over 2 rad/s clear the motion history. Values remain unavailable during warm-up and low speed.
- Pose-derived curvature is smoothed and delayed; it is not CAN steering feedback or independent ground truth. The chart performs no vehicle-response or filter-delay compensation.
- The viewer requires a fresh tracking result within **150 ms** of the displayed source pose. That is a display-pairing tolerance, not a controller update period.

The right-side chart covers **10 s** of source time: yellow desired curvature, cyan estimated vehicle curvature, and green preview target speed. Its numerical difference is **vehicle minus desired**.

## Possible PID input mappings

| PID's controlled quantity | Available reference | Available feedback / limitation |
|---|---|---|
| Curvature | `curvature_per_m` | `vehicle_curvature_per_m`; localization estimate with delay and low-speed gaps |
| Road-wheel steering angle | `steering_rad` | No measured road-wheel or steering-wheel angle in this project |
| Speed | `target_speed_mps` | `estimated_speed_mps` or `motion_speed_mps` for diagnostic comparison; no measured wheel speed |
| Yaw rate | No native desired-yaw-rate field | `estimated_yaw_rate_rad_s`; trailing pose estimate |
| Path tracking using lateral/heading errors | Reference geometry and path errors | Localized rear-axle pose; controller-specific error/sign mapping still needed |

For a curvature error in reference-minus-feedback convention:

```python
if fields['targets_valid'] and fields['vehicle_curvature_valid']:
    error_per_m = fields['curvature_per_m'] - fields['vehicle_curvature_per_m']
```

This excerpt illustrates the subtraction only. Parsing, finite-value, freshness, source-mode and timestamp checks are required before consumption. It does not define a PID gain, update rate, output unit, steering conversion or vehicle command.

## Vehicle and preview configuration

| Parameter | Current value / provenance |
|---|---|
| Vehicle | 2025 Nissan LEAF |
| Wheelbase | 2.70 m |
| LiDAR forward from rear axle | 1.25 m, user measured |
| LiDAR left of axle/body centerline | 0.01 m, inferred as `(0.905 - 0.885) / 2`, assuming symmetric sides and aligned centerlines |
| Sensor yaw in vehicle | Nominal 0 rad; connector faces rear, assuming an upright Ouster |
| `vehicle_transform_verified` | `true` for the nominal diagnostic transform; not a surveyed extrinsic calibration |
| Vehicle axes | X forward, Y left, Z up |
| Preview steering bound | +/-0.5 rad, about +/-28.6 degrees |
| Preview steering-rate bound | 0.3 rad/s |
| Tracker preview speed cap | 4.4704 m/s = 10 mph (about 16.09 km/h); curvature and endpoint limits can lower it |
| Separate DBW bridge speed cap | Defaults to 6 mph (2.68224 m/s); current validation permits explicit values up to 10 mph |
| Preview deceleration / lateral acceleration | 1.0 / 1.0 m/s² |
| Lookahead | `clamp(3 m + 1.5 s * estimated speed, 3 m, 12 m)` |
| Endpoint margin | 0.5 m |
| Route match limits | 5 m position distance and 60 degrees heading difference |

These preview limits are not measured Nissan actuator limits. The conversion is planar; vertical lever-arm and roll/pitch effects are not modeled. `os_lidar` has a different axis convention from `os_sensor` and must not replace it.

## Other available topics and files

| Interface | Available information |
|---|---|
| `/camera_lidar_bev/localization/odometry` | `nav_msgs/msg/Odometry`, ENU pose of `os_sensor`; `twist` is unset and its zeros are not measured speed |
| `/camera_lidar_bev/localization/status` | Localization acceptance and diagnostics as JSON |
| `/camera_lidar_bev/path_tracking/source_pose` | `nav_msgs/msg/Odometry`, source-selected ENU sensor pose published by the combined viewer. In live localization the tracker is remapped to read `/camera_lidar_bev/localization/odometry` directly instead, so this topic is display-side only there |
| `/points` | Raw LiDAR point cloud; new live drivers use `2048x10`, 10 Hz |
| `/imu` | Ouster IMU measurements |
| `/camera/front/image_raw` | Front camera image stream |
| `/camera/front/camera_info` | Camera calibration |
| `/camera_lidar_bev/detections_2d` | Camera object detections |
| `/camera_lidar_bev/lidar_detections_3d` | LiDAR object detections |
| `/camera_lidar_bev/localization/detections_3d` | Fresh camera-matched LiDAR detections placed in ENU |
| `maps/local_gps_recording_002/pointcloud_map.pcd` | Saved local point-cloud map |
| `maps/local_gps_recording_002/gps_alignment/trajectory_enu.csv` | Full saved ENU sensor reference trajectory; `stamp_sec,x_m,y_m,z_m,qx,qy,qz,qw` |
| `maps/local_gps_recording_002/gps_alignment/alignment.json` | Geographic origin and local-map-to-ENU alignment |

The tracker converts the sensor reference trajectory and current sensor poses to rear-axle poses using the same mounting configuration. The CSV itself is not already a rear-axle trajectory. Map/capture artifacts are local data and may not be present in a fresh Git checkout.

Perception and geographic alignment are provisional. Detections do not alter the Pure Pursuit targets. There is no obstacle-avoidance planner, persistent object-motion prediction or HD lane map in this tracking path.

## Run and inspect

Start the live localized display for a new manually driven pass on the mapped route; close conflicting replay/viewer sessions first:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
ROS_DOMAIN_ID=1 bash tools/run_localized_bev.sh \
  --map maps/local_gps_recording_002 --pure-pursuit --range-m 80
```

In a second terminal:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$PWD/config/transport/cyclonedds_large_clouds.xml"

ros2 topic info /camera_lidar_bev/path_tracking/status --verbose
ros2 topic echo /camera_lidar_bev/path_tracking/status --once
```

The JSON is inside the ROS String message's `data` field. Parse source nanoseconds as an integer; converting a Unix-epoch nanosecond timestamp to an ordinary JavaScript Number loses integer precision.

Optionally record the small diagnostic/pose streams for comparison in a new output directory:

```bash
ros2 bag record --storage mcap -o bags/pid_inputs_TEST_001 \
  /camera_lidar_bev/path_tracking/status \
  /camera_lidar_bev/path_tracking/source_pose \
  /camera_lidar_bev/localization/odometry \
  /camera_lidar_bev/localization/status
```

This command does not record raw sensor clouds or any external PID/vehicle feedback. Add the actual controller's topics once known. Stop recording with Ctrl+C.

## Integration gaps and current evidence

The current tracker input is coupled to the combined viewer. Closing or stalling the viewer expires tracking. A vehicle-control integration needs a pose feed independent of visualization, an explicit typed controller interface, matching units/signs and controller-specific timeout handling.

The project has no measured steering angle, wheel speed, steering ratio/actuator calibration, throttle/brake feedback, vehicle CAN command adapter, or validated actuation limits. Pose-derived speed/curvature cannot fill in those missing hardware interfaces automatically. No PID gains or controller-output mapping have been configured.

The latest [curvature software check](../evaluation/path_tracking_curvature_compare_001.json) passed **163 ROS checks**, with median tracker calculation time about **0.14 ms**. C++ geometry/motion tests and eight Python overlay/history tests also passed. A short saved replay produced both chart lines. These are software and recorded-data checks; a new physical-drive or closed-loop PID performance result has not been measured.

Implementation references: [tracker node](../src/bev_path_tracking/src/path_tracker.cpp), [tracking core](../src/bev_path_tracking/src/pure_pursuit.cpp), [vehicle/configuration YAML](../config/tracking/pure_pursuit.yaml), [tracking guide](path_tracking.md), and [mapping/localization guide](lidar_imu_mapping.md).
