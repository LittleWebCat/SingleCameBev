# Local integration changes

The FAST-LIO estimator, IKFoM update, IMU propagation/deskew and ikd-Tree remain
the pinned upstream implementation. Local changes support this project's Ouster
128-row input and ROS2 Jazzy build:

- C++17 CMake; remove unused Python and pcl_ros requirements; optional Livox
  code is disabled by default. OpenMP uses two threads, configurable through
  FASTLIO_THREADS from one to four on ARM and x86.
- Match current Ouster `ring:uint16`, `t:uint32` nanoseconds and float32 XYZ/
  intensity. Reject incompatible fields/layout, nonfinite coordinates and invalid
  scan geometry. Preserve native offsets from the source cloud header rather
  than guessing per-point timestamps. Time-sort retained points before scan-end
  synchronization, including organized destaggered scans.
- Configurable `frames.map_frame` and `frames.body_frame`; `publish.tf_en=false`
  by default because the project's bridge publishes the calibrated sensor pose.
  Odometry describes the IMU body origin. Current pose covariance is computed
  before publishing, and linear velocity is expressed in the body frame. The
  pinned IKFoM state order is position then rotation; its right rotation error
  is converted with diag(I, R_map_from_imu) into map fixed axes, including
  position/rotation cross-covariances, for the ROS pose covariance contract.
- Reliable large-cloud subscription; IMU sensor QoS with 256-sample history.
  Excessive queue backlogs stop mapping instead of growing without a bound.
- Timestamp rewind stops the estimator with an explicit error; restart for each
  recording. This backend does not support looping/rewinding an active replay.
- `publish.map_en=false` skips the upstream accumulated-map timer. The project
  owns voxel map accumulation and save behavior. Upstream PCD save and debug log
  writes are disabled and attempts to enable them are rejected.

The initial frame is local and has no GNSS/world alignment. The estimator has
no loop closure or saved-map relocalization. Factory LiDAR/IMU extrinsics are
required, and host receive timestamps have not been calibrated for jitter.

## Realtime backlog and diagnostic update

`common.latest_scan_only=true` in the project profile discards older queued
LiDAR registration scans when a newer cloud is available. Their paired time
entries are removed together, and a pending selection is cleared. Every queued
IMU sample is retained until it is consumed by the newly selected scan, so IMU
propagation covers the interval since the last processed cloud. Published pose
and registered-cloud stamps remain the native selected scan's end time. The
point stride, surface/map voxels, noise and factory geometry are unchanged.

The estimator processing callback runs on a wall timer, so queued measurements
can drain after bag playback pauses or reaches EOF. Measurement synchronization,
deskew and published pose stamps still use the original source timestamps.

`/fast_lio_diagnostics` publishes lightweight backend counters and timings once
per wall second; bringup remaps it to `/camera_lidar_bev/mapping/raw/diagnostics`.
It reports input/output native-header ages, full estimator time, preprocessing,
IMU/deskew, downsampling, correspondence update, map insertion, publishing,
retained points and queued/drop counts. Backlog drops are explicit.

In ikd-Tree `multi_thread_rebuild`, `rebuild_flag=false` now executes while
`working_flag_mutex` is still held, before clearing Rebuild_Ptr and unlocking.
This closes the interval in which a waiting writer could observe rebuilding as
active after completion and leave an operation in the next rebuild logger. The
upstream ERROR printf remains visible; it is not suppressed.
