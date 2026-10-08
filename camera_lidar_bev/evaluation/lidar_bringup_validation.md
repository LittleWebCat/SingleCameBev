# LiDAR input and recording/replay validation

Validated on 2026-10-06. This is a sensor transport milestone; camera/LiDAR calibration, clock alignment, odometry, and perception remain incomplete.

## Active inputs

The project's launcher started the reachable Ouster OS1-128 using the existing `ouster_ros/os_driver` executable. Configure and activate responses and resulting lifecycle states were checked sequentially. The original `/home/agxthor/sensors_ws` files were not edited.

| Input | Topic | Frame | Received in approximately 5 s | Sample receipt rate |
| --- | --- | --- | ---: | ---: |
| LiDAR | `/points` | `os_lidar` | 50 | 10.026 Hz |
| Camera image | `/camera/front/image_raw` | `camera_front_optical_frame` | 150 | 30.015 Hz |
| CameraInfo | `/camera/front/camera_info` | `camera_front_optical_frame` | 150 | 30.019 Hz |
| Ouster IMU | `/imu` | `os_imu` | 500 | 100.023 Hz |

Evidence: [lidar_inputs_verified.json](lidar_inputs_verified.json). These are short callback-receipt measurements, not sustained sensor or full-perception performance. External Xsens `/sensing/imu/imu_data` and `/lio_sam/mapping/odometry` were not discovered in ROS domain 1.

Clouds are organized 128 × 2048, with 262,144 declared points and a 48-byte stride: 12,582,912 bytes per cloud. Layout validation passed. Fields include XYZ, intensity, `t`, reflectivity, ring, ambient, and range. `t` is UINT32 at byte offset 20, with a bounded sample spanning 0–99,841,184 raw units. Local driver source defines it as relative nanoseconds. Its approximately 100 ms span is consistent with the configured 10 Hz scan mode; this does not establish camera acquisition synchronization or valid deskew.

The sensor uses internal oscillator timing. `TIME_FROM_ROS_TIME` in the driver estimates scan-start headers from host packet reception. The camera uses host time after capture returns. LiDAR header age during the initial reliable check was approximately 223–239 ms at callback receipt. Header age is not an independently verified end-to-end acquisition-latency measurement.

Firmware: v3.1.0. Ouster ROS source revision: `f56b1a0e0b12985c41d3f2630344fb5c5963f571`. Driver initialization reported Ouster client 0.16.1. The sensor metadata and original driver parameter file used for each sample are preserved with the ignored bags.

## Host and subscription settings

The sensor was reachable at `169.254.213.23`, using host interface `enP2p1s0` at `169.254.1.100/16`.

Temporary host tuning applied:

| Setting | Previous | Applied |
| --- | ---: | ---: |
| `net.core.rmem_max` | 212992 | 16777216 |
| `net.core.rmem_default` | 212992 | 4194304 |
| `net.ipv4.ipfrag_time` | 30 | 3 |
| `net.ipv4.ipfrag_high_thresh` | 4194304 | 134217728 |

No persistent sysctl file was changed. `tools/start_lidar.py` only reads these settings; it does not tune the host automatically.

The point publisher advertises Reliable, KEEP_LAST depth 1. An initial Best Effort inspection received no complete large clouds during a 5 s sample, while a Reliable inspection received 50. The probe now exposes `lidar_reliability`, defaults to Reliable, and uses a one-scan queue. The inventory chooses reliability from advertised PointCloud2 publishers. Camera and IMU subscribers retain sensor-data Best Effort.

## Recording-load checks

All bags are under ignored `bags/`; source control contains reports and tools rather than the MCAP payloads.

| Bag | Recorded span | LiDAR | Image / CameraInfo | IMU | LiDAR header gaps >150 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lidar_camera_check_001` | 9.602 s | 67 | 288 / 288 | 960 | 27 |
| `lidar_camera_check_002` | 4.615 s | 32 | 138 / 138 | 462 | 12 |
| `lidar_camera_check_003` | 4.659 s | 46 | 140 / 140 | 466 | 0 |

The first sample had the additional live point-cloud diagnostics subscriber. The second retained that subscriber while increasing the recorder's CycloneDDS receive buffer to 16 MiB; it still received about 7 scans/s. The final sample stopped that temporary observer and kept the larger recorder buffer. It recorded approximately 10 scans/s, with a median scan-header interval of 99.990 ms and a maximum of 100.563 ms. No scan-header regressions were found in any sample.

Evidence: [recording_point_gaps.json](recording_point_gaps.json), each bag's `metadata.yaml`, sibling session JSON, and recorder log. These checks demonstrate a load-sensitive input path. They do not prove loss-free capture for longer sessions, multiple consumers, or the future complete perception pipeline. Avoid redundant large-cloud subscribers during baseline recording, then inspect saved counts and gaps. The buffer-only trial did not resolve the slowdown.

The final bag also contains Ouster `/metadata` and `/tf_static`. No `/tf` stream or vehicle odometry was recorded. Driver-internal static transforms do not provide independently measured sensor-to-vehicle calibration. Vehicle stationary status was not independently verified; no moving-deskew or road-evaluation gate was attempted.

## Replay

The final bag was replayed in localhost ROS domain 42 with `/clock` and the probe's `use_sim_time=true`, excluding recorded diagnostics. The replay check received all 46 saved clouds, all 140 images and CameraInfo messages, and all 466 IMU messages. Cloud payload layout and the `t` field were preserved. Replay child processes exited cleanly.

Evidence: [replay_check_003.json](replay_check_003.json). The earlier bag also replayed, with delivery differences on Best Effort camera/IMU topics. Expected stale-input errors appear after playback finishes; calibration and missing-odometry warnings remain explicit.

## Remaining gates

- Measure camera intrinsics/distortion at the operating dimensions.
- Measure camera–LiDAR and sensor–`base_link` transforms and validate independent projections.
- Establish a usable motion trajectory. Ouster acceleration/gyro data has no orientation estimate and is not odometry.
- Validate acquisition timing and deskew on motion sequences.
- Profile recording and processing with the full intended consumer load before claiming sustained 20 Hz BEV.

Ground estimation, occupancy ray tracing, rolling-map updates, semantic fusion, and tracking have not been implemented in this milestone.
