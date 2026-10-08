# First recording report

Copy this template for each session. Save bag files outside Git and link their location here. The first milestone is reproducible recording/replay and an honest input inventory, not a 20 Hz BEV benchmark.

## Session

- Session ID and UTC start/end:
- Bag location and `ros2 bag info` output:
- Sensor driver launch/configuration paths and revision:
- Computer, JetPack, ROS distribution, CUDA/TensorRT versions, power mode:
- Sensor models/serial numbers, mounting configuration, selected camera mode:
- Scene and vehicle motion:
- Missing inputs or known limitations:

## Input inventory

| Input | Topic/type | Frame | Configured rate | Measured received rate | Timestamp source/domain | Loss or gaps |
| --- | --- | --- | --- | --- | --- | --- |
| LiDAR cloud | | | | | | |
| Front image | | | | | | |
| CameraInfo | | | | | | |
| Xsens IMU | | | | | | |
| Motion/odometry | | | | | | |
| GNSS, if used | | | | | | |
| TF/static TF | | | | | | |

The current Ouster file selects `2048x10`, `TIME_FROM_ROS_TIME`, and `xyzir`. Check the active driver parameters and actual messages; a configuration file alone does not establish live behavior.

## Point timing and alignment

- PointCloud2 field names, datatypes, offsets, `point_step`, `row_step`:
- Per-point time field, units, range, and reference time:
- Header stamp meaning: scan start/end/other:
- Sensor clock relationship and synchronization mechanism:
- Camera stamp meaning: exposure/device/host receipt/other:
- IMU and motion stamp conventions and available interpolation coverage:
- Alignment checks on moving data:

A missing point-time field prevents validated point-level deskewing. Field presence alone does not establish correct units or a shared clock. Message arrival times do not measure the physical exposure-to-scan offset.

## Recording and replay checks

- Bag topics/counts/duration match the inventory:
- QoS compatibility and any recorder warnings:
- IMU rate and point-cloud completeness under recording load:
- Camera image and CameraInfo resolution agree:
- Required TF relationships present or explicitly missing:
- Replay with `/clock` and `use_sim_time` reaches the probe:
- Header stamps remain monotonic within each stream:
- Missing/stale inputs are visible in diagnostics:
- Bag, sensor metadata, and calibration/configuration provenance preserved:

## Calibration status

- Camera intrinsics/distortion measured at operating resolution:
- Camera–LiDAR transform and independent projection checks:
- LiDAR–base and IMU–base transforms:
- Motion/odometry source and frame relationship:
- Timing validated during motion:

Keep missing calibration marked unknown. Do not publish identity transforms solely to make RViz look aligned.

## Decision

- Recording/replay gate: pass / fail / partial
- Deskew prerequisites: pass / fail / partial
- Missing evidence and next concrete action:
- Numerical thresholds to agree before geometry/semantic evaluation:

For later performance reports, distinguish publication rate, fresh-scan rate, processing latency, and sensor-to-output age. Include p95/p99 and sustained-load measurements.
