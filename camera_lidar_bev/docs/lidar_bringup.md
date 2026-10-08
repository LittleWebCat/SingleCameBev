# LiDAR bring-up and the first bounded recording

To inspect distant returns in an existing RViz session, load [raw_lidar_wide.rviz](../config/rviz/raw_lidar_wide.rviz) through **File → Open Config**. This uses the original `/points` topic with white points, a wider zoom (`Scale = 8`, `X = 40 m`), and forward upward. The approximate 0–80 m display depends on the viewport size; the preset changes visualization only. The camera-object layout's default scale of 18 and center 7 m ahead show approximately 30 m forward in an 860-pixel viewport.

The [range investigation](../evaluation/lidar_range_diagnosis_001.json) found 88.7 m maximum range in a previously recorded full geometry scan, with returns beyond 60 m. This is earlier evidence, not a current live range measurement. The active sensor configuration read during the investigation was `1024x20`, normal operation and a full 360-degree window. Seeing a distant vehicle in the camera does not guarantee enough unoccluded LiDAR returns from it; check the wide original scan before changing association thresholds.

The next milestone is a stationary LiDAR + front-camera recording that can be replayed. It remains a partial sensor session until a usable motion trajectory, calibration, and clock relationships are verified. Receiving an IMU topic does not establish odometry or a deskew trajectory.

The current user-selected mode is **`1024x20`**: 128 vertical channels and 1024 horizontal columns, giving 131,072 returns per scan at 20 Hz. [The live mode/rate check](../evaluation/lidar_20hz_001.json) verifies actual dimensions, 19.999 Hz reception, preserved driver geometry, and original per-point timestamps. Earlier recording evidence below used `2048x10`; those recordings and metadata remain historical inputs. Before the switch, metadata was preserved in `data/lidar_runtime/metadata_2048x10_before_20hz_001.json`.

Ouster documents `1024x20` among its [supported scan modes](https://static.ouster.dev/sensor-docs/image_route1/image_route2/common_sections/API/sensor_configuration_description.html). The 20 Hz choice halves horizontal points per scan compared with `2048x10`. The project configuration controls future bring-up; the driver did not write persistent sensor startup settings.

## Inspect the active driver first

Use the same `ROS_DOMAIN_ID` in the camera, LiDAR, probe, and recorder terminals. The existing sensor launcher uses domain `1`; verify the active processes before starting another driver. Source the installed ROS environment and this project's overlay:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=1
ros2 topic list -t
ros2 node list
```

If no Ouster driver is already active, inspect the project's launcher and then start it in its own terminal:

```bash
/usr/bin/python3 tools/start_lidar.py --dry-run
/usr/bin/python3 tools/start_lidar.py
```

The launcher uses [ouster_baseline.yaml](../config/sensors/ouster_baseline.yaml), starts `/os_driver` with automatic lifecycle startup disabled, and waits for configure/activate to succeed. It refuses to start a duplicate driver. `--params-file` selects a reviewed alternative and `--startup-timeout` bounds each startup wait. Its UDP-buffer check is read-only; host network tuning is a separate action. The driver workspace remains `/home/agxthor/sensors_ws`; its original files are preserved.

The initial bring-up applied these temporary host settings after inspecting the small default buffers:

```bash
sudo sysctl -w net.core.rmem_max=16777216 net.core.rmem_default=4194304 \
  net.ipv4.ipfrag_time=3 net.ipv4.ipfrag_high_thresh=134217728
```

These settings are not made persistent. The launcher does not run sudo or change them automatically. Runtime sensor metadata is saved under the ignored `data/lidar_runtime/` directory. See [first_steps.md](first_steps.md) for the integration procedure.

Inspect the effective cloud before collecting data:

```bash
ros2 topic info /points --verbose
ros2 topic echo /points --once --field header
ros2 topic echo /points --once --field fields
ros2 topic hz /points
```

Stop the rate check with Ctrl+C. Also inspect `/imu`, the camera topics, available odometry, and the diagnostic topic. Save measured rates and actual frame IDs in [first_recording.md](../evaluation/first_recording.md); configured rates are not measurements.

The configured driver publishes `/points` reliably. Initial Best Effort subscribers received no complete full-size `original` scans during a short check; a Reliable subscriber received the cloud. The probe therefore defaults to `lidar_reliability:=reliable` with a one-scan queue. Use `lidar_reliability:=best_effort` only when matching another point-cloud source. The inventory tool automatically checks advertised point-cloud reliability; camera and IMU subscriptions remain sensor-data Best Effort.

## Point timing and frame checks

### Live top-down view

With the existing driver running and the ROS environments above sourced, open the live LiDAR BEV viewer:

```bash
ROS_DOMAIN_ID=1 DISPLAY=:1 \
  CYCLONEDDS_URI=file:///home/agxthor/SingleCameraBev/camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml \
  rviz2 -d config/rviz/live_bev.rviz
```

The view uses `/points`, the driver's `os_sensor` fixed frame, and `TopDownOrtho`. White points show measured returns from above against a black background. Use the mouse wheel to zoom and left-drag to rotate. This is raw LiDAR visualization; it does not classify free space or obstacles, accumulate a rolling map, or fuse camera semantics. Close other large-cloud viewers to reduce duplicate subscription load. The live display and global status were verified OK.

The inspected legacy configuration uses `2048x10`, `TIME_FROM_ROS_TIME`, and `xyzir`. `xyzir` lacks the per-point timing required for point-level deskewing. A timing-bearing `original` cloud can include `t`; verify the actual field, its units, its reference, and its relationship to the scan header before using it. A `t` field alone does not establish synchronized sensors.

With reception-based `TIME_FROM_ROS_TIME`, the relationship between LiDAR acquisition time, the ROS host clock, and the camera exposure remains unverified. The camera bridge timestamps after capture returns. Record these limitations instead of interpreting small header ages as proof of physical synchronization.

Confirm transforms from measured mounting/calibration. Preserve LiDAR and IMU frames reported by the driver. Do not create identity transforms solely to align the displays. A stationary session can validate data transport and replay; moving deskew validation requires calibrated transforms and an adequately timed motion trajectory.

## Record for a bounded duration

The initial capture exposed load from duplicate large-cloud subscriptions: the recorder and live probe received about 7 scans/s together. With the temporary probe stopped, a final short bag captured 46 scans, 140 image/info pairs, and 466 IMU messages over 4.659 seconds. Avoid extra LiDAR viewers/probes during the baseline capture; inspect the bag afterward. This is short-run evidence, not a sustained throughput guarantee.

The optional [CycloneDDS profile](../config/transport/cyclonedds_large_clouds.xml) requests a 16 MiB receive buffer for a recorder process. It requires the host receive limit shown above. A buffer-only trial did not resolve the observed multi-subscriber slowdown; do not treat the profile alone as a fix. The final capture used the profile with one LiDAR subscriber:

```bash
CYCLONEDDS_URI=file:///home/agxthor/SingleCameraBev/camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml \
  /usr/bin/python3 tools/record_sensor_session.py \
  --output /tmp/bev_static_session_002 --duration 10
```

Keep the vehicle stationary, with visible ground, objects at several distances, and an occluded area. The helper records only on explicit invocation, stops with SIGINT after the requested duration, and falls back to bounded termination if the recorder does not stop. It refuses an existing bag, log, or report path. Save bags outside Git.

First inspect the exact command without starting DDS or creating files:

```bash
/usr/bin/python3 tools/record_sensor_session.py \
  --output /tmp/bev_static_session_001 --duration 10 --dry-run
```

After the desired live streams are verified, record:

```bash
/usr/bin/python3 tools/record_sensor_session.py \
  --output /tmp/bev_static_session_001 --duration 10
```

The defaults request `/points`, `/imu`, `/metadata`, `/camera/front/image_raw`, `/camera/front/camera_info`, `/tf`, `/tf_static`, and `/camera_lidar_bev/diagnostics`. Ouster `/metadata` and `/tf_static` can use transient-local durability; inspect their saved counts to confirm retained messages were received. ROS bag recording can finish successfully while requested topics are absent; inspect counts rather than treating the helper's `completed` status as a full sensor gate. Add verified odometry and replace topic names with the inventory from the active graph when available:

```bash
/usr/bin/python3 tools/record_sensor_session.py \
  --output /tmp/bev_motion_session_001 --duration 10 \
  --topics /points /imu /metadata /camera/front/image_raw /camera/front/camera_info \
  /verified/odometry /tf /tf_static /camera_lidar_bev/diagnostics
```

`/verified/odometry` is a placeholder: replace it with the actual validated trajectory topic before invocation. This is a later moving-session example, not a claim that odometry currently exists.

The helper selects MCAP if `rosbag2_storage_mcap` is available in the sourced ament index and otherwise requests SQLite3. It does not install packages. Its sibling `.recorder.log` captures recorder output; `.session.json` records the command, requested topics, host UTC start/end, monotonic elapsed duration, ROS domain, return status, and explicit unverified calibration/motion/clock status. These are operational metadata, not measured sensor counts or alignment evidence. Ctrl+C stops the child and marks the session interrupted; interrupted or failed bags require inspection before reuse.

## Inspect and replay

The reproducible automated check launches its own probe and bag player in localhost ROS domain 42, verifies the available input types and timing field, then stops both:

```bash
/usr/bin/python3 tools/check_sensor_replay.py bags/lidar_camera_check_003 \
  --output evaluation/replay_check_new.json
```

Use a new output path. The check explicitly expects missing odometry and pending calibration for this transport milestone; its success is a partial gate. Its result reports actual delivered counts, which can differ from saved counts with sensor-data Best Effort topics.

```bash
ros2 bag info /tmp/bev_static_session_001
```

Confirm the intended topic types, counts, duration, recorder warnings, and clock/frame inventory. Preserve Ouster metadata and effective driver parameters alongside the bag. A stationary LiDAR + camera + IMU bag with no odometry is a partial transport/replay milestone.

Replay in an unused ROS domain to avoid mixing live and recorded data. Use both terminals with ROS and this overlay sourced, setting the same unused domain:

```bash
export ROS_DOMAIN_ID=42
ros2 launch bev_bringup sensor_probe.launch.py use_sim_time:=true imu_topic:=/imu
```

In a second terminal:

```bash
export ROS_DOMAIN_ID=42
ros2 bag play /tmp/bev_static_session_001 --clock \
  --exclude-topics /camera_lidar_bev/diagnostics
```

Inspect the new probe's diagnostics for missing inputs, gaps, frames, and point fields. Excluding recorded diagnostics keeps replay observations distinct from the original reports. Complete [first_recording.md](../evaluation/first_recording.md) before calibration and LiDAR ground/occupancy implementation.
