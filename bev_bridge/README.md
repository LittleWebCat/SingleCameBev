# Live BEV to LEAF DBW bridge

The bridge validates live Pure Pursuit targets and converts road-wheel angle and target speed to the supplied adapter's expected four-field ASCII format. **Dry-run is the default.** Hardware mode requires an explicit configuration and never engages itself.

See [previous issues and solutions](../camera_lidar_bev/docs/issues_and_solutions.md#6-dbw-bridge-review-fixes-and-engagement-behavior) for the bridge review findings, replay rejection, adapter identification, stop-latch behavior, the 2026-10-14/15 hardware driving sessions and the distinction between command writes and measured vehicle response.

The software fixes do not certify vehicle safety. The ESP firmware source was not found locally. The operator reports tested hardware checks and an independent safety button, and explicitly takes responsibility for external safety. These are software limits, not measured actuator ratings.

## Current state (2026-10-15)

- **Hardware profile in use:** `hardware.json` (verified-configuration mode). The operator confirmed all six firmware/vehicle checks on 2026-10-14; the file records baud **115200**, sign **−1**, ratio **17:1**, firmware heartbeat timeout **0.5 s**, and steering caps **±280 steering-wheel degrees / 280 degrees/s**. The rate is above the **200 degrees/s** firmware limit the operator reported; this is recorded in its `verification_notes`. `hardware.json` is not covered by `.gitignore` (only `hardware.local.json` is).
- **`hardware.external.json` is rejected at preflight.** It requests ±300 degrees / 80 degrees/s, but the `operator_managed_external` mode allows at most ±90 degrees / 90 degrees/s regardless of the file. At ±90 degrees the minimum turning radius is about 29 m, which cannot follow this route's corners.
- **Measured input latency:** tracker targets reach the bridge about **213 ms** (median; 235 ms maximum) after the LiDAR scan end, against the bridge's 0.5 s source limit. See [pipeline latency](#pipeline-latency).
- **Open vehicle issue:** in hardware runs the measured vehicle speed did not follow the commanded speed in either direction (up to 17.6 mph against a 7 mph command; stationary for over 10 s while 5–7 mph was commanded). Localization agreed with GPS during those intervals, so the speeds were real motion. Firmware speed-hold, wheel-speed (`0x284`) delivery or operator pedal input has not been ruled out. See [hardware sessions](#hardware-sessions-2026-10-1415).

## Run the whole software stack

Close an existing localized BEV/viewer before starting another owned stack:

```bash
cd /home/agxthor/SingleCameraBev
bash bev_bridge/run_all.sh --dry-run
```

This supervises live sensors/localization/BEV plus a dry-run bridge. It does not publish engagement. Inspect `/bev_dbw/bridge_status` for readiness and the printed session directory for logs. `--print-only` validates and prints commands without starting or writing anything. `--duration 30` bounds a diagnostic session. Existing sensor processes reused by the BEV controller are not owned by this launcher.

The bridge requires valid live targets; GPS and localization must be ready. In live localization the path tracker reads accepted localizer poses directly from `/camera_lidar_bev/localization/odometry`; it no longer depends on the combined viewer's render loop. **Closing the viewer window still stops the whole `run_all` stack** ("view exited (0); stopping the owned stack").

`--max-speed-mph` defaults to **6 mph** (9.656 km/h, 2.68224 m/s) for both the launcher and standalone bridge. Validation and serial transport allow up to **10 mph** (16.093 km/h, 4.4704 m/s); larger values are rejected. The tracking YAML uses a 4.4704 m/s ceiling. The bridge sends the lower of its CLI cap and the tracker target. Curvature and endpoint limits can request lower speeds. Restart the stack to load changed settings; an existing process keeps its startup limits and hardware configuration.

## Explicit release and engagement

Use the same ROS environment as the launcher in a second terminal:

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///home/agxthor/SingleCameraBev/camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml
```

For a **dry-run only**, first publish an explicit release, then a short heartbeat test:

```bash
ros2 topic pub --once /bev_dbw/dry_run_engage std_msgs/msg/Bool '{data: false}'
timeout --signal=INT 2s ros2 topic pub -r 10 /bev_dbw/dry_run_engage std_msgs/msg/Bool '{data: true}'
```

An initial/latched True publisher cannot arm the bridge. Engagement is a software heartbeat, not a physical held-button deadman. After any command has been generated, invalid data, a missed heartbeat or explicit False latches a stop; new True messages cannot resume driving. Restart the bridge after resolving the fault and handing control to the operator. No launcher automatically publishes True. A standalone `bev_dbw_bridge.py` listens on `/bev_dbw/engage` unless `--engage-topic` is given; `run_all` passes `/bev_dbw/dry_run_engage` or `/bev_dbw/hardware_engage`.

## Simulated input (dry-run only)

`sim_status_publisher.py` publishes path-tracking status JSON in the accepted live shape with a sinusoidal steering target, so the bridge can be exercised without sensors. `--engage` also publishes a release followed by True heartbeats; use it **only against a dry-run bridge**. Run both with the system Python 3.12 (`/usr/bin/python3`); a conda Python earlier on `PATH` cannot import ROS Jazzy's `rclpy`.

```bash
cd /home/agxthor/SingleCameraBev/bev_bridge
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 bev_dbw_bridge.py --dry-run          # terminal 1
/usr/bin/python3 sim_status_publisher.py --engage     # terminal 2
```

With the defaults (±0.1 rad, 0.3 m/s) the dry-run log shows about ±97 steering-wheel degrees and 0.67 mph. Stopping the simulator latches the bridge stop, as designed.

## Hardware configuration

The verified-configuration profile currently used is `hardware.json`:

```bash
cd /home/agxthor/SingleCameraBev
bash bev_bridge/run_all.sh --actuate \
  --hardware-config bev_bridge/hardware.json --max-speed-mph 7
```

Run with `--print-only` first to validate a changed file without opening hardware. The launcher opens hardware but does not engage. After it reports fresh tracking readiness and the operator is ready to control the independent safety button, use a second terminal with the ROS environment above:

```bash
ros2 topic pub --once /bev_dbw/hardware_engage std_msgs/msg/Bool '{data: false}'
timeout --signal=INT 2s ros2 topic pub -r 10 /bev_dbw/hardware_engage std_msgs/msg/Bool '{data: true}'
```

The True heartbeat can produce real steering and speed commands. The example is bounded to two seconds; removing `timeout` streams until Ctrl+C. Afterward the bridge requests zero speed and latches the stop state. These topics do not read the physical safety button; the operator's external system is responsible for its enforcement. A latched stop requires restarting the bridge. Firmware stop behavior remains independently unverified by this project, so a software heartbeat timeout is not a substitute for the operator's brake or independent stop.

Before engaging, place the vehicle at the route start, aligned with the route. Engaging misaligned or past the route end makes the tracker request full steering lock immediately.

Steering limits and route geometry: the tightest corners of `maps/local_gps_recording_002` have radii of about **10.0, 11.8, 13.9, 14.4 and 16.1 m**. With a 2.70 m wheelbase and 17:1 ratio, a steering-wheel cap of 90/180/280/400 degrees allows minimum radii of about 29.1/14.4/9.1/6.2 m. The tracker itself limits road-wheel angle to 0.5 rad (about 487 steering-wheel degrees).

To create another verified profile, copy `hardware.example.json` to `hardware.local.json` (ignored by Git) and supply actual measured/verified values. The template deliberately fails hardware preflight. Required evidence includes the adapter protocol and baud, steering calibration and limits, CAN speed feedback, zero-speed braking, firmware timeout braking/watchdog interval, and independent hardware emergency stop. Do not turn an unknown check into `true` to bypass preflight. The config records human verification evidence; software cannot independently certify those assertions.

`hardware.external.json` records the operator-managed external-safety arrangement without marking firmware checks verified. Its mode is limited to ±90 degrees / 90 degrees/s and 10 mph and currently fails preflight because of its 300-degree request.

Hardware mode is restricted to firmware speed-hold. Local pose-derived pedal PI is available only for dry-run comparison, never as a fallback when CAN speed feedback is missing. Check `--help` for the standalone bridge options.

## Pipeline latency

Measured passively on the live stack (2026-10-14) as age at arrival relative to the LiDAR scan-end stamp:

| Stage | Before | After |
| --- | --- | --- |
| `/points` (12.6 MB `2048x10` cloud) reaches subscribers | ~143 ms | unchanged |
| Localizer odometry published | ~282 ms | ~212 ms |
| Tracker status (what the bridge checks) | ~347 ms median, 353 ms max | **~213 ms median, 226 ms p95, 235 ms max** |
| Largest gap between tracker updates | 200 ms (dropped scans) | 101.6 ms |

Two changes produced the improvement: the tracker takes `/camera_lidar_bev/localization/odometry` directly instead of the viewer relay (which republished the identical pose about 57 ms, up to 125 ms, later), and the localizer rate cap in `run_localized_bev.py` was raised from 10 to 20 Hz so it no longer waits or overwrites scans at the 10 Hz scan rate (~72 ms compute). Before the change, every target arrived about 0.35 s old, so one missed 10 Hz update exceeded the 0.5 s limit and latched a stop. The dense LiDAR profile remains the largest single delay.

## Hardware sessions (2026-10-14/15)

All sessions used `run_all.sh --actuate`; speeds are localization-derived, not wheel speed. Elapsed times are seconds after bridge start.

| Session | Profile / caps | Engaged driving | Stop cause |
| --- | --- | --- | --- |
| `all_20261014T233900_917070Z` | external, ±90°, 6 mph | 34 s, 3.4–4.5 mph | Target source age exceeded 0.5 s after one missed update (~0.35 s base latency) |
| `all_20261014T234332_624636Z` | external, ±90°, 10 mph | 31 s, 7.9–10 mph | Tracker invalid: pose arrived 0.55 s old |
| `all_20261014T234955_847740Z` | external, ±90°, 10 mph | 5.8 s | `target_expired_release_required`, source age 0.501 s |
| `all_20261014T235828_910824Z` | external, ±90°, 10 mph, latency fix | 90 s | Tracker invalid after steering pinned at ±90° (29% of commands; up to 456° requested) |
| `all_20261015T000744_347905Z` | `hardware.json`, 400°/200°/s, 7 mph | 27 s | Tracker invalid; at the 400° limit 63% of the time; nearly stationary for 14 s while 4.2 mph was commanded |
| `all_20261015T001056_017359Z` | 400°/200°/s, 7 mph | 99 s, ~266 m | Reached the route end at ~5 m/s versus a 2.6 m/s target and overran it; measured up to 9.7 m/s (21.6 mph) |
| `all_20261015T002212_578362Z` | 280°/90°/s, 7 mph | 58 s | Started past the route end; at the 280° limit 94% of the time; measured ≤0.11 m/s while 5–6 mph was commanded |
| `all_20261015T002759_229265Z` | 280°/280°/s, 7 mph | 109 s, ~267 m | Mostly 2.8–3.0 m/s; spike to 7.9 m/s; 12 s stationary; then 0→4.6 m/s into a right turn (2.4 m/s target) and tracker invalidation |

For the last session, LiDAR localization agreed with GPS within 1.77 m (median 0.77 m), fitness stayed ≥ 0.983 and all 1,093 engaged-interval scans were accepted, so its speed excursions were real motion rather than pose jumps. The bridge never sent more than the selected cap. The cause of the speed mismatch is unresolved.

## Corrected behavior

- Strict JSON object/number/Boolean validation. Missing/null/nonfinite values invalidate the target; they are not replaced with zero or accelerator creep.
- Require `live_localization`, ENU, configured rear axle, valid tracking/targets and strictly advancing positive integer source stamps. Both fresh source age and monotonic receipt age are checked; replay targets are rejected in all modes.
- Monotonic deadman/receipt timing and a steady-clock ROS timer. ROS clock jumps latch a restart-required fault; hardware mode rejects simulated time.
- Explicit False revokes engagement immediately. Invalid messages clear cached targets. No automatic re-engagement following a fault.
- Both target source choices receive road-wheel angle/rate and configured steering-wheel bounds. Source updates remain approximately 10 Hz; sending at 50 Hz does not create new perception information.
- Explicit device path, exclusive opening, known GPS/IMU rejection, busy-port checks, bounded writes, partial-write/error detection and no reconnect/retry of cached commands. Transport failure requires a process restart and the independent firmware/operator stop.
- No automatic serial writes at startup or inactive shutdown. Closing the serial transport emits no centering/release commands.
- On loss of engagement/data after commands have been issued, hold the last transmitted steering and request zero speed in speed-hold mode. The stop state remains latched. On normal shutdown, attempt this stop intent for up to four seconds before closing. This is **not proof of standstill** and cannot deliver braking through a failed USB link; firmware timeout braking and operator brake/estop remain required. There is no obstacle avoidance.

## Logs and checks

`/bev_dbw/bridge_status` reports readiness, restart-required stop/transport states, mode, last output and the preserved first `stop_reason`. A `stop_latched` log event records the original cause, input/source/engagement ages and configured timeouts before later messages can overwrite the input-gate reason; the bridge also prints it as a `STOP LATCHED` warning. Session `events.jsonl` also records accepted/rejected inputs, engagement and generated commands. A log write failure disables positive commands. Logs and `hardware.local.json` are ignored by Git.

Find the cause of the most recent stop:

```bash
grep stop_latched /home/agxthor/SingleCameraBev/bev_bridge/runs/all_*/bridge/events.jsonl | tail -1
```

`invalid_tracking_valid` means the tracker itself invalidated its output (for example off route by more than 5 m or more than 60 degrees of heading error); the bridge log does not record which tracker check failed. Per-scan GPS-versus-LiDAR offsets for a run-all session are written on exit to `view/localization/lidar_localization.json`.

```bash
source /opt/ros/jazzy/setup.bash
cd /home/agxthor/SingleCameraBev
/usr/bin/python3 -m unittest discover -s bev_bridge/tests -v
```

The isolated ROS check uses only dry-run children and test topics:

```bash
ROS_DOMAIN_ID=96 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  /usr/bin/python3 bev_bridge/tests/check_ros_dry_run.py \
  --output bev_bridge/runs/NEW_ros_dry_check
```

For the original camera/LiDAR replay rejection check, also source the BEV workspace and its Cyclone DDS config, then run `tests/check_recorded_replay.py --output NEW_DIRECTORY` in isolated domain 95. It starts no physical sensors and opens no serial ports.

The **2026-10-15** unit run completed **32 tests: 31 passed, one configuration error**. The error is the external profile's 300-degree steering request exceeding its mode's 90-degree limit; it is not a serial or moving-vehicle test. The suite covers the 10 mph boundary, lower/zero target preservation, serial rejection above the cap and preservation of the first stop cause. Earlier checks passed 24 isolated ROS checks and rejected 47 actual recorded-replay targets with zero bridge commands. The isolated ROS check explicitly retains its 1 mph diagnostic limit. No vehicle actuation is performed by these tests.

See [PID signal summary](../camera_lidar_bev/docs/pid_integration_summary.md) and [path tracking](../camera_lidar_bev/docs/path_tracking.md) for the upstream fields and their limitations.
