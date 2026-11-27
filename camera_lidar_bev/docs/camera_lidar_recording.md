# Record camera and LiDAR test data

Use a five-minute sensors-only recording to collect several outdoor scenes before running detector evaluation. Keep the camera/LiDAR mounting unchanged and the rig stationary for the first session. The command below is for you to run after positioning the rig; preparing these tools does not start a capture.

## Record for five minutes

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
bash tools/collect_camera_lidar.sh --duration 300 --scene 'multi_scene_test'
```

The wrapper sources ROS and the project overlays, starts or reuses the calibrated front camera at native 1024 × 768 and a requested 30 FPS, and starts or reuses the Ouster in `1024x20` mode with the original point-cloud fields. It records the sensor topics for 300 seconds. It does not require detector engines or RViz and starts no detector or viewer. Ctrl+C stops the recording early; on exit, sensors started by this invocation stop, while reused sensors remain running.

## Record dense 10 Hz LiDAR scans

Stop any existing LiDAR/BEV launcher with Ctrl+C before selecting another scan mode. Then run:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
bash tools/collect_camera_lidar.sh --lidar-mode 2048x10 --duration 300 --scene '10hz_test'
```

This selects `config/sensors/ouster_2048x10.yaml`: 10 scans per second, 2048 horizontal columns and 128 vertical channels. It records the front camera at its existing 30 FPS setting. The selected driver profile is copied into the session context and used for startup; the usual live profile stays at `1024x20`. The collector checks the selected mode against the cloud dimensions, sensor metadata and active driver parameters before recording. It rejects an existing driver in another mode instead of recording the wrong scan profile. The saved-data cadence checker receives the requested 10 Hz rate.

Each recording prints its new `bags/camera_lidar_<UTC timestamp>` path. Ctrl+C stops early and saves the partial recording. Use `--duration` for another length. `--dry-run` prints commands without starting sensors or creating a recording.

## Record 10 Hz LiDAR with GPS

For the next mapping session, use the passive Septentrio GPS stream together with the camera, original LiDAR scans and Ouster IMU:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
ROS_DOMAIN_ID=1 bash tools/collect_camera_lidar.sh \
  --lidar-mode 2048x10 --gps --duration 300 --scene '10hz_gps_mapping'
```

Keep the rig still for approximately **10 seconds after recording starts**, then collect the route, including a return along overlapping scenery for later map comparison. This gives the LiDAR–IMU mapper a quiet initialization interval. Keep the sensor mounting fixed and record scene/route timing in the saved notes.

`--gps` starts or reuses `septentrio_gps.py` in the same ROS domain. It passively reads the receiver's existing SBF TCP stream at `192.168.3.1:28784`; it sends no receiver setup commands. The collector waits for a fresh usable GPS fix before starting the bag. If no fix is available, it prints the reason while waiting; provide an unobstructed outdoor sky view. Add `--startup-timeout 120` if the receiver needs longer to acquire a fix. Camera/LiDAR collection without `--gps` retains its previous behavior.

The four additional recorded topics are:

| Topic | Content |
| --- | --- |
| `/sensing/gnss/fix` | `NavSatFix`: position, fix status and position covariance |
| `/sensing/gnss/status` | Receiver solution mode/error, RTK status and original GNSS week/TOW |
| `/sensing/gnss/raw_sbf` | Original received SBF blocks for later inspection |
| `/sensing/gnss/diagnostics` | Connection, parsing and freshness status |

The session context includes the GPS configuration with its SHA-256, `gps_runtime.yaml` and `gps_preflight.json`. The latter records the actual fix, covariance, antenna frame, source/receipt ages and available receiver status. `--gps-config /path/to/passive_gps.yaml` selects another passive bridge configuration. Its selected file is copied into the context and used to start the bridge.

For an explicit diagnostic capture without a usable position, add **`--allow-no-gps-fix`** together with `--gps`. This permits recent no-fix or unusable-coordinate messages, while still requiring fresh GPS headers and message receipt. The preflight and manifest retain `valid_fix: false`; this mode must not be interpreted as a usable GNSS trajectory.

GPS headers use **host ROS receipt time** and `gnss_antenna`; raw receiver week/TOW is retained separately. The antenna-to-LiDAR/vehicle lever arm is unmeasured, and no antenna TF is invented. A reported SBAS solution is not RTK fixed and does not establish centimeter accuracy. Receiving a valid fix does not validate shared acquisition timing, global map alignment or the entire recorded route's GPS quality. Inspect the recorded receiver status/covariance across the route. The existing offline bag checker covers camera/LiDAR consistency; the GPS preflight describes startup quality rather than continuous GNSS accuracy.

The default bag path is `bags/camera_lidar_<UTC timestamp>` and is printed at startup. Each session uses a new output path. MCAP is preferred when its ROS storage plugin is installed, using `zstd_fast` lossless chunk compression. SQLite3 is the uncompressed fallback. A sibling `<bagname>.capture/` directory stores configuration/calibration snapshots, scene notes, `sensor_params.yaml`, `run_manifest.json` and the quality report. The recorder's sibling `.session.json` and `.recorder.log` are retained too. Bags and their companions remain local and are excluded from Git.

The recorder now defaults to a 1 GiB cache (`--cache-mib 1024`), using roughly 2 GiB for its two payload buffers. This increases buffering for brief file-writing stalls. The first extended recording at the earlier 256 MiB setting logged 143 cumulative cache losses, including 23 images and 16 scans; a later recording must verify whether the larger buffer resolves the loss. The checker reads those recorder counters as well as the saved message headers. Identical cumulative warnings printed at each file split are not separate loss events.

Five minutes is **about 60 GB uncompressed**: roughly 6 MB per original LiDAR scan at 20 Hz plus 2.36 MB per raw BGR camera frame at 30 FPS. After the user's cleanup on 2026-10-09, the workspace had approximately 79 GiB free. The collector checks the available space again when run. Actual compressed size depends on the scene, point layout and measured rates. It stops the recorder cleanly when free space approaches its 2 GiB floor; inspect the actual duration and stop reason afterward. Add `--output /path/on/larger/disk/NEW_SESSION` to select another disk. Existing paths are refused. The existing [LiDAR bring-up guide](lidar_bringup.md) describes the temporary DDS buffer settings needed after a reboot.

## Scenes to include

Use the scene label to identify the session and keep a note of approximate elapsed times for each scene. A useful first set includes:

- Empty ground/background, then vehicles and people visible in both sensors.
- Objects at approximately 5, 10, 20, 30 and 50 m where they are visible; record measured distances where possible.
- Front, side and angled views, plus partially occluded examples and nearby background objects.
- Changes in lighting or shadow, with a few unobstructed views for comparison.

Record notes about object type, distance, angle, visibility in each sensor and lighting. Keep the mounting fixed between scenes. The five-minute duration is the total recording time, so prepare the scenes and move between them within that interval. Separate sessions can use different `--scene` labels for easier evaluation.

## Validate the saved session

Inspect the collection/validation reports rather than treating a recorder exit as proof of complete data. The bag checker reports saved topic counts/rates, gaps, image dimensions, exact camera Image/CameraInfo header pairing, calibration types and point fields. Nominal five-minute counts are approximately 9,000 camera images and 6,000 LiDAR scans, but measured bag counts and timing determine completeness.

The collector runs the offline checker after a completed recording and prints the report paths. A five-minute `2048x10` session nominally contains 3,000 LiDAR scans. To rerun the checker without reconnecting sensors, replace `BAG_PATH` with the printed bag directory and choose a new report name; for a 10 Hz recording also add `--lidar-hz 10`:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
/usr/bin/python3 tools/check_camera_lidar_bag.py BAG_PATH \
  --calibration BAG_PATH.capture/front_camera_1024x768.candidate.yaml \
  --output BAG_PATH.capture/quality_recheck.json
```

The checker reads the bag offline and starts no ROS publishers or playback. It refuses an existing report path; without `--output`, its default report is `BAG_PATH.qc.json`. Review missing topics, invalid messages, camera metadata/header mismatches and gaps before using the recording for detector comparisons. A partial or disk-limited recording needs the same review, with its actual duration recorded in the scene notes.

Sensor timestamps are preserved in the bag. The current camera timestamps are host ROS time after capture returns, and the LiDAR uses the configured driver timestamp convention. Their relationship to exposure/acquisition time and to one another remains unverified. Exact Image/CameraInfo pairing and close camera/LiDAR headers establish software consistency, not physical sensor synchronization. Calibration snapshots preserve the settings used; they do not validate extrinsics, metric accuracy or a motion trajectory.
