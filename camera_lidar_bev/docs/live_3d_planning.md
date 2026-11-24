# Live 3D BEV: detection, landmarks, prediction and planning

One 3D view of the localized car with CenterPoint boxes, predicted paths of moving vehicles and
pedestrians, the mapped route ("navtraj"), the kept stop signs and stop lines, and a planned trajectory.
Everything here **plans, displays and records only**: no tool publishes a vehicle command or talks to the
[DBW bridge](../../bev_bridge/README.md). Connecting the planner to the bridge is a separate step that has
not been done.

## Start

Start the 20 Hz LiDAR first, in its own terminal, and leave it running:

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash && source /home/agxthor/sensors_ws/install/setup.bash && source install/setup.bash
export ROS_DOMAIN_ID=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://$PWD/config/transport/cyclonedds_large_clouds.xml
/usr/bin/python3 tools/start_lidar.py --params-file config/sensors/ouster_baseline.yaml --startup-timeout 90
```

Then, in a second terminal:

```bash
bash /home/agxthor/SingleCameraBev/camera_lidar_bev/tools/run_bev_3d.sh
```

[`run_bev_3d.sh`](../tools/run_bev_3d.sh) starts, in order, and reuses anything already running:

| Step | Part | Tool |
| --- | --- | --- |
| 1 | Ouster LiDAR, 1024x20 (if not already running) | `tools/start_lidar.py` |
| 2 | Saved-map localization; waits up to 2 min for `localized` | `tools/run_localization_compare.sh --live --start-sensors --max-rate 20` |
| 3 | CenterPoint 3D detection, 3 sweeps | `ros2 launch bev_bringup lidar_vehicle_detector.launch.py model:=centerpoint` |
| 4 | TF `enu -> os_sensor`, route, kept landmarks | `tools/localization_rviz_bridge.py` |
| 5 | Object tracking and 4-5 s prediction | `tools/object_prediction.py` |
| 6 | Route planner and drive recording | `tools/route_planner.py` |
| 7 | Measured-vs-planned charts | `tools/planner_plots.py` |
| 8 | 3D BEV scene and RViz | `tools/bev_scene_markers.py`, `config/rviz/bev_planning_3d.rviz` |

Options: `--no-plots`, `--lanes` (lane-following prediction), `--map`. Ctrl+C stops only what the script
started; each part logs to `data/bev3d_<UTC>/`, and a part that exits prints its last log lines.

**Start the LiDAR before localization.** `run_localization_compare.sh --start-sensors` starts the Ouster in
**2048x10 (10 Hz)** when no driver is running; it reuses an existing driver, so starting
`start_lidar.py` with `ouster_baseline.yaml` first gives 1024x20. Measured on 2026-11-24 with all parts
running: LiDAR 20.0 Hz, localization 17.9 Hz, CenterPoint boxes 17.2 Hz.

## The 3D view

`config/rviz/bev_planning_3d.rviz` (fixed frame `os_sensor`, orbit view behind the car), fed by
[`bev_scene_markers.py`](../tools/bev_scene_markers.py):

- **Objects:** see-through 3D boxes with a heading tick, labelled with class and distance. Vehicles (car,
  truck, bus) violet, pedestrians amber, cyclists/motorcycles pink, other classes (barrier, cone) grey.
  Objects inside the planned path are drawn brighter and marked `in path`, or `LEAD` for the vehicle the
  planner follows.
- **Predicted paths** ([below](#object-tracking-and-prediction)): a fading line in the object's colour,
  2-sigma discs every second, a see-through box at the end of the horizon, `#id speed` or `parked` / `still`.
- **Planned trajectory:** a ribbon wider than the route, coloured by planned speed (green full speed,
  through yellow, to red stopping), with a tick every second.
- **Route (navtraj):** a thin blue-grey line on the road.
- **Stop signs and stop lines:** a red octagon on a pole and a white bar on the road for each kept stop
  sign; the stop line the planner is heading for gets a see-through red wall, `STOP 18 m (armed)` or `HOLD`.
- **Traffic lights are not drawn** in this view.
- Planner status text behind the car, e.g. `CRUISE 5.3->4.5 m/s (3 s)`.

Limits: RViz spaces out 3D text on this machine; a stopped plan has no ribbon (all points coincide).

The older live view `config/rviz/lidar_boxes_3d.rviz` remains: raw scan, CenterPoint boxes, route, kept
landmarks and the planner's own speed-coloured band (1.8 m wide). The driven path is a toggle there.
`config/rviz/landmarks_map.rviz` is a map-fixed (`enu`) overview of the whole route with the saved map,
route and enlarged landmarks; it needs only the RViz bridge.

## CenterPoint 3D detection

`model:=centerpoint` runs CenterPoint-PointPillar (OpenPCDet, nuScenes weights, 10 classes, +-51.2 m) from
raw Ouster scans in native C++/TensorRT FP16 ([configuration](../config/perception/lidar_centerpoint_detector.yaml)).
It merges the last **3 sweeps** with the nuScenes time-lag feature, placing older sweeps with localization
poses; the newest sweep's rotation comes from the Ouster gyro (bias estimated online) because localization
arrives after the scan. Per-class score thresholds start higher for rare classes.

[Offline on the 2026-10-10 drive](../evaluation/centerpoint_multisweep_replay_001.txt) (no ground truth;
consistency proxies), 1 vs 3 sweeps: boxes present in the previous and next scan 47.3 % -> 55.8 %,
isolated boxes 28.3 % -> 21.2 %, vehicle position jitter 0.146 -> 0.125 m, detector time 16.3 -> 40.4 ms
median. Ouster-domain accuracy, distant recall and size/yaw accuracy remain unvalidated.

The model also regresses a per-box velocity (`vx, vy`, channels 8-9 of the head), which the detector
currently ignores; see [prediction](#what-limits-prediction-and-next-steps).

## Stop signs, traffic lights and stop lines on the map

1. **Mapping drive:** with cameras running YOLO and localization, [`tools/map_landmarks.py`](../tools/map_landmarks.py)
   locates each camera `stop sign` / `traffic light` box in 3D from the LiDAR points inside it, converts it
   to the map frame, and merges sightings within 2.5 m. A landmark is confirmed after 3 sightings. Output:
   `maps/<map>/landmarks/landmarks.json`. One loop on 2026-11-24: 1,095 detections, 45 landmarks.
2. **Review:** [`tools/review_landmarks.py`](../tools/review_landmarks.py) serves a page on
   `http://127.0.0.1:8090/` (this machine only) with the landmarks on a top-down LiDAR map; click or tick
   to keep / drop, then **Save** writes `landmarks_reviewed.json` (the original is never changed). Kept for
   `local_gps_recording_002`: **stop signs #0 and #40, traffic lights #2, #3, #24, #25, #33**.
3. **Use:** the RViz bridge, planner and BEV view read only the kept entries.

**Stop lines are estimated**, not detected paint: a line across the road at the mapped-route point nearest
each stop sign. Treat them as approximate until checked against the painted line.

## Front-camera traffic-light check

[`tools/traffic_light_camera_view.py`](../tools/traffic_light_camera_view.py) (run with `../.venv/bin/python`)
runs `models/traffic_light_state_fp32.engine` (YOLO that detects a signal head and classifies red / yellow
/ green in one pass) on `/camera/front_wide/image_raw`, projects the kept mapped lights into the image, and
logs `data/traffic_light_eval_<UTC>/frames.jsonl` and `summary.json` (`--record` saves video).

About 2.5 minutes of driving on 2026-11-24 (front-wide, 1920x1080, 30 fps, model 8 ms median):

| Mapped light distance | Detected with a colour when in view |
| --- | --- |
| 0-20 m | 88 % (106/120 frames) |
| 20-40 m | 34 % (163/475) |
| 40-60 m | 20 % (206/1,051) |
| 60-100 m | < 2 % |

Only 14 % of the generic COCO `traffic light` boxes got a colour. The model sees the whole frame
downscaled to 640 px, so a typical 55 px signal head shrinks to about 18 px. Running it on full-resolution
tiles of the upper image, or inside the COCO boxes, is the likely fix (not done). "In view" ignores
occlusion and lights facing cross traffic, so the rates are a lower bound. The narrow front camera
(41 degrees, port 4.1.3.2), better suited to traffic lights, was not detected on USB on 2026-11-24.
**The planner does not use traffic lights.**

## Route planner (shadow mode) and drive recording

[`tools/route_planner.py`](../tools/route_planner.py) re-plans on every localization pose (~18 Hz) from
the measured state, over 8 s at 0.1 s:

- **Path:** the saved route, starting at the car and blending onto the route over max(8 m, 2 s of travel).
- **Speed:** min(cap, sqrt(lateral-acceleration limit / curvature)), reachable with comfortable braking.
  The cap is the tracker's **4.4704 m/s (10 mph)** (`--max-speed-mps`); lateral limit 1.5 m/s^2.
- **Stop lines:** a full stop with the front bumper 0.5 m before each kept stop line, held 2 s, then released;
  re-armed when far behind it again. Driving through without stopping is logged as `passed`.
- **Front vehicle:** CenterPoint objects in the lane corridor ahead (1.6 m half-width around the route)
  are followed with IDM: 4 m standstill gap plus 1.8 s time gap, lead speed tracked along the route.
- **Braking:** comfortable (1.5 m/s^2) for the cap and curves; up to 4 m/s^2 only for stop lines and the
  lead vehicle. Acceleration limit 1 m/s^2.
- **Kinematics:** vehicle frame at the rear axle (1.25 m behind the LiDAR); planned vy is 0 (no slip),
  planned ay = v^2 x curvature. Measured values: linear fit of the last 0.5 s of localized rear-axle positions
  for velocity, quadratic over 1 s for acceleration, plus the Ouster IMU with gravity removed (0.2 s low-pass).

Closed-loop simulation (separate ROS domain, simulated car following the plan with 0.3 s actuator lag):
stops with the front bumper at the line and releases after 2 s; stops 3.8 m behind a stopped car (4 m
target); settles about 7 m behind a 2 m/s car at its speed; 3 ms per plan. Not measured on the car under
control, and there is no wheel-speed or CAN feedback.

**Charts** ([`tools/planner_plots.py`](../tools/planner_plots.py)): last 30 s of vx, vy, ax, ay, measured
(localization cyan, IMU magenta) against the plan's target (yellow, speed 1 s ahead), and the newest plan's
speed / ax / ay over its horizon.

**Recording:** every planner run writes `data/drive_<UTC>/`:

| File | Contents |
| --- | --- |
| `drive.csv` | one row per cycle: pose, route progress, lateral / heading error, measured vx vy ax ay and yaw rate, IMU ax ay, planned speed / acceleration now, +1 s, +3 s, mode, stop-line distance and state, lead class / gap / speed / desired gap, plan time |
| `plans.jsonl` | every full 8 s plan (t, x, y, v, ax, ay) |
| `metadata.json` | parameters, map, stop lines, vehicle geometry, git commit |
| `bag/` | MCAP: localization odometry and status, `/imu`, `/tf`, 3D detections, planner trajectory and state, object predictions; `--bag-points` adds raw `/points` (large). About 20 MB per minute without points. |

## Object tracking and prediction

[`tools/object_prediction.py`](../tools/object_prediction.py) places each CenterPoint box in the map frame with
the localization pose of the same scan (removing the car's motion), tracks it, and predicts **5 s for
vehicles and cyclists, 4 s for pedestrians**. Models are in [`tools/prediction_models.py`](../tools/prediction_models.py):

- **Vehicles:** IMM filter (straight + coordinated turn) on position, speed, heading and turn rate, also
  using CenterPoint's box heading; predicted turns stop after 90 degrees.
- **Cyclists and pedestrians:** constant-velocity Kalman filter (the box heading made cyclists worse).
- Association by Mahalanobis gate + Hungarian per class group; confirmed after 3 matches, dropped after 1 s.
- Tracks slower than 1.5 (vehicle), 1.0 (cyclist), 0.5 (pedestrian) m/s are shown as parked / still.
- `--lanes` predicts vehicles along the mapped route when they drive along it.
- Publishes `/camera_lidar_bev/prediction/markers` and `/camera_lidar_bev/prediction/objects` (JSON).

**Benchmark** ([`tools/eval_prediction.py`](../tools/eval_prediction.py)) on the recorded 24-minute drive
`data/drive_20261124T213905Z` (28,135 frames): ground truth is where each object was actually detected
afterwards (one reference tracker for all methods). First half for tuning, second half held out.
Median error in metres (moving objects):

| Object (held-out second half unless noted) | cv (first version) | **imm (live default)** | imm + lanes |
| --- | --- | --- | --- |
| Vehicle, straight, 3 s | 1.92 | **1.54** | 1.77 |
| Vehicle, all, 5 s | 3.69 | **2.36** | 2.85 |
| Vehicle, average over 5 s (ADE) | 1.28 | **1.15** | 1.17 |
| Vehicle, turning, 3 s (first half; 50-54 samples) | 6.70 | 6.62 | 5.82 |
| Vehicle, straight, 3 s (first half) | 3.95 | 3.65 | 3.70 |
| Cyclist, 3 s | 0.87 | 0.87 | 0.87 |
| Pedestrian, 3 s / 4 s | 0.40 / 0.55 | 0.40 / 0.55 | 0.40 / 0.55 |

Moving vehicles predicted to stay put: 13 % (first half), 5 % (second half); parked vehicles predicted to move:
about 1 %. Full numbers: [prediction_benchmark_drive_20261124T213905Z.json](../evaluation/prediction_benchmark_drive_20261124T213905Z.json).
The first half has more moving traffic and larger errors than the second. On a synthetic steady turn
(6 m/s, 0.2 rad/s) IMM is 2.2 m off at 3 s, against about 5 m for a straight-line guess. The replay data, the
benchmark and the models run on any PC without ROS: see [prediction development](prediction_development.md).

### What limits prediction and next steps

On moving vehicles at 3 s the error is almost all **along the direction of travel** (median 3.4 m) rather than
sideways (0.35 m):

- **New tracks:** in a track's first 10 updates (~0.5 s) the speed error is 2.7 m/s and the 3 s error 8 m,
  because a new track starts at zero speed. Decoding CenterPoint's own velocity output would give a speed
  from the first frame.
- **Braking / accelerating:** about +-5.3 m along-track at 3 s; a constant-speed model cannot anticipate it.
- Steady speed: 2.3 m median at 3 s. Turning vehicles are few in this drive (about 50 samples).

Lane markings mainly help turns and lane assignment, not the speed error. A learned predictor needs more
recorded drives (the recordings label themselves: the real future is in the later frames) and is not started.

## Lane markings on the map (in progress)

[`tools/extract_lane_markings.py`](../tools/extract_lane_markings.py) finds road paint in the saved map from
its LiDAR intensity (paint is retro-reflective) and writes
`maps/local_gps_recording_002/lane_markings/lane_markings.json` and a preview image. First pass: 264
markings, about 1.4 km (lane, edge and centre lines, crosswalk stripes, stop-line candidates). Centre lines
and crosswalks look good; short lane/edge fragments and most stop-line candidates are noise. Cleanup and an
RViz publisher are being finished; the markings are not in the 3D view yet.

## Tree removal (optional, off)

[`tools/remove_map_trees.py`](../tools/remove_map_trees.py) marks map voxels whose points scatter through a
volume (leaves) rather than forming surfaces, more than 1 m above the local ground, keeping columns around
kept landmarks (39 % of the saved map's points). `localization_rviz_bridge.py --remove-trees` then publishes
the map and live scans without trees. The live views show the raw scan; localization and detection always
use the full data.
