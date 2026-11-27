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

## HD map in the 3D view

The HD map of 2026-11-25 (`hdmap_20261125_125612_20261125_1820`, its own LiDAR frame, no GPS) is aligned to the
saved map's `enu` frame by `tools/align_hdmap.py` (route ICP, then paint-to-paint 6-DoF ICP; median residual 0.22 m),
written to `maps/local_gps_recording_002/hdmap/hdmap3d_enu.json`. `tools/hdmap_publisher.py` (started by
`run_bev_3d.sh` when that file exists) publishes it latched on `/camera_lidar_bev/hdmap/markers`: kerbs red, lane
centres green (mapped drive direction) / orange (against it) / magenta (intersection turns), lane lines white /
yellow, crosswalks, stop lines and intersection outlines. Display "HD map" in `bev_planning_3d.rviz`; it follows the
car through the localization TF. `--layers` picks layers (`lane_boundary` is off by default). Known gap: stop sign
#40 is 21.7 m from the nearest HD-map stop line.

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
- **Front vehicle:** CenterPoint objects are tracked (constant-velocity Kalman filter; a track counts after
  detections in consecutive frames and is kept through dropouts of up to 1 s). The closest track in the lane
  corridor ahead (1.6 m half-width around the route) is followed with IDM: 4 m standstill gap plus 1.8 s time
  gap, its speed from the track (minus one standard deviation, so a new or uncertain track counts as slower).
- **Crossing and cutting in:** a moving object whose predicted path (constant velocity, 4 s; pedestrians 3 s)
  enters the corridor ahead is yielded to as if stopped where it enters, unless the car is through 1 s before it
  arrives. It must move across the lane clearly faster than its velocity uncertainty (0.5 m/s, pedestrians 0.4),
  for 0.4 s (pedestrians, cyclists) or 0.1 s (vehicles). The status line shows `yield: <class> entering lane in <t> s`.
- **Braking:** comfortable (1.5 m/s^2) for the cap and curves; up to 4 m/s^2 for stop lines and the lead
  vehicle, and up to 6.5 m/s^2 (`--emergency-decel-mps2`) only when 4 m/s^2 cannot stop in time.
  Acceleration limit 1 m/s^2. `--no-track-objects` restores the first version (raw detections, no tracking,
  no yielding).
- **Kinematics:** vehicle frame at the rear axle (1.25 m behind the LiDAR); planned vy is 0 (no slip),
  planned ay = v^2 x curvature. Measured values: linear fit of the last 0.5 s of localized rear-axle positions
  for velocity, quadratic over 1 s for acceleration, plus the Ouster IMU with gravity removed (0.2 s low-pass).

The decisions (stop lines, front vehicle, yielding, speed plan) are in
[`tools/planner_core.py`](../tools/planner_core.py) without ROS, so the simulation below runs the same code as
the car. About 5 ms per plan on the Thor.

### Traffic simulation (safety test)

[`tools/simulate_planner.py`](../tools/simulate_planner.py) drives the mapped route closed loop: the planner runs
at 20 Hz, the car follows the plan's acceleration with a 0.3 s lag, and simulated road users are seen with 0.1 s
latency, 0.15 m noise, 10 % missed frames and occasional 0.2-1 s losses (occlusion). Ten scenarios with random
parameters along the whole route: slower lead, lead braking hard (3-8 m/s^2), stopped car, cut-in, parked car
pulling out, crossing pedestrian, crossing car, oncoming and parked cars (no braking wanted), stop-and-go lead, and
all at once; plus `replay`, the recorded traffic of a drive. A collision is **avoidable** if, when the object first
got in front of the car, stopping relative to it after 0.3 s needed at most 6.5 m/s^2; the others are cut-ins and
pull-outs right beside or a few metres ahead that braking cannot prevent.

```bash
python3 tools/simulate_planner.py --episodes 40                       # all scenarios, ~30 s on the Thor
python3 tools/simulate_planner.py --max-speed-mps 8.94 --label 20mph   # faster cap
python3 tools/simulate_planner.py --track-objects false --label old   # first version, for comparison
python3 tools/simulate_planner.py --scenarios cut_in --episode-seed 3021 --trace trace.csv   # replay one episode
```

400 episodes (40 per scenario), results in `evaluation/planner_sim_*.json`:

| | 10 mph cap, first version | **10 mph, tracked (live default)** | 20 mph, first version | 20 mph, tracked |
| --- | --- | --- | --- | --- |
| Collisions (all / avoidable) | 50 / 7 | **20 / 0** | 69 / 9 | 29 / 3 |
| Lead brakes hard / stopped car: collisions, time to collision < 1.5 s | 3, 14 | **0, 0** | 3, 25 | 0, 0 |
| Crossing pedestrian / car collisions | 12 / 11 | **1 / 0** | 9 / 18 | 1 / 1 |
| Cut-in / pull-out collisions (tracked: none avoidable at 10 mph) | 10 / 9 | 10 / 9 | 12 / 17 | 11 / 15 |
| With 5 % ghost detections: episodes with hard braking (> 3 m/s^2), oncoming + stop-and-go | 0 | 4 | | |

What the simulation found in the first version and what changed:

- **A missed detection forgot the car ahead.** One missed frame (CenterPoint misses some) set the lead to none and
  the plan accelerated, even towards a stopped car seen 0.1 s before; it reached it after a few flickers.
  Tracks now coast through dropouts.
- **A newly seen car counted as stopped** (its speed came from differencing two frames), so every cut-in caused
  hard braking; the speed now comes from the track.
- **Crossing objects counted only once inside the lane**, too late at the crossing; they are now yielded to from
  their predicted path.
- **Braking was capped at 4 m/s^2** even when that could not stop in time; harder braking is now allowed then.
- Scattered false detections chained into a "confirmed" obstacle in the first tracker version (hard
  braking, 31 of 80 episodes): tracks now need consecutive frames and young tracks expire after 0.3 s.

On the real 20-minute drive `drive_20261126T013444Z` (replayed open loop, the car where it really drove) the
tracked planner had a lead in 2,261 frames (first version 1,703) and yielded 22 times (219 frames), 9 of them to
objects that did enter the lane within 5 s; the other 13 were mostly pedestrians and two-wheelers near the
road that did not (precision 41 %, after tightening from 11 of 51). Closed-loop `replay` of real traffic is only
a progress / false-braking check (the replayed cars do not react, and cars that were following the recording car
end up in front of a slower simulated car): both versions made the same progress (106 m per 30 s).

### Traffic lights (simulation)

The planner knows the three signalized intersections on the route: the five kept traffic lights grouped by the
nearest HD-map intersection (`planner_core.load_signals`), each with a stop point at the HD stop line the route
crosses, else 1 m before the route enters the intersection: intersection 1 (lights #2, #3) at route 263 m,
2 (#24, #25, the far-side heads 20 m off the route) at 400 m, 3 (#33) at 625 m. Rules: a colour is accepted after two
alike in 0.5 s (one wrong frame is ignored) and held 2 s; **red** is a stop target (front bumper 0.5 m before the
line); **yellow** stops if 3 m/s^2 is enough (`--yellow-max-decel-mps2`), else goes through, decided once per
yellow; **green** goes; a light **not seen** is ignored (`--unknown-light go`) or a stop within 30 m (`stop`). On the
car there is no colour input yet, so live the signals stay unknown and change nothing; the status line shows
`light #<id> <distance> m (<state>)`.

`python3 tools/simulate_planner.py --scenarios light_red light_green light_yellow light_cycle light_red_queue --duration-s 45`
simulates red then green, always green, yellow when the car is 0.5-6 s from the line, a random fixed-time cycle, and
red with 1-3 stopped cars queued at the line. Colours are seen like the measured front camera: 88 / 34 / 20 % per
frame at 0-20 / 20-40 / 40-60 m and 3 % wrong colour (`--light-perfect` for always right). 200 episodes per row
(40 per scenario), results in `evaluation/planner_sim_lights_*.json`, example episodes in
[planner_sim_lights.png](../evaluation/planner_sim_lights.png):

| | Red-light runs | Yellow: stopped / went through | Stops on green | Hard braking episodes |
| --- | --- | --- | --- | --- |
| 10 mph, unseen = go | 0 | 56 / 7 | 1 | 2 |
| 10 mph, unseen = stop | 0 | 56 / 7 | 1 | 2 |
| 20 mph, unseen = go | **3** | 38 / 12 (1 ran red) | 2 | 42 |
| 20 mph, unseen = stop | 0 | 40 / 11 | 2 | 11 |
| 20 mph, colours always seen | 0 | 42 / 9 | 2 | 14 |

Stopped at red: front bumper 0.3 m before the line (median); moves 1.5 s after green (colour filter + 0.3 s
actuator lag); in a queue it stops about 9.5 m back behind the cars and leaves 4.4 s after green. All three red
runs were at intersection 2, whose lights are far-side heads 20 m off the route: from 20 mph the camera recognised
red only about 12 m before the line, too late to stop comfortably, so the car stopped about 1 m past it. Treating an
unseen light as a stop removes them and also halves hard braking at 20 mph, for slightly less progress; at 10 mph
the camera range is enough either way. Before using lights live: publish the front-camera colour per signal, and use
`--unknown-light stop`.

**Watch it live:** `bash tools/run_sim_live.sh [--scenarios light_yellow light_red light_red_queue light_cycle]
[--max-speed-mps 8.94] [--unknown-light stop] [--rate 2]` plays the same simulated episodes in real time in the 3D view
with predictions and charts (own ROS domain 78, [`tools/sim_live.py`](../tools/sim_live.py)): each signal as a pole
with its true colour (big lamp) and what the planner believes (small lamp, grey = not seen), its stop line red when
the plan stops for it and green when it goes, `STOP` / `GO` with the distance, the true road users as thin outlines
over the detected boxes, and a status line (scenario, light, decision, crossing colour, collisions). Each episode's
result is printed in the terminal. Any traffic scenario works too (`--scenarios cut_in crossing_ped ...`).

**Replay a recorded drive through the current planner** (3D view and charts, own ROS domain 77, nothing from the
car's domain): `bash tools/run_planner_replay.sh [--bag data/drive_<UTC>/bag] [--start-s 870] [--duration-s 300]`
(`--old-planner` for the first version). Without ROS, `python3 tools/compare_planner_replay.py` runs both versions
over a whole exported drive and writes `evaluation/planner_replay_<drive>.json` and a speed / gap / mode chart. On
`drive_20261126T013444Z`: sudden one-frame plan drops (a flickering detection) 36 -> 11, plan jumps > 2 m/s 161 -> 72,
14 yield events (196 frames), same planned speed against driven (median +1.0 m/s: you drove above the 10 mph cap).

Limits: the simulated traffic does not react, there is no lateral planning (the plan only brakes), and
the car model is a speed lag, not vehicle dynamics. Nothing here has been run with the planner in control of the
car.

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
- **Vehicles follow the HD map lanes** (rule-based, [`tools/hdmap_lanes.py`](../tools/hdmap_lanes.py); on by
  default when `MAP/hdmap/hdmap3d_enu.json` exists, `--lanes route` for the older mapped-route following,
  `--no-lanes` for the filter alone): the lane is matched by position, heading and whether its turn over the
  next 15 m fits the car's turn rate; at forks the branch matching the turn rate is taken; mapped stop lines give
  a stop (braking from 1.5 m/s^2 needed, 1.5 s wait, then 1.5 m/s^2 away), and the offset from the lane centre fades
  out over 1.5 s. A curve speed limit made predictions worse on both drives and is off.
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

**HD map rules** (`imm+hd`, vehicles only; cyclists keep `imm`, the lane centre made them worse), median vehicle
error in metres on both recorded drives (first half / second half):

| Drive, half | 3 s straight, imm | 3 s straight, **imm+hd** | 5 s all, imm | 5 s all, **imm+hd** | turning 3 s, imm / imm+hd |
| --- | --- | --- | --- | --- | --- |
| 2026-11-24, first (busy, 54 turns) | 3.65 | **3.58** | 7.97 | **7.51** | 6.62 / **5.93** |
| 2026-11-24, second | **1.54** | 2.02 | **2.36** | 2.93 | 3 samples |
| 2026-11-26, first | 2.64 | 2.64 | **4.97** | 5.04 | 1 sample |
| 2026-11-26, second | 3.07 | **2.91** | 4.69 | **4.16** (mean 6.94 vs 8.30) | - |

Better in the busy and turning cases and on the new drive's held-out half, worse on the quiet second half of the
first drive. The error is still mostly speed (along the lane, about 2.1 m at 3 s against 0.4 m sideways), which a
map cannot fix. Building it found and fixed: matching a car that had passed a lane's end to that lane (3 m short),
putting cars going straight on the turn lane that starts at the same place (27-42 m off), the zigzag in the mapped
centre lines (shortened paths and fake curvature), and stop lines attached to lanes that only pass near them.
Diagnose with `python3 tools/eval_prediction.py --dataset datasets/prediction/drive_20261126T013444Z.npz`.

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
