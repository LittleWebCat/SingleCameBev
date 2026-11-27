# Real-time lanes, road boundaries and lane centre: camera vs LiDAR test design

**Built (2026-11-25), live only, camera vs LiDAR, no map.** Run on the car (LiDAR and narrow front camera):

```bash
cd /home/agxthor/SingleCameraBev/camera_lidar_bev
source /opt/ros/jazzy/setup.bash && source install/setup.bash
export ROS_DOMAIN_ID=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file://$PWD/config/transport/cyclonedds_large_clouds.xml
/usr/bin/python3 tools/start_lidar.py --params-file config/sensors/ouster_baseline.yaml --startup-timeout 90 &
ros2 launch bev_bringup front_camera.launch.py device:=/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.1.3.3:1.0-video-index0 \
    width:=1024 height:=768 fps:=30.0 calibration_file:=$PWD/config/calibration/rig_20261125/front_narrow_1024x768.yaml &
/usr/bin/python3 tools/lane_lidar.py &
../.venv/bin/python tools/lane_camera.py --camera front_narrow --image-topic /camera/front/image_raw &
/usr/bin/python3 tools/compare_lanes.py --methods lidar camera_front_narrow
```

| Tool | What it does |
| --- | --- |
| [`road_features.py`](../tools/road_features.py) | shared, ROS-free: fit lines from points, ego-lane left / right, lane centre, comparison |
| [`lane_lidar.py`](../tools/lane_lidar.py) | single-scan LiDAR: road plane, paint by reflectivity contrast, road edge by kerb step / gap |
| [`lane_camera.py`](../tools/lane_camera.py) | TwinLiteNet lane + drivable masks on the camera, projected to a flat road at `--road-z` |
| [`compare_lanes.py`](../tools/compare_lanes.py) | live rate, latency, compute, availability, reach, and camera-vs-LiDAR agreement at 5/10/20/30 m; logs `data/lane_compare_<UTC>/` |

The 3D view (`bev_planning_3d.rviz`) shows both: LiDAR orange lines / red road edge / green centre, camera blue /
violet / cyan. Calibration: `config/calibration/rig_20261125` (package of 2026-11-25; new narrow front camera on
USB 4.1.3.3, 1024x768, 39 degrees).

**Cameras: narrow + wide, one process each.** `bash tools/run_lane_compare.sh --rviz --view --road-z <LiDAR road z>`
starts the LiDAR, both front cameras, `lane_lidar.py`, one `lane_camera.py` per camera, `fuse_camera_lanes.py`
(combination `narrow_wide`) and the comparison. `--view` opens `tools/lane_view.py`: each camera image with its lane
pixels (blue tint), its lines and road edge (blue / violet) and the LiDAR's lines and road edge projected into it
(orange / red). The side cameras were tried and dropped (little use for the lanes ahead, slower).

Latency fix (2026-11-25, parked): the camera node first handled all cameras in one Python process with CPU resizing
and a 5-image queue, so delays were 230-300 ms. Now each camera has its own process, the image is resized on the GPU
and the masks stay at 640x360, and only the newest image is kept. Measured: narrow 30 Hz, 12 ms compute,
**17 ms** sensor-to-result; wide 23 Hz, 35 ms compute, **62 ms**; LiDAR 20 Hz, 28 ms compute, 139 ms (LiDAR stamps are
the scan start, so its delay includes the 50 ms scan).

First live check (car parked in the garage, no road in view): LiDAR 20 Hz, 24 ms compute, 135 ms latency
from the scan stamp; camera 30 Hz, 6 ms compute, 12 ms latency. The camera correctly found no road; the LiDAR
"road boundaries" there were the garage walls. The LiDAR measured the floor 1.70 m below the sensor while
the camera assumes `--road-z -1.9`: set it from a flat-road LiDAR measurement before comparing (0.2 m of height
error moves a line at 10 m by about 1 m). Accuracy needs a drive; agreement is not accuracy, so hand labels
(below) are still needed to say which method is right.

Goal: choose the detector for three road features:

| Feature | Meaning | Main use |
| --- | --- | --- |
| **Lane lines** | painted white / yellow lines, including the yellow road centre line (divider) | lane assignment, lane keeping |
| **Road boundaries** | where the drivable road ends: kerb, edge line, or road / grass / sidewalk edge when unpainted | stay on the road; required where there is no paint |
| **Lane centre** | the path midway between the ego lane's left and right boundaries (lane line, centre line or road boundary) | the line the planner follows, instead of the recorded route |

Choose them for the live BEV and planner by measuring, on the same recorded drives and
on the Thor with the full stack running, **how accurate** (metres, in the car's frame), **how far**,
**how reliable** and **how fast** each candidate is. Status: design; nothing below is implemented yet
except where marked.

## Candidates

| | Method | Output | Expected strength / risk |
| --- | --- | --- | --- |
| A | **Camera only**: TwinLiteNet lane mask and drivable-area mask (its edge = road boundary) (existing, BDD100K, TensorRT FP16, 640x360) projected to the road with the calibrated camera and a flat ground plane | lane points in `os_sensor` | fast, long range on a flat road; ground-slope and calibration errors grow with distance; trained on US dashcams, not this camera |
| B | **Camera + LiDAR**: same lane mask, but metric positions taken from LiDAR ground points that project onto lane pixels | lane points in `os_sensor` | camera recognition with LiDAR depth, no flat-ground assumption; limited by LiDAR ground density far away and camera-LiDAR timing |
| C | **LiDAR, live**: lines from ground points of the last ~0.5 s of scans (placed with localization) whose reflectivity is high relative to the local road (the live version of `tools/extract_lane_markings.py`); road boundary from the kerb height step (5-20 cm rise within ~0.3 m) or the ground / non-ground edge along each ring | lane polylines in `os_sensor` | works in the dark, metric by construction; paint is only a few beams wide at range, worn paint is faint |
| D (dropped: no map in this comparison) | **Map prior**: lane markings from the saved map (`lane_markings.json`, done) and kerbs extracted from the map the same way, placed by localization | lane polylines | best when localized on a mapped road; not a detector, fails off the map or if paint changed. Used as a baseline and as the reference below |

A learned camera lane-line model (e.g. UFLD-v2 / CLRNet, CULane-trained) is a later candidate if A/B lose
on accuracy; it plugs into the same harness.

## Common output, so every method is scored the same way

Each method is a node publishing `std_msgs/String` JSON on `/camera_lidar_bev/lanes/<method>`:
`{stamp (source scan/image), method, lines: [{points: [[x, y], ...] in os_sensor, kind: lane_line | centre_line | road_boundary, side: left | right, confidence}], lane_centre: [[x, y], ...], latency_ms}`, plus a MarkerArray for RViz. The source stamp is the camera or LiDAR stamp of the data used, so latency =
publish time - source stamp.
The lane centre is computed the same way for every method from its own ego-lane left and right lines
(midpoints along the car's direction; with one side missing, offset the other side by the last measured lane
width and mark it `one_sided`), so the comparison measures the detector, not the centre-line logic.

## Reference (ground truth)

1. **Map reference (every frame):** the reviewed map lane markings and map kerbs (to extract offline from the
   saved map's height steps, then reviewed like the landmarks), with the reference lane centre derived from
   them, transformed into the car frame with
   localization at the method's source stamp. Dense and automatic, but it came from LiDAR intensity, so it
   favours C and D; it is only trusted where it agrees with 2.
2. **Hand labels (independent):** about 200 front-camera frames sampled across the conditions below, lane
   lines, road boundaries and the yellow centre line clicked as image polylines (a small labeling page like `tools/review_landmarks.py`). Scores A/B in
   the image, and checks the map reference (projected into the same frames) for bias.

Frames where localization is not `localized`, or the map reference has no line within 30 m, are reported
separately rather than scored.

## Test data

Record two drives of the mapped loop with everything a method needs (raw `/points`, front camera images,
`/imu`, localization, TF): `tools/route_planner.py --bag-points` plus the camera topics, or
`tools/collect_camera_lidar.sh`. One in daylight, one at dusk or night; also one short off-map street for D's
failure case. Tag segments (from the route and time) as straight, curve, intersection, lane occluded by a
parked car, worn paint, shadow. The narrow front camera (41 degrees) should be added when it is reconnected;
A/B are run on each camera separately.

## Metrics (car frame, lines as polylines)

| Metric | Definition |
| --- | --- |
| Lateral error | distance from detected line to reference line at look-ahead x = 5, 10, 20, 30 m; median and p90, separately for lane lines, centre line, left / right road boundary |
| Lane-centre error | detected minus reference lane centre at 5, 10, 20 m, and its heading error; also against where the car actually drove (recorded rear-axle path), as a sanity check |
| Boundary availability | share of frames with a road boundary on each side within 0-20 m, on painted and unpainted streets |
| Recall by range | share of reference line length within 0-10, 10-20, 20-30, 30-50 m that has a detection within 0.3 m |
| False lines | detected length farther than 0.5 m from any reference line, per 100 m driven |
| Usable range | largest look-ahead where recall >= 80 % and p90 lateral error <= 0.3 m |
| Ego-lane width error | detected minus reference width at 10 m |
| Stability | frame-to-frame change of the ego-boundary lateral offset at 10 m (jitter), and dropout length |
| Latency | publish time - source stamp, median / p95 |
| Throughput and cost | output Hz, GPU ms per frame, CPU %, GPU memory, and the drop in CenterPoint / localization rates when added |

Accuracy is scored offline on replay (deterministic, repeatable); speed and cost are measured live on the Thor
with the full `run_bev_3d.sh` stack running, because GPU sharing with CenterPoint and YOLO decides the real
latency.

## Pass criteria and decision

A method is usable for the planner if, on the held-out drive: lane-centre p90 error <= 0.3 m at 10 m, road
boundary p90 error <= 0.3 m at 10 m, line p90 lateral error <= 0.3 m at 10 m, recall >= 90 %
within 20 m (daylight), >= 15 Hz, <= 50 ms latency added, and CenterPoint and localization keep >= 95 % of their
rates. Among usable methods, rank by usable range, then night/worn-paint recall, then cost. D is the reference
ceiling on mapped roads; the chosen detector is the one that is best off the map and agrees with D on it, so the
two can be fused later (detector for the current paint, map for the topology).

## Harness to build

1. `tools/lane_methods/` nodes for A, B, C (A and B share the TwinLiteNet node's mask topic; C reuses the
   extraction logic) and D (map prior publisher).
2. `tools/eval_lane_detection.py`: replays a bag at real time or as fast as possible, subscribes to all
   `/camera_lidar_bev/lanes/*`, computes the metrics above against the map reference and the hand labels, and
   writes `evaluation/lane_benchmark_<drive>.json` plus a table per condition.
3. `tools/label_lanes.py`: local page to click lane lines on sampled camera frames.
4. Live cost check: `run_bev_3d.sh` plus one method at a time, logging rates from the existing topics.

Different methods may win different features (for example LiDAR for kerbs, camera for distant lines);
the result can be one detector per feature, fused into one lane-centre estimate.

Expected order of work: D and C first (no new model, LiDAR only), then A/B on the existing TwinLiteNet
engine, then the hand labels and the comparison report.
