# Developing object prediction on any PC

The tracking and prediction code, a recorded drive and the benchmark run without ROS, a GPU or the car:
Python 3.10+ with numpy and scipy is enough. Improve a predictor on your PC, score it on the same replay
as the current one, then run it live on the car with the same code.

## Set up

```bash
git clone https://github.com/LittleWebCat/SingleCameBev.git
cd SingleCameBev/camera_lidar_bev
python3 -m pip install -r tools/requirements-prediction.txt   # numpy, scipy
python3 -m unittest discover -s tests -p 'test_prediction_models.py'
```

## Run the benchmark

```bash
python3 tools/eval_prediction.py --quick                 # first 4 minutes, ~15 s
python3 tools/eval_prediction.py                         # whole 24-minute drive, cv + imm + imm+map, ~5 min
python3 tools/eval_prediction.py --models cv imm         # chosen models
```

It prints median (mean) [count] tables for the first half of the drive (**dev**, use it for tuning) and the
second half (**test**, check it last so it stays an honest score), and writes
`evaluation/prediction_benchmark_<drive>[_quick].json`.

What it measures, for each confirmed track every 0.5 s:

- **Ground truth** is where the object was actually detected over the following seconds, linked by one
  reference tracker shared by every model (so all models are scored on the same objects).
- **Moving** objects (vehicles and cyclists that moved >= 2 m in 2 s, pedestrians >= 1 m): final error at
  1 / 3 / 5 s (pedestrians 4 s), average error over the horizon (ADE), straight vs turning (heading changed
  > 30 degrees within 3 s), and `missed_motion`, the share predicted to stay put.
- **Still** objects (moved < 0.7 m in 3 s): `still_predicted_moving`, the share predicted to move.

Errors include detection, tracking and localization noise: ground truth is only as good as the detections.

## Add your own model

Copy [`tools/prediction_plugin_example.py`](../tools/prediction_plugin_example.py), change it, and run:

```bash
python3 tools/eval_prediction.py --plugin my_model.py --models imm mine --quick
```

A plugin calls `prediction_models.register_model(name, factory)`. `factory(t, det)` builds a track for a
new detection `det = (group, class, enu_xyz, (length, width, height), enu_box_yaw)`, where `group` is
`vehicle`, `cyclist` or `pedestrian`. A track needs:

| Member | Meaning |
| --- | --- |
| `group`, `cls`, `z`, `size` | set from the detection |
| `predict(t)` | advance the state to time `t` (s) |
| `update(t, det)` | correct with the matched detection |
| `position()`, `position_cov()` | `(x, y)` and its 2x2 covariance, used to match detections |
| `velocity()` | `(vx, vy)`; tracks slower than 1.5 / 1.0 / 0.5 m/s (vehicle / cyclist / pedestrian) are predicted to stay |
| `future(horizon_s, step_s)` | `[(t, x, y, heading, 2-sigma radius)]` from `t = 0` |

Subclassing `CVTrack` or `IMMTrack` and overriding `future()` changes only the prediction; overriding
`predict` / `update` changes the filter too. A model name with `+map` adds lane following along the mapped
route (`RouteLanes`). Run the plugin live on the car by registering it in `prediction_models.MODELS` (or
importing it there) and starting `tools/object_prediction.py --model <name>` (the `--model` choices list
needs the new name added).

## Where the current models lose accuracy

Measured on `drive_20261124T213905Z` (see [the live guide](live_3d_planning.md#object-tracking-and-prediction)):
vehicle error is almost all **along the direction of travel**, mainly from

1. **new tracks**: a track starts at zero speed, so its first ~0.5 s has a 2.7 m/s speed error;
   CenterPoint regresses a velocity per box (head channels 8-9) that the detector does not decode yet;
2. **braking and accelerating** cars (about +-5 m along-track at 3 s);
3. few turning examples in this drive (about 50), so turn handling is weakly measured.

Good first experiments: a constant-acceleration or IMM model with an acceleration state; initialising
speed from the first few detections; slowing vehicles that approach a mapped stop line or a stopped car;
using the route / lane markings for direction at intersections.

## Add more drives

Every planner run on the car records a bag (`data/drive_<UTC>/bag`, ignored by git). Export it once on the
car, then commit or copy the .npz:

```bash
source /opt/ros/jazzy/setup.bash && source install/setup.bash
/usr/bin/python3 tools/export_prediction_dataset.py --bag data/drive_<UTC>/bag   # -> datasets/prediction/drive_<UTC>.npz
python3 tools/eval_prediction.py --dataset datasets/prediction/drive_<UTC>.npz
```

The dataset ([format](../tools/prediction_dataset.py)) holds, per LiDAR scan, the CenterPoint detections of
trackable classes already in the map frame (score >= 0.2; the benchmark uses >= 0.35 like the live node), the
car's pose, and the mapped route. The 24-minute drive is 12.9 MB. Several drives, more turning traffic and
more pedestrians are needed before a learned (neural network) predictor is worth training; the recordings
label themselves, because each object's real future is in the later frames.
