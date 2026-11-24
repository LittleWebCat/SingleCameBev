#!/usr/bin/env bash
# Side-by-side replay of CenterPoint multi-sweep (3 scans, FAST-LIO poses) vs single-sweep.
#
#   bash tools/compare_sweeps_replay.sh [BAG] [extra replay_combined_bev.py options]
#
# Replays BAG once (default: the Oct 10 dense drive) with FAST-LIO mapping for per-scan poses,
# runs the normal detector (sweeps from config/perception/lidar_centerpoint_detector.yaml) and
# a second single-sweep detector on /camera_lidar_bev/single_sweep/*, and opens two 3D RViz
# windows: left "multi_sweep_3", right "single_sweep". The map is written to a temporary
# folder, not maps/. Ctrl+C, or the end of playback, stops everything this script started.
set -eo pipefail
bev_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$bev_project_dir"
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source "$bev_project_dir/install/setup.bash"
export ROS_DOMAIN_ID="${BEV_REPLAY_DOMAIN_ID:-89}" DISPLAY="${DISPLAY:-:1}" RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$bev_project_dir/config/transport/cyclonedds_large_clouds.xml"
bag="${1:-bags/camera_lidar_20261010T221427_088256Z}"
shift || true
work="$(mktemp -d /tmp/compare_sweeps_XXXXXX)"

# Two layouts from the 3D view: same scene, different box topic, side by side on screen.
/usr/bin/python3 - "$work" <<'EOF'
import sys
work = sys.argv[1]
base = open('config/rviz/lidar_boxes_3d.rviz').read()
geometry = '  Height: 950\n  Hide Left Dock: true\n  Hide Right Dock: true\n  Views:\n    collapsed: false\n  Width: 1440\n  X: 40\n  Y: 40'
assert geometry in base
for name, topic, x in (('multi_sweep_3', '/camera_lidar_bev/lidar_boxes_3d_markers', 0),
                       ('single_sweep', '/camera_lidar_bev/single_sweep/lidar_boxes_3d_markers', 965)):
    s = base.replace('Value: /camera_lidar_bev/lidar_boxes_3d_markers', f'Value: {topic}')
    s = s.replace(geometry, geometry.replace('Height: 950', 'Height: 1100').replace('Width: 1440', 'Width: 955')
                  .replace('X: 40', f'X: {x}'))
    open(f'{work}/{name}.rviz', 'w').write(s)
EOF

pids=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${pids[@]}"; do kill -INT -- "-$pid" 2>/dev/null || true; done
  sleep 3
  for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  echo "Stopped side-by-side processes; map/logs in $work"
}
trap cleanup EXIT INT TERM

single=(/camera_lidar_bev/single_sweep)
setsid ros2 run bev_lidar_detection lidar_vehicle_detector --ros-args \
  --params-file config/perception/lidar_centerpoint_detector.yaml -r __node:=lidar_detector_single_sweep \
  -r __ns:=/camera_lidar_bev -p use_sim_time:=true -p sweeps:=1 -p ego_vehicle_marker:=false \
  -p engine_file:="$bev_project_dir/models/lidar_objects_002/centerpoint_pillar_fp16.plan" \
  -p detections_topic:="${single[0]}/lidar_detections_3d" -p markers_topic:="${single[0]}/lidar_vehicle_markers" \
  -p markers_3d_topic:="${single[0]}/lidar_boxes_3d_markers" -p points_topic:="${single[0]}/lidar_vehicle_points" \
  > "$work/single_sweep_detector.log" 2>&1 &
pids+=($!)
for name in multi_sweep_3 single_sweep; do
  setsid rviz2 -d "$work/$name.rviz" --ros-args -p use_sim_time:=true > "$work/rviz_$name.log" 2>&1 &
  pids+=($!)
done
echo "Single-sweep detector and two RViz windows started; replaying $bag with FAST-LIO mapping."
/usr/bin/python3 tools/replay_combined_bev.py "$bag" --mapping --mapping-output "$work/map" \
  --no-viewer --no-road-lanes --no-loop --rate 1 "$@"
