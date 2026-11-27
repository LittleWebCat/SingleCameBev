#!/usr/bin/env bash
# Replay a recorded drive through the current object prediction and route planner, in the 3D view with the charts.
# The bag's localization, TF, IMU and CenterPoint detections are played back (not its recorded planner / prediction
# topics); tools/object_prediction.py and tools/route_planner.py run fresh on them, as on the car. Runs on its own
# ROS domain (default 77), so it never mixes with the car's domain 1. The planner writes data/replay_<UTC>/ (no bag).
# Ctrl+C stops everything this script started. Planning only: nothing commands the vehicle.
#
#   bash tools/run_planner_replay.sh [--bag data/drive_<UTC>/bag] [--start-s 870] [--duration-s 300] [--rate 1.0]
#                                    [--old-planner] [--no-plots]
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MAP=maps/local_gps_recording_002
BAG=data/drive_20261126T013444Z/bag
START=870
DURATION=300
RATE=1.0
PLOTS=1
PLANNER_ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --bag) BAG=$2; shift 2 ;;
    --map) MAP=$2; shift 2 ;;
    --start-s) START=$2; shift 2 ;;
    --duration-s) DURATION=$2; shift 2 ;;
    --rate) RATE=$2; shift 2 ;;
    --old-planner) PLANNER_ARGS+=(--no-track-objects); shift ;;
    --no-plots) PLOTS=0; shift ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

export PATH=/usr/bin:$PATH
set +u
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -u
export ROS_DOMAIN_ID=${REPLAY_DOMAIN_ID:-77} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$ROOT/config/transport/cyclonedds_large_clouds.xml
export DISPLAY=${DISPLAY:-:1}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOGS=$ROOT/data/replay_logs_$STAMP
mkdir -p "$LOGS"
PIDS=()
start() {
  local name=$1; shift
  setsid "$@" > "$LOGS/$name.log" 2>&1 < /dev/null &
  PIDS+=("$!")
  echo "  started $name (log $LOGS/$name.log)"
}
stop_all() {
  trap - INT TERM EXIT
  echo; echo "Stopping the replay..."
  for (( i=${#PIDS[@]}-1; i>=0; i-- )); do kill -INT -- -"${PIDS[$i]}" 2> /dev/null; done
  sleep 2
  for pid in "${PIDS[@]}"; do kill -TERM -- -"$pid" 2> /dev/null; done
  echo "Planner output: data/replay_$STAMP/ (drive.csv, plans.jsonl)"
  exit 0
}
trap stop_all INT TERM EXIT

echo "Replay of $BAG from ${START} s for ${DURATION} s at ${RATE}x on ROS domain $ROS_DOMAIN_ID; logs in $LOGS"
start prediction /usr/bin/python3 tools/object_prediction.py --map "$MAP"
start planner /usr/bin/python3 tools/route_planner.py --map "$MAP" --no-bag --out "data/replay_$STAMP" "${PLANNER_ARGS[@]}"
[ "$PLOTS" = 1 ] && start plots /usr/bin/python3 tools/planner_plots.py
start scene /usr/bin/python3 tools/bev_scene_markers.py --map "$MAP"
[ -f "$MAP/hdmap/hdmap3d_enu.json" ] && start hdmap /usr/bin/python3 tools/hdmap_publisher.py --map "$MAP"
start rviz rviz2 -d config/rviz/bev_planning_3d.rviz
sleep 5
echo "Playing (Ctrl+C to stop early)..."
ros2 bag play "$BAG" --start-offset "$START" --playback-duration "$DURATION" --rate "$RATE" \
  --topics /camera_lidar_bev/localization/odometry /camera_lidar_bev/localization/status /imu /tf /tf_static \
           /camera_lidar_bev/lidar_detections_3d > "$LOGS/bag_play.log" 2>&1
echo "Replay finished; the view stays open. Ctrl+C to close."
while true; do sleep 1; done
