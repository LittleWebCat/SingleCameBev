#!/usr/bin/env bash
# Start the complete live 3D BEV in one command, in order:
#   1 Ouster LiDAR at 1024x20 (20 Hz)      5 object tracking + 4-5 s prediction (tools/object_prediction.py)
#   2 map localization (waits: localized)  6 route planner + drive recording (tools/route_planner.py)
#   3 CenterPoint 3D detection             7 measured-vs-planned charts (tools/planner_plots.py)
#   4 RViz bridge (TF, route, landmarks)   8 BEV scene markers + RViz (config/rviz/bev_planning_3d.rviz)
# Parts that are already running are reused and left running. Ctrl+C stops only what this script
# started. Logs: data/bev3d_<UTC>/. Planning only: nothing commands the vehicle.
#
#   bash tools/run_bev_3d.sh [--map maps/local_gps_recording_002] [--no-plots] [--lanes]
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MAP=maps/local_gps_recording_002
PLOTS=1
PRED_ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --map) MAP=$2; shift 2 ;;
    --no-plots) PLOTS=0; shift ;;
    --lanes) PRED_ARGS+=(--lanes); shift ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

export PATH=/usr/bin:$PATH
set +u
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source install/setup.bash
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$ROOT/config/transport/cyclonedds_large_clouds.xml
export DISPLAY=${DISPLAY:-:1}
LOGS=$ROOT/data/bev3d_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$LOGS"
STARTED=()

start() {  # name, command...: own process group so Ctrl+C can stop the whole part
  local name=$1; shift
  setsid "$@" > "$LOGS/$name.log" 2>&1 < /dev/null &
  STARTED+=("$!:$name")
  echo "  started $name (log $LOGS/$name.log)"
}

running() { pgrep -f "$1" > /dev/null; }

stop_all() {
  trap - INT TERM
  echo; echo "Stopping what this script started..."
  for (( i=${#STARTED[@]}-1; i>=0; i-- )); do
    local pid=${STARTED[$i]%%:*}
    kill -INT -- -"$pid" 2> /dev/null
  done
  for _ in $(seq 1 20); do
    local alive=0
    for entry in "${STARTED[@]}"; do kill -0 "${entry%%:*}" 2> /dev/null && alive=1; done
    [ $alive = 0 ] && break
    sleep 0.5
  done
  for entry in "${STARTED[@]}"; do kill -TERM -- -"${entry%%:*}" 2> /dev/null; done
  echo "Stopped. Logs: $LOGS"
  exit 0
}
trap stop_all INT TERM

wait_for() {  # seconds, description, test command...
  local seconds=$1 what=$2; shift 2
  for _ in $(seq 1 "$seconds"); do "$@" && return 0; sleep 1; done
  echo "  timed out waiting for $what"; return 1
}
localized() { timeout 3 ros2 topic echo --once /camera_lidar_bev/localization/status 2> /dev/null | grep -q '"localized"'; }

echo "3D BEV on ROS domain $ROS_DOMAIN_ID, map $MAP; logs in $LOGS"

echo "1/8 LiDAR 20 Hz"
if running 'ouster_ros/lib/ouster_ros/[o]s_driver'; then
  echo "  LiDAR driver already running (reused; its mode is whatever it was started with)"
else
  start lidar /usr/bin/python3 tools/start_lidar.py --params-file config/sensors/ouster_baseline.yaml --startup-timeout 90
  wait_for 100 "LiDAR ACTIVE" grep -q "LiDAR ACTIVE" "$LOGS/lidar.log" || stop_all
fi

echo "2/8 map localization"
if running 'tools/[l]idar_map_localizer.py'; then
  echo "  localizer already running (reused)"
else
  start localization bash tools/run_localization_compare.sh --live --start-sensors --map "$MAP" --max-rate 20
fi
echo "  waiting until localized (drive onto / stay on the mapped route)..."
wait_for 120 "localization" localized || echo "  continuing; RViz fills in once localized"

echo "3/8 CenterPoint 3D detection"
if running '[l]idar_vehicle_detector --ros'; then echo "  detector already running (reused)"; else
  start detector ros2 launch bev_bringup lidar_vehicle_detector.launch.py model:=centerpoint \
    odometry_topic:=/camera_lidar_bev/localization/odometry rviz:=false
fi

echo "4/8 RViz bridge"
running 'tools/[l]ocalization_rviz_bridge.py' && echo "  already running (reused)" || start bridge /usr/bin/python3 tools/localization_rviz_bridge.py --map "$MAP"

echo "5/8 object prediction"
running 'tools/[o]bject_prediction.py' && echo "  already running (reused)" || start prediction /usr/bin/python3 tools/object_prediction.py --map "$MAP" "${PRED_ARGS[@]}"

echo "6/8 planner + recording"
running 'tools/[r]oute_planner.py' && echo "  already running (reused)" || start planner /usr/bin/python3 tools/route_planner.py --map "$MAP"

if [ $PLOTS = 1 ]; then
  echo "7/8 planner charts"
  running 'tools/[p]lanner_plots.py' && echo "  already running (reused)" || start plots /usr/bin/python3 tools/planner_plots.py
fi

echo "8/8 BEV scene + RViz"
running 'tools/[b]ev_scene_markers.py' && echo "  scene already running (reused)" || start scene /usr/bin/python3 tools/bev_scene_markers.py --map "$MAP"
[ -f "$MAP/hdmap/hdmap3d_enu.json" ] && { running 'tools/[h]dmap_publisher.py' && echo "  HD map already running (reused)" || start hdmap /usr/bin/python3 tools/hdmap_publisher.py --map "$MAP"; }
running 'rviz2 -d config/rviz/[b]ev_planning_3d.rviz' && echo "  RViz already open (reused)" || start rviz rviz2 -d config/rviz/bev_planning_3d.rviz

echo
REC="(planner was already running: see its own output)"
if [ -e "$LOGS/planner.log" ]; then
  wait_for 15 "planner recording folder" grep -q "recording " "$LOGS/planner.log" > /dev/null
  REC=$(grep -o "recording [^ ]*" "$LOGS/planner.log" | tail -1 | cut -d' ' -f2)
fi
echo "Running. Recording: ${REC:-unknown, see $LOGS/planner.log}. Press Ctrl+C to stop what this script started."
while true; do
  for entry in "${STARTED[@]}"; do
    pid=${entry%%:*} name=${entry#*:}
    if ! kill -0 "$pid" 2> /dev/null && [ ! -e "$LOGS/.$name.reported" ]; then
      echo "  ! $name exited; last lines of $LOGS/$name.log:"; tail -5 "$LOGS/$name.log" | sed 's/^/      /'
      touch "$LOGS/.$name.reported"
    fi
  done
  sleep 5
done
