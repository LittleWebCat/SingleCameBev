#!/usr/bin/env bash
# Measure traffic-light detection on the narrow and the wide front camera at the same time, on a drive.
# Needs the live stack with map localization (bash tools/run_bev_3d.sh) running; starts the two front cameras if
# they are not running yet (rig_20261125: narrow 4.1.3.3, wide 4.1.2.1), then one tools/traffic_light_camera_view.py
# per camera (two windows). Drive through the three signalized intersections, then Ctrl+C: each writes
# data/traffic_light_eval_<UTC>_<camera>/, and the comparison runs:
#   python3 tools/compare_light_cameras.py --eval <wide folder> --narrow-eval <narrow folder>
#
#   bash tools/run_light_camera_test.sh [--record]
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
EXTRA=(); [ "${1:-}" = "--record" ] && EXTRA=(--record)
export PATH=/usr/bin:$PATH
set +u; source /opt/ros/jazzy/setup.bash; source /home/agxthor/sensors_ws/install/setup.bash; source install/setup.bash; set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp DISPLAY=${DISPLAY:-:1}
export CYCLONEDDS_URI=file://$ROOT/config/transport/cyclonedds_large_clouds.xml
LOGS=$ROOT/data/light_test_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$LOGS"; STARTED=()
RIG=$ROOT/config/calibration/rig_20261125
declare -A PORT=([front_narrow]=4.1.3.3 [front_wide]=4.1.2.1)
declare -A SIZE=([front_narrow]=1024x768 [front_wide]=1920x1080)
start() { local name=$1; shift; setsid "$@" > "$LOGS/$name.log" 2>&1 < /dev/null & STARTED+=("$!"); echo "  started $name (log $LOGS/$name.log)"; }
running() { pgrep -f "$1" > /dev/null; }
stop_all() { trap - INT TERM; for (( i=${#STARTED[@]}-1; i>=0; i-- )); do kill -INT -- -"${STARTED[$i]}" 2> /dev/null; done
  sleep 4; for p in "${STARTED[@]}"; do kill -TERM -- -"$p" 2> /dev/null; done
  W=$(ls -d data/traffic_light_eval_*_front_wide 2> /dev/null | tail -1); N=$(ls -d data/traffic_light_eval_*_front_narrow 2> /dev/null | tail -1)
  if [ -n "$W" ] && [ -n "$N" ] && [ -f "$W/summary.json" ] && [ -f "$N/summary.json" ]; then
    /usr/bin/python3 tools/compare_light_cameras.py --eval "$W" --narrow-eval "$N" | tail -25; fi
  echo "Stopped. Logs: $LOGS"; exit 0; }
trap stop_all INT TERM

pgrep -f 'tools/[l]idar_map_localizer.py' > /dev/null || echo "  ! map localization is not running: start bash tools/run_bev_3d.sh first"
for cam in front_narrow front_wide; do
  dev=/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:${PORT[$cam]}:1.0-video-index0
  if running "__node:=${cam}_camera_bridge"; then echo "  $cam camera already running"
  elif fuser "$(readlink -f "$dev")" > /dev/null 2>&1; then echo "  ! $cam device busy (another program has it); stop it first"
  else
    w=${SIZE[$cam]%x*}; h=${SIZE[$cam]#*x}
    start "camera_$cam" ros2 run bev_sensor_bridge camera_bridge --ros-args -r __node:=${cam}_camera_bridge -r __ns:=/camera_lidar_bev \
      --params-file config/sensors/front_camera.yaml -p device:=$dev -p width:=$w -p height:=$h -p fps:=30.0 -p fourcc:=MJPG \
      -p frame_id:=camera_${cam}_optical_frame -p calibration_file:=$RIG/${cam}_${SIZE[$cam]}.yaml \
      -p image_topic:=/camera/$cam/image_raw -p camera_info_topic:=/camera/$cam/camera_info
  fi
done
sleep 5
for cam in front_narrow front_wide; do
  start "lights_$cam" "$ROOT/../.venv/bin/python" tools/traffic_light_camera_view.py --camera $cam "${EXTRA[@]}"
done
echo "Drive through the signalized intersections; Ctrl+C stops and compares the two cameras."
while true; do sleep 1; done
