#!/usr/bin/env bash
# Live camera vs LiDAR lane comparison in one command: Ouster LiDAR (1024x20), the two front cameras
# (narrow 4.1.3.3, wide 4.1.2.1; calibration rig_20261125) on
# /camera/<name>/image_raw, tools/lane_lidar.py, one tools/lane_camera.py per camera, tools/fuse_camera_lanes.py
# (combination narrow_wide) and tools/compare_lanes.py in the foreground.
# Parts already running are reused; Ctrl+C stops what this started and writes
# data/lane_compare_<UTC>/summary.json. Logs: data/lane_run_<UTC>/.
#
#   bash tools/run_lane_compare.sh [--road-z -1.75] [--cameras "front_narrow front_wide"] [--rviz] [--view]
#   --view: window with the cameras' images, camera lanes (blue) and LiDAR lanes (orange) drawn on them.
#   --road-z: road height below the LiDAR for the camera projection; use the LiDAR's "road z" on a flat road.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
ROAD_Z=-1.75; RVIZ=0; VIEW=0; CAMS="front_narrow front_wide"
while [ $# -gt 0 ]; do case $1 in
  --road-z) ROAD_Z=$2; shift 2 ;; --cameras) CAMS=$2; shift 2 ;; --rviz) RVIZ=1; shift ;; --view) VIEW=1; shift ;;
  -h|--help) sed -n 2,11p "$0"; exit 0 ;; *) echo "unknown option $1"; exit 2 ;; esac; done
export PATH=/usr/bin:$PATH
set +u; source /opt/ros/jazzy/setup.bash; source /home/agxthor/sensors_ws/install/setup.bash; source install/setup.bash; set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-1} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp DISPLAY=${DISPLAY:-:1}
export CYCLONEDDS_URI=file://$ROOT/config/transport/cyclonedds_large_clouds.xml
LOGS=$ROOT/data/lane_run_$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$LOGS"; STARTED=()
RIG=$ROOT/config/calibration/rig_20261125
declare -A PORT=([front_narrow]=4.1.3.3 [front_wide]=4.1.2.1)
declare -A SIZE=([front_narrow]=1024x768 [front_wide]=1920x1080)
start() { local name=$1; shift; setsid "$@" > "$LOGS/$name.log" 2>&1 < /dev/null & STARTED+=("$!"); echo "  started $name"; }
running() { pgrep -f "$1" > /dev/null; }
stop_all() { trap - INT TERM; for (( i=${#STARTED[@]}-1; i>=0; i-- )); do kill -INT -- -"${STARTED[$i]}" 2> /dev/null; done
  sleep 3; for p in "${STARTED[@]}"; do kill -TERM -- -"$p" 2> /dev/null; done; echo "Stopped. Logs: $LOGS"; exit 0; }
trap stop_all INT TERM

if running 'ouster_ros/lib/ouster_ros/[o]s_driver'; then echo "  LiDAR already running"; else
  start lidar /usr/bin/python3 tools/start_lidar.py --params-file config/sensors/ouster_baseline.yaml --startup-timeout 90
  for _ in $(seq 1 100); do grep -q "LiDAR ACTIVE" "$LOGS/lidar.log" && break; sleep 1; done; fi
METHODS="lidar"
for cam in $CAMS; do
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
  METHODS="$METHODS camera_$cam"
done
for combo in narrow_wide; do METHODS="$METHODS camera_$combo"; done
sleep 5
running 'tools/[l]ane_lidar.py' && echo "  LiDAR lanes already running" || start lane_lidar /usr/bin/python3 tools/lane_lidar.py
for cam in $CAMS; do  # one process per camera: each receives only its own images
  running "tools/[l]ane_camera.py --camera $cam" && echo "  $cam lanes already running (its --road-z unchanged)" || \
    start lane_$cam "$ROOT/../.venv/bin/python" tools/lane_camera.py --camera $cam --road-z "$ROAD_Z"
done
running 'tools/[f]use_camera_lanes.py' && echo "  camera fusion already running" || start fuse /usr/bin/python3 tools/fuse_camera_lanes.py
[ $VIEW = 1 ] && start view /usr/bin/python3 tools/lane_view.py --cameras $(for c in front_wide front_narrow; do case " $CAMS " in *" $c "*) echo $c;; esac; done)
[ $RVIZ = 1 ] && start rviz rviz2 -d config/rviz/bev_planning_3d.rviz
echo "Comparison below every 5 s; Ctrl+C stops what this started and saves the summary."
/usr/bin/python3 -u tools/compare_lanes.py --methods $METHODS &
wait $!
stop_all
