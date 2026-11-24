#!/usr/bin/env bash
# Record a screen video and performance logs while every connected camera and the
# LiDAR run their detectors at once:
#   - ROS camera-LiDAR combined view: front-mid camera (4.1.2.1) + YOLO, Ouster +
#     CenterPoint, camera/LiDAR box fusion and its live viewer window
#   - bev_depth multi-camera pipeline: the other six cameras (configs/all_cameras_perf.yaml)
#     with YOLO, traffic-light color and depth, and its display window
#   - live LiDAR detector monitor in an xterm (also logged to CSV)
#   - tegrastats (CPU/RAM/temperature/power) and nvidia-smi (GPU load) once per second
#   - screen video of display :1 encoded on the NVENC hardware encoder
#
# Usage: scripts/record_all_cameras_perf.sh [--duration SECONDS] [--lidar-mode 1024x20|2048x10]
#        Without --duration it records until Ctrl+C. Output: runs/perf_all_cameras_<UTC time>/
set -uo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bev="$repo/camera_lidar_bev"
duration=0
lidar_mode=1024x20
while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) duration="$2"; shift 2 ;;
    --lidar-mode) lidar_mode="$2"; shift 2 ;;
    -h|--help) sed -n 2,15p "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$lidar_mode" == 1024x20 || "$lidar_mode" == 2048x10 ]] || { echo "--lidar-mode must be 1024x20 or 2048x10" >&2; exit 2; }

if pgrep -x os_driver >/dev/null || pgrep -x lidar_vehicle_d >/dev/null || pgrep -f "[m]ulticam_infer" >/dev/null; then
  echo "A LiDAR driver, detector or multi-camera pipeline is already running; stop it first." >&2
  exit 1
fi
in_use=$(fuser /dev/video* 2>/dev/null)
[[ -z "$in_use" ]] || { echo "Cameras are in use by PID(s):$in_use; close them first." >&2; exit 1; }
[[ "$(cat /sys/module/uvcvideo/parameters/mjpeg_payload 2>/dev/null)" == 800 ]] ||
  echo "WARNING: patched uvcvideo (mjpeg_payload=800) not active; some cameras may show NO FRAMES." >&2

out="$repo/runs/perf_all_cameras_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$out"
export DISPLAY="${DISPLAY:-:1}"
set +u  # the ROS setup scripts read unset variables
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source "$bev/install/setup.bash"
set -u
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}" RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$bev/config/transport/cyclonedds_large_clouds.xml"

pids=()
start() {  # start NAME LOGFILE COMMAND...: own process group so it can be stopped as a unit
  local name="$1" log="$2"; shift 2
  setsid "$@" >"$log" 2>&1 &
  pids+=("$name:$!")
  echo "started $name (pid $!)"
}
stop_all() {
  trap - INT TERM
  echo "Stopping..."
  # Screen video first so the MP4 is finalized, then the rest in reverse start order.
  # start_bev.py needs time to stop its own children (each in its own session).
  for ((i=${#pids[@]}-1; i>=0; i--)); do
    local name="${pids[$i]%%:*}" pid="${pids[$i]#*:}" grace=20 sig=INT
    [[ "$name" == ros_stack ]] && grace=60
    [[ "$name" == tegrastats || "$name" == gpu_load ]] && sig=TERM
    kill -"$sig" -- "-$pid" 2>/dev/null
    for _ in $(seq 1 $((grace * 4))); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done
    kill -0 "$pid" 2>/dev/null && { echo "forcing $name"; kill -TERM -- "-$pid" 2>/dev/null; sleep 2; kill -KILL -- "-$pid" 2>/dev/null; }
  done
  # Nothing of this kind ran before the script started (checked above), so any survivor is ours.
  local leftovers
  leftovers=$(pgrep -f "[o]s_driver|[l]idar_vehicle_detector|[c]amera_bridge|[c]amera_object_detector|[c]amera_lidar_box_fusion|[s]tart_lidar.py|[c]ombined_bev_live.py|[r]oad_lane|ros2 launch [b]ev_bringup|[m]ulticam_infer" | tr '\n' ' ')
  if [[ -n "$leftovers" ]]; then
    echo "Stopping leftover processes: $leftovers"
    kill -INT $leftovers 2>/dev/null; sleep 10; kill -TERM $leftovers 2>/dev/null
  fi
  /usr/bin/python3 "$repo/scripts/summarize_perf_run.py" "$out" | tee "$out/summary.txt"
  echo "Recording saved in $out"
}
trap stop_all INT TERM

{
  echo "started_utc: $(date -u +%FT%TZ)"
  echo "lidar_mode: $lidar_mode"
  echo "lidar_model: centerpoint"
  echo "ros_camera: front mid 4.1.2.1 (1024x768 @30)"
  echo "multicam_config: configs/all_cameras_perf.yaml"
  echo "git: $(git -C "$repo" rev-parse --short HEAD) (+ uncommitted changes)"
} >"$out/run_info.txt"

start ros_stack "$out/ros_stack.log" bash "$bev/tools/run_bev.sh" --combined-view --lidar-model centerpoint \
  --lidar-mode "$lidar_mode"
echo "Waiting for CenterPoint output..."
for _ in $(seq 1 120); do
  timeout 3 ros2 topic echo --once /camera_lidar_bev/lidar_detections_3d --field header >/dev/null 2>&1 && break
  sleep 1
done
start multicam "$out/multicam.log" env PYTHONPATH="$repo" "$repo/.venv/bin/python" -u -m bev_depth.multicam_infer \
  --config "$repo/configs/all_cameras_perf.yaml" --save-dir "$out/multicam" --no-save-images --display --profile
start monitor "$out/monitor.log" xterm -geometry 92x16+0+0 -title "LiDAR detector performance" \
  -e /usr/bin/python3 "$bev/tools/watch_lidar_detector.py" --csv "$out/lidar_detector.csv"
start tegrastats "$out/tegrastats.stdout" tegrastats --interval 1000 --logfile "$out/tegrastats.log"
start gpu_load "$out/gpu_load.csv" nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.gr,power.draw \
  --format=csv,noheader -l 1
sleep 5  # let the windows open before the video starts
start screen_video "$out/screen_video.log" ffmpeg -hide_banner -loglevel warning -y -f x11grab -framerate 15 \
  -video_size "$(xdpyinfo | awk '/dimensions/{print $2}')" -i "$DISPLAY" -c:v h264_nvenc -preset p4 -b:v 8M \
  "$out/screen.mp4"

if [[ "$duration" != 0 ]]; then
  echo "Recording for $duration s (Ctrl+C to stop early)"
  sleep "$duration" & wait $!
else
  echo "Recording; press Ctrl+C to stop"
  while true; do sleep 1; done
fi
stop_all
