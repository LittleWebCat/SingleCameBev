#!/usr/bin/env bash
# Watch the planner in simulated traffic and traffic lights live: tools/sim_live.py plays tools/simulate_planner.py
# episodes in real time, with object prediction, the planner charts, the HD map and the 3D view. Own ROS domain (78),
# so it never mixes with the car (1) or a drive replay (77). Ctrl+C stops everything. Simulation only.
#
#   bash tools/run_sim_live.sh [sim_live.py options, e.g. --scenarios light_yellow light_red --max-speed-mps 8.94 --unknown-light stop]
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
MAP=maps/local_gps_recording_002
export PATH=/usr/bin:$PATH
set +u
source /opt/ros/jazzy/setup.bash
source install/setup.bash
set -u
export ROS_DOMAIN_ID=${SIM_DOMAIN_ID:-78} RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file://$ROOT/config/transport/cyclonedds_large_clouds.xml
export DISPLAY=${DISPLAY:-:1}
LOGS=$ROOT/data/sim_live_logs_$(date -u +%Y%m%dT%H%M%SZ)
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
  echo; echo "Stopping the simulation view..."
  for (( i=${#PIDS[@]}-1; i>=0; i-- )); do kill -INT -- -"${PIDS[$i]}" 2> /dev/null; done
  sleep 2
  for pid in "${PIDS[@]}"; do kill -TERM -- -"$pid" 2> /dev/null; done
  exit 0
}
trap stop_all INT TERM EXIT

echo "Live simulation on ROS domain $ROS_DOMAIN_ID; logs in $LOGS"
start prediction /usr/bin/python3 tools/object_prediction.py --map "$MAP"
start plots /usr/bin/python3 tools/planner_plots.py
start scene /usr/bin/python3 tools/bev_scene_markers.py --map "$MAP"
[ -f "$MAP/hdmap/hdmap3d_enu.json" ] && start hdmap /usr/bin/python3 tools/hdmap_publisher.py --map "$MAP"
start rviz rviz2 -d config/rviz/bev_planning_3d.rviz
sleep 4
echo "Episodes (results below; Ctrl+C to stop):"
# in the background with wait, so Ctrl+C / pkill -INT reach the trap at once (bash defers traps during a foreground job)
setsid bash -c '/usr/bin/python3 tools/sim_live.py "$@" 2>&1 | tee "'"$LOGS"'/sim.log" | grep --line-buffered -E "episode|result" | sed -u "s/^\[INFO\] \[[0-9.]*\] \[sim_live\]: /  /"' _ --map "$MAP" "$@" &
SIM=$!
PIDS+=("$SIM")
wait "$SIM"
echo "Episodes finished; the view stays open. Ctrl+C to close."
while true; do sleep 1; done
