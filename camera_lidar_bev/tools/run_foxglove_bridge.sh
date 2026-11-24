#!/usr/bin/env bash
# Foxglove WebSocket bridge for viewing this project's ROS topics in Foxglove.
#
#   bash tools/run_foxglove_bridge.sh             # live sensors (ROS domain 1)
#   bash tools/run_foxglove_bridge.sh --replay    # tools/replay_combined_bev.sh (domain 89, recorded clock)
#   add --network to accept viewers from other machines (default: this Thor only)
#
# Then open Foxglove (desktop app, or https://app.foxglove.dev in Chromium) and choose
# "Open connection" -> Foxglove WebSocket -> ws://localhost:8765 (or ws://<thor-ip>:8765).
#
# Read-only on purpose: viewers cannot publish topics, call services or set parameters,
# because the same ROS graph can include the DBW bridge.
set -eo pipefail
bev_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
domain=1 sim_time=false address=127.0.0.1 port=8765
while [[ $# -gt 0 ]]; do
  case "$1" in
    --replay) domain="${BEV_REPLAY_DOMAIN_ID:-89}"; sim_time=true; shift ;;
    --network) address=0.0.0.0; shift ;;
    --port) port="$2"; shift 2 ;;
    -h|--help) sed -n 2,13p "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source "$bev_project_dir/install/setup.bash"
export ROS_DOMAIN_ID="$domain" RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$bev_project_dir/config/transport/cyclonedds_large_clouds.xml"
echo "Foxglove bridge: ROS domain $domain, recorded clock $sim_time, listening on $address:$port (read-only)"
exec ros2 launch foxglove_bridge foxglove_bridge_launch.xml \
  port:="$port" address:="$address" use_sim_time:="$sim_time" \
  "capabilities:=[connectionGraph,assets]" "client_topic_whitelist:=['(?!)']" \
  "service_whitelist:=['(?!)']" "param_whitelist:=['(?!)']"
