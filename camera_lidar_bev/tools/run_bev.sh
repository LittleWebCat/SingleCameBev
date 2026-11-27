#!/usr/bin/env bash
# One entry point for native sensors, color association and stationary geometry.
set -eo pipefail
bev_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$bev_project_dir"
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source "$bev_project_dir/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export DISPLAY="${DISPLAY:-:1}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$bev_project_dir/config/transport/cyclonedds_large_clouds.xml"
exec /usr/bin/python3 "$bev_project_dir/tools/start_bev.py" "$@"
