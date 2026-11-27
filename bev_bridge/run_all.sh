#!/usr/bin/env bash
set -eo pipefail
bev_bridge_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source "$bev_bridge_dir/../camera_lidar_bev/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export DISPLAY="${DISPLAY:-:1}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$bev_bridge_dir/../camera_lidar_bev/config/transport/cyclonedds_large_clouds.xml"
exec /usr/bin/python3 "$bev_bridge_dir/run_all.py" "$@"
