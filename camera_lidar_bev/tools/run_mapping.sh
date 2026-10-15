#!/usr/bin/env bash
set -eo pipefail
bev_mapping_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$bev_mapping_project"
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source "$bev_mapping_project/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-90}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$bev_mapping_project/config/transport/cyclonedds_large_clouds.xml"
exec /usr/bin/python3 "$bev_mapping_project/tools/run_mapping.py" "$@"
