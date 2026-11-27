#!/usr/bin/env bash
# Raw sensors and bounded recording only; no detector, BEV processor or RViz.
set -eo pipefail
capture_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$capture_project_dir"
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source "$capture_project_dir/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$capture_project_dir/config/transport/cyclonedds_large_clouds.xml"
exec /usr/bin/python3 "$capture_project_dir/tools/collect_camera_lidar.py" "$@"
