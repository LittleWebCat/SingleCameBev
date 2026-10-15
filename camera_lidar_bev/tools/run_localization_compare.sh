#!/usr/bin/env bash
set -eo pipefail
bev_localization_project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$bev_localization_project"
source /opt/ros/jazzy/setup.bash
source /home/agxthor/sensors_ws/install/setup.bash
source "$bev_localization_project/install/setup.bash"
export DISPLAY="${DISPLAY:-:1}"
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$bev_localization_project/config/transport/cyclonedds_large_clouds.xml"
exec /usr/bin/python3 "$bev_localization_project/tools/run_localization_compare.py" "$@"
