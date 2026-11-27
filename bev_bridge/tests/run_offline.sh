#!/usr/bin/env bash
# Every check that runs without ROS, a Jetson, an ESP or a car.
#   bash bev_bridge/tests/run_offline.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"
echo "== bridge suite (rclpy stubbed) =="
PYTHONPATH="$HERE/rclpy_stub" python3 -m unittest discover -s bev_bridge/tests -p 'test_bridge.py'
echo "== telemetry + response-monitor suite =="
python3 -m unittest discover -s bev_bridge/tests -p 'test_telemetry.py'
echo "== end-to-end loopback against a simulated dbw_esp_full =="
python3 bev_bridge/tests/check_loopback.py
