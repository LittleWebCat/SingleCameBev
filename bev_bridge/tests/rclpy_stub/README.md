# Minimal rclpy / std_msgs stub (tests only)

`test_bridge.py` imports `bev_dbw_bridge`, which imports `rclpy` at module scope, so the suite could
only run on the Jetson. These stubs provide just enough surface for the import and for the tests'
`object.__new__` node fixtures — `Node` is never constructed and nothing here talks to a ROS graph.

```bash
PYTHONPATH=bev_bridge/tests/rclpy_stub python3 -m unittest discover -s bev_bridge/tests -p 'test_bridge.py'
```

Or use `bev_bridge/tests/run_offline.sh`, which runs this plus `test_telemetry.py` and
`check_loopback.py`. On the Jetson, drop the `PYTHONPATH` so the real `rclpy` is used instead.
