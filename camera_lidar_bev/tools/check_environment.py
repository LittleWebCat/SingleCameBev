#!/usr/bin/env python3
"""Read-only checks; does not install software, access sensors, or start ROS."""

import os
from pathlib import Path
import platform
import shutil
import sys


def main():
    failures = []
    print(f"Architecture: {platform.machine()}")
    print(f"ROS_DISTRO: {os.environ.get('ROS_DISTRO', '(not sourced)')}")
    for command in ("cmake", "g++", "colcon", "ros2"):
        location = shutil.which(command)
        print(f"{command}: {location or 'MISSING'}")
        if not location:
            failures.append(command)
    prefixes = [Path(value) for value in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if value]
    if not prefixes:
        print("Source /opt/ros/jazzy/setup.bash before building or launching.")
        failures.append("sourced ROS environment")
    required = ("ament_cmake", "rclcpp", "sensor_msgs", "std_msgs", "nav_msgs", "diagnostic_msgs", "yaml_cpp_vendor",
                "ament_index_python", "launch", "launch_ros", "rosbag2_transport")
    for package in required:
        available = any((prefix / "share" / "ament_index" / "resource_index" / "packages" / package).exists()
                        for prefix in prefixes)
        print(f"{package}: {'available' if available else 'MISSING'}")
        if not available:
            failures.append(package)
    ros_root = Path("/opt/ros/jazzy/share")
    camera_drivers = [name for name in ("v4l2_camera", "usb_cam") if (ros_root / name).is_dir()]
    print(f"Optional external ROS camera drivers: {', '.join(camera_drivers) or 'none found; this project includes its own OpenCV bridge'}")
    print("Sensor rates, clocks, extrinsics and live topics require separate validation.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
