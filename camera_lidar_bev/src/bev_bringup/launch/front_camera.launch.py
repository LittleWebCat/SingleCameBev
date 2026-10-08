"""Capture the front camera; synthetic mode is explicitly for testing."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = Path(get_package_share_directory("bev_bringup")) / "config"
    defaults = {
        "params_file": (str(config / "sensors/front_camera.yaml"), str),
        "device": ("/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.1.2.2:1.0-video-index0", str),
        "calibration_file": ("", str),
        "source_mode": ("camera", str),
        "width": ("640", int),
        "height": ("480", int),
        "fps": ("30.0", float),
        "fourcc": ("MJPG", str),
        "use_sim_time": ("false", bool),
    }
    declarations = [DeclareLaunchArgument(name, default_value=value)
                    for name, (value, _) in defaults.items()]
    overrides = {
        name: ParameterValue(LaunchConfiguration(name), value_type=value_type)
        for name, (_, value_type) in defaults.items() if name != "params_file"
    }
    return LaunchDescription(declarations + [Node(
        package="bev_sensor_bridge",
        executable="camera_bridge",
        namespace="camera_lidar_bev",
        name="front_camera_bridge",
        parameters=[LaunchConfiguration("params_file"), overrides],
        output="screen",
    )])
