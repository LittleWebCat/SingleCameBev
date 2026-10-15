"""Inspect existing inputs without starting drivers or publishing perception."""

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
        "params_file": str(config / "sensors" / "sensor_probe.yaml"),
        "lidar_topic": "/points",
        "lidar_reliability": "reliable",
        "camera_topic": "/camera/front/image_raw",
        "camera_info_topic": "/camera/front/camera_info",
        "imu_topic": "/sensing/imu/imu_data",
        "odom_topic": "/lio_sam/mapping/odometry",
        "use_sim_time": "false",
    }
    declarations = [
        DeclareLaunchArgument(name, default_value=value)
        for name, value in defaults.items()
    ]
    overrides = {
        name: ParameterValue(LaunchConfiguration(name), value_type=str)
        for name in defaults
        if name not in ("params_file", "use_sim_time")
    }
    overrides["use_sim_time"] = ParameterValue(
        LaunchConfiguration("use_sim_time"), value_type=bool
    )
    return LaunchDescription(
        declarations
        + [Node(
            package="bev_sensor_bridge",
            executable="sensor_probe",
            namespace="camera_lidar_bev",
            name="sensor_probe",
            parameters=[LaunchConfiguration("params_file"), overrides],
            output="screen",
        )]
    )
