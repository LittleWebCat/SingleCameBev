"""Read the Septentrio's existing TCP SBF stream and publish GPS topics."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = Path(get_package_share_directory('bev_bringup'))
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=str(share/'config/sensors/septentrio_gps.yaml')),
        ExecuteProcess(cmd=['/usr/bin/python3', str(share/'tools/septentrio_gps.py'),
                            '--config', LaunchConfiguration('config_file')], output='screen'),
    ])
