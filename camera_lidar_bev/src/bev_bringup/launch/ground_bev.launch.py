"""Stationary ground/obstacle preview using the existing sensor-frame colored cloud."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('bev_bringup')
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        Node(
            package='bev_geometry', executable='ground_bev_node',
            name='ground_bev_node', namespace='camera_lidar_bev', output='screen',
            parameters=[os.path.join(share, 'config', 'perception', 'ground_bev.yaml')]),
        Node(
            package='rviz2', executable='rviz2', name='ground_bev_view', output='screen',
            condition=IfCondition(LaunchConfiguration('rviz')),
            arguments=['-d', os.path.join(share, 'config', 'rviz', 'ground_bev.rviz')]),
    ])
