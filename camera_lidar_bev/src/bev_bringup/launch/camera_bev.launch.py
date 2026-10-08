"""Color measured LiDAR returns with the native front camera, then view from above."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('bev_bringup')
    calibration = os.path.join(share, 'config', 'calibration', 'imported_front_001')
    return LaunchDescription([
        DeclareLaunchArgument(
            'intrinsics_file',
            default_value=os.path.join(calibration, 'front_camera_1024x768.candidate.yaml')),
        DeclareLaunchArgument(
            'extrinsics_file',
            default_value=os.path.join(calibration, 'T_B_os_sensor.candidate.yaml')),
        DeclareLaunchArgument('rviz', default_value='true'),
        Node(
            package='bev_visualization', executable='camera_rgb_cloud',
            name='camera_rgb_cloud', namespace='camera_lidar_bev', output='screen',
            parameters=[
                os.path.join(share, 'config', 'perception', 'camera_rgb_cloud.yaml'),
                {'intrinsics_file': LaunchConfiguration('intrinsics_file'),
                 'extrinsics_file': LaunchConfiguration('extrinsics_file')},
            ]),
        Node(
            package='rviz2', executable='rviz2', name='camera_bev_rviz', output='screen',
            condition=IfCondition(LaunchConfiguration('rviz')),
            arguments=['-d', os.path.join(share, 'config', 'rviz', 'camera_bev.rviz')]),
    ])
