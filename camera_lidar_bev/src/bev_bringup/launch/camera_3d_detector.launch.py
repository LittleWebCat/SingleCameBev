"""Camera-only learned 3D detections shown beside their flat camera-relative BEV."""

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
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('engine_file', default_value=(
            '/home/agxthor/SingleCameraBev/camera_lidar_bev/models/'
            'camera_3d_001/smoke_fp16_raw.engine')),
        Node(package='bev_visualization', executable='camera_3d_detector',
             name='camera_3d_detector', namespace='camera_lidar_bev', output='screen',
             parameters=[os.path.join(share, 'config', 'perception', 'camera_3d_detector.yaml'),
                         {'engine_file': LaunchConfiguration('engine_file')}]),
        Node(package='rviz2', executable='rviz2', name='camera_3d_boxes_view', output='screen',
             condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', os.path.join(share, 'config', 'rviz', 'camera_3d_boxes.rviz')]),
    ])
