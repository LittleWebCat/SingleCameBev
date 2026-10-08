"""Native camera object labels and LiDAR-supported observed boxes in stationary BEV."""

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
        DeclareLaunchArgument('detector', default_value='true'),
        DeclareLaunchArgument('fusion', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('engine_file', default_value=(
            '/home/agxthor/SingleCameraBev/camera_lidar_bev/models/'
            'camera_objects_001/yolo11n_fp16_raw.engine')),
        DeclareLaunchArgument('intrinsics_file', default_value=os.path.join(
            calibration, 'front_camera_1024x768.candidate.yaml')),
        DeclareLaunchArgument('extrinsics_file', default_value=os.path.join(
            calibration, 'T_B_os_sensor.candidate.yaml')),
        Node(package='bev_visualization', executable='camera_object_detector',
             name='camera_object_detector', namespace='camera_lidar_bev', output='screen',
             condition=IfCondition(LaunchConfiguration('detector')),
             parameters=[os.path.join(share, 'config', 'perception', 'camera_object_detector.yaml'),
                         {'engine_file': LaunchConfiguration('engine_file')}]),
        Node(package='bev_visualization', executable='camera_object_bev',
             name='camera_object_bev', namespace='camera_lidar_bev', output='screen',
             condition=IfCondition(LaunchConfiguration('fusion')),
             parameters=[os.path.join(share, 'config', 'perception', 'camera_object_bev.yaml'),
                         {'intrinsics_file': LaunchConfiguration('intrinsics_file'),
                          'extrinsics_file': LaunchConfiguration('extrinsics_file')}]),
        Node(package='rviz2', executable='rviz2', name='camera_object_bev_view', output='screen',
             condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', os.path.join(share, 'config', 'rviz', 'camera_objects.rviz')]),
    ])
