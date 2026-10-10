"""Learned LiDAR car boxes rendered as flat footprints in the stationary BEV."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('bev_bringup')
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('engine_file', default_value=(
            '/home/agxthor/SingleCameraBev/camera_lidar_bev/models/'
            'lidar_objects_001/pointpillar_fp16.plan')),
        DeclareLaunchArgument('max_forward_m', default_value='60', description=(
            'Forward detection limit in meters, within the trained model range '
            'of 69.12 m.')),
        Node(package='bev_lidar_detection', executable='lidar_vehicle_detector',
             name='lidar_vehicle_detector', namespace='camera_lidar_bev', output='screen',
             parameters=[os.path.join(share, 'config', 'perception', 'lidar_vehicle_detector.yaml'),
                         {'engine_file': LaunchConfiguration('engine_file'),
                          'max_forward_m': ParameterValue(
                              LaunchConfiguration('max_forward_m'), value_type=float)}]),
        Node(package='rviz2', executable='rviz2', name='lidar_vehicle_bev_view', output='screen',
             condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', os.path.join(share, 'config', 'rviz', 'lidar_vehicle_boxes.rviz')]),
    ])
