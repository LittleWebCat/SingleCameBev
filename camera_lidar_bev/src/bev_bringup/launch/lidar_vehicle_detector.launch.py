"""Learned LiDAR 3D boxes rendered as flat footprints in the stationary BEV.

model:=centerpoint (default) runs CenterPoint-PointPillar (nuScenes, 10 classes, 360 deg, +-51.2 m);
model:=pointpillars runs the earlier KITTI PointPillars (car/person, forward 0..69.12 m).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

MODELS_DIR = '/home/agxthor/SingleCameraBev/camera_lidar_bev/models'
MODELS = {
    'centerpoint': {'engine': 'lidar_objects_002/centerpoint_pillar_fp16.plan',
                    'config': 'lidar_centerpoint_detector.yaml', 'max_forward_m': 51.2},
    'pointpillars': {'engine': 'lidar_objects_001/pointpillar_fp16.plan',
                     'config': 'lidar_vehicle_detector.yaml', 'max_forward_m': 69.12},
}


def detector(context):
    share = get_package_share_directory('bev_bringup')
    name = LaunchConfiguration('model').perform(context)
    if name not in MODELS:
        raise RuntimeError(f'model must be one of {sorted(MODELS)}, got {name!r}')
    model = MODELS[name]
    engine = LaunchConfiguration('engine_file').perform(context) or os.path.join(MODELS_DIR, model['engine'])
    parameters = [os.path.join(share, 'config', 'perception', model['config']), {'engine_file': engine}]
    odometry = LaunchConfiguration('odometry_topic').perform(context)
    if odometry:
        parameters.append({'odometry_topic': odometry})
    requested = LaunchConfiguration('max_forward_m').perform(context)
    if requested:
        # Callers such as the 80 m combined view ask for their display range; stay within the model.
        parameters.append({'max_forward_m': min(float(requested), model['max_forward_m'])})
    return [Node(package='bev_lidar_detection', executable='lidar_vehicle_detector',
                 name='lidar_vehicle_detector', namespace='camera_lidar_bev', output='screen',
                 parameters=parameters)]


def generate_launch_description():
    share = get_package_share_directory('bev_bringup')
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='false'),
        DeclareLaunchArgument('model', default_value='centerpoint',
                              description='centerpoint or pointpillars'),
        DeclareLaunchArgument('engine_file', default_value='',
                              description="Engine path; empty selects the model's pinned engine."),
        DeclareLaunchArgument('max_forward_m', default_value='', description=(
            "Forward detection limit in meters; empty keeps the model config. Clamped to the "
            "model's trained range (CenterPoint 51.2 m, PointPillars 69.12 m).")),
        DeclareLaunchArgument('odometry_topic', default_value='', description=(
            'Odometry for CenterPoint multi-sweep; empty keeps the config '
            '(/camera_lidar_bev/mapping/odometry). Live localization publishes '
            '/camera_lidar_bev/localization/odometry.')),
        OpaqueFunction(function=detector),
        Node(package='rviz2', executable='rviz2', name='lidar_vehicle_bev_view', output='screen',
             condition=IfCondition(LaunchConfiguration('rviz')),
             arguments=['-d', os.path.join(share, 'config', 'rviz', 'lidar_vehicle_boxes.rviz')]),
    ])
