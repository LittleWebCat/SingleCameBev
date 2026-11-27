"""Front-wide, front-left and front-right cameras with YOLO for tools/map_landmarks.py.

front_wide (Global Shutter, 117 deg, port 4.1.2.1) at its calibrated 1920x1080 on /camera/front_wide/*
with YOLO on /camera_lidar_bev/front_wide/detections_2d, plus the side cameras from
front_side_cameras.launch.py (1600x1200). Calibration: config/calibration/rig_20261124. YOLO classes
include traffic light and stop sign (config/perception/camera_object_detector.yaml). No box fusion.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

ENGINE = '/home/agxthor/SingleCameraBev/camera_lidar_bev/models/camera_objects_001/yolo11n_fp16_raw.engine'


def generate_launch_description():
    share = get_package_share_directory('bev_bringup')
    launch = lambda name, arguments: IncludeLaunchDescription(  # noqa: E731
        PythonLaunchDescriptionSource(os.path.join(share, 'launch', name)), launch_arguments=arguments.items())
    config = os.path.join(share, 'config')
    frame = 'camera_front_wide_optical_frame'
    return LaunchDescription([
        Node(package='bev_sensor_bridge', executable='camera_bridge', namespace='camera_lidar_bev',
             name='front_wide_camera_bridge', output='screen',
             parameters=[os.path.join(config, 'sensors', 'front_camera.yaml'), {
                 'device': '/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.1.2.1:1.0-video-index0',
                 'width': 1920, 'height': 1080, 'fps': 30.0, 'fourcc': 'MJPG', 'frame_id': frame,
                 'calibration_file': os.path.join(config, 'calibration', 'rig_20261124', 'front_wide_1920x1080.yaml'),
                 'image_topic': '/camera/front_wide/image_raw', 'camera_info_topic': '/camera/front_wide/camera_info'}]),
        Node(package='bev_visualization', executable='camera_object_detector', namespace='camera_lidar_bev',
             name='front_wide_camera_object_detector', output='screen',
             parameters=[os.path.join(config, 'perception', 'camera_object_detector.yaml'), {
                 'engine_file': ENGINE, 'camera_frame': frame, 'image_width': 1920, 'image_height': 1080,
                 'image_topic': '/camera/front_wide/image_raw',
                 'detections_topic': '/camera_lidar_bev/front_wide/detections_2d',
                 'labeled_image_topic': '/camera_lidar_bev/front_wide/camera_labeled'}]),
        launch('front_side_cameras.launch.py', {'detector': 'true', 'fusion': 'false'}),
    ])
