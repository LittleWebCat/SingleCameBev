"""Front-left and front-right cameras (16MP USB, ~/CameraSetup layout 2026-11-19) with YOLO and LiDAR fusion.

Runs next to the main front camera (front_camera.launch.py). Each side camera gets a camera
bridge on /camera/front_left|front_right/{image_raw,camera_info} at its calibrated 1600x1200
MJPG 30 FPS, with CameraInfo from config/calibration/rig_20261124. With detector:=true a native
TensorRT YOLO detector publishes /camera_lidar_bev/<camera>/{detections_2d,camera_labeled}, and
with fusion:=true a camera_lidar_box_fusion node associates those labels with the CenterPoint
LiDAR boxes on /camera_lidar_bev/<camera>/camera_lidar_{detections_3d,markers,points}.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

BY_PATH = '/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:{}:1.0-video-index0'
CAMERAS = {'front_left': '4.1.2.4', 'front_right': '4.1.4.3'}
WIDTH, HEIGHT = 1600, 1200  # calibrated resolution of these cameras (rig_20261124)
ENGINE = ('/home/agxthor/SingleCameraBev/camera_lidar_bev/models/'
          'camera_objects_001/yolo11n_fp16_raw.engine')


def generate_launch_description():
    share = get_package_share_directory('bev_bringup')
    config = os.path.join(share, 'config')
    rig = os.path.join(config, 'calibration', 'rig_20261124')
    sim_time = ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)
    nodes = []
    for name, port in CAMERAS.items():
        frame = f'camera_{name}_optical_frame'
        intrinsics = os.path.join(rig, f'{name}_{WIDTH}x{HEIGHT}.yaml')
        nodes.append(Node(
            package='bev_sensor_bridge', executable='camera_bridge', namespace='camera_lidar_bev',
            name=f'{name}_camera_bridge', output='screen',
            parameters=[os.path.join(config, 'sensors', 'front_camera.yaml'), {
                'device': BY_PATH.format(port), 'width': WIDTH, 'height': HEIGHT,
                'fps': ParameterValue(LaunchConfiguration('fps'), value_type=float), 'fourcc': 'MJPG',
                'frame_id': frame, 'calibration_file': intrinsics,
                'image_topic': f'/camera/{name}/image_raw', 'camera_info_topic': f'/camera/{name}/camera_info',
                'use_sim_time': sim_time}]))
        nodes.append(Node(
            package='bev_visualization', executable='camera_object_detector', namespace='camera_lidar_bev',
            name=f'{name}_camera_object_detector', output='screen',
            condition=IfCondition(LaunchConfiguration('detector')),
            parameters=[os.path.join(config, 'perception', 'camera_object_detector.yaml'), {
                'engine_file': LaunchConfiguration('engine_file'), 'camera_frame': frame,
                'image_width': WIDTH, 'image_height': HEIGHT, 'image_topic': f'/camera/{name}/image_raw',
                'detections_topic': f'/camera_lidar_bev/{name}/detections_2d',
                'labeled_image_topic': f'/camera_lidar_bev/{name}/camera_labeled', 'use_sim_time': sim_time}]))
        nodes.append(Node(
            package='bev_visualization', executable='camera_lidar_box_fusion', namespace='camera_lidar_bev',
            name=f'{name}_camera_lidar_box_fusion', output='screen',
            condition=IfCondition(PythonExpression(["'", LaunchConfiguration('detector'), "' == 'true' and '",
                                                    LaunchConfiguration('fusion'), "' == 'true'"])),
            parameters=[os.path.join(config, 'perception', 'camera_lidar_boxes.yaml'), {
                'intrinsics_file': intrinsics, 'extrinsics_file': os.path.join(rig, f'{name}_T_os_sensor.yaml'),
                'camera_frame': frame, 'camera_info_topic': f'/camera/{name}/camera_info',
                'camera_detections_topic': f'/camera_lidar_bev/{name}/detections_2d',
                'markers_topic': f'/camera_lidar_bev/{name}/camera_lidar_markers',
                'points_topic': f'/camera_lidar_bev/{name}/camera_lidar_points',
                'output_detections_topic': f'/camera_lidar_bev/{name}/camera_lidar_detections_3d',
                'use_sim_time': sim_time}]))
    return LaunchDescription([
        DeclareLaunchArgument('detector', default_value='true', description='YOLO on each side camera'),
        DeclareLaunchArgument('fusion', default_value='true', description='associate YOLO labels with LiDAR boxes'),
        DeclareLaunchArgument('engine_file', default_value=ENGINE),
        DeclareLaunchArgument('fps', default_value='30.0'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
    ] + nodes)
