"""Show camera labels and road masks beside flat LiDAR 3D BEV footprints.

This launch starts inference and visualization only. The calibrated camera and
Ouster streams must come from existing sensor drivers or a rosbag player.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, GroupAction, IncludeLaunchDescription, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import SetParameter, Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('bev_bringup')
    calibration = os.path.join(share, 'config', 'calibration', 'imported_front_001')
    models = '/home/agxthor/SingleCameraBev/camera_lidar_bev/models'

    def detector_launch(filename, arguments):
        # Each include uses engine_file/rviz internally. Scope those generic
        # arguments so the detector engines cannot overwrite each other.
        return GroupAction(
            condition=IfCondition(LaunchConfiguration('detectors')),
            actions=[
                SetParameter('use_sim_time', ParameterValue(
                    LaunchConfiguration('use_sim_time'), value_type=bool)),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(os.path.join(share, 'launch', filename)),
                    launch_arguments=arguments.items()),
            ])

    declarations = [
        DeclareLaunchArgument('detectors', default_value='true', description=(
            'Start PointPillars, YOLO and camera/LiDAR association. '
            'Set false when these processors already run.')),
        DeclareLaunchArgument('viewer', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('intrinsics_file', default_value=os.path.join(
            calibration, 'front_camera_1024x768.candidate.yaml')),
        DeclareLaunchArgument('extrinsics_file', default_value=os.path.join(
            calibration, 'T_B_os_sensor.candidate.yaml')),
        DeclareLaunchArgument('lidar_engine_file', default_value=os.path.join(
            models, 'lidar_objects_001', 'pointpillar_fp16.plan')),
        DeclareLaunchArgument('camera_2d_engine_file', default_value=os.path.join(
            models, 'camera_objects_001', 'yolo11n_fp16_raw.engine')),
        DeclareLaunchArgument('viewer_script', default_value=os.path.join(
            share, 'tools', 'combined_bev_live.py')),
        DeclareLaunchArgument('range_m', default_value='80'),
        DeclareLaunchArgument('lateral_m', default_value='30'),
        DeclareLaunchArgument('view_fps', default_value='10'),
        DeclareLaunchArgument('max_points', default_value='30000'),
        DeclareLaunchArgument('max_map_points', default_value='100000'),
        DeclareLaunchArgument('saved_map', default_value='', description='Saved map directory for original-source replay'),
        DeclareLaunchArgument('localize_map', default_value='', description='Saved map directory paired with independent live odometry'),
        DeclareLaunchArgument('localization_odometry_topic', default_value='/camera_lidar_bev/localization/odometry'),
        DeclareLaunchArgument('localization_status_topic', default_value='/camera_lidar_bev/localization/status'),
        DeclareLaunchArgument('world_detections_topic', default_value='/camera_lidar_bev/localization/detections_3d'),
        DeclareLaunchArgument('replay_bag', default_value='', description='Original source bag for saved-map provenance'),
        DeclareLaunchArgument('gps_alignment', default_value='', description='GPS alignment directory for saved map'),
        DeclareLaunchArgument('road_lanes', default_value='true'),
        DeclareLaunchArgument('road_python', default_value='/home/agxthor/SingleCameraBev/.venv/bin/python'),
        DeclareLaunchArgument('road_engine_file', default_value=os.path.join(
            models, 'road_lanes_001', 'twinlitenet_fp16.engine')),
        DeclareLaunchArgument('road_fps', default_value='10'),
        DeclareLaunchArgument('path_tracking', default_value='false'),
        DeclareLaunchArgument('tracking_config', default_value=os.path.join(share, 'config/tracking/pure_pursuit.yaml')),
        DeclareLaunchArgument('reference_trajectory', default_value='',
                              description='ENU sensor trajectory CSV for diagnostic tracking'),
    ]

    viewer = ExecuteProcess(
        condition=IfCondition(LaunchConfiguration('viewer')),
        additional_env={'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                        'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'},
        cmd=[
            '/usr/bin/python3', LaunchConfiguration('viewer_script'),
            '--intrinsics', LaunchConfiguration('intrinsics_file'),
            '--extrinsics', LaunchConfiguration('extrinsics_file'),
            '--range-m', LaunchConfiguration('range_m'),
            '--lateral-m', LaunchConfiguration('lateral_m'),
            '--fps', LaunchConfiguration('view_fps'),
            '--max-points', LaunchConfiguration('max_points'),
            '--max-map-points', LaunchConfiguration('max_map_points'),
            '--saved-map', LaunchConfiguration('saved_map'),
            '--localize-map', LaunchConfiguration('localize_map'),
            '--localization-odometry-topic', LaunchConfiguration('localization_odometry_topic'),
            '--localization-status-topic', LaunchConfiguration('localization_status_topic'),
            '--world-detections-topic', LaunchConfiguration('world_detections_topic'),
            '--replay-bag', LaunchConfiguration('replay_bag'),
            '--gps-alignment', LaunchConfiguration('gps_alignment'),
            '--path-tracking', LaunchConfiguration('path_tracking'),
            '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')],
        ], output='screen')
    tracker = Node(
        condition=IfCondition(LaunchConfiguration('path_tracking')),
        package='bev_path_tracking', executable='path_tracker', name='path_tracker', output='screen',
        parameters=[LaunchConfiguration('tracking_config'), {
            'reference_trajectory': LaunchConfiguration('reference_trajectory'),
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            'source_mode': 'saved_recording_pose',
        }])
    return LaunchDescription(declarations + [
        tracker,
        RegisterEventHandler(OnProcessExit(
            target_action=tracker, on_exit=[EmitEvent(event=Shutdown(reason='Path tracker exited'))])),
        detector_launch('lidar_vehicle_detector.launch.py', {
            'rviz': 'false', 'engine_file': LaunchConfiguration('lidar_engine_file'),
            'max_forward_m': PythonExpression(['min(', LaunchConfiguration('range_m'), ', 69.12)'])}),
        detector_launch('camera_objects.launch.py', {
            'detector': 'true', 'fusion': 'false', 'rviz': 'false',
            'engine_file': LaunchConfiguration('camera_2d_engine_file')}),
        detector_launch('camera_lidar_boxes.launch.py', {
            'rviz': 'false', 'intrinsics_file': LaunchConfiguration('intrinsics_file'),
            'extrinsics_file': LaunchConfiguration('extrinsics_file')}),
        ExecuteProcess(
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('detectors'), "'.lower() == 'true' and '",
                LaunchConfiguration('road_lanes'), "'.lower() == 'true'"])),
            cmd=[LaunchConfiguration('road_python'), os.path.join(share, 'tools', 'road_lane_detector.py'),
                 '--engine', LaunchConfiguration('road_engine_file'),
                 '--fps', LaunchConfiguration('road_fps'),
                 '--ros-args', '-p', ['use_sim_time:=', LaunchConfiguration('use_sim_time')]],
            output='screen'),
        RegisterEventHandler(OnProcessExit(
            target_action=viewer,
            on_exit=[EmitEvent(event=Shutdown(reason='Combined viewer closed'))])),
        viewer,
    ])
