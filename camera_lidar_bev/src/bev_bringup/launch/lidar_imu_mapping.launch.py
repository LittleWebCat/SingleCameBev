"""FAST-LIO2 and the Ouster mapping adapter; uses existing sensor topics only."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def start_nodes(context):
    output = LaunchConfiguration('output_directory').perform(context)
    if not output:
        raise RuntimeError('output_directory is required; run tools/run_mapping.sh')
    adapter_parameters = {
        'output_directory': str(Path(output).expanduser().resolve()),
        'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
        'initialization_seconds': ParameterValue(LaunchConfiguration('initialization_seconds'), value_type=float),
    }
    adapter_config = LaunchConfiguration('adapter_config').perform(context)
    parameters = [adapter_config, adapter_parameters] if adapter_config else [adapter_parameters]
    prefix = '/camera_lidar_bev/mapping/raw'
    backend = Node(
        package='fast_lio', executable='fastlio_mapping', name='fast_lio', output='screen',
        parameters=[LaunchConfiguration('config_file'), {
            'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
            'frames.map_frame': 'map', 'frames.body_frame': 'lio_imu',
            'publish.tf_en': False, 'publish.map_en': False, 'pcd_save.pcd_save_en': False,
        }],
        remappings=[('/Odometry', prefix+'/odometry'), ('/path', prefix+'/path'),
                    ('/cloud_registered', prefix+'/cloud_registered'),
                    ('/cloud_registered_body', prefix+'/cloud_registered_body'),
                    ('/cloud_effected', prefix+'/cloud_effected'), ('/Laser_map', prefix+'/backend_map'),
                    ('/map_save', prefix+'/unused_map_save'),
                    ('/fast_lio_diagnostics', prefix+'/diagnostics')])
    adapter = Node(package='bev_mapping', executable='mapping_adapter', name='mapping_adapter',
                   output='screen', parameters=parameters)
    return [backend, adapter,
            RegisterEventHandler(OnProcessExit(target_action=backend, on_exit=[
                EmitEvent(event=Shutdown(reason='FAST-LIO estimator exited'))])),
            RegisterEventHandler(OnProcessExit(target_action=adapter, on_exit=[
                EmitEvent(event=Shutdown(reason='Mapping adapter exited'))]))]


def generate_launch_description():
    share = get_package_share_directory('bev_bringup')
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=str(Path(share)/'config/mapping/fast_lio_ouster.yaml')),
        DeclareLaunchArgument('adapter_config', default_value=''),
        DeclareLaunchArgument('output_directory', default_value=''),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('initialization_seconds', default_value='2.0'),
        OpaqueFunction(function=start_nodes),
    ])
