"""Preserve native Ouster timing, expose sensor-frame poses and export a local map.

FAST-LIO is a separate GPL-2.0 executable. This adapter does not implement its
estimator or claim calibrated clock synchronization, loop closure or GNSS pose.
"""

from collections import deque
import copy
import csv
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

for _thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.signals import SignalHandlerOptions
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import Imu, PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster


PREFIX = '/camera_lidar_bev/mapping'


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def rotation_matrix(q):
    values = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(values)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError('invalid orientation quaternion')
    x, y, z, w = values / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def cloud_array(message):
    """Read XYZ and optional intensity without assuming tightly packed rows."""
    fields = {field.name: field for field in message.fields}
    names = ['x', 'y', 'z'] + (['intensity'] if 'intensity' in fields else [])
    if any(name not in fields or fields[name].datatype != PointField.FLOAT32
           or fields[name].count != 1 for name in names):
        raise ValueError('registered cloud requires float32 XYZ/intensity')
    if (message.point_step <= 0 or message.row_step < message.width*message.point_step
            or len(message.data) < message.row_step*message.height):
        raise ValueError('invalid registered cloud layout')
    endian = '>f4' if message.is_bigendian else '<f4'
    dtype = np.dtype({'names': names, 'formats': [endian]*len(names),
                      'offsets': [fields[n].offset for n in names],
                      'itemsize': message.point_step})
    data = np.ndarray((message.height, message.width), dtype=dtype,
                      buffer=message.data, strides=(message.row_step, message.point_step))
    out = np.zeros((message.height*message.width, 4), dtype=np.float32)
    for index, name in enumerate(names):
        out[:, index] = data[name].reshape(-1)
    return out[np.isfinite(out).all(axis=1)]


class MappingAdapter(Node):
    def __init__(self):
        super().__init__('mapping_adapter')
        defaults = {
            'output_directory': '', 'map_frame': 'map', 'sensor_frame': 'os_sensor',
            'lidar_frame': 'os_lidar', 'imu_frame': 'os_imu',
            'imu_from_sensor_translation': [0.002441, 0.009725, -0.007533],
            'imu_from_lidar_translation': [0.002441, 0.009725, 0.030662],
            'imu_from_lidar_rotation': [-1., 0., 0., 0., -1., 0., 0., 0., 1.],
            'initialization_seconds': 2.0, 'gyro_mean_limit': 0.03,
            'gyro_std_limit': 0.02, 'accel_std_limit': 0.15,
            'voxel_size_m': 0.25, 'max_map_voxels': 500000,
            'map_publish_period': 2.0, 'max_path_poses': 20000,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.settings = {k: self.get_parameter(k).value for k in defaults}
        for key in ('initialization_seconds', 'voxel_size_m', 'map_publish_period'):
            if not math.isfinite(self.settings[key]) or self.settings[key] <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.settings['max_map_voxels'] <= 0 or self.settings['max_path_poses'] <= 0:
            raise ValueError('map/path capacities must be positive')
        self.output = Path(self.settings['output_directory']).expanduser()
        if not self.settings['output_directory']:
            raise ValueError('output_directory is required; use tools/run_mapping.sh')
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ('pointcloud_map.pcd', 'trajectory.csv', 'mapping_report.json'):
            if (self.output/name).exists():
                raise FileExistsError(f'refusing to overwrite an earlier mapping run: {self.output/name}')
        self.trajectory_file = (self.output/'trajectory.csv').open('x', newline='')
        self.trajectory_writer = csv.writer(self.trajectory_file)
        self.trajectory_writer.writerow(['stamp_sec', 'x_m', 'y_m', 'z_m', 'qx', 'qy', 'qz', 'qw'])
        self.map_frame = self.settings['map_frame']
        self.sensor_frame = self.settings['sensor_frame']
        self.imu_from_sensor = np.array(self.settings['imu_from_sensor_translation'])
        self.imu_window = deque()
        self.ready = False
        self.failed = False
        self.first_forwarded_imu = None
        self.latest_imu = None
        self.last_cloud_stamp = None
        self.last_odom_stamp = None
        self.first_odom_stamp = None
        self.latest_pose = None
        self.last_imu_stamp = None
        self.last_map_publish = 0.0
        self.voxels = {}
        self.voxel_limit_reached = False
        self.path = RosPath()
        self.path.header.frame_id = self.map_frame
        self.counters = {'imu_received': 0, 'scans_received': 0, 'scans_forwarded': 0,
                         'scans_skipped_initialization': 0, 'scans_skipped_imu_coverage': 0,
                         'poses': 0, 'registered_scans': 0, 'invalid_inputs': 0}
        self.stats_lock = threading.RLock()
        self.sensor_group = MutuallyExclusiveCallbackGroup()
        self.map_group = MutuallyExclusiveCallbackGroup()
        self.service_group = MutuallyExclusiveCallbackGroup()
        sensor_qos = QoSProfile(depth=500, reliability=ReliabilityPolicy.RELIABLE)
        cloud_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        persistent = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.imu_pub = self.create_publisher(Imu, PREFIX+'/input/imu', sensor_qos)
        self.cloud_pub = self.create_publisher(PointCloud2, PREFIX+'/input/points', cloud_qos)
        self.odom_pub = self.create_publisher(Odometry, PREFIX+'/odometry', 10)
        self.path_pub = self.create_publisher(RosPath, PREFIX+'/path', persistent)
        self.map_pub = self.create_publisher(PointCloud2, PREFIX+'/map', persistent)
        self.diag_pub = self.create_publisher(DiagnosticArray, PREFIX+'/diagnostics', 10)
        self.tf = TransformBroadcaster(self)
        imu_input_qos = QoSProfile(depth=500, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, '/imu', self.on_imu, imu_input_qos,
                                 callback_group=self.sensor_group)
        self.create_subscription(PointCloud2, '/points', self.on_cloud, cloud_qos,
                                 callback_group=self.sensor_group)
        self.create_subscription(String, '/metadata', self.on_metadata, persistent,
                                 callback_group=self.sensor_group)
        self.create_subscription(Odometry, PREFIX+'/raw/odometry', self.on_odometry, 100,
                                 callback_group=self.map_group)
        self.create_subscription(PointCloud2, PREFIX+'/raw/cloud_registered',
                                 self.on_registered_cloud, cloud_qos,
                                 callback_group=self.map_group)
        self.create_service(Trigger, PREFIX+'/save_map', self.on_save,
                            callback_group=self.service_group)
        self.create_timer(1.0, self.publish_status, callback_group=self.service_group)
        self.get_logger().info('Waiting for a low-motion IMU interval. Keep the rig still '
                               'for initialization; coordinates will be local, without GPS.')

    def fail_input(self, reason):
        self.counters['invalid_inputs'] += 1
        if not self.failed:
            self.get_logger().error(reason)
        self.failed = True

    def on_metadata(self, message):
        try:
            metadata = json.loads(message.data)
            lidar = np.array(metadata['lidar_intrinsics']['lidar_to_sensor_transform'], dtype=float).reshape(4, 4)
            imu = np.array(metadata['imu_intrinsics']['imu_to_sensor_transform'], dtype=float).reshape(4, 4)
            lidar[:3, 3] /= 1000.0
            imu[:3, 3] /= 1000.0
            imu_from_lidar = np.linalg.inv(imu) @ lidar
            imu_from_sensor = np.linalg.inv(imu)
            if (not np.allclose(imu_from_lidar[:3, :3], np.array(
                    self.settings['imu_from_lidar_rotation']).reshape(3, 3), atol=1e-6)
                    or not np.allclose(imu_from_lidar[:3, 3], self.settings[
                        'imu_from_lidar_translation'], atol=1e-6)
                    or not np.allclose(imu_from_sensor[:3, :3], np.eye(3), atol=1e-6)
                    or not np.allclose(imu_from_sensor[:3, 3], self.imu_from_sensor, atol=1e-6)):
                self.fail_input('Ouster factory transforms differ from mapping configuration; '
                                'regenerate the mapping parameters for this sensor.')
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
            self.fail_input(f'Cannot check Ouster factory metadata: {error}')

    def on_imu(self, message):
        self.counters['imu_received'] += 1
        if self.failed:
            return
        stamp = stamp_seconds(message.header.stamp)
        if message.header.frame_id != self.settings['imu_frame']:
            self.fail_input(f'IMU frame must be {self.settings["imu_frame"]}')
            return
        if self.last_imu_stamp is not None and stamp < self.last_imu_stamp:
            self.fail_input('IMU time moved backward. Mapping requires one-pass replay; restart it.')
            return
        self.last_imu_stamp = stamp
        values = np.array([message.angular_velocity.x, message.angular_velocity.y,
                           message.angular_velocity.z, message.linear_acceleration.x,
                           message.linear_acceleration.y, message.linear_acceleration.z])
        if not np.isfinite(values).all():
            self.fail_input('Nonfinite IMU measurement')
            return
        self.latest_imu = stamp
        if not self.ready:
            self.imu_window.append((stamp, values))
            window = self.settings['initialization_seconds']
            while len(self.imu_window) > 1 and stamp-self.imu_window[1][0] >= window:
                self.imu_window.popleft()
            if len(self.imu_window) < 20 or stamp-self.imu_window[0][0] < window:
                return
            measured = np.stack([v for _, v in self.imu_window])
            quiet = (np.linalg.norm(measured[:, :3].mean(axis=0)) < self.settings['gyro_mean_limit']
                     and measured[:, :3].std(axis=0).max() < self.settings['gyro_std_limit']
                     and measured[:, 3:].std(axis=0).max() < self.settings['accel_std_limit']
                     and 8.5 < np.linalg.norm(measured[:, 3:].mean(axis=0)) < 11.0)
            if not quiet:
                return
            self.ready = True
            self.first_forwarded_imu = stamp
            self.imu_window.clear()
            self.get_logger().info('Low-motion initialization interval found; forwarding native '
                                   'LiDAR/IMU data to FAST-LIO2.')
        self.imu_pub.publish(message)

    def on_cloud(self, message):
        self.counters['scans_received'] += 1
        if self.failed:
            return
        stamp = stamp_seconds(message.header.stamp)
        if self.last_cloud_stamp is not None and stamp < self.last_cloud_stamp:
            self.fail_input('LiDAR time moved backward. Mapping requires one-pass replay; restart it.')
            return
        self.last_cloud_stamp = stamp
        if not self.ready:
            self.counters['scans_skipped_initialization'] += 1
            return
        fields = {field.name: field for field in message.fields}
        required = {'x': PointField.FLOAT32, 'y': PointField.FLOAT32,
                    'z': PointField.FLOAT32, 't': PointField.UINT32,
                    'ring': PointField.UINT16}
        if (message.header.frame_id != self.settings['lidar_frame'] or
                any(name not in fields or fields[name].datatype != kind for name, kind in required.items())):
            self.fail_input('Expected original os_lidar XYZ, t:uint32 nanoseconds, ring:uint16')
            return
        if (message.point_step <= 0 or message.row_step < message.width*message.point_step
                or len(message.data) < message.row_step*message.height):
            self.fail_input('Invalid input cloud layout')
            return
        dtype = '>u4' if message.is_bigendian else '<u4'
        offsets = np.ndarray((message.height, message.width), dtype=dtype,
                             buffer=message.data, offset=fields['t'].offset,
                             strides=(message.row_step, message.point_step))
        duration = float(offsets.max(initial=0))*1e-9
        if not 0 < duration <= 0.25:
            self.fail_input(f'Unexpected per-point scan duration {duration}s')
            return
        if (stamp < self.first_forwarded_imu or self.latest_imu is None
                or self.latest_imu < stamp+duration):
            self.counters['scans_skipped_imu_coverage'] += 1
            return
        self.cloud_pub.publish(message)
        self.counters['scans_forwarded'] += 1

    def on_odometry(self, message):
        if self.failed or message.header.frame_id != self.map_frame or message.child_frame_id != 'lio_imu':
            return
        stamp = stamp_seconds(message.header.stamp)
        with self.stats_lock:
            if self.last_odom_stamp is not None and stamp <= self.last_odom_stamp:
                if stamp < self.last_odom_stamp:
                    self.fail_input('Estimator pose timestamp moved backward; restart mapping.')
                return
            try:
                rotation = rotation_matrix(message.pose.pose.orientation)
            except ValueError as error:
                self.fail_input(str(error))
                return
            output = copy.deepcopy(message)
            output.child_frame_id = self.sensor_frame
            position = message.pose.pose.position
            offset = rotation @ self.imu_from_sensor
            output.pose.pose.position.x = position.x + float(offset[0])
            output.pose.pose.position.y = position.y + float(offset[1])
            output.pose.pose.position.z = position.z + float(offset[2])
            # FAST-LIO reports world-frame small-angle pose errors [xyz,rot].
            # Move the covariance to the sensor origin using its lever arm.
            x, y, z = offset
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            jacobian = np.eye(6)
            jacobian[:3, 3:] = -skew
            covariance = np.array(message.pose.covariance).reshape(6, 6)
            output.pose.covariance = (jacobian@covariance@jacobian.T).ravel().tolist()
            # Linear velocity comes from the estimator; angular velocity/uncertainty
            # are not supplied, so this twist is not intended as an EKF fusion input.
            output.twist.covariance = (np.eye(6)*1e6).ravel().tolist()
            self.odom_pub.publish(output)
            transform = TransformStamped()
            transform.header = output.header
            transform.child_frame_id = self.sensor_frame
            transform.transform.translation.x = output.pose.pose.position.x
            transform.transform.translation.y = output.pose.pose.position.y
            transform.transform.translation.z = output.pose.pose.position.z
            transform.transform.rotation = output.pose.pose.orientation
            self.tf.sendTransform(transform)
            pose = PoseStamped(header=output.header, pose=output.pose.pose)
            self.path.header.stamp = output.header.stamp
            self.path.poses.append(pose)
            if len(self.path.poses) > self.settings['max_path_poses']:
                self.path.poses.pop(0)
            self.last_odom_stamp = stamp
            if self.first_odom_stamp is None:
                self.first_odom_stamp = stamp
            p, q = output.pose.pose.position, output.pose.pose.orientation
            self.latest_pose = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
            self.trajectory_writer.writerow([f'{stamp:.9f}', *self.latest_pose])
            self.counters['poses'] += 1
            if self.counters['poses'] % 10 == 0:
                self.trajectory_file.flush()

    def on_registered_cloud(self, message):
        if self.failed or message.header.frame_id != self.map_frame:
            return
        try:
            points = cloud_array(message)
        except ValueError as error:
            self.fail_input(str(error))
            return
        keys = np.floor(points[:, :3]/self.settings['voxel_size_m']).astype(np.int64)
        unique, indices = np.unique(keys, axis=0, return_index=True)
        with self.stats_lock:
            capacity = self.settings['max_map_voxels']
            for key, index in zip(unique, indices):
                cell = tuple(key)
                if cell in self.voxels:
                    continue
                if len(self.voxels) >= capacity:
                    if not self.voxel_limit_reached:
                        self.get_logger().warning('Map voxel capacity reached; existing map retained. '
                                                  'Start another mapping segment for a longer route.')
                    self.voxel_limit_reached = True
                    break
                self.voxels[cell] = points[index].copy()
            self.counters['registered_scans'] += 1
            now = time.monotonic()
            if now-self.last_map_publish >= self.settings['map_publish_period']:
                self.publish_map(message.header)
                self.last_map_publish = now

    def publish_map(self, header):
        points = np.array(list(self.voxels.values()), dtype='<f4').reshape(-1, 4)
        message = PointCloud2()
        message.header = header
        message.header.frame_id = self.map_frame
        message.height, message.width = 1, len(points)
        message.fields = [PointField(name=name, offset=4*i, datatype=PointField.FLOAT32, count=1)
                          for i, name in enumerate(('x', 'y', 'z', 'intensity'))]
        message.point_step, message.row_step = 16, 16*len(points)
        message.is_dense = True
        message.data = points.tobytes()
        self.map_pub.publish(message)
        self.path_pub.publish(self.path)

    def save_map(self):
        with self.stats_lock:
            points = np.array(list(self.voxels.values()), dtype='<f4').reshape(-1, 4)
            report = {'backend': 'FAST-LIO2', 'map_frame': self.map_frame,
                      'pose_child_frame': self.sensor_frame, 'georeferenced': False,
                      'loop_closure': False, 'clock_alignment': 'host-derived; unverified',
                      'initialization': 'low-motion IMU gate; not independently verified stationary',
                      'voxel_size_m': self.settings['voxel_size_m'],
                      'map_point_count': len(points), 'voxel_limit_reached': self.voxel_limit_reached,
                      'first_pose_stamp': self.first_odom_stamp, 'last_pose_stamp': self.last_odom_stamp,
                      'last_pose': self.latest_pose, 'failed_input': self.failed,
                      'counters': dict(self.counters), 'parameters': self.settings}
            self.trajectory_file.flush()
        if len(points):
            header = ('# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n'
                      'TYPE F F F F\nCOUNT 1 1 1 1\n'
                      f'WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n'
                      f'POINTS {len(points)}\nDATA binary\n').encode()
            temporary = self.output/'pointcloud_map.pcd.tmp'
            with temporary.open('wb') as stream:
                stream.write(header)
                stream.write(points.tobytes())
            temporary.replace(self.output/'pointcloud_map.pcd')
        temporary = self.output/'mapping_report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(self.output/'mapping_report.json')
        return len(points)

    def on_save(self, request, response):
        try:
            count = self.save_map()
            response.success = count > 0 and not self.failed
            response.message = f'Saved {count} map points to {self.output}'
        except OSError as error:
            response.success, response.message = False, str(error)
        return response

    def publish_status(self):
        status = DiagnosticStatus()
        status.name = 'camera_lidar_bev/local_mapping'
        status.hardware_id = 'Ouster LiDAR and internal IMU'
        status.level = DiagnosticStatus.ERROR if self.failed else (
            DiagnosticStatus.OK if self.counters['poses'] else DiagnosticStatus.WARN)
        status.message = ('Invalid input: restart required' if self.failed else
                          'Local odometry and mapping' if self.counters['poses'] else
                          'Waiting for estimator' if self.ready else 'Waiting for low-motion initialization')
        values = {**self.counters, 'initialized_input': self.ready, 'map_voxels': len(self.voxels),
                  'voxel_limit_reached': self.voxel_limit_reached, 'georeferenced': False}
        status.values = [KeyValue(key=key, value=str(value)) for key, value in values.items()]
        array = DiagnosticArray(status=[status])
        array.header.stamp = self.get_clock().now().to_msg()
        self.diag_pub.publish(array)


def main(args=None):
    # Stop callbacks before invalidating the ROS context. The default ROS
    # signal handler tears it down while map/IMU worker callbacks may publish.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    stop_requested = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: stop_requested.set())
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_requested.set())
    node = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        node = MappingAdapter()
        executor.add_node(node)
        while rclpy.ok() and not stop_requested.is_set() and not node.failed:
            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        # ros2 launch forwards SIGINT after our process group already receives
        # it. A second interrupt must not abort map export during cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        executor.shutdown(timeout_sec=2.0)
        if node is not None:
            count = node.save_map()
            print(f'Saved {count} voxel map points and trajectory to {node.output}', flush=True)
            node.trajectory_file.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 1 if node is not None and node.failed else 0


if __name__ == '__main__':
    main()
