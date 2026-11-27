"""Check map-frame geometry and padded clouds without starting ROS nodes."""
import csv
import io
import math
from pathlib import Path
import struct
import sys
import threading
from types import SimpleNamespace
import unittest

import numpy as np
from nav_msgs.msg import Odometry, Path as RosPath
from sensor_msgs.msg import PointCloud2, PointField

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/bev_mapping'))
from bev_mapping.mapping_adapter import MappingAdapter, cloud_array


class Collector:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)

    def sendTransform(self, message):
        self.messages.append(message)


def adapter_state():
    trajectory = io.StringIO()
    state = SimpleNamespace(
        failed=False, map_frame='map', sensor_frame='os_sensor',
        imu_from_sensor=np.array([0.002441, 0.009725, -0.007533]),
        stats_lock=threading.RLock(), last_odom_stamp=None, first_odom_stamp=None,
        latest_pose=None, settings={'max_path_poses': 20000}, path=RosPath(),
        counters={'poses': 0}, odom_pub=Collector(), tf=Collector(),
        trajectory_writer=csv.writer(trajectory), trajectory_file=trajectory,
        errors=[])
    state.fail_input = state.errors.append
    return state


def imu_pose():
    message = Odometry()
    message.header.frame_id = 'map'
    message.header.stamp.sec = 100
    message.child_frame_id = 'lio_imu'
    p = message.pose.pose.position
    p.x, p.y, p.z = 1., 2., 3.
    q = message.pose.pose.orientation
    q.z, q.w = math.sqrt(.5), math.sqrt(.5)
    message.pose.covariance = np.diag([1., 2., 3., 4., 5., 6.]).ravel().tolist()
    return message


class MappingFrameTests(unittest.TestCase):
    def test_sensor_origin_uses_rotated_factory_lever_arm_and_covariance(self):
        state = adapter_state()
        MappingAdapter.on_odometry(state, imu_pose())
        output = state.odom_pub.messages[0]
        p = output.pose.pose.position
        # At 90-degree yaw, IMU +x points along map +y; IMU +y along map -x.
        r = np.array([-0.009725, 0.002441, -0.007533])
        np.testing.assert_allclose([p.x, p.y, p.z], np.array([1., 2., 3.])+r)
        self.assertEqual(output.header.frame_id, 'map')
        self.assertEqual(output.child_frame_id, 'os_sensor')
        covariance = np.array(output.pose.covariance).reshape(6, 6)
        rx, ry, rz = r
        self.assertAlmostEqual(covariance[0, 0], 1+rz*rz*5+ry*ry*6)
        self.assertAlmostEqual(covariance[1, 1], 2+rz*rz*4+rx*rx*6)
        self.assertAlmostEqual(covariance[2, 2], 3+ry*ry*4+rx*rx*5)
        self.assertAlmostEqual(covariance[0, 5], -ry*6)
        self.assertAlmostEqual(covariance[1, 5], rx*6)
        np.testing.assert_allclose(covariance, covariance.T)
        self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0)
        self.assertEqual(state.tf.messages[0].child_frame_id, 'os_sensor')
        self.assertEqual(state.counters['poses'], 1)
        self.assertEqual(len(state.path.poses), 1)
        row = next(csv.reader(io.StringIO(state.trajectory_file.getvalue())))
        np.testing.assert_allclose([float(value) for value in row[1:4]], [p.x, p.y, p.z])

    def test_unexpected_frames_and_duplicate_pose_do_not_create_map_poses(self):
        state = adapter_state()
        message = imu_pose()
        message.header.frame_id = 'odom'
        MappingAdapter.on_odometry(state, message)
        self.assertFalse(state.odom_pub.messages)
        message.header.frame_id = 'map'
        message.child_frame_id = 'base_link'
        MappingAdapter.on_odometry(state, message)
        self.assertFalse(state.odom_pub.messages)
        message.child_frame_id = 'lio_imu'
        MappingAdapter.on_odometry(state, message)
        MappingAdapter.on_odometry(state, message)
        self.assertEqual(state.counters['poses'], 1)
        message.header.stamp.sec = 99
        MappingAdapter.on_odometry(state, message)
        self.assertEqual(state.counters['poses'], 1)
        self.assertTrue(state.errors)

    def test_registered_cloud_preserves_padding_and_big_endian_coordinates(self):
        cloud = PointCloud2()
        cloud.height, cloud.width = 2, 2
        cloud.point_step, cloud.row_step = 20, 48
        cloud.is_bigendian = True
        cloud.fields = [PointField(name=name, offset=index*4,
                                   datatype=PointField.FLOAT32, count=1)
                        for index, name in enumerate(('x', 'y', 'z', 'intensity'))]
        values = [(1., 2., 3., 4.), (5., 6., 7., 8.),
                  (9., 10., 11., 12.), (float('nan'), 14., 15., 16.)]
        data = bytearray(96)
        for index, point in enumerate(values):
            struct.pack_into('>ffff', data, (index//2)*48+(index % 2)*20, *point)
        cloud.data = bytes(data)
        np.testing.assert_allclose(cloud_array(cloud), np.array(values[:3], dtype=np.float32))


if __name__ == '__main__':
    unittest.main()
