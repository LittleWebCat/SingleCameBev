"""Source-paired world detections and the flat saved-map display contract."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from combined_bev_live import (mark_production_matches, point_cloud_xyzi, select_source_cloud, world_detections_from_source,
                               world_source_pair_reason)
from combined_bev_display import CombinedBevRenderer, MATCHED, LIDAR_3D
from live_map_source import transform_boxes
from saved_map_source import scan_end_nanoseconds


def header(frame='os_sensor', sec=100, nanosec=0):
    return NS(frame_id=frame, stamp=NS(sec=sec, nanosec=nanosec))


def pose(position, quaternion):
    return NS(position=NS(x=position[0], y=position[1], z=position[2]),
              orientation=NS(x=quaternion[0], y=quaternion[1], z=quaternion[2], w=quaternion[3]))


def source_message():
    center = pose([8., -2., .4], Rotation.from_euler('z', .35).as_quat())
    hypothesis = NS(hypothesis=NS(class_id='car', score=.87),
                    pose=NS(pose=deepcopy(center), covariance=np.diag([1., 2., 3., 4., 5., 6.]).ravel().tolist()))
    detection = NS(header=header(), id='object_7', results=[hypothesis],
                   bbox=NS(center=center, size=NS(x=4.5, y=1.8, z=1.4)))
    return NS(header=header(), detections=[detection])


class CombinedLiveLocalizationTests(unittest.TestCase):
    def test_display_sampling_preserves_padded_named_fields_and_full_native_timing(self):
        width, height, point_step = 7, 2, 24
        row_step = width*point_step+16
        payload = bytearray(row_step*height)
        fields = [NS(name=name, datatype=kind, offset=offset, count=1)
                  for name, kind, offset in [('x', 7, 0), ('y', 7, 4), ('z', 7, 8),
                                             ('reflectivity', 4, 16), ('t', 6, 20)]]
        indices = np.arange(width*height).reshape(height, width)
        for name, offset, dtype, values in [('x', 0, '<f4', indices+5), ('y', 4, '<f4', indices*.1),
                                            ('z', 8, '<f4', indices*.03), ('reflectivity', 16, '<u2', indices+1),
                                            ('t', 20, '<u4', indices*1_000_000)]:
            view = np.ndarray((height, width), dtype=dtype, buffer=payload,
                              offset=offset, strides=(row_step, point_step))
            view[:] = values
        cloud = NS(header=header('os_lidar'), width=width, height=height, point_step=point_step,
                   row_step=row_step, fields=fields, data=bytes(payload), is_bigendian=False)
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('z', .2).as_matrix()
        full = point_cloud_xyzi(cloud, transform)
        sampled = point_cloud_xyzi(cloud, transform, max_points=4)
        np.testing.assert_allclose(sampled, full[::4])
        self.assertLessEqual(len(sampled), 4)
        # The last native return is absent from these display samples, but
        # still supplies the pose-query stamp from the unchanged original scan.
        self.assertEqual(scan_end_nanoseconds(cloud), 100_013_000_000)

    def test_completed_older_scan_wins_over_latest_scan_without_live_pose(self):
        older = (NS(header=header('os_lidar', 100, 0)), 10.)
        latest = (NS(header=header('os_lidar', 100, 100_000_000)), 10.1)
        image = header('front_camera', 100, 100_000_000)
        selected = select_source_cloud([older, latest], image, 150_000_000,
                                       lambda record: record is older,
                                       preferred_source_stamp_ns=100_100_000_000)
        self.assertIs(selected, older)
        # Once both poses exist, the exact detector/cloud source remains
        # preferred even when a different ready scan is closer to the image.
        selected = select_source_cloud([older, latest], image, 150_000_000,
                                       lambda record: True,
                                       preferred_source_stamp_ns=100_000_000_000)
        self.assertIs(selected, older)
        # A completed pose outside the camera window cannot widen that gate.
        selected = select_source_cloud([older, latest], header('front_camera', 100, 200_000_000),
                                       150_000_000, lambda record: record is older)
        self.assertIs(selected, latest)
        self.assertIsNone(select_source_cloud([older], image, 50_000_000, lambda record: True))

    def test_full_pose_and_covariance_transform_preserves_dimensions_and_hypotheses(self):
        source = source_message()
        # ROS Python permits a result pose to alias the bbox pose; it still
        # represents one source position and must be transformed exactly once.
        source.detections[0].results[0].pose.pose = source.detections[0].bbox.center
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('xyz', [.3, -.2, 1.1]).as_matrix()
        transform[:3, 3] = [13., 25., 3.]
        output = world_detections_from_source(source, transform, 100_099_999_731)
        self.assertEqual(output.header.frame_id, 'enu')
        self.assertEqual((output.header.stamp.sec, output.header.stamp.nanosec), (100, 99_999_731))
        detection = output.detections[0]
        self.assertEqual(detection.header.frame_id, 'enu')
        self.assertEqual(detection.id, 'object_7')
        self.assertEqual(vars(detection.bbox.size), {'x': 4.5, 'y': 1.8, 'z': 1.4})
        self.assertEqual(vars(detection.results[0].hypothesis), {'class_id': 'car', 'score': .87})
        expected_xyz = transform[:3, :3]@np.array([8., -2., .4])+transform[:3, 3]
        expected_rotation = transform[:3, :3]@Rotation.from_euler('z', .35).as_matrix()
        for actual in (detection.bbox.center, detection.results[0].pose.pose):
            np.testing.assert_allclose([actual.position.x, actual.position.y, actual.position.z], expected_xyz)
            q = actual.orientation
            np.testing.assert_allclose(Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), expected_rotation, atol=1e-12)
        block = np.zeros((6, 6))
        block[:3, :3] = block[3:, 3:] = transform[:3, :3]
        np.testing.assert_allclose(np.asarray(detection.results[0].pose.covariance).reshape(6, 6),
                                   block@np.diag([1., 2., 3., 4., 5., 6.])@block.T)
        self.assertEqual(source.header.frame_id, 'os_sensor')
        self.assertEqual(source.detections[0].bbox.center.position.x, 8.)

    def test_wrong_frame_or_invalid_quaternion_cannot_publish_world_boxes(self):
        source = source_message()
        source.header.frame_id = 'camera'
        with self.assertRaises(ValueError):
            world_detections_from_source(source, np.eye(4), 100_100_000_000)
        source.header.frame_id = 'os_sensor'
        source.detections[0].bbox.center.orientation = NS(x=0., y=0., z=0., w=0.)
        with self.assertRaises(ValueError):
            world_detections_from_source(source, np.eye(4), 100_100_000_000)

    def test_missing_stale_or_different_source_clears_world_detections(self):
        fused = (source_message(), 10.)
        cloud = (NS(header=header('os_lidar')), 10.)
        paired = {'query_stamp_ns': 100_100_000_000}
        args = (100_100_000_000, 100_300_000_000, 10.3)
        self.assertIsNone(world_source_pair_reason(fused, cloud, paired, *args))
        self.assertIsNotNone(world_source_pair_reason(None, cloud, paired, *args))
        self.assertIsNotNone(world_source_pair_reason(fused, cloud, None, *args))
        self.assertIsNotNone(world_source_pair_reason(fused, cloud, paired, args[0], 101_000_000_000, 10.3))
        self.assertIsNotNone(world_source_pair_reason(fused, cloud, paired, args[0], args[1], 11.))
        cloud[0].header.stamp.nanosec = 1
        self.assertIsNotNone(world_source_pair_reason(fused, cloud, paired, *args))
        cloud[0].header.stamp.nanosec = 0
        paired['query_stamp_ns'] += 100_000_000
        self.assertIsNotNone(world_source_pair_reason(fused, cloud, paired, *args))

    def test_empty_matched_subset_clears_instead_of_publishing_raw_objects(self):
        message = source_message()
        message.detections.clear()
        output = world_detections_from_source(message, np.eye(4), 100_100_000_000)
        self.assertEqual(output.detections, [])
        raw = [{'class': 'car', 'score': .9, 'os_sensor_xyz_m': [20., 1., 0.],
                'size_lwh_m': [4., 2., 1.5], 'yaw_rad': .1}]
        self.assertFalse(mark_production_matches(raw, [])[0]['camera_matched'])
        fused = deepcopy(raw)
        self.assertTrue(mark_production_matches(raw, fused)[0]['camera_matched'])
        fused[0]['os_sensor_xyz_m'][0] += .1
        self.assertFalse(mark_production_matches(raw, fused)[0]['camera_matched'])

    def test_flat_renderer_draws_world_footprints_only_when_pose_mapping_present(self):
        calibration = {'K': [600., 0., 400., 0., 600., 300., 0., 0., 1.],
                       'D': [0.]*5, 'T': np.eye(4).ravel().tolist(), 'width': 800, 'height': 600}
        renderer = CombinedBevRenderer(calibration)
        sample = {'sample_id': 'test', 'lidar_boxes': [], 'camera_2d_boxes': [], 'associations': []}
        image = np.zeros((600, 800, 3), np.uint8)
        boxes = [{'class': 'car', 'score': .9, 'os_sensor_xyz_m': [15., 0., 0.],
                  'size_lwh_m': [4., 2., 1.5], 'yaw_rad': .4, 'camera_matched': False}]
        mapping = {'points': np.zeros((0, 3)), 'path': np.array([[0., 0., 0.]]),
                   'world_path': np.array([[0., 0., 0.], [5., 0., 0.]]),
                   'world_position': np.array([5., 0., 0.]), 'world_yaw_rad': 0.,
                   'georeferenced': True, 'live_localization': True,
                   'overview_points': np.array([[-30., -30., 0.], [30., 30., 0.]])}
        without = renderer.render(image, sample, mapping=mapping)
        mapping['world_boxes'] = transform_boxes(boxes, np.eye(4))
        with_boxes = renderer.render(image, sample, mapping=mapping)
        self.assertGreater(np.count_nonzero(np.all(with_boxes == LIDAR_3D, axis=2)),
                           np.count_nonzero(np.all(without == LIDAR_3D, axis=2)))
        no_pose = renderer.render(image, sample, mapping=None)
        self.assertEqual(np.count_nonzero(np.all(no_pose == LIDAR_3D, axis=2)),
                         np.count_nonzero(np.all(without == LIDAR_3D, axis=2)))


if __name__ == '__main__':
    unittest.main()
