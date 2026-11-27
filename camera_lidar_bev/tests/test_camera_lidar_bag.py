"""Offline recording QC regressions; no ROS nodes, graph, sensors or bag playback."""

import importlib.util
import math
from pathlib import Path
import struct
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[1]/"tools/check_camera_lidar_bag.py"
spec = importlib.util.spec_from_file_location("camera_lidar_bag_qc", path)
qc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc)


class RecordedTiming(unittest.TestCase):
    def test_camera_sensor_cadence_and_record_receipts_are_separate(self):
        headers = [1_000_000_000+round(i*1e9/30) for i in range(60)]
        receipts = [stamp+100_000_000+i*2_000_000 for i, stamp in enumerate(headers)]
        result = qc.summarize_timing(headers, receipts, 30)
        self.assertEqual(result["saved_messages"], 60)
        self.assertAlmostEqual(result["header_rate_hz"], 30, places=6)
        self.assertLess(result["saved_receipt_rate_hz"], 29)
        self.assertEqual(result["gaps_above_1_5_nominal_period"], 0)
        self.assertFalse(result["gap_slots_are_confirmed_drops"])

    def test_long_gap_is_nominal_unfilled_slots_not_confirmed_loss(self):
        headers = [1_000_000_000, 1_050_000_000, 1_100_000_000, 1_300_000_000]
        result = qc.summarize_timing(headers, headers, 20)
        self.assertEqual(result["gaps_above_1_5_nominal_period"], 1)
        self.assertEqual(result["nominal_unfilled_slots_estimate"], 3)
        self.assertAlmostEqual(result["positive_header_interval_s"]["max"], .2)
        self.assertFalse(result["gap_slots_are_confirmed_drops"])

    def test_clock_steps_duplicates_and_zero_headers_remain_visible(self):
        result = qc.summarize_timing([0, 100, 100, 50, 200], [100, 200, 300, 400, 500])
        self.assertEqual(result["nonpositive_header_stamps"], 1)
        self.assertEqual(result["duplicate_adjacent_header_stamps"], 1)
        self.assertEqual(result["backward_header_steps"], 1)
        self.assertEqual(result["distinct_header_stamps"], 4)
        self.assertIsNone(qc.summarize_timing([100], [200])["header_rate_hz"])


class PairingAndClockMeaning(unittest.TestCase):
    def test_reordered_info_pairs_exactly_but_wrong_frame_never_pairs(self):
        images = [("front", 10), ("front", 20), ("front", 30)]
        infos = [("front", 30), ("front", 10), ("wrong_frame", 20)]
        result = qc.pair_camera_headers(images, infos)
        self.assertEqual(result["exact_header_pairs"], 2)
        self.assertEqual(result["unpaired_images"], 1)
        self.assertEqual(result["orphan_camera_infos"], 1)
        self.assertFalse(result["all_images_have_exact_camera_info"])

    def test_duplicate_image_cannot_reuse_one_info_as_two_saved_pairs(self):
        result = qc.pair_camera_headers([("front", 10), ("front", 10)], [("front", 10)])
        self.assertEqual(result["exact_header_pairs"], 1)
        self.assertEqual(result["unpaired_images"], 1)
        self.assertFalse(qc.pair_camera_headers([], [])["all_images_have_exact_camera_info"])
        self.assertTrue(qc.pair_camera_headers([("front", 10)], [("front", 10)])["all_images_have_exact_camera_info"])

    def test_nearest_cross_sensor_headers_report_signed_offsets_without_sync_claim(self):
        result = qc.nearest_header_offsets([100_000_000, 150_000_000, 190_000_000],
                                           [110_000_000, 180_000_000])
        self.assertEqual(result["samples"], 3)
        self.assertAlmostEqual(result["camera_header_minus_nearest_lidar_s"]["median"], -.01)
        self.assertAlmostEqual(result["absolute_header_delta_s"]["max"], .03)
        self.assertFalse(result["physical_synchronization_validated"])
        self.assertEqual(qc.nearest_header_offsets([10], [])["samples"], 0)


class NativeCalibration(unittest.TestCase):
    def setUp(self):
        self.calibration = {
            "camera_matrix": {"data": [1000., 0., 512., 0., 1000., 384., 0., 0., 1.]},
            "distortion_coefficients": {"data": [-.1, .01, 0., 0., 0.]},
            "rectification_matrix": {"data": [1., 0., 0., 0., 1., 0., 0., 0., 1.]},
            "projection_matrix": {"data": [1000., 0., 512., 0., 0., 1000., 384., 0., 0., 0., 1., 0.]}}
        self.info = SimpleNamespace(width=1024, height=768, distortion_model="plumb_bob",
                                    k=list(self.calibration["camera_matrix"]["data"]),
                                    d=list(self.calibration["distortion_coefficients"]["data"]),
                                    r=list(self.calibration["rectification_matrix"]["data"]),
                                    p=list(self.calibration["projection_matrix"]["data"]),
                                    binning_x=0, binning_y=0,
                                    roi=SimpleNamespace(x_offset=0, y_offset=0, width=0, height=0, do_rectify=False))

    def test_unchanged_native_info_accepts_but_rescale_or_crop_does_not(self):
        self.assertTrue(qc.camera_info_matches(self.info, self.calibration))
        self.info.width = 640
        self.assertFalse(qc.camera_info_matches(self.info, self.calibration))
        self.info.width = 1024
        self.info.k[0] += 1
        self.assertFalse(qc.camera_info_matches(self.info, self.calibration))
        self.info.k[0] -= 1
        self.info.roi.x_offset = 1
        self.assertFalse(qc.camera_info_matches(self.info, self.calibration))

    def test_nonfinite_and_missing_coefficients_cannot_look_stable(self):
        self.info.r[0] = math.nan
        self.assertFalse(qc.camera_info_matches(self.info, self.calibration))
        self.info.r[0] = 1
        self.info.d.pop()
        self.assertFalse(qc.camera_info_matches(self.info, self.calibration))
        self.info.d.append(0)
        self.calibration["camera_matrix"]["data"][0] = math.nan
        self.assertFalse(qc.camera_info_matches(self.info, self.calibration))


try:
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import Image, PointCloud2, PointField
    from std_msgs.msg import Header
    ros_available = True
except ImportError:
    ros_available = False


@unittest.skipUnless(ros_available, "Source ROS to exercise actual serialization, without initializing a node")
class ActualCdrLayouts(unittest.TestCase):
    def test_actual_image_serialization_layout_and_truncation(self):
        h = Header(frame_id="camera_front_optical_frame")
        h.stamp.sec, h.stamp.nanosec = 2, 123
        image = Image(header=h, width=2, height=1, encoding="bgr8", step=6, data=bytes(range(6)))
        serialized = serialize_message(image)
        frame, stamp, layout = qc.image_layout(serialized, 2, 1)
        self.assertEqual((frame, stamp), ("camera_front_optical_frame", 2_000_000_123))
        self.assertEqual(layout["payload_bytes"], 6)
        with self.assertRaises(ValueError):
            qc.image_layout(serialized, 1024, 768)
        with self.assertRaises(ValueError):
            qc.image_layout(serialized[:-3], 2, 1)

    def test_actual_original_cloud_retains_t_and_rejects_float_time_schema(self):
        h = Header(frame_id="os_lidar")
        h.stamp.sec, h.stamp.nanosec = 3, 456
        data = bytearray(96)
        struct.pack_into("<I", data, 20, 3)
        struct.pack_into("<I", data, 68, 7)
        fields = [PointField(name=name, offset=offset, datatype=kind, count=1)
                  for name, offset, kind in (("x", 0, 7), ("y", 4, 7), ("z", 8, 7),
                                            ("intensity", 16, 7), ("t", 20, 6), ("reflectivity", 24, 4),
                                            ("ring", 26, 4), ("ambient", 28, 4), ("range", 32, 6))]
        cloud = PointCloud2(header=h, width=2, height=1, fields=fields, point_step=48, row_step=96,
                            data=bytes(data), is_dense=False)
        frame, stamp, layout = qc.lidar_layout(serialize_message(cloud))
        self.assertEqual((frame, stamp), ("os_lidar", 3_000_000_456))
        self.assertEqual(layout["sample_point_t_uint32"], [3, 7])
        cloud.fields[4].datatype = 7
        with self.assertRaisesRegex(ValueError, "t"):
            qc.lidar_layout(serialize_message(cloud))
        cloud.fields[4].datatype = 6
        cloud.data = cloud.data[:-1]
        with self.assertRaises(ValueError):
            qc.lidar_layout(serialize_message(cloud))


if __name__ == "__main__":
    unittest.main()
