#!/usr/bin/python3
"""Display existing ROS camera/LiDAR predictions together, without new inference.

Images and their camera predictions use exact matching source headers. LiDAR
headers may differ by --pair-delta seconds. Every layer expires independently.
The renderer is shared with the saved-data preview in combined_bev_display.py.
"""

import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import math
import os
import signal
import time

# These operations multiply many XYZ rows by tiny 3x3 matrices. Large BLAS
# thread pools cost more than the calculation and starve the odometry process.
# Scope the limits to this CPU viewer; native inference/FAST-LIO keep their own.
for _thread_variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                         'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '1'

import cv2
import numpy as np
import yaml

cv2.setNumThreads(1)


def header_stamp(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def header_key(header):
    return header_stamp(header), header.frame_id


def rotation_matrix(quaternion):
    q = np.asarray([quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm < 1e-8:
        raise ValueError("Box/TF quaternion is invalid")
    x, y, z, w = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def best_result(detection):
    if not detection.results:
        return None
    valid = [result.hypothesis for result in detection.results
             if math.isfinite(result.hypothesis.score)]
    return max(valid, key=lambda result: result.score) if valid else None


def camera_2d_boxes(message):
    boxes = []
    for item in message.detections:
        result = best_result(item)
        if result is None:
            continue
        b = item.bbox
        x, y = b.center.position.x, b.center.position.y
        bounds = [x-b.size_x/2, y-b.size_y/2, x+b.size_x/2, y+b.size_y/2]
        if not np.isfinite(bounds).all() or b.size_x <= 0 or b.size_y <= 0:
            continue
        boxes.append({"class": result.class_id, "score": float(result.score),
                      "native_xyxy_px": bounds})
    return boxes


def boxes_3d(message):
    boxes = []
    for item in message.detections:
        result = best_result(item)
        if result is None:
            continue
        b = item.bbox
        position = [b.center.position.x, b.center.position.y, b.center.position.z]
        size = [b.size.x, b.size.y, b.size.z]
        if not np.isfinite(position+size).all() or min(size) <= 0:
            continue
        try:
            rotation = rotation_matrix(b.center.orientation)
        except ValueError:
            continue
        q = b.center.orientation
        quaternion = np.asarray([q.x, q.y, q.z, q.w], float)
        quaternion /= np.linalg.norm(quaternion)
        box = {"class": result.class_id, "score": float(result.score)}
        box.update(os_sensor_xyz_m=position, size_lwh_m=size,
                   quaternion_xyzw=quaternion.tolist(),
                   yaw_rad=math.atan2(rotation[1, 0], rotation[0, 0]))
        boxes.append(box)
    return boxes


def world_detections_from_source(message, world_from_sensor, scan_end_ns):
    """Copy camera-matched LiDAR boxes into ENU at their native scan end.

    The caller must supply the exact fused source scan and a fresh paired pose.
    Camera-only detections never enter this function. IDs/hypotheses/dimensions
    survive; all metric poses and their covariance rotate into the world frame.
    """
    from scipy.spatial.transform import Rotation
    if message.header.frame_id != "os_sensor":
        raise ValueError("World detections require source boxes in os_sensor")
    transform = np.asarray(world_from_sensor, float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("World detection transform is invalid")
    world_rotation = Rotation.from_matrix(transform[:3, :3])
    output = deepcopy(message)
    output.header.frame_id = "enu"
    output.header.stamp.sec, output.header.stamp.nanosec = divmod(int(scan_end_ns), 10**9)
    covariance_rotation = np.zeros((6, 6))
    covariance_rotation[:3, :3] = transform[:3, :3]
    covariance_rotation[3:, 3:] = transform[:3, :3]

    def transform_pose(pose, source_pose):
        q = source_pose.orientation
        quaternion = np.asarray([q.x, q.y, q.z, q.w], float)
        position = np.asarray([source_pose.position.x, source_pose.position.y, source_pose.position.z], float)
        if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-8:
            raise ValueError("World detection source pose is invalid")
        position = transform[:3, :3]@position+transform[:3, 3]
        quaternion = (world_rotation*Rotation.from_quat(quaternion)).as_quat()
        pose.position.x, pose.position.y, pose.position.z = map(float, position)
        q = pose.orientation
        q.x, q.y, q.z, q.w = map(float, quaternion)

    for detection, source_detection in zip(output.detections, message.detections):
        dimensions = [detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z]
        if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
            raise ValueError("World detection dimensions are invalid")
        detection.header = deepcopy(output.header)
        transform_pose(detection.bbox.center, source_detection.bbox.center)
        for result, source_result in zip(detection.results, source_detection.results):
            transform_pose(result.pose.pose, source_result.pose.pose)
            covariance = np.asarray(source_result.pose.covariance, float).reshape(6, 6)
            if not np.isfinite(covariance).all():
                raise ValueError("World detection covariance is invalid")
            result.pose.covariance = (covariance_rotation@covariance@covariance_rotation.T).reshape(-1).tolist()
    return output


def mark_production_matches(raw_boxes, fused_boxes):
    """Only the production fusion subset grants a world camera-match label."""
    for box in raw_boxes:
        box['camera_matched'] = any(
            box['class'] == fused['class'] and
            np.allclose(box['os_sensor_xyz_m'], fused['os_sensor_xyz_m'], atol=1e-5, rtol=0) and
            np.allclose(box['size_lwh_m'], fused['size_lwh_m'], atol=1e-5, rtol=0)
            for fused in fused_boxes)
    return raw_boxes


def world_source_pair_reason(fused, cloud, pose, scan_end_ns, now_ns, wall_now, maximum_age=.5):
    """Return a clearing reason unless every world box has current source data."""
    if fused is None or cloud is None or pose is None:
        return 'missing camera-matched source, native scan, or paired localization'
    for record in (fused, cloud):
        source_age = (now_ns-header_stamp(record[0].header))/1e9
        receipt_age = wall_now-record[1]
        if not (-.05 <= source_age <= maximum_age and 0 <= receipt_age <= maximum_age):
            return 'expired source detections or native scan'
    if fused[0].header.frame_id != 'os_sensor':
        return 'source detections are not in os_sensor'
    if cloud[0].header.frame_id not in ('os_lidar', 'os_sensor'):
        return 'unsupported native scan frame'
    if header_stamp(fused[0].header) != header_stamp(cloud[0].header):
        return 'camera-matched detections and native scan have different source stamps'
    if pose.get('query_stamp_ns') != scan_end_ns:
        return 'localization was not queried at this native scan end'
    return None


def select_source_cloud(records, image_header, maximum_delta_ns, pose_available,
                        preferred_source_stamp_ns=None):
    """Choose a fresh caller-filtered scan that has completed localization.

    A newer scan can arrive before its ICP pose. Search the whole existing
    camera pairing window so that it does not hide a completed older scan.
    The caller retains the usual source/wall freshness gates; pose_available
    retains exact/short-bracket scan-end pairing with no extrapolation.
    """
    image_stamp = header_stamp(image_header)
    candidates = [record for record in records
                  if record[0].header.frame_id in ('os_lidar', 'os_sensor') and
                  abs(header_stamp(record[0].header)-image_stamp) <= maximum_delta_ns]
    if not candidates:
        return None

    def score(record):
        stamp = header_stamp(record[0].header)
        ready = bool(pose_available(record))
        return (int(ready), int(stamp == preferred_source_stamp_ns),
                -abs(stamp-image_stamp), stamp)

    return max(candidates, key=score)


def point_cloud_xyzi(message, sensor_from_lidar, max_points=None):
    """Read display samples before XYZ work, honoring row padding/field names.

    This cap affects only the viewer. Native scan-end timing still inspects the
    full original t field, and detector/localizer subscriptions are unchanged.
    """
    if message.header.frame_id not in ("os_lidar", "os_sensor"):
        raise ValueError(f"Unsupported point-cloud frame: {message.header.frame_id}")
    types = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}
    fields = {field.name: field for field in message.fields}
    if message.width == 0 or message.height == 0:
        return np.empty((0, 4), np.float32)
    if (message.row_step < message.width*message.point_step or
            len(message.data) < message.row_step*message.height):
        raise ValueError("PointCloud2 data/row_step is incomplete")
    endian = ">" if message.is_bigendian else "<"

    def column(name):
        field = fields.get(name)
        if field is None or field.count != 1 or field.datatype not in types:
            raise ValueError(f"Missing or unsupported PointCloud2 field: {name}")
        dtype = np.dtype(endian+types[field.datatype])
        if field.offset+dtype.itemsize > message.point_step:
            raise ValueError(f"PointCloud2 field exceeds point_step: {name}")
        return np.ndarray((message.height, message.width), dtype=dtype,
                          buffer=message.data, offset=field.offset,
                          strides=(message.row_step, message.point_step)).reshape(-1)

    if max_points is not None and max_points < 1:
        raise ValueError("Display sample limit must be positive")
    stride = max(1, math.ceil(message.width*message.height/max_points)) if max_points else 1
    xyz = np.column_stack([column(name)[::stride] for name in ("x", "y", "z")]).astype(np.float32)
    good = np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) >= .5)
    intensity_name = "reflectivity" if "reflectivity" in fields else "intensity"
    intensity = column(intensity_name)[::stride].astype(np.float32) if intensity_name in fields else np.zeros(len(xyz), np.float32)
    if intensity_name == "reflectivity":
        intensity /= 255.
    xyz, intensity = xyz[good], intensity[good]
    if message.header.frame_id == "os_lidar":
        xyz = xyz @ sensor_from_lidar[:3, :3].T + sensor_from_lidar[:3, 3]
    # Measured ego-vehicle envelope, as used by the native LiDAR detector.
    own_vehicle = ((xyz[:, 0] >= -2.131) & (xyz[:, 0] <= 2.350) &
                   (xyz[:, 1] >= -.905) & (xyz[:, 1] <= .885) &
                   (xyz[:, 2] >= -1.8) & (xyz[:, 2] <= 0))
    return np.column_stack([xyz[~own_vehicle], intensity[~own_vehicle]]).astype(np.float32)


def map_cloud_xyz(message, max_points):
    """Read the bounded, persistent XYZ map without applying scan/ego filters."""
    if message.header.frame_id != "map":
        raise ValueError(f"Unsupported local-map frame: {message.header.frame_id}")
    if message.width == 0 or message.height == 0:
        return np.empty((0, 3), np.float32)
    if (message.point_step <= 0 or message.row_step < message.width*message.point_step or
            len(message.data) < message.row_step*message.height):
        raise ValueError("Local-map PointCloud2 data/row_step is incomplete")
    fields = {field.name: field for field in message.fields}
    types = {7: "f4", 8: "f8"}
    endian = ">" if message.is_bigendian else "<"
    stride = max(1, math.ceil(message.width*message.height/max_points))
    columns = []
    for name in ("x", "y", "z"):
        field = fields.get(name)
        if field is None or field.count != 1 or field.datatype not in types:
            raise ValueError(f"Missing or unsupported local-map field: {name}")
        dtype = np.dtype(endian+types[field.datatype])
        if field.offset < 0 or field.offset+dtype.itemsize > message.point_step:
            raise ValueError(f"Local-map field exceeds point_step: {name}")
        column = np.ndarray((message.height, message.width), dtype=dtype,
                            buffer=message.data, offset=field.offset,
                            strides=(message.row_step, message.point_step)).reshape(-1)
        columns.append(column[::stride])
    xyz = np.column_stack(columns).astype(np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)]


def map_from_sensor_odometry(message):
    """Validate the mapping bridge's pose contract before using its transform."""
    if message.header.frame_id != "map" or message.child_frame_id != "os_sensor":
        raise ValueError("Mapping odometry must describe map -> os_sensor")
    pose = message.pose.pose
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix(pose.orientation)
    transform[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
    if not np.isfinite(transform).all():
        raise ValueError("Mapping odometry position is invalid")
    return transform


def mapping_in_sensor(map_xyz, path_xyz, map_from_sensor):
    """Invert T_map_from_os_sensor; do not confuse world XYZ with local XYZ."""
    rotation, translation = map_from_sensor[:3, :3], map_from_sensor[:3, 3]
    return dict(points=(np.asarray(map_xyz)-translation)@rotation,
                path=(np.asarray(path_xyz)-translation)@rotation)


def arguments():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intrinsics", type=Path,
                        default=root/"config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml")
    parser.add_argument("--extrinsics", type=Path,
                        default=root/"config/calibration/imported_front_001/T_B_os_sensor.candidate.yaml")
    parser.add_argument("--range-m", type=float, default=80.)
    parser.add_argument("--lateral-m", type=float, default=30.)
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--stale-after", type=float, default=.5)
    parser.add_argument("--pair-delta", type=float, default=.15)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--max-map-points", type=int, default=100000)
    optional_path = lambda value: Path(value) if value else None
    map_mode = parser.add_mutually_exclusive_group()
    map_mode.add_argument("--saved-map", type=optional_path,
                        help="saved map directory; only for its original source recording")
    map_mode.add_argument("--localize-map", type=optional_path,
                         help="frozen saved map placed by independent live localization")
    parser.add_argument("--replay-bag", type=optional_path,
                        help="original source bag; required with --saved-map")
    parser.add_argument("--gps-alignment", type=optional_path,
                        help="GPS alignment directory for this saved map")
    parser.add_argument("--status-topic", default="/camera_lidar_bev/combined_view/status")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--display", action="store_true", dest="display", default=True)
    display.add_argument("--no-display", action="store_false", dest="display")
    parser.add_argument("--image-topic", default="/camera/front/image_raw")
    parser.add_argument("--info-topic", default="/camera/front/camera_info")
    parser.add_argument("--points-topic", default="/points")
    parser.add_argument("--camera-2d-topic", default="/camera_lidar_bev/detections_2d")
    parser.add_argument("--lidar-3d-topic", default="/camera_lidar_bev/lidar_detections_3d")
    parser.add_argument("--fused-topic", default="/camera_lidar_bev/camera_lidar_detections_3d")
    parser.add_argument("--output-topic", default="/camera_lidar_bev/combined_view")
    parser.add_argument("--road-topic", default="/camera_lidar_bev/road_mask")
    parser.add_argument("--lane-topic", default="/camera_lidar_bev/lane_mask")
    parser.add_argument("--mapping-odometry-topic", default="/camera_lidar_bev/mapping/odometry")
    parser.add_argument("--mapping-map-topic", default="/camera_lidar_bev/mapping/map")
    parser.add_argument("--mapping-path-topic", default="/camera_lidar_bev/mapping/path")
    parser.add_argument("--localization-odometry-topic", default="/camera_lidar_bev/localization/odometry")
    parser.add_argument("--localization-status-topic", default="/camera_lidar_bev/localization/status")
    parser.add_argument("--world-detections-topic", default="/camera_lidar_bev/localization/detections_3d")
    parser.add_argument("--path-tracking", type=lambda value: value.lower() == 'true', default=False)
    args, ros_args = parser.parse_known_args()
    for name in ("range_m", "lateral_m", "fps", "stale_after", "pair_delta"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.max_points < 1:
        parser.error("--max-points must be positive")
    if args.max_map_points < 1:
        parser.error("--max-map-points must be positive")
    if args.saved_map and not args.replay_bag:
        parser.error("--saved-map requires --replay-bag to check source provenance")
    if args.gps_alignment and not (args.saved_map or args.localize_map):
        parser.error("--gps-alignment requires --saved-map or --localize-map")
    if args.path_tracking and not (args.saved_map or args.localize_map):
        parser.error("--path-tracking requires a GPS-aligned saved or localized map")
    return args, ros_args


def main():
    args, ros_args = arguments()
    import rclpy
    from cv_bridge import CvBridge
    from diagnostic_msgs.msg import DiagnosticArray
    from nav_msgs.msg import Odometry, Path as RosPath
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data, QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage
    from vision_msgs.msg import Detection2DArray, Detection3DArray
    from combined_bev_display import CombinedBevRenderer, DEFAULT_LAYERS, associations_for_sample
    from saved_map_source import SavedMapSource, scan_end_nanoseconds
    from live_map_source import LiveMapSource, transform_boxes
    from scipy.spatial.transform import Rotation
    from path_tracking_overlay import paired_overlay, source_transform

    saved_map = (SavedMapSource(args.saved_map, args.replay_bag, args.gps_alignment)
                 if args.saved_map else None)
    live_map = (LiveMapSource(args.localize_map, args.gps_alignment, max_overview_points=50000)
                if args.localize_map else None)
    if args.path_tracking and saved_map is not None and saved_map.enu_from_map is None:
        raise ValueError("path tracking requires the saved map GPS alignment")

    extrinsics = yaml.safe_load(args.extrinsics.read_text())
    camera_from_sensor = np.asarray(extrinsics["transforms"]["T_camera_from_os_sensor"]["matrix"], float).reshape(4, 4)
    sensor_from_lidar = np.asarray(extrinsics["observed_driver_transform"]["matrix"], float).reshape(4, 4)
    expected = yaml.safe_load(args.intrinsics.read_text())
    camera_frame = extrinsics["transforms"]["T_camera_from_os_sensor"]["target_frame"]

    class CombinedView(Node):
        def __init__(self):
            super().__init__("combined_bev_view")
            self.bridge = CvBridge()
            self.streams = {name: deque(maxlen=24 if name != "points" else 12)
                            for name in ("image", "info", "points", "camera2d", "lidar3d", "fused", "road", "lane", "odometry")}
            self.map_points = np.empty((0, 3), np.float32)
            self.map_header = None
            self.map_path = np.empty((0, 3), np.float32)
            self.map_path_stamps = np.empty(0, np.int64)
            self.mapping_seen = saved_map is not None or live_map is not None
            self.localizer_status = None
            self.latest_valid_pose_stamp_ns = None
            self.latest_valid_pose_received_wall = None
            self.latest_pose_callback_ros_ns = None
            self.pose_receive_count = 0
            self.pose_reject_count = 0
            self.latest_pose_reject_reason = None
            self.point_receive_count = 0
            self.latest_point_source_gap_ms = None
            self.selection_pose_cache = {}
            self.selection_pose_queries = 0
            self.selection_pose_cache_hits = 0
            self.update_started_wall = None
            self.scan_end_cache = {}
            self.live_pose = None
            self.live_mapping_key = None
            self.live_mapping = None
            self.world_publish_key = None
            self.world_box_count = 0
            self.mapping_clock_ns = None
            self.saved_pose_key = None
            self.saved_pose = None
            self.saved_pose_reason = None
            self.saved_mapping_key = None
            self.saved_mapping = None
            self.tracking_record = None
            self.tracking_pose_key = None
            self.sensor_from_lidar = sensor_from_lidar.copy()
            self.transform_source = "saved driver TF"
            self.renderer = None
            self.calibration_key = None
            self.points_key = None
            self.points = None
            self.point_count = 0
            self.sampled_display_point_count = 0
            self.diagnostics = {}
            self.layers = DEFAULT_LAYERS.copy()
            self.window = "Camera 2D + LiDAR 3D + road/lane | BEV"
            self.window_opened = False
            self.viewer_subscriptions = []
            # Rendering may span a scan period. Preserve the small bounded
            # acquisition queue so the scan awaiting its ICP result reaches
            # our source cache instead of being overwritten by the next scan.
            cloud_qos = QoSProfile(depth=4 if live_map is not None else 1,
                                   reliability=ReliabilityPolicy.RELIABLE)
            pose_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
            for name, topic, kind in (("image", args.image_topic, Image), ("info", args.info_topic, CameraInfo),
                                     ("points", args.points_topic, PointCloud2), ("camera2d", args.camera_2d_topic, Detection2DArray),
                                     ("lidar3d", args.lidar_3d_topic, Detection3DArray),
                                     ("fused", args.fused_topic, Detection3DArray),
                                     ("road", args.road_topic, Image), ("lane", args.lane_topic, Image)):
                self.viewer_subscriptions.append(self.create_subscription(
                    kind, topic, lambda message, stream=name: self.receive(stream, message),
                    cloud_qos if name == "points" else qos_profile_sensor_data))
            tf_qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                reliability=ReliabilityPolicy.RELIABLE)
            self.viewer_subscriptions.append(self.create_subscription(TFMessage, "/tf_static", self.receive_tf, tf_qos))
            map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
            if live_map is not None:
                self.viewer_subscriptions.append(self.create_subscription(
                    Odometry, args.localization_odometry_topic, self.receive_localization, pose_qos))
                self.viewer_subscriptions.append(self.create_subscription(
                    String, args.localization_status_topic, self.receive_localizer_status, cloud_qos))
            elif saved_map is None:
                self.viewer_subscriptions.append(self.create_subscription(
                    Odometry, args.mapping_odometry_topic,
                    lambda message: self.receive("odometry", message), cloud_qos))
                self.viewer_subscriptions.append(self.create_subscription(
                    PointCloud2, args.mapping_map_topic, self.receive_map, map_qos))
                self.viewer_subscriptions.append(self.create_subscription(
                    RosPath, args.mapping_path_topic, self.receive_path, map_qos))
            self.viewer_subscriptions.append(self.create_subscription(
                DiagnosticArray, "/camera_lidar_bev/diagnostics", self.receive_diagnostics, 5))
            self.viewer_subscriptions.append(self.create_subscription(
                DiagnosticArray, "/camera_lidar_bev/mapping/diagnostics", self.receive_diagnostics, 5))
            self.publisher = self.create_publisher(Image, args.output_topic, 1)
            self.status_publisher = self.create_publisher(String, args.status_topic, 1)
            self.tracking_pose_publisher = (self.create_publisher(
                Odometry, "/camera_lidar_bev/path_tracking/source_pose", 10) if args.path_tracking else None)
            if args.path_tracking:
                self.viewer_subscriptions.append(self.create_subscription(
                    String, "/camera_lidar_bev/path_tracking/status", self.receive_tracking, 10))
            self.world_publisher = (self.create_publisher(Detection3DArray, args.world_detections_topic, 1)
                                    if live_map is not None else None)
            self.timer = self.create_timer(1/args.fps, self.update)
            self.get_logger().info(f"Publishing {args.output_topic} at up to {args.fps:g} Hz; no model inference in this viewer")
            if saved_map is not None:
                self.get_logger().info(f"Saved source map loaded: {saved_map.point_count:,} points, "
                                       f"{len(saved_map.stamps):,} poses; "
                                       f"GPS aligned: {saved_map.enu_from_map is not None}")
            if live_map is not None:
                self.get_logger().info(f"Frozen live map loaded: {live_map.point_count:,} points; "
                                       "positions come only from independent scan-to-map localization")

        def receive_localization(self, message):
            pose = message.pose.pose
            q = pose.orientation
            accepted, reason = live_map.add_pose(
                header_stamp(message.header),
                [pose.position.x, pose.position.y, pose.position.z],
                [q.x, q.y, q.z, q.w],
                parent_frame=message.header.frame_id, child_frame=message.child_frame_id,
                source_now_ns=self.get_clock().now().nanoseconds, received_wall=time.monotonic())
            if not accepted:
                self.pose_reject_count += 1
                self.latest_pose_reject_reason = reason
                self.get_logger().warning(f"Live localization pose: {reason}", throttle_duration_sec=3.)
            else:
                self.pose_receive_count += 1
                self.latest_valid_pose_stamp_ns = header_stamp(message.header)
                self.latest_valid_pose_received_wall = time.monotonic()
                self.latest_pose_callback_ros_ns = self.get_clock().now().nanoseconds

        def receive_tracking(self, message):
            try:
                fields = json.loads(message.data)
                if not isinstance(fields, dict):
                    raise ValueError('tracking status must be an object')
                self.tracking_record = fields, time.monotonic()
            except (ValueError, TypeError):
                self.tracking_record = None

        def tracking_for_source(self, mapping):
            if self.tracking_pose_publisher is None:
                return None, None
            transform, stamp = source_transform(saved_map, live_map, self.saved_pose, self.live_pose)
            if mapping is None:
                transform, stamp = None, None
            if transform is not None and stamp != self.tracking_pose_key:
                pose = Odometry()
                pose.header.frame_id, pose.child_frame_id = 'enu', 'os_sensor'
                pose.header.stamp.sec, pose.header.stamp.nanosec = divmod(int(stamp), 10**9)
                p, q = pose.pose.pose.position, pose.pose.pose.orientation
                p.x, p.y, p.z = map(float, transform[:3, 3])
                quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
                q.x, q.y, q.z, q.w = map(float, quaternion)
                self.tracking_pose_publisher.publish(pose)
                self.tracking_pose_key = stamp
            return paired_overlay(self.tracking_record, transform, stamp,
                                  self.get_clock().now().nanoseconds, time.monotonic())

        def receive_localizer_status(self, message):
            try:
                fields = json.loads(message.data)
                if not isinstance(fields, dict):
                    raise ValueError("status is not an object")
            except (ValueError, TypeError):
                fields = {'accepted': False, 'status': 'invalid localizer status'}
            self.localizer_status = fields, time.monotonic()
            if not fields.get('accepted', False):
                live_map.invalidate(str(fields.get('status', 'localization unavailable')))
                self.live_pose = None

        def localization_ready(self):
            return (self.localizer_status is not None and
                    time.monotonic()-self.localizer_status[1] <= 1. and
                    self.localizer_status[0].get('accepted') is True)

        def scan_end_for_record(self, record):
            key = header_key(record[0].header)
            if key not in self.scan_end_cache:
                self.scan_end_cache[key] = scan_end_nanoseconds(record[0])
                if len(self.scan_end_cache) > 24:
                    self.scan_end_cache.pop(next(iter(self.scan_end_cache)))
            return self.scan_end_cache[key]

        def receive(self, stream, message):
            if stream == "odometry":
                self.mapping_seen = True
            if stream == 'points':
                self.point_receive_count += 1
                if self.streams['points']:
                    self.latest_point_source_gap_ms = (header_stamp(message.header)-
                                                       header_stamp(self.streams['points'][-1][0].header))/1e6
            self.streams[stream].append((message, time.monotonic()))

        def receive_map(self, message):
            self.mapping_seen = True
            try:
                points = map_cloud_xyz(message, args.max_map_points)
            except ValueError as error:
                self.get_logger().warning(f"Local map: {error}", throttle_duration_sec=3.)
                self.map_points = np.empty((0, 3), np.float32)
                self.map_header = None
                return
            self.map_points = points
            self.map_header = message.header

        def receive_path(self, message):
            self.mapping_seen = True
            # Path/map persist between low-rate publications; only the pose
            # used to place them in BEV must pass the source-stamp freshness gate.
            if message.header.frame_id != "map" or any(
                    item.header.frame_id != "map" for item in message.poses):
                self.map_path = np.empty((0, 3), np.float32)
                self.map_path_stamps = np.empty(0, np.int64)
                self.get_logger().warning("Local path must use frame map", throttle_duration_sec=3.)
                return
            poses = message.poses[::max(1, math.ceil(len(message.poses)/10000))]
            self.map_path = np.asarray([[item.pose.position.x, item.pose.position.y, item.pose.position.z]
                                        for item in poses], dtype=np.float32).reshape(-1, 3)
            self.map_path_stamps = np.asarray([header_stamp(item.header) for item in poses], dtype=np.int64)

        def mapping_pose_for_source(self, header, cloud_record=None):
            if live_map is not None:
                if not self.localization_ready():
                    reason = ('expired localization status' if self.localizer_status is not None and
                              time.monotonic()-self.localizer_status[1] > 1. else
                              self.localizer_status[0].get('status') if self.localizer_status else 'waiting for localization')
                    return None, str(reason or 'expired localization')
                if cloud_record is None or not self.fresh(cloud_record):
                    return None, "live map needs a fresh native source scan"
                try:
                    end_stamp = self.scan_end_for_record(cloud_record)
                    key = (end_stamp, self.pose_receive_count, live_map.epoch)
                    if key in self.selection_pose_cache:
                        cached = self.selection_pose_cache[key]
                        if cached[0] is None or time.monotonic()-cached[0]['received_wall'] <= live_map.max_receive_age_s:
                            self.selection_pose_cache_hits += 1
                            return cached
                    self.selection_pose_queries += 1
                    result = live_map.pose_at(end_stamp,
                                              source_now_ns=self.get_clock().now().nanoseconds,
                                              wall_now=time.monotonic())
                    self.selection_pose_cache[key] = result
                    return result
                except ValueError as error:
                    return None, str(error)
            if saved_map is not None:
                if cloud_record is None or not self.fresh(cloud_record):
                    return None, "saved map needs a fresh original source scan with native timing"
                key = header_key(cloud_record[0].header)
                if key != self.saved_pose_key:
                    try:
                        end_stamp = self.scan_end_for_record(cloud_record)
                        self.saved_pose, self.saved_pose_reason = saved_map.pose_at(end_stamp)
                    except ValueError as error:
                        self.saved_pose, self.saved_pose_reason = None, str(error)
                    self.saved_pose_key = key
                return self.saved_pose, self.saved_pose_reason
            valid_records = [record for record in self.streams["odometry"]
                             if record[0].header.frame_id == "map" and record[0].child_frame_id == "os_sensor"]
            records = [record for record in valid_records if self.fresh(record)]
            if not records:
                if valid_records:
                    latest = max(valid_records, key=lambda record: header_stamp(record[0].header))
                    lag = (self.get_clock().now().nanoseconds-header_stamp(latest[0].header))/1e9
                    if lag > args.stale_after:
                        return None, f"pose lag {lag:.2f} s | freshness limit {args.stale_after:.2f} s"
                return None, "waiting/expired pose"
            pose = min(records, key=lambda record: abs(header_stamp(record[0].header)-header_stamp(header)))[0]
            delta_ns = header_stamp(pose.header)-header_stamp(header)
            if abs(delta_ns) > min(args.pair_delta, .15)*1e9:
                return None, "no pose within 150 ms of source scan"
            try:
                map_from_sensor_odometry(pose)
            except ValueError as error:
                return None, str(error)
            return pose, "source paired"

        def mapping_for_source(self, header, cloud_record=None):
            pose, reason = self.mapping_pose_for_source(header, cloud_record)
            if pose is None:
                self.live_pose = None
                return None, reason
            if live_map is not None:
                self.live_pose = pose
                key = (pose['query_stamp_ns'], live_map.epoch, args.range_m, args.lateral_m, args.max_map_points)
                if key != self.live_mapping_key:
                    self.live_mapping = live_map.mapping_for_pose(pose, args.range_m, args.lateral_m, args.max_map_points)
                    self.live_mapping_key = key
                # Per-frame world boxes must not persist when the detection
                # source disappears while this native map pose stays current.
                mapping = dict(self.live_mapping)
                mapping['live_localization'] = True
                xyz = mapping['world_position']
                return mapping, (f"LIVE scan-end pose {pose['pose_delta_ms']:+.3f} ms; "
                                 f"ENU {xyz[0]:+.1f}, {xyz[1]:+.1f}, {xyz[2]:+.1f} m; "
                                 f"{len(mapping['points']):,} local map points")
            if saved_map is not None:
                key = (pose['query_stamp_ns'], args.range_m, args.lateral_m, args.max_map_points)
                if key != self.saved_mapping_key:
                    self.saved_mapping = saved_map.mapping_for_pose(
                        pose, args.range_m, args.lateral_m, args.max_map_points)
                    self.saved_mapping_key = key
                mapping = self.saved_mapping
                xyz = pose['transform'][:3, 3]
                status = (f"saved scan-end pose {pose['pose_delta_ms']:+.3f} ms; "
                          f"map XYZ {xyz[0]:+.1f}, {xyz[1]:+.1f}, {xyz[2]:+.1f} m; "
                          f"{len(mapping['points']):,}/{mapping['visible_point_count']:,} local map points")
                return mapping, status
            delta_ns = header_stamp(pose.header)-header_stamp(header)
            transform = map_from_sensor_odometry(pose)
            source_stamp = header_stamp(pose.header)
            path = self.map_path[self.map_path_stamps <= source_stamp]
            # The chosen scan-end pose is the trajectory endpoint even when
            # the persistent Path publication runs more slowly than odometry.
            path = np.vstack((path, transform[:3, 3]))
            mapping = mapping_in_sensor(self.map_points, path, transform)
            xyz = transform[:3, 3]
            yaw = math.degrees(math.atan2(transform[1, 0], transform[0, 0]))
            mapping.update(world_path=path, world_position=xyz,
                           world_yaw_rad=math.atan2(transform[1, 0], transform[0, 0]))
            status = (f"source paired {delta_ns/1e6:+.1f} ms; map XYZ "
                      f"{xyz[0]:+.1f}, {xyz[1]:+.1f}, {xyz[2]:+.1f} m; yaw {yaw:+.1f} deg; "
                      f"{len(self.map_points):,} cached map points")
            return mapping, status

        def mapping_readiness(self, mapping=None, pairing_reason=None):
            if live_map is not None:
                if mapping is not None:
                    return (f"LIVE LOCALIZED | {len(mapping['points']):,} map points | "
                            f"{len(mapping['world_path']):,} live poses")
                return "LIVE LOCALIZATION | "+(pairing_reason or 'waiting for fresh source-paired pose')
            if saved_map is not None:
                if mapping is not None:
                    gps = " | GPS ALIGNED" if saved_map.enu_from_map is not None else ""
                    return (f"SAVED{gps} | {len(mapping['points']):,} local points | "
                            f"{len(mapping['world_path']):,} recorded poses")
                return "SAVED | "+(pairing_reason or "waiting for original source scan")
            if mapping is not None:
                return f"ready | {len(self.map_points):,} map points | {len(mapping['world_path']):,} poses"
            diagnostic = self.diagnostics.get("camera_lidar_bev/local_mapping")
            if diagnostic and time.monotonic()-diagnostic[1] <= 2.5:
                fields = diagnostic[0]
                if fields.get("_level") == "2":
                    return fields.get("_message", "mapping input error")
                if fields.get("initialized_input", "False").lower() != "true":
                    return ("waiting for low-motion IMU | "
                            f"{fields.get('scans_skipped_initialization', '0')} scans held")
                return pairing_reason or fields.get("_message", "waiting for mapping pose")
            if self.mapping_seen:
                return pairing_reason or "waiting for fresh source-paired pose"
            return "no mapping data | start replay with --mapping"

        def receive_diagnostics(self, message):
            for status in message.status:
                fields = {entry.key: entry.value for entry in status.values}
                fields.update(_message=status.message, _level=str(status.level))
                self.diagnostics[status.name] = (fields, time.monotonic())

        def receive_tf(self, message):
            for item in message.transforms:
                if item.header.frame_id == "os_sensor" and item.child_frame_id == "os_lidar":
                    try:
                        transform = np.eye(4)
                        transform[:3, :3] = rotation_matrix(item.transform.rotation)
                        t = item.transform.translation
                        transform[:3, 3] = [t.x, t.y, t.z]
                        if not np.isfinite(transform).all():
                            continue
                        self.sensor_from_lidar = transform
                        self.transform_source = "received /tf_static"
                    except ValueError:
                        continue

        def fresh(self, record):
            message, received = record
            age = time.monotonic()-received
            source_age = (self.get_clock().now().nanoseconds-header_stamp(message.header))/1e9
            return 0 <= age <= args.stale_after and -.05 <= source_age <= args.stale_after

        def exact(self, name, header):
            for record in reversed(self.streams[name]):
                if header_key(record[0].header) == header_key(header) and self.fresh(record):
                    return record
            return None

        def closest(self, name, header, required_frame=None):
            records = [record for record in self.streams[name] if self.fresh(record)
                       and (required_frame is None or record[0].header.frame_id == required_frame)]
            if not records:
                return None
            best = min(records, key=lambda record: abs(header_stamp(record[0].header)-header_stamp(header)))
            return best if abs(header_stamp(best[0].header)-header_stamp(header)) <= args.pair_delta*1e9 else None

        def cloud_for_lidar(self, lidar):
            if lidar is None:
                return None
            # The detector retains the raw source stamp while changing the frame
            # from os_lidar to os_sensor. Match the stamp before transforming.
            stamp = header_stamp(lidar[0].header)
            return next((record for record in reversed(self.streams["points"])
                         if self.fresh(record) and header_stamp(record[0].header) == stamp), None)

        def paired_lidar(self, header):
            records = [record for record in self.streams["lidar3d"]
                       if self.fresh(record) and record[0].header.frame_id == "os_sensor"
                       and abs(header_stamp(record[0].header)-header_stamp(header)) <= args.pair_delta*1e9
                       and self.cloud_for_lidar(record) is not None]
            if records:
                return min(records, key=lambda record: abs(header_stamp(record[0].header)-header_stamp(header)))
            return self.closest("lidar3d", header, "os_sensor")

        def source_for_image(self, header):
            """Share the same cloud choice between image scoring and rendering."""
            lidar = self.paired_lidar(header)
            if live_map is None:
                cloud = self.cloud_for_lidar(lidar)
                return lidar, cloud if cloud is not None else self.closest('points', header)
            preferred_stamp = header_stamp(lidar[0].header) if lidar is not None else None
            cloud = select_source_cloud(
                [record for record in self.streams['points'] if self.fresh(record)],
                header, args.pair_delta*1e9,
                lambda record: self.mapping_pose_for_source(record[0].header, record)[0] is not None,
                preferred_stamp)
            if cloud is not None:
                stamp = header_stamp(cloud[0].header)
                same_source = [record for record in self.streams['lidar3d']
                               if self.fresh(record) and record[0].header.frame_id == 'os_sensor' and
                               header_stamp(record[0].header) == stamp]
                if same_source:
                    lidar = same_source[-1]
            return lidar, cloud

        def select_image(self):
            images = [record for record in self.streams["image"] if self.fresh(record)
                      and record[0].header.frame_id == camera_frame and self.exact("info", record[0].header)]
            if not images:
                return None
            # Dense raw scans arrive after their acquisition headers. Select a
            # still-fresh image with completed LiDAR/cloud and exact camera labels
            # instead of advancing the image beyond the available scan pair.
            # The same 150 ms camera/LiDAR pair limit still applies.
            def score(record):
                header = record[0].header
                lidar, cloud = self.source_for_image(header)
                source_header = cloud[0].header if cloud else lidar[0].header if lidar else None
                map_ready = (self.layers["mapping"] and source_header is not None and
                             self.mapping_pose_for_source(source_header, cloud)[0] is not None)
                # Prefer a completed mapping pair among the same fresh images,
                # retaining exact CameraInfo and both existing clock/pair gates.
                return (int(map_ready),
                        4*int(cloud is not None)+int(self.exact("camera2d", header) is not None)+
                        2*int(self.exact("road", header) is not None and self.exact("lane", header) is not None),
                        header_stamp(header))
            return max(images, key=score)

        def blank(self, reason):
            canvas = np.zeros((900, 1920, 3), np.uint8)
            cv2.putText(canvas, "Camera + LiDAR / BEV", (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (240, 240, 240), 2)
            cv2.putText(canvas, reason, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, .7, (180, 210, 245), 1)
            cv2.putText(canvas, "Waiting for fresh image and exactly paired CameraInfo", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, .65, (190, 190, 190), 1)
            state = "ON" if self.layers["mapping"] else "OFF"
            cv2.putText(canvas, f"MAP {state} [9] | {self.mapping_readiness()}", (40, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (220, 200, 100), 1)
            return canvas

        def publish_world_source(self, fused=None, cloud=None, pose=None):
            if self.world_publisher is None:
                return {}
            metrics = {'world_detection_count': 0, 'world_detection_frame': 'enu',
                       'world_detection_semantics': 'production camera-matched LiDAR subset only',
                       'world_detection_topic': args.world_detections_topic,
                       'world_detection_source_topic': args.fused_topic,
                       'world_detection_source_start_ns': None,
                       'world_detection_source_end_ns': None,
                       'world_detection_pose_stamp_ns': None,
                       'world_detection_pose_delta_ms': None}
            end_stamp = self.scan_end_for_record(cloud) if cloud is not None else None
            reason = world_source_pair_reason(fused, cloud, pose, end_stamp,
                                             self.get_clock().now().nanoseconds,
                                             time.monotonic(), args.stale_after)
            metrics['world_detection_state'] = reason or 'source-paired camera-matched boxes'
            if reason is None:
                output = world_detections_from_source(fused[0], pose['transform'], end_stamp)
                metrics.update(world_detection_count=len(output.detections),
                               world_detection_source_start_ns=header_stamp(cloud[0].header),
                               world_detection_source_end_ns=end_stamp,
                               world_detection_pose_stamp_ns=pose['pose_stamp_ns'],
                               world_detection_pose_delta_ms=pose['pose_delta_ms'])
            else:
                # Empty messages retire world boxes immediately; image-only
                # camera labels retain their pixel coordinates in the UI.
                output = Detection3DArray()
                output.header.frame_id = 'enu'
                output.header.stamp = self.get_clock().now().to_msg()
            self.world_publisher.publish(output)
            self.world_box_count = len(output.detections)
            return metrics

        def show(self, canvas, source_header=None, metrics=None):
            output = self.bridge.cv2_to_imgmsg(canvas, encoding="bgr8")
            if source_header is not None:
                output.header = source_header
            else:
                output.header.stamp = self.get_clock().now().to_msg()
                output.header.frame_id = camera_frame
            self.publisher.publish(output)
            status = {'map_ready': False, **(metrics or {})}
            status.update(image_stamp_ns=header_stamp(output.header),
                          image_frame=output.header.frame_id,
                          map_layer_on=self.layers["mapping"],
                          saved_map_directory=str(saved_map.directory) if saved_map else None,
                          localized_map_directory=str(live_map.directory) if live_map else None,
                          original_source_bag=str(saved_map.source_bag) if saved_map else None,
                          map_mode="independent live localization" if live_map else "saved source trajectory" if saved_map else "live odometry",
                          gps_aligned=bool(live_map or (saved_map and saved_map.enu_from_map is not None)))
            if live_map is not None:
                status.update(self.localization_buffer_status(source_header, status.get('cloud_stamp_ns')))
            self.status_publisher.publish(String(data=json.dumps(status, allow_nan=False)))
            if args.display:
                if not self.window_opened:
                    cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(self.window, 1600, 900)
                    self.window_opened = True
                cv2.imshow(self.window, canvas)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord("q")) or cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    rclpy.shutdown()
                elif key in range(ord("1"), ord("1")+len(self.layers)):
                    name = list(self.layers)[key-ord("1")]
                    self.layers[name] = not self.layers[name]

        def localization_buffer_status(self, image_header, selected_cloud_stamp=None):
            now_ns, wall_now = self.get_clock().now().nanoseconds, time.monotonic()
            images = [record for record in self.streams['image'] if self.fresh(record)]
            points = [record for record in self.streams['points'] if self.fresh(record)]
            candidates = []
            for record in points:
                pose, reason = self.mapping_pose_for_source(record[0].header, record)
                start = header_stamp(record[0].header)
                try:
                    end = self.scan_end_for_record(record)
                except ValueError:
                    end = None
                candidates.append({'start_ns': start, 'end_ns': end,
                                   'source_age_s': (now_ns-start)/1e9,
                                   'receipt_age_s': wall_now-record[1],
                                   'camera_delta_ms': (start-header_stamp(image_header))/1e6 if image_header else None,
                                   'pose_ready': pose is not None, 'pose_reason': reason})
            latest_cloud = self.streams['points'][-1] if self.streams['points'] else None
            latest_start = header_stamp(latest_cloud[0].header) if latest_cloud else None
            try:
                latest_end = self.scan_end_for_record(latest_cloud) if latest_cloud else None
            except ValueError:
                latest_end = None
            return {'viewer_clock_ns': now_ns,
                    'latest_valid_pose_stamp_ns': self.latest_valid_pose_stamp_ns,
                    'latest_valid_pose_age_s': (now_ns-self.latest_valid_pose_stamp_ns)/1e9
                                              if self.latest_valid_pose_stamp_ns else None,
                    'latest_valid_pose_receipt_age_s': wall_now-self.latest_valid_pose_received_wall
                                                      if self.latest_valid_pose_received_wall else None,
                    'latest_pose_callback_source_age_s': (self.latest_pose_callback_ros_ns-self.latest_valid_pose_stamp_ns)/1e9
                                                        if self.latest_valid_pose_stamp_ns else None,
                    'viewer_pose_receive_count': self.pose_receive_count,
                    'viewer_pose_reject_count': self.pose_reject_count,
                    'viewer_latest_pose_reject_reason': self.latest_pose_reject_reason,
                    'latest_cloud_start_ns': latest_start, 'latest_cloud_end_ns': latest_end,
                    'selected_cloud_source_age_s': (now_ns-selected_cloud_stamp)/1e9 if selected_cloud_stamp else None,
                    'selected_cloud_end_ns': next((row['end_ns'] for row in candidates
                                                   if row['start_ns'] == selected_cloud_stamp), None),
                    'selected_image_age_s': (now_ns-header_stamp(image_header))/1e9 if image_header else None,
                    'fresh_image_count': len(images),
                    'fresh_exact_image_info_count': sum(self.exact('info', record[0].header) is not None for record in images),
                    'fresh_cloud_count': len(points),
                    'fresh_pose_paired_cloud_count': sum(row['pose_ready'] for row in candidates),
                    'viewer_points_receive_count': self.point_receive_count,
                    'viewer_points_last_source_gap_ms': self.latest_point_source_gap_ms,
                    'viewer_cloud_history_start_ns': [header_stamp(record[0].header) for record in self.streams['points']],
                    'viewer_pose_queries_this_update': self.selection_pose_queries,
                    'viewer_pose_cache_hits_this_update': self.selection_pose_cache_hits,
                    'viewer_update_ms_so_far': (wall_now-self.update_started_wall)*1000 if self.update_started_wall else None,
                    'cloud_pose_candidates': candidates,
                    'localizer_status_age_s': wall_now-self.localizer_status[1] if self.localizer_status else None,
                    'localizer_status_stamp_ns': self.localizer_status[0].get('stamp_ns') if self.localizer_status else None}

        def update(self):
            self.update_started_wall = time.monotonic()
            self.selection_pose_cache.clear()
            self.selection_pose_queries = 0
            self.selection_pose_cache_hits = 0
            clock_ns = self.get_clock().now().nanoseconds
            if self.mapping_clock_ns is not None and clock_ns < self.mapping_clock_ns-50_000_000:
                # A replay loop creates a new local-map history. Never combine
                # poses from the new loop with a persistent previous-loop map.
                self.streams["odometry"].clear()
                self.map_points = np.empty((0, 3), np.float32)
                self.map_header = None
                self.map_path = np.empty((0, 3), np.float32)
                self.map_path_stamps = np.empty(0, np.int64)
                self.saved_pose_key = None
                self.saved_mapping_key = None
                self.tracking_pose_key = None
                self.tracking_record = None
                self.scan_end_cache.clear()
                self.live_pose = None
                if live_map is not None:
                    live_map.observe_clock(clock_ns, reset_on_rewind=bool(self.get_parameter('use_sim_time').value))
            self.mapping_clock_ns = clock_ns
            image_record = self.select_image()
            if image_record is None:
                metrics = self.publish_world_source()
                self.show(self.blank("No current camera pair; expired boxes and points cleared"), metrics=metrics)
                return
            image = image_record[0]
            info_record = self.exact("info", image.header)
            if info_record is None or not self.fresh(image_record):
                self.show(self.blank("Camera pair expired during source selection; waiting for a fresh pair"),
                          metrics=self.publish_world_source())
                return
            info = info_record[0]
            try:
                if (image.width != info.width or image.height != info.height or
                        info.width != expected["image_width"] or info.height != expected["image_height"]):
                    raise ValueError("Image/CameraInfo dimensions differ from the native camera configuration")
                k, d = np.asarray(info.k, float).reshape(3, 3), np.asarray(info.d, float)
                if (not np.isfinite(k).all() or not np.isfinite(d).all() or k[0, 0] <= 0 or k[1, 1] <= 0
                        or info.distortion_model != "plumb_bob" or len(d) != 5):
                    raise ValueError("CameraInfo has unsupported raw-image intrinsics/distortion")
                if (info.binning_x not in (0, 1) or info.binning_y not in (0, 1) or
                        info.roi.x_offset or info.roi.y_offset or
                        info.roi.width not in (0, info.width) or info.roi.height not in (0, info.height)):
                    raise ValueError("Cropped/binned CameraInfo is unsupported by the raw-image viewer")
                calibration = {"width": info.width, "height": info.height, "K": k.reshape(-1).tolist(),
                               "D": d.tolist(), "T": camera_from_sensor.reshape(-1).tolist()}
                calibration_key = (info.width, info.height, tuple(info.k), tuple(info.d))
                if self.renderer is None or calibration_key != self.calibration_key:
                    self.renderer = CombinedBevRenderer(calibration, forward_m=args.range_m,
                                                        lateral_m=args.lateral_m, max_points=args.max_points)
                    self.calibration_key = calibration_key
                camera2d = self.exact("camera2d", image.header)
                road = self.exact("road", image.header)
                lane = self.exact("lane", image.header)
                lidar, cloud = self.source_for_image(image.header)
                points = None
                if cloud is not None:
                    points_key = (header_key(cloud[0].header), self.sensor_from_lidar.tobytes())
                    if points_key != self.points_key:
                        self.points = point_cloud_xyzi(cloud[0], self.sensor_from_lidar, args.max_points)
                        self.point_count = cloud[0].width*cloud[0].height
                        self.sampled_display_point_count = len(self.points)
                        self.points_key = points_key
                    points = self.points
                lidar_stamp = header_stamp(lidar[0].header) if lidar else header_stamp(cloud[0].header) if cloud else 0
                sample = {"sample_id": "replay" if self.get_parameter('use_sim_time').value else "live",
                          "image_stamp_ns": header_stamp(image.header),
                          "lidar_stamp_ns": lidar_stamp,
                          "pair_delta_ms": (header_stamp(image.header)-lidar_stamp)/1e6 if lidar_stamp else None,
                          "input_points": self.point_count if points is not None else 0,
                          "camera_2d_boxes": camera_2d_boxes(camera2d[0]) if camera2d else [],
                          "lidar_boxes": boxes_3d(lidar[0]) if lidar else []}
                sample["associations"] = associations_for_sample(sample, calibration)
                fused = self.exact("fused", lidar[0].header) if lidar else None
                status = {"mode": "SAVED recording predictions" if saved_map else "LIVE existing predictions", "camera2d": "source paired" if camera2d else "waiting/expired",
                          "lidar3d": "nearby source" if lidar else "waiting/expired",
                          "cloud": ("source paired" if cloud and lidar and
                                    header_stamp(cloud[0].header) == header_stamp(lidar[0].header)
                                    else "nearby raw source" if cloud else "waiting/expired"),
                          "production_matched_boxes": len(fused[0].detections) if fused else "waiting/expired",
                          "transform": self.transform_source,
                          "road/lane": "source paired" if road and lane else "waiting/expired"}
                status["image age"] = f'{(self.get_clock().now().nanoseconds-header_stamp(image.header))/1e9:.3f} s'
                mapping = None
                mapping_reason = None
                if self.mapping_seen:
                    source_header = cloud[0].header if cloud else lidar[0].header if lidar else None
                    if source_header is not None:
                        mapping, status["local odometry"] = self.mapping_for_source(source_header, cloud)
                        mapping_reason = status["local odometry"]
                    else:
                        status["local odometry"] = "waiting for source LiDAR scan"
                        mapping_reason = status["local odometry"]
                status["mapping readiness"] = self.mapping_readiness(mapping, mapping_reason)
                status["map layer"] = "ON [9]" if self.layers["mapping"] else "OFF [9]"
                if saved_map is not None:
                    status["map source"] = saved_map.directory.name+" | original recorded trajectory"
                    if saved_map.enu_from_map is not None and mapping is not None:
                        enu = mapping["world_position"]
                        status["GPS alignment"] = (f"ENU {enu[0]:+.1f}, {enu[1]:+.1f}, {enu[2]:+.1f} m "
                                                   "| fitted GPS anchor; no online localization")
                world_metrics = {}
                if live_map is not None:
                    status['map source'] = live_map.directory.name+' | independent live localization'
                    status['world objects'] = 'production camera-matched LiDAR boxes in ENU'
                    matched_boxes = boxes_3d(fused[0]) if fused else []
                    source_boxes = mark_production_matches(sample['lidar_boxes'], matched_boxes)
                    object_pose = None
                    if (lidar is not None and cloud is not None and
                            header_stamp(lidar[0].header) == header_stamp(cloud[0].header)):
                        object_pose, _ = self.mapping_pose_for_source(lidar[0].header, cloud)
                    world_metrics = self.publish_world_source(fused, cloud, object_pose)
                    if mapping is not None and object_pose is not None:
                        mapping['world_boxes'] = transform_boxes(source_boxes, object_pose['transform'])
                    status['world matched boxes'] = world_metrics.get('world_detection_count', 0)
                for node_name, label_name in (("camera_object_detector", "YOLO"),
                                              ("lidar_vehicle_detector", "PointPillars")):
                    diagnostic = self.diagnostics.get(node_name)
                    if diagnostic and time.monotonic()-diagnostic[1] <= 2.5:
                        fields = diagnostic[0]
                        if "processing_ms" in fields:
                            status[label_name] = f'{float(fields["processing_ms"]):.1f} ms (latest diagnostic)'
                frame = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
                segmentation = None
                if road and lane:
                    segmentation = dict(road_mask=self.bridge.imgmsg_to_cv2(road[0], desired_encoding="mono8"),
                                        lane_mask=self.bridge.imgmsg_to_cv2(lane[0], desired_encoding="mono8"))
                diagnostic = self.diagnostics.get("road_lane_detector")
                if diagnostic and time.monotonic()-diagnostic[1] <= 2.5:
                    fields = diagnostic[0]
                    if "runtime_ms" in fields:
                        status["TwinLiteNet"] = f'{float(fields["runtime_ms"]):.1f} ms (latest diagnostic)'
                tracking, tracking_summary = self.tracking_for_source(mapping)
                if tracking_summary is not None:
                    status['Pure Pursuit'] = tracking_summary
                canvas = self.renderer.render(frame, sample, points=points, status=status, layers=self.layers,
                                               segmentation=segmentation, mapping=mapping, tracking=tracking,
                                               source_now_ns=self.get_clock().now().nanoseconds)
                metrics = {'map_ready': mapping is not None, 'mapping_readiness': status["mapping readiness"],
                           'cloud_stamp_ns': header_stamp(cloud[0].header) if cloud else None,
                           'camera_2d_stamp_ns': header_stamp(camera2d[0].header) if camera2d else None,
                           'lidar_3d_stamp_ns': header_stamp(lidar[0].header) if lidar else None,
                           'camera_box_count': len(sample["camera_2d_boxes"]),
                           'lidar_box_count': len(sample["lidar_boxes"]),
                           'association_count': sum(row.get('camera_index', -1) >= 0
                                                    for row in sample["associations"]),
                           'raw_point_count': self.point_count if points is not None else 0,
                           'sampled_display_point_count': self.sampled_display_point_count if points is not None else 0,
                           'display_map_point_count': len(mapping['points']) if mapping is not None else 0,
                           'recorded_path_pose_count': len(mapping['world_path']) if mapping is not None else 0}
                metrics.update(world_metrics)
                if args.path_tracking:
                    metrics['path_tracking'] = tracking['fields'] if tracking is not None else None
                    metrics['path_tracking_state'] = tracking_summary
                if live_map is not None and mapping is not None and self.live_pose is not None:
                    pose = self.live_pose
                    metrics.update(scan_end_stamp_ns=pose['query_stamp_ns'],
                                   pose_stamp_ns=pose['pose_stamp_ns'], pose_delta_ms=pose['pose_delta_ms'],
                                   pose_method=pose['method'], pose_bracket_ms=pose['bracket_ms'],
                                   pose_world=pose['transform'].tolist(),
                                   enu_position_m=mapping['world_position'].tolist(),
                                   live_path_pose_count=len(mapping['world_path']),
                                   map_total_point_count=live_map.point_count,
                                   world_inset_box_count=len(mapping.get('world_boxes', [])))
                if saved_map is not None and mapping is not None:
                    pose = self.saved_pose
                    metrics.update(scan_end_stamp_ns=pose['query_stamp_ns'],
                                   pose_stamp_ns=pose['pose_stamp_ns'], pose_delta_ms=pose['pose_delta_ms'],
                                   pose_method=pose['method'], pose_bracket_ms=pose['bracket_ms'],
                                   source_pose_xyz_m=pose['transform'][:3, 3].tolist(),
                                   map_total_point_count=saved_map.point_count,
                                   map_visible_point_count=mapping['visible_point_count'],
                                   enu_position_m=mapping['world_position'].tolist() if saved_map.enu_from_map is not None else None)
                self.show(canvas, image.header, metrics)
            except Exception as error:
                self.get_logger().warning(f"Combined view: {error}", throttle_duration_sec=3.)
                self.show(self.blank(str(error)), image.header, self.publish_world_source())

    rclpy.init(args=ros_args)
    node = CombinedView()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
