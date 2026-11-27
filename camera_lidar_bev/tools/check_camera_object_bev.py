#!/usr/bin/env python3
"""Bounded software integration check for camera labels and observed LiDAR boxes.

Uses synthetic geometry in isolated ROS domain 75 with the real candidate YAMLs.
Starts one C++ fusion child, no camera, LiDAR, detector, GPU, or display. Passing
does not establish physical calibration, learned detection accuracy, or speed.
"""

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def stamp_ns(header):
    return int(header.stamp.sec)*1_000_000_000 + int(header.stamp.nanosec)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path,
                        default=root / "install/bev_visualization/lib/bev_visualization/camera_object_bev")
    parser.add_argument("--intrinsics", type=Path,
                        default=root / "config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml")
    parser.add_argument("--extrinsics", type=Path,
                        default=root / "config/calibration/imported_front_001/T_B_os_sensor.candidate.yaml")
    parser.add_argument("--output", type=Path,
                        default=root / "evaluation/camera_object_bev_replay_001.json")
    parser.add_argument("--domain-id", type=int, default=75)
    parser.add_argument("--timeout", type=float, default=27.0)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new evidence path")
    if not 2 <= args.domain_id <= 231:
        parser.error("domain-id must be between 2 and 231, separate from live domain 1")
    if not math.isfinite(args.timeout) or not 20 <= args.timeout <= 30:
        parser.error("timeout must be between 20 and 30 seconds")
    for path in (args.executable, args.intrinsics, args.extrinsics):
        if not path.is_file():
            parser.error(f"Required built executable or calibration file is missing: {path}")

    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ.pop("ROS_LOCALHOST_ONLY", None)
    os.environ.pop("ROS_STATIC_PEERS", None)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    os.environ.setdefault("CYCLONEDDS_URI", "file://" + str(root / "config/transport/cyclonedds_large_clouds.xml"))
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/camera-lidar-bev-object-check-logs")

    import cv2
    import numpy as np
    import rclpy
    import yaml
    from diagnostic_msgs.msg import DiagnosticArray
    from geometry_msgs.msg import TransformStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
    from std_msgs.msg import Header
    from tf2_msgs.msg import TFMessage
    from vision_msgs.msg import Detection2D, Detection2DArray, Detection3DArray, ObjectHypothesisWithPose
    from visualization_msgs.msg import Marker, MarkerArray

    native = yaml.safe_load(args.intrinsics.read_text())
    external = yaml.safe_load(args.extrinsics.read_text())
    camera_frame = external["transforms"]["T_camera_from_os_sensor"]["target_frame"]
    transform = np.asarray(external["transforms"]["T_camera_from_os_sensor"]["matrix"], dtype=np.float64)
    k = np.asarray(native["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    d = np.asarray(native["distortion_coefficients"]["data"], dtype=np.float64)
    width, height = native["image_width"], native["image_height"]
    xx, yy, zz = np.meshgrid(np.linspace(6.0, 6.3, 5), np.linspace(-0.2, 0.2, 5),
                             np.linspace(-1.4, -0.2, 9), indexing="ij")
    xyz = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float32).astype(np.float64)
    source_ids = 90_001 + 13*np.arange(len(xyz), dtype=np.uint32)
    camera_xyz = xyz @ transform[:3, :3].T + transform[:3, 3]
    pixels = cv2.projectPoints(camera_xyz, np.zeros(3), np.zeros(3), k, d)[0].reshape(-1, 2)
    x_min, y_min = pixels.min(axis=0)-5
    x_max, y_max = pixels.max(axis=0)+5
    assert np.all(camera_xyz[:, 2] > 0), "Synthetic person must be in front of the camera"
    assert 0 < x_min < x_max < width and 0 < y_min < y_max < height, "Synthetic native box must fit in image"
    # Place full-only returns on exactly the same optical rays at half depth.
    occluders = (camera_xyz*0.5-transform[:3, 3]) @ transform[:3, :3]
    occluders = occluders.astype(np.float32).astype(np.float64)
    occluder_ids = 900_001 + 17*np.arange(len(occluders), dtype=np.uint32)

    full_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                           ("rgb", "<u4"), ("source_index", "<u4"),
                           ("camera_colored", "u1"), ("padding", "u1", (3,))])
    classified_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                 ("rgb", "<u4"), ("source_index", "<u4"),
                                 ("height_above_ground", "<f4"), ("classification", "u1"),
                                 ("padding", "u1", (3,))])
    base_full = np.zeros(len(xyz), dtype=full_dtype)
    base_classified = np.zeros(len(xyz), dtype=classified_dtype)
    for column, name in enumerate(("x", "y", "z")):
        base_full[name] = xyz[:, column]
        base_classified[name] = xyz[:, column]
    base_full["rgb"] = 0x123456
    base_full["source_index"] = source_ids
    base_classified["source_index"] = source_ids
    base_classified["rgb"] = 0xff3030
    base_classified["height_above_ground"] = xyz[:, 2]+1.8
    base_classified["classification"] = 3
    shadow_full = np.zeros(len(xyz)+len(occluders), dtype=full_dtype)
    shadow_full[:len(xyz)] = base_full
    for column, name in enumerate(("x", "y", "z")):
        shadow_full[name][len(xyz):] = occluders[:, column]
    shadow_full["source_index"][len(xyz):] = occluder_ids
    no_floor = base_classified.copy()
    no_floor["classification"] = 1
    no_floor["height_above_ground"] = np.nan

    full_fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("rgb", 12, 7),
                   ("source_index", 16, 6), ("camera_colored", 20, 2)]
    classified_fields = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("rgb", 12, 7),
                         ("source_index", 16, 6), ("height_above_ground", 20, 7), ("classification", 24, 2)]
    prefix = "/camera_object_software_check"
    topic = {name: prefix+"/"+name for name in ("detections_2d", "info", "full", "classified",
                                              "detections_3d", "markers", "vehicle_bev_markers", "semantic", "diagnostics")}
    command = [str(args.executable), "--ros-args",
               "-p", f"intrinsics_file:={args.intrinsics}", "-p", f"extrinsics_file:={args.extrinsics}",
               "-p", "allow_provisional_calibration:=true", "-p", f"camera_frame:={camera_frame}",
               "-p", "model_name:=software_synthetic_person", "-p", f"detections_topic:={topic['detections_2d']}",
               "-p", f"camera_info_topic:={topic['info']}", "-p", f"full_points_topic:={topic['full']}",
               "-p", f"classified_points_topic:={topic['classified']}",
               "-p", f"detections_3d_topic:={topic['detections_3d']}", "-p", f"markers_topic:={topic['markers']}",
               "-p", f"vehicle_bev_markers_topic:={topic['vehicle_bev_markers']}",
               "-p", f"semantic_points_topic:={topic['semantic']}",
               "-r", f"/camera_lidar_bev/diagnostics:={topic['diagnostics']}"]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "domain_id": args.domain_id, "software_only": True, "physical_accuracy_validated": False,
              "learned_detector_accuracy_validated": False, "real_time_speed_validated": False,
              "calibration_status": "external_candidate_unvalidated", "stages": [],
              "synthetic_points": len(xyz), "input_source_indices_nonsequential": True,
              "executable": str(args.executable),
              "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
              "executable_mtime_ns": args.executable.stat().st_mtime_ns}
    rclpy.init()
    node = rclpy.create_node("camera_object_software_verifier")
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
    tf_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pubs = {key: node.create_publisher(kind, topic[key], qos) for key, kind in
            (("detections_2d", Detection2DArray), ("info", CameraInfo),
             ("full", PointCloud2), ("classified", PointCloud2))}
    tf_pub = node.create_publisher(TFMessage, "/tf_static", tf_qos)
    received = {key: [] for key in ("detections_3d", "markers", "vehicle_bev_markers", "semantic", "diagnostics")}

    def on_message(key, message):
        received[key].append((time.monotonic(), message))
        del received[key][:-150]

    subscriptions = [node.create_subscription(kind, topic[key],
                     lambda message, key=key: on_message(key, message), qos) for key, kind in
                     (("detections_3d", Detection3DArray), ("markers", MarkerArray),
                      ("vehicle_bev_markers", MarkerArray),
                      ("semantic", PointCloud2), ("diagnostics", DiagnosticArray))]
    driver = external["observed_driver_transform"]["observed_tf"]
    tf = TransformStamped()
    tf.header.frame_id = "os_sensor"
    tf.child_frame_id = "os_lidar"
    tf.header.stamp = node.get_clock().now().to_msg()
    tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = driver["translation_m"]
    tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = driver["quaternion_xyzw"]
    tf_message = TFMessage(transforms=[tf])
    tf_pub.publish(tf_message)
    deadline = time.monotonic()+args.timeout
    phase_cloud_stamps = set()
    last_positive_stamp = 0
    empty_barrier_stamp = 0

    def header(ns, frame):
        item = Header()
        item.stamp.sec, item.stamp.nanosec = divmod(int(ns), 1_000_000_000)
        item.frame_id = frame
        return item

    def make_cloud(array, fields, cloud_header):
        cloud = PointCloud2(header=cloud_header, height=1, width=len(array), is_bigendian=False,
                            point_step=array.dtype.itemsize, row_step=len(array)*array.dtype.itemsize,
                            is_dense=True, data=array.tobytes())
        cloud.fields = [PointField(name=name, offset=offset, datatype=datatype, count=1)
                        for name, offset, datatype in fields]
        return cloud

    def send(kind):
        image_stamp = node.get_clock().now().nanoseconds-20_000_000
        cloud_stamp = image_stamp-20_000_000
        image_frame = "incorrect_optical_frame" if kind == "wrong_frame" else camera_frame
        image_header = header(image_stamp, image_frame)
        if kind == "pair_delta":
            cloud_stamp = image_stamp-300_000_000
        if kind == "delayed_classified":
            cloud_stamp = image_stamp-220_000_000
        if kind == "empty_barrier":
            cloud_stamp = empty_barrier_stamp-200_000_000
        cloud_header = header(cloud_stamp, "os_sensor")
        phase_cloud_stamps.add(cloud_stamp)
        info = CameraInfo(header=image_header, width=width, height=height, distortion_model="plumb_bob")
        info.k = native["camera_matrix"]["data"]
        info.d = native["distortion_coefficients"]["data"]
        info.r = native["rectification_matrix"]["data"]
        info.p = native["projection_matrix"]["data"]
        if kind == "bad_info":
            info.k[0] += 1.0
        detections = Detection2DArray(header=image_header)
        if kind != "empty":
            detection = Detection2D(header=image_header)
            detection.bbox.center.position.x = float((x_min+x_max)/2)
            detection.bbox.center.position.y = float((y_min+y_max)/2)
            detection.bbox.size_x = float(x_max-x_min)
            detection.bbox.size_y = float(y_max-y_min)
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = {
                "vehicle_positive": "car", "bicycle_positive": "bicycle"}.get(kind, "person")
            hypothesis.hypothesis.score = 0.91
            detection.results.append(hypothesis)
            detections.detections.append(detection)
        full = shadow_full if kind == "masked_occlusion" else base_full
        classified = no_floor if kind == "no_floor" else base_classified
        full_header = cloud_header
        if kind == "full_stamp_mismatch":
            full_header = header(cloud_stamp-7_000_000, "os_sensor")
        if kind != "empty":
            pubs["full"].publish(make_cloud(full, full_fields, full_header))
            pubs["classified"].publish(make_cloud(classified, classified_fields, cloud_header))
        if kind == "pair_delta":
            # Populate the bounded 16-image cache with current image stamps.
            # Ordinary persistent cloud delay is legitimately handled by the
            # cache and must not be treated as a pairing failure.
            for offset in range(17):
                new_header = header(image_stamp+offset*100_000, image_frame)
                next_info = copy.deepcopy(info)
                next_info.header = new_header
                next_detections = copy.deepcopy(detections)
                next_detections.header = new_header
                for item in next_detections.detections:
                    item.header = new_header
                pubs["info"].publish(next_info)
                pubs["detections_2d"].publish(next_detections)
                rclpy.spin_once(node, timeout_sec=0.003)
        else:
            pubs["info"].publish(info)
            pubs["detections_2d"].publish(detections)

    def scalar(cloud, name):
        fields = [item for item in cloud.fields if item.name == name]
        assert len(fields) == 1
        field = fields[0]
        types = {2: "u1", 4: "u2", 6: "u4", 7: "f4", 8: "f8"}
        code = "u4" if name == "rgb" else types[field.datatype]
        return np.ndarray((cloud.height, cloud.width), dtype=(">" if cloud.is_bigendian else "<")+code,
                          buffer=bytes(cloud.data), offset=field.offset,
                          strides=(cloud.row_step, cloud.point_step)).reshape(-1).copy()

    def by_stamp(key, start):
        result = {}
        for receipt, message in received[key]:
            if receipt < start:
                continue
            if key in ("markers", "vehicle_bev_markers"):
                if not message.markers:
                    continue
                message_header = message.markers[0].header
            else:
                message_header = message.header
            result[stamp_ns(message_header)] = message
        return result

    def matching_outputs(start):
        objects, markers, vehicle_markers, points = (
            by_stamp(key, start) for key in ("detections_3d", "markers", "vehicle_bev_markers", "semantic"))
        for stamp in sorted(objects.keys() & markers.keys() & vehicle_markers.keys() & points.keys(), reverse=True):
            yield stamp, objects[stamp], markers[stamp], vehicle_markers[stamp], points[stamp]

    def check_marker_clear(markers, stamp):
        assert len(markers.markers) == 1 and markers.markers[0].action == Marker.DELETEALL
        assert markers.markers[0].header.frame_id == "os_sensor"
        assert stamp_ns(markers.markers[0].header) == stamp

    def check_positive(start, expected_class="person"):
        nonlocal last_positive_stamp
        expected_id, expected_rgb = {"person": (0, 0xff00ff), "bicycle": (1, 0x00ffff),
                                     "car": (2, 0xffa500)}[expected_class]
        expected_color = ((expected_rgb >> 16 & 255)/255, (expected_rgb >> 8 & 255)/255,
                          (expected_rgb & 255)/255, 1.0)
        for stamp, objects, markers, vehicle_markers, points in matching_outputs(start):
            if stamp not in phase_cloud_stamps or not objects.detections:
                continue
            assert objects.header.frame_id == "os_sensor" and points.header.frame_id == "os_sensor"
            assert len(objects.detections) == 1
            item = objects.detections[0]
            assert stamp_ns(item.header) == stamp and item.id == ""
            assert len(item.results) == 1 and item.results[0].hypothesis.class_id == expected_class
            assert math.isclose(item.results[0].hypothesis.score, 0.91, abs_tol=1e-9)
            q = item.bbox.center.orientation
            assert (q.x, q.y, q.z, q.w) == (0.0, 0.0, 0.0, 1.0)
            assert points.height == 1 and points.point_step == 28 and points.width >= 8
            assert len(points.data) == points.row_step
            ids = scalar(points, "source_index")
            assert len(set(map(int, ids))) == len(ids) and set(map(int, ids)) <= set(map(int, source_ids))
            assert np.all(scalar(points, "class_id") == expected_id)
            assert np.all(scalar(points, "rgb") == expected_rgb)
            assert np.allclose(scalar(points, "confidence"), 0.91, atol=1e-6)
            selected = np.column_stack([scalar(points, axis) for axis in ("x", "y", "z")]).astype(np.float64)
            expected_center = (selected.min(axis=0)+selected.max(axis=0))/2
            expected_size = selected.max(axis=0)-selected.min(axis=0)
            actual_center = np.array([item.bbox.center.position.x, item.bbox.center.position.y, item.bbox.center.position.z])
            actual_size = np.array([item.bbox.size.x, item.bbox.size.y, item.bbox.size.z])
            assert np.allclose(actual_center, expected_center, atol=1e-6)
            assert np.allclose(actual_size, expected_size, atol=1e-6)
            assert np.all(selected >= xyz.min(axis=0)-1e-6) and np.all(selected <= xyz.max(axis=0)+1e-6)
            assert len(markers.markers) == 3 and markers.markers[0].action == Marker.DELETEALL
            line, text = markers.markers[1:]
            assert line.type == Marker.LINE_LIST and len(line.points) == 24
            assert text.type == Marker.TEXT_VIEW_FACING and text.text == expected_class+" 0.91"
            for marker in (line, text):
                assert marker.header.frame_id == "os_sensor" and stamp_ns(marker.header) == stamp
                assert np.allclose((marker.color.r, marker.color.g, marker.color.b, marker.color.a),
                                   expected_color, atol=1e-7, rtol=0)
                assert 0 < marker.lifetime.sec+marker.lifetime.nanosec/1e9 <= 0.5
            assert vehicle_markers.markers[0].action == Marker.DELETEALL
            assert vehicle_markers.markers[0].header.frame_id == "os_sensor"
            assert stamp_ns(vehicle_markers.markers[0].header) == stamp
            if expected_class != "car":
                check_marker_clear(vehicle_markers, stamp)
            else:
                assert len(vehicle_markers.markers) == 3
                footprint, label = vehicle_markers.markers[1:]
                assert footprint.ns == "observed_vehicle_bev_extent" and footprint.type == Marker.LINE_STRIP
                assert label.ns == "vehicle_bev_labels" and label.type == Marker.TEXT_VIEW_FACING
                assert label.text == "car 0.91"
                for marker in (footprint, label):
                    assert marker.action == Marker.ADD and marker.header.frame_id == "os_sensor"
                    assert stamp_ns(marker.header) == stamp
                    q = marker.pose.orientation
                    assert (q.x, q.y, q.z, q.w) == (0.0, 0.0, 0.0, 1.0)
                    assert np.allclose((marker.color.r, marker.color.g, marker.color.b, marker.color.a),
                                       expected_color, atol=1e-7, rtol=0)
                    assert math.isclose(marker.lifetime.sec+marker.lifetime.nanosec/1e9, 0.5, abs_tol=1e-9)
                assert math.isclose(footprint.scale.x, 0.07, abs_tol=1e-9)
                assert (footprint.pose.position.x, footprint.pose.position.y, footprint.pose.position.z) == (0.0, 0.0, 0.0)
                corners = np.asarray([(point.x, point.y, point.z) for point in footprint.points])
                assert corners.shape == (5, 3) and np.allclose(corners[0], corners[-1], atol=1e-9)
                assert np.allclose(corners[:, 2], 0.15, atol=1e-9)
                bounds = selected[:, :2].min(axis=0), selected[:, :2].max(axis=0)
                expected_corners = {(float(x), float(y)) for x in (bounds[0][0], bounds[1][0])
                                    for y in (bounds[0][1], bounds[1][1])}
                actual_corners = corners[:4, :2]
                assert len(set(map(tuple, actual_corners))) == 4
                assert all(any(np.allclose(corner, expected, atol=1e-6, rtol=0)
                               for expected in expected_corners) for corner in actual_corners)
                # Each edge must follow exactly one BEV axis, never cross the rectangle.
                for edge in np.diff(corners[:, :2], axis=0):
                    assert np.count_nonzero(np.abs(edge) > 1e-8) == 1
                assert np.allclose((label.pose.position.x, label.pose.position.y), expected_center[:2], atol=1e-6)
                assert math.isclose(label.pose.position.z, 0.17, abs_tol=1e-9)
                assert math.isclose(label.scale.z, 0.35, abs_tol=1e-9)
            last_positive_stamp = stamp
            return {"cloud_stamp_ns": stamp, "semantic_points": int(points.width),
                    "observed_center_m": actual_center.tolist(), "observed_size_m": actual_size.tolist(),
                    "vehicle_bev_markers": len(vehicle_markers.markers),
                    "vehicle_bev_footprint_validated": expected_class == "car"}
        return None

    def check_empty(start, phase_start_ns):
        for stamp, objects, markers, vehicle_markers, points in matching_outputs(start):
            if objects.detections and stamp in phase_cloud_stamps:
                raise AssertionError("Unsupported/failing phase emitted a positive 3D object")
            # Reject queued clears from the prior phase; require a new source or
            # fresh failure-clear header and all four outputs at the same stamp.
            if stamp <= last_positive_stamp or stamp < phase_start_ns-50_000_000:
                continue
            if objects.detections:
                continue
            assert objects.header.frame_id == "os_sensor" and points.header.frame_id == "os_sensor"
            assert points.width == 0 and len(points.data) == 0
            check_marker_clear(markers, stamp)
            check_marker_clear(vehicle_markers, stamp)
            return {"clear_stamp_ns": stamp, "empty_detections": True, "empty_semantic_cloud": True,
                    "markers_delete_all": True, "vehicle_bev_markers_delete_all": True}
        return None

    process = None
    with tempfile.TemporaryFile() as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            # All input subscribers must be discovered before bounded phase traffic.
            discovery_deadline = min(deadline, time.monotonic()+5)
            while time.monotonic() < discovery_deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                tf_pub.publish(tf_message)
                if all(pub.get_subscription_count() for pub in pubs.values()):
                    break
                if process.poll() is not None:
                    raise RuntimeError("Fusion child exited during discovery")
            else:
                raise TimeoutError("Isolated ROS discovery did not complete")

            for kind, positive in (("positive", True), ("delayed_classified", True), ("empty", False),
                                   ("empty_barrier", False), ("wrong_frame", False),
                                   ("bad_info", False), ("pair_delta", False), ("no_floor", False),
                                   ("masked_occlusion", False), ("full_stamp_mismatch", False),
                                   ("recovery", True), ("bicycle_positive", True), ("vehicle_positive", True)):
                phase_cloud_stamps.clear()
                start = time.monotonic()
                start_ns = node.get_clock().now().nanoseconds
                phase_deadline = min(deadline, start+1.25)
                next_send = start
                evidence = None
                # Allow rings from the previous stage to age outside the pair
                # window before accepting an unavailable-state clear.
                minimum_observation = 0.45 if kind in ("pair_delta", "full_stamp_mismatch") else 0.15
                if kind == "delayed_classified":
                    minimum_observation = 0.65
                if kind == "empty":
                    minimum_observation = 0.25
                while time.monotonic() < phase_deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Fusion child exited during "+kind)
                    now = time.monotonic()
                    if now >= next_send:
                        send(kind)
                        next_send = now+0.045
                    rclpy.spin_once(node, timeout_sec=0.01)
                    if time.monotonic()-start < minimum_observation:
                        continue
                    observation_start = start+0.55 if kind == "delayed_classified" else start
                    expected_class = {"vehicle_positive": "car", "bicycle_positive": "bicycle"}.get(kind, "person")
                    evidence = check_positive(observation_start, expected_class) if positive else check_empty(start, start_ns)
                    if evidence and kind == "delayed_classified":
                        diagnostics = [status for receipt, array in received["diagnostics"] if receipt >= start
                                       for status in array.status if status.name == "camera_object_bev"]
                        values = {item.key: item.value for item in diagnostics[-1].values} if diagnostics else {}
                        if int(values.get("cloud_header_ns", "0")) not in phase_cloud_stamps:
                            evidence = None
                        else:
                            assert float(values["header_delta_s"]) <= 0.04, (
                                "Cached alignment should choose near-zero delta; actual="+values["header_delta_s"])
                            assert float(values["source_age_at_processing_s"]) >= 0.17, "Expected older cached camera image"
                            evidence["cached_image_pair_delta_s"] = float(values["header_delta_s"])
                            evidence["cached_image_source_age_s"] = float(values["source_age_at_processing_s"])
                    if evidence:
                        break
                if not evidence:
                    raise AssertionError("Missing expected output during "+kind)
                if kind == "empty":
                    empty_barrier_stamp = evidence["clear_stamp_ns"]
                report["stages"].append(dict(name=kind, passed=True, **evidence))
                print(kind+": passed", flush=True)

            # Leave an actual vehicle rectangle without new input and require a fresh
            # explicit clear; marker lifetime alone is insufficient.
            phase_cloud_stamps.clear()
            start = time.monotonic()
            start_ns = node.get_clock().now().nanoseconds
            clear = None
            while time.monotonic() < min(deadline, start+1.5):
                rclpy.spin_once(node, timeout_sec=0.03)
                if time.monotonic()-start < 0.85:
                    continue
                clear = check_empty(start+0.6, start_ns)
                if clear:
                    break
            if not clear:
                raise AssertionError("Stale source did not explicitly clear all outputs")
            report["stages"].append(dict(name="stale_clear", passed=True, **clear))
            print("stale_clear: passed", flush=True)
            statuses = [status for _, message in received["diagnostics"] for status in message.status
                        if status.name == "camera_object_bev"]
            assert statuses, "Missing unique fusion diagnostics"
            values = {item.key: item.value for item in statuses[-1].values}
            for key, expected in (("observed_extent", "true"), ("heading_estimated", "false"),
                                  ("tracking_enabled", "false"), ("tf_published", "false"),
                                  ("occupancy_modified", "false"), ("accuracy_validated", "false"),
                                  ("output_frame", "os_sensor")):
                assert values.get(key) == expected, (key, values.get(key))
            report["last_diagnostics"] = values
            report["passed"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            report["failed_stage"] = locals().get("kind", "discovery")
            statuses = [status for _, array in received["diagnostics"] for status in array.status
                        if status.name == "camera_object_bev"]
            if statuses:
                report["last_diagnostics"] = {item.key: item.value for item in statuses[-1].values}
            print(report["error"], flush=True)
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            if process is not None:
                report["child_returncode"] = process.returncode
            log.seek(0)
            report["node_log_tail"] = log.read().decode("utf-8", errors="replace")[-5000:]
            node.destroy_node()
            rclpy.shutdown()
            del subscriptions
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
    print("Evidence: "+str(args.output), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
