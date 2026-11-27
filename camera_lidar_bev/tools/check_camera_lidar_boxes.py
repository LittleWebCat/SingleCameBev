#!/usr/bin/env python3
"""Bounded software replay of camera labels associated to complete LiDAR boxes.

Uses real candidate calibration files and independently projected synthetic box
geometry in isolated ROS domain 77. No GPU, camera or LiDAR driver is started.
Passing establishes software association behavior, not calibration or accuracy.
"""

import argparse
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
    return int(header.stamp.sec)*1_000_000_000+int(header.stamp.nanosec)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path,
                        default=root/"install/bev_visualization/lib/bev_visualization/camera_lidar_box_fusion")
    parser.add_argument("--intrinsics", type=Path,
                        default=root/"config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml")
    parser.add_argument("--extrinsics", type=Path,
                        default=root/"config/calibration/imported_front_001/T_B_os_sensor.candidate.yaml")
    parser.add_argument("--output", type=Path,
                        default=root/"evaluation/camera_lidar_boxes_replay_001.json")
    parser.add_argument("--domain-id", type=int, default=77)
    parser.add_argument("--require-camera-match", action="store_true",
                        help="verify that unmatched LiDAR boxes and their points are hidden")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a fresh evidence path")
    if not 2 <= args.domain_id <= 231:
        parser.error("domain-id must be between 2 and 231, separate from live domain 1")
    if not math.isfinite(args.timeout) or not 20 <= args.timeout <= 30:
        parser.error("timeout must be between 20 and 30 seconds")
    for path in (args.executable, args.intrinsics, args.extrinsics):
        if not path.is_file():
            parser.error(f"Required executable or candidate calibration is missing: {path}")
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ.pop("ROS_LOCALHOST_ONLY", None)
    os.environ.pop("ROS_STATIC_PEERS", None)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    os.environ.setdefault("CYCLONEDDS_URI", "file://"+str(root/"config/transport/cyclonedds_large_clouds.xml"))
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/camera-lidar-bev-camera-lidar-box-check-logs")

    import cv2
    import numpy as np
    import rclpy
    import yaml
    from diagnostic_msgs.msg import DiagnosticArray
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, PointCloud2, PointField
    from std_msgs.msg import Header
    from vision_msgs.msg import (Detection2D, Detection2DArray, Detection3D, Detection3DArray,
                                ObjectHypothesisWithPose)
    from visualization_msgs.msg import Marker, MarkerArray

    native = yaml.safe_load(args.intrinsics.read_text())
    external = yaml.safe_load(args.extrinsics.read_text())
    camera_frame = external["transforms"]["T_camera_from_os_sensor"]["target_frame"]
    transform = np.asarray(external["transforms"]["T_camera_from_os_sensor"]["matrix"], dtype=np.float64)
    k = np.asarray(native["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(native["distortion_coefficients"]["data"], dtype=np.float64)
    width, height = native["image_width"], native["image_height"]
    center, size, yaw = np.asarray((20.0, 0.0, -1.0)), np.asarray((4.4, 1.8, 1.6)), 0.4
    co, si = math.cos(yaw), math.sin(yaw)
    corners = np.asarray([(center[0]+co*sx*size[0]/2-si*sy*size[1]/2,
                           center[1]+si*sx*size[0]/2+co*sy*size[1]/2,
                           center[2]+sz*size[2]/2)
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    camera_corners = corners@transform[:3, :3].T+transform[:3, 3]
    camera_center = center@transform[:3, :3].T+transform[:3, 3]
    assert np.all(camera_corners[:, 2] > 0) and camera_center[2] > 0
    pixels = cv2.projectPoints(camera_corners, np.zeros(3), np.zeros(3), k, distortion)[0].reshape(-1, 2)
    limits = (np.maximum(pixels.min(axis=0), (0, 0)),
              np.minimum(pixels.max(axis=0), (width, height)))
    assert np.all(limits[1] > limits[0]), "Independent projected cuboid must intersect the native image"
    roi_center, roi_size = (limits[0]+limits[1])/2, limits[1]-limits[0]
    expected_footprint = np.asarray([(center[0]+co*sx*size[0]/2-si*sy*size[1]/2,
                                     center[1]+si*sx*size[0]/2+co*sy*size[1]/2, 0.15)
                                    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
    fixtures = {}
    for name, location, dimensions, angle in (("car", center, size, yaw),
                                             ("person", np.asarray((8.0, 0.0, -0.9)), np.asarray((0.6, 0.6, 1.7)), 0.1)):
        cosine, sine = math.cos(angle), math.sin(angle)
        vertices = np.asarray([(location[0]+cosine*sx*dimensions[0]/2-sine*sy*dimensions[1]/2,
                                location[1]+sine*sx*dimensions[0]/2+cosine*sy*dimensions[1]/2,
                                location[2]+sz*dimensions[2]/2)
                               for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        optical = vertices@transform[:3, :3].T+transform[:3, 3]
        assert np.all(optical[:, 2] > 0)
        projected = cv2.projectPoints(optical, np.zeros(3), np.zeros(3), k, distortion)[0].reshape(-1, 2)
        lo, hi = np.maximum(projected.min(axis=0), (0, 0)), np.minimum(projected.max(axis=0), (width, height))
        assert np.all(hi > lo)
        footprint = np.asarray([(location[0]+cosine*sx*dimensions[0]/2-sine*sy*dimensions[1]/2,
                                 location[1]+sine*sx*dimensions[0]/2+cosine*sy*dimensions[1]/2, 0.15)
                                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))])
        fixtures[name] = dict(center=location, size=dimensions, yaw=angle, co=cosine, si=sine,
                              roi_center=(lo+hi)/2, roi_size=hi-lo, footprint=footprint,
                              rgb=0xff00ff if name == "person" else 0xffa500)
    point_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                            ("rgb", "<u4"), ("source_index", "<u4")])
    point_sources = {}
    prefix = "/camera_lidar_box_software_check"
    topics = {name: prefix+"/"+name for name in ("lidar", "camera", "info", "markers", "lidar_points", "fused_points", "fused_detections", "diagnostics")}
    command = [str(args.executable), "--ros-args",
               "-p", f"intrinsics_file:={args.intrinsics}", "-p", f"extrinsics_file:={args.extrinsics}",
               "-p", "allow_provisional_calibration:=true", "-p", f"camera_frame:={camera_frame}",
               "-p", f"lidar_detections_topic:={topics['lidar']}",
               "-p", f"camera_detections_topic:={topics['camera']}", "-p", f"camera_info_topic:={topics['info']}",
               "-p", f"markers_topic:={topics['markers']}",
               "-p", f"lidar_points_topic:={topics['lidar_points']}", "-p", f"points_topic:={topics['fused_points']}",
               "-p", f"output_detections_topic:={topics['fused_detections']}",
               "-p", f"require_camera_match:={'true' if args.require_camera_match else 'false'}",
               "-r", f"/camera_lidar_bev/diagnostics:={topics['diagnostics']}"]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "domain_id": args.domain_id, "software_only": True, "physical_accuracy_validated": False,
              "require_camera_match": args.require_camera_match,
              "real_time_speed_validated": False, "synthetic_messages": True,
              "independent_projection": "OpenCV projectPoints of eight oriented cuboid corners using native K/D and candidate B",
              "native_image_size": [width, height], "synthetic_projected_roi_xyxy": np.concatenate(limits).tolist(),
              "synthetic_lidar_center_m": center.tolist(), "synthetic_lidar_size_m": size.tolist(),
              "synthetic_lidar_yaw_rad": yaw, "executable": str(args.executable),
              "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
              "intrinsics_sha256": hashlib.sha256(args.intrinsics.read_bytes()).hexdigest(),
              "extrinsics_sha256": hashlib.sha256(args.extrinsics.read_bytes()).hexdigest(), "stages": []}
    rclpy.init()
    node = rclpy.create_node("camera_lidar_box_software_verifier")
    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
    publishers = {key: node.create_publisher(kind, topics[key], qos) for key, kind in
                  (("lidar", Detection3DArray), ("camera", Detection2DArray), ("info", CameraInfo),
                   ("lidar_points", PointCloud2))}
    received = {"markers": [], "fused_points": [], "fused_detections": [], "diagnostics": []}

    def on_message(key, message):
        received[key].append((time.monotonic(), message))
        del received[key][:-250]

    subscriptions = [node.create_subscription(kind, topics[key],
                     lambda message, key=key: on_message(key, message), qos) for key, kind in
                     (("markers", MarkerArray), ("fused_points", PointCloud2),
                      ("fused_detections", Detection3DArray), ("diagnostics", DiagnosticArray))]
    deadline = time.monotonic()+args.timeout
    last_positive_stamp = 0
    active_phase = "discovery"

    def header(stamp, frame):
        item = Header(frame_id=frame)
        item.stamp.sec, item.stamp.nanosec = divmod(int(stamp), 1_000_000_000)
        return item

    def send(kind, send_lidar=True):
        class_name = "person" if kind.startswith("person_") else "car"
        fixture = fixtures[class_name]
        location, dimensions, angle = fixture["center"], fixture["size"], fixture["yaw"]
        lidar_stamp = node.get_clock().now().nanoseconds-20_000_000
        camera_stamp = lidar_stamp+5_000_000
        if kind == "pair_delta":
            camera_stamp = lidar_stamp-300_000_000
        if kind == "camera_future":
            camera_stamp += 2_000_000_000
        lidar_frame = "incorrect_lidar_frame" if kind == "wrong_lidar_frame" else "os_sensor"
        lidar_header = header(lidar_stamp, lidar_frame)
        detections = Detection3DArray(header=lidar_header)
        item = Detection3D(header=lidar_header)
        item.bbox.center.position.x, item.bbox.center.position.y, item.bbox.center.position.z = map(float, location)
        item.bbox.center.orientation.z, item.bbox.center.orientation.w = math.sin(angle/2), math.cos(angle/2)
        item.bbox.size.x, item.bbox.size.y, item.bbox.size.z = map(float, dimensions)
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id, hypothesis.hypothesis.score = class_name, 0.77
        hypothesis.pose.pose = item.bbox.center
        item.results.append(hypothesis)
        detections.detections.append(item)
        camera_header = header(camera_stamp, "incorrect_camera_frame" if kind == "wrong_camera_frame" else camera_frame)
        camera = Detection2DArray(header=camera_header)
        if kind not in ("fresh_empty_camera", "person_unmatched"):
            for _ in range(2 if kind == "ambiguous_camera" else 1):
                item2d = Detection2D(header=camera_header)
                item2d.bbox.center.position.x, item2d.bbox.center.position.y = map(float, fixture["roi_center"])
                item2d.bbox.size_x, item2d.bbox.size_y = map(float, fixture["roi_size"])
                hypothesis2d = ObjectHypothesisWithPose()
                camera_class = ("car" if class_name == "person" else "person") if kind.endswith("wrong_camera_class") else class_name
                hypothesis2d.hypothesis.class_id, hypothesis2d.hypothesis.score = camera_class, 0.87
                item2d.results.append(hypothesis2d)
                camera.detections.append(item2d)
        info = CameraInfo(header=camera_header, width=width, height=height, distortion_model="plumb_bob")
        info.k, info.d = native["camera_matrix"]["data"], native["distortion_coefficients"]["data"]
        info.r, info.p = native["rectification_matrix"]["data"], native["projection_matrix"]["data"]
        if kind == "bad_camera_info":
            info.k[0] += 1.0
        if kind == "camera_info_stamp_mismatch":
            info.header = header(camera_stamp-7_000_000, camera_frame)
        if kind not in ("no_camera", "stale_camera"):
            publishers["info"].publish(info)
            publishers["camera"].publish(camera)
        if send_lidar:
            rows = np.zeros(3, dtype=point_dtype)
            inside_offset = np.asarray((fixture["co"]*0.1-fixture["si"]*0.1,
                                        fixture["si"]*0.1+fixture["co"]*0.1, 0.1))
            xyz = np.vstack((location, location+inside_offset, location+(10.0, 0.0, 0.0)))
            for axis_index, axis in enumerate(("x", "y", "z")):
                rows[axis] = xyz[:, axis_index]
            rows["rgb"] = fixture["rgb"]
            rows["source_index"] = (70_001, 900_001, 1_234_567)
            points_header = lidar_header
            if kind == "points_stamp_mismatch":
                points_header = header(lidar_stamp-7_000_000, "os_sensor")
            if kind == "points_wrong_frame":
                points_header = header(lidar_stamp, "os_lidar")
            points = PointCloud2(header=points_header, height=1, width=len(rows), point_step=20, row_step=60,
                                 is_bigendian=False, is_dense=True, data=rows.tobytes())
            points.fields = [PointField(name=name, offset=index*4,
                                       datatype=PointField.UINT32 if name == "source_index" else PointField.FLOAT32, count=1)
                             for index, name in enumerate(("x", "y", "z", "rgb", "source_index"))]
            point_sources[lidar_stamp] = rows
            publishers["lidar_points"].publish(points)
            publishers["lidar"].publish(detections)
        return lidar_stamp

    def cloud_for_stamp(start, stamp):
        for receipt, cloud in reversed(received["fused_points"]):
            if receipt >= start and stamp_ns(cloud.header) == stamp:
                return cloud
        return None

    def detections_for_stamp(start, stamp):
        for receipt, message in reversed(received["fused_detections"]):
            if receipt >= start and stamp_ns(message.header) == stamp:
                return message
        return None

    def check_points(cloud, stamp, expected_count, rgb):
        assert cloud.header.frame_id == "os_sensor" and stamp_ns(cloud.header) == stamp
        assert cloud.height == 1 and cloud.point_step == 20 and cloud.width == expected_count
        assert cloud.row_step == expected_count*20 and len(cloud.data) == cloud.row_step
        if not expected_count:
            return
        fields = {field.name: field for field in cloud.fields}
        assert set(fields) == {"x", "y", "z", "rgb", "source_index"}
        for index, name in enumerate(("x", "y", "z", "rgb", "source_index")):
            assert fields[name].offset == index*4 and fields[name].count == 1
            assert fields[name].datatype == (PointField.UINT32 if name == "source_index" else PointField.FLOAT32)
        assert not cloud.is_bigendian
        rows = np.frombuffer(bytes(cloud.data), dtype=point_dtype)
        expected = point_sources[stamp][:2]
        assert np.array_equal(rows["source_index"], expected["source_index"])
        for axis in ("x", "y", "z"):
            assert np.array_equal(rows[axis], expected[axis])
        assert np.all(rows["rgb"] == rgb)

    def check_geometry(start, source_stamps, matched, kind):
        nonlocal last_positive_stamp
        class_name = "person" if kind.startswith("person_") else "car"
        fixture = fixtures[class_name]
        location, dimensions = fixture["center"], fixture["size"]
        expected_label = (f"{class_name} L:0.77 C:{class_name+' 0.87' if matched else '--'}\n"
                          f"L {dimensions[0]:.2f} W {dimensions[1]:.2f} H {dimensions[2]:.2f} m")
        for receipt, array in reversed(received["markers"]):
            if receipt < start or len(array.markers) != 3:
                continue
            first, rectangle, label = array.markers
            stamp = stamp_ns(first.header)
            if stamp not in source_stamps or label.text != expected_label:
                continue
            expected_points = 2 if kind not in ("points_stamp_mismatch", "points_wrong_frame") else 0
            cloud = cloud_for_stamp(start, stamp)
            detections = detections_for_stamp(start, stamp)
            if cloud is None or cloud.width != expected_points or detections is None:
                continue
            check_points(cloud, stamp, expected_points, fixture["rgb"])
            assert detections.header.frame_id == "os_sensor" and len(detections.detections) == 1
            detection = detections.detections[0]
            assert detection.header.frame_id == "os_sensor" and stamp_ns(detection.header) == stamp
            assert len(detection.results) == 1 and detection.results[0].hypothesis.class_id == class_name
            assert detection.results[0].hypothesis.score == 0.77 and detection.id == ""
            assert np.allclose((detection.bbox.center.position.x, detection.bbox.center.position.y,
                                detection.bbox.center.position.z), location, atol=1e-9, rtol=0)
            assert np.allclose((detection.bbox.size.x, detection.bbox.size.y, detection.bbox.size.z),
                               dimensions, atol=1e-9, rtol=0)
            q_box = detection.bbox.center.orientation
            assert (q_box.x, q_box.y) == (0.0, 0.0)
            assert np.allclose((q_box.z, q_box.w), (math.sin(fixture["yaw"]/2), math.cos(fixture["yaw"]/2)),
                               atol=1e-9, rtol=0)
            assert first.action == Marker.DELETEALL
            for marker in array.markers:
                assert marker.header.frame_id == "os_sensor" and stamp_ns(marker.header) == stamp
            assert rectangle.action == label.action == Marker.ADD
            assert rectangle.type == Marker.LINE_STRIP and rectangle.ns == "camera_lidar_bev_extent"
            assert label.type == Marker.TEXT_VIEW_FACING and label.ns == "camera_lidar_labels"
            q = rectangle.pose.orientation
            assert (q.x, q.y, q.z, q.w) == (0.0, 0.0, 0.0, 1.0)
            assert (rectangle.pose.position.x, rectangle.pose.position.y, rectangle.pose.position.z) == (0.0, 0.0, 0.0)
            actual = np.asarray([(point.x, point.y, point.z) for point in rectangle.points])
            assert actual.shape == (5, 3) and np.allclose(actual[0], actual[-1], atol=1e-9)
            assert np.allclose(actual, fixture["footprint"], atol=1e-8, rtol=0)
            # Recover dimensions in the input yaw frame: camera metadata must
            # never resize, recenter or rotate complete trained LiDAR geometry.
            delta = actual[:4, :2]-location[:2]
            local = np.column_stack((fixture["co"]*delta[:, 0]+fixture["si"]*delta[:, 1],
                                     -fixture["si"]*delta[:, 0]+fixture["co"]*delta[:, 1]))
            assert np.allclose(local.max(axis=0)-local.min(axis=0), dimensions[:2], atol=1e-8, rtol=0)
            assert np.allclose((label.pose.position.x, label.pose.position.y), location[:2], atol=1e-8, rtol=0)
            for marker in (rectangle, label):
                expected_rgb = fixture["rgb"]
                assert np.allclose((marker.color.r, marker.color.g, marker.color.b, marker.color.a),
                                   ((expected_rgb >> 16 & 255)/255, (expected_rgb >> 8 & 255)/255,
                                    (expected_rgb & 255)/255, 1.0), atol=1e-7, rtol=0)
            last_positive_stamp = max(last_positive_stamp, stamp)
            return {"lidar_stamp_ns": stamp, "camera_associated": matched, "label": label.text,
                    "full_lidar_footprint_unchanged": True, "flat_closed_oriented_footprint": True,
                    "all_marker_headers_match_lidar": True, "fused_points": expected_points,
                    "native_point_provenance_verified": True, "lidar_class": class_name,
                    "full_3d_detection_geometry_and_score_unchanged": True}
        return None

    def check_clear(start, minimum_stamp):
        for receipt, array in reversed(received["markers"]):
            if receipt < start or len(array.markers) != 1:
                continue
            first = array.markers[0]
            stamp = stamp_ns(first.header)
            if stamp < minimum_stamp or stamp <= last_positive_stamp:
                continue
            cloud = cloud_for_stamp(start, stamp)
            detections = detections_for_stamp(start, stamp)
            if cloud is None or cloud.width or detections is None:
                continue
            check_points(cloud, stamp, 0, 0)
            assert detections.header.frame_id == "os_sensor" and not detections.detections
            assert first.header.frame_id == "os_sensor" and first.action == Marker.DELETEALL
            return {"clear_stamp_ns": stamp, "explicit_delete_all": True, "empty_fused_points": True,
                    "empty_fused_detections": True, "output_headers_match": True}
        return None

    def record(name, evidence):
        report["stages"].append(dict(name=name, passed=True, **evidence))
        print(name+": passed", flush=True)

    def phase(kind, matched):
        nonlocal active_phase
        active_phase = kind
        start = time.monotonic()
        start_ns = node.get_clock().now().nanoseconds
        stamps = set()
        next_send = start
        minimum_wait = 0.65 if kind == "stale_camera" else 0.22
        while time.monotonic() < min(deadline, start+1.65):
            now = time.monotonic()
            if now >= next_send:
                stamps.add(send(kind))
                next_send = now+0.075
            rclpy.spin_once(node, timeout_sec=0.01)
            if time.monotonic()-start < minimum_wait:
                continue
            observation_start = start+minimum_wait-0.05
            evidence = (check_clear(observation_start, start_ns-30_000_000) if args.require_camera_match and not matched
                        else check_geometry(observation_start, stamps, matched, kind))
            if evidence:
                record(kind, evidence)
                return
            if process.poll() is not None:
                raise RuntimeError("Fusion child exited during "+kind)
        raise AssertionError("Missing expected preserved box/camera metadata during "+kind)

    def lidar_failure(kind):
        nonlocal active_phase
        active_phase = kind
        start = time.monotonic()
        start_ns = node.get_clock().now().nanoseconds
        next_send = start
        minimum_wait = 0.65 if kind == "stale_lidar" else 0.20
        while time.monotonic() < min(deadline, start+1.6):
            now = time.monotonic()
            if now >= next_send:
                # Fresh camera traffic alone cannot keep expired LiDAR geometry.
                send("matched" if kind == "stale_lidar" else kind, send_lidar=kind != "stale_lidar")
                next_send = now+0.075
            rclpy.spin_once(node, timeout_sec=0.02)
            if time.monotonic()-start >= minimum_wait:
                evidence = check_clear(start, start_ns-30_000_000)
                if evidence:
                    record(kind, evidence)
                    return
        raise AssertionError("Missing explicit geometry clear during "+kind)

    process = None
    with tempfile.TemporaryFile() as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            discover_deadline = min(deadline, time.monotonic()+5)
            while time.monotonic() < discover_deadline:
                rclpy.spin_once(node, timeout_sec=0.03)
                if all(pub.get_subscription_count() for pub in publishers.values()):
                    break
                if process.poll() is not None:
                    raise RuntimeError("Fusion child exited during discovery")
            else:
                raise TimeoutError("Isolated ROS discovery did not complete")
            for kind, matched in (("no_camera", False), ("matched", True), ("stale_camera", False),
                                  ("camera_recovery", True), ("ambiguous_camera", False),
                                  ("fresh_empty_camera", False), ("bad_camera_info", False),
                                  ("camera_info_stamp_mismatch", False), ("wrong_camera_frame", False),
                                  ("camera_future", False), ("pair_delta", False),
                                  ("matched_before_wrong_lidar", True)):
                phase(kind, matched)
            lidar_failure("wrong_lidar_frame")
            phase("matched_before_stale_lidar", True)
            lidar_failure("stale_lidar")
            phase("recovery", True)
            for kind, matched in (("person_matched", True), ("person_unmatched", False),
                                  ("person_wrong_camera_class", False), ("car_wrong_camera_class", False),
                                  ("points_stamp_mismatch", True), ("points_wrong_frame", True),
                                  ("points_recovery", True)):
                phase(kind, matched)
            report["passed"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            report["failed_stage"] = active_phase
            report["received_output_messages"] = {key: len(items) for key, items in received.items()}
            if received["markers"]:
                report["last_marker_texts"] = [marker.text for marker in received["markers"][-1][1].markers]
            print(report["error"], flush=True)
        finally:
            if received["diagnostics"]:
                report["last_diagnostics"] = {status.name: {item.key: item.value for item in status.values}
                                              for status in received["diagnostics"][-1][1].status}
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
            report["node_log_tail"] = log.read().decode("utf-8", errors="replace")[-6000:]
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
