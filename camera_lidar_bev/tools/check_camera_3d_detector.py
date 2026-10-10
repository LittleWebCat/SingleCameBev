#!/usr/bin/env python3
"""Bounded isolated ROS replay through the real camera-only SMOKE detector.

Uses a previously captured outdoor image and its native candidate CameraInfo.
No LiDAR, camera driver or synthetic model outputs are used. Valid output may
contain zero learned predictions; this photograph is not a ground-truth dataset.
The replay verifies source/coordinate contracts and invalid/stale clearing.
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
    return int(header.stamp.sec)*1_000_000_000+int(header.stamp.nanosec)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path,
                        default=root/"install/bev_visualization/lib/bev_visualization/camera_3d_detector")
    parser.add_argument("--engine", type=Path, default=root/"models/camera_3d_001/smoke_fp16_raw.engine")
    parser.add_argument("--image", type=Path,
                        default=root/"data/camera_3d_demo_001/kitti_000008.png")
    parser.add_argument("--intrinsics", type=Path,
                        help="native CameraInfo YAML; otherwise uses --calibration and the image dimensions")
    parser.add_argument("--calibration", type=Path,
                        default=root/"data/camera_3d_demo_001/kitti_000008.calib.txt",
                        help="rectified KITTI projection (12 or 16 numbers), using its 3x3 intrinsics and zero distortion")
    parser.add_argument("--require-predictions", action="store_true",
                        help="require at least one real learned prediction in every valid/recovery phase")
    parser.add_argument("--output", type=Path,
                        default=root/"evaluation/camera_3d_detector_replay_001.json")
    parser.add_argument("--domain-id", type=int, default=88)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a fresh evidence path")
    if not 2 <= args.domain_id <= 231:
        parser.error("domain-id must be between 2 and 231, separate from live domain 1")
    if not math.isfinite(args.timeout) or not 10 <= args.timeout <= 20:
        parser.error("timeout must be between 10 and 20 seconds")
    calibration_path = args.intrinsics if args.intrinsics is not None else args.calibration
    for path in (args.executable, args.engine, args.image, calibration_path):
        if not path.is_file():
            parser.error(f"Required executable, real engine, image or calibration is missing: {path}")
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ.pop("ROS_LOCALHOST_ONLY", None)
    os.environ.pop("ROS_STATIC_PEERS", None)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    os.environ.setdefault("CYCLONEDDS_URI", "file://"+str(root/"config/transport/cyclonedds_large_clouds.xml"))
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/camera-lidar-bev-camera-3d-check-logs")

    import cv2
    import numpy as np
    import rclpy
    import yaml
    from diagnostic_msgs.msg import DiagnosticArray
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import Header
    from tf2_msgs.msg import TFMessage
    from vision_msgs.msg import Detection3DArray
    from visualization_msgs.msg import Marker, MarkerArray

    source = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if source is None:
        parser.error("Recorded image cannot be read")
    if args.intrinsics is not None:
        native = yaml.safe_load(args.intrinsics.read_text())
    else:
        projection = np.loadtxt(args.calibration, dtype=np.float64).reshape(-1)
        if projection.size not in (12, 16) or not np.isfinite(projection).all():
            parser.error("KITTI calibration must be a finite 3x4 or 4x4 projection")
        intrinsic = projection.reshape(-1, 4)[:3, :3]
        output_projection = np.column_stack((intrinsic, np.zeros(3)))
        native = {"image_width": source.shape[1], "image_height": source.shape[0],
                  "camera_matrix": {"data": intrinsic.reshape(-1).tolist()},
                  "distortion_coefficients": {"data": [0.0]*5},
                  "rectification_matrix": {"data": np.eye(3).reshape(-1).tolist()},
                  "projection_matrix": {"data": output_projection.reshape(-1).tolist()}}
    width, height = native["image_width"], native["image_height"]
    if source is None or source.shape != (height, width, 3):
        parser.error("Recorded image must match native calibrated dimensions")
    K = np.asarray(native["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(native["distortion_coefficients"]["data"], dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(K, distortion, None, K, (width, height), cv2.CV_32FC1)
    expected_rectified = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR)
    camera_frame, bev_frame = "camera_front_optical_frame", "camera_front_bev"
    prefix = "/camera_3d_software_check"
    topics = {name: prefix+"/"+name for name in ("image", "info", "detections", "markers", "labeled", "diagnostics", "tf_static")}
    child_name = "software_camera_3d_child"
    command = [str(args.executable), "--ros-args", "-r", "__node:="+child_name,
               "-p", f"engine_file:={args.engine}", "-p", f"image_topic:={topics['image']}",
               "-p", f"camera_info_topic:={topics['info']}", "-p", f"detections_topic:={topics['detections']}",
               "-p", f"markers_topic:={topics['markers']}", "-p", f"labeled_image_topic:={topics['labeled']}",
               "-p", "classes:=[car,person]", "-p", "score_threshold:=0.25",
               "-p", f"width:={width}", "-p", f"height:={height}",
               "-p", f"camera_frame:={camera_frame}", "-p", f"bev_frame:={bev_frame}",
               "-r", f"/camera_lidar_bev/diagnostics:={topics['diagnostics']}",
               "-r", f"/tf_static:={topics['tf_static']}"]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "domain_id": args.domain_id, "software_only": True, "actual_trained_engine": True,
              "physical_accuracy_validated": False, "learned_detection_accuracy_validated": False,
              "real_time_speed_validated": False, "lidar_inputs_used": False,
              "source_image": str(args.image), "source_image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
              "engine": str(args.engine), "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
              "executable": str(args.executable), "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
              "calibration": str(calibration_path), "native_size": [width, height],
              "require_predictions": args.require_predictions,
              "intrinsics_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(), "stages": [],
              "limitations": ["Recorded photograph is not annotated 3D ground truth; emitted boxes are model predictions.",
                              "Default fixture is an official rectified KITTI outdoor image; explicit native image/YAML overrides are supported.",
                              "Zero predictions are valid observations and do not establish detection accuracy.",
                              "Candidate camera calibration and KITTI-to-this-camera domain accuracy remain unvalidated."]}
    rclpy.init()
    node = rclpy.create_node("camera_3d_software_verifier")
    reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
    sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    tf_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL)
    image_pub = node.create_publisher(Image, topics["image"], sensor)
    info_pub = node.create_publisher(CameraInfo, topics["info"], sensor)
    received = {name: [] for name in ("detections", "markers", "labeled", "diagnostics", "tf_static")}

    def on_message(name, message):
        received[name].append((time.monotonic(), message))
        del received[name][:-100]

    subscriptions = [node.create_subscription(kind, topics[name],
                     lambda message, name=name: on_message(name, message), qos) for name, kind, qos in
                     (("detections", Detection3DArray, reliable), ("markers", MarkerArray, reliable),
                      ("labeled", Image, sensor), ("diagnostics", DiagnosticArray, reliable),
                      ("tf_static", TFMessage, tf_qos))]
    deadline = time.monotonic()+args.timeout
    active_phase = "discovery"
    last_valid_stamp = 0
    last_valid_image = None
    source_headers = set()

    def header(stamp, frame=camera_frame):
        message = Header(frame_id=frame)
        message.stamp.sec, message.stamp.nanosec = divmod(int(stamp), 1_000_000_000)
        return message

    def inputs(kind="valid"):
        stamp = node.get_clock().now().nanoseconds-20_000_000
        if kind == "future":
            stamp += 2_000_000_000
        frame = "incorrect_optical_frame" if kind == "image_frame" else camera_frame
        h = header(stamp, frame)
        rgb = kind == "rgb_valid"
        pixels = cv2.cvtColor(source, cv2.COLOR_BGR2RGB) if rgb else source
        image = Image(header=h, width=width, height=height, encoding="rgb8" if rgb else "bgr8",
                      is_bigendian=False, step=width*3, data=pixels.tobytes())
        if kind == "payload":
            image.data = image.data[:-1]
        info = CameraInfo(header=h, width=width, height=height, distortion_model="plumb_bob")
        info.k, info.d = native["camera_matrix"]["data"], native["distortion_coefficients"]["data"]
        info.r, info.p = native["rectification_matrix"]["data"], native["projection_matrix"]["data"]
        if kind == "info_stamp":
            info.header = header(stamp-7_000_000)
        if kind == "info_K":
            info.k[0] = 0.0
        if kind == "info_size":
            info.width -= 1
        if kind == "info_roi":
            info.roi.x_offset = 1
            info.roi.width = width-1
        if kind == "info_nonfinite_R":
            info.r[0] = float("nan")
        return image, info

    def matches(start):
        grouped = {}
        for name in ("detections", "markers", "labeled"):
            grouped[name] = {}
            for receipt, message in received[name]:
                if receipt < start:
                    continue
                if name == "markers":
                    assert message.markers, "Every marker update needs explicit DELETEALL"
                    h = message.markers[0].header
                else:
                    h = message.header
                grouped[name][stamp_ns(h)] = message
        common = grouped["detections"].keys() & grouped["markers"].keys() & grouped["labeled"].keys()
        for stamp in sorted(common, reverse=True):
            yield stamp, grouped["detections"][stamp], grouped["markers"][stamp], grouped["labeled"][stamp]

    def image_array(message):
        assert message.encoding == "bgr8" and message.width == width and message.height == height
        assert message.step == width*3 and len(message.data) == width*height*3
        return np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(height, width, 3)

    def valid_output(start, sent):
        nonlocal last_valid_stamp, last_valid_image
        for stamp, detections, markers, labeled in matches(start):
            if stamp not in sent:
                continue
            if args.require_predictions and not detections.detections:
                continue
            assert detections.header.frame_id == labeled.header.frame_id == camera_frame
            assert markers.markers[0].action == Marker.DELETEALL
            assert len(markers.markers) == 1+2*len(detections.detections)
            classes_seen = []
            for index, item in enumerate(detections.detections):
                assert item.header.frame_id == camera_frame and stamp_ns(item.header) == stamp
                assert len(item.results) == 1
                hypothesis = item.results[0]
                name, score = hypothesis.hypothesis.class_id, hypothesis.hypothesis.score
                assert name in ("car", "person") and math.isfinite(score) and .25 < score <= 1
                location = np.asarray([getattr(item.bbox.center.position, axis) for axis in ("x", "y", "z")])
                size = np.asarray([getattr(item.bbox.size, axis) for axis in ("x", "y", "z")])
                assert np.isfinite(location).all() and .5 <= location[2] <= 60 and abs(location[0]) <= 40
                assert np.isfinite(size).all() and np.all(size > 0) and np.all(size < 20)
                q = item.bbox.center.orientation
                assert q.x == q.z == 0 and math.isclose(q.y*q.y+q.w*q.w, 1, abs_tol=1e-7)
                pose = hypothesis.pose.pose
                assert np.allclose((pose.position.x, pose.position.y, pose.position.z), location, atol=1e-9, rtol=0)
                assert np.allclose((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w),
                                   (q.x, q.y, q.z, q.w), atol=1e-9, rtol=0)
                yaw = 2*math.atan2(q.y, q.w)
                co, si = math.cos(yaw), math.sin(yaw)
                expected_corners = []
                for sx in (-1, 1):
                    for sz in (-1, 1):
                        x, z = sx*size[0]/2, sz*size[2]/2
                        optical_x = location[0]+co*x+si*z
                        optical_z = location[2]-si*x+co*z
                        assert optical_z > .3, "Displayed box must have every corner in front of the near plane"
                        expected_corners.append((optical_z, -optical_x))
                rectangle, label = markers.markers[1+2*index:3+2*index]
                assert rectangle.ns == "camera_3d_footprints" and rectangle.type == Marker.LINE_STRIP
                assert label.ns == "camera_3d_labels" and label.type == Marker.TEXT_VIEW_FACING
                actual = np.asarray([(point.x, point.y, point.z) for point in rectangle.points])
                assert actual.shape == (5, 3) and np.allclose(actual[0], actual[-1], atol=1e-8)
                assert np.allclose(actual[:, 2], .15, atol=1e-9)
                assert all(any(np.allclose(corner[:2], expected, atol=1e-4, rtol=0)
                               for expected in expected_corners) for corner in actual[:4])
                assert len(set(map(tuple, actual[:4, :2]))) == 4
                assert np.allclose((label.pose.position.x, label.pose.position.y, label.pose.position.z),
                                   (location[2], -location[0], .3), atol=1e-9, rtol=0)
                expected_text = f"{name} {score:.2f}  L {size[0]:.1f} W {size[2]:.1f} H {size[1]:.1f} m"
                assert label.text == expected_text
                rgb = (1, 0, 1, 1) if name == "person" else (1, 165/255, 0, 1)
                for marker in (rectangle, label):
                    assert marker.action == Marker.ADD
                    assert np.allclose((marker.color.r, marker.color.g, marker.color.b, marker.color.a), rgb,
                                       atol=1e-7, rtol=0)
                classes_seen.append(name)
            for marker in markers.markers:
                assert marker.header.frame_id == bev_frame and stamp_ns(marker.header) == stamp
            canvas = image_array(labeled)
            identical = np.mean(np.max(np.abs(canvas.astype(np.int16)-expected_rectified.astype(np.int16)), axis=2) <= 1)
            assert identical > .70, "Labeled native view must preserve the independently rectified recorded image"
            assert stamp > last_valid_stamp
            last_valid_stamp, last_valid_image = stamp, sent[stamp]
            source_headers.add(stamp)
            return {"source_stamp_ns": stamp, "learned_boxes_observed": len(classes_seen),
                    "classes_observed": classes_seen, "all_geometry_validated_if_emitted": True,
                    "native_rectified_image_preserved_fraction": float(identical),
                    "optical_3d_and_camera_bev_headers_verified": True}
        return None

    def cleared_output(start, minimum_stamp):
        for stamp, detections, markers, labeled in matches(start):
            if stamp <= last_valid_stamp or stamp < minimum_stamp or stamp in source_headers:
                continue
            assert detections.header.frame_id == labeled.header.frame_id == camera_frame
            assert not detections.detections
            assert len(markers.markers) == 1 and markers.markers[0].action == Marker.DELETEALL
            assert markers.markers[0].header.frame_id == bev_frame
            canvas = image_array(labeled)
            assert np.mean(np.all(canvas == 0, axis=2)) > .90, "Explicit invalid/stale image must be blank except status text"
            return {"clear_stamp_ns": stamp, "empty_3d": True, "explicit_markers_delete_all": True,
                    "blank_status_image": True, "three_source_stamps_matched": True}
        return None

    def record(name, evidence):
        report["stages"].append(dict(name=name, passed=True, **evidence))
        print(name+": passed", flush=True)

    def valid(name="valid", rgb=False):
        nonlocal active_phase
        active_phase = name
        start = time.monotonic()
        sent = {}
        next_send = start
        while time.monotonic() < min(deadline, start+1.3):
            now = time.monotonic()
            if now >= next_send:
                image, info = inputs("rgb_valid" if rgb else "valid")
                sent[stamp_ns(image.header)] = image
                info_pub.publish(info)
                image_pub.publish(image)
                next_send = now+.10
            rclpy.spin_once(node, timeout_sec=.01)
            evidence = valid_output(start, sent)
            if evidence:
                record(name, evidence)
                return
            if process.poll() is not None:
                raise RuntimeError("Real camera-only node exited during "+name)
        raise AssertionError("Missing source-preserving actual 3D inference output during "+name)

    def failure(kind):
        nonlocal active_phase
        active_phase = kind
        start = time.monotonic()
        start_ns = node.get_clock().now().nanoseconds
        if kind == "duplicate":
            assert last_valid_image is not None
            image_pub.publish(copy.deepcopy(last_valid_image))
        elif kind != "stale":
            image, info = inputs(kind)
            info_pub.publish(info)
            image_pub.publish(image)
        while time.monotonic() < min(deadline, start+1.15):
            rclpy.spin_once(node, timeout_sec=.02)
            if kind == "stale" and time.monotonic()-start < .60:
                continue
            evidence = cleared_output(start, start_ns-20_000_000)
            if evidence:
                record(kind, evidence)
                return
        raise AssertionError("Invalid/stale source did not explicitly clear all outputs during "+kind)

    process = None
    with tempfile.TemporaryFile() as log:
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            discovery_deadline = min(deadline, time.monotonic()+4)
            while time.monotonic() < discovery_deadline:
                rclpy.spin_once(node, timeout_sec=.03)
                if image_pub.get_subscription_count() and info_pub.get_subscription_count():
                    break
                if process.poll() is not None:
                    raise RuntimeError("Camera-only node exited during discovery")
            else:
                raise TimeoutError("Private ROS discovery did not complete")
            subscriptions_seen = node.get_subscriber_names_and_types_by_node(child_name, "/")
            report["actual_child_subscriptions"] = subscriptions_seen
            data_subscriptions = {name for name, _ in subscriptions_seen if name != "/parameter_events"}
            assert data_subscriptions == {topics["image"], topics["info"]}, "Camera-only child must have only image/CameraInfo data inputs"
            valid()
            for kind in ("payload", "image_frame", "info_stamp", "info_K", "info_size", "info_roi", "info_nonfinite_R", "future", "duplicate"):
                failure(kind)
                valid("recovery_after_"+kind)
            failure("stale")
            valid("rgb_valid_recovery", rgb=True)
            active_phase = "static_axes_and_diagnostics"
            diagnostic_deadline = min(deadline, time.monotonic()+1.2)
            values = {}
            while time.monotonic() < diagnostic_deadline:
                rclpy.spin_once(node, timeout_sec=.03)
                statuses = [s for _, message in received["diagnostics"] for s in message.status if s.name == "camera_3d_detector"]
                if statuses:
                    values = {item.key: item.value for item in statuses[-1].values}
                    if int(values.get("processed_frames", "0")) >= 11:
                        break
            for key, expected in (("camera_only", "true"), ("trained_3d_detector", "true"), ("lidar_used", "false"),
                                  ("physical_accuracy_validated", "false"), ("tracking_enabled", "false"),
                                  ("camera_frame", camera_frame), ("bev_frame", bev_frame)):
                assert values.get(key) == expected, (key, values.get(key))
            transforms = [t for _, message in received["tf_static"] for t in message.transforms]
            assert len(transforms) == 1 and transforms[0].header.frame_id == camera_frame and transforms[0].child_frame_id == bev_frame
            t = transforms[0].transform
            assert (t.translation.x, t.translation.y, t.translation.z) == (0.0, 0.0, 0.0)
            assert np.allclose((t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w), (.5, -.5, .5, .5), atol=1e-9, rtol=0)
            report["last_diagnostics"] = values
            report["coordinate_only_tf_verified"] = True
            report["passed"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            report["failed_stage"] = active_phase
            report["received_output_messages"] = {name: len(messages) for name, messages in received.items()}
            print(report["error"], flush=True)
        finally:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
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
