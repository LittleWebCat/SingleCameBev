#!/usr/bin/env python3
"""Bounded ROS replay of real captured LiDAR through the trained PointPillars node.

Publishes a recorded Ouster scan only in isolated ROS domain 76. Uses the actual
TensorRT engine, never a model mock, and verifies oriented predictions, flat BEV
markers, source-point provenance, and explicit invalid/stale clearing. This is a
software integration check; it does not validate Ouster detection accuracy.
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
                        default=root/"install/bev_lidar_detection/lib/bev_lidar_detection/lidar_vehicle_detector")
    parser.add_argument("--engine", type=Path, default=root/"models/lidar_objects_001/pointpillar_fp16.plan")
    parser.add_argument("--capture", type=Path,
                        default=root/"recordings/lidar_detector_input_001/raw_points.npz")
    parser.add_argument("--kitti-reference", type=Path,
                        default=root/"models/lidar_objects_001/kitti_reference_000000.bin")
    parser.add_argument("--output", type=Path,
                        default=root/"evaluation/lidar_vehicle_detector_replay_001.json")
    parser.add_argument("--domain-id", type=int, default=76)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a fresh evidence path")
    if not 2 <= args.domain_id <= 231:
        parser.error("domain-id must be between 2 and 231, separate from live domain 1")
    if not math.isfinite(args.timeout) or not 20 <= args.timeout <= 30:
        parser.error("timeout must be between 20 and 30 seconds")
    for path in (args.executable, args.engine, args.capture, args.kitti_reference):
        if not path.is_file():
            parser.error(f"Required executable, real engine or capture is missing: {path}")

    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ.pop("ROS_LOCALHOST_ONLY", None)
    os.environ.pop("ROS_STATIC_PEERS", None)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    os.environ.setdefault("CYCLONEDDS_URI", "file://"+str(root/"config/transport/cyclonedds_large_clouds.xml"))
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/camera-lidar-bev-lidar-detector-check-logs")

    import numpy as np
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from geometry_msgs.msg import TransformStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    from tf2_msgs.msg import TFMessage
    from vision_msgs.msg import Detection3DArray
    from visualization_msgs.msg import Marker, MarkerArray

    with np.load(args.capture, allow_pickle=False) as captured:
        raw_xyz = np.column_stack([captured[axis] for axis in ("x", "y", "z")]).astype(np.float32)
        reflectivity = np.asarray(captured["reflectivity"], dtype=np.uint16).copy()
    if raw_xyz.ndim != 2 or raw_xyz.shape[1] != 3 or not 1 <= len(raw_xyz) <= 300_000:
        parser.error("Capture has invalid XYZ shape or point count")
    if reflectivity.shape != (len(raw_xyz),) or not np.isfinite(raw_xyz).all():
        parser.error("Capture must contain matching finite native XYZ and reflectivity")
    # These are the exact source rows of the minimal replay cloud. The NPZ has
    # already retained valid original returns; its rows are not original scan IDs.
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("reflectivity", "<u2"), ("padding", "u1", (2,))])
    packed = np.zeros(len(raw_xyz), dtype=dtype)
    for index, axis in enumerate(("x", "y", "z")):
        packed[axis] = raw_xyz[:, index]
    packed["reflectivity"] = reflectivity
    native_xyz = raw_xyz.copy()
    native_xyz[:, :2] *= -1
    native_xyz[:, 2] += np.float32(0.038195)
    kitti = np.fromfile(args.kitti_reference, dtype="<f4")
    if kitti.size % 4 or not kitti.size:
        parser.error("KITTI reference must contain nonempty XYZI float32 records")
    kitti = kitti.reshape(-1, 4)
    if not np.isfinite(kitti).all() or np.any(kitti[:, 3] < 0) or np.any(kitti[:, 3] > 1):
        parser.error("KITTI reference must contain finite native XYZ and normalized reflectivity")
    kitti_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("reflectivity", "<f4")])
    kitti_packed = np.zeros(len(kitti), dtype=kitti_dtype)
    for index, axis in enumerate(("x", "y", "z")):
        kitti_packed[axis] = kitti[:, index]
    kitti_packed["reflectivity"] = kitti[:, 3]*np.float32(255.0)
    prefix = "/lidar_vehicle_software_check"
    topics = {name: prefix+"/"+name for name in ("input", "detections", "markers", "points", "diagnostics", "tf_static")}
    command = [str(args.executable), "--ros-args", "-p", f"engine_file:={args.engine}",
               "-p", f"input_topic:={topics['input']}", "-p", f"detections_topic:={topics['detections']}",
               "-p", f"markers_topic:={topics['markers']}", "-p", f"points_topic:={topics['points']}",
               "-p", "score_threshold:=0.35", "-p", "output_frame:=os_sensor",
               "-p", "classes:=[car,person]",
               "-p", "intensity_field:=reflectivity", "-p", "intensity_scale:=255.0",
               "-r", f"/camera_lidar_bev/diagnostics:={topics['diagnostics']}",
               "-r", f"/tf_static:={topics['tf_static']}"]
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "domain_id": args.domain_id, "software_only": True, "trained_engine_used": True,
              "physical_accuracy_validated": False, "real_time_speed_validated": False,
              "source_is_real_captured_scan": True, "source_index_refers_to_replayed_npz_row": True,
              "capture": str(args.capture), "capture_sha256": hashlib.sha256(args.capture.read_bytes()).hexdigest(),
              "capture_points": len(raw_xyz), "capture_input_frame": "os_lidar",
              "kitti_reference": str(args.kitti_reference),
              "kitti_reference_sha256": hashlib.sha256(args.kitti_reference.read_bytes()).hexdigest(),
              "kitti_reference_points": len(kitti), "kitti_input_frame": "os_sensor",
              "kitti_intensity_adaptation": "FLOAT32 reflectivity = original normalized intensity * 255",
              "engine": str(args.engine), "engine_sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
              "executable": str(args.executable),
              "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(), "stages": []}
    rclpy.init()
    node = rclpy.create_node("lidar_vehicle_software_verifier")
    output_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
    input_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    tf_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(PointCloud2, topics["input"], input_qos)
    tf_pub = node.create_publisher(TFMessage, topics["tf_static"], tf_qos)
    received = {key: [] for key in ("detections", "markers", "points", "diagnostics")}

    def on_message(key, message):
        received[key].append((time.monotonic(), message))
        del received[key][:-200]

    subscriptions = [node.create_subscription(kind, topics[key],
                     lambda message, key=key: on_message(key, message), output_qos) for key, kind in
                     (("detections", Detection3DArray), ("markers", MarkerArray),
                      ("points", PointCloud2), ("diagnostics", DiagnosticArray))]
    deadline = time.monotonic()+args.timeout
    last_positive_stamp = 0
    last_positive_cloud = None
    active_phase = "discovery"

    def header(stamp, frame="os_lidar"):
        item = Header(frame_id=frame)
        item.stamp.sec, item.stamp.nanosec = divmod(int(stamp), 1_000_000_000)
        return item

    def cloud(kind="positive"):
        use_kitti = kind == "kitti_person"
        rows = kitti_packed if use_kitti else packed
        stamp = node.get_clock().now().nanoseconds-20_000_000
        if kind == "future":
            stamp += 2_000_000_000
        frame = "wrong_lidar_frame" if kind == "wrong_frame" else "os_sensor" if use_kitti else "os_lidar"
        message = PointCloud2(header=header(stamp, frame),
                              height=1, width=len(rows), point_step=rows.dtype.itemsize,
                              row_step=len(rows)*rows.dtype.itemsize, is_bigendian=False, is_dense=True,
                              data=rows.tobytes())
        message.fields = [PointField(name=axis, offset=index*4, datatype=PointField.FLOAT32, count=1)
                          for index, axis in enumerate(("x", "y", "z"))]
        if kind != "missing_reflectivity":
            message.fields.append(PointField(name="reflectivity", offset=12,
                                             datatype=PointField.FLOAT32 if use_kitti else PointField.UINT16, count=1))
        return message

    def driver_tf(conflict=False):
        item = TransformStamped(header=header(node.get_clock().now().nanoseconds, "os_sensor"), child_frame_id="os_lidar")
        item.transform.translation.x = 0.02 if conflict else 0.0
        item.transform.translation.z = 0.038195
        item.transform.rotation.z = 1.0
        item.transform.rotation.w = 0.0
        return TFMessage(transforms=[item])

    def matching(start):
        messages = {}
        for key in ("detections", "markers", "points"):
            messages[key] = {}
            for receipt, message in received[key]:
                if receipt < start:
                    continue
                if key == "markers":
                    assert message.markers, "Every marker output must include explicit DELETEALL"
                    source_header = message.markers[0].header
                else:
                    source_header = message.header
                messages[key][stamp_ns(source_header)] = message
        common = messages["detections"].keys() & messages["markers"].keys() & messages["points"].keys()
        for stamp in sorted(common, reverse=True):
            yield stamp, messages["detections"][stamp], messages["markers"][stamp], messages["points"][stamp]

    def scalar(points, name):
        fields = [item for item in points.fields if item.name == name]
        assert len(fields) == 1 and fields[0].count == 1
        field = fields[0]
        code = "u4" if name in ("rgb", "source_index") else "f4"
        expected = PointField.UINT32 if name == "source_index" else PointField.FLOAT32
        assert field.datatype == expected
        return np.ndarray((points.height, points.width), dtype=(">" if points.is_bigendian else "<")+code,
                          buffer=bytes(points.data), offset=field.offset,
                          strides=(points.row_step, points.point_step)).reshape(-1).copy()

    def check_positive(start, source_clouds, require_person=False):
        nonlocal last_positive_stamp, last_positive_cloud
        for stamp, detections, markers, points in matching(start):
            if stamp not in source_clouds or not detections.detections:
                continue
            assert detections.header.frame_id == points.header.frame_id == "os_sensor"
            assert markers.markers[0].action == Marker.DELETEALL
            assert len(markers.markers) == 1+2*len(detections.detections)
            boxes = []
            classes = [item.results[0].hypothesis.class_id for item in detections.detections if len(item.results) == 1]
            if require_person and "person" not in classes:
                continue
            for index, item in enumerate(detections.detections):
                assert item.header.frame_id == "os_sensor" and stamp_ns(item.header) == stamp and item.id == ""
                assert len(item.results) == 1 and item.results[0].hypothesis.class_id in ("car", "person")
                class_name = item.results[0].hypothesis.class_id
                rgb = 0xff00ff if class_name == "person" else 0xffa500
                score = item.results[0].hypothesis.score
                assert math.isfinite(score) and 0.35 <= score <= 1
                center = np.asarray([getattr(item.bbox.center.position, axis) for axis in ("x", "y", "z")])
                size = np.asarray([getattr(item.bbox.size, axis) for axis in ("x", "y", "z")])
                assert np.isfinite(center).all() and np.isfinite(size).all() and 0 <= center[0] <= 60
                assert np.all(size >= ((0.1, 0.1, 0.4) if class_name == "person" else (0.5, 0.2, 0.2)))
                assert np.all(size <= (15.0, 6.0, 5.0))
                q = item.bbox.center.orientation
                assert q.x == q.y == 0 and math.isclose(q.z*q.z+q.w*q.w, 1.0, abs_tol=1e-7)
                yaw = 2*math.atan2(q.z, q.w)
                co, si = math.cos(yaw), math.sin(yaw)
                rectangle, label = markers.markers[1+2*index:3+2*index]
                assert rectangle.ns == "predicted_vehicle_bev_extent" and rectangle.type == Marker.LINE_STRIP
                assert label.ns == "predicted_vehicle_bev_labels" and label.type == Marker.TEXT_VIEW_FACING
                assert label.text == f"{class_name} {score:.2f}\nL {size[0]:.2f} W {size[1]:.2f} H {size[2]:.2f} m"
                for marker in (rectangle, label):
                    assert np.allclose((marker.color.r, marker.color.g, marker.color.b, marker.color.a),
                                       ((rgb >> 16 & 255)/255, (rgb >> 8 & 255)/255, (rgb & 255)/255, 1.0),
                                       atol=1e-7, rtol=0)
                assert np.allclose((label.pose.position.x, label.pose.position.y, label.pose.position.z),
                                   (center[0], center[1], 0.17), atol=1e-8, rtol=0)
                assert math.isclose(rectangle.scale.x, 0.08, abs_tol=1e-9)
                assert math.isclose(label.scale.z, 0.55, abs_tol=1e-9)
                corners = np.asarray([(point.x, point.y, point.z) for point in rectangle.points])
                assert corners.shape == (5, 3) and np.allclose(corners[0], corners[-1], atol=1e-9)
                expected = []
                for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)):
                    dx, dy = sx*size[0]/2, sy*size[1]/2
                    expected.append((center[0]+co*dx-si*dy, center[1]+si*dx+co*dy, 0.15))
                assert np.allclose(corners, expected, atol=1e-6, rtol=0)
                boxes.append((center, size, co, si, class_name, rgb))
            for marker in markers.markers:
                assert marker.header.frame_id == "os_sensor" and stamp_ns(marker.header) == stamp
                if marker.action == Marker.DELETEALL:
                    continue
                assert marker.action == Marker.ADD
                q = marker.pose.orientation
                assert (q.x, q.y, q.z, q.w) == (0.0, 0.0, 0.0, 1.0)
                assert math.isclose(marker.lifetime.sec+marker.lifetime.nanosec/1e9, 0.5, abs_tol=1e-9)
            assert points.height == 1 and points.width > 0 and points.point_step == 20
            assert points.row_step == points.width*points.point_step and len(points.data) == points.row_step
            ids = scalar(points, "source_index")
            source_cloud = source_clouds[stamp]
            expected_xyz = kitti[:, :3] if source_cloud.header.frame_id == "os_sensor" else native_xyz
            assert len(set(map(int, ids))) == len(ids) and np.all(ids < len(expected_xyz))
            selected = np.column_stack([scalar(points, axis) for axis in ("x", "y", "z")])
            assert np.allclose(selected, expected_xyz[ids], atol=1e-6, rtol=0)
            point_colors = scalar(points, "rgb")
            assert np.all(np.isin(point_colors, (0xffa500, 0xff00ff)))
            supported = np.zeros(len(selected), dtype=bool)
            for center, size, co, si, class_name, rgb in boxes:
                delta = selected-center
                local = np.column_stack((co*delta[:, 0]+si*delta[:, 1],
                                         -si*delta[:, 0]+co*delta[:, 1], delta[:, 2]))
                supported |= np.all(np.abs(local) <= size/2+1e-6, axis=1) & (point_colors == rgb)
            assert supported.all(), "Published point lies outside every predicted oriented cuboid"
            last_positive_stamp, last_positive_cloud = stamp, source_clouds[stamp]
            return {"source_stamp_ns": stamp, "cars": classes.count("car"), "people": classes.count("person"),
                    "vehicle_points": int(points.width), "input_frame": source_cloud.header.frame_id,
                    "predicted_size_m": [box[1].tolist() for box in boxes],
                    "oriented_flat_footprints_verified": True, "size_labels_verified": True,
                    "transformed_source_rows_verified": True, "points_inside_predicted_cuboids": True}
        return None

    def check_clear(start, minimum_stamp=0):
        for stamp, detections, markers, points in matching(start):
            if stamp < minimum_stamp or stamp <= last_positive_stamp or detections.detections:
                continue
            assert detections.header.frame_id == points.header.frame_id == "os_sensor"
            assert points.width == 0 and points.height == 1 and points.point_step == 20
            assert points.row_step == 0 and len(points.data) == 0
            assert len(markers.markers) == 1 and markers.markers[0].action == Marker.DELETEALL
            assert markers.markers[0].header.frame_id == "os_sensor" and stamp_ns(markers.markers[0].header) == stamp
            return {"clear_stamp_ns": stamp, "empty_detections": True, "empty_points": True,
                    "explicit_markers_delete_all": True, "three_headers_matched": True}
        return None

    def record(name, evidence):
        report["stages"].append(dict(name=name, passed=True, **evidence))
        print(name+": passed", flush=True)

    def positive(name, fixture="positive", require_person=False):
        nonlocal active_phase
        active_phase = name
        start = time.monotonic()
        sources = {}
        next_send = start
        while time.monotonic() < min(deadline, start+2.0):
            now = time.monotonic()
            if now >= next_send:
                message = cloud(fixture)
                sources[stamp_ns(message.header)] = message
                pub.publish(message)
                next_send = now+0.12
            rclpy.spin_once(node, timeout_sec=0.01)
            if time.monotonic()-start >= 0.15:
                evidence = check_positive(start, sources, require_person)
                if evidence:
                    record(name, evidence)
                    return
            if process.poll() is not None:
                raise RuntimeError("Trained node exited during "+name)
        raise AssertionError("Missing real trained positive output during "+name)

    def failure(kind):
        nonlocal active_phase
        active_phase = kind
        start = time.monotonic()
        start_ns = node.get_clock().now().nanoseconds
        if kind == "duplicate":
            assert last_positive_cloud is not None
            pub.publish(copy.deepcopy(last_positive_cloud))
        elif kind == "conflicting_driver_tf":
            tf_pub.publish(driver_tf(conflict=True))
        elif kind != "stale":
            pub.publish(cloud(kind))
        while time.monotonic() < min(deadline, start+1.6):
            rclpy.spin_once(node, timeout_sec=0.02)
            if time.monotonic()-start < (0.65 if kind == "stale" else 0.08):
                continue
            evidence = check_clear(start, start_ns-30_000_000)
            if evidence:
                record(kind, evidence)
                return
            if process.poll() is not None:
                raise RuntimeError("Trained node exited during "+kind)
        raise AssertionError("Missing explicit three-output clear during "+kind)

    process = None
    with tempfile.TemporaryFile() as log:
        try:
            launched = time.monotonic()
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            discovery_deadline = min(deadline, launched+8.0)
            while time.monotonic() < discovery_deadline:
                rclpy.spin_once(node, timeout_sec=0.03)
                if pub.get_subscription_count() and tf_pub.get_subscription_count():
                    break
                if process.poll() is not None:
                    raise RuntimeError("Trained node exited during discovery")
            else:
                raise TimeoutError("Isolated ROS discovery did not complete")
            active_phase = "missing_driver_tf"
            pub.publish(cloud())
            evidence = None
            start = time.monotonic()
            while time.monotonic() < min(deadline, start+1.5):
                rclpy.spin_once(node, timeout_sec=0.02)
                evidence = check_clear(launched)
                if evidence:
                    break
            if not evidence:
                raise AssertionError("Node did not explicitly clear while actual driver TF was missing")
            record(active_phase, evidence)
            tf_pub.publish(driver_tf())
            for kind in ("wrong_frame", "missing_reflectivity", "future", "duplicate", "stale"):
                positive("positive_before_"+kind)
                failure(kind)
            positive("kitti_trained_person", fixture="kitti_person", require_person=True)
            positive("positive_before_tf_conflict")
            failure("conflicting_driver_tf")
            # Wait for one diagnostic update with the final permanent TF gate.
            active_phase = "diagnostics"
            diagnostic_deadline = min(deadline, time.monotonic()+1.2)
            values = {}
            while time.monotonic() < diagnostic_deadline:
                rclpy.spin_once(node, timeout_sec=0.03)
                statuses = [status for _, array in received["diagnostics"] for status in array.status
                            if status.name == "lidar_vehicle_detector"]
                if statuses:
                    values = {item.key: item.value for item in statuses[-1].values}
                    if (values.get("driver_tf_valid") == "false" and
                            values.get("publishing_fresh_output") == "false" and
                            int(values.get("processed_scans", "0")) > 0):
                        break
            for key, expected in (("trained_3d_detector", "true"), ("full_dimensions_predicted", "true"),
                                  ("heading_predicted", "true"), ("physical_accuracy_validated", "false"),
                                  ("tracking_enabled", "false"), ("frame", "os_sensor"),
                                  ("intensity_field", "reflectivity"), ("vehicle_class", "Car_and_Pedestrian"),
                                  ("driver_tf_valid", "false"), ("publishing_fresh_output", "false")):
                assert values.get(key) == expected, (key, values.get(key))
            assert int(values.get("processed_scans", "0")) > 0, "Final TF-gate diagnostics must follow real inference"
            report["last_diagnostics"] = values
            report["passed"] = True
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            report["failed_stage"] = active_phase
            report["received_output_messages"] = {key: len(items) for key, items in received.items()}
            statuses = [status for _, array in received["diagnostics"] for status in array.status
                        if status.name == "lidar_vehicle_detector"]
            if statuses:
                report["last_diagnostic_message"] = statuses[-1].message
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
