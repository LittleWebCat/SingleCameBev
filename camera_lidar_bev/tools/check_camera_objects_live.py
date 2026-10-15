#!/usr/bin/env python3
"""Observe the running camera labels and stationary BEV without opening sensors."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import traceback


CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {0: 0xff00ff, 1: 0x00ffff, 2: 0xffa500, 3: 0xffa500, 5: 0xffa500, 7: 0xffa500}


def stamp(header):
    return int(header.stamp.sec)*1_000_000_000+int(header.stamp.nanosec)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--output", type=Path, default=root / "evaluation/camera_objects_live.json")
    parser.add_argument("--first-image", type=Path)
    parser.add_argument("--last-image", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 10 <= args.duration <= 20:
        parser.error("duration must be between 10 and 20 seconds")
    paths = [path for path in (args.output, args.first_image, args.last_image) if path is not None]
    if len({str(path.resolve()) for path in paths}) != len(paths):
        parser.error("report and image paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in paths):
        parser.error("output already exists; choose fresh report and image paths")
    import cv2
    import numpy as np
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from nav_msgs.msg import OccupancyGrid
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, PointCloud2
    from vision_msgs.msg import Detection2DArray, Detection3DArray
    from visualization_msgs.msg import MarkerArray
    report = {"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "read_only_live_camera_object_bev_observation", "duration_requested_s": args.duration,
              "provisional_stationary_mode": True, "stationarity_independently_verified": False,
              "physical_calibration_accuracy_validated": False, "timing_and_motion_validated": False,
              "learned_3d_detection_validated": False, "complete_object_dimensions_validated": False,
              "positive_camera_object_observed": False, "positive_live_object_observed": False,
              "errors": [], "streams": {}, "diagnostics": {}, "saved_images": {}}
    streams = {name: {"receipts": [], "stamps": [], "counts": []} for name in
               ("detections_2d", "labeled_image", "detections_3d", "markers", "semantic_points", "occupancy")}
    diagnostic_samples = {}
    last_image = None
    node = None
    started = time.monotonic()
    def record(name, header, count):
        key = stamp(header)
        if key <= 0 or header.stamp.nanosec >= 1_000_000_000:
            raise ValueError(f"{name}: invalid output source timestamp")
        item = streams[name]
        item["receipts"].append(time.monotonic())
        item["stamps"].append(key)
        item["counts"].append(int(count))
    def optical(header):
        if header.frame_id != "camera_front_optical_frame":
            raise ValueError("Camera output frame is not the native optical frame")
    def sensor(header):
        if header.frame_id != "os_sensor":
            raise ValueError("BEV object output frame is not os_sensor")
    def identity(q):
        if not np.allclose([q.x, q.y, q.z, q.w], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("Observed extents must have identity orientation, without an inferred heading")
    def hypothesis(detection, header):
        if stamp(detection.header) != stamp(header) or detection.header.frame_id != header.frame_id:
            raise ValueError("Detection child header differs from array source header")
        if len(detection.results) != 1 or detection.id:
            raise ValueError("Detection has an invalid hypothesis count or invented tracking ID")
        item = detection.results[0].hypothesis
        if item.class_id not in CLASSES.values() or not math.isfinite(item.score) or not 0.35 <= item.score <= 1:
            raise ValueError("Invalid camera class or confidence")
    def image_callback(message):
        nonlocal last_image
        optical(message.header)
        if (message.width != 1024 or message.height != 768 or message.encoding != "bgr8"
                or message.is_bigendian or message.step < 3072 or len(message.data) != message.step*768):
            raise ValueError("Labeled image is not native 1024x768 BGR8")
        last_image = np.ndarray((768, 1024, 3), dtype=np.uint8, buffer=bytes(message.data),
                                strides=(message.step, 3, 1)).copy()
        if not streams["labeled_image"]["receipts"] and args.first_image:
            args.first_image.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.first_image), last_image):
                raise ValueError("Could not save first labeled camera image")
            report["saved_images"]["first"] = str(args.first_image.resolve())
        record("labeled_image", message.header, 1)
    def detections2d_callback(message):
        optical(message.header)
        for detection in message.detections:
            hypothesis(detection, message.header)
            box = detection.bbox
            x, y, w, h = box.center.position.x, box.center.position.y, box.size_x, box.size_y
            if (not np.isfinite([x, y, w, h, box.center.theta]).all() or w <= 0 or h <= 0
                    or box.center.theta != 0 or x-w/2 < -1e-6 or y-h/2 < -1e-6
                    or x+w/2 > 1024+1e-6 or y+h/2 > 768+1e-6):
                raise ValueError("2D camera box is outside native image coordinates")
        report["positive_camera_object_observed"] |= bool(message.detections)
        record("detections_2d", message.header, len(message.detections))
    def detections3d_callback(message):
        sensor(message.header)
        for detection in message.detections:
            hypothesis(detection, message.header)
            box = detection.bbox
            if (not np.isfinite([box.center.position.x, box.center.position.y, box.center.position.z,
                                 box.size.x, box.size.y, box.size.z]).all()
                    or min(box.size.x, box.size.y, box.size.z) <= 0
                    or max(box.size.x, box.size.y) > 12 or box.size.z > 4.5):
                raise ValueError("3D observed extent contains invalid position or dimensions")
            identity(box.center.orientation)
        report["positive_live_object_observed"] |= bool(message.detections)
        record("detections_3d", message.header, len(message.detections))
    def markers_callback(message):
        if not message.markers or message.markers[0].action != 3:
            raise ValueError("Each object-marker update must begin with DELETEALL")
        header = message.markers[0].header
        labels = {item.id-1: item.text.split()[0] for item in message.markers[1:] if item.type == 9 and item.text}
        for item in message.markers:
            sensor(item.header)
            if stamp(item.header) != stamp(header):
                raise ValueError("Object marker array mixes source headers")
            if item.action == 3:
                continue
            lifetime = item.lifetime.sec+item.lifetime.nanosec/1e9
            identity(item.pose.orientation)
            name = item.text.split()[0] if item.type == 9 and item.text else labels.get(item.id)
            class_id = next((key for key, value in CLASSES.items() if value == name), None)
            if class_id is None or item.action != 0 or not 0 < lifetime <= 0.75 or item.type not in (5, 9):
                raise ValueError("Object marker class, action, type or short lifetime is invalid")
            rgb = [(COLORS[class_id] >> shift & 255)/255 for shift in (16, 8, 0)]
            if not np.allclose([item.color.r, item.color.g, item.color.b, item.color.a], rgb+[1], atol=1e-6):
                raise ValueError("Object marker colors differ from the camera class palette")
            if item.type == 5 and (len(item.points) != 24 or not np.isfinite([[p.x, p.y, p.z] for p in item.points]).all()):
                raise ValueError("Observed box marker must contain twelve finite edges")
        record("markers", header, len(message.markers)-1)
    def semantic_callback(message):
        sensor(message.header)
        fields = {item.name: item for item in message.fields}
        expected = {"x": (0, 7), "y": (4, 7), "z": (8, 7), "rgb": (12, 7),
                    "source_index": (16, 6), "class_id": (20, 4), "confidence": (24, 7)}
        if (message.is_bigendian or message.point_step != 28 or len(fields) != 7 or len(message.fields) != 7
                or message.width*28 > message.row_step or len(message.data) != message.row_step*message.height
                or any(name not in fields or (fields[name].offset, fields[name].datatype) != layout
                       or fields[name].count != 1 for name, layout in expected.items())):
            raise ValueError("Semantic point subset layout is invalid")
        dtype = np.dtype({"names": list(expected), "formats": ["<f4", "<f4", "<f4", "<u4", "<u4", "<u2", "<f4"],
                          "offsets": [0, 4, 8, 12, 16, 20, 24], "itemsize": 28})
        points = np.ndarray((message.height, message.width), dtype=dtype, buffer=bytes(message.data),
                            strides=(message.row_step, 28)).reshape(-1)
        if len(points) and (not all(np.isfinite(points[name]).all() for name in ("x", "y", "z", "confidence"))
                           or not np.isin(points["class_id"], list(CLASSES)).all()
                           or np.any((points["confidence"] < .35) | (points["confidence"] > 1))
                           or len(np.unique(points["source_index"])) != len(points)):
            raise ValueError("Semantic subset has invalid class, confidence, geometry or duplicate source IDs")
        for class_id, color in COLORS.items():
            if np.any((points["rgb"][points["class_id"] == class_id] & 0xffffff) != color):
                raise ValueError("Semantic point colors differ from camera classes")
        record("semantic_points", message.header, len(points))
    def occupancy_callback(message):
        sensor(message.header)
        if (message.info.width != 200 or message.info.height != 150 or abs(message.info.resolution-.2) > 1e-6
                or len(message.data) != 30000 or not np.isin(message.data, [-1, 0, 100]).all()
                or not np.allclose([message.info.origin.position.x, message.info.origin.position.y,
                                    message.info.origin.position.z], [0, -15, 0], atol=1e-6)):
            raise ValueError("Stationary occupancy grid format or states changed")
        identity(message.info.origin.orientation)
        record("occupancy", message.header, int(np.count_nonzero(np.asarray(message.data) != -1)))
    def diagnostics_callback(message):
        for status in message.status:
            if status.name not in ("camera_object_detector", "camera_object_bev", "ground_bev"):
                continue
            values = {item.key: item.value for item in status.values}
            if status.name == "ground_bev" and (values.get("provisional") != "true" or values.get("stationary_mode") != "true"):
                raise ValueError("Ground diagnostics lost provisional stationary mode")
            if status.name == "camera_object_bev" and (values.get("accuracy_validated") != "false" or values.get("occupancy_modified") != "false"):
                raise ValueError("Fusion diagnostics claim unvalidated accuracy or change occupancy")
            level = status.level[0] if isinstance(status.level, (bytes, bytearray)) else int(status.level)
            diagnostic_samples.setdefault(status.name, []).append({"receipt": time.monotonic(), "level": level,
                                                                   "message": status.message, "values": values})
    def guarded(callback):
        def receive(message):
            try:
                callback(message)
            except Exception as error:
                if str(error) not in report["errors"]:
                    report["errors"].append(str(error))
        return receive
    try:
        rclpy.init()
        node = rclpy.create_node("camera_objects_live_observer")
        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        latest_image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        subscriptions = [node.create_subscription(kind, "/camera_lidar_bev/"+topic, guarded(callback),
                         latest_image_qos if kind is Image else reliable)
                         for kind, topic, callback in [(Image, "camera_labeled", image_callback),
                         (Detection2DArray, "detections_2d", detections2d_callback),
                         (Detection3DArray, "detections_3d", detections3d_callback),
                         (MarkerArray, "object_markers", markers_callback),
                         (PointCloud2, "semantic_points", semantic_callback),
                         (DiagnosticArray, "diagnostics", diagnostics_callback)]]
        subscriptions.append(node.create_subscription(OccupancyGrid, "/camera_lidar_bev/occupancy", guarded(occupancy_callback),
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)))
        while time.monotonic()-started < args.duration:
            rclpy.spin_once(node, timeout_sec=.02)
        if args.last_image and last_image is not None:
            args.last_image.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.last_image), last_image):
                raise ValueError("Could not save last labeled camera image")
            report["saved_images"]["last"] = str(args.last_image.resolve())
        _ = subscriptions
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["traceback"] = traceback.format_exc()
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    for name, item in streams.items():
        times, unique, counts = item["receipts"], sorted(set(item["stamps"])), item["counts"]
        report["streams"][name] = {"messages": len(times), "distinct_source_headers": len(unique),
            "receipt_span_fps": (len(times)-1)/(times[-1]-times[0]) if len(times) > 1 else 0,
            "source_header_span_fps": (len(unique)-1)/((unique[-1]-unique[0])/1e9) if len(unique) > 1 else 0,
            "nonempty_messages": sum(count > 0 for count in counts), "first_count": counts[0] if counts else None,
            "last_count": counts[-1] if counts else None, "max_count": max(counts) if counts else 0}
        if len(times) < 2:
            report["errors"].append(f"Insufficient live {name} outputs observed")
    for name, samples in diagnostic_samples.items():
        metrics = {}
        for key in {key for sample in samples for key in sample["values"] if key.endswith(("_ms", "_s"))}:
            values = [float(sample["values"][key]) for sample in samples if key in sample["values"]]
            values = [value for value in values if math.isfinite(value)]
            if values:
                metrics[key] = {"samples": len(values), "min": min(values), "median": float(np.median(values)),
                                "p95": float(np.percentile(values, 95)), "max": max(values)}
        counter = {"camera_object_detector": "processed_frames", "camera_object_bev": "processed_arrays", "ground_bev": "processed_scans"}[name]
        rate = ((int(samples[-1]["values"].get(counter, 0))-int(samples[0]["values"].get(counter, 0))) /
                (samples[-1]["receipt"]-samples[0]["receipt"])) if len(samples) > 1 else 0
        report["diagnostics"][name] = {"samples": len(samples), "processed_counter_rate_hz": rate,
            "timing_sample_policy": "one reported last-frame timing per approximately 1 Hz diagnostic, not every inference",
            "metrics": metrics, "first": samples[0], "last": samples[-1]}
    for name in ("camera_object_detector", "camera_object_bev", "ground_bev"):
        if name not in report["diagnostics"]:
            report["errors"].append(f"Missing live {name} diagnostics")
    report["elapsed_s"] = time.monotonic()-started
    report["passed"] = not report["errors"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.output), "errors": report["errors"],
                      "positive_live_object_observed": report["positive_live_object_observed"]}))
    return 0 if report["passed"] else 1
if __name__ == "__main__":
    raise SystemExit(main())
