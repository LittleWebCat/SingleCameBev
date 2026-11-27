#!/usr/bin/env python3
"""Bounded, isolated native TensorRT camera-detector integration check.

Use the repo .venv interpreter after sourcing the ROS/workspace setups. The
verifier replays a bundled photographic asset on private localhost ROS topics;
it does not open a camera or LiDAR, access the live domain, or download models.
It compares native C++ output to the same TensorRT engine through Ultralytics.
"""

import argparse
from contextlib import contextmanager
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
import traceback


ROAD_IDS = [0, 1, 2, 3, 5, 7]
ROAD_NAMES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
FRAME = "camera_front_optical_frame"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stamp_key(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


@contextmanager
def child_process(arguments, report):
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield process
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 6000))
            report["node_log_tail"] = log.read().decode("utf-8", errors="replace")
            report["child_returncode"] = process.returncode


def main():
    root = Path(__file__).resolve().parents[1]
    repo = root.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True, help="raw native TensorRT engine")
    parser.add_argument("--ultralytics-engine", type=Path, required=True,
                        help="same engine with fresh Ultralytics COCO metadata")
    parser.add_argument("--source-asset", type=Path,
                        default=repo / ".venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg")
    parser.add_argument("--executable", type=Path,
                        default=root / "install/bev_visualization/lib/bev_visualization/camera_object_detector")
    parser.add_argument("--output", type=Path, default=root / "evaluation/camera_detector_replay.json")
    parser.add_argument("--domain-id", type=int, default=74)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new evidence path")
    if not 2 <= args.domain_id <= 231:
        parser.error("domain-id must be between 2 and 231, separate from live domain 1")
    if not math.isfinite(args.timeout) or not 20 <= args.timeout <= 30:
        parser.error("timeout must be between 20 and 30 seconds")
    for path in (args.engine, args.ultralytics_engine, args.source_asset, args.executable):
        if not path.is_file():
            parser.error(f"Required local file is missing: {path}")

    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ.pop("ROS_LOCALHOST_ONLY", None)
    os.environ.pop("ROS_STATIC_PEERS", None)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    os.environ.setdefault("CYCLONEDDS_URI", "file://" + str(root / "config/transport/cyclonedds_large_clouds.xml"))
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/camera_lidar_bev-camera-detector-check-logs")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = "/tmp/camera_lidar_bev-camera-detector-check-yolo"

    report = {"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "native_tensor_rt_camera_detector_software_check",
              "ros_domain_id": args.domain_id, "live_hardware_used": False,
              "physical_calibration_accuracy_validated": False,
              "timing_and_motion_validated": False, "learned_3d_detection_validated": False,
              "image_size": [1024, 768], "camera_frame": FRAME,
              "source_files": {}, "reference": {}, "stage_results": {}, "errors": [],
              "detection_messages": 0, "labeled_image_messages": 0,
              "verified_source_pairs": 0, "verified_clear_pairs": 0}
    started = time.monotonic()
    deadline = started + args.timeout
    node = None
    previous_alarm = signal.getsignal(signal.SIGALRM)

    def alarm(unused_signal, unused_frame):
        raise TimeoutError("Bounded camera detector verifier deadline exceeded")

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, args.timeout)
    try:
        import cv2
        import numpy as np
        import rclpy
        from diagnostic_msgs.msg import DiagnosticArray
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image
        from ultralytics import YOLO
        from vision_msgs.msg import Detection2DArray

        # Ensure the native plan and reference execute identical serialized bytes.
        with args.ultralytics_engine.open("rb") as stream:
            metadata_length = int.from_bytes(stream.read(4), "little")
            if not 1 <= metadata_length <= 65536:
                raise ValueError("Reference engine lacks bounded Ultralytics metadata")
            metadata = json.loads(stream.read(metadata_length).decode("utf-8"))
            reference_plan_hash = hashlib.sha256(stream.read()).hexdigest()
        if reference_plan_hash != sha256(args.engine):
            raise ValueError("Native and Ultralytics engines do not contain the same TensorRT plan")
        names = {int(key): value for key, value in metadata.get("names", {}).items()}
        if metadata.get("batch") != 1 or metadata.get("imgsz") != [640, 640] or len(names) != 80:
            raise ValueError("Reference engine metadata must identify batch-one 640x640 COCO detection")
        if [names.get(key) for key in ROAD_IDS] != ROAD_NAMES:
            raise ValueError("Reference metadata does not match validated COCO road-user labels")
        for path in (args.engine, args.ultralytics_engine, args.source_asset):
            report["source_files"][str(path.resolve())] = sha256(path)
        source = cv2.imread(str(args.source_asset), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError("Bundled source asset could not be decoded")
        frame = cv2.resize(source, (1024, 768), interpolation=cv2.INTER_LINEAR)
        inference_started = time.monotonic()
        model = YOLO(str(args.ultralytics_engine), task="detect")
        result = model.predict(source=frame, imgsz=640, rect=False, conf=0.35, iou=0.5,
                               classes=ROAD_IDS, max_det=100, device=0, verbose=False)[0]
        expected = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            expected.append({"class_name": result.names[class_id],
                             "score": float(box.conf.item()),
                             "bbox_xyxy": [float(value) for value in box.xyxy[0].tolist()]})
        report["reference"] = {"detections": expected,
                               "setup_and_prediction_ms": (time.monotonic()-inference_started)*1000,
                               "speed_ms": {key: float(value) for key, value in result.speed.items()},
                               "confidence": 0.35, "nms_iou": 0.5, "rect": False,
                               "engine_metadata": {key: metadata.get(key) for key in
                                                   ("version", "description", "batch", "imgsz", "license")},
                               "native_box_tolerance_px": 2.0, "score_tolerance": 0.005}
        if not expected or not {item["class_name"] for item in expected}.intersection({"person", "bus"}):
            raise ValueError("Real bundled asset produced no reference person/bus detections")

        rclpy.init()
        node = rclpy.create_node("camera_detector_isolated_check")
        prefix = "/camera_detector_check"
        topics = {"image_topic": prefix + "/input", "detections_topic": prefix + "/detections",
                  "labeled_image_topic": prefix + "/labeled_image", "diagnostics_topic": prefix + "/diagnostics"}
        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latest_image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        publisher = node.create_publisher(Image, topics["image_topic"], reliable)
        inputs = {}
        records = {}
        verified = set()
        clear_records = {}
        clear_verified = set()
        active_stage = "discovery"
        diagnostics = []

        def record_error(error):
            if str(error) not in report["errors"]:
                report["errors"].append(str(error))
                report.setdefault("callback_tracebacks", []).append(traceback.format_exc())

        def compare_detections(actual):
            if len(actual) != len(expected):
                raise ValueError(f"C++/Ultralytics detection counts differ: {len(actual)} versus {len(expected)}")
            unmatched = list(expected)
            matches = []
            for detection in actual:
                class_matches = [(float(np.max(np.abs(np.asarray(detection["bbox_xyxy"])-item["bbox_xyxy"]))), index)
                                 for index, item in enumerate(unmatched)
                                 if item["class_name"] == detection["class_name"]]
                if not class_matches:
                    raise ValueError("C++ emitted a class absent from the reference detections")
                difference, index = min(class_matches)
                reference = unmatched.pop(index)
                score_difference = abs(detection["score"]-reference["score"])
                if difference > 2.0 or score_difference > 0.005:
                    raise ValueError(f"C++/Ultralytics native boxes or scores differ: {difference:.3f}px / {score_difference:.6f}")
                matches.append({"class_name": detection["class_name"], "max_box_difference_px": difference,
                                "score_difference": score_difference})
            return matches

        def pair(key):
            if key in verified or set(records.get(key, {})) != {"detections", "image"}:
                return
            origin = inputs[key]
            matches = compare_detections(records[key]["detections"])
            pixels = records[key]["image"]
            if np.array_equal(pixels, frame):
                raise ValueError("Labeled image contains no annotation changes")
            report["stage_results"][origin["stage"]] = {"exact_source_header_preserved": True,
                  "detections": records[key]["detections"], "reference_matches": matches,
                  "native_labeled_image_size": [1024, 768], "annotated_pixels_changed": True}
            report["verified_source_pairs"] += 1
            verified.add(key)

        def clear_pair(key):
            if key in clear_verified or set(clear_records.get(key, {})) != {"detections", "image"}:
                return
            if clear_records[key]["detections"] or not clear_records[key]["image"]:
                raise ValueError("Cleared output contains old detections or old camera pixels")
            report["verified_clear_pairs"] += 1
            clear_verified.add(key)

        def receive_detections(message):
            try:
                report["detection_messages"] += 1
                if message.header.frame_id != FRAME:
                    raise ValueError("Detection array optical frame changed")
                key = stamp_key(message.header)
                actual = []
                for detection in message.detections:
                    if stamp_key(detection.header) != key or detection.header.frame_id != FRAME:
                        raise ValueError("Per-detection source header differs from array header")
                    if len(detection.results) != 1 or detection.id:
                        raise ValueError("2D detector must emit one hypothesis and no invented persistent identity")
                    hypothesis = detection.results[0].hypothesis
                    center = detection.bbox.center
                    width, height = float(detection.bbox.size_x), float(detection.bbox.size_y)
                    bounds = [center.position.x-width/2, center.position.y-height/2,
                              center.position.x+width/2, center.position.y+height/2]
                    if (hypothesis.class_id not in ROAD_NAMES or not math.isfinite(hypothesis.score)
                            or not 0.35 <= hypothesis.score <= 1 or not np.isfinite(bounds).all()
                            or width <= 0 or height <= 0 or center.theta != 0
                            or bounds[0] < -1e-6 or bounds[1] < -1e-6
                            or bounds[2] > 1024+1e-6 or bounds[3] > 768+1e-6):
                        raise ValueError("Malformed native detection class, score or pixel box")
                    actual.append({"class_name": hypothesis.class_id, "score": float(hypothesis.score),
                                   "bbox_xyxy": [float(value) for value in bounds]})
                if key in inputs:
                    if not inputs[key]["valid"]:
                        raise ValueError("Rejected input was republished with its invalid source header")
                    record = records.setdefault(key, {})
                    if "detections" in record:
                        raise ValueError("The same camera source header was processed twice")
                    record["detections"] = actual
                    pair(key)
                else:
                    if actual:
                        raise ValueError("Nonempty detections have an unknown source header")
                    age = (node.get_clock().now().nanoseconds-key)/1e9
                    if key <= 0 or not 0 <= age <= 1.0:
                        raise ValueError("Clear detection header is not fresh")
                    clear_records.setdefault(key, {})["detections"] = actual
                    clear_pair(key)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                record_error(error)

        def receive_image(message):
            try:
                report["labeled_image_messages"] += 1
                if (message.header.frame_id != FRAME or message.width != 1024 or message.height != 768
                        or message.encoding != "bgr8" or message.is_bigendian or message.step < 3072
                        or len(message.data) != message.step*message.height):
                    raise ValueError("Labeled image header or native BGR8 layout is invalid")
                key = stamp_key(message.header)
                pixels = np.ndarray((768, 1024, 3), dtype=np.uint8, buffer=bytes(message.data),
                                    strides=(message.step, 3, 1)).copy()
                if key in inputs:
                    if not inputs[key]["valid"]:
                        raise ValueError("Invalid input was republished as a labeled image")
                    record = records.setdefault(key, {})
                    if "image" in record:
                        raise ValueError("The same camera image source was republished")
                    record["image"] = pixels
                    pair(key)
                else:
                    # Clear-image text is permitted, but photographic pixels and
                    # old boxes must be replaced by the gray unavailable canvas.
                    clear_canvas = bool(np.all(pixels[0, 0] == 35) and
                                        np.mean(np.all(pixels == 35, axis=2)) > 0.95)
                    clear_records.setdefault(key, {})["image"] = clear_canvas
                    clear_pair(key)
            except (ValueError, TypeError, KeyError, IndexError) as error:
                record_error(error)

        def receive_diagnostics(message):
            for status in message.status:
                if status.name != "camera_object_detector":
                    continue
                level = status.level[0] if isinstance(status.level, (bytes, bytearray)) else int(status.level)
                snapshot = {"stage": active_stage, "level": level, "message": status.message,
                            "values": {entry.key: entry.value for entry in status.values}}
                diagnostics.append(snapshot)
                report["last_diagnostic"] = snapshot

        subscriptions = [node.create_subscription(Detection2DArray, topics["detections_topic"], receive_detections, reliable),
                         node.create_subscription(Image, topics["labeled_image_topic"], receive_image, latest_image_qos),
                         node.create_subscription(DiagnosticArray, topics["diagnostics_topic"], receive_diagnostics, reliable)]
        discovery_until = time.monotonic()+0.75
        while time.monotonic() < discovery_until:
            rclpy.spin_once(node, timeout_sec=0.02)
        others = [(name, namespace) for name, namespace in node.get_node_names_and_namespaces()
                  if name != "camera_detector_isolated_check"]
        if others:
            raise ValueError(f"Private ROS domain is occupied: {others}")
        arguments = [str(args.executable), "--ros-args", "-r", "__node:=camera_detector_check_cpp",
                     "-r", f"/camera_lidar_bev/diagnostics:={topics['diagnostics_topic']}"]
        parameters = {name: topic for name, topic in topics.items() if name != "diagnostics_topic"}
        parameters.update(engine_file=str(args.engine.resolve()), camera_frame=FRAME,
                          image_width=1024, image_height=768, max_publish_hz=20.0,
                          stale_after_s=0.75, confidence=0.35, nms_iou=0.5, max_detections=100,
                          classes=ROAD_NAMES)
        for name, value in parameters.items():
            text = json.dumps(value) if isinstance(value, list) else str(value)
            arguments += ["-p", f"{name}:={text}"]
        report["node_arguments"] = arguments

        with child_process(arguments, report) as process:
            def spin():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Bounded camera detector verifier deadline exceeded")
                if process.poll() is not None:
                    raise RuntimeError(f"Native camera detector exited with {process.returncode}")
                rclpy.spin_once(node, timeout_sec=0.02)
                if report["errors"]:
                    raise ValueError(report["errors"][0])

            def wait_for(condition, timeout=3.0):
                until = min(deadline, time.monotonic()+timeout)
                while not condition():
                    spin()
                    if time.monotonic() >= until:
                        raise TimeoutError(f"Camera detector stage timed out: {active_stage}")

            wait_for(lambda: publisher.get_subscription_count() > 0, timeout=6.0)

            def publish(stage, mode="valid", reuse=None):
                nonlocal active_stage
                active_stage = stage
                if reuse is not None:
                    publisher.publish(reuse)
                    return stamp_key(reuse.header), reuse
                message = Image()
                key = node.get_clock().now().nanoseconds
                if mode == "stale": key -= 2_000_000_000
                if mode == "future": key += 2_000_000_000
                message.header.stamp.sec, message.header.stamp.nanosec = divmod(key, 1_000_000_000)
                message.header.frame_id = "wrong_optical_frame" if mode == "wrong_frame" else FRAME
                message.width, message.height = (1023 if mode == "wrong_size" else 1024), 768
                message.encoding = "mono8" if mode == "wrong_encoding" else "rgb8" if mode == "rgb" else "bgr8"
                message.is_bigendian = False
                message.step = 3072
                message.data = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes() if mode == "rgb" else frame.tobytes())
                if mode == "wrong_layout": message.data = message.data[:-1]
                inputs[key] = {"stage": stage, "valid": mode in ("valid", "rgb")}
                publisher.publish(message)
                return key, message

            def good(stage, mode="valid"):
                key, message = publish(stage, mode)
                wait_for(lambda: key in verified)
                return key, message

            key, baseline_message = good("native_bgr_reference")
            publish("duplicate_source_header", reuse=baseline_message)
            settle_until = time.monotonic()+0.2
            while time.monotonic() < settle_until:
                spin()
            report["stage_results"]["duplicate_source_header"] = {"original_source_not_republished": True}
            good("native_rgb_reference", "rgb")
            for mode in ("wrong_size", "wrong_frame", "wrong_encoding", "wrong_layout", "stale", "future"):
                before = set(clear_verified)
                key, unused = publish(mode, mode)
                wait_for(lambda: bool(clear_verified-before))
                if key in records:
                    raise ValueError(f"Rejected {mode} source produced outputs")
                report["stage_results"][mode] = {"invalid_source_not_republished": True,
                    "empty_detections_and_cleared_image_with_matching_fresh_header": True}
                good("recovered_after_"+mode)

            # A previously valid source must also clear when input stops entirely.
            active_stage = "source_stopped"
            before = set(clear_verified)
            wait_for(lambda: bool(clear_verified-before), timeout=2.0)
            report["stage_results"]["source_stopped"] = {
                "stale_detections_and_image_cleared_without_new_input": True}
            good("final_recovered")
            active_stage = "final_diagnostics"
            def final_diagnostic(item):
                stamp = int(item["values"].get("source_stamp_ns", "0"))
                return (stamp in inputs and inputs[stamp]["stage"] == "final_recovered"
                        and item["values"].get("outputs_cleared") == "0")

            diagnostic_until = min(deadline, time.monotonic()+1.5)
            next_heartbeat = time.monotonic()+0.15
            while not any(final_diagnostic(item) for item in diagnostics):
                spin()
                if time.monotonic() >= diagnostic_until:
                    raise TimeoutError("Recovered native diagnostics did not arrive")
                if time.monotonic() >= next_heartbeat:
                    # Keep the source fresh while waiting for the 1 Hz diagnostic
                    # timer; its phase is independent of the test stages.
                    publish("final_recovered")
                    next_heartbeat = time.monotonic()+0.15
            final = next(item for item in reversed(diagnostics) if final_diagnostic(item))
            values = final["values"]
            for name in ("decode_ms", "preprocess_ms", "inference_ms", "gpu_compute_ms", "postprocess_ms",
                         "annotate_ms", "processing_ms", "publish_ms", "source_age_at_publish_s"):
                value = float(values.get(name, "nan"))
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Missing or invalid native stage timing {name}")
            if int(values.get("duplicate_frames", "0")) < 1 or int(values.get("inference_errors", "-1")) != 0:
                raise ValueError("Duplicate skip or error counters disagree with successful replay")
            report["verified_diagnostics"] = final
            report["passed"] = True
            # Keep subscriptions alive until the child has stopped.
            _ = subscriptions
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["traceback"] = traceback.format_exc()
        report["passed"] = False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_alarm)
        if node is not None:
            node.destroy_node()
        try:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        report["elapsed_s"] = time.monotonic()-started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.output),
                      "verified_source_pairs": report["verified_source_pairs"],
                      "verified_clear_pairs": report["verified_clear_pairs"], "errors": report["errors"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
