#!/usr/bin/env python3
"""Bounded, read-only verification of the provisionally applied front intrinsics."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from diagnostic_msgs.msg import DiagnosticArray
from sensor_msgs.msg import CameraInfo, Image
import yaml


def stamp(message):
    return int(message.header.stamp.sec), int(message.header.stamp.nanosec)


def values_close(actual, expected):
    return len(actual) == len(expected) and all(
        math.isfinite(float(a)) and math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-9)
        for a, b in zip(actual, expected)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--calibration", type=Path, default=root /
                        "config/calibration/front_camera_640x480_002.candidate.yaml")
    parser.add_argument("--output", type=Path, default=Path("/tmp/front_camera_calibration_check.json"))
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    if not 1 <= args.timeout <= 15:
        parser.error("timeout must be between 1 and 15 seconds")
    calibration_bytes = args.calibration.read_bytes()
    calibration = yaml.safe_load(calibration_bytes)
    expected = {
        "width": int(calibration["image_width"]),
        "height": int(calibration["image_height"]),
        "frame_id": "camera_front_optical_frame",
        "distortion_model": calibration["distortion_model"],
        "k": calibration["camera_matrix"]["data"],
        "d": calibration["distortion_coefficients"]["data"],
        "r": calibration["rectification_matrix"]["data"],
        "p": calibration["projection_matrix"]["data"],
    }
    rclpy.init()
    node = rclpy.create_node("front_camera_provisional_application_check")
    started = time.monotonic()
    report = {
        "calibration_file": str(args.calibration.resolve()),
        "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "calibration_qualification": "candidate_unvalidated",
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "image_messages": 0,
        "camera_info_messages": 0,
        "pre_candidate_camera_info_ignored": 0,
        "matching_candidate_camera_info_messages": 0,
        "matching_image_info_headers": 0,
        "verified_timing_warning_messages": 0,
        "errors": [],
    }
    images = {}
    infos = {}
    candidate_seen = False
    first_candidate_stamp = None
    last_publishers = {}

    def error(text):
        if text not in report["errors"]:
            report["errors"].append(text)

    def match(key):
        if key not in images or key not in infos:
            return
        image = images.pop(key)
        info = infos.pop(key)
        if key == (0, 0) or not 0 <= key[1] < 1_000_000_000:
            error("Matched camera header has an invalid or zero stamp")
        if image["frame_id"] != expected["frame_id"] or info["frame_id"] != expected["frame_id"]:
            error("Matched image or CameraInfo has the wrong optical frame")
        if image["width"] != expected["width"] or image["height"] != expected["height"]:
            error("Matched image size does not equal candidate calibration size")
        if image["encoding"] != "bgr8":
            error("Matched raw image encoding is not bgr8")
        if image["step"] != image["width"] * 3 or image["data_bytes"] != image["step"] * image["height"]:
            error("Matched raw image stride or byte count is inconsistent with packed bgr8")
        report["matching_image_info_headers"] += 1
        report["sample_image"] = image
        report["sample_camera_info"] = info

    def receive_image(message):
        report["image_messages"] += 1
        key = stamp(message)
        images[key] = {
            "stamp_sec": key[0], "stamp_nanosec": key[1],
            "frame_id": message.header.frame_id,
            "width": int(message.width), "height": int(message.height),
            "encoding": message.encoding, "step": int(message.step),
            "data_bytes": len(message.data),
        }
        match(key)
        while len(images) > 200:
            images.pop(next(iter(images)))

    def receive_info(message):
        nonlocal candidate_seen, first_candidate_stamp
        report["camera_info_messages"] += 1
        arrays = {name: list(getattr(message, name)) for name in ("k", "d", "r", "p")}
        has_candidate = (
            message.width == expected["width"] and message.height == expected["height"]
            and message.distortion_model == expected["distortion_model"]
            and all(values_close(arrays[name], expected[name]) for name in arrays)
        )
        if not has_candidate:
            if candidate_seen:
                error("CameraInfo changed or disagrees with candidate after candidate was observed")
            else:
                report["pre_candidate_camera_info_ignored"] += 1
            return
        candidate_seen = True
        key = stamp(message)
        if first_candidate_stamp is None:
            first_candidate_stamp = key
        report["matching_candidate_camera_info_messages"] += 1
        infos[key] = {
            "stamp_sec": key[0], "stamp_nanosec": key[1],
            "frame_id": message.header.frame_id,
            "width": int(message.width), "height": int(message.height),
            "distortion_model": message.distortion_model,
            **arrays,
        }
        match(key)
        while len(infos) > 200:
            infos.pop(next(iter(infos)))

    def receive_diagnostics(message):
        if not candidate_seen or stamp(message) < first_candidate_stamp:
            return
        for status in message.status:
            if status.name != "camera_lidar_bev/front_camera_bridge":
                continue
            level = (int.from_bytes(status.level, "little")
                     if isinstance(status.level, (bytes, bytearray)) else int(status.level))
            current = {item.key: item.value for item in status.values}
            report["last_camera_diagnostic"] = {
                "level": level, "message": status.message, "values": current,
            }
            if level >= 2:
                error("Camera diagnostics ERROR after applied candidate: " + status.message)
                continue
            required = {
                "calibrated": "true",
                "timestamp_source": "host_receipt_after_read",
                "exposure_timestamps_available": "false",
                "clock_domain": "host_ros_system_time",
                "synchronization_validated": "false",
                "frame_id": expected["frame_id"],
                "actual_width": str(expected["width"]),
                "actual_height": str(expected["height"]),
            }
            mismatch = [name for name, value in required.items() if current.get(name) != value]
            if level != 1:
                error("Camera diagnostic no longer retains WARN for unverified sensor timing")
            if mismatch:
                error("Camera diagnostic fields disagree: " + ", ".join(mismatch))
            if level == 1 and not mismatch:
                report["verified_timing_warning_messages"] += 1

    subscriptions = [
        node.create_subscription(Image, "/camera/front/image_raw", receive_image, qos_profile_sensor_data),
        node.create_subscription(CameraInfo, "/camera/front/camera_info", receive_info, qos_profile_sensor_data),
        node.create_subscription(DiagnosticArray, "/camera_lidar_bev/diagnostics", receive_diagnostics, qos_profile_sensor_data),
    ]
    try:
        while rclpy.ok() and time.monotonic() - started < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - started >= 2.0:
                last_publishers = {
                    "/camera/front/image_raw": node.count_publishers("/camera/front/image_raw"),
                    "/camera/front/camera_info": node.count_publishers("/camera/front/camera_info"),
                }
                if report["matching_image_info_headers"] >= 10 and report["verified_timing_warning_messages"] >= 2:
                    break
        report["elapsed_s"] = round(time.monotonic() - started, 6)
        report["publisher_counts"] = last_publishers
        if report["matching_image_info_headers"] < 10:
            error("Fewer than ten candidate CameraInfo/image matching headers received within timeout")
        if report["verified_timing_warning_messages"] < 1:
            error("No calibrated diagnostic with retained unverified-timing WARN received")
        if any(last_publishers.get(topic) != 1 for topic in ("/camera/front/image_raw", "/camera/front/camera_info")):
            error("Image and CameraInfo must each have exactly one publisher")
        report["checks_passed"] = not report["errors"]
        report["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({
            "checks_passed": report["checks_passed"],
            "paired_headers": report["matching_image_info_headers"],
            "candidate_camera_info_messages": report["matching_candidate_camera_info_messages"],
            "timing_warn_messages": report["verified_timing_warning_messages"],
            "publisher_counts": last_publishers,
            "errors": report["errors"],
            "evidence_file": str(args.output),
        }, indent=2), flush=True)
        return 0 if report["checks_passed"] else 2
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
