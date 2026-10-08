#!/usr/bin/python3
"""Observe ROS sensor inputs for a bounded interval without changing drivers.

Source ROS and the workspace first, then run with /usr/bin/python3. Header age
is only a clock comparison; it does not establish synchronization or latency.
PointCloud2 payloads are sampled in place and are never included in the report.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import struct
import time


EXPECTED_TOPICS = {
    "/points": "sensor_msgs/msg/PointCloud2",
    "/camera/front/image_raw": "sensor_msgs/msg/Image",
    "/camera/front/camera_info": "sensor_msgs/msg/CameraInfo",
    "/sensing/imu/imu_data": "sensor_msgs/msg/Imu",
    "/imu": "sensor_msgs/msg/Imu",
    "/lio_sam/mapping/odometry": "nav_msgs/msg/Odometry",
}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def point_cloud_summary(message, max_samples=4096):
    """Describe the layout and sample UINT32 t, respecting rows and byte order."""
    width, height = int(message.width), int(message.height)
    point_step, row_step = int(message.point_step), int(message.row_step)
    payload_bytes = len(message.data)
    fields = [
        {"name": field.name, "offset": int(field.offset),
         "datatype": int(field.datatype), "count": int(field.count)}
        for field in message.fields
    ]
    errors = []
    if width <= 0 or height <= 0:
        errors.append("empty_dimensions")
    if point_step <= 0:
        errors.append("nonpositive_point_step")
    if row_step < width * point_step:
        errors.append("row_step_smaller_than_row")
    if payload_bytes < row_step * height:
        errors.append("truncated_payload")
    scalar_bytes = {1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 8}
    names = set()
    for field in fields:
        if field["name"] in names:
            errors.append("duplicate_field:" + field["name"])
        names.add(field["name"])
        size = scalar_bytes.get(field["datatype"])
        if size is None or field["count"] <= 0:
            errors.append("invalid_field_type_or_count:" + field["name"])
        elif field["offset"] < 0 or field["offset"] + size * field["count"] > point_step:
            errors.append("field_outside_point:" + field["name"])
    timing_fields = [field for field in fields if field["name"] in ("t", "time", "timestamp", "ts")]
    t_fields = [field for field in fields if field["name"] == "t"]
    t_valid = bool(len(t_fields) == 1 and t_fields[0]["datatype"] == 6
                   and t_fields[0]["count"] == 1
                   and 0 <= t_fields[0]["offset"]
                   and t_fields[0]["offset"] + 4 <= point_step)
    timing = {
        "field_present": bool(timing_fields),
        "candidate_fields": timing_fields,
        "t_present": bool(t_fields),
        "t_uint32_layout_valid": t_valid,
        "expected_ouster_original_t_datatype": "UINT32 (6)",
        "units_and_reference": "unverified; confirm nanoseconds and relative reference from driver profile",
        "deskew_ready": False,
    }
    result = {
        "width": width, "height": height, "point_count": width * height,
        "point_step": point_step, "row_step": row_step,
        "payload_bytes": payload_bytes, "is_bigendian": bool(message.is_bigendian),
        "fields": fields, "layout_valid": not errors, "layout_errors": errors,
        "point_time": timing,
    }
    if errors or not t_valid or max_samples < 1:
        return result
    count = width * height
    samples = min(count, max_samples)
    values = []
    unpacker = struct.Struct(">I" if message.is_bigendian else "<I")
    data = memoryview(message.data)
    for sample in range(samples):
        index = sample * (count - 1) // (samples - 1) if samples > 1 else 0
        row, column = divmod(index, width)
        offset = row * row_step + column * point_step + t_fields[0]["offset"]
        values.append(unpacker.unpack_from(data, offset)[0])
    timing.update({
        "sample_count": samples, "sample_limit": max_samples,
        "sample_min_raw": min(values), "sample_max_raw": max(values),
        "sample_span_raw": max(values) - min(values),
        "sample_zero_count": values.count(0),
        "sampling": "evenly spaced point indices; first/last included when sample count permits",
    })
    return result


def message_summary(message, type_name, max_samples):
    if type_name == "sensor_msgs/msg/PointCloud2":
        return point_cloud_summary(message, max_samples)
    if type_name == "sensor_msgs/msg/CameraInfo":
        return {
            "width": int(message.width), "height": int(message.height),
            "distortion_model": message.distortion_model,
            "intrinsics_present": bool(math.isfinite(float(message.k[0])) and message.k[0] > 0),
            "intrinsics_validation": "K[0] presence only; calibration accuracy unverified",
        }
    if type_name == "sensor_msgs/msg/Image":
        return {
            "width": int(message.width), "height": int(message.height),
            "encoding": message.encoding, "step": int(message.step),
            "payload_bytes": len(message.data),
        }
    if type_name == "sensor_msgs/msg/Imu":
        return {
            "orientation_marked_available": bool(message.orientation_covariance[0] != -1),
            "angular_velocity_marked_available": bool(message.angular_velocity_covariance[0] != -1),
            "linear_acceleration_marked_available": bool(message.linear_acceleration_covariance[0] != -1),
            "trajectory_note": "IMU measurements alone are not an odometry trajectory for deskewing",
        }
    if type_name == "nav_msgs/msg/Odometry":
        return {"child_frame_id": message.child_frame_id,
                "trajectory_validation": "pose/twist accuracy and frame transforms unverified"}
    return {"trajectory_validation": "PoseStamped has no velocity; suitability unverified"}


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def point_cloud_reliability(requested, publisher_information):
    """Choose reliable only when every advertised publisher supports it."""
    if requested != "auto":
        return requested
    if publisher_information and all(
            publisher["qos"]["reliability"] == "RELIABLE"
            for publisher in publisher_information):
        return "reliable"
    return "best_effort"


def qos_summary(profile):
    return {
        "reliability": profile.reliability.name,
        "durability": profile.durability.name,
        "history": profile.history.name,
        "depth": int(profile.depth),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=positive_float, default=8.0,
                        help="observation duration in seconds after discovery (default: 8)")
    parser.add_argument("--discovery-duration", type=positive_float, default=2.0,
                        help="initial discovery duration in seconds (default: 2)")
    parser.add_argument("--stale-after", type=positive_float, default=2.0,
                        help="seconds without receipt before a topic is stale (default: 2)")
    parser.add_argument("--max-point-samples", type=int, default=4096,
                        help="bounded t-field samples per cloud (1..4096)")
    parser.add_argument("--include-pose", action="store_true", help="also discover PoseStamped topics")
    parser.add_argument("--point-cloud-reliability", choices=("auto", "reliable", "best_effort"),
                        default="auto", help="PointCloud2 QoS; auto selects reliable when all publishers are reliable")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()
    if not 1 <= args.max_point_samples <= 4096:
        parser.error("--max-point-samples must be between 1 and 4096")

    # Delay imports so --help and dependency-free layout checks need no ROS.
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2

    supported_types = {
        "sensor_msgs/msg/PointCloud2": PointCloud2,
        "sensor_msgs/msg/Image": Image,
        "sensor_msgs/msg/CameraInfo": CameraInfo,
        "sensor_msgs/msg/Imu": Imu,
        "nav_msgs/msg/Odometry": Odometry,
    }
    if args.include_pose:
        from geometry_msgs.msg import PoseStamped
        supported_types["geometry_msgs/msg/PoseStamped"] = PoseStamped

    rclpy.init(args=[])
    node = None
    report = None
    try:
        node = rclpy.create_node("bev_readonly_input_inventory", enable_rosout=False,
                                start_parameter_services=False)
        start_utc = utc_now()
        clocks_start = {"system_time_ns": time.time_ns(), "ros_time_ns": node.get_clock().now().nanoseconds}
        deadline = time.monotonic() + args.discovery_duration
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, deadline - time.monotonic())))
        graph = dict(node.get_topic_names_and_types())
        sensor_topics = {topic: types for topic, types in sorted(graph.items())
                         if any(type_name in supported_types for type_name in types)}
        statistics = {}
        subscriptions = []
        observation_start = time.monotonic()

        def make_callback(topic, type_name):
            def receive(message):
                state = statistics[topic]
                receipt = time.monotonic()
                state.setdefault("first_receipt_monotonic", receipt)
                state["last_receipt_monotonic"] = receipt
                state["count"] += 1
                frame = message.header.frame_id
                if frame not in state["frames"] and len(state["frames"]) < 16:
                    state["frames"].append(frame)
                state["last_frame_id"] = frame
                sec, nanos = int(message.header.stamp.sec), int(message.header.stamp.nanosec)
                stamp = sec * 1_000_000_000 + nanos
                valid = sec >= 0 and 0 <= nanos < 1_000_000_000 and stamp > 0
                state["last_header_time_ns"] = stamp
                if not valid:
                    state["invalid_or_zero_header_count"] += 1
                else:
                    previous = state.get("previous_valid_header_ns")
                    if previous is not None:
                        state["header_regression_count"] += int(stamp < previous)
                        state["repeated_header_count"] += int(stamp == previous)
                    state["previous_valid_header_ns"] = stamp
                    age = (node.get_clock().now().nanoseconds - stamp) / 1e9
                    state["last_header_age_at_receipt_s_assuming_common_clock"] = round(age, 6)
                    state["min_header_age_at_receipt_s_assuming_common_clock"] = min(
                        state.get("min_header_age_at_receipt_s_assuming_common_clock", age), age)
                    state["max_header_age_at_receipt_s_assuming_common_clock"] = max(
                        state.get("max_header_age_at_receipt_s_assuming_common_clock", age), age)
                state["latest_message"] = message_summary(message, type_name, args.max_point_samples)
            return receive

        for topic, type_names in sensor_topics.items():
            matches = [type_name for type_name in type_names if type_name in supported_types]
            publishers = [
                {"node_name": endpoint.node_name,
                 "node_namespace": endpoint.node_namespace,
                 "topic_type": endpoint.topic_type,
                 "qos": qos_summary(endpoint.qos_profile)}
                for endpoint in node.get_publishers_info_by_topic(topic)
            ]
            statistics[topic] = {
                "types": type_names, "count": 0, "frames": [],
                "header_regression_count": 0, "repeated_header_count": 0,
                "invalid_or_zero_header_count": 0,
                "publishers_at_subscription": publishers,
            }
            if len(type_names) != 1 or len(matches) != 1:
                statistics[topic]["skip_reason"] = "ambiguous_topic_message_types"
                continue
            type_name = matches[0]
            subscription_qos = qos_profile_sensor_data
            if type_name == "sensor_msgs/msg/PointCloud2":
                selected_reliability = point_cloud_reliability(args.point_cloud_reliability, publishers)
                subscription_qos = QoSProfile(
                    history=HistoryPolicy.KEEP_LAST, depth=1,
                    reliability=(ReliabilityPolicy.RELIABLE if selected_reliability == "reliable"
                                 else ReliabilityPolicy.BEST_EFFORT),
                    durability=DurabilityPolicy.VOLATILE)
                statistics[topic]["point_cloud_reliability_requested"] = args.point_cloud_reliability
            statistics[topic]["subscription_qos"] = qos_summary(subscription_qos)
            subscriptions.append(node.create_subscription(
                supported_types[type_name], topic, make_callback(topic, type_name), subscription_qos))
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))
        observation_end = time.monotonic()
        clocks_end = {"system_time_ns": time.time_ns(), "ros_time_ns": node.get_clock().now().nanoseconds}
        for state in statistics.values():
            first = state.pop("first_receipt_monotonic", None)
            last = state.pop("last_receipt_monotonic", None)
            state.pop("previous_valid_header_ns", None)
            interval = last - first if last is not None else 0.0
            state["receipt_rate_hz"] = round((state["count"] - 1) / interval, 3) if interval > 0 else None
            state["receipt_rate_note"] = "callback receipts during sample; not verified sensor capture rate"
            state["seconds_since_last_receipt"] = round(observation_end - last, 6) if last is not None else None
            state["status"] = "missing" if last is None else (
                "stale" if observation_end - last > args.stale_after else "observed")
            if last is None:
                state["status_reason"] = state.get("skip_reason", "no_messages_received")
            if state.get("last_header_time_ns", 0) > 0:
                state["last_header_age_at_end_s_assuming_common_clock"] = round(
                    (clocks_end["ros_time_ns"] - state["last_header_time_ns"]) / 1e9, 6)
        expected = {}
        for topic, type_name in EXPECTED_TOPICS.items():
            state = statistics.get(topic)
            expected[topic] = {"expected_type": type_name,
                               "status": state["status"] if state else "missing"}
            if state is None:
                expected[topic]["reason"] = "topic_not_discovered_with_supported_sensor_type"
            elif type_name not in state["types"]:
                expected[topic].update(status="missing", reason="unexpected_message_type")
            elif state["status"] != "observed":
                expected[topic]["reason"] = state.get("status_reason", "no_recent_receipt")
        report = {
            "schema_version": 1, "started_utc": start_utc, "finished_utc": utc_now(),
            "ros_domain_id": int(os.environ.get("ROS_DOMAIN_ID", "0")),
            "environment": {key: os.environ.get(key) for key in
                            ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "ROS_AUTOMATIC_DISCOVERY_RANGE", "ROS_LOCALHOST_ONLY")},
            "clock_start": clocks_start, "clock_end": clocks_end,
            "use_sim_time": bool(node.get_parameter("use_sim_time").value),
            "discovery_duration_s": args.discovery_duration,
            "observation_duration_s": round(observation_end - observation_start, 6),
            "stale_after_s": args.stale_after,
            "point_cloud_reliability_requested": args.point_cloud_reliability,
            "clock_caveat": "Header ages assume a common epoch and synchronized clock; receipt time is not exposure/scan time.",
            "motion_caveat": "Ouster IMU or other raw IMU is not equivalent to an odometry trajectory.",
            "expected_topics": expected,
            "observed_topics": [topic for topic, state in statistics.items() if state["count"] > 0],
            "discovered_sensor_topics": sensor_topics,
            "topics": statistics,
        }
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
