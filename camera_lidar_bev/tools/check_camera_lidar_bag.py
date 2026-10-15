#!/usr/bin/env python3
"""Validate a saved raw camera/Ouster bag entirely offline; never play or record it.

MCAP/sqlite bags are scanned once. Large Image/PointCloud2 CDR headers and layouts
are checked without copying/deserializing their pixel/point arrays. Header timing
and nominal cadence gaps are recording observations, not exposure synchronization
or proof that the sensor dropped frames. Reports are created exclusively.
"""

import argparse
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import re
from pathlib import Path
import statistics
import struct
import time


TOPICS = {"image": "/camera/front/image_raw", "info": "/camera/front/camera_info",
          "lidar": "/points", "imu": "/imu", "metadata": "/metadata", "tf_static": "/tf_static"}
TYPES = {"image": "sensor_msgs/msg/Image", "info": "sensor_msgs/msg/CameraInfo",
         "lidar": "sensor_msgs/msg/PointCloud2", "imu": "sensor_msgs/msg/Imu",
         "metadata": "std_msgs/msg/String", "tf_static": "tf2_msgs/msg/TFMessage"}


def recorder_cache_losses(text):
    """rosbag2 0.26 retains these counters and reprints them at every split.

    Repeated cumulative reports must never be summed as separate losses.
    This is recorder-cache evidence, not a sensor or DDS loss measurement.
    """
    blocks, current = [], None
    for line in text.splitlines():
        if "Cache buffers lost messages per topic:" in line:
            current = {}
        elif current is not None:
            topic = re.fullmatch(r"\s*(/[^\s:]+):\s*(\d+)\s*", line)
            total = re.fullmatch(r"\s*Total lost:\s*(\d+)\s*", line)
            if topic:
                current[topic.group(1)] = int(topic.group(2))
            elif total:
                blocks.append((current, int(total.group(1))))
                current = None
    per_topic = {}
    for values, _ in blocks:
        for topic, count in values.items():
            per_topic[topic] = max(per_topic.get(topic, 0), count)
    maximum = max((total for _, total in blocks), default=0)
    return {"source": "rosbag2_recorder_cache", "cumulative_reports_seen": len(blocks),
            "reported_total_max": maximum, "per_topic_cumulative_max": per_topic,
            "has_confirmed_recorder_cache_loss": maximum > 0 or any(per_topic.values()),
            "is_sensor_drop_count": False, "repeated_reports_summed": False}


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low]+(ordered[high]-ordered[low])*(position-low)


def summarize_timing(headers, recorded, expected_hz=None):
    """Describe saved stream ordering and cadence; never infer actual lost frames."""
    intervals = [(b-a)/1e9 for a, b in zip(headers, headers[1:])]
    positive = [value for value in intervals if value > 0]
    span = (headers[-1]-headers[0])/1e9 if len(headers) > 1 else 0
    record_span = (recorded[-1]-recorded[0])/1e9 if len(recorded) > 1 else 0
    period = 1/expected_hz if expected_hz else None
    gaps = [value for value in positive if period and value > 1.5*period]
    return {"saved_messages": len(headers), "distinct_header_stamps": len(set(headers)),
            "nonpositive_header_stamps": sum(value <= 0 for value in headers),
            "backward_header_steps": sum(value < 0 for value in intervals),
            "duplicate_adjacent_header_stamps": sum(value == 0 for value in intervals),
            "header_span_s": span, "header_rate_hz": (len(headers)-1)/span if span > 0 else None,
            "recorded_timestamp_span_s": record_span,
            "saved_receipt_rate_hz": (len(recorded)-1)/record_span if record_span > 0 else None,
            "positive_header_interval_s": {"median": statistics.median(positive) if positive else None,
                                            "p95": percentile(positive, .95), "max": max(positive) if positive else None},
            "configured_expected_hz": expected_hz, "gaps_above_1_5_nominal_period": len(gaps),
            "nominal_unfilled_slots_estimate": sum(max(0, round(value/period)-1) for value in gaps) if period else None,
            "gap_slots_are_confirmed_drops": False}


def pair_camera_headers(images, infos):
    """Pair complete (frame_id, stamp_ns) identities, including multiplicities."""
    image_counts, info_counts = Counter(images), Counter(infos)
    matched = sum(min(count, info_counts[key]) for key, count in image_counts.items())
    return {"images": len(images), "camera_infos": len(infos), "exact_header_pairs": matched,
            "unpaired_images": len(images)-matched, "orphan_camera_infos": len(infos)-matched,
            "all_images_have_exact_camera_info": bool(images) and matched == len(images)}


def nearest_header_offsets(camera, lidar):
    """Camera minus nearest LiDAR header; this is explicitly not physical sync."""
    ordered = sorted(set(lidar))
    differences = []
    if ordered:
        for stamp in camera:
            index = bisect_left(ordered, stamp)
            candidates = ordered[max(0, index-1):min(len(ordered), index+1)]
            nearest = min(candidates, key=lambda value: (abs(value-stamp), value))
            differences.append((stamp-nearest)/1e9)
    absolute = [abs(value) for value in differences]
    return {"samples": len(differences), "camera_header_minus_nearest_lidar_s": {
                "median": statistics.median(differences) if differences else None,
                "min": min(differences) if differences else None, "max": max(differences) if differences else None},
            "absolute_header_delta_s": {"median": statistics.median(absolute) if absolute else None,
                                        "p95": percentile(absolute, .95), "max": max(absolute) if absolute else None},
            "physical_synchronization_validated": False}


class Cdr:
    """Bounded CDR1 reader for standard ROS headers and large payload layouts."""
    def __init__(self, serialized):
        self.data = memoryview(serialized)
        if len(self.data) < 4:
            raise ValueError("Truncated CDR encapsulation")
        kind = struct.unpack_from(">H", self.data, 0)[0]
        if kind not in (0, 1):
            raise ValueError(f"Unsupported CDR encapsulation {kind}; expected CDR1")
        self.endian = "<" if kind == 1 else ">"
        self.position = 4

    def scalar(self, code):
        size = struct.calcsize(code)
        self.position += (-(self.position-4)) % size
        if self.position+size > len(self.data):
            raise ValueError("Truncated CDR scalar")
        value = struct.unpack_from(self.endian+code, self.data, self.position)[0]
        self.position += size
        return value

    def string(self, limit=4096):
        length = self.scalar("I")
        if not 1 <= length <= limit or self.position+length > len(self.data):
            raise ValueError("Invalid or truncated CDR string")
        if self.data[self.position+length-1] != 0:
            raise ValueError("CDR string lacks terminator")
        value = bytes(self.data[self.position:self.position+length-1]).decode("utf-8")
        self.position += length
        return value

    def header(self):
        seconds, nanoseconds = self.scalar("i"), self.scalar("I")
        frame = self.string()
        if nanoseconds >= 1_000_000_000:
            raise ValueError("Invalid ROS nanosecond field")
        return frame, seconds*1_000_000_000+nanoseconds

    def payload(self):
        length = self.scalar("I")
        start = self.position
        if start+length > len(self.data):
            raise ValueError("Truncated CDR payload")
        self.position += length
        return start, length


def image_layout(serialized, width=1024, height=768):
    reader = Cdr(serialized)
    frame, stamp = reader.header()
    rows, columns, encoding = reader.scalar("I"), reader.scalar("I"), reader.string()
    endian, step = reader.scalar("B"), reader.scalar("I")
    _, size = reader.payload()
    if columns != width or rows != height or encoding not in ("bgr8", "rgb8") or endian not in (0, 1):
        raise ValueError("Image does not have expected native dimensions/encoding")
    if step < columns*3 or size != rows*step:
        raise ValueError("Image payload does not match native row layout")
    return frame, stamp, {"width": columns, "height": rows, "encoding": encoding, "payload_bytes": size}


def lidar_layout(serialized):
    reader = Cdr(serialized)
    frame, stamp = reader.header()
    height, width, count = reader.scalar("I"), reader.scalar("I"), reader.scalar("I")
    if not 1 <= count <= 256:
        raise ValueError("Invalid PointCloud2 field count")
    fields = {}
    for _ in range(count):
        name = reader.string()
        field = {"offset": reader.scalar("I"), "datatype": reader.scalar("B"), "count": reader.scalar("I")}
        if name in fields:
            raise ValueError("Duplicate PointCloud2 field")
        fields[name] = field
    endian, point_step, row_step = reader.scalar("B"), reader.scalar("I"), reader.scalar("I")
    start, size = reader.payload()
    dense = reader.scalar("B")
    if not width or not height or endian not in (0, 1) or dense not in (0, 1) or not point_step:
        raise ValueError("Invalid original PointCloud2 dimensions/layout")
    if row_step < width*point_step or size != row_step*height:
        raise ValueError("PointCloud2 payload size does not match rows")
    required = {"x": (7, 4), "y": (7, 4), "z": (7, 4), "t": (6, 4), "reflectivity": (4, 2)}
    spans = []
    for name, (datatype, field_size) in required.items():
        field = fields.get(name)
        if field is None or field["datatype"] != datatype or field["count"] != 1 or field["offset"]+field_size > point_step:
            raise ValueError(f"Original LiDAR field missing or invalid: {name}")
        spans.append((field["offset"], field["offset"]+field_size))
    if any(a < d and c < b for i, (a, b) in enumerate(spans) for c, d in spans[i+1:]):
        raise ValueError("Original required LiDAR fields overlap")
    times = []
    for index in sorted({0, width*height//2, width*height-1}):
        row, column = divmod(index, width)
        offset = start+row*row_step+column*point_step+fields["t"]["offset"]
        times.append(struct.unpack_from((">" if endian else "<")+"I", reader.data, offset)[0])
    return frame, stamp, {"width": width, "height": height, "point_step": point_step,
                          "fields": fields, "sample_point_t_uint32": times, "payload_bytes": size}


def camera_info_matches(info, calibration, width=1024, height=768):
    """Exact calibrated native schema, permitting only insignificant serialization noise."""
    if info.width != width or info.height != height or info.distortion_model != "plumb_bob":
        return False
    arrays = ((info.k, calibration["camera_matrix"]["data"]),
              (info.d, calibration["distortion_coefficients"]["data"]),
              (info.r, calibration["rectification_matrix"]["data"]),
              (info.p, calibration["projection_matrix"]["data"]))
    if any(len(actual) != len(expected) or any(not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > 1e-8
                                              for a, b in zip(actual, expected)) for actual, expected in arrays):
        return False
    return (info.binning_x <= 1 and info.binning_y <= 1 and not info.roi.x_offset and not info.roi.y_offset and
            not info.roi.do_rectify and info.roi.width in (0, width) and info.roi.height in (0, height))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="saved MCAP/sqlite bag directory with metadata.yaml")
    parser.add_argument("--output", type=Path, help="new JSON path; default <bag>.qc.json beside the bag")
    parser.add_argument("--calibration", type=Path,
                        default=root/"config/calibration/imported_front_001/front_camera_1024x768.candidate.yaml")
    parser.add_argument("--camera-hz", type=float, default=30.0)
    parser.add_argument("--lidar-hz", type=float, default=20.0)
    parser.add_argument("--max-read-gib", type=float, default=80.0, help="refuse larger storage/payload scans (default80GiB)")
    parser.add_argument("--timeout", type=float, default=180.0, help="bounded offline scan seconds (default180)")
    for name, topic in TOPICS.items():
        parser.add_argument("--"+name.replace("_", "-")+"-topic", default=topic)
    args = parser.parse_args()
    bag = args.bag.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else bag.parent/(bag.name+".qc.json")
    if not (bag/"metadata.yaml").is_file() or not args.calibration.is_file():
        parser.error("bag metadata.yaml and the native calibration YAML must exist")
    if output.exists() or output.is_symlink():
        parser.error(f"refusing to overwrite report: {output}")
    if any(not math.isfinite(value) or value <= 0 for value in (args.camera_hz, args.lidar_hz, args.max_read_gib, args.timeout)):
        parser.error("rates, read size and timeout must be finite and positive")
    topics = {name: getattr(args, name+"_topic") for name in TOPICS}
    if len(set(topics.values())) != len(topics) or any(not topic.startswith("/") for topic in topics.values()):
        parser.error("required topics must be distinct absolute names")
    import yaml
    metadata_yaml = yaml.safe_load((bag/"metadata.yaml").read_text())["rosbag2_bagfile_information"]
    storage = metadata_yaml["storage_identifier"]
    if storage not in ("mcap", "sqlite3"):
        parser.error("only MCAP and sqlite3 recordings are supported")
    file_bytes = 0
    for relative in metadata_yaml["relative_file_paths"]:
        file = (bag/relative).resolve()
        if not file.is_relative_to(bag) or not file.is_file():
            parser.error("metadata references an absent file or a path outside the bag")
        file_bytes += file.stat().st_size
    if file_bytes > args.max_read_gib*1024**3:
        parser.error("bag exceeds --max-read-gib; refusing an unexpectedly large scan")
    calibration = yaml.safe_load(args.calibration.read_text())
    if calibration["image_width"] != 1024 or calibration["image_height"] != 768:
        parser.error("this native capture check requires 1024x768 calibration")

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CameraInfo, Imu
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage
    classes = {"info": CameraInfo, "imu": Imu, "metadata": String, "tf_static": TFMessage}
    metadata = rosbag2_py.Info().read_metadata(str(bag), storage)
    stored = {item.topic_metadata.name: {"count": item.message_count, "type": item.topic_metadata.type,
                                       "serialization_format": item.topic_metadata.serialization_format}
              for item in metadata.topics_with_message_count}
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "bag": str(bag), "offline_only": True,
              "passed": False, "scan_complete": False, "storage": storage, "storage_bytes": file_bytes,
              "bag_metadata_message_count_all_topics": metadata.message_count, "calibration": str(args.calibration),
              "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
              "physical_accuracy_validated": False, "sensor_exposure_synchronization_validated": False,
              "point_t_units_and_clock_origin_validated": False, "issues": [], "warnings": [],
              "limitations": ["Only saved messages are counted; source-emitted and recorder-dropped totals are unknown.",
                              "Header and bag-recording timestamps do not establish simultaneous sensor acquisition.",
                              "Nominal cadence gaps may reflect exposure/rate changes, transport or recording loss; they are not confirmed drops.",
                              "Large payload layout is checked for every message; no pixel content or complete point-value accuracy is inferred.",
                              "Camera calibration/extrinsic accuracy and odometry are not validated by this check."]}
    state = {name: {"headers": [], "recorded": [], "frames": Counter(), "keys": [], "bad": 0} for name in TOPICS}
    reader = None
    began = time.monotonic()
    bytes_read = 0
    tf_good = tf_conflicting = 0
    metadata_valid = 0
    try:
        for name, topic in topics.items():
            entry = stored.get(topic)
            if not entry or not entry["count"]:
                report["issues"].append(f"Missing nonempty required stream: {topic}")
            elif entry["type"] != TYPES[name] or entry["serialization_format"] != "cdr":
                raise ValueError(f"Wrong raw stream type/serialization: {topic}")
        compression = metadata.compression_mode
        reader = rosbag2_py.SequentialCompressionReader() if compression else rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=storage), rosbag2_py.ConverterOptions("cdr", "cdr"))
        report["received_timestamp_read_order_supported"] = reader.set_read_order(
            rosbag2_py.ReadOrder(rosbag2_py.ReadOrderSortBy.ReceivedTimestamp))
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics.values())))
        names = {topic: name for name, topic in topics.items()}
        while reader.has_next():
            if time.monotonic()-began > args.timeout:
                raise TimeoutError("Offline scan exceeded --timeout; counts are partial")
            topic, serialized, recorded = reader.read_next()
            bytes_read += len(serialized)
            if bytes_read > args.max_read_gib*1024**3:
                raise ValueError("Serialized payload scan exceeded --max-read-gib")
            name, entry = names[topic], state[names[topic]]
            entry["recorded"].append(int(recorded))
            try:
                if name == "image":
                    frame, stamp, layout = image_layout(serialized)
                    entry["last_layout"] = layout
                elif name == "lidar":
                    frame, stamp, layout = lidar_layout(serialized)
                    entry["last_layout"] = layout
                else:
                    message = deserialize_message(serialized, classes[name])
                    if name in ("info", "imu"):
                        frame = message.header.frame_id
                        stamp = message.header.stamp.sec*1_000_000_000+message.header.stamp.nanosec
                        if message.header.stamp.nanosec >= 1_000_000_000:
                            raise ValueError("Invalid header nanosecond value")
                        if name == "info" and not camera_info_matches(message, calibration):
                            raise ValueError("CameraInfo is not stable native calibrated K/D/R/P")
                        if name == "imu":
                            values = [getattr(vector, axis) for vector in (message.angular_velocity, message.linear_acceleration)
                                      for axis in ("x", "y", "z")]
                            if not all(math.isfinite(value) for value in values):
                                raise ValueError("IMU acceleration/rates are nonfinite")
                    elif name == "metadata":
                        value = json.loads(message.data)
                        if not isinstance(value, dict) or not value:
                            raise ValueError("Ouster metadata must contain a nonempty JSON object")
                        metadata_valid += 1
                        entry["metadata_top_level_keys"] = sorted(value)[:40]
                        continue
                    else:
                        for transform in message.transforms:
                            if transform.header.frame_id != "os_sensor" or transform.child_frame_id != "os_lidar":
                                continue
                            p, q = transform.transform.translation, transform.transform.rotation
                            values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
                            expected = (0, 0, .038195, 0, 0, 1, 0)
                            actual = (p.x, p.y, p.z, q.x, q.y, abs(q.z), q.w)
                            if all(math.isfinite(x) and abs(x-y) <= 1e-6 for x, y in zip(actual, expected)):
                                tf_good += 1
                            else:
                                tf_conflicting += 1
                        continue
                entry["headers"].append(int(stamp))
                entry["frames"][frame] += 1
                entry["keys"].append((frame, int(stamp)))
            except (ValueError, UnicodeError, struct.error) as error:
                entry["bad"] += 1
                if entry["bad"] <= 3:
                    report["issues"].append(f"{topic}: {error}")
        report["scan_complete"] = True
        expected_frames = {"image": "camera_front_optical_frame", "info": "camera_front_optical_frame",
                           "lidar": "os_lidar", "imu": "os_imu"}
        for name, frame in expected_frames.items():
            entry = state[name]
            if set(entry["frames"]) != {frame}:
                report["issues"].append(f"Unexpected/missing source frame for {topics[name]}: {dict(entry['frames'])}")
        pairing = pair_camera_headers(state["image"]["keys"], state["info"]["keys"])
        report["camera_info_pairing"] = pairing
        if not pairing["all_images_have_exact_camera_info"]:
            report["issues"].append("Saved images are missing exactly paired valid native CameraInfo")
        if pairing["orphan_camera_infos"]:
            report["warnings"].append("Some CameraInfo messages have no saved image with the exact same header")
        if not tf_good or tf_conflicting:
            report["issues"].append("Actual driver os_sensor<-os_lidar TF missing or conflicting")
        if not metadata_valid:
            report["issues"].append("Valid Ouster metadata JSON was not retained")
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        report["issues"].append(str(error))
    finally:
        if reader is not None:
            reader.close()
    report["serialized_bytes_scanned"] = bytes_read
    report["elapsed_seconds"] = time.monotonic()-began
    report["driver_tf"] = {"matching_observations": tf_good, "conflicting_observations": tf_conflicting}
    report["valid_metadata_messages"] = metadata_valid
    report["streams"] = {}
    for name, entry in state.items():
        expected_hz = args.camera_hz if name in ("image", "info") else args.lidar_hz if name == "lidar" else None
        timing = summarize_timing(entry["headers"], entry["recorded"], expected_hz)
        timing["saved_messages"] = len(entry["recorded"])
        timing.update(topic=topics[name], metadata_saved_messages=stored.get(topics[name], {}).get("count", 0),
                      invalid_messages=entry["bad"], frame_counts=dict(entry["frames"]))
        if "last_layout" in entry:
            timing["last_valid_payload_layout"] = entry["last_layout"]
        if "metadata_top_level_keys" in entry:
            timing["metadata_top_level_keys"] = entry["metadata_top_level_keys"]
        report["streams"][name] = timing
        if report["scan_complete"] and timing["saved_messages"] != timing["metadata_saved_messages"]:
            report["issues"].append(f"Scanned count differs from bag metadata for {topics[name]}")
        if name in ("image", "info", "lidar", "imu") and (timing["nonpositive_header_stamps"] or timing["backward_header_steps"]):
            report["issues"].append(f"Invalid/nonmonotonic source headers in {topics[name]}")
        if timing["duplicate_adjacent_header_stamps"] or timing["gaps_above_1_5_nominal_period"]:
            report["warnings"].append(f"Header duplicates or nominal cadence gaps observed in {topics[name]}; see stream metrics")
        if entry["bad"] and not any(topics[name] in issue for issue in report["issues"]):
            report["issues"].append(f"Malformed saved messages in {topics[name]}")
    report["camera_lidar_header_offsets"] = nearest_header_offsets(state["image"]["headers"], state["lidar"]["headers"])
    recorder_log = bag.parent/(bag.name+".recorder.log")
    if recorder_log.is_file():
        report["recorder_cache_losses"] = recorder_cache_losses(recorder_log.read_text(errors="replace"))
        report["recorder_cache_losses"]["log_path"] = str(recorder_log)
        if report["recorder_cache_losses"]["has_confirmed_recorder_cache_loss"]:
            report["issues"].append("Recorder cache overflow caused confirmed message loss; see cumulative recorder counters")
    else:
        report["warnings"].append("Recorder log unavailable; recorded cadence alone cannot prove absence of recorder cache loss")
    report["passed"] = report["scan_complete"] and not report["issues"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"report": str(output), "passed": report["passed"], "scan_complete": report["scan_complete"],
                      "saved_counts": {name: entry["saved_messages"] for name, entry in report["streams"].items()},
                      "issues": report["issues"], "warnings": report["warnings"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
