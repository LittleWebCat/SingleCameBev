#!/usr/bin/env python3
"""Replay a recorded native scan into the C++ camera colorizer in an isolated domain.

This verifies software association and input rejection. It does not verify physical
calibration accuracy or sensor exposure synchronization, and uses no sensor hardware.
Source /opt/ros/jazzy/setup.bash and this workspace's install/setup.bash first.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


def cloud_metadata(message):
    return {"width": int(message.width), "height": int(message.height),
            "point_step": int(message.point_step), "row_step": int(message.row_step),
            "is_bigendian": bool(message.is_bigendian), "is_dense": bool(message.is_dense),
            "fields": [{"name": item.name, "offset": int(item.offset),
                        "datatype": int(item.datatype), "count": int(item.count)}
                       for item in message.fields]}


def scalar_field(message, name):
    """Decode a scalar without converting FLOAT32 RGB bits through Python floats."""
    import numpy as np
    fields = {item.name: item for item in message.fields}
    if name not in fields:
        raise ValueError(f"Output cloud is missing {name}")
    field = fields[name]
    types = {2: "u1", 6: "u4", 7: "f4"}
    if field.count != 1 or field.datatype not in types:
        raise ValueError(f"Output {name} has an unsupported scalar type")
    if name == "source_index" and field.datatype != 6:
        raise ValueError("Output source_index must be UINT32")
    if name == "camera_colored" and field.datatype != 2:
        raise ValueError("Output camera_colored must be UINT8")
    if name == "rgb" and field.datatype not in (6, 7):
        raise ValueError("Output packed RGB must be UINT32 or FLOAT32")
    size = 1 if field.datatype == 2 else 4
    if field.offset + size > message.point_step:
        raise ValueError(f"Output {name} extends beyond point_step")
    if message.row_step < message.width * message.point_step:
        raise ValueError("Output row_step is shorter than packed points")
    payload = bytes(message.data)
    if len(payload) != message.row_step * message.height:
        raise ValueError("Output cloud byte length disagrees with its layout")
    order = ">" if message.is_bigendian else "<"
    code = "u4" if name == "rgb" else types[field.datatype]
    return np.ndarray((message.height, message.width), dtype=order + code,
                      buffer=payload, offset=field.offset,
                      strides=(message.row_step, message.point_step)).reshape(-1).copy()


@contextmanager
def child_process(arguments, report):
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield process, log
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            log.seek(0)
            report["node_log_tail"] = log.read().decode("utf-8", errors="replace")[-5000:]


def main():
    root = Path(__file__).resolve().parents[1]
    imported = root / "config/calibration/imported_front_001"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path,
                        default=root / "data/calibration/imported_projection_B_001")
    parser.add_argument("--camera-calibration", type=Path,
                        default=imported / "front_camera_1024x768.candidate.yaml")
    parser.add_argument("--extrinsics", type=Path,
                        default=imported / "T_B_os_sensor.candidate.yaml")
    parser.add_argument("--executable", type=Path,
                        default=root / "install/bev_visualization/lib/bev_visualization/camera_rgb_cloud")
    parser.add_argument("--output", type=Path,
                        default=root / "evaluation/camera_bev_replay.json")
    parser.add_argument("--domain-id", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists; choose a new evidence path")
    if not 2 <= args.domain_id <= 231:
        parser.error("domain-id must be between 2 and 231, separate from live domain 1")
    if not 12 <= args.timeout <= 25:
        parser.error("timeout must be between 12 and 25 seconds")
    if not args.executable.is_file():
        parser.error(f"C++ executable has not been built: {args.executable}")

    # All child traffic is private and local. Do this before importing rclpy.
    os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
    os.environ.pop("ROS_LOCALHOST_ONLY", None)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")
    os.environ.setdefault("CYCLONEDDS_URI", "file://" + str(root / "config/transport/cyclonedds_large_clouds.xml"))
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/camera_lidar_bev-rgb-check-logs")

    import cv2
    import numpy as np
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from geometry_msgs.msg import TransformStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
    from tf2_msgs.msg import TFMessage
    from extrinsic_capture import decode_point_cloud
    from camera_bev import associate_camera_colors

    report = {"schema_version": 1, "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "recorded_fixture_software_integration_check",
              "ros_domain_id": args.domain_id, "live_hardware_used": False,
              "calibration_qualification": "provisional_no_independent_accuracy_validation",
              "physical_calibration_accuracy_validated": False,
              "exposure_synchronization_validated": False, "errors": [],
              "output_cloud_messages": 0, "output_empty_messages": 0,
              "output_verified_messages": 0, "published_cloud_messages": 0,
              "verified_provisional_diagnostics": 0, "verified_fresh_diagnostics": 0,
              "source_files": {}}
    started = time.monotonic()
    deadline = started + args.timeout
    node = None
    log = None
    try:
        for path in (args.camera_calibration, args.extrinsics,
                     args.fixture / "camera_raw.png", args.fixture / "cloud.bin",
                     args.fixture / "cloud_metadata.json", args.fixture / "camera_info.json",
                     args.fixture / "projection_report.json"):
            report["source_files"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        cloud_meta = json.loads((args.fixture / "cloud_metadata.json").read_text())
        info_meta = json.loads((args.fixture / "camera_info.json").read_text())
        projection_report = json.loads((args.fixture / "projection_report.json").read_text())
        cloud_bytes = (args.fixture / "cloud.bin").read_bytes()
        image = cv2.imread(str(args.fixture / "camera_raw.png"), cv2.IMREAD_COLOR)
        if image is None or image.shape != (info_meta["height"], info_meta["width"], 3):
            raise ValueError("Recorded image does not match exact native CameraInfo dimensions")
        source = decode_point_cloud(cloud_meta, cloud_bytes)
        if len(source.xyz) != 262144:
            raise ValueError("Expected the original 262144-return fixture")
        sensor_from_lidar = np.asarray(projection_report["T_os_sensor_from_os_lidar"], dtype=np.float64)
        camera_from_lidar = np.asarray(projection_report["hypotheses"]["via_os_sensor"]["T_camera_from_os_lidar"], dtype=np.float64)
        reference_intrinsics = {"model": "pinhole", "image_size": [info_meta["width"], info_meta["height"]],
                                "K": np.asarray(info_meta["k"]).reshape(3, 3).tolist(), "D": info_meta["d"]}
        reference_extrinsics = {"direction": "lidar_to_camera", "units": "m",
                                "T_camera_from_lidar": (camera_from_lidar @ np.linalg.inv(sensor_from_lidar)).tolist()}
        expected = associate_camera_colors(image, source.xyz, reference_intrinsics,
                                           reference_extrinsics, sensor_from_lidar)
        expected_indices = np.flatnonzero(expected.valid_mask)
        expected_xyz = expected.xyz_sensor[expected_indices]
        expected_colors = expected.colors_bgr[expected_indices, ::-1]
        expected_mask = expected.colored_mask[expected_indices].astype(np.uint8)
        report["fixture"] = {"original_point_count": len(source.xyz),
                             "valid_point_count": len(expected_indices),
                             "camera_colored_point_count": int(np.count_nonzero(expected_mask)),
                             "native_image_size": [info_meta["width"], info_meta["height"]],
                             "original_header_difference_s": projection_report["absolute_header_difference_s"],
                             "replay_header_policy": "fresh_host_stamps_preserving_recorded_pair_offset",
                             "output_frame": "os_sensor"}
        if report["fixture"]["camera_colored_point_count"] < 1000:
            raise ValueError("Independent fixture association contains too few camera-colored points")
        rclpy.init()
        node = rclpy.create_node("camera_bev_recorded_fixture_check")
        prefix = "/camera_bev_check"
        topics = {"image_topic": prefix + "/image_raw", "camera_info_topic": prefix + "/camera_info",
                  "points_topic": prefix + "/points", "output_topic": prefix + "/rgb_points"}
        if any(node.count_publishers(topic) for topic in topics.values()):
            raise ValueError("Private verifier topics already have a publisher; choose another isolated domain")
        reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        static = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        publishers = {"image": node.create_publisher(Image, topics["image_topic"], qos_profile_sensor_data),
                      "info": node.create_publisher(CameraInfo, topics["camera_info_topic"], qos_profile_sensor_data),
                      "cloud": node.create_publisher(PointCloud2, topics["points_topic"], reliable),
                      "tf": node.create_publisher(TFMessage, "/tf_static", static)}
        published = {}
        verified_by_stage = {}
        verified_stamps_by_stage = {}
        wrong_info_diagnostic_stages = set()
        active_stage = "discovery"
        seen_stamps = set()
        last_empty = True
        last_nonempty_receipt = None

        def receive(message):
            nonlocal last_empty, last_nonempty_receipt
            report["output_cloud_messages"] += 1
            if message.width * message.height == 0:
                report["output_empty_messages"] += 1
                last_empty = True
                return
            last_empty = False
            last_nonempty_receipt = time.monotonic()
            key = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            try:
                if key not in published:
                    raise ValueError("Nonempty output does not preserve a published source cloud stamp")
                stage = published[key]
                if stage.startswith("bad"):
                    raise ValueError("Nonempty output was published for mismatched CameraInfo")
                if key in seen_stamps:
                    raise ValueError("Previously consumed cloud was republished")
                seen_stamps.add(key)
                if message.header.frame_id != "os_sensor":
                    raise ValueError("Output XYZ is not labeled os_sensor")
                xyz = decode_point_cloud(cloud_metadata(message), bytes(message.data)).xyz
                indices = scalar_field(message, "source_index")
                colored = scalar_field(message, "camera_colored")
                rgb = scalar_field(message, "rgb") & 0x00ffffff
                if not np.array_equal(indices, expected_indices):
                    raise ValueError("Original source indices differ from independent Python association")
                if not np.allclose(xyz, expected_xyz, atol=1e-5, rtol=1e-6):
                    raise ValueError("XYZ differs from observed sensor-from-lidar transform")
                if not np.array_equal(colored, expected_mask):
                    raise ValueError("Camera-colored mask differs from independent nearest-depth projection")
                expected_rgb = (expected_colors[:, 0].astype(np.uint32) << 16
                                | expected_colors[:, 1].astype(np.uint32) << 8
                                | expected_colors[:, 2].astype(np.uint32))
                if not np.array_equal(rgb, expected_rgb):
                    raise ValueError("RGB differs from exact raw camera pixel samples or white fallback")
                verified_by_stage[stage] = verified_by_stage.get(stage, 0) + 1
                verified_stamps_by_stage.setdefault(stage, []).append(key)
                report["output_verified_messages"] += 1
            except (ValueError, TypeError) as error:
                if str(error) not in report["errors"]:
                    report["errors"].append(str(error))

        def receive_diagnostics(message):
            for status in message.status:
                if status.name != "camera_lidar_bev/camera_rgb_cloud":
                    continue
                values = {item.key: item.value for item in status.values}
                level = (int.from_bytes(status.level, "little")
                         if isinstance(status.level, (bytes, bytearray)) else int(status.level))
                report["last_diagnostic"] = {"level": level, "message": status.message, "values": values}
                required = {"calibration_status": "external_candidate_unvalidated",
                            "source_frame_selection": "user_visual_B_os_sensor", "accuracy_validated": "false",
                            "synchronization_validated": "false", "tf_published": "false", "output_frame": "os_sensor"}
                if level != 1 or any(values.get(name) != value for name, value in required.items()):
                    report["errors"].append("RGB diagnostics did not retain provisional B geometry and synchronization WARN")
                    continue
                report["verified_provisional_diagnostics"] += 1
                if active_stage.startswith("bad") and "CameraInfo" in status.message and "match" in status.message:
                    wrong_info_diagnostic_stages.add(active_stage)
                if values.get("publishing_fresh_cloud") == "true":
                    try:
                        if (int(values.get("input_points", "-1")) != len(source.xyz)
                                or int(values.get("valid_points", "-1")) != len(expected_indices)
                                or int(values.get("colored_points", "-1")) != report["fixture"]["camera_colored_point_count"]
                                or not 0 <= float(values.get("header_delta_s", "nan")) <= 0.15):
                            raise ValueError("Fresh RGB diagnostic point counts or header pairing differ from fixture")
                        report["verified_fresh_diagnostics"] += 1
                    except ValueError as error:
                        report["errors"].append(str(error))

        subscription = node.create_subscription(PointCloud2, topics["output_topic"], receive, reliable)
        diagnostic_subscription = node.create_subscription(DiagnosticArray, "/camera_lidar_bev/diagnostics",
                                                           receive_diagnostics, 5)
        observed = projection_report["observed_driver_tf"]
        transform = TransformStamped()
        transform.header.frame_id = observed["parent_frame"]
        transform.child_frame_id = observed["child_frame"]
        for axis, value in zip(("x", "y", "z"), observed["translation_m"]):
            setattr(transform.transform.translation, axis, value)
        for axis, value in zip(("x", "y", "z", "w"), observed["quaternion_xyzw"]):
            setattr(transform.transform.rotation, axis, value)
        tf_message = TFMessage(transforms=[transform])
        offset_ns = round(projection_report["absolute_header_difference_s"] * 1_000_000_000)

        def publish_pair(stage):
            now_ns = node.get_clock().now().nanoseconds
            image_ns = now_ns - offset_ns
            raw = Image()
            raw.header.frame_id = info_meta["header"]["frame_id"]
            raw.header.stamp.sec, raw.header.stamp.nanosec = divmod(image_ns, 1_000_000_000)
            raw.width, raw.height = info_meta["width"], info_meta["height"]
            raw.encoding, raw.step = "bgr8", raw.width * 3
            raw.data = image.tobytes()
            info = CameraInfo()
            info.header = raw.header
            info.width, info.height, info.distortion_model = raw.width, raw.height, info_meta["distortion_model"]
            for name in ("k", "d", "r", "p"):
                setattr(info, name, info_meta[name])
            if stage.startswith("bad"):
                info.k[0] += 10.0
            cloud = PointCloud2()
            cloud.header.frame_id = cloud_meta["header"]["frame_id"]
            cloud.header.stamp.sec, cloud.header.stamp.nanosec = divmod(now_ns, 1_000_000_000)
            for name in ("width", "height", "point_step", "row_step", "is_bigendian", "is_dense"):
                setattr(cloud, name, cloud_meta[name])
            cloud.fields = [PointField(**item) for item in cloud_meta["fields"]]
            cloud.data = cloud_bytes
            published[now_ns] = stage
            report["published_cloud_messages"] += 1
            publishers["tf"].publish(tf_message)
            publishers["info"].publish(info)
            publishers["image"].publish(raw)
            publishers["cloud"].publish(cloud)

        arguments = [str(args.executable), "--ros-args", "-r", "__node:=camera_bev_check_cpp"]
        parameters = dict(topics, intrinsics_file=str(args.camera_calibration.resolve()),
                          extrinsics_file=str(args.extrinsics.resolve()), max_pair_delta_s=0.15,
                          stale_after_s=0.5, max_publish_hz=3.0)
        for name, value in parameters.items():
            arguments += ["-p", f"{name}:={value}"]
        report["node_arguments"] = arguments
        with child_process(arguments, report) as (process, log):
            def spin():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Bounded replay verifier deadline exceeded")
                if process.poll() is not None:
                    raise RuntimeError(f"C++ colorizer exited with {process.returncode}")
                rclpy.spin_once(node, timeout_sec=0.03)

            discovery_until = min(deadline, time.monotonic() + 5.0)
            while time.monotonic() < discovery_until:
                spin()
                if all(publishers[key].get_subscription_count() for key in ("image", "info", "cloud")):
                    break
            else:
                raise TimeoutError("C++ node did not subscribe to private replay topics")

            def drive(stage, duration, required=0):
                nonlocal active_stage
                active_stage = stage
                phase_until = time.monotonic() + duration
                next_publish = 0.0
                while time.monotonic() < phase_until or verified_by_stage.get(stage, 0) < required:
                    if time.monotonic() >= next_publish:
                        publish_pair(stage)
                        next_publish = time.monotonic() + 0.4
                    spin()
                    if report["errors"]:
                        raise ValueError(report["errors"][0])

            drive("bad_initial", 1.2)
            drive("valid", 2.0, required=3)
            # Expire previous valid camera frames before introducing wrong intrinsics.
            quiet_until = time.monotonic() + 0.7
            while time.monotonic() < quiet_until:
                spin()
            drive("bad_after_valid", 1.2)
            drive("valid_recovered", 1.0, required=1)
            stopped = time.monotonic()
            # Permit in-flight delivery, then observe beyond the configured age bound.
            while time.monotonic() - stopped < 1.7:
                spin()
            if last_nonempty_receipt is not None and last_nonempty_receipt - stopped > 0.7:
                raise ValueError("Nonempty cloud persisted after input age limit")
            if not last_empty:
                raise ValueError("Stale camera-colored cloud was not cleared")
            if report["verified_fresh_diagnostics"] < 1:
                raise ValueError("No fresh diagnostic verified provisional B qualification and fixture counts")
            if wrong_info_diagnostic_stages != {"bad_initial", "bad_after_valid"}:
                raise ValueError("Both wrong-CameraInfo phases must report that CameraInfo was received and rejected")
            report["verified_by_stage"] = verified_by_stage
            valid_stamps = verified_stamps_by_stage["valid"]
            report["valid_replay_output_rate_hz"] = round((len(valid_stamps)-1) * 1e9 /
                                                        (max(valid_stamps)-min(valid_stamps)), 6)
            report["bad_camera_info_rejected_before_and_after_valid_input"] = True
            report["wrong_camera_info_diagnostic_stages"] = sorted(wrong_info_diagnostic_stages)
            report["recovered_after_matching_camera_info"] = True
            report["stale_output_cleared"] = True
            report["old_cloud_republication_observed"] = False
            report["post_input_observation_s"] = round(time.monotonic() - stopped, 6)
        node.destroy_subscription(subscription)
        node.destroy_subscription(diagnostic_subscription)
    except Exception as error:
        if str(error) not in report["errors"]:
            report["errors"].append(str(error))
        if log is not None:
            try:
                log.seek(0)
                report["node_log_tail"] = log.read().decode("utf-8", errors="replace")[-5000:]
            except ValueError:
                pass
    finally:
        if node is not None:
            node.destroy_node()
        if "rclpy" in locals() and rclpy.ok():
            rclpy.shutdown()
        report["elapsed_s"] = round(time.monotonic() - started, 6)
        report["checks_passed"] = not report["errors"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"checks_passed": report["checks_passed"],
                          "verified_clouds": report["output_verified_messages"],
                          "fixture": report.get("fixture"), "errors": report["errors"],
                          "evidence_file": str(args.output)}, indent=2))
    return 0 if report["checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
