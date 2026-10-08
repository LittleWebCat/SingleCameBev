#!/usr/bin/python3
"""Compare candidate source frames or view user-selected B on the native camera.

This diagnostic never publishes TF or establishes accuracy or exposure
synchronization. --view b records a user visual preference; calibration remains
provisional. Keep the sensor rig and observed scene stationary.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from capture_extrinsics import (CloudRecord, ImageRecord, camera_info_matches,
                                choose_pair, header_key)
from extrinsic_capture import decode_point_cloud
from import_camera_lidar_calibration import _read_json, validate_documents
from import_projection import render_calibration_preview


def transform_from_tf(record):
    """Construct parent-from-child from an observed, explicitly named ROS TF."""
    if record.get("parent_frame") != "os_sensor" or record.get("child_frame") != "os_lidar":
        raise ValueError("Expected observed os_sensor parent / os_lidar child TF")
    translation = np.asarray(record.get("translation_m"))
    quaternion = np.asarray(record.get("quaternion_xyzw"))
    if (translation.shape != (3,) or quaternion.shape != (4,)
            or translation.dtype.kind not in "fi" or quaternion.dtype.kind not in "fi"
            or not np.isfinite(translation).all() or not np.isfinite(quaternion).all()):
        raise ValueError("Observed TF translation and quaternion must be finite numbers")
    norm = float(np.linalg.norm(quaternion))
    if not math.isclose(norm, 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("Observed TF quaternion must have unit length")
    x, y, z, w = quaternion / norm
    result = np.eye(4)
    result[:3, :3] = [[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                       [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                       [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]
    result[:3, 3] = translation
    return result


def expected_camera_info(camera):
    width, height = camera["image_size"]
    return {"width": width, "height": height, "distortion_model": "plumb_bob",
            "k": camera["K"].reshape(-1).tolist(), "d": camera["D"].tolist(),
            "r": np.eye(3).reshape(-1).tolist(),
            "p": np.column_stack([camera["K"], np.zeros(3)]).reshape(-1).tolist()}


def validate_pair(pair, expected, now, maximum_age_s=1.0):
    """Reject stale, cropped, mismatched or wrong-frame inputs before projection."""
    if pair is None:
        raise ValueError("Waiting for image, exact-header CameraInfo and nearby LiDAR header")
    image, info, cloud, difference = pair
    if header_key(image.header) != header_key(info["header"]):
        raise ValueError("CameraInfo header does not exactly match image header")
    if image.header["frame_id"] != "camera_front_optical_frame":
        raise ValueError("Camera image must use camera_front_optical_frame")
    if cloud.metadata["header"]["frame_id"] != "os_lidar":
        raise ValueError("This comparison requires cloud frame os_lidar")
    if not math.isfinite(difference) or not 0 <= difference <= 0.15:
        raise ValueError("Absolute header difference exceeds 0.15 seconds")
    width, height = expected["width"], expected["height"]
    if image.frame.dtype != np.uint8 or image.frame.shape != (height, width, 3):
        raise ValueError(f"Camera must capture native {width}x{height}; no resizing intrinsics")
    if not camera_info_matches(info, expected):
        raise ValueError("CameraInfo dimensions/K/D/R/P do not match the imported candidate")
    if any(info.get(key, 0) not in (0, 1) for key in ("binning_x", "binning_y")):
        raise ValueError("CameraInfo binning is incompatible with native calibration")
    roi = info.get("roi", {})
    if (roi.get("x_offset", 0) != 0 or roi.get("y_offset", 0) != 0
            or roi.get("width", 0) not in (0, width)
            or roi.get("height", 0) not in (0, height)):
        raise ValueError("CameraInfo ROI is incompatible with native calibration")
    ages = {"image": now - image.received_at, "cloud": now - cloud.received_at,
            "camera_info": now - info.get("_received_at", image.received_at)}
    if any(not math.isfinite(age) or age < 0 or age > maximum_age_s for age in ages.values()):
        raise ValueError("STALE: image, CameraInfo or LiDAR receipt age exceeds 1 second")
    return ages


def load_import(import_dir):
    intrinsics_path = import_dir / "front.json"
    extrinsics_path = import_dir / "front_external.json"
    intrinsics_bytes, intrinsics = _read_json(intrinsics_path)
    extrinsics_bytes, extrinsics = _read_json(extrinsics_path)
    camera, _, _, _ = validate_documents(intrinsics, extrinsics)
    sources = {"intrinsics": {"path": str(intrinsics_path.resolve()),
                              "sha256": hashlib.sha256(intrinsics_bytes).hexdigest()},
               "extrinsics": {"path": str(extrinsics_path.resolve()),
                              "sha256": hashlib.sha256(extrinsics_bytes).hexdigest()}}
    manifest_path = import_dir / "import_report.json"
    if manifest_path.exists():
        _, manifest = _read_json(manifest_path)
        for name in sources:
            if sources[name]["sha256"] != manifest["source_files"][name]["sha256"]:
                raise ValueError(f"Preserved {name} file hash differs from the import manifest")
    return intrinsics, extrinsics, camera, sources


def projection_report(pair, ages, preview, tf_record, sources, max_display_points):
    image, info, cloud, difference = pair
    selected_b = preview.view == "b"
    return {
        "schema_version": 1, "accepted_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": ("provisional overlay using user-selected B" if selected_b
                    else "visual comparison of two unconfirmed external source frames"),
        "view": preview.view, "frame_unselected": not selected_b,
        "selected_source_frame": "os_sensor" if selected_b else None,
        "selected_hypothesis": "via_os_sensor" if selected_b else None,
        "selection_basis": "user_visual_preference" if selected_b else None,
        "accuracy_qualification": "provisional_no_independent_accuracy_validation",
        "validation_passed": False, "tf_published": False,
        "calibration_status": "external_candidate_unvalidated", "fusion_ready": False,
        "source_files": sources, "image_header": image.header,
        "camera_info_header": info["header"], "cloud_header": cloud.metadata["header"],
        "native_image_width_px": image.frame.shape[1], "native_image_height_px": image.frame.shape[0],
        "native_dimensions_and_camera_info_k_d_r_p_match": True,
        "intrinsics_rescaled": False, "projection_domain": "raw_distorted_image",
        "camera_info": {key: value for key, value in info.items() if not key.startswith("_")},
        "absolute_header_difference_s": difference, "receipt_age_s": ages,
        "pairing_qualification": "header_pairing_heuristic_only",
        "exposure_synchronization_verified": False,
        "scene_requirement": "Keep rig and scene stationary; motion is not verified by this tool",
        "observed_driver_tf": {key: value for key, value in tf_record.items() if not key.startswith("_")},
        "T_os_sensor_from_os_lidar": transform_from_tf(tf_record).tolist(),
        "hypotheses": {name: {"T_camera_from_os_lidar": preview.hypotheses[name].tolist(),
                              "input_point_count": projection.input_count,
                              "valid_point_count": projection.valid_count,
                              "positive_optical_depth_point_count": projection.front_count,
                              "radial_domain_excluded_point_count": projection.radial_domain_excluded_count,
                              "in_frame_point_count": projection.in_frame_count,
                              "unique_projected_pixel_count": projection.projected_count,
                              "drawn_pixel_count": min(projection.projected_count, max_display_points)}
                       for name, projection in preview.projections.items()},
    }


def save_snapshot(path, report_path, evidence_dir, preview, report, pair):
    """Preserve a native-resolution diagnostic plus optional replayable raw inputs."""
    paths = [path] + ([report_path] if report_path else [])
    if any(target.exists() or target.is_symlink() for target in paths):
        raise FileExistsError("Refusing to overwrite an existing snapshot or report")
    if evidence_dir is not None:
        evidence_dir.mkdir(parents=True, exist_ok=False)
        image, info, cloud, _ = pair
        if not cv2.imwrite(str(evidence_dir / "camera_raw.png"), image.frame):
            raise OSError("Could not save raw camera evidence")
        (evidence_dir / "cloud.bin").write_bytes(memoryview(cloud.data).cast("B"))
        (evidence_dir / "cloud_metadata.json").write_text(json.dumps(cloud.metadata, indent=2, allow_nan=False) + "\n")
        (evidence_dir / "camera_info.json").write_text(json.dumps(
            {key: value for key, value in info.items() if not key.startswith("_")}, indent=2, allow_nan=False) + "\n")
        report["raw_evidence_directory"] = str(evidence_dir.resolve())
        (evidence_dir / "projection_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    separation_ms = report["absolute_header_difference_s"] * 1000
    snapshot = status_display(preview.image,
                              f"Header separation {separation_ms:.1f} ms | exposure synchronization unverified", True,
                              view=preview.view)
    if not cv2.imwrite(str(path), snapshot):
        raise OSError(f"Could not write snapshot {path}")
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Saved native diagnostic: {path.resolve()}", flush=True)


def status_display(content, message, fresh, view="compare"):
    """Keep diagnostic limits visible above both native-size image panes."""
    display = np.zeros((content.shape[0] + 80, content.shape[1], 3), dtype=np.uint8)
    display[80:] = content
    color = (255, 255, 255) if fresh else (0, 180, 255)
    lines = ["STATIC RIG + SCENE REQUIRED | " + message,
             ("B chosen visually; accuracy provisional; no TF | Q/Esc close | S save"
              if view == "b" else "Source frame remains UNSELECTED; no TF published | Q/Esc: close | S: save fresh snapshot")]
    for text, y in zip(lines, (27, 58)):
        text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
        scale = min(0.65, 0.65 * (content.shape[1] - 20) / max(text_width, 1))
        cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color if y == 27 else (255, 255, 255), 1, cv2.LINE_AA)
    return display


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-dir", type=Path, default=root / "config/calibration/imported_front_001")
    parser.add_argument("--image-topic", default="/camera/front/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/front/camera_info")
    parser.add_argument("--cloud-topic", default="/points")
    parser.add_argument("--view", choices=("compare", "b"), default="compare",
                        help="both candidates, or B chosen by user visual inspection (accuracy provisional)")
    saves = parser.add_mutually_exclusive_group()
    saves.add_argument("--snapshot", type=Path, help="save one fresh pair headlessly, then exit")
    saves.add_argument("--initial-snapshot", type=Path, help="save first fresh pair and continue live GUI")
    parser.add_argument("--report", type=Path, help="JSON evidence saved with first snapshot")
    parser.add_argument("--evidence-dir", type=Path, help="new directory for first snapshot raw image/cloud/metadata")
    parser.add_argument("--timeout", type=float, default=20.0, help="maximum startup wait for a usable pair, seconds")
    parser.add_argument("--display-width", type=int, help="GUI width only; defaults to 1440 compare / 1024 B; native pixel math")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--max-display-points", type=int, default=12000)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if not math.isfinite(args.fps) or not 0 < args.fps <= 5:
        parser.error("fps must be finite and in (0, 5]")
    display_width = args.display_width if args.display_width is not None else (1440 if args.view == "compare" else 1024)
    if not 640 <= display_width <= 1600 or args.max_display_points <= 0:
        parser.error("display-width must be 640..1600 and max-display-points positive")
    if (args.report or args.evidence_dir) and not (args.snapshot or args.initial_snapshot):
        parser.error("report/evidence-dir require snapshot or initial-snapshot")
    try:
        intrinsics, extrinsics, camera, sources = load_import(args.import_dir)
        expected = expected_camera_info(camera)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    # Offline import, unit tests and --help need no ROS installation or discovery.
    from cv_bridge import CvBridge
    import rclpy
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2
    from tf2_msgs.msg import TFMessage

    rclpy.init()
    node = rclpy.create_node("imported_camera_lidar_frame_comparison")
    bridge = CvBridge()
    images, infos = deque(maxlen=50), {}
    latest_cloud = None
    observed_tf = None
    callback_error = None

    def ros_header(header):
        return {"frame_id": header.frame_id, "stamp_sec": int(header.stamp.sec),
                "stamp_nanosec": int(header.stamp.nanosec)}

    def receive_image(message):
        nonlocal callback_error
        try:
            frame = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            images.append(ImageRecord(frame, ros_header(message.header), time.monotonic()))
        except Exception as error:
            callback_error = f"Image decoding rejected: {error}"

    def receive_info(message):
        header = ros_header(message.header)
        infos[header_key(header)] = {
            "header": header, "width": int(message.width), "height": int(message.height),
            "distortion_model": message.distortion_model,
            "k": list(map(float, message.k)), "d": list(map(float, message.d)),
            "r": list(map(float, message.r)), "p": list(map(float, message.p)),
            "binning_x": int(message.binning_x), "binning_y": int(message.binning_y),
            "roi": {"x_offset": int(message.roi.x_offset), "y_offset": int(message.roi.y_offset),
                    "width": int(message.roi.width), "height": int(message.roi.height),
                    "do_rectify": bool(message.roi.do_rectify)}, "_received_at": time.monotonic(),
        }
        while len(infos) > 100:
            del infos[next(iter(infos))]

    def receive_cloud(message):
        nonlocal latest_cloud
        metadata = {"header": ros_header(message.header), "width": int(message.width),
                    "height": int(message.height), "point_step": int(message.point_step),
                    "row_step": int(message.row_step), "is_bigendian": bool(message.is_bigendian),
                    "is_dense": bool(message.is_dense),
                    "fields": [{"name": field.name, "offset": int(field.offset),
                                "datatype": int(field.datatype), "count": int(field.count)}
                               for field in message.fields]}
        latest_cloud = CloudRecord(metadata, message.data, time.monotonic())

    def receive_tf(message):
        nonlocal observed_tf, callback_error
        for entry in message.transforms:
            if entry.header.frame_id != "os_sensor" or entry.child_frame_id != "os_lidar":
                continue
            t, q = entry.transform.translation, entry.transform.rotation
            record = {"parent_frame": entry.header.frame_id, "child_frame": entry.child_frame_id,
                      "header": ros_header(entry.header), "translation_m": [t.x, t.y, t.z],
                      "quaternion_xyzw": [q.x, q.y, q.z, q.w], "source_topic": "/tf_static"}
            try:
                matrix = transform_from_tf(record)
                if observed_tf is not None and not np.allclose(matrix, transform_from_tf(observed_tf), atol=1e-9, rtol=0):
                    raise ValueError("Conflicting os_sensor<-os_lidar static TF observed")
                observed_tf = record
            except ValueError as error:
                callback_error = str(error)

    subscriptions = [
        node.create_subscription(Image, args.image_topic, receive_image,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)),
        node.create_subscription(CameraInfo, args.camera_info_topic, receive_info,
                                 QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)),
        node.create_subscription(PointCloud2, args.cloud_topic, receive_cloud,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)),
        node.create_subscription(TFMessage, "/tf_static", receive_tf,
                                 QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                                            durability=DurabilityPolicy.TRANSIENT_LOCAL)),
    ]
    window = ("Imported calibration: B via os_sensor | USER VISUAL CHOICE, ACCURACY PROVISIONAL"
              if args.view == "b" else "Imported calibration: A os_lidar / B os_sensor | SOURCE UNCONFIRMED")
    started = time.monotonic()
    last_render = -math.inf
    accepted = False
    saved_first = False
    manual_count = 0
    preview = report = pair = None
    reason = "Waiting for native camera / CameraInfo / LiDAR and observed driver TF"
    width, height = camera["image_size"]
    try:
        print("Keep the rig and scene stationary. " + ("B is a user visual choice; accuracy remains provisional."
              if args.view == "b" else "Both panes are unconfirmed frame hypotheses."), flush=True)
        print("Native raw projection; white LiDAR markers with black outlines. No TF is published.", flush=True)
        if args.snapshot is None:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            pane_count = 2 if args.view == "compare" else 1
            cv2.resizeWindow(window, display_width, int(display_width * (height + 140) / (pane_count * width)))
        while rclpy.ok():
            for _ in range(4):
                rclpy.spin_once(node, timeout_sec=0.002)
            now = time.monotonic()
            if callback_error:
                raise RuntimeError(callback_error)
            if now - last_render >= 1 / args.fps:
                last_render = now
                fresh_images = [image for image in images if 0 <= now - image.received_at <= 1.0]
                candidate = choose_pair(fresh_images, infos, latest_cloud, 0.15)
                try:
                    ages = validate_pair(candidate, expected, now)
                    if observed_tf is None:
                        raise ValueError("Waiting for actual driver os_sensor<-os_lidar static TF")
                    decoded = decode_point_cloud(candidate[2].metadata, candidate[2].data)
                    rendered = render_calibration_preview(candidate[0].frame, decoded.xyz, intrinsics, extrinsics,
                                                          transform_from_tf(observed_tf), args.max_display_points,
                                                          view=args.view)
                    evidence = projection_report(candidate, ages, rendered, observed_tf, sources, args.max_display_points)
                    preview, report, pair = rendered, evidence, candidate
                    reason = f"Header separation {candidate[3]*1000:.1f} ms | exposure synchronization unverified"
                    accepted = True
                    first_path = args.snapshot or args.initial_snapshot
                    if first_path is not None and not saved_first:
                        save_snapshot(first_path, args.report, args.evidence_dir, preview, report, pair)
                        saved_first = True
                        if args.snapshot is not None:
                            return 0
                except ValueError as error:
                    reason = str(error)
                    pair = None
            if not accepted and now - started > args.timeout:
                raise RuntimeError(f"No usable pair within {args.timeout:g}s: {reason}")
            if args.snapshot is None:
                pane_count = 2 if args.view == "compare" else 1
                content = preview.image if preview is not None else np.zeros((height + 60, width*pane_count, 3), dtype=np.uint8)
                valid_live = False
                if pair is not None:
                    try:
                        validate_pair(pair, expected, time.monotonic())
                        valid_live = True
                    except ValueError:
                        reason = "STALE: showing last accepted pair; waiting for fresh matching inputs"
                if preview is not None and not valid_live and not reason.startswith("STALE"):
                    reason = "STALE: last accepted view | " + reason
                display = status_display(content, reason, valid_live, view=args.view)
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xff
                if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord("s"), ord("S")) and valid_live:
                    manual_count += 1
                    basename = args.initial_snapshot or root / "data/calibration/imported_projection_preview.png"
                    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                    path = basename.with_name(f"{basename.stem}_{timestamp}_{manual_count:03d}{basename.suffix or '.png'}")
                    save_snapshot(path, path.with_suffix(".json"), None, preview, report, pair)
        return 0
    except (RuntimeError, ValueError, OSError) as error:
        print(f"Projection preview stopped: {error}", flush=True)
        return 1
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
