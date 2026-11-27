#!/usr/bin/python3
"""Collect stationary checkerboard camera/LiDAR pairs; do not fit extrinsics."""

import argparse
from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np

from extrinsic_capture import BoardConfig, ExtrinsicCaptureSession, decode_point_cloud
from extrinsic_preview import CornerStability, render_range_view


@dataclass
class ImageRecord:
    frame: np.ndarray
    header: dict
    received_at: float


@dataclass
class CloudRecord:
    metadata: dict
    data: object
    received_at: float


def header_key(header):
    return header["stamp_sec"], header["stamp_nanosec"], header["frame_id"]


def stamp_ns(header):
    return int(header["stamp_sec"]) * 1_000_000_000 + int(header["stamp_nanosec"])


def choose_pair(images, infos, cloud, maximum_difference_s):
    """Nearest image with exact-header CameraInfo; no synchronization claim."""
    if cloud is None:
        return None
    cloud_time = stamp_ns(cloud.metadata["header"])
    candidates = [(abs(stamp_ns(image.header) - cloud_time), image)
                  for image in images if header_key(image.header) in infos]
    if not candidates:
        return None
    difference, image = min(candidates, key=lambda item: item[0])
    if difference > maximum_difference_s * 1_000_000_000:
        return None
    return image, infos[header_key(image.header)], cloud, difference / 1_000_000_000


def camera_info_matches(info, expected):
    if any(info.get(key) != expected[key] for key in ("width", "height", "distortion_model")):
        return False
    for key in ("k", "d", "r", "p"):
        values = np.asarray(info.get(key), dtype=float)
        reference = np.asarray(expected[key], dtype=float)
        if (values.shape != reference.shape or not np.isfinite(values).all()
                or not np.allclose(values, reference, rtol=1e-9, atol=1e-9)):
            return False
    return True


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new dataset directory")
    parser.add_argument("--board-config", type=Path,
                        default=root / "config/calibration/checkerboard.yaml")
    parser.add_argument("--calibration-file", type=Path,
                        default=root / "config/calibration/front_camera_640x480_002.candidate.yaml")
    parser.add_argument("--image-topic", default="/camera/front/image_raw")
    parser.add_argument("--camera-info-topic", default="/camera/front/camera_info")
    parser.add_argument("--cloud-topic", default="/points")
    parser.add_argument("--target-poses", type=int, default=15)
    parser.add_argument("--max-header-difference-s", type=float, default=0.15,
                        help="header pairing limit only; not exposure synchronization")
    parser.add_argument("--min-board-span-px", type=float, default=140.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--range-color-max-m", type=float, default=15.0)
    parser.add_argument("--lidar-display", choices=("white", "range"), default="white",
                        help="LiDAR points: white on black (default) or range colors")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    if args.target_poses < 1:
        parser.error("target-poses must be positive")
    for name in ("max_header_difference_s", "min_board_span_px", "settle_seconds",
                 "range_color_max_m", "timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive")
    if args.max_header_difference_s > 0.15:
        parser.error("max-header-difference-s cannot exceed the 0.15 s dataset limit")
    try:
        board = BoardConfig.from_yaml(args.board_config)
        session = ExtrinsicCaptureSession(args.output, board, args.calibration_file,
                                          image_topic=args.image_topic,
                                          camera_info_topic=args.camera_info_topic,
                                          cloud_topic=args.cloud_topic,
                                          min_board_span_px=args.min_board_span_px)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))

    # Parse --help and validate inputs without importing ROS.
    from cv_bridge import CvBridge
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2

    rclpy.init()
    node = rclpy.create_node("stationary_camera_lidar_extrinsic_capture")
    bridge = CvBridge()
    images = deque(maxlen=40)
    infos = {}
    latest_cloud = None
    cached_pair = None
    cached_corners = None
    cached_found = False
    cached_span = 0.0
    cached_info_matches = False
    raster = None
    last_preview_at = -math.inf
    last_cloud_key = None
    last_corner_key = None
    stability = CornerStability(args.settle_seconds)
    stable_elapsed = 0.0
    stable_ready = False
    selection = None
    current_hint = None
    zoom_direction = None
    message = f"Hold the rigid board and sensors still for at least {args.settle_seconds:g} seconds before saving"
    window = "Camera - LiDAR stationary pairs | extrinsics UNMEASURED"
    pane_width, pane_height, top_height, band_height = 640, 480, 34, 166
    opened = False
    started = time.monotonic()

    def ros_header(header):
        return {"frame_id": header.frame_id, "stamp_sec": int(header.stamp.sec),
                "stamp_nanosec": int(header.stamp.nanosec)}

    def receive_image(image):
        try:
            frame = bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            header = ros_header(image.header)
            header.update(timestamp_source="host_receipt_after_read", image_encoding=image.encoding)
            images.append(ImageRecord(frame, header, time.monotonic()))
        except Exception as error:
            node.get_logger().error(f"Image decode failed: {error}", throttle_duration_sec=5.0)

    def receive_info(info):
        header = ros_header(info.header)
        infos[header_key(header)] = {
            "header": header, "width": int(info.width), "height": int(info.height),
            "distortion_model": info.distortion_model,
            "k": list(map(float, info.k)), "d": list(map(float, info.d)),
            "r": list(map(float, info.r)), "p": list(map(float, info.p)),
            "binning_x": int(info.binning_x), "binning_y": int(info.binning_y),
        }
        while len(infos) > 80:
            del infos[next(iter(infos))]

    def receive_cloud(cloud):
        nonlocal latest_cloud
        metadata = {"header": ros_header(cloud.header),
                    "width": int(cloud.width), "height": int(cloud.height),
                    "point_step": int(cloud.point_step), "row_step": int(cloud.row_step),
                    "is_bigendian": bool(cloud.is_bigendian), "is_dense": bool(cloud.is_dense),
                    "fields": [{"name": field.name, "offset": int(field.offset),
                                "datatype": int(field.datatype), "count": int(field.count)}
                               for field in cloud.fields]}
        latest_cloud = CloudRecord(metadata, cloud.data, time.monotonic())

    def click(event, x, y, flags, param):
        nonlocal selection, current_hint, zoom_direction, last_cloud_key, message
        if x < pane_width or not top_height <= y < top_height + pane_height:
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            zoom_direction = None
            last_cloud_key = None
            message = "LiDAR 360-degree view restored; click a return on the actual board"
            return
        if event != cv2.EVENT_LBUTTONDOWN or raster is None or cached_pair is None:
            return
        sample = raster.sample(x - pane_width, y - top_height)
        if sample is None:
            message = "No return near that pixel; select a colored return on the board"
            return
        header = dict(cached_pair[2].metadata["header"])
        selection = {"direction": (sample["azimuth_deg"], sample["elevation_deg"]),
                     "original_click_cloud_header": header}
        current_hint = sample
        zoom_direction = selection["direction"]
        last_cloud_key = None
        message = "Selected return. In zoom, click the actual board; SPACE confirms both sensors and board stayed still"

    subscriptions = [
        node.create_subscription(Image, args.image_topic, receive_image,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)),
        node.create_subscription(CameraInfo, args.camera_info_topic, receive_info,
                                 QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)),
        node.create_subscription(PointCloud2, args.cloud_topic, receive_cloud,
                                 QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)),
    ]
    try:
        print(f"Dataset: {args.output.resolve()}", flush=True)
        print(f"Board: {board.pattern[0]}x{board.pattern[1]} inner corners, {board.square_size_mm:g} mm squares", flush=True)
        print(f"Hold the board and both sensors still >={args.settle_seconds:g} seconds. Click actual board returns in LiDAR view.", flush=True)
        print("Left click selects/zooms; right click restores 360 degrees. SPACE/S saves; Q/ESC closes.", flush=True)
        print("CameraInfo must exactly match image headers and the referenced provisional intrinsics.", flush=True)
        print("No extrinsics or exposure synchronization have been measured.", flush=True)
        while rclpy.ok():
            # Keep callbacks drained while corner detection / point rendering are capped at 5 Hz.
            for _ in range(4):
                rclpy.spin_once(node, timeout_sec=0.002)
            now = time.monotonic()
            if not images:
                if now - started > args.timeout:
                    raise RuntimeError(f"No camera images on {args.image_topic}; check bridge and ROS_DOMAIN_ID")
                continue
            if not opened:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, pane_width * 2, pane_height + top_height + band_height)
                cv2.setMouseCallback(window, click)
                opened = True

            if now - last_preview_at >= 0.2:
                last_preview_at = now
                pair = choose_pair(images, infos, latest_cloud, args.max_header_difference_s)
                if pair is not None:
                    cached_pair = pair
                    image, info, cloud, difference = pair
                    cached_info_matches = camera_info_matches(info, session.expected_camera_info)
                    image_key = header_key(image.header)
                    if image_key != last_corner_key:
                        gray = cv2.cvtColor(image.frame, cv2.COLOR_BGR2GRAY)
                        cached_found, cached_corners = cv2.findChessboardCornersSB(
                            gray, board.pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
                        cached_span = (float(np.ptp(cached_corners.reshape(-1, 2), axis=0).max())
                                       if cached_found else 0.0)
                        result = stability.observe(cached_corners if cached_found else None, now)
                        stable_elapsed, stable_ready = result.elapsed_s, result.ready
                        if result.reset:
                            selection = current_hint = None
                        last_corner_key = image_key
                    cloud_key = header_key(cloud.metadata["header"])
                    if cloud_key != last_cloud_key:
                        try:
                            decoded = decode_point_cloud(cloud.metadata, cloud.data)
                            raster = render_range_view(decoded.xyz, pane_width, pane_height,
                                                       args.range_color_max_m, zoom_direction,
                                                       color_mode=args.lidar_display)
                            last_cloud_key = cloud_key
                        except ValueError as error:
                            raster = None
                            message = f"LiDAR preview rejected: {error}"
                    if selection is not None and raster is not None:
                        current_hint = raster.sample_direction(*selection["direction"])
                        if current_hint is None:
                            selection = None
                            message = "Selected direction lost its return; click the board again"
                else:
                    cached_pair = None
                    raster = None
                    last_cloud_key = None
                    stable_ready = False
                    stability.clear()
                    selection = current_hint = None

            display = np.full((pane_height + top_height + band_height, pane_width * 2, 3),
                              24, dtype=np.uint8)
            frame = cached_pair[0].frame if cached_pair is not None else images[-1].frame
            camera = frame.copy()
            if cached_pair is not None and cached_found:
                cv2.drawChessboardCorners(camera, board.pattern, cached_corners, True)
            display[top_height:top_height + pane_height, :pane_width] = cv2.resize(
                camera, (pane_width, pane_height))
            if raster is not None:
                range_pane = raster.image.copy()
                if current_hint is not None:
                    pixel = raster.pixel_for_direction(current_hint["azimuth_deg"],
                                                       current_hint["elevation_deg"])
                    if pixel is not None:
                        cv2.circle(range_pane, tuple(map(lambda value: int(round(value)), pixel)),
                                   8, (255, 255, 255), 2, cv2.LINE_AA)
                display[top_height:top_height + pane_height, pane_width:] = range_pane
            else:
                cv2.putText(display, "Waiting for valid XYZ LiDAR cloud", (pane_width + 20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 255), 1)
            cv2.putText(display, "CAMERA RAW | detected corners; provisional intrinsics", (10, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)
            cloud_frame = (cached_pair[2].metadata["header"]["frame_id"] if cached_pair is not None else "waiting")
            cv2.putText(display, f"LiDAR frame: {cloud_frame} | separate view, no fitted transform",
                        (pane_width + 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1)
            now = time.monotonic()
            image_age = now - cached_pair[0].received_at if cached_pair is not None else math.inf
            cloud_age = now - cached_pair[2].received_at if cached_pair is not None else math.inf
            if session.count >= args.target_poses:
                status = "CAPTURE GOAL REACHED: press Q; no extrinsics have been calculated"
            elif cached_pair is None:
                status = "WAITING: image + exact CameraInfo + nearest LiDAR header"
            elif image_age > 1.0 or cloud_age > 1.0:
                status = "STALE INPUT: save disabled"
            elif not cached_info_matches:
                status = "CameraInfo differs from reference calibration: save disabled"
            elif not cached_found:
                status = f"SHOW ALL {board.pattern[0] * board.pattern[1]} INNER CORNERS"
            elif cached_span < args.min_board_span_px:
                status = f"MOVE CLOSER: corner span {cached_span:.0f} / {args.min_board_span_px:g} px"
            elif not stable_ready:
                status = f"HOLD STILL: camera corner stability {stable_elapsed:.1f} / {args.settle_seconds:g} s"
            elif current_hint is None:
                status = "CLICK A RETURN ON THE ACTUAL BOARD in LiDAR view"
            elif raster is None:
                status = "LiDAR XYZ preview unavailable: save disabled"
            else:
                status = "READY: SPACE confirms board and sensors stayed still; saves displayed pair"
            ready = (cached_pair is not None and image_age <= 1.0 and cloud_age <= 1.0
                     and cached_info_matches and cached_found and cached_span >= args.min_board_span_px
                     and stable_ready and current_hint is not None and raster is not None
                     and session.count < args.target_poses)
            baseline = top_height + pane_height
            lines = [
                f"Saved {session.count}/{args.target_poses} | {status}",
                (f"Camera age {image_age * 1000:.0f} ms | LiDAR age {cloud_age * 1000:.0f} ms | "
                 f"header difference {cached_pair[3] * 1000:.1f} ms (pairing only)" if cached_pair else
                 "Header pairing is a capture heuristic; exposure synchronization is unmeasured"),
                (f"Selected USER return: range {current_hint['range_m']:.2f} m | "
                 f"azimuth {current_hint['azimuth_deg']:.1f} deg | elevation {current_hint['elevation_deg']:.1f} deg"
                 if current_hint is not None else "No board return selected; a click is user guidance, not automatic board detection"),
                "Left click: select return / zoom    Right click: 360 deg    SPACE/S: save    Q/ESC: close",
                message[:155],
                "Use varied tilts and positions; keep all corners and board returns visible. Extrinsics remain UNMEASURED.",
            ]
            for index, text in enumerate(lines):
                color = (0, 240, 0) if ready and index == 0 else (0, 170, 255) if index == 0 else (210, 210, 210)
                cv2.putText(display, text, (10, baseline + 24 + index * 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.46 if index < 4 else 0.43, color, 1, cv2.LINE_AA)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (32, ord("s"), ord("S")):
                if not ready:
                    message = f"Not saved: {status}"
                else:
                    image, info, cloud, difference = cached_pair
                    hint = dict(current_hint)
                    hint["point_index"] = hint.pop("decoded_xyz_index")
                    hint["source"] = "user_selected_return"
                    hint["header"] = dict(cloud.metadata["header"])
                    hint["original_click_cloud_header"] = selection["original_click_cloud_header"]
                    hint["manual_selection_note"] = "User identified a board return; not automatic board detection"
                    result = session.capture(image.frame, cached_corners, image.header, info,
                                             cloud.metadata, cloud.data,
                                             time.monotonic() - image.received_at,
                                             time.monotonic() - cloud.received_at,
                                             board_stationary_confirmed=True, lidar_target_hint=hint)
                    message = ("Saved. Move/tilt board, hold still, and click its current LiDAR return" if result.accepted
                               else f"Not saved: {result.reason}")
                    print(f"{result.count}/{args.target_poses}: {message}", flush=True)
                    if result.accepted:
                        selection = current_hint = None
                        stability.clear()
                        stable_ready = False
                        last_corner_key = None
                        if result.count >= args.target_poses:
                            message = f"{args.target_poses}-pose capture goal reached; press Q. No extrinsics have been fitted."
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if opened:
            cv2.destroyAllWindows()
        print(f"Saved {session.count} pairs. Manifest: {args.output / 'session.json'}", flush=True)


if __name__ == "__main__":
    main()
