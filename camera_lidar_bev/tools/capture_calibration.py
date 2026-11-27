#!/usr/bin/python3
"""Collect distinct checkerboard views from the live ROS camera topic."""

import argparse
import math
from pathlib import Path
import time

import cv2

from calibration_capture import BoardConfig, CaptureSession


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new dataset directory")
    parser.add_argument("--board-config", type=Path, default=root / "config/calibration/checkerboard.yaml")
    parser.add_argument("--topic", default="/camera/front/image_raw")
    parser.add_argument("--target-images", type=int, default=25)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--min-board-span-px", type=float, default=0.0,
                        help="minimum larger inner-corner span in pixels; capture guide, 0 disables")
    parser.add_argument("--capture-first", action="store_true", help="save the first valid view as a starting image")
    args = parser.parse_args()
    if args.target_images < 1 or not math.isfinite(args.timeout) or args.timeout <= 0.0:
        parser.error("target image count and timeout must be positive")
    if not math.isfinite(args.min_board_span_px) or args.min_board_span_px < 0.0:
        parser.error("min-board-span-px must be finite and nonnegative")
    board = BoardConfig.from_yaml(args.board_config)
    session = CaptureSession(args.output, board, args.topic,
                             min_board_span_px=args.min_board_span_px)

    from cv_bridge import CvBridge
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("front_camera_calibration_capture")
    bridge = CvBridge()
    latest = None
    latest_header = None
    received_at = 0.0
    sequence = 0
    displayed_sequence = -1
    corners = None
    found = False
    count = 0
    auto_first_pending = args.capture_first
    message = "Hold board steady; move and tilt between captures"
    window = f"Calibration Capture - {board.pattern[0]}x{board.pattern[1]} corners, {board.square_size_mm:g} mm"
    opened = False

    def receive(image):
        nonlocal latest, latest_header, received_at, sequence
        try:
            latest = bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            latest_header = {
                "frame_id": image.header.frame_id,
                "stamp_sec": int(image.header.stamp.sec),
                "stamp_nanosec": int(image.header.stamp.nanosec),
                "timestamp_source": "host_receipt_after_read",
                "image_encoding": image.encoding,
            }
            received_at = time.monotonic()
            sequence += 1
        except Exception as error:
            node.get_logger().error(f"Image decode failed: {error}", throttle_duration_sec=5.0)

    subscription = node.create_subscription(
        Image, args.topic, receive, QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
    started = time.monotonic()
    try:
        print(f"Dataset: {args.output.resolve()}", flush=True)
        print(f"Board: {board.pattern[0]}x{board.pattern[1]} inner corners, {board.square_size_mm:g} mm squares", flush=True)
        if args.min_board_span_px > 0.0:
            print(f"Capture size guide: inner corners span at least {args.min_board_span_px:g} px on one axis.", flush=True)
        print("SPACE or S saves a fresh detected view; Q or ESC closes. Move/tilt the board for each image.", flush=True)
        print("Vary tilt and cover top, bottom, and image edges; keep the whole printed board visible.", flush=True)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if latest is None:
                if time.monotonic() - started > args.timeout:
                    raise RuntimeError(f"No image received on {args.topic}; check camera bridge and ROS_DOMAIN_ID")
                continue
            if not opened:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, 960, 900)
                opened = True
            if sequence != displayed_sequence:
                gray = cv2.cvtColor(latest, cv2.COLOR_BGR2GRAY)
                found, corners = cv2.findChessboardCornersSB(
                    gray, board.pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
                displayed_sequence = sequence
            fresh = time.monotonic() - received_at <= 1.0
            preview = latest.copy()
            span_px = None
            if found:
                cv2.drawChessboardCorners(preview, board.pattern, corners, found)
                points = corners.reshape(-1, 2)
                span_px = points.max(axis=0) - points.min(axis=0)
            large_enough = found and float(span_px.max()) >= args.min_board_span_px
            display = cv2.copyMakeBorder(preview, 0, 120, 0, 0,
                                         cv2.BORDER_CONSTANT, value=(24, 24, 24))
            baseline = latest.shape[0]
            ready = found and fresh and large_enough
            status = (f"DETECTED {board.pattern[0]*board.pattern[1]} CORNERS" if ready
                      else "STALE CAMERA" if not fresh else "MOVE CLOSER" if found
                      else "SHOW ALL INNER CORNERS")
            cv2.putText(display, f"Saved {count}/{args.target_images} | {status}", (10, baseline + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if ready else (0, 120, 255), 1)
            span_text = (f"Corner span: {span_px[0]:.0f} x {span_px[1]:.0f} px" if found
                         else "Corner span: waiting for detection")
            span_text += f" | min on one axis: {args.min_board_span_px:g} px"
            cv2.putText(display, span_text, (10, baseline + 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            cv2.putText(display, "SPACE/S: capture    Q/ESC: close", (10, baseline + 66),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
            cv2.putText(display, message[:85], (10, baseline + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)
            cv2.putText(display, "Vary tilt; cover top, bottom and edges; keep the whole board visible",
                        (10, baseline + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            capture_requested = key in (ord("s"), ord("S"), 32) or (auto_first_pending and ready)
            if capture_requested and count < args.target_images:
                if not found or not fresh:
                    message = f"Not saved: hold board steady with all {board.pattern[0]*board.pattern[1]} corners visible"
                else:
                    result = session.capture(latest, corners, latest_header, time.monotonic() - received_at)
                    count = result.count
                    message = ("Saved. Move or tilt the board for a different view" if result.accepted
                               else f"Not saved: {result.reason}")
                    print(f"{count}/{args.target_images}: {message}", flush=True)
                    if result.accepted:
                        auto_first_pending = False
                    if count >= args.target_images:
                        message = "Capture goal reached. Press Q; calibration fitting is the next step"
                        print("Capture goal reached; no calibration has been calculated.", flush=True)
            if key in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if opened:
            cv2.destroyAllWindows()
        print(f"Saved {count} images. Manifest: {args.output / 'session.json'}", flush=True)


if __name__ == "__main__":
    main()
