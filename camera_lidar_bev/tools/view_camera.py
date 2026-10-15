#!/usr/bin/env python3
"""Show ROS camera images using the publisher's sensor-data QoS."""

import argparse
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from camera_preview import load_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/camera/front/image_raw")
    parser.add_argument("--snapshot", type=Path, help="Save one received frame and exit without opening a window")
    parser.add_argument("--timeout", type=float, default=10.0, help="Initial-frame timeout in seconds")
    parser.add_argument("--checkerboard", nargs=2, type=int, metavar=("COLUMNS", "ROWS"),
                        help="Overlay detected inner corners, e.g. --checkerboard 5 4")
    parser.add_argument("--calibration-file", type=Path,
                        help="Show raw and undistorted candidate views; does not apply calibration to ROS")
    args = parser.parse_args()
    if args.timeout <= 0.0:
        parser.error("--timeout must be positive")
    if args.checkerboard and min(args.checkerboard) < 2:
        parser.error("checkerboard inner-corner dimensions must be at least 2")
    if args.checkerboard and args.calibration_file:
        parser.error("Choose either --checkerboard or --calibration-file")
    calibration = None
    if args.calibration_file:
        try:
            calibration = load_calibration(args.calibration_file)
        except ValueError as error:
            parser.error(str(error))
    rclpy.init()
    node = rclpy.create_node("front_camera_view")
    bridge = CvBridge()
    latest = None
    latest_received = 0.0
    sequence = 0
    shown_sequence = -1
    window = "Checkerboard — 5x4 inner corners" if args.checkerboard == [5, 4] else "Front Camera — ROS"
    if calibration:
        window = "Front Camera — Raw / Candidate Undistortion (unvalidated)"
    opened = False
    shown_preview = None

    def receive(message):
        nonlocal latest, latest_received, sequence
        try:
            latest = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            latest_received = time.monotonic()
            sequence += 1
        except Exception as error:
            node.get_logger().error(f"Cannot decode camera image: {error}", throttle_duration_sec=5.0)

    subscription = node.create_subscription(Image, args.topic, receive, qos_profile_sensor_data)
    started = time.monotonic()
    try:
        print(f"Waiting for {args.topic} (sensor-data Best Effort QoS)", flush=True)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            if latest is None:
                if time.monotonic() - started > args.timeout:
                    raise RuntimeError(f"No image received on {args.topic}; check the camera bridge and ROS_DOMAIN_ID")
                continue
            if args.snapshot is not None:
                snapshot = calibration.comparison(latest) if calibration else latest
                if not cv2.imwrite(str(args.snapshot), snapshot):
                    raise RuntimeError(f"Could not save image to {args.snapshot}")
                print(f"Saved {snapshot.shape[1]}x{snapshot.shape[0]} BGR image to {args.snapshot}", flush=True)
                return
            if not opened:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                if calibration:
                    cv2.resizeWindow(window, 1280, 516)
                else:
                    cv2.resizeWindow(window, 960, 720)
                opened = True
            if sequence != shown_sequence:
                preview = calibration.comparison(latest) if calibration else latest
                if args.checkerboard:
                    preview = latest.copy()
                    pattern = tuple(args.checkerboard)
                    gray = cv2.cvtColor(latest, cv2.COLOR_BGR2GRAY)
                    found, corners = cv2.findChessboardCornersSB(
                        gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
                    if found:
                        cv2.drawChessboardCorners(preview, pattern, corners, found)
                    label = (f"DETECTED: {pattern[0]}x{pattern[1]} = {pattern[0]*pattern[1]} corners"
                             if found else "BOARD NOT DETECTED: show the entire checkerboard")
                    cv2.putText(preview, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0) if found else (0, 0, 255), 1)
                cv2.imshow(window, preview)
                shown_preview = preview
                shown_sequence = sequence
            elif time.monotonic() - latest_received > 1.0:
                # Make stopped input visible rather than presenting an old frame
                # as live. The stored source image is never modified.
                stale = shown_preview.copy() if calibration else latest.copy()
                cv2.putText(stale, "CAMERA INPUT STALE", (15, 70 if calibration else 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(window, stale)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if opened:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
