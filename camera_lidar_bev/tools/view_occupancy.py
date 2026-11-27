#!/usr/bin/python3
"""Display the C++ stationary occupancy image with nearest-cell scaling."""

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/camera_lidar_bev/occupancy_image')
    parser.add_argument('--timeout', type=float, default=15.0)
    args = parser.parse_args()
    if not 0 < args.timeout <= 30:
        parser.error('timeout must be in (0, 30] seconds')
    import cv2
    import numpy as np
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.executors import ExternalShutdownException
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node('stationary_occupancy_view')
    bridge = CvBridge()
    latest = None
    receipt = 0.0
    sequence = 0
    shown = -1
    window = 'Stationary occupancy | green observed ground, red obstacle, gray unknown'

    def receive(message):
        nonlocal latest, receipt, sequence
        if message.header.frame_id != 'os_sensor':
            return
        image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        if image.shape != (200, 150, 3):
            node.get_logger().error('Occupancy image must match the configured 40x30m / 0.20m grid')
            return
        latest = image.copy()
        receipt = time.monotonic()
        sequence += 1

    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
    subscription = node.create_subscription(Image, args.topic, receive, qos)
    started = time.monotonic()
    opened = False
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.02)
            if latest is None:
                if time.monotonic() - started > args.timeout:
                    raise RuntimeError('No occupancy image; start the ground processor in the same ROS domain')
                continue
            if not opened:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, 620, 910)
                opened = True
            stale = time.monotonic() - receipt > 1.0
            if sequence != shown or stale:
                cells = np.full_like(latest, 48) if stale else latest
                raster = cv2.resize(cells, (600, 800), interpolation=cv2.INTER_NEAREST)
                display = np.zeros((910, 620, 3), dtype=np.uint8)
                display[100:900, 10:610] = raster
                lines = ['Stationary os_sensor grid | forward up, left left',
                         '40 m forward x 30 m wide | cells 0.20 m',
                         'Green: observed ground | Red: obstacle | Gray: unknown',
                         'SOURCE STALE: view cleared' if stale else 'Approx. height 1.8 m | Q/Esc close']
                for text, y in zip(lines, (20, 43, 66, 89)):
                    scale = min(.45, .45 * 600 / max(1, cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]))
                    cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                                (0, 190, 255) if stale and y == 89 else (230, 230, 230), 1, cv2.LINE_AA)
                cv2.imshow(window, display)
                shown = sequence
            if cv2.waitKey(1) & 0xff in (ord('q'), ord('Q'), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        return 0
    except (KeyboardInterrupt, ExternalShutdownException):
        return 0
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if opened:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    raise SystemExit(main())
