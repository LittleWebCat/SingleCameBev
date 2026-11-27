#!/usr/bin/env python3
"""Show the cameras' annotated lane images side by side: camera lane pixels (blue tint), camera lane lines and
road edge (blue / violet), and the LiDAR's lane lines and road edge projected into each image (orange / red).

Images come from /camera_lidar_bev/lanes/camera_<name>/image (tools/lane_camera.py --annotate-hz). q or Esc quits.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/lane_view.py [--cameras front_wide front_narrow]
"""
import argparse

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cameras', nargs='+', default=['front_wide', 'front_narrow'])
    parser.add_argument('--height', type=int, default=360, help='height of each tile in the window')
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('lane_view')
    latest = {}
    for c in args.cameras:
        node.create_subscription(Image, f'/camera_lidar_bev/lanes/camera_{c}/image',
                                 lambda m, c=c: latest.__setitem__(c, m), qos_profile_sensor_data)
    window = 'Lanes: camera (blue) vs LiDAR (orange)'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            tiles = []
            for c in args.cameras:
                m = latest.get(c)
                if m is None:
                    tile = np.full((args.height, args.height * 4 // 3, 3), 30, np.uint8)
                    cv2.putText(tile, f'waiting for {c}', (10, args.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                else:
                    img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
                    tile = cv2.resize(img, (round(m.width * args.height / m.height), args.height), interpolation=cv2.INTER_AREA)
                tiles.append(tile)
            cv2.imshow(window, np.hstack(tiles))
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
