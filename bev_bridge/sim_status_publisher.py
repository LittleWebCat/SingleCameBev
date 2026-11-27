#!/usr/bin/env python3
"""Publish simulated path-tracking status so the bridge can be checked without the tracker.

Sends JSON on /camera_lidar_bev/path_tracking/status in the shape bridge_core.InputGate accepts,
with a slow sinusoidal steering target. Engage heartbeats are opt-in (--engage): False first
(the bridge requires a release), then True. Only use --engage against a --dry-run bridge.
"""
import argparse
import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Bool


class SimStatus(Node):
    def __init__(self, args):
        super().__init__('sim_path_tracking_status')
        self.a = args
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.status_pub = self.create_publisher(String, args.topic, qos)
        self.engage_pub = self.create_publisher(Bool, args.engage_topic, qos) if args.engage else None
        self.create_subscription(String, args.bridge_status_topic, self.on_bridge_status, 10)
        self.start = time.monotonic()
        self.sent = 0
        self.last_bridge_state = None
        self.create_timer(1/args.rate, self.tick)

    def tick(self):
        t = time.monotonic()-self.start
        steering = self.a.steer_amp*math.sin(2*math.pi*t/self.a.steer_period)
        status = dict(
            tracking_valid=True, targets_valid=True, vehicle_transform_verified=True, diagnostic_only=True,
            source_mode='live_localization', frame='enu', pose_reference='rear_axle', state='tracking',
            source_stamp_ns=self.get_clock().now().nanoseconds,
            steering_rad=steering, curvature_per_m=math.tan(steering)/self.a.wheelbase,
            target_speed_mps=self.a.speed, estimated_speed_mps=self.a.speed)
        self.status_pub.publish(String(data=json.dumps(status)))
        self.sent += 1
        if self.engage_pub is not None:
            # Release (False) for the first second, then hold the deadman (True).
            self.engage_pub.publish(Bool(data=t >= self.a.release_s))
        if self.sent % int(self.a.rate) == 0:
            self.get_logger().info(f'sent {self.sent} status msgs; steering_rad={steering:+.3f}; '
                                   f'bridge state={self.last_bridge_state}')

    def on_bridge_status(self, message):
        try:
            self.last_bridge_state = json.loads(message.data).get('state')
        except ValueError:
            self.last_bridge_state = 'unparseable'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topic', default='/camera_lidar_bev/path_tracking/status')
    parser.add_argument('--engage-topic', default='/bev_dbw/engage')
    parser.add_argument('--bridge-status-topic', default='/bev_dbw/bridge_status')
    parser.add_argument('--rate', type=float, default=20.)
    parser.add_argument('--speed', type=float, default=0.3, help='target/estimated speed, m/s')
    parser.add_argument('--steer-amp', type=float, default=0.1, help='road-wheel steering amplitude, rad')
    parser.add_argument('--steer-period', type=float, default=6., help='seconds per steering cycle')
    parser.add_argument('--wheelbase', type=float, default=2.70)
    parser.add_argument('--engage', action='store_true', help='also publish engage heartbeats (dry-run bridge only)')
    parser.add_argument('--release-s', type=float, default=1.)
    args = parser.parse_args()
    rclpy.init()
    node = SimStatus(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
