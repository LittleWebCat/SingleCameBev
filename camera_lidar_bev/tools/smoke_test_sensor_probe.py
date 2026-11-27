#!/usr/bin/env python3
"""Exercise the installed C++ probe with synthetic data in a separate ROS domain."""

import os
from pathlib import Path
import signal
import struct
import subprocess
import time

# The test uses a separate domain and test-only topics, without hardware drivers.
os.environ["ROS_DOMAIN_ID"] = "187"
os.environ.pop("ROS_LOCALHOST_ONLY", None)
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
os.environ.setdefault("ROS_LOG_DIR", "/tmp/camera_lidar_bev-smoke-logs")

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2, PointField


def main():
    root = Path(__file__).resolve().parents[1]
    executable = root / "install/bev_sensor_bridge/lib/bev_sensor_bridge/sensor_probe"
    if not executable.is_file():
        raise RuntimeError("Build the project before running the probe smoke test")
    topics = {
        "lidar_topic": "/bev_probe_test/lidar",
        "camera_topic": "/bev_probe_test/image",
        "camera_info_topic": "/bev_probe_test/info",
        "imu_topic": "/bev_probe_test/imu",
        "odom_topic": "/bev_probe_test/odom",
    }
    arguments = [str(executable), "--ros-args", "-r", "__ns:=/bev_probe_test"]
    for parameter, topic in topics.items():
        arguments += ["-p", f"{parameter}:={topic}"]
    process = subprocess.Popen(arguments)
    node = None
    try:
        rclpy.init()
        node = rclpy.create_node("probe_smoke_test", namespace="bev_probe_test")
        reports = []
        subscription = node.create_subscription(
            DiagnosticArray, "/bev_probe_test/diagnostics", reports.append, 5
        )
        publishers = [
            node.create_publisher(message_type, topics[parameter],
                                  QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
                                  if parameter == "lidar_topic" else qos_profile_sensor_data)
            for message_type, parameter in (
                (PointCloud2, "lidar_topic"), (Image, "camera_topic"),
                (CameraInfo, "camera_info_topic"), (Imu, "imu_topic"),
                (Odometry, "odom_topic"),
            )
        ]

        def receive_until(predicate, publish=None, timeout_s=8.0):
            reports.clear()
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Probe exited early with code {process.returncode}")
                if publish:
                    publish()
                rclpy.spin_once(node, timeout_sec=0.05)
                if reports:
                    statuses = {item.name.split("/")[-1]: item for item in reports[-1].status}
                    if predicate(statuses):
                        return statuses
            raise AssertionError("Expected diagnostics did not arrive")

        receive_until(lambda states: states["lidar"].level == DiagnosticStatus.WARN
                      and "Waiting" in states["lidar"].message)

        def publish_inputs(with_point_time=False, future=False, zero_stamp=False, malformed=False):
            stamp = node.get_clock().now().to_msg()
            if future:
                stamp.sec += 5
            if zero_stamp:
                stamp.sec = stamp.nanosec = 0
            cloud = PointCloud2()
            cloud.header.stamp = stamp
            cloud.header.frame_id = "os_lidar"
            cloud.width = cloud.height = 1
            cloud.fields = [
                PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
                for index, name in enumerate(("x", "y", "z"))
            ]
            if with_point_time:
                cloud.fields.append(PointField(name="t", offset=12, datatype=PointField.UINT32, count=1))
                cloud.data = struct.pack("<fffI", 5.0, 0.0, 1.0, 1000)
                cloud.point_step = 16
            else:
                cloud.data = struct.pack("<fff", 5.0, 0.0, 1.0)
                cloud.point_step = 12
            cloud.row_step = cloud.point_step
            if malformed:
                cloud.data = b""
            image = Image()
            image.header.stamp = stamp
            image.header.frame_id = "camera_front_optical_frame"
            image.width = image.height = image.step = 1
            image.encoding = "mono8"
            image.data = [0]
            info = CameraInfo()
            info.header = image.header
            info.width = info.height = 1
            info.k = [500.0, 0.0, 0.0, 0.0, 500.0, 0.0, 0.0, 0.0, 1.0]
            imu = Imu()
            imu.header.stamp = stamp
            imu.header.frame_id = "imu_link"
            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = "odom"
            for publisher, message in zip(publishers, (cloud, image, info, imu, odom)):
                publisher.publish(message)

        def values(status):
            return {item.key: item.value for item in status.values}

        receive_until(lambda states: states["lidar"].level == DiagnosticStatus.WARN
                      and values(states["lidar"]).get("point_time_field_present") == "false",
                      publish_inputs)
        valid = receive_until(lambda states: all(states[name].level == DiagnosticStatus.OK
                              for name in ("lidar", "front_camera", "camera_info", "imu", "odometry")),
                              lambda: publish_inputs(with_point_time=True))
        assert valid["calibration_and_fusion"].level == DiagnosticStatus.WARN
        assert values(valid["lidar"])["point_time_units_and_origin"].startswith("unverified")
        receive_until(lambda states: states["lidar"].level == DiagnosticStatus.ERROR
                      and "payload" in states["lidar"].message,
                      lambda: publish_inputs(with_point_time=True, malformed=True))
        receive_until(lambda states: states["lidar"].level == DiagnosticStatus.WARN
                      and "future" in states["lidar"].message,
                      lambda: publish_inputs(with_point_time=True, future=True))
        receive_until(lambda states: states["lidar"].level == DiagnosticStatus.WARN
                      and "zero/invalid" in states["lidar"].message,
                      lambda: publish_inputs(with_point_time=True, zero_stamp=True))
        receive_until(lambda states: states["lidar"].level == DiagnosticStatus.ERROR
                      and "stale" in states["lidar"].message)
        print("Probe smoke test passed: missing input, point timing, malformed payload, clocks, stale input, and pending calibration")
        # Keep the subscription referenced until the node has completed its work.
        del subscription
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
