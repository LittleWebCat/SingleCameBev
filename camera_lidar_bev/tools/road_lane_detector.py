#!/usr/bin/env python3
"""Publish native-image road and lane masks with the pinned TwinLiteNet engine.

Run with the workspace .venv Python after sourcing ROS Jazzy. Input is the raw
camera image, without undistortion; masks retain its exact source header. This
node estimates image segmentation only. It does not measure a road boundary or
perform camera/LiDAR geometry. TensorRT buffers and a non-default CUDA stream
are allocated once, and callbacks retain only the latest image.
"""

import argparse
import math
from pathlib import Path
import time

import cv2
import numpy as np


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def image_bgr(message):
    """Decode an 8-bit ROS image including row padding, without cv_bridge."""
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1}
    if message.encoding not in channels:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    count = channels[message.encoding]
    if (message.height < 1 or message.width < 1 or
            message.step < message.width * count or
            len(message.data) < message.step * message.height):
        raise ValueError("Camera image dimensions, row stride or payload are invalid")
    array = np.ndarray((message.height, message.width, count), np.uint8,
                       buffer=memoryview(message.data),
                       strides=(message.step, count, 1))
    conversions = {"rgb8": cv2.COLOR_RGB2BGR, "bgra8": cv2.COLOR_BGRA2BGR,
                   "rgba8": cv2.COLOR_RGBA2BGR, "mono8": cv2.COLOR_GRAY2BGR}
    if message.encoding == "mono8":
        array = array[:, :, 0]
    return cv2.cvtColor(array, conversions[message.encoding]) if message.encoding in conversions else array


class TwinLiteRuntime:
    """Static TensorRT FLOAT I/O contract, FP16 internal engine computation."""

    def __init__(self, engine_path):
        import torch
        import tensorrt as trt

        self.torch = torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use the workspace .venv Python")
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, "")
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create TensorRT execution context")
        self.input_name = "image"
        self.output_names = ("drivable_logits", "lane_logits")
        expected = {self.input_name: (1, 3, 360, 640),
                    self.output_names[0]: (1, 2, 360, 640),
                    self.output_names[1]: (1, 2, 360, 640)}
        actual = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        if actual != set(expected):
            raise RuntimeError(f"Unexpected TwinLiteNet tensor names: {sorted(actual)}")
        for name, shape in expected.items():
            if (tuple(self.engine.get_tensor_shape(name)) != shape or
                    self.engine.get_tensor_dtype(name) != trt.float32):
                raise RuntimeError(f"Unexpected TwinLiteNet tensor shape/dtype: {name}")
            expected_mode = trt.TensorIOMode.INPUT if name == self.input_name else trt.TensorIOMode.OUTPUT
            if self.engine.get_tensor_mode(name) != expected_mode:
                raise RuntimeError(f"Unexpected TwinLiteNet tensor mode: {name}")
        self.stream = torch.cuda.Stream()
        self.input_host = torch.empty(expected[self.input_name], dtype=torch.float32, pin_memory=True)
        self.input_array = self.input_host.numpy()
        self.input_cuda = torch.empty_like(self.input_host, device="cuda")
        self.logits = [torch.empty(expected[name], dtype=torch.float32, device="cuda")
                       for name in self.output_names]
        self.argmax_cuda = torch.empty((1, 360, 640), dtype=torch.int64, device="cuda")
        self.mask_cuda = torch.empty((360, 640), dtype=torch.uint8, device="cuda")
        self.masks_host = [torch.empty((360, 640), dtype=torch.uint8, pin_memory=True) for _ in self.output_names]
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        for name, tensor in zip((self.input_name,) + self.output_names, [self.input_cuda] + self.logits):
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"Cannot assign TensorRT tensor buffer: {name}")

    def infer(self, bgr):
        started = time.perf_counter()
        height, width = bgr.shape[:2]
        # Official preprocessing: direct BGR -> RGB resize, CHW /255. No
        # letterboxing, ImageNet normalization, cropping or undistortion.
        resized = cv2.resize(bgr, (640, 360), interpolation=cv2.INTER_LINEAR)
        np.copyto(self.input_array[0], resized[:, :, ::-1].transpose(2, 0, 1), casting="unsafe")
        self.input_array *= 1.0 / 255.0
        torch = self.torch
        with torch.cuda.stream(self.stream):
            self.input_cuda.copy_(self.input_host, non_blocking=True)
            self.start_event.record(self.stream)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TwinLiteNet TensorRT inference failed")
            self.end_event.record(self.stream)
            for logits, host_mask in zip(self.logits, self.masks_host):
                torch.argmax(logits, dim=1, out=self.argmax_cuda)
                self.mask_cuda.copy_(self.argmax_cuda[0])
                self.mask_cuda.mul_(255)
                host_mask.copy_(self.mask_cuda, non_blocking=True)
        self.stream.synchronize()
        masks = [cv2.resize(mask.numpy(), (width, height), interpolation=cv2.INTER_NEAREST)
                 for mask in self.masks_host]
        return masks, (time.perf_counter() - started) * 1000.0, self.start_event.elapsed_time(self.end_event)


def arguments():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=root / "models/road_lanes_001/twinlitenet_fp16.engine")
    parser.add_argument("--fps", type=float, default=10.0, help="Maximum processed images per wall-clock second, up to 10")
    parser.add_argument("--stale-after", type=float, default=0.5)
    parser.add_argument("--image-topic", default="/camera/front/image_raw")
    parser.add_argument("--road-topic", default="/camera_lidar_bev/road_mask")
    parser.add_argument("--lane-topic", default="/camera_lidar_bev/lane_mask")
    parser.add_argument("--diagnostics-topic", default="/camera_lidar_bev/diagnostics")
    args, ros_args = parser.parse_known_args()
    if not math.isfinite(args.fps) or not 0 < args.fps <= 10:
        parser.error("--fps must be positive, finite and no greater than 10")
    if not math.isfinite(args.stale_after) or args.stale_after <= 0:
        parser.error("--stale-after must be positive and finite")
    if not args.engine.is_file():
        parser.error(f"TwinLiteNet engine is missing: {args.engine}")
    return args, ros_args


def main():
    args, ros_args = arguments()
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from rclpy.clock import Clock
    from rclpy.clock_type import ClockType
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image

    rclpy.init(args=ros_args)
    node = None
    try:
        model = TwinLiteRuntime(args.engine)

        class RoadLaneDetector(Node):
            def __init__(self):
                super().__init__("road_lane_detector")
                qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                                 durability=DurabilityPolicy.VOLATILE)
                self.road_publisher = self.create_publisher(Image, args.road_topic, qos)
                self.lane_publisher = self.create_publisher(Image, args.lane_topic, qos)
                self.diagnostic_publisher = self.create_publisher(DiagnosticArray, args.diagnostics_topic, 5)
                self.subscription = self.create_subscription(Image, args.image_topic, self.receive, qos)
                self.latest = None
                self.last_key = None
                self.last_shape = None
                self.last_frame = ""
                self.cleared = True
                self.received = self.processed = self.skipped = self.errors = 0
                self.runtime_ms = self.engine_ms = 0.0
                self.input_stamp = 0
                self.input_age_ms = 0.0
                self.road_pixels = self.lane_pixels = 0
                self.state = "waiting for camera"
                self.level = DiagnosticStatus.WARN
                self.last_error_log = -math.inf
                # Steady timers keep the processing cap and stale clearing
                # effective when playback is slowed, paused or looping.
                self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
                self.timer = self.create_timer(1.0 / args.fps, self.update, clock=self.steady_clock)
                self.diagnostic_timer = self.create_timer(1.0, self.diagnostics, clock=self.steady_clock)
                self.get_logger().info(
                    f"TwinLiteNet ready; native-image masks at up to {args.fps:g} Hz from {args.image_topic}")

            def receive(self, message):
                self.received += 1
                if self.latest is not None:
                    old_message = self.latest[0]
                    if (stamp_ns(old_message.header), old_message.header.frame_id) != self.last_key:
                        self.skipped += 1
                self.latest = message, time.monotonic()

            def fresh(self, record):
                message, received_at = record
                receive_age = time.monotonic() - received_at
                source_age = (self.get_clock().now().nanoseconds - stamp_ns(message.header)) / 1e9
                self.input_age_ms = source_age * 1000.0
                return (0 <= receive_age <= args.stale_after and
                        -0.05 <= source_age <= args.stale_after)

            def mask_message(self, mask, header):
                output = Image()
                output.header = header
                output.height, output.width = mask.shape
                output.encoding = "mono8"
                output.is_bigendian = 0
                output.step = output.width
                output.data = np.ascontiguousarray(mask).tobytes()
                return output

            def clear(self):
                if not self.cleared and self.last_shape is not None:
                    blank = np.zeros(self.last_shape, np.uint8)
                    output = Image()
                    output.header.stamp = self.get_clock().now().to_msg()
                    output.header.frame_id = self.last_frame
                    clear_message = self.mask_message(blank, output.header)
                    self.road_publisher.publish(clear_message)
                    self.lane_publisher.publish(clear_message)
                self.cleared = True
                self.road_pixels = self.lane_pixels = 0

            def update(self):
                if self.latest is None or not self.fresh(self.latest):
                    self.state = "camera input stale" if self.latest is not None else "waiting for camera"
                    self.level = DiagnosticStatus.WARN
                    self.clear()
                    return
                record = self.latest
                message = record[0]
                key = stamp_ns(message.header), message.header.frame_id
                if key == self.last_key:
                    return
                self.last_key = key
                self.input_stamp, self.last_frame = key
                try:
                    bgr = image_bgr(message)
                    self.last_shape = bgr.shape[:2]
                    masks, self.runtime_ms, self.engine_ms = model.infer(bgr)
                    if not self.fresh(record):
                        self.skipped += 1
                        self.state = "prediction expired before publication"
                        self.level = DiagnosticStatus.WARN
                        self.clear()
                        return
                    self.road_pixels, self.lane_pixels = (int(np.count_nonzero(mask)) for mask in masks)
                    self.road_publisher.publish(self.mask_message(masks[0], message.header))
                    self.lane_publisher.publish(self.mask_message(masks[1], message.header))
                    self.processed += 1
                    self.cleared = False
                    self.state = "predicting road and lane masks"
                    self.level = DiagnosticStatus.OK
                except Exception as error:
                    self.errors += 1
                    self.state = f"segmentation error: {error}"
                    self.level = DiagnosticStatus.ERROR
                    self.clear()
                    now = time.monotonic()
                    if now - self.last_error_log >= 5.0:
                        self.get_logger().error(self.state)
                        self.last_error_log = now

            def diagnostics(self):
                status = DiagnosticStatus()
                status.name = "road_lane_detector"
                status.hardware_id = "front_camera/TwinLiteNet"
                status.level, status.message = self.level, self.state
                height, width = self.last_shape or (0, 0)
                pixels = max(1, height * width)
                values = {
                    "model": "TwinLiteNet", "backend": "TensorRT FP16", "engine_file": str(args.engine),
                    "image_width": width, "image_height": height, "input_stamp_ns": self.input_stamp,
                    "input_frame": self.last_frame, "runtime_ms": f"{self.runtime_ms:.3f}",
                    "engine_ms": f"{self.engine_ms:.3f}", "source_age_ms": f"{self.input_age_ms:.3f}",
                    "fps_limit": args.fps, "received": self.received, "processed": self.processed,
                    "skipped": self.skipped, "errors": self.errors,
                    "road_pixels": self.road_pixels, "lane_pixels": self.lane_pixels,
                    "road_fraction": f"{self.road_pixels / pixels:.6f}",
                    "lane_fraction": f"{self.lane_pixels / pixels:.6f}",
                    "cleared": str(self.cleared).lower(), "accuracy_validated": "false",
                }
                status.values = [KeyValue(key=key, value=str(value)) for key, value in values.items()]
                report = DiagnosticArray()
                report.header.stamp = self.get_clock().now().to_msg()
                report.status = [status]
                self.diagnostic_publisher.publish(report)

        node = RoadLaneDetector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
