#!/usr/bin/env python3
"""Exercise the C++ camera bridge with test images; no camera hardware is used.

Run after sourcing /opt/ros/jazzy/setup.bash and this workspace's install/setup.bash:
    /usr/bin/python3 tools/smoke_test_camera_bridge.py

The YAML below is a test fixture, not a calibration for a physical sensor.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

# Keep test traffic local and separate from the project's normal sensor domain.
os.environ["ROS_DOMAIN_ID"] = "188"
os.environ.pop("ROS_LOCALHOST_ONLY", None)
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
os.environ.setdefault("ROS_LOG_DIR", "/tmp/camera_lidar_bev-camera-smoke-logs")

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


WIDTH = 96
HEIGHT = 64
FRAME_ID = "camera_front_test_optical_frame"
EXPECTED_K = [80.0, 0.0, 47.5, 0.0, 81.0, 31.5, 0.0, 0.0, 1.0]
EXPECTED_D = [0.01, -0.02, 0.001, -0.001, 0.0]
EXPECTED_R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
EXPECTED_P = [80.0, 0.0, 47.5, 0.0, 0.0, 81.0, 31.5, 0.0, 0.0, 0.0, 1.0, 0.0]


def log_tail(log):
    log.seek(0)
    return log.read().decode("utf-8", errors="replace")[-4000:]


@contextmanager
def bridge_process(arguments):
    """Capture output without pipe backpressure, and always stop the child."""
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(arguments, stdout=log, stderr=subprocess.STDOUT)
        try:
            yield process, log
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=4)


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def header_key(message):
    return (message.header.stamp.sec, message.header.stamp.nanosec, message.header.frame_id)


def diagnostic_values(status):
    return {item.key: item.value for item in status.values}


def check_image(image, info, calibration_expected):
    assert image.header == info.header, "Image and CameraInfo headers differ"
    assert image.header.frame_id == FRAME_ID, "Unexpected optical frame"
    assert stamp_ns(image.header) > 0, "Zero or invalid image timestamp"
    assert (image.width, image.height) == (WIDTH, HEIGHT)
    assert (info.width, info.height) == (WIDTH, HEIGHT)
    assert image.encoding == "bgr8", f"Unexpected encoding: {image.encoding}"
    assert image.is_bigendian == 0
    assert image.step == WIDTH * 3
    payload = bytes(image.data)
    assert len(payload) == image.step * HEIGHT, "Incorrect image payload length"
    assert payload == bytes((payload[0], 64, 128)) * (WIDTH * HEIGHT), (
        "Synthetic frame is not the documented constant BGR image"
    )
    if calibration_expected:
        assert info.distortion_model == "plumb_bob"
        assert list(info.k) == EXPECTED_K, "Camera matrix was not loaded from YAML"
        assert list(info.d) == EXPECTED_D, "Distortion was not loaded from YAML"
        assert list(info.r) == EXPECTED_R, "Rectification matrix differs from YAML"
        assert list(info.p) == EXPECTED_P, "Projection matrix differs from YAML"
    else:
        assert info.k[0] == 0.0, "Uncalibrated or mismatched frames claim valid intrinsics"


def run_synthetic_case(executable, case, calibration_file="", expected_level=DiagnosticStatus.WARN):
    namespace = f"/bev_camera_bridge_test/{case}"
    image_topic = f"{namespace}/image_raw"
    info_topic = f"{namespace}/camera_info"
    node = rclpy.create_node(f"camera_bridge_smoke_{case}")
    images = {}
    infos = {}
    reports = []
    subscriptions = [
        node.create_subscription(Image, image_topic,
                                 lambda msg: images.__setitem__(header_key(msg), msg),
                                 qos_profile_sensor_data),
        node.create_subscription(CameraInfo, info_topic,
                                 lambda msg: infos.__setitem__(header_key(msg), msg),
                                 qos_profile_sensor_data),
        node.create_subscription(DiagnosticArray, f"{namespace}/diagnostics", reports.append, 5),
    ]
    arguments = [str(executable), "--ros-args", "-r", f"__ns:={namespace}"]
    for name, value in {
        "source_mode": "synthetic",
        "width": WIDTH,
        "height": HEIGHT,
        "fps": 20.0,
        "frame_id": FRAME_ID,
        "image_topic": image_topic,
        "camera_info_topic": info_topic,
        "use_sim_time": "false",
    }.items():
        arguments += ["-p", f"{name}:={value}"]
    # Omit the empty value to exercise the uncalibrated default without relying
    # on command-line YAML handling of empty scalar strings.
    if calibration_file:
        arguments += ["-p", f"calibration_file:={calibration_file}"]
    checked = 0
    previous_stamp = 0
    previous_blue = None
    diagnostic_ok = False
    try:
        with bridge_process(arguments) as (process, log):
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"{case}: bridge exited with {process.returncode}\n{log_tail(log)}")
                rclpy.spin_once(node, timeout_sec=0.05)
                for key in sorted(images.keys() & infos.keys()):
                    image = images.pop(key)
                    info = infos.pop(key)
                    check_image(image, info, calibration_expected=(case == "calibrated"))
                    current_stamp = stamp_ns(image.header)
                    assert current_stamp > previous_stamp, "Frame timestamps did not increase"
                    now_ns = node.get_clock().now().nanoseconds
                    assert -100_000_000 <= now_ns - current_stamp < 3_000_000_000, (
                        "Image timestamp is not a recent host ROS timestamp"
                    )
                    blue = image.data[0]
                    if previous_blue is not None:
                        # DDS may drop sensor-data frames; sequence gaps are valid.
                        assert 0 < (blue - previous_blue) % 256 < 128, (
                            "Synthetic sequence did not advance"
                        )
                    previous_stamp = current_stamp
                    previous_blue = blue
                    checked += 1
                for report in reports:
                    for status in report.status:
                        values = diagnostic_values(status)
                        if values.get("timestamp_source") == "host_receipt_after_read":
                            assert values.get("exposure_timestamps_available") == "false", (
                                "Bridge claims an exposure timestamp for host-stamped input"
                            )
                            assert values.get("source_mode") == "synthetic"
                            published = int(values.get("frames_published", "-1"))
                            assert published >= 0, "Missing or invalid publication count"
                            if published > 0:
                                expected_calibrated = "true" if case == "calibrated" else "false"
                                assert values.get("calibrated") == expected_calibrated
                            if published > 0 and status.level == expected_level:
                                diagnostic_ok = True
                reports.clear()
                if checked >= 6 and diagnostic_ok:
                    # In the mismatch case these successful frames also prove
                    # that the calibration error did not stop image publication.
                    print(f"PASS {case}: {checked} matched frames and expected diagnostics")
                    return
            raise AssertionError(
                f"{case}: timed out; matched_frames={checked}, diagnostics_ok={diagnostic_ok}\n"
                f"{log_tail(log)}"
            )
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()


def require_bounded_exit(arguments, expected_success, description):
    with bridge_process(arguments) as (process, log):
        try:
            returncode = process.wait(timeout=8.0)
        except subprocess.TimeoutExpired as error:
            raise AssertionError(f"{description}: bridge did not exit within 8 seconds") from error
        assert returncode >= 0, f"{description}: bridge terminated by a signal\n{log_tail(log)}"
        assert (returncode == 0) == expected_success, (
            f"{description}: unexpected exit code {returncode}\n{log_tail(log)}"
        )
    print(f"PASS {description}")


def write_calibration(path, width=WIDTH):
    path.write_text(
        f"""image_width: {width}
image_height: 64
camera_name: synthetic_front_test
camera_matrix:
  rows: 3
  cols: 3
  data: [80.0, 0.0, 47.5, 0.0, 81.0, 31.5, 0.0, 0.0, 1.0]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [0.01, -0.02, 0.001, -0.001, 0.0]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
projection_matrix:
  rows: 3
  cols: 4
  data: [80.0, 0.0, 47.5, 0.0, 0.0, 81.0, 31.5, 0.0, 0.0, 0.0, 1.0, 0.0]
""",
        encoding="utf-8",
    )


def main():
    root = Path(__file__).resolve().parents[1]
    executable = root / "install/bev_sensor_bridge/lib/bev_sensor_bridge/camera_bridge"
    if not executable.is_file():
        raise RuntimeError("Build bev_sensor_bridge before running the camera bridge smoke test")
    with tempfile.TemporaryDirectory(prefix="bev-camera-smoke-") as temporary:
        directory = Path(temporary)
        calibration = directory / "camera.yaml"
        mismatch = directory / "wrong_dimensions.yaml"
        write_calibration(calibration)
        write_calibration(mismatch, width=WIDTH + 1)
        rclpy.init()
        try:
            run_synthetic_case(executable, "uncalibrated")
            # Loading intrinsics does not validate exposure timing; host-stamped
            # synthetic input retains WARN even when its test YAML matches.
            run_synthetic_case(executable, "calibrated", str(calibration), DiagnosticStatus.WARN)
            run_synthetic_case(executable, "mismatch", str(mismatch), DiagnosticStatus.ERROR)
            require_bounded_exit(
                [str(executable), "--ros-args", "-r", "__ns:=/bev_camera_bridge_test/finite",
                 "-p", "source_mode:=synthetic", "-p", "width:=96", "-p", "height:=64",
                 "-p", "fps:=20.0", "-p", "max_frames:=3"],
                expected_success=True,
                description="max_frames stops synthetic capture cleanly",
            )
            require_bounded_exit(
                [str(executable), "--ros-args", "-r", "__ns:=/bev_camera_bridge_test/missing",
                 "-p", "source_mode:=camera", "-p", "device:=/dev/bev-camera-smoke-missing-device"],
                expected_success=False,
                description="missing camera device fails cleanly",
            )
            require_bounded_exit(
                [str(executable), "--ros-args", "-r", "__ns:=/bev_camera_bridge_test/sim_clock",
                 "-p", "source_mode:=synthetic", "-p", "use_sim_time:=true"],
                expected_success=False,
                description="simulated clock is rejected for host-stamped capture",
            )
        finally:
            if rclpy.ok():
                rclpy.shutdown()
    print("Camera bridge smoke test passed; no physical sensor calibration or timing was validated")


if __name__ == "__main__":
    main()
