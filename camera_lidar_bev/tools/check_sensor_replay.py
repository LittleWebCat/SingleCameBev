#!/usr/bin/env python3
"""Verify a partial sensor recording with the installed probe on localhost domain 42."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be finite and positive")
    return seconds


def diagnostic_level(value):
    """ROS byte fields can deserialize as a one-byte bytes object or an int."""
    if isinstance(value, (bytes, bytearray)):
        if len(value) != 1:
            raise ValueError(f"expected one diagnostic level byte, received {len(value)}")
        return value[0]
    return int(value)


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for stop_signal, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
            try:
                os.killpg(process.pid, stop_signal)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                pass
    finally:
        signal.signal(signal.SIGINT, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="recorded bag directory containing metadata.yaml")
    parser.add_argument("--output", type=Path, help="optional new JSON report path")
    parser.add_argument("--timeout", type=positive_seconds, default=30.0, help="overall check timeout in seconds (default: 30)")
    args = parser.parse_args()
    bag = args.bag.expanduser().resolve()
    root = Path(__file__).resolve().parents[1]
    executable = root / "install/bev_sensor_bridge/lib/bev_sensor_bridge/sensor_probe"
    if not (bag / "metadata.yaml").is_file():
        parser.error("bag must contain metadata.yaml")
    if not executable.is_file() or shutil.which("ros2") is None:
        parser.error("build the project and source ROS and its install/setup.bash first")
    output = args.output.expanduser().resolve() if args.output else None
    if output and (output.exists() or output.is_symlink()):
        parser.error(f"refusing to overwrite report: {output}")

    # Configure discovery before importing or initializing any ROS component.
    os.environ["ROS_DOMAIN_ID"] = "42"
    os.environ.pop("ROS_LOCALHOST_ONLY", None)
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    os.environ.pop("ROS_STATIC_PEERS", None)
    logs = Path(tempfile.mkdtemp(prefix="bev-replay-check-"))
    os.environ["ROS_LOG_DIR"] = str(logs / "ros_logs")
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray

    probe_command = [
        str(executable), "--ros-args", "-r", "__ns:=/bev_replay_check",
        "-p", "use_sim_time:=true", "-p", "imu_topic:=/imu",
        "-p", "lidar_reliability:=reliable",
    ]
    replay_command = [
        "ros2", "bag", "play", str(bag), "--clock",
        "--exclude-topics", "/camera_lidar_bev/diagnostics",
    ]
    report = {
        "bag_path": str(bag),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "ros_domain_id": 42,
        "discovery_range": "LOCALHOST",
        "probe_command": probe_command,
        "replay_command": replay_command,
        "logs": str(logs),
        "probe_messages_received": {},
        "observed_stream_statuses": {},
        "latest_statuses": {},
        "sensor_clock_alignment": "unverified; replay receipt does not validate acquisition timing",
        "point_time_units_and_origin": "unverified",
        "calibration_status": "pending",
        "motion_trajectory_status": "missing odometry expected for this partial input session",
        "full_sensor_gate": "partial",
        "result": "failed",
    }
    node = None
    probe = None
    replay = None
    log_files = []
    latest = {}
    observed = {}
    report_count = 0
    exit_code = 1
    started = time.monotonic()
    deadline = started + args.timeout
    try:
        rclpy.init()
        node = rclpy.create_node("replay_check", namespace="/bev_replay_check")

        def receive(message):
            nonlocal report_count
            report_count += 1
            for status in message.status:
                name = status.name.split("/")[-1]
                values = {item.key: item.value for item in status.values}
                snapshot = {"name": status.name, "level": diagnostic_level(status.level), "message": status.message, "values": values}
                latest[name] = snapshot
                count = int(values.get("messages_received", "0"))
                previous = observed.get(name)
                if previous is None or count > int(previous["values"].get("messages_received", "0")):
                    observed[name] = snapshot

        subscription = node.create_subscription(DiagnosticArray, "/bev_replay_check/diagnostics", receive, 5)

        def spin_for(seconds):
            until = min(time.monotonic() + seconds, deadline)
            while time.monotonic() < until:
                if probe is not None and probe.poll() is not None:
                    raise RuntimeError(f"probe exited early with code {probe.returncode}")
                rclpy.spin_once(node, timeout_sec=min(0.05, max(0, until - time.monotonic())))
            if time.monotonic() >= deadline:
                raise TimeoutError(f"replay check exceeded {args.timeout:g} seconds")

        # Refuse an occupied domain rather than silently merging another session.
        spin_for(2.0)
        other_nodes = [(name, namespace) for name, namespace in node.get_node_names_and_namespaces()
                       if (name, namespace) != ("replay_check", "/bev_replay_check")]
        if other_nodes:
            raise RuntimeError(f"domain 42 is already occupied: {other_nodes}; stop its existing replay processes first")
        probe_log = (logs / "probe.log").open("x", encoding="utf-8")
        log_files.append(probe_log)
        probe = subprocess.Popen(probe_command, stdout=probe_log, stderr=subprocess.STDOUT, start_new_session=True)
        ready_until = min(time.monotonic() + 8.0, deadline)
        while not latest:
            spin_for(0.1)
            if time.monotonic() >= ready_until:
                raise TimeoutError("probe diagnostics did not arrive during discovery")
        spin_for(2.0)
        replay_log = (logs / "replay.log").open("x", encoding="utf-8")
        log_files.append(replay_log)
        replay = subprocess.Popen(replay_command, stdout=replay_log, stderr=subprocess.STDOUT, start_new_session=True)
        while replay.poll() is None:
            spin_for(0.1)
        report["replay_returncode"] = replay.returncode
        if replay.returncode != 0:
            raise RuntimeError(f"bag replay exited with code {replay.returncode}; inspect {logs / 'replay.log'}")
        # The probe reports cumulative receipt counts on its one-second wall timer.
        spin_for(1.5)
        required = ("lidar", "front_camera", "camera_info", "imu")
        counts = {name: int(latest.get(name, {}).get("values", {}).get("messages_received", "0"))
                  for name in (*required, "odometry")}
        lidar = observed.get("lidar", {}).get("values", {})
        calibration = latest.get("calibration_and_fusion", {})
        odometry = latest.get("odometry", {})
        checks = {f"{name}_received": counts[name] > 0 for name in required}
        checks.update({
            "timing_field_t_present": "t" in lidar.get("point_fields", "").split(",")
                and lidar.get("point_time_field_present") == "true",
            "cloud_payload_layout_valid": lidar.get("payload_layout_valid") == "true",
            "calibration_pending_warn": calibration.get("level") == 1
                and calibration.get("values", {}).get("extrinsic_calibration") == "unverified",
            "missing_odometry_warn": counts["odometry"] == 0 and odometry.get("level") == 1
                and "Waiting for input" in odometry.get("message", ""),
        })
        report["checks"] = checks
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError("replay verification failed: " + ", ".join(failed))
        report["result"] = "passed_partial_recording_replay"
        exit_code = 0
        # Keep the subscription alive for the entire run.
        del subscription
    except KeyboardInterrupt:
        report["result"] = "interrupted"
        exit_code = 130
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        report["error"] = str(error)
        print(str(error), file=sys.stderr)
    finally:
        stop_process(replay)
        stop_process(probe)
        for handle in log_files:
            handle.close()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        report["diagnostic_arrays_received"] = report_count
        report["latest_statuses"] = latest
        report["observed_stream_statuses"] = observed
        report["probe_messages_received"] = {
            name: int(snapshot["values"].get("messages_received", "0"))
            for name, snapshot in latest.items() if "messages_received" in snapshot["values"]
        }
        report["ended_utc"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        if replay is not None:
            report["replay_returncode"] = replay.poll()
        if probe is not None:
            report["probe_returncode_after_cleanup"] = probe.poll()
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
    print(json.dumps(report, indent=2))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
