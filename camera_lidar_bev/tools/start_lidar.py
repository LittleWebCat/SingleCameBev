#!/usr/bin/env python3
"""Start the existing Ouster driver with verified lifecycle transitions."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def driver_arguments(params_file):
    return ["ros2", "run", "ouster_ros", "os_driver", "--ros-args",
            "--params-file", str(params_file), "-r", "__node:=os_driver"]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-file", type=Path, default=root / "config/sensors/ouster_baseline.yaml")
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0.0 or args.startup_timeout > 300.0:
        parser.error("--startup-timeout must be in (0, 300] seconds")
    params_file = args.params_file.resolve()
    if not params_file.is_file():
        parser.error(f"Missing parameter file: {params_file}")
    command = driver_arguments(params_file)
    print(f"ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0')}", flush=True)
    print("Driver command: " + json.dumps(command), flush=True)
    if args.dry_run:
        print("Dry run: no driver, lifecycle call, or sensor configuration change")
        return 0
    for setting in ("net/core/rmem_max", "net/core/rmem_default", "net/ipv4/ipfrag_time", "net/ipv4/ipfrag_high_thresh"):
        try:
            value = (Path("/proc/sys") / setting).read_text().strip()
        except OSError:
            value = "unavailable"
        print(f"Host {setting.replace('/', '.')}={value} (read-only check)", flush=True)
    if not shutil.which("ros2"):
        parser.error("Source ROS Jazzy and sensors_ws/install/setup.bash first")
    # A process in another ROS domain can still own the same sensor/UDP ports.
    existing = subprocess.run(["pgrep", "-x", "os_driver"], capture_output=True, text=True)
    if existing.returncode == 0:
        raise RuntimeError("An os_driver process already exists (PID " + existing.stdout.strip().replace("\n", ",") +
                           "); inspect its ROS domain and state before starting another")

    import rclpy
    from lifecycle_msgs.srv import ChangeState, GetState

    rclpy.init()
    controller = rclpy.create_node("bev_lidar_lifecycle_controller")
    state_client = controller.create_client(GetState, "/os_driver/get_state")
    transition_client = controller.create_client(ChangeState, "/os_driver/change_state")
    process = None
    try:
        if state_client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("/os_driver already exists; refusing to start or reconfigure an existing driver")
        runtime = root / "data/lidar_runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(command, cwd=runtime, start_new_session=True)

        def wait_future(future, description):
            deadline = time.monotonic() + args.startup_timeout
            while not future.done() and time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"Driver exited with {process.returncode} while {description}")
                rclpy.spin_once(controller, timeout_sec=0.1)
            if not future.done():
                raise RuntimeError(f"Timed out while {description}")
            result = future.result()
            if result is None:
                raise RuntimeError(f"No lifecycle response while {description}")
            return result

        def get_state():
            return wait_future(state_client.call_async(GetState.Request()), "reading driver state").current_state

        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"Driver exited with {process.returncode} before lifecycle registration")
            if state_client.wait_for_service(timeout_sec=0.2) and transition_client.wait_for_service(timeout_sec=0.2):
                break
        else:
            raise RuntimeError("Driver lifecycle services did not appear")
        state = get_state()
        if state.id != 1:
            raise RuntimeError(f"Expected unconfigured state, got {state.id}:{state.label}")
        for transition_id, expected_state, label in ((1, 2, "configure"), (3, 3, "activate")):
            print(f"Sending {label.upper()} and waiting for completion", flush=True)
            request = ChangeState.Request()
            request.transition.id = transition_id
            response = wait_future(transition_client.call_async(request), f"waiting for {label}")
            if not response.success:
                raise RuntimeError(f"Driver rejected {label}")
            state = get_state()
            if state.id != expected_state:
                raise RuntimeError(f"Unexpected state after {label}: {state.id}:{state.label}")
        print("LiDAR ACTIVE: /points, /imu, /metadata. Point timing and clocks still require measurement.", flush=True)
        print("Keep this process running; Ctrl+C stops only its own driver process group.", flush=True)
        while rclpy.ok() and process.poll() is None:
            rclpy.spin_once(controller, timeout_sec=0.2)
        if process.poll() is not None:
            return process.returncode
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
