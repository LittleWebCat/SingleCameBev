#!/usr/bin/env python3
"""Record an explicitly requested, bounded ROS bag; dry-run needs no ROS imports."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time


DEFAULT_TOPICS = [
    "/points",
    "/imu",
    "/metadata",
    "/camera/front/image_raw",
    "/camera/front/camera_info",
    "/tf",
    "/tf_static",
    "/camera_lidar_bev/diagnostics",
]


def positive_duration(value):
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise argparse.ArgumentTypeError("duration must be finite and greater than zero")
    return duration


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def storage_id():
    """Inspect the sourced ament index without starting a node or importing ROS."""
    prefixes = [Path(item) for item in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if item]
    for prefix in prefixes:
        marker = prefix / "share/ament_index/resource_index/packages/rosbag2_storage_mcap"
        if marker.is_file():
            return "mcap"
    return "sqlite3"


def exists(path):
    return path.exists() or path.is_symlink()


def stop_recorder(process):
    """Stop the entire recorder group, then escalate only if graceful stop fails."""
    if process.poll() is not None:
        return "already_exited"
    old_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for stop_signal, timeout, outcome in (
            (signal.SIGINT, 15, "sigint"),
            (signal.SIGTERM, 5, "sigterm_after_timeout"),
            (signal.SIGKILL, 5, "sigkill_after_timeout"),
        ):
            try:
                os.killpg(process.pid, stop_signal)
            except ProcessLookupError:
                return "already_exited"
            try:
                process.wait(timeout=timeout)
                return outcome
            except subprocess.TimeoutExpired:
                pass
        return "did_not_exit"
    finally:
        signal.signal(signal.SIGINT, old_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new bag directory; existing paths are refused")
    parser.add_argument("--duration", type=positive_duration, default=10.0, help="recording seconds (default: 10)")
    parser.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS, help="replace the default absolute topic list")
    parser.add_argument("--storage-preset-profile", choices=("none", "fastwrite", "zstd_fast", "zstd_small"), default="none")
    parser.add_argument("--qos-overrides", type=Path, help="explicit ROS recorder QoS YAML")
    parser.add_argument("--max-bag-size", type=int, default=0, help="split files at this byte size; 0 disables splitting")
    parser.add_argument("--max-cache-size", type=int, default=104857600, help="recorder message cache in bytes")
    parser.add_argument("--min-free-gib", type=float, default=0, help="stop cleanly before free space falls below this floor")
    parser.add_argument("--dry-run", action="store_true", help="print the exact argument list without DDS or file writes")
    args = parser.parse_args()
    output = Path(os.path.abspath(args.output.expanduser()))
    report_path = output.parent / (output.name + ".session.json")
    log_path = output.parent / (output.name + ".recorder.log")
    for path in (output, report_path, log_path):
        if exists(path):
            parser.error(f"refusing to overwrite existing path: {path}")
    for topic in args.topics:
        if not topic.startswith("/") or any(character.isspace() for character in topic):
            parser.error(f"expected an absolute ROS topic without whitespace: {topic!r}")
    topics = list(dict.fromkeys(args.topics))
    if args.max_bag_size < 0 or args.max_cache_size < 0 or not math.isfinite(args.min_free_gib) or args.min_free_gib < 0:
        parser.error("bag/cache sizes and free-space floor must be finite and nonnegative")
    if args.qos_overrides and not args.qos_overrides.is_file():
        parser.error("QoS overrides file is missing")
    storage = storage_id()
    command = ["ros2", "bag", "record", "--storage", storage, "--output", str(output), "--topics", *topics]
    command += ["--max-cache-size", str(args.max_cache_size), "--disable-keyboard-controls"]
    if storage == "mcap" and args.storage_preset_profile != "none":
        command += ["--storage-preset-profile", args.storage_preset_profile]
    if args.max_bag_size:
        command += ["--max-bag-size", str(args.max_bag_size)]
    if args.qos_overrides:
        command += ["--qos-profile-overrides-path", str(args.qos_overrides.resolve())]
    print("Command arguments: " + json.dumps(command), flush=True)
    print("Shell equivalent: " + shlex.join(command), flush=True)
    print(f"Duration: {args.duration:g} seconds; stop with SIGINT, with a bounded termination fallback.", flush=True)
    print(f"Recorder log: {log_path}", flush=True)
    print(f"Session report: {report_path}", flush=True)
    if args.dry_run:
        print("Dry run: no recorder started and no directories or files created.")
        return 0
    if shutil.which("ros2") is None:
        print("ros2 was not found; source the ROS environment before recording.", file=sys.stderr)
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a file appearing after preflight.
    try:
        report_file = report_path.open("x", encoding="utf-8")
    except OSError as error:
        print(f"Cannot create session report: {error}", file=sys.stderr)
        return 2
    report = {
        "bag_path": str(output),
        "recorder_log": str(log_path),
        "command": command,
        "commanded_topics": topics,
        "storage_id": storage,
        "storage_preset_profile": args.storage_preset_profile if storage == "mcap" else "none",
        "minimum_free_gib": args.min_free_gib,
        "commanded_duration_seconds": args.duration,
        "started_utc": None,
        "ended_utc": None,
        "elapsed_seconds": None,
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", "default"),
        "cyclonedds_uri": os.environ.get("CYCLONEDDS_URI"),
        "clock_domain": {
            "report_timestamps": "host UTC wall clock",
            "duration_timer": "host monotonic clock",
            "sensor_clock_alignment": "unverified",
        },
        "calibration_status": "unverified",
        "motion_trajectory_status": "unverified",
        "topic_presence_and_counts": "not measured by this helper; inspect ros2 bag info",
        "status": "launch_failed",
        "stop_method": None,
        "recorder_returncode": None,
        "bag_metadata_present": False,
    }
    process = None
    started = time.monotonic()
    exit_code = 1
    try:
        with log_path.open("x", encoding="utf-8") as log_file:
            # A final check prevents accidentally using a bag created since preflight.
            if exists(output):
                raise FileExistsError(f"bag path appeared after preflight: {output}")
            report["started_utc"] = utc_now()
            process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
            started = time.monotonic()
            deadline = started + args.duration
            report["status"] = "recording"
            while process.poll() is None:
                if args.min_free_gib and shutil.disk_usage(output.parent).free < args.min_free_gib*1024**3:
                    report["status"] = "disk_space_low"
                    report["error"] = "Recording stopped before the configured free-space floor"
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    report["status"] = "duration_reached"
                    break
                try:
                    process.wait(timeout=min(remaining, 0.5))
                except subprocess.TimeoutExpired:
                    pass
            if report["status"] == "recording":
                report["status"] = "recorder_exited_early"
            report["stop_method"] = stop_recorder(process)
            report["recorder_returncode"] = process.poll()
            report["bag_metadata_present"] = (output / "metadata.yaml").is_file()
            acceptable_return = process.returncode in (0, -signal.SIGINT, 128 + signal.SIGINT)
            graceful_stop = report["stop_method"] in ("sigint", "already_exited")
            if report["status"] == "duration_reached" and acceptable_return and graceful_stop and report["bag_metadata_present"]:
                report["status"] = "completed"
                exit_code = 0
            elif report["status"] == "duration_reached":
                report["status"] = "recorder_stop_error"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        exit_code = 130
    except (OSError, subprocess.SubprocessError) as error:
        report["error"] = str(error)
        print(f"Recording failed: {error}", file=sys.stderr)
    finally:
        if process is not None:
            if process.poll() is None:
                report["stop_method"] = stop_recorder(process)
            report["recorder_returncode"] = process.poll()
        report["bag_metadata_present"] = (output / "metadata.yaml").is_file()
        report["ended_utc"] = utc_now()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        with report_file:
            json.dump(report, report_file, indent=2)
            report_file.write("\n")
    print(f"Recording status: {report['status']}; recorder exit: {report['recorder_returncode']}")
    print(f"Inspect recorder output in {log_path} and topic counts with ros2 bag info {shlex.quote(str(output))}.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
