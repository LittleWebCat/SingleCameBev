#!/usr/bin/env python3
"""Live LiDAR 3D detector benchmark across Ouster modes (default: 1024x20 then 2048x10).

For each mode: starts the Ouster driver (tools/start_lidar.py) and the detector node,
waits for warmup, then measures for --duration seconds:

* /points: scan rate, points per scan
* /camera_lidar_bev/lidar_detections_3d: output rate, boxes per scan, end-to-end latency
  (receipt time minus scan header stamp, which the driver sets at scan reception)
* /camera_lidar_bev/diagnostics: detector timing breakdown (1 Hz samples)

Both processes are stopped before the next mode. Run with ROS Jazzy and sensors_ws
sourced, using /usr/bin/python3. Read-only with respect to maps and recordings.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

MODES = {'1024x20': 'ouster_baseline.yaml', '2048x10': 'ouster_2048x10.yaml'}
TIMING_KEYS = ('processing_ms', 'detector_ms', 'preprocess_ms', 'inference_ms', 'postprocess_ms',
               'model_input_points', 'input_pillars', 'object_boxes', 'truncated_points')


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def summary(values):
    if not values:
        return None
    v = sorted(values)
    return {'n': len(v), 'median': round(statistics.median(v), 3), 'mean': round(statistics.fmean(v), 3),
            'p95': round(v[min(len(v) - 1, int(0.95 * len(v)))], 3), 'max': round(v[-1], 3)}


def stop(process, name):
    if process is None or process.poll() is not None:
        return
    for sig, wait in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=wait)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            print(f'{name} did not stop after {sig.name}', flush=True)


def measure(node, rclpy, duration, warmup):
    from diagnostic_msgs.msg import DiagnosticArray
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from vision_msgs.msg import Detection3DArray

    data = {'scans': [], 'detections': [], 'diagnostics': []}
    state = {'recording': False}

    def on_points(m):
        if state['recording']:
            data['scans'].append((time.monotonic(), m.width * m.height, stamp_ns(m.header)))

    def on_detections(m):
        if state['recording'] and stamp_ns(m.header) > 0:
            latency = (node.get_clock().now().nanoseconds - stamp_ns(m.header)) / 1e6
            data['detections'].append((time.monotonic(), latency, len(m.detections), stamp_ns(m.header)))

    def on_diagnostics(m):
        if not state['recording']:
            return
        for s in m.status:
            if s.name == 'lidar_vehicle_detector':
                data['diagnostics'].append({v.key: v.value for v in s.values})

    subs = [node.create_subscription(PointCloud2, '/points', on_points, qos_profile_sensor_data),
            node.create_subscription(Detection3DArray, '/camera_lidar_bev/lidar_detections_3d', on_detections, 10),
            node.create_subscription(DiagnosticArray, '/camera_lidar_bev/diagnostics', on_diagnostics, 10)]
    end = time.monotonic() + warmup
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    state['recording'] = True
    start = time.monotonic()
    while time.monotonic() - start < duration:
        rclpy.spin_once(node, timeout_sec=0.05)
    for s in subs:
        node.destroy_subscription(s)
    return data, time.monotonic() - start


def analyse(data, elapsed):
    # Detector clears publish empty arrays with a fresh stamp; keep only outputs for real scans.
    scan_stamps = {s[2] for s in data['scans']}
    fresh = [d for d in data['detections'] if d[3] in scan_stamps]
    distinct = {d[3] for d in fresh}
    diag = [d for d in data['diagnostics'] if d.get('publishing_fresh_output') == 'true']
    out = {
        'seconds': round(elapsed, 1),
        'scan_rate_hz': round(len(data['scans']) / elapsed, 2),
        'points_per_scan': summary([s[1] for s in data['scans']]),
        'detection_rate_hz': round(len(distinct) / elapsed, 2),
        'scans_processed_fraction': round(len(distinct) / max(1, len(data['scans'])), 3),
        'end_to_end_latency_ms': summary([d[1] for d in fresh]),
        'boxes_per_scan': summary([d[2] for d in fresh]),
        'diagnostic_samples': len(diag),
        'model': diag[-1].get('model') if diag else None,
    }
    for key in TIMING_KEYS:
        values = [float(d[key]) for d in diag if key in d]
        out[key] = summary(values)
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--modes', nargs='+', choices=sorted(MODES), default=['1024x20', '2048x10'])
    parser.add_argument('--model', choices=['centerpoint', 'pointpillars'], default='centerpoint')
    parser.add_argument('--duration', type=float, default=60.0, help='measured seconds per mode')
    parser.add_argument('--warmup', type=float, default=10.0, help='seconds after first detection')
    parser.add_argument('--startup-timeout', type=float, default=90.0)
    parser.add_argument('--output', type=Path,
                        default=root / f'evaluation/lidar_detector_live_modes_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json')
    args = parser.parse_args()
    if subprocess.run(['pgrep', '-x', 'os_driver'], capture_output=True).returncode == 0:
        parser.error('An os_driver is already running; stop it first so the mode can be set')
    # Kernel process names (comm) are truncated to 15 characters.
    if subprocess.run(['pgrep', '-x', 'lidar_vehicle_d'], capture_output=True).returncode == 0:
        parser.error('A lidar_vehicle_detector is already running; stop it first')

    import rclpy
    from vision_msgs.msg import Detection3DArray
    rclpy.init()
    node = rclpy.create_node('lidar_detector_live_benchmark')
    results = {'model': args.model, 'ros_domain_id': os.environ.get('ROS_DOMAIN_ID', '0'),
               'started_utc': datetime.now(timezone.utc).isoformat(), 'modes': {},
               'scope': 'Live stationary run; timings from 1 Hz node diagnostics, latency per published scan. '
                        'Not physical detection accuracy.'}
    try:
        for mode in args.modes:
            print(f'=== {mode}: starting driver and {args.model} detector', flush=True)
            driver = detector = None
            try:
                driver = subprocess.Popen(
                    ['/usr/bin/python3', str(root / 'tools/start_lidar.py'), '--params-file',
                     str(root / 'config/sensors' / MODES[mode]), '--startup-timeout', '60'],
                    start_new_session=True)
                detector = subprocess.Popen(
                    ['ros2', 'launch', 'bev_bringup', 'lidar_vehicle_detector.launch.py', 'rviz:=false',
                     f'model:={args.model}'], start_new_session=True)
                seen = {'n': 0}
                sub = node.create_subscription(Detection3DArray, '/camera_lidar_bev/lidar_detections_3d',
                                               lambda m: seen.update(n=seen['n'] + 1), 10)
                deadline = time.monotonic() + args.startup_timeout
                while seen['n'] < 5 and time.monotonic() < deadline:
                    if driver.poll() is not None or detector.poll() is not None:
                        raise RuntimeError('driver or detector exited during startup')
                    rclpy.spin_once(node, timeout_sec=0.2)
                node.destroy_subscription(sub)
                if seen['n'] < 5:
                    raise RuntimeError('no detector output before the startup timeout')
                data, elapsed = measure(node, rclpy, args.duration, args.warmup)
                results['modes'][mode] = analyse(data, elapsed)
                print(json.dumps({mode: results['modes'][mode]}, indent=1), flush=True)
            except Exception as error:  # record and continue with the next mode
                results['modes'][mode] = {'error': str(error)}
                print(f'{mode} failed: {error}', flush=True)
            finally:
                stop(detector, 'detector')
                stop(driver, 'driver')
                time.sleep(3)  # let the sensor UDP ports and DDS participants go away
    finally:
        node.destroy_node()
        rclpy.shutdown()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(f'Wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
