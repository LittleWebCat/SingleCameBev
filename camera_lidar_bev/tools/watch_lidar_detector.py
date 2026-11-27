#!/usr/bin/env python3
"""Live terminal view of the LiDAR 3D detector's performance (read-only).

Refreshes every second from /camera_lidar_bev/lidar_detections_3d and the detector's
/camera_lidar_bev/diagnostics. Scan and detection rates come from the node's own
received/processed counters (subscribing to the 6 MB clouds from Python would be the
bottleneck). Latency is receipt time minus the scan header stamp, which the driver
sets on scan reception. Ctrl+C to exit.
"""
import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timezone
import statistics
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from vision_msgs.msg import Detection3DArray


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window', type=float, default=3.0, help='seconds of history for rates/latency')
    parser.add_argument('--csv', help='also append one row per second to this CSV file')
    args = parser.parse_args()
    columns = ['utc', 'scan_hz', 'detection_hz', 'latency_median_ms', 'latency_max_ms', 'processing_ms',
               'detector_ms', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'model_input_points',
               'input_pillars', 'object_boxes', 'model']
    log = None
    if args.csv:
        stream = open(args.csv, 'a', newline='')
        log = csv.writer(stream)
        if stream.tell() == 0:
            log.writerow(columns)
    rclpy.init()
    node = rclpy.create_node('lidar_detector_watch')
    outputs = deque()  # (monotonic, latency_ms) within the window
    counters = deque()  # (monotonic, received_scans, processed_scans) from diagnostics
    latest = {'diag': {}, 'classes': Counter()}

    def on_detections(m):
        latency = (node.get_clock().now().nanoseconds - stamp_ns(m.header)) / 1e6
        if latency < 5:
            return  # clear message stamped "now", not a processed scan
        outputs.append((time.monotonic(), latency))
        latest['classes'] = Counter(d.results[0].hypothesis.class_id for d in m.detections if d.results)

    def on_diagnostics(m):
        for s in m.status:
            if s.name == 'lidar_vehicle_detector':
                latest['diag'] = {v.key: v.value for v in s.values}
                latest['message'] = s.message
                try:
                    counters.append((time.monotonic(), int(latest['diag']['received_scans']),
                                     int(latest['diag']['processed_scans'])))
                except (KeyError, ValueError):
                    pass

    node.create_subscription(Detection3DArray, '/camera_lidar_bev/lidar_detections_3d', on_detections, 10)
    node.create_subscription(DiagnosticArray, '/camera_lidar_bev/diagnostics', on_diagnostics, 10)

    def num(key, fmt='{:.1f}'):
        try:
            return fmt.format(float(latest['diag'][key]))
        except (KeyError, ValueError):
            return '-'

    next_draw = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if now < next_draw:
                continue
            next_draw = now + 1.0
            while outputs and now - outputs[0][0] > args.window:
                outputs.popleft()
            if not outputs:
                latest['classes'] = Counter()  # no fresh detections: do not show stale boxes
            while len(counters) > 2 and now - counters[0][0] > args.window + 1:
                counters.popleft()
            lat = sorted(o[1] for o in outputs)
            if len(counters) >= 2 and counters[-1][0] > counters[0][0]:
                dt = counters[-1][0] - counters[0][0]
                scan_hz = (counters[-1][1] - counters[0][1]) / dt
                det_hz = (counters[-1][2] - counters[0][2]) / dt
            else:
                scan_hz = det_hz = 0.0
            lines = [
                f"LiDAR 3D detector live view   model: {latest['diag'].get('model', 'waiting...')}",
                f"status: {latest.get('message', 'waiting for diagnostics')}",
                '',
                f"scan rate        {scan_hz:6.1f} Hz   (received by the detector node)",
                f"detection rate   {det_hz:6.1f} Hz     processed   {det_hz / scan_hz if scan_hz else 0:6.0%} of scans",
                f"latency (scan->boxes)  median {statistics.median(lat) if lat else 0:6.1f} ms   max {lat[-1] if lat else 0:6.1f} ms",
                '',
                f"node processing  {num('processing_ms'):>6} ms  (decode + detect + publish)",
                f"  detector       {num('detector_ms'):>6} ms  = pillars {num('preprocess_ms')} + GPU {num('inference_ms')} + boxes {num('postprocess_ms')}",
                f"  input points   {num('model_input_points', '{:.0f}'):>6}      pillars {num('input_pillars', '{:.0f}')}",
                '',
                'boxes: ' + (', '.join(f'{k} {v}' for k, v in latest['classes'].most_common()) or 'none'),
                '',
                f'(last {args.window:.0f} s window; diagnostics update at 1 Hz; Ctrl+C to exit)',
            ]
            print('\033[2J\033[H' + '\n'.join(lines), flush=True)
            if log and latest['diag']:
                d = latest['diag']
                log.writerow([datetime.now(timezone.utc).isoformat(timespec='milliseconds'), f'{scan_hz:.2f}',
                              f'{det_hz:.2f}', f'{statistics.median(lat):.1f}' if lat else '',
                              f'{lat[-1]:.1f}' if lat else '', *(d.get(k, '') for k in columns[5:13]),
                              d.get('model', '')])
                stream.flush()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
