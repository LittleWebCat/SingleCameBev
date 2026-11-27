#!/usr/bin/env python3
"""Live front-camera traffic-light detection and colour (red / yellow / green) with performance numbers.

Runs models/traffic_light_state_fp32.engine (YOLO that detects a signal head and classifies its colour
in one pass) on the newest frame of the front camera (default /camera/front_wide/image_raw). Read-only:
it subscribes to topics the running camera pipeline already publishes and controls nothing.

Window (q or Esc quits):
  * coloured boxes: state-model detections with confidence; enlarged crops along the bottom;
  * thin white boxes: generic COCO "traffic light" from the running camera_object_detector, as a
    reference for lights the state model did / did not classify;
  * cyan circles: kept mapped traffic lights (MAP/landmarks/landmarks_reviewed.json) projected with
    map localization and the rig calibration; a mapped light counts as detected when a colour box lies
    within --match-px of it. Occlusion is not modelled, so "expected" means "in the field of view".
  * top bar: camera / model FPS, model latency, image age, colour counts, COCO coverage, and the
    mapped-light detection rate by distance.
Writes data/traffic_light_eval_<UTC>/frames.jsonl (one line per processed frame) and summary.json on
exit; --record also saves the annotated view as annotated.mp4.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  ../.venv/bin/python tools/traffic_light_camera_view.py [--camera front_wide|front_narrow] [--record]
  # both at once (two windows, two result folders): bash tools/run_light_camera_test.sh
"""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from map_landmarks import CAMERAS as LANDMARK_CAMERAS, Camera, quat_to_matrix, stamp_ns

RIG = Path(__file__).resolve().parents[1] / 'config/calibration/rig_20261125'
# name: (COCO detections topic for the reference boxes, intrinsics, extrinsics); the two front cameras use the
# 2026-11-25 rig (new narrow camera on USB 4.1.3.3, image /camera/front_narrow/image_raw as run_lane_compare.sh starts it)
CAMERAS = {**LANDMARK_CAMERAS,
           'front_wide': ('/camera_lidar_bev/front_wide/detections_2d', RIG / 'front_wide_1920x1080.yaml', RIG / 'front_wide_T_os_sensor.yaml'),
           'front_narrow': ('/camera_lidar_bev/detections_2d', RIG / 'front_narrow_1024x768.yaml', RIG / 'front_narrow_T_os_sensor.yaml')}

ROOT = Path(__file__).resolve().parents[1]
COLOURS = {'red': (40, 40, 255), 'yellow': (0, 210, 255), 'green': (60, 220, 60)}  # BGR
BINS = (0, 20, 40, 60, 80, 120)
WARMUP_FRAMES = 10


def bin_name(d):
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        if lo <= d < hi:
            return f'{lo}-{hi} m'
    return f'{BINS[-1]}+ m'


def percentiles(values):
    if not values:
        return None
    a = np.asarray(values)
    return {'mean': round(float(a.mean()), 2), 'p50': round(float(np.percentile(a, 50)), 2),
            'p95': round(float(np.percentile(a, 95)), 2), 'max': round(float(a.max()), 2)}


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def box_gap(box, u, v):
    """Pixel distance from (u, v) to the box (0 inside)."""
    return math.hypot(max(box[0] - u, 0.0, u - box[2]), max(box[1] - v, 0.0, v - box[3]))


def project_mapped(lights, cam, pose, max_range_m):
    """Mapped lights inside the image: [{'id', 'distance_m', 'uv'}] (occlusion not modelled)."""
    if pose is None:
        return []
    rot, pos = pose
    out = []
    for l in lights:
        p_sensor = rot.T @ (np.asarray(l['position_enu_m'], float) - pos)
        p_cam = cam.cam_from_sensor[:3, :3] @ p_sensor + cam.cam_from_sensor[:3, 3]
        distance = float(np.hypot(p_sensor[0], p_sensor[1]))
        if p_cam[2] < 1.0 or distance > max_range_m:
            continue
        # Pinhole bounds first: lens distortion terms can fold far off-axis points back into the image.
        u0 = cam.k[0, 0] * p_cam[0] / p_cam[2] + cam.k[0, 2]
        v0 = cam.k[1, 1] * p_cam[1] / p_cam[2] + cam.k[1, 2]
        if not (-0.2 * cam.w < u0 < 1.2 * cam.w and -0.2 * cam.h < v0 < 1.2 * cam.h):
            continue
        uv, _ = cv2.projectPoints(p_cam.reshape(1, 1, 3), np.zeros(3), np.zeros(3), cam.k, cam.d)
        u, v = (float(x) for x in uv.reshape(2))
        if 0 <= u < cam.w and 0 <= v < cam.h:
            out.append({'id': l['id'], 'distance_m': round(distance, 1), 'uv': [round(u), round(v)]})
    return out


class Feeds:
    """Newest image, COCO detections by stamp, and latest localization pose (filled by a ROS thread)."""

    def __init__(self, node, cam, image_topic):
        self.lock = threading.Lock()
        self.image = None
        self.image_times = deque(maxlen=60)
        self.received = 0
        self.coco = deque(maxlen=60)  # (stamp ns, [(box, score)])
        self.pose = None
        node.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)
        node.create_subscription(Detection2DArray, cam.topic, self.on_coco, 10)
        node.create_subscription(Odometry, '/camera_lidar_bev/localization/odometry', self.on_odometry, 10)

    def on_image(self, m):
        with self.lock:
            self.image = m
            self.received += 1
            self.image_times.append(time.monotonic())

    def on_coco(self, m):
        boxes = []
        for d in m.detections:
            if d.results and d.results[0].hypothesis.class_id == 'traffic light':
                c, sx, sy = d.bbox.center.position, d.bbox.size_x, d.bbox.size_y
                boxes.append(((c.x - sx / 2, c.y - sy / 2, c.x + sx / 2, c.y + sy / 2), d.results[0].hypothesis.score))
        with self.lock:
            self.coco.append((stamp_ns(m.header), boxes))

    def on_odometry(self, m):
        q, p = m.pose.pose.orientation, m.pose.pose.position
        with self.lock:
            self.pose = (quat_to_matrix((q.x, q.y, q.z, q.w)), np.array([p.x, p.y, p.z]))

    def take(self):
        with self.lock:
            image, self.image = self.image, None
            return image, self.pose

    def coco_for(self, stamp, tolerance_ns=40_000_000):
        with self.lock:
            best = min(self.coco, key=lambda c: abs(c[0] - stamp), default=None)
        return best[1] if best is not None and abs(best[0] - stamp) <= tolerance_ns else None

    def camera_fps(self):
        with self.lock:
            t = list(self.image_times)
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0


def text(img, s, org, scale=0.6, colour=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def crop_strip(frame, dets, width, height=150):
    strip = np.full((height, width, 3), 24, np.uint8)
    x = 8
    for cls, conf, (x0, y0, x1, y1) in sorted(dets, key=lambda d: -d[1])[:10]:
        pw, ph = 0.6 * (x1 - x0), 0.3 * (y1 - y0)
        crop = frame[max(0, int(y0 - ph)):int(y1 + ph), max(0, int(x0 - pw)):int(x1 + pw)]
        if crop.size == 0:
            continue
        h = height - 34
        w = max(8, min(int(crop.shape[1] * h / crop.shape[0]), 180))
        if x + w > width - 8:
            break
        strip[6:6 + h, x:x + w] = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(strip, (x - 2, 4), (x + w + 1, 7 + h), COLOURS[cls], 2)
        text(strip, f'{cls} {conf:.2f}', (x, height - 10), 0.5, COLOURS[cls])
        x += w + 14
    if x == 8:
        text(strip, 'no red / yellow / green detection in this frame', (12, height // 2), 0.6, (170, 170, 170))
    return strip


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera', choices=sorted(CAMERAS), default='front_wide')
    parser.add_argument('--image-topic', help='default: /camera/<camera>/image_raw')
    parser.add_argument('--model', type=Path, default=ROOT.parent / 'models/traffic_light_state_fp32.engine')
    parser.add_argument('--conf', type=float, default=0.35, help='state-model confidence threshold')
    parser.add_argument('--map', type=Path, default=ROOT / 'maps/local_gps_recording_002')
    parser.add_argument('--max-range-m', type=float, default=100.0, help='mapped lights farther than this are not expected')
    parser.add_argument('--match-px', type=float, default=80.0, help='colour box within this many px of a mapped light')
    parser.add_argument('--display-width', type=int, default=1280)
    parser.add_argument('--record', action='store_true', help='also save the annotated view as annotated.mp4')
    args = parser.parse_args()

    from ultralytics import YOLO  # import after argparse so --help works without the model stack
    model = YOLO(str(args.model), task='detect')
    cam = Camera(args.camera, *CAMERAS[args.camera])
    reviewed = args.map / 'landmarks/landmarks_reviewed.json'
    lights = [l for l in json.loads(reviewed.read_text())['landmarks']
              if l.get('keep') and l['type'] == 'traffic_light'] if reviewed.exists() else []
    out = ROOT / 'data' / f"traffic_light_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{args.camera}"
    out.mkdir(parents=True)
    log = (out / 'frames.jsonl').open('w')

    rclpy.init()
    node = rclpy.create_node('traffic_light_camera_view')
    feeds = Feeds(node, cam, args.image_topic or f'/camera/{args.camera}/image_raw')
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    print(f'{args.camera}: {args.model.name} classes {model.names}; {len(lights)} mapped traffic lights; '
          f'writing {out}', flush=True)

    latencies, ages, processed_times = [], [], deque(maxlen=60)
    colour_total, recent = Counter(), deque()  # recent: (monotonic time, colour)
    coco_stats = Counter()  # boxes, with_colour
    by_bin = {bin_name(lo): Counter() for lo in BINS[:-1]}  # expected, detected
    per_light = {}
    processed, start, writer = 0, time.monotonic(), None
    window = f'Traffic lights: {args.camera}'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while rclpy.ok():
            msg, pose = feeds.take()
            if msg is None:
                if cv2.waitKey(2) & 0xFF in (27, ord('q')):
                    break
                continue
            stamp = stamp_ns(msg.header)
            frame = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
            frame = frame.reshape(msg.height, msg.width, 3)
            if msg.encoding == 'rgb8':
                frame = frame[:, :, ::-1]
            t0 = time.perf_counter()
            result = model.predict(frame, imgsz=640, conf=args.conf, iou=0.5, verbose=False)[0]
            latency_ms = (time.perf_counter() - t0) * 1000
            now = time.monotonic()
            processed += 1
            processed_times.append(now)
            age_ms = (node.get_clock().now().nanoseconds - stamp) / 1e6
            if processed > WARMUP_FRAMES:
                latencies.append(latency_ms)
                ages.append(age_ms)
            dets = [(result.names[int(c)], float(s), tuple(float(v) for v in b))
                    for c, s, b in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist())]
            for cls, _, _ in dets:
                colour_total[cls] += 1
                recent.append((now, cls))
            while recent and recent[0][0] < now - 10:
                recent.popleft()

            coco = feeds.coco_for(stamp)
            coco_matched = []
            for box, _ in coco or []:
                hit = any(iou(box, b) >= 0.2 or (box[0] <= (b[0] + b[2]) / 2 <= box[2] and box[1] <= (b[1] + b[3]) / 2 <= box[3])
                          for _, _, b in dets)
                coco_matched.append(hit)
                coco_stats['boxes'] += 1
                coco_stats['with_colour'] += hit

            mapped = project_mapped(lights, cam, pose, args.max_range_m)
            for m in mapped:
                gaps = [(box_gap(b, *m['uv']), cls) for cls, _, b in dets]
                gap, cls = min(gaps, default=(math.inf, None))
                m['detected'], m['colour'] = gap <= args.match_px, cls if gap <= args.match_px else None
                b = by_bin.setdefault(bin_name(m['distance_m']), Counter())
                b['expected'] += 1
                b['detected'] += m['detected']
                p = per_light.setdefault(m['id'], {'expected_frames': 0, 'detected_frames': 0, 'colours': Counter(),
                                                   'max_detected_distance_m': None, 'min_expected_distance_m': None})
                p['expected_frames'] += 1
                p['min_expected_distance_m'] = min(p['min_expected_distance_m'] or math.inf, m['distance_m'])
                if m['detected']:
                    p['detected_frames'] += 1
                    p['colours'][cls] += 1
                    p['max_detected_distance_m'] = max(p['max_detected_distance_m'] or 0.0, m['distance_m'])

            log.write(json.dumps({'stamp_ns': stamp, 'latency_ms': round(latency_ms, 2), 'image_age_ms': round(age_ms, 1),
                                  'detections': [{'colour': c, 'conf': round(s, 3), 'box': [round(v) for v in b]} for c, s, b in dets],
                                  'coco_traffic_lights': None if coco is None else
                                  [{'box': [round(v) for v in box], 'score': round(sc, 3), 'with_colour': hit}
                                   for (box, sc), hit in zip(coco, coco_matched)],
                                  'mapped': mapped, 'localized': pose is not None}) + '\n')

            # Draw at full resolution, then scale for display.
            view = frame.copy()
            for (box, _), hit in zip(coco or [], coco_matched):
                cv2.rectangle(view, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 1)
            for cls, conf, b in dets:
                cv2.rectangle(view, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), COLOURS[cls], 3)
                text(view, f'{cls} {conf:.2f}', (int(b[0]), max(18, int(b[1]) - 8)), 0.8, COLOURS[cls])
            for m in mapped:
                u, v = m['uv']
                cv2.circle(view, (u, v), int(args.match_px), (255, 255, 0), 2)
                text(view, f"#{m['id']} {m['distance_m']:.0f} m " + (m['colour'] if m['detected'] else 'MISSED'),
                     (u + int(args.match_px) + 6, v + 6), 0.8, (255, 255, 0))
            scale = args.display_width / view.shape[1]
            view = cv2.resize(view, (args.display_width, round(view.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            cv2.rectangle(view, (0, 0), (view.shape[1], 92), (0, 0, 0), -1)
            fps = (len(processed_times) - 1) / (processed_times[-1] - processed_times[0]) if len(processed_times) > 1 else 0.0
            lat = percentiles(latencies[-300:])
            counts = Counter(c for _, c in recent)
            text(view, f"{args.camera} {msg.width}x{msg.height} | camera {feeds.camera_fps():.1f} fps | model {fps:.1f} fps | "
                       f"latency {lat['mean'] if lat else latency_ms:.1f} ms mean, {lat['p95'] if lat else latency_ms:.1f} p95 | "
                       f"image age {age_ms:.0f} ms", (10, 24))
            coco_rate = f"{coco_stats['with_colour']}/{coco_stats['boxes']} ({coco_stats['with_colour'] / coco_stats['boxes']:.0%})" \
                if coco_stats['boxes'] else 'none yet'
            text(view, f"last 10 s: red {counts['red']}  yellow {counts['yellow']}  green {counts['green']} | "
                       f"COCO traffic lights given a colour: {coco_rate}", (10, 50))
            rates = '  '.join(f"{k} {v['detected'] / v['expected']:.0%} ({v['expected']})" for k, v in by_bin.items() if v['expected'])
            text(view, f"mapped lights detected when in view: {rates or ('none in view yet' if pose is not None else 'no localization')}",
                 (10, 76), 0.6, (255, 255, 0))
            view = np.vstack([view, crop_strip(frame, dets, view.shape[1])])
            cv2.imshow(window, view)
            if args.record:
                if writer is None:
                    writer = cv2.VideoWriter(str(out / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 20.0,
                                             (view.shape[1], view.shape[0]))
                writer.write(view)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    except KeyboardInterrupt:
        pass
    finally:
        duration = time.monotonic() - start
        log.close()
        if writer is not None:
            writer.release()
        summary = {
            'camera': args.camera, 'model': args.model.name, 'confidence_threshold': args.conf,
            'duration_s': round(duration, 1), 'frames_received': feeds.received, 'frames_processed': processed,
            'model_fps': round(processed / duration, 2) if duration > 0 else None,
            'model_latency_ms': percentiles(latencies), 'image_age_ms_at_result': percentiles(ages),
            'detections_by_colour': dict(colour_total),
            'coco_traffic_lights': {**coco_stats, 'rate_with_colour': round(coco_stats['with_colour'] / coco_stats['boxes'], 3)
                                    if coco_stats['boxes'] else None},
            'mapped_lights_by_distance': {k: {**v, 'rate': round(v['detected'] / v['expected'], 3)}
                                          for k, v in by_bin.items() if v['expected']},
            'mapped_lights': {str(k): {**v, 'colours': dict(v['colours'])} for k, v in sorted(per_light.items())},
            'notes': f'mapped lights are expected when they project inside the image within {args.max_range_m:g} m '
                     f'(occlusion not modelled); detected = a colour box within {args.match_px:g} px',
        }
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2), flush=True)
        cv2.destroyAllWindows()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
