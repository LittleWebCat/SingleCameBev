#!/usr/bin/env python3
"""Live charts for tools/route_planner.py: measured vs planned vx / vy / ax / ay, and the plan ahead.

Left, last 30 s (vehicle frame at the rear axle, x forward, y left):
  vx, vy, ax, ay: measured from localization (cyan), Ouster IMU for ax/ay (magenta), planned for now
  (yellow: the plan's target speed 1 s ahead, planned ax / ay now; planned vy is 0 by design).
Right, the newest plan over its horizon: speed, longitudinal and lateral acceleration versus time.
Top: planner mode, stop line, front vehicle, route progress and tracking errors. q or Esc quits.
Display only; subscribes to /camera_lidar_bev/planner/state.

  source /opt/ros/jazzy/setup.bash && source install/setup.bash
  /usr/bin/python3 tools/planner_plots.py [--history-s 30]
"""
import argparse
from collections import deque
import json

import cv2
import numpy as np
import rclpy
from std_msgs.msg import String

CYAN, MAGENTA, YELLOW, GREY, WHITE = (255, 220, 0), (255, 80, 255), (0, 215, 255), (90, 90, 90), (235, 235, 235)


def put(img, s, org, scale=0.55, colour=WHITE, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def chart(img, rect, title, series, x_range, min_span, x_label):
    """series: [(xs, ys, colour, name)]; y range from the data with at least min_span, always showing 0."""
    x0, y0, w, h = rect
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), -1)
    values = [v for _, ys, _, _ in series for v in ys if v is not None and np.isfinite(v)]
    lo, hi = min(values + [0.0]), max(values + [0.0])
    if hi - lo < min_span:
        mid = (hi + lo) / 2 if values else 0.0
        lo, hi = mid - min_span / 2, mid + min_span / 2
        if lo > 0 or hi < 0:
            lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    px = lambda x: int(x0 + 46 + (x - x_range[0]) / (x_range[1] - x_range[0]) * (w - 56))  # noqa: E731
    py = lambda y: int(y0 + h - 22 - (y - lo) / (hi - lo) * (h - 46))  # noqa: E731
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = lo + frac * (hi - lo)
        cv2.line(img, (x0 + 46, py(yv)), (x0 + w - 10, py(yv)), GREY if frac in (0.0, 1.0) else (60, 60, 60), 1)
        put(img, f'{yv:.1f}', (x0 + 4, py(yv) + 4), 0.4, (170, 170, 170))
    cv2.line(img, (x0 + 46, py(0)), (x0 + w - 10, py(0)), (130, 130, 130), 1)
    put(img, title, (x0 + 50, y0 + 16), 0.5, WHITE)
    put(img, x_label, (x0 + w - 110, y0 + h - 5), 0.4, (170, 170, 170))
    lx = x0 + 50 + 9 * len(title)
    for xs, ys, colour, name in series:
        pts = [(px(x), py(y)) for x, y in zip(xs, ys) if y is not None and np.isfinite(y) and x_range[0] <= x <= x_range[1]]
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, colour, 2, cv2.LINE_AA)
        put(img, name, (lx, y0 + 16), 0.45, colour)
        lx += 10 * len(name) + 14


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--history-s', type=float, default=30.0)
    parser.add_argument('--width', type=int, default=1500)
    parser.add_argument('--height', type=int, default=900)
    args = parser.parse_args()

    rclpy.init()
    node = rclpy.create_node('planner_plots')
    hist = deque()  # (stamp, measured, imu, plan-now)
    latest = {}

    def on_state(m):
        r = json.loads(m.data)
        latest['r'] = r
        p = r.get('plan')
        now = None if p is None else {'v': p['v'][min(10, len(p['v']) - 1)], 'vy': 0.0, 'ax': p['ax'][0], 'ay': p['ay'][0]}
        if hist and r['stamp'] < hist[-1][0]:
            hist.clear()  # clock went back (replay restart)
        hist.append((r['stamp'], r.get('measured'), r.get('imu'), now))
        while hist and r['stamp'] - hist[0][0] > args.history_s:
            hist.popleft()

    node.create_subscription(String, '/camera_lidar_bev/planner/state', on_state, 20)
    window = 'Planner: measured vs planned'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.width, args.height)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            img = np.full((args.height, args.width, 3), 18, np.uint8)
            r = latest.get('r')
            if r is None:
                put(img, 'waiting for /camera_lidar_bev/planner/state (start tools/route_planner.py and localization)',
                    (20, 40), 0.7)
            else:
                t_now = r['stamp']
                ts = [h[0] - t_now for h in hist]
                get = lambda i, k: [None if h[i] is None else h[i].get(k) for h in hist]  # noqa: E731
                top = 70
                lw, rw = int(args.width * 0.62), args.width - int(args.width * 0.62) - 30
                ch = (args.height - top - 20) // 4
                xr = (-args.history_s, 0.0)
                chart(img, (10, top, lw, ch - 8), 'vx forward m/s', [(ts, get(1, 'vx'), CYAN, 'measured'),
                      (ts, get(3, 'v'), YELLOW, 'plan target +1 s')], xr, 2.0, 'time s')
                chart(img, (10, top + ch, lw, ch - 8), 'vy left m/s', [(ts, get(1, 'vy'), CYAN, 'measured'),
                      (ts, get(3, 'vy'), YELLOW, 'planned (0, no slip)')], xr, 0.5, 'time s')
                chart(img, (10, top + 2 * ch, lw, ch - 8), 'ax forward m/s2', [(ts, get(1, 'ax'), CYAN, 'localization'),
                      (ts, get(2, 'ax'), MAGENTA, 'IMU'), (ts, get(3, 'ax'), YELLOW, 'planned')], xr, 1.0, 'time s')
                chart(img, (10, top + 3 * ch, lw, ch - 8), 'ay left m/s2', [(ts, get(1, 'ay'), CYAN, 'localization'),
                      (ts, get(2, 'ay'), MAGENTA, 'IMU'), (ts, get(3, 'ay'), YELLOW, 'planned')], xr, 1.0, 'time s')
                p = r.get('plan')
                rx = lw + 30
                if p:
                    hr = (0.0, p['t'][-1])
                    ch3 = (args.height - top - 20) // 3
                    chart(img, (rx, top, rw, ch3 - 8), 'plan: speed m/s', [(p['t'], p['v'], YELLOW, 'v')], hr, 2.0, 'ahead s')
                    chart(img, (rx, top + ch3, rw, ch3 - 8), 'plan: ax m/s2', [(p['t'], p['ax'], YELLOW, 'ax')], hr, 1.0, 'ahead s')
                    chart(img, (rx, top + 2 * ch3, rw, ch3 - 8), 'plan: ay m/s2', [(p['t'], p['ay'], YELLOW, 'ay')], hr, 1.0, 'ahead s')
                meas = r.get('measured') or {}
                line1 = (f"{r['mode'].replace('_', ' ').upper()}  |  measured vx {meas.get('vx', float('nan')):.2f} m/s  "
                         f"plan {p['v'][min(10, len(p['v']) - 1)]:.2f} in 1 s, {p['v'][min(30, len(p['v']) - 1)]:.2f} m/s in 3 s  |  cap {r['max_speed']:.2f} m/s"
                         if p else f"{r['mode'].replace('_', ' ').upper()}  |  not on the mapped route: no plan")
                put(img, line1, (12, 26), 0.65, YELLOW if p else (80, 80, 255), 2)
                parts = []
                if r.get('route_s') is not None:
                    parts.append(f"route {r['route_s']:.0f}/{r['route_length']:.0f} m, lateral {r['lateral_error']:+.2f} m, "
                                 f"heading {np.degrees(r['heading_error']):+.1f} deg")
                if r.get('stop'):
                    parts.append(f"stop line #{r['stop']['id']}: {r['stop']['distance']:.1f} m ({r['stop']['state']})")
                if r.get('lead'):
                    L = r['lead']
                    parts.append(f"lead {L['class']}: gap {L['gap']:.1f} m (want {L['desired_gap']:.1f}), {L['speed']:.1f} m/s")
                if r.get('plan_ms') is not None:
                    parts.append(f"plan {r['plan_ms']:.1f} ms")
                put(img, '  |  '.join(parts), (12, 52), 0.5)
            cv2.imshow(window, img)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
