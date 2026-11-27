"""Rule-based trajectory prediction along the HD map lane graph (ROS-free; numpy + scipy).

Used by tools/prediction_models.Tracker as its `lanes` (same follow() interface as RouteLanes) for the '+hd' models,
with the aligned HD map MAP/hdmap/hdmap3d_enu.json (tools/align_hdmap.py). For a moving vehicle (cyclists and
pedestrians keep the tracker's prediction):

  1. Lane match: the lane centre line (lanes are directed, points in the drive direction) closest in a cost of
     lateral offset, heading difference and how well the lane's turn over the next 15 m fits the object's turn rate;
     it must be within half the lane width + 1 m and 35 degrees of the direction of motion, and the object must not
     be past its end. Otherwise None, and the tracker's own prediction is kept.
  2. Route through the lane graph: successors are followed until the horizon distance is covered. At a fork the
     branch whose heading change best matches what the object is doing is taken: expected turn = turn rate x the
     time to drive the branch (capped at 90 degrees) plus the current heading offset from the lane, so a car
     already turning is sent into the turn lane and a car going straight into the straightest branch.
  3. Speed: the current speed (a curve limit sqrt(LAT_ACCEL / curvature) is available but off: it made predictions
     worse on both recorded drives), and at mapped stop lines a stop: the object keeps its speed until it needs STOP_DECEL to stop at
     the line, brakes to it, waits STOP_WAIT_S, then accelerates back at RESUME_ACCEL. No stop is predicted when
     it would need more than MAX_STOP_DECEL (too close or fast: it is going through) or the line is under 1 m ahead.
  4. Lateral: the current offset from the lane centre fades out with LATERAL_FADE_S (it merges onto the lane).
  Past the end of the mapped lanes the path continues straight.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree

LAT_ACCEL = math.inf  # m/s^2 curve speed limit; off: on two drives every finite value (2, 3, 5) made predictions worse
CURVE_MIN = 0.02  # 1/m: gentler bends (radius over 50 m) do not slow anyone down
CURVE_DECEL = 2.0  # m/s^2
STOP_DECEL = 1.5  # m/s^2: braking for a stop line starts when this is needed
MAX_STOP_DECEL = 4.0  # m/s^2: needing more means the object is not going to stop there
STOP_WAIT_S = 1.5
RESUME_ACCEL = 1.5  # m/s^2
LATERAL_FADE_S = 1.5
MAX_ANGLE = math.radians(35)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Lane:
    def __init__(self, e):
        self.id, self.successors, self.kind = e['id'], e.get('successors') or [], e.get('lane_type', 'VEHICLE')
        self.width = float(e.get('width_m') or 3.5)
        xy = np.array(e['points'], float)[:, :2]
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-3]
        self.xy = smooth(xy[keep])
        seg = np.diff(self.xy, axis=0)
        self.s = np.r_[0.0, np.cumsum(np.linalg.norm(seg, axis=1))]
        self.length = float(self.s[-1])
        h = np.arctan2(seg[:, 1], seg[:, 0]) if len(seg) else np.zeros(1)
        self.heading = np.r_[h, h[-1]] if len(seg) else np.zeros(len(self.xy))
        self.stops = []  # arc lengths of stop lines on this lane

    def turn(self, upto_m=25.0):
        """Heading change over the first upto_m metres (left positive)."""
        k = int(np.searchsorted(self.s, min(upto_m, self.length)))
        return float(np.sum(wrap(np.diff(np.unwrap(self.heading[:max(k, 1) + 1]))))) if len(self.heading) > 1 else 0.0


class HDMapLanes:
    uses_turn_rate = True
    groups = ('vehicle',)  # cyclists ride at the road edge: lane centres made them worse (0.60 -> 0.74 m at 3 s)

    def __init__(self, source):
        hd = json.loads(Path(source).read_text()) if not isinstance(source, dict) else source
        layers = hd['layers']
        self.lanes = {e['id']: Lane(e) for e in layers.get('lane_centerline', []) if len(e['points']) >= 2}
        self.lanes = {k: v for k, v in self.lanes.items() if len(v.xy) >= 2}
        for stop in layers.get('stop_line', []):
            a, b = (np.array(p[:2], float) for p in (stop['points'][0], stop['points'][-1]))
            for lane in self.lanes.values():
                if lane.kind != 'VEHICLE':
                    continue
                i = crossing(lane.xy, a, b)
                if i is not None:
                    lane.stops.append(float(lane.s[i]))
        pts, owner, index = [], [], []
        for k, lane in self.lanes.items():
            pts.append(lane.xy)
            owner += [k] * len(lane.xy)
            index += list(range(len(lane.xy)))
        self.tree = cKDTree(np.vstack(pts))
        self.owner, self.index = np.array(owner), np.array(index)

    def match(self, x, y, heading, turn_rate=0.0, speed=5.0):
        """(lane, s, lateral) of the best lane for a position and direction of motion, or None. Lanes that agree
        with what the object is doing over the next 15 m are preferred (a car going straight is not put on the turn
        lane that starts at the same place)."""
        best = None
        seen = {}
        for j in self.tree.query_ball_point([x, y], 4.0):
            seen.setdefault(int(self.owner[j]), []).append(int(self.index[j]))
        for k, idx in seen.items():
            lane = self.lanes[k]
            i = min(idx, key=lambda i: (lane.xy[i, 0] - x) ** 2 + (lane.xy[i, 1] - y) ** 2)
            h = lane.heading[i]
            dx, dy = x - lane.xy[i, 0], y - lane.xy[i, 1]
            lateral = -math.sin(h) * dx + math.cos(h) * dy
            along = math.cos(h) * dx + math.sin(h) * dy
            dh = abs(wrap(heading - h))
            s_on = lane.s[i] + along
            if abs(lateral) > min(lane.width / 2 + 1.0, 3.0) or dh > MAX_ANGLE:
                continue
            if (s_on > lane.length + 0.3 and lane.successors) or (s_on < -0.3 and i == 0):
                continue  # past its end (on the next lane) or before its start
            s_on = float(np.clip(s_on, 0, lane.length))
            ahead = min(15.0, lane.length - s_on)
            j = min(int(np.searchsorted(lane.s, s_on + ahead)), len(lane.s) - 1)
            lane_turn = float(np.sum(wrap(np.diff(lane.heading[i:j + 1])))) if j > i else 0.0
            expected = float(np.clip(turn_rate * ahead / max(speed, 1.0) + wrap(heading - h), -math.pi / 2, math.pi / 2))
            cost = (lateral / 1.0) ** 2 + (dh / 0.25) ** 2 + ((lane_turn - expected) / 0.35) ** 2 + (2.0 if lane.kind != 'VEHICLE' else 0.0)
            if best is None or cost < best[0]:
                best = (cost, lane, s_on, lateral)
        return None if best is None else best[1:]

    def route(self, lane, s0, distance, heading_offset, turn_rate, speed):
        """Chain of lanes from (lane, s0) covering distance: [(lane, s_from, s_to)]."""
        chain, need = [(lane, s0, lane.length)], distance - (lane.length - s0)
        while need > 0:
            nxt = [self.lanes[k] for k in chain[-1][0].successors if k in self.lanes]
            if not nxt or len(chain) > 20:
                break
            if len(nxt) > 1:
                def miss(c):
                    t = min(c.length, 25.0) / max(speed, 1.0)
                    expected = float(np.clip(turn_rate * t + heading_offset, -math.pi / 2, math.pi / 2))
                    return abs(c.turn() - expected) + (1.0 if c.kind != 'VEHICLE' else 0.0)
                nxt.sort(key=miss)
            chain.append((nxt[0], 0.0, nxt[0].length))
            need -= nxt[0].length
        return chain

    def follow(self, x, y, heading, speed, horizon_s, step_s=0.1, turn_rate=0.0):
        m = self.match(x, y, heading, turn_rate, speed)
        if m is None:
            return None
        lane, s0, lateral = m
        offset = wrap(heading - lane.heading[min(int(np.searchsorted(lane.s, s0)), len(lane.s) - 1)])
        chain = self.route(lane, s0, speed * horizon_s + 30.0, offset, turn_rate, speed)
        # one polyline in path arc length u, with stop lines and curvature on it
        lx, ly, _ = along(lane.xy, lane.s, np.array([s0]), heading)
        xy, u, stops, base = [np.array([[lx[0], ly[0]]])], [0.0], [], 0.0  # u = 0 at the object's place on the lane
        for ln, a, b in chain:
            i, j = int(np.searchsorted(ln.s, a, side='right')), int(np.searchsorted(ln.s, b, side='right'))
            xy.append(ln.xy[i:j])
            u.extend(base + ln.s[i:j] - a)
            stops += [base + s - a for s in ln.stops if a <= s <= b]
            base += b - a
        xy, u = np.vstack(xy), np.array(u)
        keep = np.r_[True, np.diff(u) > 1e-3]
        xy, u = xy[keep], u[keep]
        hdg = np.unwrap(np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))) if len(u) > 1 else np.array([heading])
        curv = np.abs(np.gradient(hdg, u[1:])) if len(u) > 2 else np.zeros(len(hdg))
        curv = uniform_filter1d(curv, 7, mode='nearest')  # half-metre points: smooth over ~3 m
        v_curve = np.where(curv > CURVE_MIN, np.sqrt(LAT_ACCEL / np.maximum(curv, 1e-4)), 1e9)

        t_all = np.arange(0.0, horizon_s + 1e-9, step_s)
        pos, v, wait, stop_done, out_u = 0.0, speed, 0.0, False, []
        stop = next((s for s in stops if s > 1.0), None)
        if stop is not None and speed ** 2 / (2 * stop) > MAX_STOP_DECEL:
            stop = None  # cannot stop there any more: going through
        for _ in t_all:
            out_u.append(pos)
            target = speed
            if len(u) > 1:  # slow for curves ahead (lowest allowed speed within the braking distance)
                ahead = (u[1:] >= pos) & (u[1:] <= pos + v * v / (2 * CURVE_DECEL) + 5.0)
                if ahead.any():
                    lim = np.sqrt(v_curve[ahead] ** 2 + 2 * CURVE_DECEL * np.maximum(u[1:][ahead] - pos - 2.0, 0))
                    target = min(target, float(lim.min()))
            a = float(np.clip((target - v) / step_s, -CURVE_DECEL, RESUME_ACCEL))
            if stop is not None and not stop_done:
                d = stop - pos
                if wait > 0 or d <= 0.3 or (v < 0.3 and d < 2.0):
                    v, a = 0.0, 0.0
                    wait += step_s
                    if wait >= STOP_WAIT_S:
                        stop_done = True
                elif v * v / (2 * d) >= STOP_DECEL:
                    a = -v * v / (2 * d)
            v = max(v + a * step_s, 0.0)
            pos += v * step_s
        out_u = np.array(out_u)
        px, py, ph = along(xy, u, out_u, heading)
        fade = lateral * np.exp(-t_all / LATERAL_FADE_S)
        px, py = px - np.sin(ph) * fade, py + np.cos(ph) * fade
        return list(zip(t_all, px, py, ph))


def smooth(xy, step=0.5, window_m=4.0):
    """Resample every step metres and average over window_m (ends fixed, so connected lanes still meet): the
    mapped centre lines zigzag by a few centimetres, which shortens progress along them and fakes curvature."""
    if len(xy) < 3:
        return xy
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    u = np.linspace(0.0, s[-1], max(int(round(s[-1] / step)) + 1, 2))
    r = np.c_[np.interp(u, s, xy[:, 0]), np.interp(u, s, xy[:, 1])]
    k = int(window_m / step) // 2
    if len(r) <= 2 * k + 1:
        return r
    out = r.copy()
    for i in range(1, len(r) - 1):
        w = min(k, i, len(r) - 1 - i)  # shrink the window at the ends
        out[i] = r[i - w:i + w + 1].mean(axis=0)
    return out


def crossing(xy, a, b):
    """Index where the lane polyline crosses the stop line a-b (extended 0.5 m at each end) at a steep angle."""
    ab = b - a
    n = ab / max(np.linalg.norm(ab), 1e-9)
    a, b = a - 0.5 * n, b + 0.5 * n
    side = (xy[:, 0] - a[0]) * (b[1] - a[1]) - (xy[:, 1] - a[1]) * (b[0] - a[0])
    for i in np.nonzero((side[:-1] > 0) != (side[1:] > 0))[0]:  # also a point exactly on the line
        d = xy[i + 1] - xy[i]
        t = ((xy[i] - a) @ (b - a)) / max((b - a) @ (b - a), 1e-9)
        if 0.0 <= t <= 1.0 and abs(d @ n) / max(np.linalg.norm(d), 1e-9) < 0.5:  # within the line, lane crosses it at > 60 deg
            return int(i)
    return None


def seg_distance(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def along(xy, u, q, heading0):
    """Positions and headings at arc lengths q on the polyline (straight on past its end)."""
    if len(u) < 2:
        return xy[0, 0] + q * math.cos(heading0), xy[0, 1] + q * math.sin(heading0), np.full(len(q), heading0)
    seg_h = np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0]))
    k = np.clip(np.searchsorted(u, q, side='right') - 1, 0, len(u) - 2)
    r = q - u[k]
    h = seg_h[k]
    return xy[k, 0] + r * np.cos(h), xy[k, 1] + r * np.sin(h), h
