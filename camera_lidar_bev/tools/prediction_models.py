"""Object tracking and motion prediction shared by tools/object_prediction.py (live) and
tools/eval_prediction.py (offline benchmark on recorded drives).

Trackers (one per detected object, positions in enu after removing the car's own motion):
  * 'cv'  : constant-velocity Kalman filter on x, y, vx, vy (the first live version). Prediction
            continues with the recent velocity-heading turn rate, fading over --turn-fade-s.
  * 'imm' : interacting multiple model filter on x, y, speed, heading, turn rate with two models,
            straight (turn rate held at 0) and coordinated turn (turn rate random walk), blended by how
            well each explains the detections. Vehicles and cyclists also use CenterPoint's box heading
            as a measurement (its front/back ambiguity is resolved against the track heading), which
            gives a heading before the object has moved far. Predicted turns stop after 90 degrees.
Cyclists and pedestrians always use 'cv': their box heading is unreliable (IMM made cyclists 0.6 -> 3.8 m worse
at 3 s in tools/eval_prediction.py).
Models are looked up by name in MODELS (register_model adds your own). Detections are tuples
(group, class, enu position (3), size (3), enu box yaw). Lane following ('+map'): a moving vehicle or cyclist within 6 m of the mapped route and heading along it
(same or opposite direction, within 35 degrees) is predicted along the route at its lateral offset;
anything else keeps the tracker's own prediction.
"""
from collections import deque
import itertools
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

GROUPS = {'car': 'vehicle', 'truck': 'vehicle', 'bus': 'vehicle', 'trailer': 'vehicle',
          'construction_vehicle': 'vehicle', 'bicycle': 'cyclist', 'motorcycle': 'cyclist', 'person': 'pedestrian'}
ACCEL_NOISE = {'vehicle': 2.0, 'cyclist': 1.5, 'pedestrian': 1.0}  # m/s^2, white-noise acceleration
MEAS_STD = {'vehicle': 0.4, 'cyclist': 0.3, 'pedestrian': 0.25}  # m, box centre
MIN_SPEED = {'vehicle': 1.5, 'cyclist': 1.0, 'pedestrian': 0.5}  # m/s; slower tracks are predicted to stay
GATE_CHI2 = 9.21  # 99 % for 2 degrees of freedom
VELOCITY_LAG_S = 0.7  # 'cv': its velocity heading trails a turning object (synthetic turning-car check)
IMM_SWITCH = np.array([[0.97, 0.03], [0.03, 0.97]])  # per update (~18 Hz): straight <-> turning
YAW_STD = 0.15  # rad, CenterPoint box heading
MAX_TURN = math.pi / 2  # predicted turns end after 90 degrees (intersection turns)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def detections_in_enu(msg, rot, pos, min_score=0.35, max_range_m=80.0):
    """[(group, class, enu position (3), size (3), enu box yaw)] for trackable classes."""
    ego_yaw = math.atan2(rot[1, 0], rot[0, 0])
    out = []
    for d in msg.detections:
        if not d.results or d.results[0].hypothesis.score < min_score:
            continue
        cls = d.results[0].hypothesis.class_id
        group = GROUPS.get(cls)
        c = d.bbox.center.position
        if group is None or math.hypot(c.x, c.y) > max_range_m:
            continue
        q = d.bbox.center.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        out.append((group, cls, rot @ np.array([c.x, c.y, c.z]) + pos, (d.bbox.size.x, d.bbox.size.y, d.bbox.size.z),
                    wrap(yaw + ego_yaw)))
    return out


def cv_model(dt, sigma_a):
    f = np.eye(4)
    f[0, 2] = f[1, 3] = dt
    g = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
    return f, sigma_a ** 2 * g @ g.T


def radius(p2):
    return min(2.5, 2 * math.sqrt(max(np.linalg.eigvalsh(p2)[-1], 0.0)))


class CVTrack:
    def __init__(self, t, det, turn_fade_s=4.0):
        group, cls, pos, size, yaw = det
        self.group, self.cls, self.turn_fade_s = group, cls, turn_fade_s
        self.x = np.array([pos[0], pos[1], 0.0, 0.0])
        self.P = np.diag([MEAS_STD[group] ** 2] * 2 + [4.0 ** 2] * 2)  # unknown velocity at birth
        self.t = t
        self.z, self.size = pos[2], size
        self.headings = deque()  # (t, velocity heading) while moving, for the turn rate
        self.turn_rate = 0.0

    def predict(self, t):
        f, q = cv_model(t - self.t, ACCEL_NOISE[self.group])
        self.x, self.P, self.t = f @ self.x, f @ self.P @ f.T + q, t

    def position(self):
        return self.x[:2]

    def position_cov(self):
        return self.P[:2, :2]

    def velocity(self):
        return self.x[2:4]

    def update(self, t, det):
        _, self.cls, pos, self.size, _ = det
        self.z = pos[2]
        s = self.P[:2, :2] + np.eye(2) * MEAS_STD[self.group] ** 2
        k = self.P[:, :2] @ np.linalg.inv(s)
        self.x = self.x + k @ (pos[:2] - self.x[:2])
        self.P = (np.eye(4) - k @ np.eye(2, 4)) @ self.P
        if math.hypot(self.x[2], self.x[3]) > 1.5:
            self.headings.append((t, math.atan2(self.x[3], self.x[2])))
        while self.headings and t - self.headings[0][0] > 1.0:
            self.headings.popleft()
        if len(self.headings) >= 5 and self.headings[-1][0] - self.headings[0][0] > 0.5:
            h = np.array(self.headings)
            rate = float(np.polyfit(h[:, 0] - h[-1, 0], np.unwrap(h[:, 1]), 1)[0])
            self.turn_rate += 0.3 * (float(np.clip(rate, -0.6, 0.6)) - self.turn_rate)
        else:
            self.turn_rate *= 0.8

    def future(self, horizon_s, step_s=0.1):
        """[(t, x, y, heading, 2-sigma radius)] every step over the horizon."""
        x, y, vx, vy = self.x
        speed, heading = math.hypot(vx, vy), math.atan2(vy, vx)
        turn = self.turn_rate if self.group != 'pedestrian' and abs(self.turn_rate) > 0.03 else 0.0
        heading += turn * VELOCITY_LAG_S
        f, q = cv_model(step_s, ACCEL_NOISE[self.group])
        p = self.P.copy()
        out = [(0.0, x, y, heading, radius(p[:2, :2]))]
        for k in range(1, int(round(horizon_s / step_s)) + 1):
            t = k * step_s
            heading += turn * math.exp(-t / self.turn_fade_s) * step_s
            x, y = x + speed * math.cos(heading) * step_s, y + speed * math.sin(heading) * step_s
            p = f @ p @ f.T + q
            out.append((t, x, y, heading, radius(p[:2, :2])))
        return out


class IMMTrack:
    """Straight + coordinated-turn IMM on [x, y, v, heading, turn rate]."""
    TURN_NOISE = (0.02, 0.6)  # rad/s^2 per model

    def __init__(self, t, det):
        group, cls, pos, size, yaw = det
        self.group, self.cls, self.t, self.z, self.size = group, cls, t, pos[2], size
        s0 = np.array([pos[0], pos[1], 0.0, yaw, 0.0])
        p0 = np.diag([MEAS_STD[group] ** 2] * 2 + [4.0 ** 2, 0.5 ** 2, 0.3 ** 2])
        self.s, self.P, self.mu = [s0.copy(), s0.copy()], [p0.copy(), p0.copy()], np.array([0.8, 0.2])
        self.use_yaw = group != 'pedestrian'

    @staticmethod
    def step(s, dt, turning):
        x, y, v, h, w = s
        if not turning:
            w = 0.0
        if abs(w) > 1e-4:
            x2, y2 = x + v / w * (math.sin(h + w * dt) - math.sin(h)), y + v / w * (math.cos(h) - math.cos(h + w * dt))
        else:
            x2, y2 = x + v * math.cos(h) * dt, y + v * math.sin(h) * dt
        return np.array([x2, y2, v, h + w * dt, w])

    @staticmethod
    def jacobian(s, dt, turning):
        x, y, v, h, w = s
        f = np.eye(5)
        if not turning:
            f[4, 4] = 0.0
            w = 0.0
        if abs(w) > 1e-4:
            s1, c1, s0, c0 = math.sin(h + w * dt), math.cos(h + w * dt), math.sin(h), math.cos(h)
            f[0, 2], f[0, 3] = (s1 - s0) / w, v / w * (c1 - c0)
            f[1, 2], f[1, 3] = (c0 - c1) / w, v / w * (s1 - s0)
            f[0, 4] = v / w * dt * c1 - v / (w * w) * (s1 - s0)
            f[1, 4] = v / w * dt * s1 - v / (w * w) * (c0 - c1)
        else:
            f[0, 2], f[0, 3] = math.cos(h) * dt, -v * math.sin(h) * dt
            f[1, 2], f[1, 3] = math.sin(h) * dt, v * math.cos(h) * dt
            if turning:
                f[0, 4], f[1, 4] = -v * math.sin(h) * dt * dt / 2, v * math.cos(h) * dt * dt / 2
        f[3, 4] = dt if turning else 0.0
        return f

    def noise(self, s, dt, j):
        h = s[3]
        g = np.array([[dt * dt / 2 * math.cos(h), 0], [dt * dt / 2 * math.sin(h), 0], [dt, 0], [0, dt * dt / 2], [0, dt]])
        return g @ np.diag([ACCEL_NOISE[self.group] ** 2, self.TURN_NOISE[j] ** 2]) @ g.T

    def combined(self):
        ref = self.s[int(np.argmax(self.mu))][3]
        s = sum(m * np.r_[si[:3], ref + wrap(si[3] - ref), si[4]] for m, si in zip(self.mu, self.s))
        p = sum(m * (pi + np.outer(si - s, si - s)) for m, si, pi in
                zip(self.mu, [np.r_[si[:3], ref + wrap(si[3] - ref), si[4]] for si in self.s], self.P))
        return s, p

    def predict(self, t):
        dt = t - self.t
        c = IMM_SWITCH.T @ self.mu
        mix = IMM_SWITCH * self.mu[:, None] / c[None, :]
        new_s, new_p = [], []
        for j in range(2):
            ref = self.s[j][3]
            aligned = [np.r_[si[:3], ref + wrap(si[3] - ref), si[4]] for si in self.s]
            s0 = sum(mix[i, j] * aligned[i] for i in range(2))
            p0 = sum(mix[i, j] * (self.P[i] + np.outer(aligned[i] - s0, aligned[i] - s0)) for i in range(2))
            f = self.jacobian(s0, dt, j == 1)
            new_s.append(self.step(s0, dt, j == 1))
            new_p.append(f @ p0 @ f.T + self.noise(s0, dt, j))
        self.s, self.P, self.c, self.t = new_s, new_p, c, t

    def position(self):
        return self.combined()[0][:2]

    def position_cov(self):
        return self.combined()[1][:2, :2]

    def velocity(self):
        s = self.combined()[0]
        return np.array([s[2] * math.cos(s[3]), s[2] * math.sin(s[3])])

    def update(self, t, det):
        _, self.cls, pos, self.size, yaw = det
        self.z = pos[2]
        heading = self.combined()[0][3]
        if abs(wrap(yaw - heading)) > math.pi / 2:
            yaw = wrap(yaw + math.pi)  # box heading is front/back ambiguous
        n = 3 if self.use_yaw else 2
        h = np.zeros((n, 5))
        h[0, 0] = h[1, 1] = 1.0
        r = np.eye(n) * MEAS_STD[self.group] ** 2
        if self.use_yaw:
            h[2, 3], r[2, 2] = 1.0, YAW_STD ** 2
        like = np.zeros(2)
        for j in range(2):
            y = np.r_[pos[:2] - self.s[j][:2], [wrap(yaw - self.s[j][3])] if self.use_yaw else []]
            s = h @ self.P[j] @ h.T + r
            si = np.linalg.inv(s)
            k = self.P[j] @ h.T @ si
            self.s[j] = self.s[j] + k @ y
            self.P[j] = (np.eye(5) - k @ h) @ self.P[j]
            like[j] = math.exp(-0.5 * float(y @ si @ y)) / math.sqrt((2 * math.pi) ** n * np.linalg.det(s))
        mu = self.c * like + 1e-12
        self.mu = np.maximum(mu / mu.sum(), 1e-3)
        self.mu /= self.mu.sum()

    def future(self, horizon_s, step_s=0.1):
        """Each model rolled forward (turns end after 90 degrees), positions blended by model probability."""
        paths = []
        for j in range(2):
            s, p, start = self.s[j].copy(), self.P[j].copy(), self.s[j][3]
            path = [(s[0], s[1], s[3], radius(p[:2, :2]))]
            for _ in range(int(round(horizon_s / step_s))):
                turning = j == 1 and abs(wrap(s[3] - start)) < MAX_TURN
                f = self.jacobian(s, step_s, turning)
                p = f @ p @ f.T + self.noise(s, step_s, j)
                s = self.step(s, step_s, turning)
                path.append((s[0], s[1], s[3], radius(p[:2, :2])))
            paths.append(np.array(path))
        a = self.mu[0] * paths[0] + self.mu[1] * paths[1]
        a[:, 2] = np.where(self.mu[0] >= self.mu[1], paths[0][:, 2], paths[1][:, 2])
        return [(k * step_s, *row) for k, row in enumerate(a)]


class RouteLanes:
    """Predict along the mapped route for vehicles driving along it (same or opposite direction)."""

    def __init__(self, route, max_offset_m=6.0, max_angle=math.radians(35)):
        self.r, self.max_offset, self.max_angle = route, max_offset_m, max_angle

    def follow(self, x, y, heading, speed, horizon_s, step_s=0.1):
        r = self.r
        d = np.hypot(r.x - x, r.y - y)
        i = int(np.argmin(d))
        if d[i] > self.max_offset:
            return None
        h = r.heading[i]
        if abs(wrap(heading - h)) < self.max_angle:
            direction = 1.0
        elif abs(wrap(heading - h - math.pi)) < self.max_angle:
            direction = -1.0
        else:
            return None
        lateral = -math.sin(h) * (x - r.x[i]) + math.cos(h) * (y - r.y[i])
        s0 = r.s[i] + math.cos(h) * (x - r.x[i]) + math.sin(h) * (y - r.y[i])
        t = np.arange(0.0, horizon_s + 1e-9, step_s)
        s = s0 + direction * speed * t
        inside = (s >= 0) & (s <= r.length)
        px, py, _, ph, _ = r.at(np.clip(s, 0, r.length))
        px, py = px - np.sin(ph) * lateral, py + np.cos(ph) * lateral
        heading_out = ph + (math.pi if direction < 0 else 0.0)
        if not inside.all():  # past the end of the mapped route: continue straight
            last = int(np.argmax(~inside)) - 1 if (~inside).any() else len(s) - 1
            last = max(last, 0)
            for k in range(last + 1, len(s)):
                dt = (k - last) * step_s
                px[k] = px[last] + speed * math.cos(heading_out[last]) * dt
                py[k] = py[last] + speed * math.sin(heading_out[last]) * dt
                heading_out[k] = heading_out[last]
        return list(zip(t, px, py, heading_out))


class Tracker:
    """Detections -> tracks (Mahalanobis gate + Hungarian per class group), with a chosen motion model."""

    def __init__(self, model='imm', confirm_hits=3, max_coast_s=1.0, lanes=None):
        if model not in MODELS:
            raise ValueError(f'unknown model {model!r}; registered: {sorted(MODELS)}')
        self.model, self.confirm_hits, self.max_coast_s, self.lanes = model, confirm_hits, max_coast_s, lanes
        self.tracks, self.ids = [], itertools.count(1)

    def new_track(self, t, det):
        tr = MODELS[self.model](t, det)
        tr.id, tr.hits, tr.last_seen, tr.last_det = next(self.ids), 1, t, det
        return tr

    def step(self, t, dets):
        """Advance to time t with this frame's detections; returns the tracks matched this frame."""
        for tr in self.tracks:
            tr.predict(t)
        matched, used = [], set()
        for group in MIN_SPEED:
            ti = [i for i, tr in enumerate(self.tracks) if tr.group == group]
            di = [j for j, d in enumerate(dets) if d[0] == group]
            if not ti or not di:
                continue
            pos = np.array([self.tracks[i].position() for i in ti])
            cov = np.array([self.tracks[i].position_cov() + np.eye(2) * MEAS_STD[group] ** 2 for i in ti])
            meas = np.array([dets[j][2][:2] for j in di])
            diff = meas[None, :, :] - pos[:, None, :]
            md = np.einsum('tdi,tij,tdj->td', diff, np.linalg.inv(cov), diff)
            cost = np.where((md < GATE_CHI2) & (np.linalg.norm(diff, axis=2) < 5.0), md, 1e6)
            for a, b in zip(*linear_sum_assignment(cost)):
                if cost[a, b] < 1e6:
                    tr = self.tracks[ti[a]]
                    tr.update(t, dets[di[b]])
                    tr.hits, tr.last_seen, tr.last_det = tr.hits + 1, t, dets[di[b]]
                    matched.append(tr)
                    used.add(di[b])
        for j, d in enumerate(dets):
            if j not in used:
                self.tracks.append(self.new_track(t, d))
        self.tracks = [tr for tr in self.tracks if t - tr.last_seen <= self.max_coast_s]
        return matched

    def confirmed(self, t, max_unseen_s=0.3):
        return [tr for tr in self.tracks if tr.hits >= self.confirm_hits and t - tr.last_seen <= max_unseen_s]

    def predict_track(self, tr, horizon_s, step_s=0.1):
        """(moving, [(t, x, y, heading, radius)]); stationary tracks get their current position."""
        vel = tr.velocity()
        speed = float(np.hypot(*vel))
        x, y = tr.position()
        if speed < MIN_SPEED[tr.group]:
            return False, [(k * step_s, x, y, 0.0, 0.0) for k in range(int(round(horizon_s / step_s)) + 1)]
        path = tr.future(horizon_s, step_s)
        if self.lanes is not None and tr.group != 'pedestrian':
            lane = self.lanes.follow(x, y, math.atan2(vel[1], vel[0]), speed, horizon_s, step_s)  # direction of motion
            if lane is not None:
                path = [(t, px, py, ph, r) for (t, px, py, ph), (_, _, _, _, r) in zip(lane, path)]
        return True, path


# Motion models by name: factory(t, detection) -> track. A track needs .group, .cls, .z, .size and
# predict(t), update(t, det), position() (x, y), position_cov() (2x2), velocity() (vx, vy) and
# future(horizon_s, step_s) -> [(t, x, y, heading, 2-sigma radius)]. Add your own with register_model;
# see docs/prediction_development.md.
MODELS = {
    'cv': lambda t, det: CVTrack(t, det),
    'imm': lambda t, det: IMMTrack(t, det) if det[0] == 'vehicle' else CVTrack(t, det),
}


def register_model(name, factory):
    MODELS[name] = factory
