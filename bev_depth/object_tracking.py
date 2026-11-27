"""Object tracking with persistent IDs, in two levels.

1. ``CameraTrackIds``: ByteTrack on each camera's image boxes. Stable
   image-space IDs within one camera, at well under a millisecond per frame.
2. ``BEVObjectTracker``: one tracker for the fused BEV, which owns the IDs
   that are shown and logged. Each object has a constant-velocity Kalman
   filter in the ego frame (x lateral, z forward, meters). Detections are
   associated with a Mahalanobis gate and the Hungarian algorithm, using a
   measurement covariance that grows with range because monocular depth does.
   A detection whose per-camera ByteTrack ID is already bound to a global
   track stays on that track (track-to-track hint), which keeps the ID when
   depth jumps or when an object moves between cameras.

Track lifecycle: tentative -> confirmed after ``min_hits`` detections ->
coasting (predicted only) while unseen -> deleted after ``max_coast_s``.
IDs are never reused within a run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .types import Detection, ProjectedDetection

CLASS_GROUPS = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "motorcycle": "vehicle",
    "bicycle": "bicycle",
    "person": "person",
    "stop sign": "static",
    "traffic light": "static",
    "red": "static",
    "yellow": "static",
    "green": "static",
}


def class_group(class_name: str) -> str:
    return CLASS_GROUPS.get(class_name, class_name)


class _Boxes:
    """The minimal Results-like view Ultralytics' BYTETracker reads."""

    def __init__(self, conf: np.ndarray, xywh: np.ndarray, cls: np.ndarray) -> None:
        self.conf, self.xywh, self.cls = conf, xywh, cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, index) -> "_Boxes":
        return _Boxes(self.conf[index], self.xywh[index], self.cls[index])


class CameraTrackIds:
    """ByteTrack for one camera: returns a stable image-space ID per detection."""

    def __init__(
        self,
        track_high_thresh: float = 0.25,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.25,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        self._tracker = BYTETracker(
            SimpleNamespace(
                track_high_thresh=track_high_thresh,
                track_low_thresh=track_low_thresh,
                new_track_thresh=new_track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                fuse_score=True,
            )
        )
        self._class_ids: dict[str, int] = {}

    def assign(self, detections: Sequence[Detection]) -> list[int | None]:
        """ByteTrack ID for each detection, or None while a new track is unconfirmed."""
        if not detections:
            self._tracker.update(_Boxes(np.zeros(0), np.zeros((0, 4)), np.zeros(0)))
            return []
        xyxy = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
        xywh = np.column_stack(
            ((xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
             xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1])
        )
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        cls = np.array(
            [self._class_ids.setdefault(d.class_name, len(self._class_ids)) for d in detections],
            dtype=np.float32,
        )
        ids: list[int | None] = [None] * len(detections)
        for row in self._tracker.update(_Boxes(conf, xywh, cls)):
            ids[int(row[7])] = int(row[4])  # row = x1, y1, x2, y2, id, score, cls, det index
        return ids


@dataclass(frozen=True)
class TrackerConfig:
    min_hits: int = 3                 # detections before a track is confirmed and shown
    tentative_max_misses: int = 2     # unconfirmed tracks die after this many misses in a row
    max_coast_s: float = 1.0          # confirmed tracks are kept this long without detections
    gate_chi2: float = 9.21           # squared Mahalanobis gate: 99% for 2 degrees of freedom
    hint_gate_chi2: float = 25.0      # looser gate when the camera-track hint agrees
    # Monocular depth error grows with range: sigma = base + per_m * range.
    range_sigma_base_m: float = 0.5
    range_sigma_per_m: float = 0.10
    bearing_sigma_rad: float = 0.03   # about 1.7 degrees of box-center jitter
    accel_sigma: dict[str, float] = field(default_factory=lambda: {
        "vehicle": 3.0, "bicycle": 2.0, "person": 1.5, "static": 0.3,
    })
    default_accel_sigma: float = 2.0
    max_speed_mps: float = 45.0       # clamp for unphysical velocity estimates
    yaw_rate_sign: float = 1.0        # +1 if the IMU's positive z-rate is a left turn


@dataclass
class _Track:
    track_id: int
    group: str
    x: np.ndarray                      # state: x, z, vx, vz
    P: np.ndarray
    last_update_s: float
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    class_votes: dict[str, float] = field(default_factory=dict)
    last_detection: ProjectedDetection | None = None

    @property
    def class_name(self) -> str:
        return max(self.class_votes, key=self.class_votes.get)


@dataclass(frozen=True)
class TrackOutput:
    track_id: int
    class_name: str
    x_m_lateral: float
    z_m_forward: float
    vx_mps: float
    vz_mps: float
    coasting: bool
    hits: int
    detection: ProjectedDetection | None  # latest associated detection


class BEVObjectTracker:
    """Global multi-object tracker on the fused BEV object list."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.cfg = config or TrackerConfig()
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._hints: dict[tuple[str, int], int] = {}   # (camera, ByteTrack id) -> track id
        self._last_time_s: float | None = None

    # ---- geometry -----------------------------------------------------------
    def _measurement_cov(self, x: float, z: float) -> np.ndarray:
        rng = max(math.hypot(x, z), 0.5)
        sr = self.cfg.range_sigma_base_m + self.cfg.range_sigma_per_m * rng
        st = rng * self.cfg.bearing_sigma_rad
        c, s = z / rng, x / rng       # unit vector along the line of sight (x, z)
        rot = np.array([[s, c], [c, -s]])   # columns: radial, tangential
        return rot @ np.diag([sr * sr, st * st]) @ rot.T

    def _predict(self, track: _Track, dt: float) -> None:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = self.cfg.accel_sigma.get(track.group, self.cfg.default_accel_sigma) ** 2
        g = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * q
        Q = np.zeros((4, 4))
        Q[np.ix_([0, 2], [0, 2])] = g
        Q[np.ix_([1, 3], [1, 3])] = g
        track.x = F @ track.x
        track.P = F @ track.P @ F.T + Q

    def _compensate_ego(self, track: _Track, speed_mps: float, yaw_rate_rps: float, dt: float) -> None:
        """Move a track into the current ego frame after the car drove and turned."""
        dpsi = self.cfg.yaw_rate_sign * yaw_rate_rps * dt   # positive = turned left
        c, s = math.cos(dpsi), math.sin(dpsi)
        R = np.array([[c, s], [-s, c]])   # rotate points clockwise for a left turn
        pos = R @ (track.x[:2] - np.array([0.0, speed_mps * dt]))
        vel = R @ track.x[2:]
        track.x = np.concatenate([pos, vel])
        T = np.zeros((4, 4))
        T[:2, :2] = T[2:, 2:] = R
        track.P = T @ track.P @ T.T

    def _mahalanobis2(self, track: _Track, z: np.ndarray, R: np.ndarray) -> tuple[float, np.ndarray]:
        S = track.P[:2, :2] + R
        innovation = z - track.x[:2]
        return float(innovation @ np.linalg.solve(S, innovation)), S

    def _update(self, track: _Track, det: ProjectedDetection, now_s: float) -> None:
        z = np.array([det.x_m_lateral, det.z_m_forward])
        R = self._measurement_cov(*z)
        S = track.P[:2, :2] + R
        K = track.P[:, :2] @ np.linalg.inv(S)
        track.x = track.x + K @ (z - track.x[:2])
        track.P = (np.eye(4) - K @ np.hstack([np.eye(2), np.zeros((2, 2))])) @ track.P
        speed = math.hypot(track.x[2], track.x[3])
        if speed > self.cfg.max_speed_mps:
            track.x[2:] *= self.cfg.max_speed_mps / speed
        track.hits += 1
        track.misses = 0
        track.last_update_s = now_s
        track.last_detection = det
        name = det.detection.class_name
        track.class_votes[name] = track.class_votes.get(name, 0.0) + det.detection.confidence
        if track.hits >= self.cfg.min_hits:
            track.confirmed = True

    def _new_track(self, det: ProjectedDetection, now_s: float) -> _Track:
        z = np.array([det.x_m_lateral, det.z_m_forward])
        P = np.zeros((4, 4))
        P[:2, :2] = self._measurement_cov(*z)
        P[2, 2] = P[3, 3] = 10.0 ** 2      # unknown velocity at birth
        track = _Track(
            track_id=self._next_id,
            group=class_group(det.detection.class_name),
            x=np.array([z[0], z[1], 0.0, 0.0]),
            P=P,
            last_update_s=now_s,
            class_votes={det.detection.class_name: det.detection.confidence},
            last_detection=det,
        )
        self._next_id += 1
        track.confirmed = track.hits >= self.cfg.min_hits
        return track

    # ---- main entry ---------------------------------------------------------
    def step(
        self,
        detections: Sequence[ProjectedDetection],
        now_s: float,
        ego_speed_mps: float | None = None,
        ego_yaw_rate_rps: float | None = None,
    ) -> tuple[list[TrackOutput], list[int | None]]:
        """Advance to ``now_s`` and associate this frame's fused detections.

        Returns the confirmed tracks and, for each input detection, the ID of the
        confirmed track it was assigned to (None if unassigned or still tentative).
        """
        dt = 0.0 if self._last_time_s is None else max(0.0, now_s - self._last_time_s)
        self._last_time_s = now_s
        for track in self._tracks:
            if dt > 0.0:
                if ego_speed_mps is not None and ego_yaw_rate_rps is not None:
                    self._compensate_ego(track, ego_speed_mps, ego_yaw_rate_rps, dt)
                self._predict(track, dt)

        valid = [
            i for i, d in enumerate(detections)
            if d.x_m_lateral is not None and d.z_m_forward is not None
        ]
        assigned: dict[int, int] = {}          # detection index -> track index
        taken: set[int] = set()

        # 1. Track-to-track hint: same camera track as last time, loose gate.
        by_id = {t.track_id: k for k, t in enumerate(self._tracks)}
        for i in valid:
            det = detections[i]
            key = (det.camera_name, det.camera_track_id)
            k = by_id.get(self._hints.get(key)) if det.camera_track_id is not None else None
            if k is None or k in taken:
                continue
            track = self._tracks[k]
            if track.group != class_group(det.detection.class_name):
                continue
            z = np.array([det.x_m_lateral, det.z_m_forward])
            d2, _ = self._mahalanobis2(track, z, self._measurement_cov(*z))
            if d2 <= self.cfg.hint_gate_chi2:
                assigned[i] = k
                taken.add(k)

        # 2. Global assignment (Hungarian) for the rest, gated by Mahalanobis distance.
        rest_d = [i for i in valid if i not in assigned]
        rest_t = [k for k in range(len(self._tracks)) if k not in taken]
        if rest_d and rest_t:
            big = 1e6
            cost = np.full((len(rest_d), len(rest_t)), big)
            for a, i in enumerate(rest_d):
                det = detections[i]
                z = np.array([det.x_m_lateral, det.z_m_forward])
                R = self._measurement_cov(*z)
                group = class_group(det.detection.class_name)
                for b, k in enumerate(rest_t):
                    track = self._tracks[k]
                    if track.group != group:
                        continue
                    d2, _ = self._mahalanobis2(track, z, R)
                    if d2 <= self.cfg.gate_chi2:
                        cost[a, b] = d2
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if cost[a, b] < big:
                    assigned[rest_d[a]] = rest_t[b]
                    taken.add(rest_t[b])

        # 3. Update matched tracks and remember their camera-track IDs.
        for i, k in assigned.items():
            self._update(self._tracks[k], detections[i], now_s)
            det = detections[i]
            if det.camera_track_id is not None:
                self._hints[(det.camera_name, det.camera_track_id)] = self._tracks[k].track_id

        # 4. Age unmatched tracks; drop dead ones.
        survivors: list[_Track] = []
        for k, track in enumerate(self._tracks):
            if k not in taken:
                track.misses += 1
            if not track.confirmed and track.misses > self.cfg.tentative_max_misses:
                continue
            if track.confirmed and now_s - track.last_update_s > self.cfg.max_coast_s:
                continue
            survivors.append(track)

        # 5. Unmatched detections start tentative tracks.
        det_track: dict[int, _Track] = {i: self._tracks[k] for i, k in assigned.items()}
        for i in valid:
            if i not in assigned:
                track = self._new_track(detections[i], now_s)
                survivors.append(track)
                det_track[i] = track
                det = detections[i]
                if det.camera_track_id is not None:
                    self._hints[(det.camera_name, det.camera_track_id)] = track.track_id
        self._tracks = survivors

        alive = {t.track_id for t in self._tracks}
        self._hints = {key: tid for key, tid in self._hints.items() if tid in alive}

        ids: list[int | None] = [None] * len(detections)
        for i, track in det_track.items():
            if track.confirmed and track.track_id in alive:
                ids[i] = track.track_id
        outputs = [
            TrackOutput(
                track_id=t.track_id,
                class_name=t.class_name,
                x_m_lateral=float(t.x[0]),
                z_m_forward=float(t.x[1]),
                vx_mps=float(t.x[2]),
                vz_mps=float(t.x[3]),
                coasting=t.misses > 0,
                hits=t.hits,
                detection=t.last_detection,
            )
            for t in self._tracks
            if t.confirmed
        ]
        return outputs, ids

    def reset(self) -> None:
        self._tracks.clear()
        self._hints.clear()
        self._last_time_s = None
