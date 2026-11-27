"""Mapped-route geometry (no ROS): the saved drive resampled by arc length, for planning and prediction.

Used by tools/route_planner.py, tools/object_prediction.py (--lanes) and tools/eval_prediction.py.
"""
import math
from pathlib import Path

import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Route:
    """Saved route resampled every 0.25 m: x, y, z, heading and smoothed curvature by arc length s."""

    def __init__(self, source, step_m=0.25):
        """source: MAP/gps_alignment/trajectory_enu.csv, or an (N, 3) array of route x, y, z in enu."""
        if isinstance(source, (str, Path)):
            t = np.genfromtxt(source, delimiter=',', names=True)
            source = np.c_[t['x_m'], t['y_m'], t['z_m']]
        source = np.asarray(source, float)
        xy, z = source[:, :2], source[:, 2]
        raw_s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        keep = np.r_[True, np.diff(raw_s) > 1e-3]
        self.raw_s = raw_s  # arc length of every original sample (stop lines refer to these indices)
        self.s = np.arange(0.0, raw_s[keep][-1], step_m)
        self.x = np.interp(self.s, raw_s[keep], xy[keep, 0])
        self.y = np.interp(self.s, raw_s[keep], xy[keep, 1])
        self.z = np.interp(self.s, raw_s[keep], z[keep])
        k = 8  # headings over +-2 m smooth the recorded localization noise
        dx = np.gradient(np.convolve(np.pad(self.x, k, mode='edge'), np.ones(2 * k + 1) / (2 * k + 1), 'valid'))
        dy = np.gradient(np.convolve(np.pad(self.y, k, mode='edge'), np.ones(2 * k + 1) / (2 * k + 1), 'valid'))
        self.heading = np.arctan2(dy, dx)
        kappa = np.gradient(np.unwrap(self.heading)) / step_m
        self.kappa = np.convolve(np.pad(kappa, k, mode='edge'), np.ones(2 * k + 1) / (2 * k + 1), 'valid')
        self.step = step_m
        self.length = float(self.s[-1])

    def at(self, s):
        i = np.clip(np.asarray(s) / self.step, 0, len(self.s) - 1)
        return (np.interp(i, np.arange(len(self.s)), self.x), np.interp(i, np.arange(len(self.s)), self.y),
                np.interp(i, np.arange(len(self.s)), self.z), self.heading[np.round(i).astype(int)],
                self.kappa[np.round(i).astype(int)])

    def project(self, x, y, yaw, s_prev=None, max_dist=5.0, max_heading=math.radians(60), ahead_m=30.0):
        """(s, signed lateral offset left+, heading error) of the nearest compatible route point, or None."""
        if s_prev is None:
            lo, hi = 0, len(self.s)
        else:  # stay near previous progress so crossings / the loop start do not jump
            lo, hi = int(max(0, (s_prev - 5) / self.step)), int(min(len(self.s), (s_prev + ahead_m) / self.step))
        d = np.hypot(self.x[lo:hi] - x, self.y[lo:hi] - y)
        ok = np.abs(wrap(self.heading[lo:hi] - yaw)) < max_heading
        if not ok.any():
            return None
        d = np.where(ok, d, np.inf)
        i = lo + int(np.argmin(d))
        if d[i - lo] > max_dist:
            return None
        h = self.heading[i]
        lateral = -math.sin(h) * (x - self.x[i]) + math.cos(h) * (y - self.y[i])
        along = math.cos(h) * (x - self.x[i]) + math.sin(h) * (y - self.y[i])
        return float(np.clip(self.s[i] + along, 0, self.length)), lateral, wrap(yaw - h)
