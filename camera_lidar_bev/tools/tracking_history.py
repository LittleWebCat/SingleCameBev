"""Bounded diagnostic history; invalid targets are gaps rather than zeros."""
from collections import deque
import math


class TrackingHistory:
    def __init__(self, window_s=10., max_samples=4000):
        if not math.isfinite(window_s) or window_s <= 0 or max_samples < 2:
            raise ValueError('invalid tracking history limits')
        self.window_s = float(window_s)
        self.samples = deque(maxlen=max_samples)
        self.vehicle_samples = deque(maxlen=max_samples)
        self.now_ns = None
        self.last_source_stamp = None
        self.available = False
        self.latest = (None, None)
        self.vehicle_available = False
        self.latest_vehicle = None

    def update(self, now_ns, fields=None):
        now_ns = int(now_ns)
        if self.now_ns is not None and now_ns < self.now_ns:
            self.samples.clear()
            self.vehicle_samples.clear()
            self.last_source_stamp = None
            self.available = False
            self.vehicle_available = False
        self.now_ns = now_ns
        values = (None, None)
        valid = False
        vehicle = None
        fresh = False
        stamp = None
        if fields and fields.get('tracking_valid') is True:
            try:
                stamp = fields['source_stamp_ns']
                fresh = (isinstance(stamp, int) and not isinstance(stamp, bool) and
                         stamp > 0 and -.05 <= (now_ns-stamp)/1e9 <= .5)
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
        if fresh and self.last_source_stamp is not None and stamp < self.last_source_stamp:
            fresh = False
        if fresh and fields.get('targets_valid') is True:
            try:
                curvature = float(fields['curvature_per_m'])
                speed = float(fields['target_speed_mps'])
                valid = math.isfinite(curvature) and math.isfinite(speed) and speed >= 0
                if valid:
                    values = curvature, speed
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
        if (fresh and fields.get('vehicle_curvature_valid') is True and
                fields.get('vehicle_transform_verified') is True and
                fields.get('vehicle_curvature_source') == 'localization_pose_estimate'):
            try:
                candidate = float(fields['vehicle_curvature_per_m'])
                if math.isfinite(candidate):
                    vehicle = candidate
            except (KeyError, TypeError, ValueError, OverflowError):
                pass
        if fresh and stamp != self.last_source_stamp:
            self.samples.append((stamp, *values))
            self.vehicle_samples.append((stamp, vehicle))
            self.last_source_stamp = stamp
        elif not fresh and (self.available or self.vehicle_available or not self.samples):
            self.samples.append((now_ns, None, None))
            self.vehicle_samples.append((now_ns, None))
        self.available = valid
        self.latest = values
        self.vehicle_available = vehicle is not None
        self.latest_vehicle = vehicle
        cutoff = now_ns-int(self.window_s*1e9)
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()
        while self.vehicle_samples and self.vehicle_samples[0][0] < cutoff:
            self.vehicle_samples.popleft()
        return self
