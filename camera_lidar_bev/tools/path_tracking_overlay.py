"""Validate source-paired tracking diagnostics and transform ENU into sensor BEV."""
import math
import numpy as np


def source_transform(saved_map, live_map, saved_pose, live_pose):
    if live_map is not None and live_pose is not None:
        return np.asarray(live_pose['transform'], float), live_pose['query_stamp_ns']
    if saved_map is not None and saved_pose is not None and saved_map.enu_from_map is not None:
        return saved_map.enu_from_map @ saved_pose['transform'], saved_pose['query_stamp_ns']
    return None, None


def paired_overlay(record, transform, source_stamp_ns, now_ns, wall_now):
    if record is None or transform is None or source_stamp_ns is None:
        return None, 'waiting for source-paired ENU pose'
    fields, received = record
    try:
        stamp = fields['source_stamp_ns']
        if stamp == 0 and fields.get('state') == 'waiting':
            return None, 'waiting for tracking pose'
        if not isinstance(stamp, int) or isinstance(stamp, bool) or stamp <= 0:
            raise ValueError('invalid source stamp')
        if fields.get('frame') != 'enu' or fields.get('diagnostic_only') is not True:
            raise ValueError('invalid frame or output contract')
        if (not -.05 <= (now_ns-stamp)/1e9 <= .5 or
                not 0 <= wall_now-received <= .5 or abs(source_stamp_ns-stamp) > 150_000_000):
            return None, 'expired or unpaired tracking result'
        state = str(fields.get('state', 'waiting'))
        if fields.get('tracking_valid') is not True:
            return None, state
        transform = np.asarray(transform, float).reshape(4, 4)
        if not np.isfinite(transform).all():
            raise ValueError('invalid transform')
        def local(key, optional=False):
            value = fields.get(key)
            if optional and value is None:
                return None
            points = np.asarray(value, float)
            if points.ndim not in (1, 2) or points.shape[-1] != 3 or not np.isfinite(points).all():
                raise ValueError('invalid geometry')
            return (points-transform[:3, 3]) @ transform[:3, :3]
        path = local('reference_xyz')
        if path.ndim != 2 or len(path) > 256:
            raise ValueError('invalid reference horizon')
        lateral, heading = float(fields['lateral_error_m']), float(fields['heading_error_rad'])
        if not math.isfinite(lateral+heading):
            raise ValueError('invalid error metrics')
        overlay = dict(path=path, target=local('lookahead_xyz'), projection=local('projection_xyz'),
                       ego=local('ego_xyz'), fields=fields)
        basis = 'rear axle' if fields.get('vehicle_transform_verified') is True else 'sensor; axle transform UNSET'
        summary = f'PURE PURSUIT | {basis} | lateral {lateral:+.2f} m | heading {math.degrees(heading):+.1f} deg'
        if fields.get('targets_valid') is True:
            steering, speed = float(fields['steering_rad']), float(fields['target_speed_mps'])
            if not math.isfinite(steering+speed):
                raise ValueError('invalid targets')
            summary += f' | steering {math.degrees(steering):+.1f} deg | target {speed:.1f} m/s (preview)'
        else:
            summary += ' | steering unavailable'
        return overlay, summary
    except (KeyError, TypeError, ValueError, OverflowError):
        return None, 'invalid tracking diagnostics'
