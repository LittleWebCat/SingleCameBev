"""Transport-independent validation and deadman state for the vehicle bridge.

No ROS imports, device access or commands. Clock arguments are explicit so the
same checks can be exercised without an adapter or a running vehicle.
"""
from dataclasses import dataclass
import json
import math


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('expected a numeric value')
    if not math.isfinite(value):
        raise ValueError('expected a finite value')
    return float(value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


@dataclass(frozen=True)
class Target:
    stamp_ns: int
    steering_rad: float
    speed_mps: float
    feedback_mps: float
    received_mono: float


class InputGate:
    def __init__(self, cmd_timeout=.3, engage_timeout=.3, source_timeout=.5,
                 source='steering_rad', wheelbase=2.7):
        for value in (cmd_timeout, engage_timeout, source_timeout, wheelbase):
            if number(value) <= 0:
                raise ValueError('timeouts and wheelbase must be positive')
        if source not in ('steering_rad', 'curvature'):
            raise ValueError('unsupported target source')
        self.cmd_timeout, self.engage_timeout = cmd_timeout, engage_timeout
        self.source_timeout, self.source, self.wheelbase = source_timeout, source, wheelbase
        self.target = None
        self.last_stamp_ns = 0
        self.engage_rx = None
        self.release_required = True
        self.reason = 'waiting_for_live_target_and_release'
        self.clock_fault = False
        self.last_clock = None

    def invalidate(self, reason):
        self.target = None
        self.engage_rx = None
        self.release_required = True
        self.reason = reason
        return False

    def observe_clock(self, mono, ros_ns):
        if not math.isfinite(mono) or not isinstance(ros_ns, int) or ros_ns <= 0:
            self.clock_fault = True
        if self.last_clock is not None:
            previous_mono, previous_ros = self.last_clock
            drift = (ros_ns-previous_ros)/1e9-(mono-previous_mono)
            if mono < previous_mono or ros_ns < previous_ros or abs(drift) > .2:
                self.clock_fault = True
        self.last_clock = mono, ros_ns
        if self.clock_fault:
            return self.invalidate('clock_jump_restart_required')
        return True

    def receive(self, payload, mono, ros_ns):
        if not self.observe_clock(mono, ros_ns):
            return False
        try:
            fields = json.loads(payload, object_pairs_hook=unique_object)
            if not isinstance(fields, dict):
                raise ValueError('JSON must be an object')
            for key in ('tracking_valid', 'targets_valid', 'vehicle_transform_verified', 'diagnostic_only'):
                if fields.get(key) is not True:
                    raise ValueError('invalid_'+key)
            if (fields.get('source_mode') != 'live_localization' or
                    fields.get('frame') != 'enu' or fields.get('pose_reference') != 'rear_axle'):
                raise ValueError('not_live_enu_rear_axle')
            state = fields.get('state')
            if state not in ('tracking', 'endpoint'):
                raise ValueError('invalid_tracking_state')
            stamp = fields.get('source_stamp_ns')
            if type(stamp) is not int or stamp <= 0 or stamp <= self.last_stamp_ns:
                raise ValueError('nonadvancing_source_stamp')
            age = (ros_ns-stamp)/1e9
            if not -.05 <= age <= self.source_timeout:
                raise ValueError('source_pose_expired_or_future')
            angle = (math.atan(number(fields.get('curvature_per_m'))*self.wheelbase)
                     if self.source == 'curvature' else number(fields.get('steering_rad')))
            speed = number(fields.get('target_speed_mps'))
            feedback = number(fields.get('estimated_speed_mps'))
            if abs(angle) > math.pi/2 or not 0 <= speed <= 40 or not 0 <= feedback <= 40:
                raise ValueError('target_or_feedback_out_of_range')
            if state == 'endpoint' and speed > .05:
                raise ValueError('endpoint_must_request_stop')
            self.target = Target(stamp, angle, speed, feedback, mono)
            self.last_stamp_ns = stamp
            self.reason = 'ready' if self.engage_rx is not None else 'waiting_for_release_then_engage'
            return True
        except (ValueError, TypeError, OverflowError, RecursionError) as error:
            return self.invalidate(str(error))

    def engage(self, enabled, mono):
        if enabled is False:
            self.engage_rx = None
            self.release_required = False
            self.reason = 'operator_disengaged'
            return
        if enabled is not True or self.release_required or self.target is None or self.clock_fault:
            return
        if not 0 <= mono-self.target.received_mono <= self.cmd_timeout:
            self.invalidate('target_receipt_expired')
            return
        self.engage_rx = mono
        self.reason = 'engaged'

    def ready(self, mono, ros_ns):
        if not self.observe_clock(mono, ros_ns):
            return False
        if self.target is None:
            return False
        if (not 0 <= mono-self.target.received_mono <= self.cmd_timeout or
                not -.05 <= (ros_ns-self.target.stamp_ns)/1e9 <= self.source_timeout):
            return self.invalidate('target_expired_release_required')
        if self.engage_rx is None:
            return False
        if not 0 <= mono-self.engage_rx <= self.engage_timeout:
            return self.invalidate('deadman_expired_release_required')
        return not self.release_required


class ResponseMonitor:
    """Trip on what the CAR did, not on what we wrote.

    dbw_esp_full echoes 0x741 (measured wheel angle, measured mph, engaged, firmware setpoint) at
    ~20 Hz. The command path alone cannot see a runaway, a dead speed-hold or a stuck wheel, so the
    October 2026 hardware sessions recorded 17.6 mph against a 7 mph command and 10+ s stationary
    against a 5-7 mph command with nothing in the bridge able to notice either.

    Trips (first one wins; the caller latches a stop):
      telemetry_never_arrived / telemetry_lost -- no fresh 0x741 after we started commanding.
      overspeed                 -- measured above the cap + margin. dbw_esp_full has NO overspeed trip.
      speed_hold_not_responding -- speed commanded, car stationary. The stale-0x284 case: the firmware
                                   rests the pedals whenever wheel speed is more than 300 ms old.
      steering_not_tracking     -- measured wheel far from the commanded wheel for steer_grace.

    No ROS, no device, explicit clock -- so the whole trip matrix is testable off the car.
    """
    def __init__(self, max_speed_mph, overspeed_margin_mph=2., telemetry_timeout=1.,
                 no_motion_grace=3., steer_tol_deg=60., steer_grace=2.,
                 motion_cmd_mph=1., motion_meas_mph=.5):
        for value in (overspeed_margin_mph, telemetry_timeout, no_motion_grace, steer_tol_deg, steer_grace):
            if number(value) <= 0:
                raise ValueError('response monitor thresholds must be positive')
        if number(max_speed_mph) < 0:
            raise ValueError('max_speed_mph must be nonnegative')
        self.max_speed_mph, self.overspeed_margin_mph = max_speed_mph, overspeed_margin_mph
        self.telemetry_timeout, self.no_motion_grace = telemetry_timeout, no_motion_grace
        self.steer_tol_deg, self.steer_grace = steer_tol_deg, steer_grace
        self.motion_cmd_mph, self.motion_meas_mph = motion_cmd_mph, motion_meas_mph
        self.seen = False
        self.no_motion_since = None
        self.steer_bad_since = None

    def reset(self):
        self.no_motion_since = self.steer_bad_since = None

    def update(self, now, telem_age, meas_mph, meas_deg, cmd_speed_mph, cmd_steer_deg, armed_since):
        """`armed_since` is the monotonic time of the first command, or None if we have not commanded
        yet (nothing is armed before that). Returns a trip name, or None."""
        if armed_since is None:
            return None
        if telem_age <= self.telemetry_timeout:
            self.seen = True
        elif now-armed_since > self.telemetry_timeout:
            return 'telemetry_lost' if self.seen else 'telemetry_never_arrived'
        if not self.seen or telem_age > self.telemetry_timeout:
            return None
        if meas_mph is not None and meas_mph > self.max_speed_mph+self.overspeed_margin_mph:
            return 'overspeed'
        if meas_mph is not None and cmd_speed_mph > self.motion_cmd_mph and meas_mph < self.motion_meas_mph:
            if self.no_motion_since is None:
                self.no_motion_since = now
            elif now-self.no_motion_since > self.no_motion_grace:
                return 'speed_hold_not_responding'
        else:
            self.no_motion_since = None
        if meas_deg is not None and abs(meas_deg-cmd_steer_deg) > self.steer_tol_deg:
            if self.steer_bad_since is None:
                self.steer_bad_since = now
            elif now-self.steer_bad_since > self.steer_grace:
                return 'steering_not_tracking'
        else:
            self.steer_bad_since = None
        return None


class PedalPI:
    """PI for diagnostic comparison; no open-loop creep without feedback."""
    def __init__(self, kp=.15, ki=.05, limit=.5, stop_brake=.2):
        self.kp, self.ki, self.limit, self.stop_brake = kp, ki, limit, stop_brake
        self.integral = 0.

    def reset(self):
        self.integral = 0.

    def update(self, target, feedback, dt):
        target, feedback, dt = number(target), number(feedback), number(dt)
        if target < 0 or feedback < 0 or not 0 < dt <= .1:
            self.reset()
            raise ValueError('invalid PI feedback or timestep')
        if target <= .05:
            self.reset()
            return -self.stop_brake
        error = target-feedback
        candidate = max(-self.limit, min(self.limit, self.integral+self.ki*error*dt))
        requested = self.kp*error+candidate
        if abs(requested) <= self.limit or requested*error < 0:
            self.integral = candidate
        return max(-self.limit, min(self.limit, self.kp*error+self.integral))
