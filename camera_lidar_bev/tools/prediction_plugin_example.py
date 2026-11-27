"""Example prediction plugin: copy this file to try your own motion model on recorded drives.

  python3 tools/eval_prediction.py --plugin tools/prediction_plugin_example.py --models cv cv_slowdown --quick

A plugin registers a factory with prediction_models.register_model(name, factory); factory(t, det) builds a
track for one new detection det = (group, class, enu xyz, size (l, w, h), enu box yaw). The easiest start is
to subclass an existing track and override future(); see docs/prediction_development.md for the interface.
"""
import math

from prediction_models import CVTrack, IMMTrack, register_model


class SlowdownTrack(CVTrack):
    """Constant velocity, but assume a gentle 0.5 m/s^2 slowdown (a braking prior, never below 0)."""

    def future(self, horizon_s, step_s=0.1):
        path = super().future(horizon_s, step_s)
        x0, y0 = path[0][1], path[0][2]
        speed = math.hypot(*self.velocity())
        out = []
        for t, x, y, heading, radius in path:
            travelled = speed * t
            wanted = max(0.0, speed * t - 0.25 * t * t) if speed > 0 else 0.0
            k = wanted / travelled if travelled > 1e-6 else 0.0
            out.append((t, x0 + (x - x0) * k, y0 + (y - y0) * k, heading, radius))
        return out


# Vehicles get the new model; cyclists and pedestrians keep the constant-velocity filter.
register_model('cv_slowdown', lambda t, det: SlowdownTrack(t, det) if det[0] == 'vehicle' else CVTrack(t, det))
# Also usable as a starting point: IMMTrack for vehicles, as in the default 'imm'.
register_model('imm_copy', lambda t, det: IMMTrack(t, det) if det[0] == 'vehicle' else CVTrack(t, det))
