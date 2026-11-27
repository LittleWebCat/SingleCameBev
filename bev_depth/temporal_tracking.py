from __future__ import annotations

import importlib.util
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from .config import BevConfig


DEFAULT_TRACKER_DIR = Path(__file__).resolve().parents[1] / "bef-tracking-kf-main"


@dataclass(frozen=True)
class TemporalTrackingResult:
    frame_index: int
    cleaned_matrix: np.ndarray
    objects: list[dict[str, Any]]
    raw_target_speed_mph: float
    cleaned_target_speed_mph: float
    tracker_latency_ms: float
    dt_s: float
    active_tracks: int
    coasted_tracks: int
    ego_speed_mps: float
    ego_yaw_rate_rps: float
    safety_input_ego_speed_mps: float | None = None
    tracker_ego_speed_mps: float = 0.0
    nominal_target_speed_mph: float = 0.0
    safe_target_speed_mph: float = 0.0
    safety_zone: str = "UNAVAILABLE"
    critical_boundary_m: float | None = None
    collision_boundary_m: float | None = None
    obstacle_distance_m: float | None = None
    safety_reason: str = "Safety evaluation is unavailable"
    safety_result: Any | None = None
    obstacle_gap_description: str = "provisional BEV range"
    elapsed_s: float = 0.0

    @property
    def latency_ms(self) -> float:
        return self.tracker_latency_ms


def _load_source_module(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"Temporal tracker source is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load temporal tracker source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _load_tracker_viewer(
    root: Path,
    temporal: ModuleType,
    target_speed: ModuleType,
    control: ModuleType,
) -> ModuleType:
    aliases = {
        "temporal_matrix_cleaner": temporal,
        "target_speed": target_speed,
        "control_params": control,
    }
    previous = {name: sys.modules.get(name) for name in aliases}
    root_string = str(root)
    added_to_path = root_string not in sys.path
    if added_to_path:
        # Newer tracker branches split the viewer across sibling modules
        # (for example ``longitudinal_safety.py``).  Make those ordinary
        # imports resolvable while the externally supplied viewer is loaded.
        sys.path.insert(0, root_string)
    sys.modules.update(aliases)
    try:
        return _load_source_module(
            "_three_camera_bev_live_bev_viewer",
            root / "live_bev_viewer.py",
        )
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
        if added_to_path:
            sys.path.remove(root_string)


def validate_tracker_geometry(bev: BevConfig) -> None:
    geometry = (
        bev.forward_range_m,
        bev.rear_range_m,
        bev.lateral_range_m,
    )
    if not np.allclose(geometry, (80.0, 40.0, 80.0)):
        raise ValueError(
            "Temporal tracking requires an 80 m forward, 40 m rear, "
            f"80 m lateral BEV; got {geometry}"
        )


class TemporalBEVTracker:
    """Rate-limited bridge to the causal BEV Kalman tracker."""

    def __init__(
        self,
        bev: BevConfig,
        tracker_dir: str | Path = DEFAULT_TRACKER_DIR,
        update_hz: float = 10.0,
    ) -> None:
        validate_tracker_geometry(bev)
        if update_hz <= 0.0:
            raise ValueError("Temporal tracker update_hz must be positive")

        root = Path(tracker_dir).expanduser().resolve()
        temporal = _load_source_module(
            "_three_camera_bev_temporal_matrix_cleaner",
            root / "temporal_matrix_cleaner.py",
        )
        target_speed = _load_source_module(
            "_three_camera_bev_target_speed",
            root / "target_speed.py",
        )
        control = _load_source_module(
            "_three_camera_bev_control_params",
            root / "control_params.py",
        )
        viewer = _load_tracker_viewer(root, temporal, target_speed, control)

        self.bev = bev
        self.tracker_dir = root
        self.update_period_s = 1.0 / float(update_hz)
        processor_parameters = inspect.signature(
            viewer.LiveBEVProcessor
        ).parameters
        processor_kwargs = (
            {"nominal_dt": self.update_period_s}
            if "nominal_dt" in processor_parameters
            else {}
        )
        self._processor = viewer.LiveBEVProcessor(**processor_kwargs)
        self.cleaner = self._processor.cleaner
        self._renderer = viewer.BEVRenderer()
        self._renderer_accepts_dt = "dt" in inspect.signature(
            self._renderer.render
        ).parameters
        self._last_update_s: float | None = None
        self._last_raw_matrix: np.ndarray | None = None
        self._elapsed_s = 0.0
        self.last_result: TemporalTrackingResult | None = None

    def maybe_update(
        self,
        matrix: np.ndarray,
        *,
        monotonic_s: float | None = None,
        ego_yaw_rate: float | None = None,
        ego_speed_mps: float | None = None,
    ) -> TemporalTrackingResult | None:
        """Update when the configured tracker period has elapsed."""

        now = time.monotonic() if monotonic_s is None else float(monotonic_s)
        if (
            self._last_update_s is not None
            and now - self._last_update_s + 1e-9 < self.update_period_s
        ):
            return None

        dt = (
            self.update_period_s
            if self._last_update_s is None
            else max(1e-3, now - self._last_update_s)
        )
        raw = np.asarray(matrix)
        live_result = self._processor.update(
            raw,
            dt=dt,
            ego_yaw_rate=ego_yaw_rate,
            ego_speed_mps=ego_speed_mps,
        )
        cleaned = live_result.cleaned_matrix
        objects = live_result.objects
        cleaned_target_speed = float(live_result.cleaned_target_speed_mph)
        tracker_ego_speed = float(
            getattr(live_result, "tracker_ego_speed_mps", live_result.ego_speed_mps)
        )
        self._elapsed_s += dt
        result = TemporalTrackingResult(
            frame_index=int(live_result.frame_index),
            cleaned_matrix=cleaned,
            objects=objects,
            raw_target_speed_mph=float(live_result.raw_target_speed_mph),
            cleaned_target_speed_mph=cleaned_target_speed,
            tracker_latency_ms=float(live_result.tracker_latency_ms),
            dt_s=dt,
            active_tracks=int(live_result.active_tracks),
            coasted_tracks=int(live_result.coasted_tracks),
            ego_speed_mps=float(live_result.ego_speed_mps),
            ego_yaw_rate_rps=float(live_result.ego_yaw_rate_rps),
            safety_input_ego_speed_mps=getattr(
                live_result, "safety_input_ego_speed_mps", None
            ),
            tracker_ego_speed_mps=tracker_ego_speed,
            nominal_target_speed_mph=float(
                getattr(
                    live_result,
                    "nominal_target_speed_mph",
                    cleaned_target_speed,
                )
            ),
            safe_target_speed_mph=float(
                getattr(
                    live_result,
                    "safe_target_speed_mph",
                    cleaned_target_speed,
                )
            ),
            safety_zone=str(getattr(live_result, "safety_zone", "UNAVAILABLE")),
            critical_boundary_m=getattr(
                live_result, "critical_boundary_m", None
            ),
            collision_boundary_m=getattr(
                live_result, "collision_boundary_m", None
            ),
            obstacle_distance_m=getattr(
                live_result, "obstacle_distance_m", None
            ),
            safety_reason=str(
                getattr(
                    live_result,
                    "safety_reason",
                    "Safety evaluation is unavailable",
                )
            ),
            safety_result=getattr(live_result, "safety_result", None),
            obstacle_gap_description=str(
                getattr(
                    live_result,
                    "obstacle_gap_description",
                    "provisional BEV range",
                )
            ),
            elapsed_s=float(getattr(live_result, "elapsed_s", self._elapsed_s)),
        )
        self._last_update_s = now
        self._last_raw_matrix = raw.copy()
        self.last_result = result
        return result

    def render_viewer(self) -> np.ndarray:
        """Render with bef-tracking-kf-main's raw-vs-causal BEVRenderer."""

        if self._last_raw_matrix is None or self.last_result is None:
            raise RuntimeError("Temporal tracker has not produced a frame yet")
        if self._renderer_accepts_dt:
            return self._renderer.render(
                self._last_raw_matrix,
                self.last_result,
                dt=self.last_result.dt_s,
            )
        return self._renderer.render(self._last_raw_matrix, self.last_result)

    def reset(self) -> None:
        self._processor.reset()
        reset_display_history = getattr(
            self._renderer, "reset_display_history", None
        )
        if reset_display_history is not None:
            reset_display_history()
        self._last_update_s = None
        self._last_raw_matrix = None
        self._elapsed_s = 0.0
        self.last_result = None
