"""Tracker stage for CP-X planning."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

# from opencda.planning_module.pipeline.prediction import (
#     PredictionFrame,
#     build_prediction_frame,
# )

from .prediction import (PredictionFrame,build_prediction_frame)


class CPXObstacleTracker:
    """Obstacle tracker with TTL hold and simple prediction validity gates."""

    def __init__(
        self,
        *,
        max_stale_s: float = 0.5,
        max_speed_mps: float = 45.0,
        max_acceleration_mps2: float = 12.0,
        max_position_jump_m: float = 12.0,
    ) -> None:
        self._latest_obstacles: List[Dict[str, object]] = []
        self._timestamp_s = 0.0
        self._signal_context: Dict[str, object] = {}
        self._stop_target: Optional[Dict[str, object]] = None
        self.max_stale_s = max(0.0, float(max_stale_s))
        self.max_speed_mps = max(0.0, float(max_speed_mps))
        self.max_acceleration_mps2 = max(0.0, float(max_acceleration_mps2))
        self.max_position_jump_m = max(0.0, float(max_position_jump_m))
        self._tracks: Dict[str, Dict[str, object]] = {}
        self._last_validity_reason = "tracker_empty"
        self._last_stale_count = 0

    def update(
        self,
        *,
        obstacle_snapshots: Sequence[Mapping[str, Any]],
        timestamp_s: float,
        signal_context: Optional[Mapping[str, object]] = None,
        stop_target: Optional[Mapping[str, object]] = None,
    ) -> List[Dict[str, object]]:
        timestamp_s = float(timestamp_s)
        active_keys = set()
        accepted: List[Dict[str, object]] = []
        rejected_reasons: List[str] = []
        for snapshot in list(obstacle_snapshots or []):
            if not isinstance(snapshot, Mapping):
                continue
            normalized = dict(snapshot)
            key = self._track_key(normalized)
            active_keys.add(str(key))
            previous = self._tracks.get(str(key))
            valid, reason = self._valid_transition(
                previous=previous,
                current=normalized,
                timestamp_s=float(timestamp_s),
            )
            if not bool(valid):
                rejected_reasons.append(str(reason))
                continue
            normalized["track_id"] = str(key)
            normalized["track_age_s"] = 0.0
            normalized["track_stale"] = False
            normalized["prediction_valid"] = True
            normalized["prediction_validity_reason"] = str(reason)
            self._tracks[str(key)] = {
                "snapshot": dict(normalized),
                "timestamp_s": float(timestamp_s),
            }
            accepted.append(dict(normalized))

        stale_count = 0
        for key, record in list(self._tracks.items()):
            if key in active_keys:
                continue
            age_s = float(timestamp_s) - float(record.get("timestamp_s", timestamp_s))
            if age_s > float(self.max_stale_s):
                self._tracks.pop(key, None)
                continue
            stale = dict(record.get("snapshot", {}) or {})
            if not stale:
                continue
            stale["track_age_s"] = max(0.0, float(age_s))
            stale["track_stale"] = True
            stale["prediction_valid"] = True
            stale["prediction_validity_reason"] = "held_by_tracker_ttl"
            self._propagate_stale(stale, age_s=float(age_s))
            accepted.append(stale)
            stale_count += 1

        self._latest_obstacles = accepted
        self._timestamp_s = float(timestamp_s)
        self._signal_context = dict(signal_context or {})
        self._stop_target = dict(stop_target or {}) if isinstance(stop_target, Mapping) else None
        self._last_stale_count = int(stale_count)
        if rejected_reasons:
            self._last_validity_reason = "tracker_rejected:" + ",".join(sorted(set(rejected_reasons))[:3])
        elif stale_count:
            self._last_validity_reason = f"tracker_ttl_hold:{int(stale_count)}"
        else:
            self._last_validity_reason = "tracker_valid"
        return [dict(snapshot) for snapshot in self._latest_obstacles]

    def predict(
        self,
        *,
        ego_snapshot: Mapping[str, object],
        lane_assignments: Mapping[str, int],
        available_lane_ids: Sequence[int],
        horizon_s: float,
        dt_s: float,
        min_front_gap_m: float,
        min_rear_gap_m: float,
        min_ttc_s: float,
        prediction_model: str = "constant_acceleration",
        max_abs_acceleration_mps2: float = 4.0,
    ) -> PredictionFrame:
        return build_prediction_frame(
            ego_snapshot=ego_snapshot,
            obstacle_snapshots=self._latest_obstacles,
            lane_assignments=lane_assignments,
            available_lane_ids=available_lane_ids,
            horizon_s=float(horizon_s),
            dt_s=float(dt_s),
            min_front_gap_m=float(min_front_gap_m),
            min_rear_gap_m=float(min_rear_gap_m),
            min_ttc_s=float(min_ttc_s),
            prediction_model=str(prediction_model),
            max_abs_acceleration_mps2=float(max_abs_acceleration_mps2),
        )

    @property
    def latest_obstacles(self) -> List[Dict[str, object]]:
        return [dict(snapshot) for snapshot in self._latest_obstacles]

    @property
    def diagnostics(self) -> Dict[str, object]:
        return {
            "tracker_active_count": len(self._latest_obstacles),
            "tracker_stale_count": int(self._last_stale_count),
            "prediction_validity_reason": str(self._last_validity_reason),
        }

    def _valid_transition(
        self,
        *,
        previous: Optional[Mapping[str, object]],
        current: Mapping[str, object],
        timestamp_s: float,
    ) -> tuple[bool, str]:
        speed = _speed_mps(current)
        if self.max_speed_mps > 0.0 and abs(float(speed)) > float(self.max_speed_mps):
            return False, "speed_gate"
        if previous is None:
            return True, "new_track"
        prev_snapshot = dict(previous.get("snapshot", {}) or {})
        prev_time = float(previous.get("timestamp_s", timestamp_s))
        dt_s = max(1.0e-3, float(timestamp_s) - float(prev_time))
        dx = _float(current, "x", "x_m") - _float(prev_snapshot, "x", "x_m")
        dy = _float(current, "y", "y_m") - _float(prev_snapshot, "y", "y_m")
        jump_m = math.hypot(float(dx), float(dy))
        if self.max_position_jump_m > 0.0 and jump_m > max(self.max_position_jump_m, 2.0 * abs(float(speed)) * dt_s + 2.0):
            return False, "position_jump_gate"
        prev_speed = _speed_mps(prev_snapshot)
        acceleration = (float(speed) - float(prev_speed)) / float(dt_s)
        if self.max_acceleration_mps2 > 0.0 and abs(float(acceleration)) > float(self.max_acceleration_mps2):
            return False, "acceleration_gate"
        return True, "valid_transition"

    @staticmethod
    def _track_key(snapshot: Mapping[str, object]) -> str:
        for key in ("track_id", "object_id", "id", "vehicle_id", "actor_id"):
            value = snapshot.get(key)
            if value is not None and str(value).strip() != "":
                return str(value)
        x_m = _float(snapshot, "x", "x_m")
        y_m = _float(snapshot, "y", "y_m")
        return f"xy:{round(float(x_m), 1)}:{round(float(y_m), 1)}"

    @staticmethod
    def _propagate_stale(snapshot: Dict[str, object], *, age_s: float) -> None:
        heading = _float(snapshot, "psi", "yaw_rad", default=0.0)
        speed = _speed_mps(snapshot)
        snapshot["x"] = _float(snapshot, "x", "x_m") + float(speed) * math.cos(float(heading)) * float(age_s)
        snapshot["y"] = _float(snapshot, "y", "y_m") + float(speed) * math.sin(float(heading)) * float(age_s)
        snapshot["x_m"] = float(snapshot["x"])
        snapshot["y_m"] = float(snapshot["y"])


def _float(snapshot: Mapping[str, object], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        try:
            value = snapshot.get(key)
        except Exception:
            value = None
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return float(default)


def _speed_mps(snapshot: Mapping[str, object]) -> float:
    return _float(snapshot, "v", "speed_mps", "velocity_mps", default=0.0)
