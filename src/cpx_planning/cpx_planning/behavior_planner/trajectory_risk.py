"""Prediction-aware behavior-planner risk checks."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric_value = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(numeric_value):
        return float(default)
    return float(numeric_value)


def _trajectory_points(snapshot: Mapping[str, object]) -> list[dict]:
    raw_trajectory = snapshot.get("predicted_trajectory", None)
    if not isinstance(raw_trajectory, Sequence) or isinstance(raw_trajectory, (str, bytes, bytearray)):
        return []
    points: list[dict] = []
    for index, raw_point in enumerate(list(raw_trajectory)):
        if isinstance(raw_point, Mapping):
            x_m = _safe_float(raw_point.get("x", snapshot.get("x", 0.0)))
            y_m = _safe_float(raw_point.get("y", snapshot.get("y", 0.0)))
            t_s = _safe_float(raw_point.get("t", raw_point.get("time_s", index)))
        elif isinstance(raw_point, Sequence) and not isinstance(raw_point, (str, bytes, bytearray)) and len(raw_point) >= 2:
            x_m = _safe_float(raw_point[0])
            y_m = _safe_float(raw_point[1])
            t_s = _safe_float(raw_point[4] if len(raw_point) >= 5 else index)
        else:
            continue
        points.append({"x": float(x_m), "y": float(y_m), "t": float(t_s)})
    return points


def _constant_velocity_points(
    snapshot: Mapping[str, object],
    *,
    horizon_s: float,
    dt_s: float,
) -> list[dict]:
    x0_m = _safe_float(snapshot.get("x", 0.0))
    y0_m = _safe_float(snapshot.get("y", 0.0))
    speed_mps = max(0.0, _safe_float(snapshot.get("v", 0.0)))
    heading_rad = _safe_float(snapshot.get("psi", 0.0))
    dt_s = max(1.0e-3, float(dt_s))
    horizon_s = max(dt_s, float(horizon_s))
    count = max(1, int(math.ceil(float(horizon_s) / float(dt_s))))
    return [
        {
            "x": float(x0_m + speed_mps * math.cos(heading_rad) * dt_s * step),
            "y": float(y0_m + speed_mps * math.sin(heading_rad) * dt_s * step),
            "t": float(dt_s * step),
        }
        for step in range(1, count + 1)
    ]


def _longitudinal_acceleration_mps2(snapshot: Mapping[str, object]) -> float:
    for key in (
        "a",
        "acceleration",
        "acceleration_mps2",
        "longitudinal_acceleration_mps2",
        "a_mps2",
    ):
        if key in snapshot:
            return _safe_float(snapshot.get(key, 0.0))
    if "ax" in snapshot or "ay" in snapshot:
        heading_rad = _safe_float(snapshot.get("psi", 0.0))
        ax_mps2 = _safe_float(snapshot.get("ax", 0.0))
        ay_mps2 = _safe_float(snapshot.get("ay", 0.0))
        return float(math.cos(heading_rad) * ax_mps2 + math.sin(heading_rad) * ay_mps2)
    return 0.0


def _constant_acceleration_points(
    snapshot: Mapping[str, object],
    *,
    horizon_s: float,
    dt_s: float,
    max_abs_acceleration_mps2: float = 4.0,
) -> list[dict]:
    x0_m = _safe_float(snapshot.get("x", 0.0))
    y0_m = _safe_float(snapshot.get("y", 0.0))
    speed_mps = max(0.0, _safe_float(snapshot.get("v", 0.0)))
    heading_rad = _safe_float(snapshot.get("psi", 0.0))
    accel_mps2 = max(
        -float(max_abs_acceleration_mps2),
        min(float(max_abs_acceleration_mps2), _longitudinal_acceleration_mps2(snapshot)),
    )
    dt_s = max(1.0e-3, float(dt_s))
    horizon_s = max(dt_s, float(horizon_s))
    count = max(1, int(math.ceil(float(horizon_s) / float(dt_s))))
    points: list[dict] = []
    for step in range(1, count + 1):
        t_s = float(dt_s * step)
        distance_m = max(0.0, float(speed_mps) * t_s + 0.5 * float(accel_mps2) * t_s * t_s)
        points.append({
            "x": float(x0_m + distance_m * math.cos(heading_rad)),
            "y": float(y0_m + distance_m * math.sin(heading_rad)),
            "t": float(t_s),
            "v": float(max(0.0, float(speed_mps) + float(accel_mps2) * t_s)),
            "a": float(accel_mps2),
            "model": "constant_acceleration",
        })
    return points


def obstacle_future_trajectory(
    snapshot: Mapping[str, object],
    *,
    horizon_s: float,
    dt_s: float,
    model: str = "constant_acceleration",
    max_abs_acceleration_mps2: float = 4.0,
) -> list[dict]:
    points = _trajectory_points(snapshot)
    if len(points) > 0:
        return [
            point
            for point in points
            if 0.0 <= float(point.get("t", 0.0)) <= float(horizon_s)
        ]
    if str(model).strip().lower() in {"ca", "constant_acceleration", "constant-acceleration"}:
        return _constant_acceleration_points(
            snapshot,
            horizon_s=float(horizon_s),
            dt_s=float(dt_s),
            max_abs_acceleration_mps2=float(max_abs_acceleration_mps2),
        )
    return _constant_velocity_points(snapshot, horizon_s=float(horizon_s), dt_s=float(dt_s))


def lane_prediction_risk(
    *,
    ego_snapshot: Mapping[str, object],
    obstacle_snapshots: Sequence[Mapping[str, Any]],
    lane_assignments: Mapping[str, int],
    target_lane_id: int,
    horizon_s: float = 3.0,
    dt_s: float = 0.2,
    min_front_gap_m: float = 12.0,
    min_rear_gap_m: float = 8.0,
    min_ttc_s: float = 2.5,
    prediction_model: str = "constant_acceleration",
    max_abs_acceleration_mps2: float = 4.0,
) -> Dict[str, object]:
    """Estimate whether a lane will stay safe over a short future horizon.

    The check is lane-local and ego-frame longitudinal. It is intentionally
    conservative for behavior planning: if a target lane has a future front or
    rear conflict with low gap/TTC, the behavior planner should stay in
    PREPARE instead of committing to EXECUTE.
    """

    ego_x = _safe_float(ego_snapshot.get("x", 0.0))
    ego_y = _safe_float(ego_snapshot.get("y", 0.0))
    ego_v = max(0.0, _safe_float(ego_snapshot.get("v", 0.0)))
    ego_psi = _safe_float(ego_snapshot.get("psi", 0.0))
    cos_h = math.cos(ego_psi)
    sin_h = math.sin(ego_psi)
    horizon_s = max(1.0e-3, float(horizon_s))
    dt_s = max(1.0e-3, float(dt_s))

    min_front_gap = float("inf")
    min_rear_gap = float("inf")
    min_ttc = float("inf")
    risky_obstacle_id = ""
    risky_reason = ""

    for snapshot in list(obstacle_snapshots or []):
        if not isinstance(snapshot, Mapping):
            continue
        obstacle_id = str(snapshot.get("vehicle_id", snapshot.get("id", ""))).strip()
        if lane_assignments.get(obstacle_id, None) != int(target_lane_id):
            continue
        obstacle_v = max(0.0, _safe_float(snapshot.get("v", 0.0)))
        points = obstacle_future_trajectory(
            snapshot,
            horizon_s=float(horizon_s),
            dt_s=float(dt_s),
            model=str(prediction_model),
            max_abs_acceleration_mps2=float(max_abs_acceleration_mps2),
        )
        for point in points:
            t_s = max(0.0, _safe_float(point.get("t", 0.0)))
            ego_future_x = float(ego_x) + float(ego_v) * math.cos(ego_psi) * float(t_s)
            ego_future_y = float(ego_y) + float(ego_v) * math.sin(ego_psi) * float(t_s)
            dx = _safe_float(point.get("x", 0.0)) - float(ego_future_x)
            dy = _safe_float(point.get("y", 0.0)) - float(ego_future_y)
            longitudinal_gap = float(cos_h * dx + sin_h * dy)
            if longitudinal_gap >= 0.0:
                min_front_gap = min(float(min_front_gap), float(longitudinal_gap))
                closing_speed = float(ego_v) - float(obstacle_v)
                if closing_speed > 0.0:
                    min_ttc = min(float(min_ttc), float(longitudinal_gap) / max(1.0e-6, closing_speed))
                if float(longitudinal_gap) < float(min_front_gap_m):
                    risky_obstacle_id = obstacle_id
                    risky_reason = "front_gap"
            else:
                rear_gap = abs(float(longitudinal_gap))
                min_rear_gap = min(float(min_rear_gap), float(rear_gap))
                closing_speed = float(obstacle_v) - float(ego_v)
                if closing_speed > 0.0:
                    min_ttc = min(float(min_ttc), float(rear_gap) / max(1.0e-6, closing_speed))
                if float(rear_gap) < float(min_rear_gap_m):
                    risky_obstacle_id = obstacle_id
                    risky_reason = "rear_gap"
            if math.isfinite(float(min_ttc)) and float(min_ttc) < float(min_ttc_s):
                risky_obstacle_id = obstacle_id
                risky_reason = "ttc"

    risky = bool(
        float(min_front_gap) < float(min_front_gap_m)
        or float(min_rear_gap) < float(min_rear_gap_m)
        or (math.isfinite(float(min_ttc)) and float(min_ttc) < float(min_ttc_s))
    )
    return {
        "risk": bool(risky),
        "target_lane_id": int(target_lane_id),
        "min_front_gap_m": None if not math.isfinite(min_front_gap) else float(min_front_gap),
        "min_rear_gap_m": None if not math.isfinite(min_rear_gap) else float(min_rear_gap),
        "min_ttc_s": None if not math.isfinite(min_ttc) else float(min_ttc),
        "risky_obstacle_id": str(risky_obstacle_id),
        "reason": str(risky_reason),
    }
