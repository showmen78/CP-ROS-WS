"""Prediction stage for the CP-X planning pipeline.

The runner already receives cooperative-perception obstacle snapshots from
CP-X or from the local CARLA/SUMO tracker.  This module makes the prediction
stage explicit: every obstacle gets a short-horizon future trajectory, then
each candidate lane receives a future-risk summary used by the behavior FSM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence

from cpx_planning.behavior_planner.trajectory_risk import (
    lane_prediction_risk,
    obstacle_future_trajectory,
)


def obstacle_track_id(snapshot: Mapping[str, object]) -> str:
    """Public alias of the id resolution ``PredictionFrame`` keys are built
    with, so callers matching MPC obstacle snapshots against
    ``obstacle_future_trajectories`` use the exact same identity rule."""

    return _obstacle_id(snapshot)


def mpc_stage_trajectory(
    points: Sequence[Mapping[str, object]],
    *,
    fallback_heading_rad: float,
    horizon_steps: int,
    dt_s: float,
) -> List[List[float]]:
    """Convert ``obstacle_future_trajectory``-style ``{x, y, t, v}`` points
    into the ``[x, y, v, psi]``-per-stage list
    ``MPC._get_object_state_at_stage`` reads directly (see MPC/mpc.py).

    Without this, MPC's own obstacle-avoidance cost never sees this
    module's prediction at all: it only recognizes a ``predicted_trajectory``
    already shaped as one ``[x, y, v, psi]`` entry per stage, and silently
    falls back to its own constant-velocity extrapolation for anything else
    (including the ``{x, y, t}`` dict points this module produces). The
    heading is held constant at ``fallback_heading_rad`` because the
    constant-acceleration/constant-velocity models this module falls back to
    do not turn -- a real turning prediction would need to supply its own
    per-point heading in ``points``.
    """

    stages: List[List[float]] = []
    last_x: float | None = None
    last_y: float | None = None
    last_v = 0.0
    for step in range(max(0, int(horizon_steps))):
        if step < len(points):
            point = points[step]
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            v = float(point.get("v", last_v))
        elif last_x is not None:
            # The supplied trajectory is shorter than MPC's horizon (e.g. a
            # CP-supplied real prediction that stops early). Hold the last
            # known speed/heading rather than leaving later stages unset.
            x = float(last_x) + float(last_v) * math.cos(float(fallback_heading_rad)) * float(dt_s)
            y = float(last_y) + float(last_v) * math.sin(float(fallback_heading_rad)) * float(dt_s)
            v = float(last_v)
        else:
            break
        last_x, last_y, last_v = x, y, v
        stages.append([float(x), float(y), float(v), float(fallback_heading_rad)])
    return stages


def _obstacle_id(snapshot: Mapping[str, object]) -> str:
    for key in ("track_id", "object_id", "vehicle_id", "actor_id", "id"):
        value = snapshot.get(key)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    try:
        return "xy:{:.1f}:{:.1f}".format(
            float(snapshot.get("x", snapshot.get("x_m", 0.0))),
            float(snapshot.get("y", snapshot.get("y_m", 0.0))),
        )
    except Exception:
        return ""


@dataclass
class PredictionFrame:
    """Prediction output consumed by behavior decision and trajectory planning."""

    ego_snapshot: Dict[str, float]
    obstacle_snapshots: List[dict]
    obstacle_future_trajectories: Dict[str, List[dict]] = field(default_factory=dict)
    lane_prediction_risks: Dict[int, Dict[str, object]] = field(default_factory=dict)

    def risk_for_lane(self, lane_id: int) -> Dict[str, object]:
        return dict(self.lane_prediction_risks.get(int(lane_id), {}))


def build_prediction_frame(
    *,
    ego_snapshot: Mapping[str, object],
    obstacle_snapshots: Sequence[Mapping[str, Any]],
    lane_assignments: Mapping[str, int],
    available_lane_ids: Sequence[int],
    horizon_s: float,
    dt_s: float,
    min_front_gap_m: float,
    min_rear_gap_m: float,
    min_ttc_s: float,
    prediction_model: str = "constant_acceleration",
    max_abs_acceleration_mps2: float = 4.0,
    lane_step_fn: Callable[[float, float, float], Any] | None = None,
) -> PredictionFrame:
    """Build an Apollo-style prediction frame for one planning tick.

    Existing CP-X messages may already include `predicted_trajectory`.  When
    they do not, the default fallback is a constant-acceleration prediction
    (`prediction_model="constant_acceleration"`).  With no acceleration field
    available this degenerates to constant velocity, so existing snapshots keep
    their previous behaviour.

    ``lane_step_fn``, when supplied, lets the fallback follow the obstacle's
    own lane centerline (curved) instead of a straight line -- see
    ``behavior_planner.trajectory_risk._lane_following_points``. Passing None
    (the default) preserves the exact previous straight-line behaviour.
    """

    normalized_ego = {
        "x": float(ego_snapshot.get("x", 0.0)),
        "y": float(ego_snapshot.get("y", 0.0)),
        "v": float(ego_snapshot.get("v", 0.0)),
        "psi": float(ego_snapshot.get("psi", 0.0)),
    }
    normalized_obstacles = [
        dict(snapshot)
        for snapshot in list(obstacle_snapshots or [])
        if isinstance(snapshot, Mapping)
    ]
    obstacle_future_trajectories = {
        obstacle_id: obstacle_future_trajectory(
            snapshot,
            horizon_s=float(horizon_s),
            dt_s=float(dt_s),
            model=str(prediction_model),
            max_abs_acceleration_mps2=float(max_abs_acceleration_mps2),
            lane_step_fn=lane_step_fn,
        )
        for snapshot in normalized_obstacles
        for obstacle_id in [_obstacle_id(snapshot)]
        if obstacle_id
    }
    lane_prediction_risks = {
        int(lane_id): lane_prediction_risk(
            ego_snapshot=normalized_ego,
            obstacle_snapshots=normalized_obstacles,
            lane_assignments=lane_assignments,
            target_lane_id=int(lane_id),
            horizon_s=float(horizon_s),
            dt_s=float(dt_s),
            min_front_gap_m=float(min_front_gap_m),
            min_rear_gap_m=float(min_rear_gap_m),
            min_ttc_s=float(min_ttc_s),
            prediction_model=str(prediction_model),
            max_abs_acceleration_mps2=float(max_abs_acceleration_mps2),
            lane_step_fn=lane_step_fn,
        )
        for lane_id in list(available_lane_ids or [])
    }
    return PredictionFrame(
        ego_snapshot=normalized_ego,
        obstacle_snapshots=normalized_obstacles,
        obstacle_future_trajectories=obstacle_future_trajectories,
        lane_prediction_risks=lane_prediction_risks,
    )
