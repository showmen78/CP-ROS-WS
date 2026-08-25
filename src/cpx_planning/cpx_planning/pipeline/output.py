"""Structured output contract for the CP-X planning pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class BehaviorCommand:
    """Behavior-layer decision exposed in PlannerOutput."""

    decision: str = "lane_follow"
    target_lane_id: int = 0
    target_speed_mps: float = 0.0
    stop_target: Optional[Mapping[str, object]] = None
    reroute_requested: bool = False
    normal_stop: bool = False
    stop_requested: bool = False
    emergency_brake: bool = False
    fsm_state: str = "LANE_KEEP"
    debug_reason: str = ""

    @classmethod
    def from_debug(
        cls,
        *,
        behavior_debug: Mapping[str, object],
        target_speed_mps: float,
    ) -> "BehaviorCommand":
        decision = str(behavior_debug.get("decision", "lane_follow"))
        normal_stop = decision in {"stop_at_intersection", "stop_sign"}
        static_obstacle_stop = decision == "static_obstacle_stop"
        emergency_brake = decision == "emergency_brake"
        return cls(
            decision=decision,
            target_lane_id=_to_int(behavior_debug.get("target_lane_id", 0)),
            target_speed_mps=float(target_speed_mps),
            stop_target=(
                behavior_debug.get("stop_target")
                if isinstance(behavior_debug.get("stop_target"), Mapping)
                else None
            ),
            reroute_requested=bool(behavior_debug.get("reroute_requested", False)),
            normal_stop=bool(normal_stop),
            stop_requested=bool(
                normal_stop or static_obstacle_stop or emergency_brake
            ),
            emergency_brake=bool(emergency_brake),
            fsm_state=str(behavior_debug.get("lc_state", "LANE_KEEP")),
            debug_reason=str(
                behavior_debug.get("debug_reason", behavior_debug.get("pipeline_error", ""))
            ),
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "decision": str(self.decision),
            "target_lane_id": int(self.target_lane_id),
            "target_speed_mps": float(self.target_speed_mps),
            "stop_target": dict(self.stop_target or {}),
            "reroute_requested": bool(self.reroute_requested),
            "normal_stop": bool(self.normal_stop),
            "stop_requested": bool(self.stop_requested),
            "emergency_brake": bool(self.emergency_brake),
            "fsm_state": str(self.fsm_state),
            "debug_reason": str(self.debug_reason),
        }


@dataclass(frozen=True)
class PlannerDiagnostics:
    """Serializable planner diagnostics."""

    fields: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return dict(self.fields or {})


@dataclass(frozen=True)
class PlannerOutput:
    """Complete internal output for one planning tick.

    OpenCDA still receives only ``control``.  The rest of the fields are the
    planning-module contract used for logging, metrics, and debugging.
    """

    control: Any
    behavior_command: BehaviorCommand
    reference_trajectory: Sequence[Mapping[str, object]] = field(default_factory=list)
    planned_trajectory: Sequence[Sequence[float]] = field(default_factory=list)
    predictions: Mapping[str, object] = field(default_factory=dict)
    acceleration_mps2: float = 0.0
    steering_rad: float = 0.0
    diagnostics: PlannerDiagnostics = field(default_factory=PlannerDiagnostics)

    def diagnostics_dict(self) -> Dict[str, object]:
        diagnostics = self.diagnostics.as_dict()
        diagnostics.setdefault("planner_output_behavior_decision", self.behavior_command.decision)
        diagnostics.setdefault("planner_output_target_lane_id", self.behavior_command.target_lane_id)
        diagnostics.setdefault("planner_output_reference_count", len(list(self.reference_trajectory or [])))
        diagnostics.setdefault("planner_output_planned_trajectory_count", len(list(self.planned_trajectory or [])))
        diagnostics.setdefault("planner_output_prediction_count", len(dict(self.predictions or {})))
        diagnostics.setdefault("planner_output_accel_mps2", float(self.acceleration_mps2))
        diagnostics.setdefault("planner_output_steer_rad", float(self.steering_rad))
        return diagnostics


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)
