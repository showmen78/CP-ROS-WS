"""Scenario-aware speed planning for the CP-X pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class SpeedPlan:
    target_speed_mps: float
    speed_cap_mps: float
    stop_goal_active: bool
    reason: str = ""

    def as_debug_fields(self) -> dict[str, object]:
        return {
            "speed_plan_target_mps": float(self.target_speed_mps),
            "speed_plan_cap_mps": float(self.speed_cap_mps),
            "speed_plan_stop_goal_active": bool(self.stop_goal_active),
            "speed_plan_reason": str(self.reason),
        }


def build_speed_plan(
    *,
    scenario_decision: object,
    behavior_decision: object,
    requested_speed_mps: float,
    ego_speed_mps: float,
    config: Mapping[str, object],
) -> SpeedPlan:
    """Return the speed target owned by the scenario/behavior layer."""

    requested = max(0.0, float(requested_speed_mps))
    scenario_cap = getattr(scenario_decision, "speed_cap_mps", None)
    cap = requested if scenario_cap is None else min(requested, max(0.0, float(scenario_cap)))
    stop_goal = bool(getattr(scenario_decision, "stop_goal_active", False))
    decision = str(behavior_decision or "").strip().lower()
    if decision in {"stop_at_intersection", "stop_sign", "emergency_brake"}:
        stop_goal = True
    if decision in {"intersection_turn_left", "intersection_turn_right"}:
        cap = min(
            float(cap),
            max(0.1, float(config.get("full_intersection_turn_speed_cap_mps", 2.2))),
        )
    if stop_goal:
        cap = 0.0
    if not math.isfinite(cap):
        cap = 0.0
    reason = str(getattr(scenario_decision, "reason", "") or "")
    if decision in {"intersection_turn_left", "intersection_turn_right"}:
        reason = _join_reason(reason, "speed_plan_turn_cap")
    if stop_goal:
        reason = _join_reason(reason, "speed_plan_stop_zero")
    elif scenario_cap is not None and float(cap) < float(requested):
        reason = _join_reason(reason, "speed_plan_scenario_cap")
    return SpeedPlan(
        target_speed_mps=float(cap),
        speed_cap_mps=float(cap),
        stop_goal_active=bool(stop_goal),
        reason=str(reason),
    )


def _join_reason(first: str, second: str) -> str:
    if not first:
        return str(second)
    if str(second) in str(first).split(";"):
        return str(first)
    return f"{first};{second}"
