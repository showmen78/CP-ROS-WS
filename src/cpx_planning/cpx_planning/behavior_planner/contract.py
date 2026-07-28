"""
Typed contract for the behavior layer's decision output.

``RuleBasedBehaviorPlanner.update()`` returns a plain ``dict`` at runtime so
that existing call sites (tests, ``planning_runner.py``) keep working
unchanged. ``BehaviorCommand`` documents the field contract for that dict —
which keys are always present vs conditionally present — without changing
the runtime type.
"""

from __future__ import annotations

from typing import Any

try:
    from typing import TypedDict
except ImportError:  # Python < 3.8
    from typing_extensions import TypedDict


class _BehaviorCommandRequired(TypedDict):
    decision: str
    target_lane_id: int
    selected_lane_id: int
    lc_state: str
    fsm_state: str
    blue_dot_rolling: bool
    stop: bool
    blocking_obstacle_id: str
    decision_reason: str


class BehaviorCommand(_BehaviorCommandRequired, total=False):
    mode_override: str | None
    stopping_point: dict[str, Any] | None
    reroute_messages: list[dict[str, Any]]
    stop_target: dict[str, Any]
    follow_target: dict[str, Any]
    traffic_light_debug: dict[str, Any]
    selected_candidate: str
    candidate_scores: list[dict[str, Any]]
    rejected_candidates: list[dict[str, Any]]
    candidate_evaluation: dict[str, Any]
    invariant_violations: list[str]
    lane_change_phase: str
    lane_change_abort_reason: str
    lane_change_cancel_reason: str
    lane_change_source_lane_id: int | str
    lane_change_target_lane_id: int | str
    lane_change_recovery_lane_id: int | str
    lane_change_state_elapsed_s: float
    lane_change_state_min_hold_s: float
