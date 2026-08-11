"""Scenario-aware speed planning for the CP-X pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class SpeedPlan:
    target_speed_mps: float
    speed_cap_mps: float
    stop_goal_active: bool
    reason: str = ""
    front_gap_m: Optional[float] = None
    desired_follow_gap_m: Optional[float] = None
    continuous_following_active: bool = False
    requested_speed_mps: float = 0.0
    scenario_cap_mps: Optional[float] = None
    turn_cap_mps: Optional[float] = None
    lane_change_cap_mps: Optional[float] = None
    following_cap_mps: Optional[float] = None
    turn_approach_cap_mps: Optional[float] = None
    upcoming_turn_distance_m: Optional[float] = None
    limiting_owner: str = "behavior_request"
    active_constraints: tuple[str, ...] = ()

    def as_debug_fields(self) -> dict[str, object]:
        return {
            "speed_plan_target_mps": float(self.target_speed_mps),
            "speed_plan_cap_mps": float(self.speed_cap_mps),
            "speed_plan_stop_goal_active": bool(self.stop_goal_active),
            "speed_plan_reason": str(self.reason),
            "speed_plan_front_gap_m": (
                "" if self.front_gap_m is None else float(self.front_gap_m)
            ),
            "speed_plan_desired_follow_gap_m": (
                ""
                if self.desired_follow_gap_m is None
                else float(self.desired_follow_gap_m)
            ),
            "speed_plan_continuous_following_active": bool(
                self.continuous_following_active
            ),
            "speed_owner_requested_mps": float(self.requested_speed_mps),
            "speed_owner_scenario_cap_mps": (
                "" if self.scenario_cap_mps is None else float(self.scenario_cap_mps)
            ),
            "speed_owner_turn_cap_mps": (
                "" if self.turn_cap_mps is None else float(self.turn_cap_mps)
            ),
            "speed_owner_lane_change_cap_mps": (
                ""
                if self.lane_change_cap_mps is None
                else float(self.lane_change_cap_mps)
            ),
            "speed_owner_following_cap_mps": (
                "" if self.following_cap_mps is None else float(self.following_cap_mps)
            ),
            "speed_owner_turn_approach_cap_mps": (
                ""
                if self.turn_approach_cap_mps is None
                else float(self.turn_approach_cap_mps)
            ),
            "speed_owner_upcoming_turn_distance_m": (
                ""
                if self.upcoming_turn_distance_m is None
                else float(self.upcoming_turn_distance_m)
            ),
            "speed_owner_selected_target_mps": float(self.target_speed_mps),
            "speed_owner_limiting_owner": str(self.limiting_owner),
            "speed_owner_active_constraints": ";".join(self.active_constraints),
        }


@dataclass(frozen=True)
class SpeedCeilingResult:
    target_speed_mps: float
    destination_state: list[float]
    reference_samples: list[dict[str, object]]
    applied: bool
    reduction_mps: float


def turn_approach_lookahead_m(
    *,
    cruise_speed_mps: float,
    config: Mapping[str, object],
) -> float:
    """Return enough route preview to decelerate to the turn-entry speed."""

    cruise_speed = max(0.0, float(cruise_speed_mps))
    turn_speed = max(
        0.1,
        float(config.get("full_intersection_turn_speed_cap_mps", 2.2)),
    )
    comfortable_decel = max(
        0.1,
        float(config.get("turn_approach_comfort_decel_mps2", 2.5)),
    )
    braking_distance = max(
        0.0,
        (cruise_speed * cruise_speed - turn_speed * turn_speed)
        / (2.0 * comfortable_decel),
    )
    entry_buffer = max(
        0.0,
        float(config.get("turn_approach_entry_buffer_m", 5.0)),
    )
    preview_margin = max(
        0.0,
        float(config.get("turn_approach_preview_margin_m", 10.0)),
    )
    configured_minimum = max(
        0.0,
        float(config.get("scenario_turn_prepare_lookahead_m", 15.0)),
    )
    return max(
        configured_minimum,
        braking_distance + entry_buffer + preview_margin,
    )


def enforce_speed_ceiling(
    *,
    proposed_target_mps: float,
    ceiling_mps: float,
    destination_state: Sequence[float],
    reference_samples: Sequence[Mapping[str, object]],
) -> SpeedCeilingResult:
    """Apply the planner speed ceiling without changing reference geometry."""

    proposed = max(0.0, float(proposed_target_mps))
    ceiling = max(0.0, float(ceiling_mps))
    selected = min(proposed, ceiling)
    destination = list(destination_state)
    if len(destination) >= 3:
        destination[2] = min(max(0.0, float(destination[2])), ceiling)
    samples = [dict(sample) for sample in reference_samples]
    for sample in samples:
        for key in ("speed_ref_mps", "v_ref_mps", "speed_mps", "v"):
            if key not in sample:
                continue
            try:
                sample[key] = min(max(0.0, float(sample[key])), ceiling)
            except (TypeError, ValueError):
                continue
    reduction = max(0.0, proposed - selected)
    return SpeedCeilingResult(
        target_speed_mps=float(selected),
        destination_state=destination,
        reference_samples=samples,
        applied=bool(reduction > 1.0e-6),
        reduction_mps=float(reduction),
    )


def build_speed_plan(
    *,
    scenario_decision: object,
    behavior_decision: object,
    requested_speed_mps: float,
    ego_speed_mps: float,
    config: Mapping[str, object],
    front_gap_m: Optional[float] = None,
    upcoming_turn_direction: str = "",
    upcoming_turn_distance_m: Optional[float] = None,
) -> SpeedPlan:
    """Return the speed target owned by the scenario/behavior layer."""

    requested = max(0.0, float(requested_speed_mps))
    scenario_cap = getattr(scenario_decision, "speed_cap_mps", None)
    scenario_cap_value = (
        None if scenario_cap is None else max(0.0, float(scenario_cap))
    )
    cap = requested if scenario_cap is None else min(requested, max(0.0, float(scenario_cap)))
    limiting_owner = "behavior_request"
    active_constraints = []
    if scenario_cap_value is not None:
        active_constraints.append("scenario_cap")
        if scenario_cap_value < requested:
            limiting_owner = "scenario_cap"
    stop_goal = bool(getattr(scenario_decision, "stop_goal_active", False))
    reason = str(getattr(scenario_decision, "reason", "") or "")
    decision = str(behavior_decision or "").strip().lower()
    if decision in {"stop_at_intersection", "stop_sign", "emergency_brake"}:
        stop_goal = True
    turn_cap_mps = None
    turn_approach_cap_mps = None
    finite_turn_distance_m = None
    turn_direction = str(upcoming_turn_direction or "").strip().lower()
    if (
        turn_direction in {"left", "right"}
        and upcoming_turn_distance_m is not None
        and math.isfinite(float(upcoming_turn_distance_m))
    ):
        finite_turn_distance_m = max(0.0, float(upcoming_turn_distance_m))
    if (
        finite_turn_distance_m is not None
        and decision not in {"intersection_turn_left", "intersection_turn_right"}
        and not stop_goal
    ):
        turn_entry_speed_mps = max(
            0.1,
            float(config.get("full_intersection_turn_speed_cap_mps", 2.2)),
        )
        comfortable_decel_mps2 = max(
            0.1,
            float(config.get("turn_approach_comfort_decel_mps2", 2.5)),
        )
        entry_buffer_m = max(
            0.0,
            float(config.get("turn_approach_entry_buffer_m", 5.0)),
        )
        braking_distance_m = max(0.0, finite_turn_distance_m - entry_buffer_m)
        turn_approach_cap_mps = math.sqrt(
            turn_entry_speed_mps * turn_entry_speed_mps
            + 2.0 * comfortable_decel_mps2 * braking_distance_m
        )
        active_constraints.append("turn_approach_cap")
        previous_cap = float(cap)
        cap = min(float(cap), float(turn_approach_cap_mps))
        if float(cap) < previous_cap:
            limiting_owner = "turn_approach_cap"
    if decision in {"intersection_turn_left", "intersection_turn_right"}:
        turn_cap_mps = max(
            0.1, float(config.get("full_intersection_turn_speed_cap_mps", 2.2))
        )
        active_constraints.append("turn_cap")
        previous_cap = float(cap)
        cap = min(
            float(cap),
            float(turn_cap_mps),
        )
        if float(cap) < previous_cap:
            limiting_owner = "turn_cap"
    lane_change_cap_mps = None
    if decision in {"lane_change_left", "lane_change_right"}:
        # A route-required lane change can start immediately after ego is
        # still accelerating from a stop (e.g. CHANGELANELEFT right at
        # spawn, with no straight lead-in to reach cruise speed first).
        # Without a cap, the lateral S-curve gets executed while
        # longitudinal speed is still an unsettled transient, and the
        # combined demand can press the actual footprint into the lane
        # boundary. Capping speed for the whole maneuver -- not just
        # smoothing the transition into it -- keeps the lateral maneuver's
        # dynamics decoupled from however unsettled the longitudinal speed
        # still is.
        lane_change_cap_mps = max(
            0.1, float(config.get("full_lane_change_speed_cap_mps", 3.0))
        )
        active_constraints.append("lane_change_cap")
        previous_cap = float(cap)
        cap = min(float(cap), float(lane_change_cap_mps))
        if float(cap) < previous_cap:
            limiting_owner = "lane_change_cap"
    following_active = False
    following_cap_mps = None
    following_gap_m = None
    desired_gap_m = None
    if front_gap_m is not None and math.isfinite(float(front_gap_m)) and not stop_goal:
        gap_m = max(0.0, float(front_gap_m))
        following_gap_m = float(gap_m)
        emergency_gap_m = max(
            0.5,
            float(config.get("following_emergency_gap_m", 3.0)),
        )
        standstill_gap_m = max(
            emergency_gap_m,
            float(config.get("following_standstill_gap_m", 5.0)),
        )
        time_headway_s = max(
            0.1,
            float(config.get("following_time_headway_s", 1.5)),
        )
        free_gap_margin_m = max(
            1.0,
            float(config.get("following_free_gap_margin_m", 6.0)),
        )
        minimum_follow_speed_mps = max(
            0.0,
            float(config.get("following_minimum_speed_mps", 0.35)),
        )
        desired_gap_m = standstill_gap_m + time_headway_s * max(0.0, float(ego_speed_mps))
        free_gap_m = desired_gap_m + free_gap_margin_m
        if gap_m <= emergency_gap_m:
            cap = 0.0
            stop_goal = True
            reason = _join_reason(reason, "speed_plan_obstacle_emergency_stop")
        elif gap_m < free_gap_m:
            ratio = (gap_m - emergency_gap_m) / max(
                1.0e-6,
                free_gap_m - emergency_gap_m,
            )
            smooth_ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            follow_cap = minimum_follow_speed_mps + smooth_ratio * max(
                0.0,
                float(cap) - minimum_follow_speed_mps,
            )
            following_cap_mps = float(follow_cap)
            active_constraints.append("following_cap")
            previous_cap = float(cap)
            cap = min(float(cap), float(follow_cap))
            if float(cap) < previous_cap:
                limiting_owner = "following_cap"
            following_active = True
    if stop_goal:
        cap = 0.0
        active_constraints.append("stop_zero")
        limiting_owner = (
            "emergency_stop"
            if decision == "emergency_brake" or "obstacle_emergency_stop" in reason
            else "normal_stop"
        )
    if not math.isfinite(cap):
        cap = 0.0
    if decision in {"intersection_turn_left", "intersection_turn_right"}:
        reason = _join_reason(reason, "speed_plan_turn_cap")
    elif turn_approach_cap_mps is not None and turn_approach_cap_mps < requested:
        reason = _join_reason(reason, "speed_plan_turn_approach_cap")
    if decision in {"lane_change_left", "lane_change_right"}:
        reason = _join_reason(reason, "speed_plan_lane_change_cap")
    if stop_goal:
        reason = _join_reason(reason, "speed_plan_stop_zero")
    elif following_active:
        reason = _join_reason(reason, "speed_plan_continuous_following")
    elif scenario_cap is not None and float(cap) < float(requested):
        reason = _join_reason(reason, "speed_plan_scenario_cap")
    return SpeedPlan(
        target_speed_mps=float(cap),
        speed_cap_mps=float(cap),
        stop_goal_active=bool(stop_goal),
        reason=str(reason),
        front_gap_m=following_gap_m,
        desired_follow_gap_m=desired_gap_m,
        continuous_following_active=bool(following_active),
        requested_speed_mps=float(requested),
        scenario_cap_mps=scenario_cap_value,
        turn_cap_mps=turn_cap_mps,
        lane_change_cap_mps=lane_change_cap_mps,
        following_cap_mps=following_cap_mps,
        turn_approach_cap_mps=turn_approach_cap_mps,
        upcoming_turn_distance_m=finite_turn_distance_m,
        limiting_owner=str(limiting_owner),
        active_constraints=tuple(active_constraints),
    )


def _join_reason(first: str, second: str) -> str:
    if not first:
        return str(second)
    if str(second) in str(first).split(";"):
        return str(first)
    return f"{first};{second}"
