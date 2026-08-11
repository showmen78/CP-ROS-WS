"""Candidate-level behavior/reference selection before MPC tracking.

This module is the explicit boundary between behavior intent generation and
the final MPC call.  It keeps the orchestration testable: each candidate owns a
behavior command, a reference, a contract result, and a lightweight feasibility
score.  The bridge may optionally add an expensive MPC probe later, but the
default gate is deterministic and side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Callable, Dict, Mapping, Optional, Sequence

from cpx_planning.MPC.lane_keep import RoadEnvelopeBlock, normalize_lane_reference_sample
from .reference_contract import ReferenceValidationResult
from .stage_contracts import ManeuverCommitment


@dataclass(frozen=True)
class CandidateBehaviorIntent:
    name: str
    decision: str
    target_lane_id: int
    target_speed_mps: float
    base_cost: float = 0.0
    reason: str = ""
    stop_goal_active: bool = False
    stop_target: Optional[Mapping[str, object]] = None
    trajectory_variant: str = ""
    lane_change_duration_s: float = 0.0


@dataclass
class CandidateReferenceResult:
    intent: CandidateBehaviorIntent
    destination_state: Sequence[float]
    lane_center_reference: Sequence[Mapping[str, object]]
    reference_debug: Dict[str, object] = field(default_factory=dict)
    contract_result: Optional[ReferenceValidationResult] = None
    feasibility_status: str = "unchecked"
    feasibility_reason: str = ""
    feasibility_cost: float = 0.0
    total_cost: float = 0.0
    risk_bucket: str = ""

    @property
    def feasible(self) -> bool:
        return str(self.feasibility_status) in {"feasible", "mpc_probe_solved"}

    def summary_row(self) -> Dict[str, object]:
        contract_reason = ""
        if self.contract_result is not None and not bool(self.contract_result.valid):
            contract_reason = str(self.contract_result.reason())
        return {
            "name": str(self.intent.name),
            "decision": str(self.intent.decision),
            "target_lane_id": int(self.intent.target_lane_id),
            "target_speed_mps": float(self.intent.target_speed_mps),
            "trajectory_variant": str(self.intent.trajectory_variant),
            "lane_change_duration_s": float(self.intent.lane_change_duration_s),
            "base_cost": float(self.intent.base_cost),
            "feasibility_status": str(self.feasibility_status),
            "feasibility_reason": str(self.feasibility_reason),
            "contract_reason": str(contract_reason),
            "total_cost": float(self.total_cost),
        }


@dataclass(frozen=True)
class CandidateSelectionOutcome:
    selected: Optional[CandidateReferenceResult]
    status: str
    reason: str


@dataclass(frozen=True)
class HumanLaneChangeProfile:
    variant: str
    duration_s: float
    speed_scale: float
    extra_cost: float
    reason: str


def build_human_lane_change_profiles(
    *,
    ego_speed_mps: float,
    lane_width_m: float,
    target_lane_prediction_risk: Mapping[str, object],
    available_distance_m: Optional[float] = None,
    minimum_duration_s: float = 3.0,
    maximum_duration_s: float = 6.5,
) -> tuple[HumanLaneChangeProfile, ...]:
    """Build comfort-bounded lane-change styles without automatic braking.

    A quintic lateral transition has peak acceleration proportional to
    ``lane_width / duration**2`` and peak jerk proportional to
    ``lane_width / duration**3``. Durations therefore come from physical
    comfort limits; speed only changes the spatial length of the maneuver.
    Longitudinal speed is preserved on a clear target lane and reduced only
    when the predicted front gap is genuinely constraining.
    """

    speed = max(0.0, float(ego_speed_mps))
    width = max(0.5, abs(float(lane_width_m)))
    min_duration = max(1.0, float(minimum_duration_s))
    max_duration = max(min_duration, float(maximum_duration_s))
    risk = dict(target_lane_prediction_risk or {})

    def duration_for_limits(accel_limit: float, jerk_limit: float) -> float:
        accel_duration = math.sqrt(5.7735 * width / max(0.1, accel_limit))
        jerk_duration = (60.0 * width / max(0.1, jerk_limit)) ** (1.0 / 3.0)
        return min(max_duration, max(min_duration, accel_duration, jerk_duration))

    front_gap = _finite_optional(risk.get("min_front_gap_m"))
    rear_gap = _finite_optional(risk.get("min_rear_gap_m"))
    desired_front_gap = 5.0 + 1.5 * speed
    front_constrained = (
        front_gap is not None and float(front_gap) < float(desired_front_gap + 6.0)
    )
    rear_constrained = rear_gap is not None and float(rear_gap) < max(10.0, 1.0 * speed)

    normal_scale = 1.0
    conservative_scale = 1.0
    speed_reason = "clear_gap_maintain_speed"
    if bool(front_constrained) and not bool(rear_constrained):
        gap_ratio = max(
            0.0,
            min(1.0, (float(front_gap) - 5.0) / max(1.0, desired_front_gap)),
        )
        normal_scale = max(0.85, 0.85 + 0.15 * gap_ratio)
        conservative_scale = max(0.75, 0.75 + 0.20 * gap_ratio)
        speed_reason = "predicted_front_gap_mild_slowdown"
    elif bool(rear_constrained):
        # Slowing while a rear vehicle is close shrinks the rear gap. Keep
        # speed and let authorization/prediction defer the maneuver if needed.
        speed_reason = "predicted_rear_gap_maintain_speed"

    assertive_duration = duration_for_limits(1.8, 4.0)
    normal_duration = duration_for_limits(1.3, 2.5)
    conservative_duration = duration_for_limits(1.0, 1.5)

    urgency_reason = ""
    if (
        available_distance_m is not None
        and math.isfinite(float(available_distance_m))
        and speed > 0.5
    ):
        available_time_s = max(0.0, float(available_distance_m)) / speed
        if available_time_s < normal_duration:
            urgency_reason = ";route_distance_prefers_assertive"

    return (
        HumanLaneChangeProfile(
            variant="assertive",
            duration_s=float(assertive_duration),
            speed_scale=1.0,
            extra_cost=2.8 if urgency_reason else 3.2,
            reason=f"human_profile:{speed_reason}{urgency_reason}",
        ),
        HumanLaneChangeProfile(
            variant="normal",
            duration_s=float(normal_duration),
            speed_scale=float(normal_scale),
            extra_cost=2.0 if not urgency_reason else 3.5,
            reason=f"human_profile:{speed_reason}{urgency_reason}",
        ),
        HumanLaneChangeProfile(
            variant="conservative",
            duration_s=float(conservative_duration),
            speed_scale=float(conservative_scale),
            extra_cost=2.6 if not urgency_reason else 4.5,
            reason=f"human_profile:{speed_reason}{urgency_reason}",
        ),
    )


def build_candidate_intents(
    *,
    selected_decision: str,
    selected_target_lane_id: int,
    current_lane_id: int,
    target_speed_mps: float,
    candidate_lane_ids: Sequence[int],
    lane_safety_scores: Mapping[int, float],
    lane_prediction_risks: Mapping[int, Mapping[str, object]],
    stop_goal_active: bool,
    traffic_stop_active: bool,
    lane_change_authorized: bool,
    lane_change_authorized_target_lane_id: int,
    allow_lane_change_candidates: bool,
    stop_target: Optional[Mapping[str, object]] = None,
    lane_change_assertive_duration_s: float = 3.2,
    lane_change_normal_duration_s: float = 4.0,
    lane_change_conservative_duration_s: float = 5.5,
    lane_change_assertive_speed_scale: float = 1.0,
    lane_change_normal_speed_scale: float = 0.9,
    lane_change_conservative_speed_scale: float = 0.7,
    lane_change_authorization_source: str = "route",
    lane_change_defer_cost: float = 10.0,
    turn_obstacle_stop_defer_cost: float = 90.0,
    human_like_lane_change_enabled: bool = False,
    ego_speed_mps: float = 0.0,
    lane_width_m: float = 3.5,
    lane_change_available_distance_m: Optional[float] = None,
    human_lane_change_min_duration_s: float = 3.0,
    human_lane_change_max_duration_s: float = 6.5,
) -> list[CandidateBehaviorIntent]:
    """Generate behavior candidates for the reference/MPC boundary."""

    intents: list[CandidateBehaviorIntent] = []
    current_lane_id = int(current_lane_id or 0)
    selected_target_lane_id = int(selected_target_lane_id or current_lane_id)
    target_speed_mps = max(0.0, float(target_speed_mps))

    def add(intent: CandidateBehaviorIntent) -> None:
        key = (
            str(intent.decision),
            int(intent.target_lane_id),
            round(float(intent.target_speed_mps), 2),
            str(intent.trajectory_variant),
            round(float(intent.lane_change_duration_s), 2),
        )
        existing = {
            (
                str(row.decision),
                int(row.target_lane_id),
                round(float(row.target_speed_mps), 2),
                str(row.trajectory_variant),
                round(float(row.lane_change_duration_s), 2),
            )
            for row in intents
        }
        if key not in existing:
            intents.append(intent)

    mandatory_stop_required = bool(traffic_stop_active) or str(selected_decision) in {
        "stop_at_intersection",
        "stop_sign",
        "emergency_brake",
    }
    if bool(mandatory_stop_required):
        add(CandidateBehaviorIntent(
            name="stop",
            decision=(
                str(selected_decision)
                if str(selected_decision) in {"stop_at_intersection", "stop_sign", "emergency_brake"}
                else "stop_at_intersection"
            ),
            target_lane_id=int(current_lane_id),
            target_speed_mps=0.0,
            base_cost=0.0,
            reason="stop_required",
            stop_goal_active=True,
            stop_target=dict(stop_target or {}) if isinstance(stop_target, Mapping) else None,
        ))
        return intents

    # A close lead obstacle is different from a traffic-control stop. Keep a
    # stop candidate available, but do not erase an already authorized escape
    # lane. Candidate prediction/contract/MPC checks decide whether changing
    # lane is actually safer than stopping behind the obstacle.
    turn_in_progress = str(selected_decision) in {
        "intersection_turn_left",
        "intersection_turn_right",
    }
    if bool(stop_goal_active):
        add(CandidateBehaviorIntent(
            name="obstacle_stop",
            decision="stop_at_intersection",
            target_lane_id=int(current_lane_id),
            target_speed_mps=0.0,
            # A turn's own reference generator is structurally different
            # from the stop reference's (arc vs. straight-line heading), so
            # winning this candidate for only a tick or two -- e.g. a
            # vehicle briefly crossing close during an unprotected turn --
            # forces two back-to-back reference-generator handoffs whose
            # recomputed headings can jump 30+ degrees, well beyond what the
            # position-only blend absorbs. Handicapping this candidate while
            # a turn is already committed means a momentary proximity blip
            # gets absorbed by the turn's own (continuous) speed reduction
            # instead of a generator swap; a genuinely close/sustained
            # threat still outweighs the handicap and wins.
            base_cost=(
                max(0.0, float(turn_obstacle_stop_defer_cost))
                if bool(turn_in_progress)
                else 0.0
            ),
            reason=(
                "front_obstacle_stop_candidate_defer_turn_in_progress"
                if bool(turn_in_progress)
                else "front_obstacle_stop_candidate"
            ),
            stop_goal_active=True,
            stop_target=(
                dict(stop_target)
                if isinstance(stop_target, Mapping)
                else None
            ),
        ))

    if str(selected_decision) in {"intersection_turn_left", "intersection_turn_right"}:
        add(CandidateBehaviorIntent(
            name=str(selected_decision),
            decision=str(selected_decision),
            target_lane_id=int(current_lane_id),
            target_speed_mps=float(target_speed_mps),
            base_cost=_lane_cost(
                lane_id=int(current_lane_id),
                current_lane_id=int(current_lane_id),
                lane_safety_scores=lane_safety_scores,
                lane_prediction_risks=lane_prediction_risks,
            ),
            reason="route_option_turn_required",
        ))
        return intents

    lane_change_committed = bool(lane_change_authorized) and (
        str(lane_change_authorization_source or "").strip().lower() == "route"
        or str(selected_decision) in {"lane_change_left", "lane_change_right"}
    )
    add(CandidateBehaviorIntent(
        name="keep_lane",
        decision="lane_follow",
        target_lane_id=int(current_lane_id),
        target_speed_mps=float(target_speed_mps),
        base_cost=(
            max(0.0, float(lane_change_defer_cost))
            if bool(lane_change_committed)
            else 0.0
        ) + _lane_cost(
            lane_id=int(current_lane_id),
            current_lane_id=int(current_lane_id),
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=lane_prediction_risks,
        ),
        reason=(
            "defer_route_lane_change"
            if bool(lane_change_committed)
            else "default_keep_lane"
        ),
    ))
    add(CandidateBehaviorIntent(
        name="yield_slow_down",
        decision="lane_follow",
        target_lane_id=int(current_lane_id),
        # The 0.6 m/s floor keeps this candidate from proposing an
        # unreasonably slow creep when the baseline speed is comfortably
        # above it. But once the baseline itself has already decayed below
        # 0.6 (e.g. braking for a close lead vehicle/red light), the floor
        # would make "yield" target a HIGHER speed than "keep_lane" itself --
        # backwards for a candidate meant to be the more conservative option.
        # That paces this candidate's own reference faster than the vehicle
        # can actually be going, producing a reference whose points sit
        # further ahead than ego's real trajectory reaches -- a large,
        # sustained lane-center tracking-cost mismatch for as long as this
        # candidate keeps winning. Capping at the baseline speed preserves
        # the floor's purpose everywhere it doesn't invert the ordering.
        target_speed_mps=min(
            float(target_speed_mps),
            max(0.6, 0.55 * float(target_speed_mps)),
        ),
        base_cost=4.0 + (
            max(0.0, float(lane_change_defer_cost))
            if bool(lane_change_committed)
            else 0.0
        ) + _lane_cost(
            lane_id=int(current_lane_id),
            current_lane_id=int(current_lane_id),
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=lane_prediction_risks,
        ),
        reason=(
            "conservative_yield_defer_route_lane_change"
            if bool(lane_change_committed)
            else "conservative_yield_candidate"
        ),
    ))
    if str(selected_decision) == "lane_follow":
        add(CandidateBehaviorIntent(
            name="selected_behavior",
            decision=str(selected_decision),
            target_lane_id=int(selected_target_lane_id),
            target_speed_mps=float(target_speed_mps),
            base_cost=1.0 + _lane_cost(
                lane_id=int(selected_target_lane_id),
                current_lane_id=int(current_lane_id),
                lane_safety_scores=lane_safety_scores,
                lane_prediction_risks=lane_prediction_risks,
            ),
            reason="behavior_planner_selected",
        ))

    if bool(allow_lane_change_candidates) and bool(lane_change_authorized):
        target_lane_id = int(lane_change_authorized_target_lane_id or 0)
        if target_lane_id != 0 and target_lane_id != int(current_lane_id):
            authorization_source = str(
                lane_change_authorization_source or "route"
            ).strip().lower()
            if authorization_source not in {"route", "opportunistic"}:
                authorization_source = "route"
            decision = (
                str(selected_decision)
                if authorization_source == "opportunistic"
                and str(selected_decision) in {"lane_change_left", "lane_change_right"}
                else (
                    "lane_change_left"
                    if target_lane_id > int(current_lane_id)
                    else "lane_change_right"
                )
            )
            lane_cost = _lane_cost(
                lane_id=int(target_lane_id),
                current_lane_id=int(current_lane_id),
                lane_safety_scores=lane_safety_scores,
                lane_prediction_risks=lane_prediction_risks,
            )
            if bool(human_like_lane_change_enabled):
                profiles = build_human_lane_change_profiles(
                    ego_speed_mps=float(ego_speed_mps),
                    lane_width_m=float(lane_width_m),
                    target_lane_prediction_risk=dict(
                        lane_prediction_risks.get(int(target_lane_id), {}) or {}
                    ),
                    available_distance_m=lane_change_available_distance_m,
                    minimum_duration_s=float(human_lane_change_min_duration_s),
                    maximum_duration_s=float(human_lane_change_max_duration_s),
                )
            else:
                profiles = (
                    HumanLaneChangeProfile("assertive", lane_change_assertive_duration_s, lane_change_assertive_speed_scale, 3.0, "legacy_profile"),
                    HumanLaneChangeProfile("normal", lane_change_normal_duration_s, lane_change_normal_speed_scale, 2.0, "legacy_profile"),
                    HumanLaneChangeProfile("conservative", lane_change_conservative_duration_s, lane_change_conservative_speed_scale, 2.5, "legacy_profile"),
                )
            for profile in profiles:
                add(CandidateBehaviorIntent(
                    name=f"{authorization_source}_{decision}_{profile.variant}",
                    decision=str(decision),
                    target_lane_id=int(target_lane_id),
                    target_speed_mps=max(
                        0.8,
                        float(target_speed_mps) * float(profile.speed_scale),
                    ),
                    base_cost=float(profile.extra_cost) + float(lane_cost),
                    reason=(
                        f"{authorization_source}_lane_change_authorized"
                        if str(profile.reason) == "legacy_profile"
                        else (
                            f"{authorization_source}_lane_change_authorized;"
                            f"{profile.reason}"
                        )
                    ),
                    trajectory_variant=str(profile.variant),
                    lane_change_duration_s=float(profile.duration_s),
                ))

    for lane_id in list(candidate_lane_ids or []):
        try:
            candidate_lane_id = int(lane_id)
        except Exception:
            continue
        if candidate_lane_id == 0 or candidate_lane_id == int(current_lane_id):
            continue
        if not bool(allow_lane_change_candidates):
            continue
        if not bool(lane_change_authorized) or candidate_lane_id != int(lane_change_authorized_target_lane_id or 0):
            continue
        decision = "lane_change_left" if candidate_lane_id > int(current_lane_id) else "lane_change_right"
        # The authorized target already owns assertive/normal/conservative
        # variants above. Other lanes remain forbidden by the authorization
        # boundary and must not leak into reference generation.

    return intents


def shape_lane_change_reference(
    *,
    target_reference: Sequence[Mapping[str, object]],
    source_reference: Sequence[Mapping[str, object]],
    duration_s: float,
    dt_s: float,
    current_lane_id: int,
    target_lane_id: int,
    target_speed_mps: float,
    ego_x_m: Optional[float] = None,
    ego_y_m: Optional[float] = None,
    initial_progress_floor: float = 0.0,
    blend_geometry: bool = True,
) -> list[Dict[str, object]]:
    """Blend lane paths without resetting lateral progress every replan.

    When blend_geometry is False, the returned samples keep the target
    lane's own (unblended) x/y -- only the progress bookkeeping below (which
    other code relies on for commitment/completion gating) is still
    computed. This lets MPC's own QP determine the transient path via its
    hard dynamics/actuator constraints instead of tracking a pre-shaped
    geometric blend that can demand more steering rate than those
    constraints allow.
    """

    target = [dict(sample) for sample in list(target_reference or [])]
    source = [dict(sample) for sample in list(source_reference or [])]
    if not target or not source:
        return target
    target = _align_target_reference_to_source(
        source_reference=source,
        target_reference=target,
    )
    count = min(len(target), len(source))
    duration = max(float(dt_s), float(duration_s))
    separations_sq = [
        _sample_separation_sq(source[index], target[index])
        for index in range(count)
    ]
    # Anchor the lateral-progress projection on the EARLIEST point where the
    # source/target lane-center paths have meaningfully diverged, not the
    # point of maximum separation. Right after a turn/intersection the two
    # paths are not parallel; projecting ego onto a distant, non-parallel
    # anchor can spuriously read as "already most of the way to the target
    # lane" even for a lane change that has just started, producing a first
    # reference sample whose lateral offset trips the reference-contract
    # veto and permanently stops the vehicle (the same spurious projection
    # then repeats every subsequent tick, since nothing about the geometry
    # changes while the vehicle is stopped). A nearby, meaningfully
    # separated anchor stays representative of the true local lane gap
    # while still measuring genuine progress on an already-in-progress lane
    # change, where source/target are close to parallel near ego.
    min_anchor_separation_sq = min(max(separations_sq, default=0.0), 1.0 ** 2)
    anchor_index = next(
        (
            index
            for index, separation_sq in enumerate(separations_sq)
            if separation_sq >= min_anchor_separation_sq
        ),
        max(range(count), key=lambda index: separations_sq[index]),
    )
    initial_alpha = max(
        min(0.98, max(0.0, float(initial_progress_floor))),
        _lane_change_initial_progress(
        source_sample=source[anchor_index],
        target_sample=target[anchor_index],
        ego_x_m=ego_x_m,
        ego_y_m=ego_y_m,
        ),
    )
    remaining_duration = max(
        float(dt_s),
        float(duration) * max(0.15, 1.0 - float(initial_alpha)),
    )
    result: list[Dict[str, object]] = []
    for index in range(count):
        t_s = float(index + 1) * max(1.0e-3, float(dt_s))
        u = min(1.0, max(0.0, t_s / remaining_duration))
        smooth_progress = 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5
        alpha = float(initial_alpha) + (
            1.0 - float(initial_alpha)
        ) * float(smooth_progress)
        source_sample = source[index]
        target_sample = target[index]
        sx = float(source_sample.get("x_ref_m", source_sample.get("x", 0.0)))
        sy = float(source_sample.get("y_ref_m", source_sample.get("y", 0.0)))
        tx = float(target_sample.get("x_ref_m", target_sample.get("x", sx)))
        ty = float(target_sample.get("y_ref_m", target_sample.get("y", sy)))
        sample = dict(target_sample)
        if bool(blend_geometry):
            sample["x_ref_m"] = sx + float(alpha) * (tx - sx)
            sample["y_ref_m"] = sy + float(alpha) * (ty - sy)
        else:
            sample["x_ref_m"] = float(tx)
            sample["y_ref_m"] = float(ty)
        sample["x"] = float(sample["x_ref_m"])
        sample["y"] = float(sample["y_ref_m"])
        sample["lane_id"] = (
            int(current_lane_id) if float(alpha) < 0.5 else int(target_lane_id)
        )
        sample["lane_transition_kind"] = "lateral_lane_change"
        sample["lane_change_progress"] = float(alpha)
        sample["lane_change_initial_progress"] = float(initial_alpha)
        sample["speed_ref_mps"] = max(0.0, float(target_speed_mps))
        sample["v_ref_mps"] = max(0.0, float(target_speed_mps))
        sample["speed_mps"] = max(0.0, float(target_speed_mps))
        result.append(sample)
    result.extend(dict(sample) for sample in target[count:])
    for index, sample in enumerate(result):
        if len(result) < 2:
            break
        first = sample if index + 1 < len(result) else result[index - 1]
        second = result[index + 1] if index + 1 < len(result) else sample
        dx = float(second["x_ref_m"]) - float(first["x_ref_m"])
        dy = float(second["y_ref_m"]) - float(first["y_ref_m"])
        if math.hypot(dx, dy) > 1.0e-6:
            sample["heading_rad"] = math.atan2(dy, dx)
    return result


def select_comfortable_lane_change_duration_s(
    *,
    target_reference: Sequence[Mapping[str, object]],
    source_reference: Sequence[Mapping[str, object]],
    initial_duration_s: float,
    duration_max_s: float,
    dt_s: float,
    current_lane_id: int,
    target_lane_id: int,
    target_speed_mps: float,
    lateral_accel_limit_mps2: float,
    curvature_fn: Callable[[Sequence[Mapping[str, object]]], float],
    ego_x_m: Optional[float] = None,
    ego_y_m: Optional[float] = None,
    initial_progress_floor: float = 0.0,
    duration_growth_factor: float = 1.3,
    max_iterations: int = 8,
) -> tuple[float, list[Dict[str, object]], str]:
    """Widen duration_s (never shrink it) until shape_lane_change_reference's
    blended path keeps v^2*curvature within the comfort limit, or the max
    duration cap is hit.

    A fixed blend duration ignores how far the lane actually is or how fast
    the vehicle is going, so a short duration can demand more lateral
    acceleration than is comfortable. Widening the schedule spreads the same
    lateral crossing over more distance/time, which only ever makes the path
    gentler -- never a shorter, sharper one.
    """

    duration_s = max(float(dt_s), float(initial_duration_s))
    duration_cap_s = max(float(duration_s), float(duration_max_s))
    growth_factor = max(1.0 + 1.0e-3, float(duration_growth_factor))
    attempts = max(1, int(max_iterations))
    shaped: list[Dict[str, object]] = []
    for attempt in range(attempts):
        shaped = shape_lane_change_reference(
            target_reference=target_reference,
            source_reference=source_reference,
            duration_s=float(duration_s),
            dt_s=float(dt_s),
            current_lane_id=int(current_lane_id),
            target_lane_id=int(target_lane_id),
            target_speed_mps=float(target_speed_mps),
            ego_x_m=ego_x_m,
            ego_y_m=ego_y_m,
            initial_progress_floor=float(initial_progress_floor),
            blend_geometry=True,
        )
        curvature_1pm = max(0.0, float(curvature_fn(shaped)))
        implied_lateral_accel_mps2 = float(target_speed_mps) ** 2 * curvature_1pm
        if float(implied_lateral_accel_mps2) <= float(lateral_accel_limit_mps2):
            return float(duration_s), shaped, "lane_change_duration_within_comfort_limit"
        # Return using *this* attempt's own (duration_s, shaped) pair, not a
        # duration_s that was grown for a next attempt that never runs --
        # otherwise the returned duration and shaped path would mismatch.
        if float(duration_s) >= float(duration_cap_s) or attempt == attempts - 1:
            return float(duration_s), shaped, "lane_change_duration_capped_at_max"
        duration_s = min(float(duration_cap_s), float(duration_s) * float(growth_factor))
    return float(duration_s), shaped, "lane_change_duration_capped_at_max"


def _align_target_reference_to_source(
    *,
    source_reference: Sequence[Mapping[str, object]],
    target_reference: Sequence[Mapping[str, object]],
) -> list[Dict[str, object]]:
    """Match parallel-lane samples by a monotonic longitudinal station.

    CARLA lane-center queries may start the adjacent lane one or more samples
    ahead of the current lane. Blending equal array indices then adds an
    unintended longitudinal motion to the lateral lane-change polynomial and
    creates artificial curvature spikes.
    """

    source = [dict(sample) for sample in list(source_reference or [])]
    target = [dict(sample) for sample in list(target_reference or [])]
    if not source or not target:
        return target
    aligned: list[Dict[str, object]] = []
    target_index = 0
    for source_index, source_sample in enumerate(source):
        sx = float(source_sample.get("x_ref_m", source_sample.get("x", 0.0)))
        sy = float(source_sample.get("y_ref_m", source_sample.get("y", 0.0)))
        search_end = min(len(target), target_index + 8)
        best_index = min(
            range(target_index, search_end),
            key=lambda index: _sample_distance_sq_xy(
                x_m=sx,
                y_m=sy,
                sample=target[index],
            ),
        )
        target_index = max(target_index, int(best_index))
        matched = dict(target[target_index])
        tx = float(matched.get("x_ref_m", matched.get("x", sx)))
        ty = float(matched.get("y_ref_m", matched.get("y", sy)))
        previous_source = source[max(0, source_index - 1)]
        next_source = source[min(len(source) - 1, source_index + 1)]
        tangent_x = float(
            next_source.get("x_ref_m", next_source.get("x", sx))
        ) - float(
            previous_source.get("x_ref_m", previous_source.get("x", sx))
        )
        tangent_y = float(
            next_source.get("y_ref_m", next_source.get("y", sy))
        ) - float(
            previous_source.get("y_ref_m", previous_source.get("y", sy))
        )
        tangent_norm = max(1.0e-6, math.hypot(tangent_x, tangent_y))
        normal_x = -tangent_y / tangent_norm
        normal_y = tangent_x / tangent_norm
        lateral_offset = (tx - sx) * normal_x + (ty - sy) * normal_y
        matched["x_ref_m"] = sx + float(lateral_offset) * normal_x
        matched["y_ref_m"] = sy + float(lateral_offset) * normal_y
        matched["x"] = float(matched["x_ref_m"])
        matched["y"] = float(matched["y_ref_m"])
        aligned.append(matched)
    return aligned


def _sample_distance_sq_xy(
    *,
    x_m: float,
    y_m: float,
    sample: Mapping[str, object],
) -> float:
    tx = float(sample.get("x_ref_m", sample.get("x", 0.0)))
    ty = float(sample.get("y_ref_m", sample.get("y", 0.0)))
    return float((tx - float(x_m)) ** 2 + (ty - float(y_m)) ** 2)


def _sample_separation_sq(
    source_sample: Mapping[str, object],
    target_sample: Mapping[str, object],
) -> float:
    try:
        source_x = float(source_sample.get("x_ref_m", source_sample.get("x", 0.0)))
        source_y = float(source_sample.get("y_ref_m", source_sample.get("y", 0.0)))
        target_x = float(target_sample.get("x_ref_m", target_sample.get("x", source_x)))
        target_y = float(target_sample.get("y_ref_m", target_sample.get("y", source_y)))
        return float((target_x - source_x) ** 2 + (target_y - source_y) ** 2)
    except Exception:
        return 0.0


def _lane_change_initial_progress(
    *,
    source_sample: Mapping[str, object],
    target_sample: Mapping[str, object],
    ego_x_m: Optional[float],
    ego_y_m: Optional[float],
) -> float:
    """Project ego between source and target lanes as replanning memory."""

    if ego_x_m is None or ego_y_m is None:
        return 0.0
    try:
        source_x = float(source_sample.get("x_ref_m", source_sample.get("x", 0.0)))
        source_y = float(source_sample.get("y_ref_m", source_sample.get("y", 0.0)))
        target_x = float(target_sample.get("x_ref_m", target_sample.get("x", source_x)))
        target_y = float(target_sample.get("y_ref_m", target_sample.get("y", source_y)))
        delta_x = target_x - source_x
        delta_y = target_y - source_y
        denominator = delta_x * delta_x + delta_y * delta_y
        if denominator <= 1.0e-6:
            return 0.0
        progress = (
            (float(ego_x_m) - source_x) * delta_x
            + (float(ego_y_m) - source_y) * delta_y
        ) / denominator
        return min(0.98, max(0.0, float(progress)))
    except Exception:
        return 0.0


def build_route_tracking_lane_change_envelope_blocks(
    *,
    source_reference: Sequence[Mapping[str, object]],
    target_reference: Sequence[Mapping[str, object]],
    master_step_count: int,
    step_distance_m: float,
    road_boundary_margin_m: float = 0.5,
    default_lane_width_m: float = 4.0,
    length_pad_m: float = 3.0,
    min_half_width_m: float = 0.3,
) -> list[RoadEnvelopeBlock]:
    """Build two static drivable-corridor blocks for a locked lane change.

    One block anchors on the source lane, one on the target lane, both
    computed once here (at lock time) and never moved again for the
    duration of the maneuver. This is what lets a road-boundary constraint
    built from these blocks stay satisfiable even as the *tracked
    reference* switches from the source lane to the target lane mid-
    maneuver -- unlike a single reference line, the union of these two
    fixed blocks never jumps.
    """

    source = [dict(sample) for sample in list(source_reference or [])]
    target = [dict(sample) for sample in list(target_reference or [])]
    if not source or not target:
        return []
    aligned_target = _align_target_reference_to_source(
        source_reference=source,
        target_reference=target,
    )
    if not aligned_target:
        return []

    source_anchor = normalize_lane_reference_sample(
        source[0],
        default_lane_width_m=float(default_lane_width_m),
    )
    target_anchor = normalize_lane_reference_sample(
        aligned_target[0],
        default_lane_width_m=float(default_lane_width_m),
    )
    if source_anchor is None or target_anchor is None:
        return []

    full_length_m = max(0.0, float(master_step_count) - 1.0) * max(0.0, float(step_distance_m))
    half_length_m = 0.5 * full_length_m + max(0.0, float(length_pad_m))
    half_length_m = max(1.0, float(half_length_m))
    margin_m = max(0.0, float(road_boundary_margin_m))

    blocks: list[RoadEnvelopeBlock] = []
    for anchor in (source_anchor, target_anchor):
        half_width_m = max(
            float(min_half_width_m),
            0.5 * (float(anchor.left_road_width_m) + float(anchor.right_road_width_m)) - margin_m,
        )
        heading_rad = float(anchor.heading_rad)
        center_x_m = float(anchor.x_center_m) + half_length_m * math.cos(heading_rad)
        center_y_m = float(anchor.y_center_m) + half_length_m * math.sin(heading_rad)
        blocks.append(
            RoadEnvelopeBlock(
                x_center_m=float(center_x_m),
                y_center_m=float(center_y_m),
                heading_rad=float(heading_rad),
                half_length_m=float(half_length_m),
                half_width_m=float(half_width_m),
            )
        )
    return blocks


def evaluate_candidate_reference(
    *,
    candidate: CandidateReferenceResult,
    ego_state: Sequence[float],
    object_snapshots: Sequence[Mapping[str, object]],
    prediction_trajectories: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    current_lane_id: int,
    min_object_distance_m: float = 2.0,
    contract_invalid_cost: float = 1000.0,
    infeasible_cost: float = 10000.0,
    lane_change_duration_cost_per_s: float = 0.75,
    previous_risk_bucket: str = "",
    risk_hysteresis_margin_m: float = 0.0,
) -> CandidateReferenceResult:
    """Attach lightweight feasibility and cost to a generated reference."""

    cost = float(candidate.intent.base_cost)
    reasons: list[str] = []
    feasible = True
    if candidate.contract_result is None:
        feasible = False
        reasons.append("missing_contract_result")
        cost += float(contract_invalid_cost)
    elif not bool(candidate.contract_result.valid):
        contract_reason = str(candidate.contract_result.reason())
        if _is_soft_turn_contract_violation(
            decision=str(candidate.intent.decision),
            contract_reason=str(contract_reason),
        ):
            reasons.append("turn_contract_softened:" + str(contract_reason))
            cost += 0.25 * float(contract_invalid_cost)
        else:
            feasible = False
            reasons.append("contract_invalid:" + str(contract_reason))
            cost += float(contract_invalid_cost)

    lane_risk_cost, lane_risk_reason, risk_bucket = _lane_change_risk_cost(
        lane_id=int(candidate.intent.target_lane_id),
        current_lane_id=int(current_lane_id),
        object_snapshots=object_snapshots,
        prediction_trajectories=prediction_trajectories,
        reference_samples=candidate.lane_center_reference,
        min_object_distance_m=float(min_object_distance_m),
        previous_bucket=str(previous_risk_bucket),
        hysteresis_margin_m=float(risk_hysteresis_margin_m),
    )
    candidate.risk_bucket = str(risk_bucket)
    cost += float(lane_risk_cost)
    if lane_risk_reason:
        reasons.append(str(lane_risk_reason))
        if float(lane_risk_cost) >= float(infeasible_cost):
            feasible = False

    comfort_cost = _trajectory_comfort_cost(
        reference_samples=candidate.lane_center_reference,
        target_speed_mps=float(candidate.intent.target_speed_mps),
    )
    cost += float(comfort_cost)
    if float(comfort_cost) > 0.0:
        reasons.append(f"trajectory_comfort_cost:{float(comfort_cost):.2f}")

    if str(candidate.intent.decision) in {"lane_change_left", "lane_change_right"}:
        duration_cost = max(
            0.0,
            float(candidate.intent.lane_change_duration_s),
        ) * max(0.0, float(lane_change_duration_cost_per_s))
        cost += float(duration_cost)
        if duration_cost > 0.0:
            reasons.append(
                f"maneuver_duration_cost:{float(duration_cost):.2f}"
            )

    candidate.feasibility_status = "feasible" if bool(feasible) else "infeasible"
    candidate.feasibility_reason = ";".join(dict.fromkeys(reasons))
    candidate.feasibility_cost = float(cost) - float(candidate.intent.base_cost)
    candidate.total_cost = float(cost)
    return candidate


def apply_mpc_probe_result(
    *,
    candidate: CandidateReferenceResult,
    solved: bool,
    status: str,
    solve_time_ms: float,
    dynamic_cost: float = 0.0,
) -> CandidateReferenceResult:
    """Attach a side-effect-free MPC feasibility probe to a candidate."""

    if bool(solved):
        candidate.feasibility_status = "mpc_probe_solved"
        candidate.total_cost += max(0.0, float(dynamic_cost))
    else:
        candidate.feasibility_status = "mpc_probe_infeasible"
        candidate.total_cost += 10000.0
        candidate.feasibility_reason = ";".join(
            reason
            for reason in (
                str(candidate.feasibility_reason),
                "mpc_probe:" + str(status or "not_solved"),
            )
            if reason
        )
    candidate.reference_debug["candidate_mpc_probe_status"] = str(status)
    candidate.reference_debug["candidate_mpc_probe_solved"] = bool(solved)
    candidate.reference_debug["candidate_mpc_probe_solve_time_ms"] = float(
        solve_time_ms
    )
    return candidate


def mark_mpc_probe_skipped(
    candidate: CandidateReferenceResult,
) -> CandidateReferenceResult:
    candidate.feasibility_status = "mpc_probe_skipped"
    candidate.reference_debug["candidate_mpc_probe_status"] = "skipped_top_k"
    candidate.reference_debug["candidate_mpc_probe_solved"] = False
    return candidate


def select_best_candidate(candidates: Sequence[CandidateReferenceResult]) -> CandidateReferenceResult:
    rows = [candidate for candidate in list(candidates or [])]
    if not rows:
        raise ValueError("select_best_candidate requires at least one candidate")
    feasible_rows = [candidate for candidate in rows if candidate.feasible]
    pool = feasible_rows if feasible_rows else rows
    return min(
        pool,
        key=lambda candidate: (
            float(candidate.total_cost),
            0 if str(candidate.intent.decision) == "lane_follow" else 1,
            abs(int(candidate.intent.target_lane_id)),
        ),
    )


def select_candidate_with_commitment(
    candidates: Sequence[CandidateReferenceResult],
    *,
    commitment: ManeuverCommitment,
    required_decision: str = "",
    required_target_lane_id: int = 0,
) -> CandidateSelectionOutcome:
    """Select a feasible candidate while preserving route/commitment ownership.

    A route-authorized lane change is a required maneuver, not merely another
    soft-cost alternative to lane keeping.  Before execution is committed we
    therefore prefer a feasible candidate matching that requirement.  Hard
    contract, prediction, and MPC feasibility checks still retain veto power.
    """

    rows = [candidate for candidate in list(candidates or [])]
    if not rows:
        return CandidateSelectionOutcome(
            selected=None,
            status="no_candidates",
            reason="candidate_selection_empty",
        )
    normalized_required_decision = str(required_decision or "").strip().lower()
    route_requirement_active = (
        normalized_required_decision in {"lane_change_left", "lane_change_right"}
        and int(required_target_lane_id or 0) != 0
    )
    if not commitment.active and bool(route_requirement_active):
        required_rows = [
            candidate
            for candidate in rows
            if candidate.feasible
            and str(candidate.intent.decision).strip().lower()
            == normalized_required_decision
            and int(candidate.intent.target_lane_id)
            == int(required_target_lane_id)
        ]
        if required_rows:
            return CandidateSelectionOutcome(
                selected=select_best_candidate(required_rows),
                status="selected_route_required",
                reason="feasible_route_required_candidate",
            )
    if not commitment.active:
        return CandidateSelectionOutcome(
            selected=select_best_candidate(rows),
            status="selected",
            reason=(
                "route_required_candidate_infeasible_defer"
                if bool(route_requirement_active)
                else "no_active_maneuver_commitment"
            ),
        )

    committed_rows = [
        candidate
        for candidate in rows
        if commitment.accepts(
            decision=str(candidate.intent.decision),
            target_lane_id=int(candidate.intent.target_lane_id),
        )
    ]
    locked_continuations = [
        candidate
        for candidate in committed_rows
        if str(candidate.intent.name) == "committed_lane_change_continuation"
    ]
    feasible_locked_continuations = [
        candidate for candidate in locked_continuations if candidate.feasible
    ]
    if feasible_locked_continuations:
        return CandidateSelectionOutcome(
            selected=select_best_candidate(feasible_locked_continuations),
            status="selected_committed",
            reason="locked_maneuver_reference_preserved",
        )
    # Once execution starts, newly generated lane-change variants are not
    # substitutes for the locked trajectory. Switching between them resets
    # lateral progress and can reverse the geometry seen by MPC.
    if locked_continuations:
        return CandidateSelectionOutcome(
            selected=None,
            status="committed_reference_required",
            reason="locked_maneuver_reference_infeasible",
        )
    return CandidateSelectionOutcome(
        selected=None,
        status="committed_reference_required",
        reason="committed_maneuver_missing_locked_continuation",
    )


def _trajectory_comfort_cost(
    *,
    reference_samples: Sequence[Mapping[str, object]],
    target_speed_mps: float,
) -> float:
    points = []
    for sample in list(reference_samples or []):
        try:
            points.append((
                float(sample.get("x_ref_m", sample.get("x", ""))),
                float(sample.get("y_ref_m", sample.get("y", ""))),
            ))
        except Exception:
            continue
    if len(points) < 3:
        return 0.0
    headings = []
    segment_lengths = []
    for first, second in zip(points[:-1], points[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        ds = math.hypot(dx, dy)
        if ds <= 1.0e-6:
            continue
        headings.append(math.atan2(dy, dx))
        segment_lengths.append(ds)
    curvatures = []
    for index, (first, second) in enumerate(zip(headings[:-1], headings[1:])):
        delta = math.atan2(math.sin(second - first), math.cos(second - first))
        ds = max(1.0e-3, segment_lengths[min(index + 1, len(segment_lengths) - 1)])
        curvatures.append(abs(float(delta)) / float(ds))
    if not curvatures:
        return 0.0
    max_curvature = max(curvatures)
    curvature_variation = sum(
        abs(second - first)
        for first, second in zip(curvatures[:-1], curvatures[1:])
    )
    lateral_accel = float(target_speed_mps) ** 2 * float(max_curvature)
    return (
        2.0 * float(max_curvature)
        + 0.5 * float(curvature_variation)
        + 0.25 * float(lateral_accel)
    )


def summarize_candidate_results(candidates: Sequence[CandidateReferenceResult]) -> str:
    rows = [candidate.summary_row() for candidate in list(candidates or [])]
    return json.dumps(rows, sort_keys=True, default=str)


def _is_soft_turn_contract_violation(*, decision: str, contract_reason: str) -> bool:
    """Treat geometric sharpness in a turn as a slow-down cue, not a hard stop."""

    if str(decision) not in {"intersection_turn_left", "intersection_turn_right"}:
        return False
    tokens = {
        str(token).strip()
        for token in str(contract_reason or "").split(";")
        if str(token).strip()
    }
    if not tokens:
        return False
    soft_tokens = {
        "curvature_out_of_contract",
        "heading_jump_out_of_contract",
    }
    return bool(tokens.issubset(soft_tokens))


def _lane_cost(
    *,
    lane_id: int,
    current_lane_id: int,
    lane_safety_scores: Mapping[int, float],
    lane_prediction_risks: Mapping[int, Mapping[str, object]],
) -> float:
    safety = max(0.0, min(1.0, float(lane_safety_scores.get(int(lane_id), 0.0))))
    risk = dict(lane_prediction_risks.get(int(lane_id), {}) or {})
    risk_cost = 80.0 if bool(risk.get("risk", False)) else 0.0
    lane_change_cost = 5.0 if int(lane_id) != int(current_lane_id) else 0.0
    return float(10.0 * (1.0 - safety) + risk_cost + lane_change_cost)


def _finite_optional(value: object) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(number) if math.isfinite(number) else None


def _lane_change_risk_cost(
    *,
    lane_id: int,
    current_lane_id: int,
    object_snapshots: Sequence[Mapping[str, object]],
    prediction_trajectories: Mapping[str, Sequence[Mapping[str, object]]] | None,
    reference_samples: Sequence[Mapping[str, object]],
    min_object_distance_m: float,
    previous_bucket: str = "",
    hysteresis_margin_m: float = 0.0,
) -> tuple[float, str, str]:
    """Score a candidate's proximity risk, plus the discrete bucket it fell in.

    min_pred_distance is computed against THIS candidate's own reference
    (index-matched to a predicted obstacle trajectory), and two competing
    same-lane candidates (e.g. a full-speed "keep lane" vs a slowed-down
    "yield" variant) build references of different speed/extent. When their
    total costs are close, tiny per-tick differences in that distance -- not
    real obstacle motion -- can flip which discrete bucket (and therefore
    which candidate/reference) wins every other tick, which then reads to
    MPC as a discontinuous reference and shows up as steering noise.
    hysteresis_margin_m widens the boundary only on the transition toward a
    LESS severe bucket (matching previous_bucket), so escalating to a more
    severe bucket stays instant while relaxing back requires clearing a
    wider margin -- fast to react to real danger, slow to let go of it.
    """

    margin_m = max(0.0, float(hysteresis_margin_m))

    def _relaxed_bound_m(bound_m: float, sticky_bucket: str) -> float:
        return (
            float(bound_m) + margin_m
            if str(previous_bucket) == str(sticky_bucket)
            else float(bound_m)
        )

    if not reference_samples:
        return 10000.0, "empty_reference", "collision_risk"
    min_pred_distance = _min_prediction_distance_m(
        reference_samples=reference_samples,
        prediction_trajectories=prediction_trajectories,
    )
    if min_pred_distance is not None:
        same_lane = int(lane_id) == int(current_lane_id)
        near_bucket = "lead_follow" if same_lane else "collision_risk"
        near_bound_m = _relaxed_bound_m(min_object_distance_m, near_bucket)
        if float(min_pred_distance) < near_bound_m:
            if same_lane:
                # A lead vehicle predicted on the ego lane is primarily a
                # longitudinal-following constraint.  Rejecting the keep-lane
                # candidate here forces the bridge to rebuild a lateral
                # fallback reference, which makes dense same-lane traffic
                # produce steering oscillation.  Keep a strong cost so the
                # slower yield candidate wins; MPC/stop-gap logic still owns
                # the longitudinal safety constraint.
                return (
                    120.0,
                    f"candidate_prediction_lead_follow:{min_pred_distance:.2f}",
                    "lead_follow",
                )
            return (
                10000.0,
                f"candidate_prediction_collision_risk:{min_pred_distance:.2f}",
                "collision_risk",
            )
        clear_bound_m = _relaxed_bound_m(
            2.0 * float(min_object_distance_m), "near_object"
        )
        if float(min_pred_distance) < clear_bound_m:
            return (
                60.0,
                f"candidate_prediction_near_object:{min_pred_distance:.2f}",
                "near_object",
            )

    if int(lane_id) == int(current_lane_id):
        return 0.0, "", "clear"
    min_distance = _min_static_obstacle_distance_m(
        reference_samples=reference_samples,
        object_snapshots=object_snapshots,
    )
    if min_distance is None:
        return 0.0, "", "clear"
    static_near_bound_m = _relaxed_bound_m(
        min_object_distance_m, "static_collision_risk"
    )
    if float(min_distance) < static_near_bound_m:
        return (
            10000.0,
            f"candidate_reference_collision_risk:{min_distance:.2f}",
            "static_collision_risk",
        )
    static_clear_bound_m = _relaxed_bound_m(
        2.0 * float(min_object_distance_m), "static_near_object"
    )
    if float(min_distance) < static_clear_bound_m:
        return (
            30.0,
            f"candidate_reference_near_object:{min_distance:.2f}",
            "static_near_object",
        )
    return 0.0, "", "clear"


def _min_prediction_distance_m(
    *,
    reference_samples: Sequence[Mapping[str, object]],
    prediction_trajectories: Mapping[str, Sequence[Mapping[str, object]]] | None,
) -> float | None:
    if not prediction_trajectories:
        return None
    min_distance = float("inf")
    reference = list(reference_samples or [])
    for trajectory in dict(prediction_trajectories or {}).values():
        points = list(trajectory or [])
        for index, ref_sample in enumerate(reference[: max(1, min(len(reference), len(points), 12))]):
            if index >= len(points):
                break
            try:
                rx = float(ref_sample.get("x_ref_m", ref_sample.get("x", 0.0)))
                ry = float(ref_sample.get("y_ref_m", ref_sample.get("y", 0.0)))
                px = float(points[index].get("x", points[index].get("x_m", 0.0)))
                py = float(points[index].get("y", points[index].get("y_m", 0.0)))
            except Exception:
                continue
            min_distance = min(float(min_distance), math.hypot(float(rx) - float(px), float(ry) - float(py)))
    return None if not math.isfinite(float(min_distance)) else float(min_distance)


def _min_static_obstacle_distance_m(
    *,
    reference_samples: Sequence[Mapping[str, object]],
    object_snapshots: Sequence[Mapping[str, object]],
) -> float | None:
    min_distance = float("inf")
    horizon_points = list(reference_samples or [])[:8]
    for sample in horizon_points:
        try:
            sx = float(sample.get("x_ref_m", sample.get("x", 0.0)))
            sy = float(sample.get("y_ref_m", sample.get("y", 0.0)))
        except Exception:
            continue
        for obstacle in list(object_snapshots or []):
            try:
                ox = float(obstacle.get("x", obstacle.get("x_m", 0.0)))
                oy = float(obstacle.get("y", obstacle.get("y_m", 0.0)))
            except Exception:
                continue
            min_distance = min(float(min_distance), math.hypot(float(sx) - float(ox), float(sy) - float(oy)))
    return None if not math.isfinite(float(min_distance)) else float(min_distance)
