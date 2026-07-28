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
from typing import Dict, Mapping, Optional, Sequence

from .reference_contract import ReferenceValidationResult


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
            "base_cost": float(self.intent.base_cost),
            "feasibility_status": str(self.feasibility_status),
            "feasibility_reason": str(self.feasibility_reason),
            "contract_reason": str(contract_reason),
            "total_cost": float(self.total_cost),
        }


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
) -> list[CandidateBehaviorIntent]:
    """Generate behavior candidates for the reference/MPC boundary."""

    intents: list[CandidateBehaviorIntent] = []
    current_lane_id = int(current_lane_id or 0)
    selected_target_lane_id = int(selected_target_lane_id or current_lane_id)
    target_speed_mps = max(0.0, float(target_speed_mps))

    def add(intent: CandidateBehaviorIntent) -> None:
        key = (str(intent.decision), int(intent.target_lane_id), round(float(intent.target_speed_mps), 2))
        existing = {
            (
                str(row.decision),
                int(row.target_lane_id),
                round(float(row.target_speed_mps), 2),
            )
            for row in intents
        }
        if key not in existing:
            intents.append(intent)

    if bool(stop_goal_active) or bool(traffic_stop_active) or str(selected_decision) in {
        "stop_at_intersection",
        "stop_sign",
        "emergency_brake",
    }:
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

    add(CandidateBehaviorIntent(
        name="keep_lane",
        decision="lane_follow",
        target_lane_id=int(current_lane_id),
        target_speed_mps=float(target_speed_mps),
        base_cost=_lane_cost(
            lane_id=int(current_lane_id),
            current_lane_id=int(current_lane_id),
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=lane_prediction_risks,
        ),
        reason="default_keep_lane",
    ))
    add(CandidateBehaviorIntent(
        name="yield_slow_down",
        decision="lane_follow",
        target_lane_id=int(current_lane_id),
        target_speed_mps=max(0.6, min(float(target_speed_mps), 0.55 * float(target_speed_mps))),
        base_cost=4.0 + _lane_cost(
            lane_id=int(current_lane_id),
            current_lane_id=int(current_lane_id),
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=lane_prediction_risks,
        ),
        reason="conservative_yield_candidate",
    ))
    if str(selected_decision) in {"lane_follow", "lane_change_left", "lane_change_right"}:
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
            decision = "lane_change_left" if target_lane_id > int(current_lane_id) else "lane_change_right"
            add(CandidateBehaviorIntent(
                name=f"route_{decision}",
                decision=str(decision),
                target_lane_id=int(target_lane_id),
                target_speed_mps=float(target_speed_mps),
                base_cost=2.0 + _lane_cost(
                    lane_id=int(target_lane_id),
                    current_lane_id=int(current_lane_id),
                    lane_safety_scores=lane_safety_scores,
                    lane_prediction_risks=lane_prediction_risks,
                ),
                reason="authorized_route_lane_change",
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
        add(CandidateBehaviorIntent(
            name=f"candidate_{decision}_{candidate_lane_id}",
            decision=str(decision),
            target_lane_id=int(candidate_lane_id),
            target_speed_mps=float(target_speed_mps),
            base_cost=3.0 + _lane_cost(
                lane_id=int(candidate_lane_id),
                current_lane_id=int(current_lane_id),
                lane_safety_scores=lane_safety_scores,
                lane_prediction_risks=lane_prediction_risks,
            ),
            reason="authorized_candidate_lane",
        ))

    return intents


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

    lane_risk_cost, lane_risk_reason = _lane_change_risk_cost(
        lane_id=int(candidate.intent.target_lane_id),
        current_lane_id=int(current_lane_id),
        object_snapshots=object_snapshots,
        prediction_trajectories=prediction_trajectories,
        reference_samples=candidate.lane_center_reference,
        min_object_distance_m=float(min_object_distance_m),
    )
    cost += float(lane_risk_cost)
    if lane_risk_reason:
        reasons.append(str(lane_risk_reason))
        if float(lane_risk_cost) >= float(infeasible_cost):
            feasible = False

    candidate.feasibility_status = "feasible" if bool(feasible) else "infeasible"
    candidate.feasibility_reason = ";".join(dict.fromkeys(reasons))
    candidate.feasibility_cost = float(cost) - float(candidate.intent.base_cost)
    candidate.total_cost = float(cost)
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


def _lane_change_risk_cost(
    *,
    lane_id: int,
    current_lane_id: int,
    object_snapshots: Sequence[Mapping[str, object]],
    prediction_trajectories: Mapping[str, Sequence[Mapping[str, object]]] | None,
    reference_samples: Sequence[Mapping[str, object]],
    min_object_distance_m: float,
) -> tuple[float, str]:
    if not reference_samples:
        return 10000.0, "empty_reference"
    min_pred_distance = _min_prediction_distance_m(
        reference_samples=reference_samples,
        prediction_trajectories=prediction_trajectories,
    )
    if min_pred_distance is not None:
        if float(min_pred_distance) < float(min_object_distance_m):
            return 10000.0, f"candidate_prediction_collision_risk:{min_pred_distance:.2f}"
        if float(min_pred_distance) < 2.0 * float(min_object_distance_m):
            return 60.0, f"candidate_prediction_near_object:{min_pred_distance:.2f}"

    if int(lane_id) == int(current_lane_id):
        return 0.0, ""
    min_distance = _min_static_obstacle_distance_m(
        reference_samples=reference_samples,
        object_snapshots=object_snapshots,
    )
    if min_distance is None:
        return 0.0, ""
    if float(min_distance) < float(min_object_distance_m):
        return 10000.0, f"candidate_reference_collision_risk:{min_distance:.2f}"
    if float(min_distance) < 2.0 * float(min_object_distance_m):
        return 30.0, f"candidate_reference_near_object:{min_distance:.2f}"
    return 0.0, ""


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
