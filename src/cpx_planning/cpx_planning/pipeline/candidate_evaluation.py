"""Candidate behavior evaluation for the planning pipeline.

This is the strategy layer between prediction and the rule-based FSM.  It
generates lane-level behavior candidates and scores them with safety, future
prediction risk, route alignment, and maneuver cost.  The first version is
intentionally lightweight: it evaluates candidate lanes before the final MPC
solve rather than solving a full MPC problem for every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


_DECISION_FOLLOW = "lane_follow"
_DECISION_CHANGE_LEFT = "lane_change_left"
_DECISION_CHANGE_RIGHT = "lane_change_right"


@dataclass
class BehaviorCandidate:
    name: str
    decision: str
    target_lane_id: int
    feasible: bool
    total_cost: float
    cost_terms: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


@dataclass
class CandidateEvaluationFrame:
    candidates: List[BehaviorCandidate]
    selected: BehaviorCandidate

    def summary(self) -> str:
        selected = self.selected
        feasibility = "ok" if selected.feasible else "blocked"
        return (
            f"{selected.name}->L{int(selected.target_lane_id)} "
            f"cost={float(selected.total_cost):.2f} {feasibility}"
        )


def _risk_for_lane(
    lane_prediction_risks: Optional[Mapping[int, Mapping[str, object]]],
    lane_id: int,
) -> dict:
    if lane_prediction_risks is None:
        return {}
    return dict(lane_prediction_risks.get(int(lane_id), {}))


def _lane_change_decision(
    *,
    source_lane_id: int,
    target_lane_id: int,
    available_lane_ids: Sequence[int],
) -> str:
    ordered = [int(lane_id) for lane_id in list(available_lane_ids or []) if int(lane_id) != 0]
    if int(source_lane_id) not in ordered or int(target_lane_id) not in ordered:
        return _DECISION_FOLLOW
    if int(source_lane_id) == int(target_lane_id):
        return _DECISION_FOLLOW
    source_idx = ordered.index(int(source_lane_id))
    target_idx = ordered.index(int(target_lane_id))
    return _DECISION_CHANGE_LEFT if int(target_idx) > int(source_idx) else _DECISION_CHANGE_RIGHT


def _candidate_cost(
    *,
    target_lane_id: int,
    source_lane_id: int,
    route_optimal_lane_id: Optional[int],
    lane_safety_scores: Mapping[int, float],
    lane_prediction_risks: Optional[Mapping[int, Mapping[str, object]]],
    mpc_feedback_blocked_lane_ids: Optional[Sequence[int]],
    safety_weight: float,
    prediction_risk_weight: float,
    route_deviation_weight: float,
    lane_change_weight: float,
    mpc_feedback_weight: float,
) -> Tuple[bool, Dict[str, float], str]:
    lane_score = max(0.0, min(1.0, float(lane_safety_scores.get(int(target_lane_id), 0.0))))
    prediction_risk = _risk_for_lane(lane_prediction_risks, int(target_lane_id))
    prediction_blocked = bool(prediction_risk.get("risk", False))
    mpc_feedback_blocked = (
        int(target_lane_id) != int(source_lane_id)
        and int(target_lane_id)
        in {int(lane_id) for lane_id in list(mpc_feedback_blocked_lane_ids or [])}
    )
    route_lane_id = int(route_optimal_lane_id or 0)
    route_distance = 0.0 if route_lane_id == 0 else abs(int(target_lane_id) - int(route_lane_id))
    lane_change_distance = 0.0 if int(target_lane_id) == int(source_lane_id) else abs(int(target_lane_id) - int(source_lane_id))
    cost_terms = {
        "safety_cost": float(safety_weight) * (1.0 - float(lane_score)),
        "prediction_risk_cost": float(prediction_risk_weight) if prediction_blocked else 0.0,
        "route_deviation_cost": float(route_deviation_weight) * float(route_distance),
        "lane_change_cost": float(lane_change_weight) * float(lane_change_distance),
        "mpc_feedback_cost": float(mpc_feedback_weight) if bool(mpc_feedback_blocked) else 0.0,
    }
    feasible = not bool(prediction_blocked or mpc_feedback_blocked)
    if prediction_blocked:
        reason = str(prediction_risk.get("reason", "prediction_risk") or "prediction_risk")
    elif mpc_feedback_blocked:
        reason = "recent_mpc_infeasible"
    else:
        reason = ""
    return bool(feasible), cost_terms, reason


def evaluate_behavior_candidates(
    *,
    lane_safety_scores: Mapping[int, float],
    lane_prediction_risks: Optional[Mapping[int, Mapping[str, object]]],
    ego_lane_id: int,
    selected_lane_id: int,
    available_lane_ids: Sequence[int],
    route_optimal_lane_id: Optional[int] = None,
    mode: str = "NORMAL",
    safety_weight: float = 10.0,
    prediction_risk_weight: float = 100.0,
    route_deviation_weight: float = 2.0,
    lane_change_weight: float = 1.0,
    mpc_feedback_blocked_lane_ids: Optional[Sequence[int]] = None,
    mpc_feedback_weight: float = 80.0,
) -> CandidateEvaluationFrame:
    """Evaluate lane-level behavior candidates for one planning tick."""

    del mode
    lanes = [int(lane_id) for lane_id in list(available_lane_ids or []) if int(lane_id) != 0]
    if len(lanes) == 0:
        lanes = [int(ego_lane_id)] if int(ego_lane_id) != 0 else [int(selected_lane_id)]

    source_lane_id = int(selected_lane_id or ego_lane_id or lanes[0])
    if int(source_lane_id) not in lanes:
        source_lane_id = int(ego_lane_id) if int(ego_lane_id) in lanes else int(lanes[0])

    candidate_lane_ids = {int(source_lane_id)}
    if route_optimal_lane_id is not None and int(route_optimal_lane_id) in lanes:
        candidate_lane_ids.add(int(route_optimal_lane_id))
    source_index = lanes.index(int(source_lane_id)) if int(source_lane_id) in lanes else 0
    if source_index > 0:
        candidate_lane_ids.add(int(lanes[source_index - 1]))
    if source_index < len(lanes) - 1:
        candidate_lane_ids.add(int(lanes[source_index + 1]))

    candidates: List[BehaviorCandidate] = []
    for target_lane_id in sorted(candidate_lane_ids, key=lambda lane_id: (abs(int(lane_id) - int(source_lane_id)), int(lane_id))):
        decision = _lane_change_decision(
            source_lane_id=int(source_lane_id),
            target_lane_id=int(target_lane_id),
            available_lane_ids=lanes,
        )
        name = "keep_lane" if decision == _DECISION_FOLLOW else decision
        feasible, cost_terms, reason = _candidate_cost(
            target_lane_id=int(target_lane_id),
            source_lane_id=int(source_lane_id),
            route_optimal_lane_id=route_optimal_lane_id,
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=lane_prediction_risks,
            mpc_feedback_blocked_lane_ids=mpc_feedback_blocked_lane_ids,
            safety_weight=float(safety_weight),
            prediction_risk_weight=float(prediction_risk_weight),
            route_deviation_weight=float(route_deviation_weight),
            lane_change_weight=float(lane_change_weight),
            mpc_feedback_weight=float(mpc_feedback_weight),
        )
        total_cost = float(sum(float(value) for value in cost_terms.values()))
        candidates.append(
            BehaviorCandidate(
                name=str(name),
                decision=str(decision),
                target_lane_id=int(target_lane_id),
                feasible=bool(feasible),
                total_cost=float(total_cost),
                cost_terms=dict(cost_terms),
                reason=str(reason),
            )
        )

    feasible_candidates = [candidate for candidate in candidates if candidate.feasible]
    selection_pool = feasible_candidates if len(feasible_candidates) > 0 else candidates
    selected = min(
        selection_pool,
        key=lambda candidate: (
            float(candidate.total_cost),
            0 if int(candidate.target_lane_id) == int(source_lane_id) else 1,
            abs(int(candidate.target_lane_id) - int(source_lane_id)),
        ),
    )
    return CandidateEvaluationFrame(candidates=list(candidates), selected=selected)
