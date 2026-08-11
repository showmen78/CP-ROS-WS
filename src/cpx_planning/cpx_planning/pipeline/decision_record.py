"""Decision ownership diagnostics for the integrated CP-X pipeline.

This module is intentionally diagnostic-only.  It does not change planner
behavior; it explains which layer proposed, constrained, vetoed, or finally
executed the current tick's action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Mapping


@dataclass(frozen=True)
class DecisionVeto:
    owner: str
    reason: str
    effect: str

    def as_dict(self) -> dict[str, object]:
        return {
            "owner": str(self.owner),
            "reason": str(self.reason),
            "effect": str(self.effect),
        }


@dataclass(frozen=True)
class DecisionRecord:
    scenario_state: str = ""
    behavior_decision: str = ""
    behavior_fsm_state: str = ""
    candidate_selected_decision: str = ""
    reference_source: str = ""
    reference_stage: str = ""
    mpc_status: str = ""
    final_action: str = ""
    control_source: str = ""
    veto_chain: list[DecisionVeto] = field(default_factory=list)

    def as_debug_fields(self) -> dict[str, object]:
        veto_rows = [row.as_dict() for row in self.veto_chain]
        veto_text = "|".join(
            f"{row.owner}:{row.effect}:{row.reason}"
            for row in self.veto_chain
            if str(row.reason)
        )
        return {
            "decision_scenario_state": str(self.scenario_state),
            "decision_behavior": str(self.behavior_decision),
            "decision_behavior_fsm": str(self.behavior_fsm_state),
            "decision_candidate": str(self.candidate_selected_decision),
            "decision_reference_source": str(self.reference_source),
            "decision_reference_stage": str(self.reference_stage),
            "decision_mpc_status": str(self.mpc_status),
            "decision_final_action": str(self.final_action),
            "decision_control_source": str(self.control_source),
            "decision_veto_count": int(len(veto_rows)),
            "decision_veto_chain": json.dumps(veto_rows, sort_keys=True, default=str),
            "decision_veto_chain_text": str(veto_text),
            "decision_owner_summary": self._summary_text(veto_rows),
        }

    def _summary_text(self, veto_rows: list[Mapping[str, object]]) -> str:
        parts = [
            f"scenario={self.scenario_state or 'n/a'}",
            f"behavior={self.behavior_decision or 'n/a'}",
        ]
        if self.candidate_selected_decision:
            parts.append(f"candidate={self.candidate_selected_decision}")
        parts.append(f"reference={self.reference_source or 'n/a'}")
        parts.append(f"mpc={self.mpc_status or 'n/a'}")
        parts.append(f"final={self.final_action or 'n/a'}")
        if veto_rows:
            parts.append(
                "veto="
                + ">".join(str(row.get("owner", "")) for row in veto_rows)
            )
        return "; ".join(parts)


def build_decision_record(
    *,
    scenario_state: object = "",
    behavior_decision: object = "",
    behavior_fsm_state: object = "",
    candidate_selected_name: object = "",
    candidate_selected_decision: object = "",
    candidate_selected_status: object = "",
    candidate_selected_reason: object = "",
    candidate_pipeline_summary: object = "",
    candidate_mpc_probe_summary: object = "",
    reference_source: object = "",
    reference_stage: object = "",
    reference_fallback_reason: object = "",
    reference_lateral_guard_reason: object = "",
    reference_stabilizer_reason: object = "",
    final_reference_gate_reason: object = "",
    lane_change_authorized: object = "",
    lane_change_gate_reason: object = "",
    route_lane_change_required: object = "",
    behavior_override_reason: object = "",
    mode_transition_guard_reason: object = "",
    mpc_status: object = "",
    mpc_fallback_reason: object = "",
    control_guard_reason: object = "",
    control_buffer_reason: object = "",
    safety_supervisor_reason: object = "",
    applied_throttle: object = 0.0,
    applied_brake: object = 0.0,
    applied_steer: object = 0.0,
) -> DecisionRecord:
    vetoes: list[DecisionVeto] = []

    def add(owner: str, reason: object, effect: str) -> None:
        text = str(reason or "").strip()
        if text and not _is_routine_reason(text):
            vetoes.append(DecisionVeto(owner=str(owner), reason=text, effect=str(effect)))

    if _truthy(route_lane_change_required) and not _truthy(lane_change_authorized):
        add("LaneChangeAuthorization", lane_change_gate_reason, "block_lane_change")
    elif str(lane_change_gate_reason or "").strip():
        add("LaneChangeAuthorization", lane_change_gate_reason, "suppress_lane_change")

    add("BehaviorOverride", behavior_override_reason, "override_behavior")
    add("BehaviorModeGuard", mode_transition_guard_reason, "delay_or_block_mode_change")

    candidate_status = str(candidate_selected_status or "").strip().lower()
    if candidate_status and candidate_status not in {"feasible", "mpc_probe_solved", "baseline_no_candidates"}:
        add("CandidateEvaluator", candidate_selected_reason or candidate_selected_status, "candidate_not_clean")
    selected_name = str(candidate_selected_name or "").strip()
    for row in _candidate_rows(candidate_pipeline_summary):
        name = str(row.get("name", "") or "").strip()
        if selected_name and name == selected_name:
            continue
        status = str(row.get("feasibility_status", "") or "").strip().lower()
        reason = str(
            row.get("feasibility_reason", "")
            or row.get("contract_reason", "")
            or status
        ).strip()
        if status == "mpc_probe_infeasible":
            add(
                "CandidateMPCFeasibility",
                f"{name}:{reason}",
                "reject_candidate",
            )
        elif status == "mpc_probe_skipped":
            add(
                "CandidateMPCFeasibility",
                f"{name}:outside_top_k",
                "prune_candidate",
            )
        elif status == "infeasible":
            add(
                "CandidateEvaluator",
                f"{name}:{reason}",
                "reject_candidate",
            )
    probe_text = str(candidate_mpc_probe_summary or "").strip()
    if probe_text and probe_text not in {
        "mpc_probe_not_applicable",
        "mpc_probe_no_feasible_top_k",
    }:
        failed_probes = [
            token
            for token in probe_text.split("|")
            if token and not token.lower().endswith((":solved", ":solved inaccurate"))
        ]
        if failed_probes and not any(
            row.owner == "CandidateMPCFeasibility"
            and row.effect == "reject_candidate"
            for row in vetoes
        ):
            add(
                "CandidateMPCFeasibility",
                "|".join(failed_probes),
                "reject_candidate",
            )

    add("ReferencePipeline", reference_fallback_reason, "fallback_reference")
    add("ReferenceValidator", reference_lateral_guard_reason, "guard_reference")
    add("ReferenceStabilizer", reference_stabilizer_reason, "stabilize_or_rebuild_reference")
    add("FinalReferenceGate", final_reference_gate_reason, "reject_reference")

    normalized_mpc_status = str(mpc_status or "").strip().lower()
    if normalized_mpc_status and normalized_mpc_status not in {
        "solved",
        "solved inaccurate",
        "buffer_reuse",
    }:
        add("MPCSolver", normalized_mpc_status, "mpc_not_solved")
    add("MPCFallback", mpc_fallback_reason, "fallback_control")

    add("MPCControlBuffer", control_buffer_reason, "reuse_or_buffer_control")
    add("MPCBridgeControlGuard", control_guard_reason, "clamp_control")
    add("SafetySupervisor", safety_supervisor_reason, "final_safety_filter")

    control_source = _control_source(
        control_guard_reason=control_guard_reason,
        mpc_fallback_reason=mpc_fallback_reason,
        control_buffer_reason=control_buffer_reason,
    )
    return DecisionRecord(
        scenario_state=str(scenario_state or ""),
        behavior_decision=str(behavior_decision or ""),
        behavior_fsm_state=str(behavior_fsm_state or ""),
        candidate_selected_decision=str(candidate_selected_decision or ""),
        reference_source=str(reference_source or ""),
        reference_stage=str(reference_stage or ""),
        mpc_status=str(mpc_status or ""),
        final_action=_final_action(
            throttle=applied_throttle,
            brake=applied_brake,
            steer=applied_steer,
        ),
        control_source=str(control_source),
        veto_chain=vetoes,
    )


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y"}


def _candidate_rows(value: object) -> list[Mapping[str, object]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, Mapping)]
    text = str(value or "").strip()
    if not text:
        return []
    try:
        rows = json.loads(text)
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]


def _is_routine_reason(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    if not text:
        return True
    routine_exact = {
        "traffic_memory_green_release",
        "raw_stop",
    }
    if text in routine_exact:
        return True
    routine_prefixes = (
        "object_memory_tracks=",
    )
    return any(text.startswith(prefix) for prefix in routine_prefixes)


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _final_action(*, throttle: object, brake: object, steer: object) -> str:
    throttle_v = _to_float(throttle)
    brake_v = _to_float(brake)
    steer_v = _to_float(steer)
    if brake_v > 0.05 and brake_v >= throttle_v:
        base = "brake"
    elif throttle_v > 0.05:
        base = "throttle"
    else:
        base = "coast"
    if abs(steer_v) > 0.05:
        base += "_steer"
    return base


def _control_source(
    *,
    control_guard_reason: object,
    mpc_fallback_reason: object,
    control_buffer_reason: object,
) -> str:
    guard = str(control_guard_reason or "")
    fallback = str(mpc_fallback_reason or "")
    buffer_reason = str(control_buffer_reason or "")
    if "pid" in guard or "pid" in fallback:
        return "pid_fallback"
    if "buffer" in guard or "buffer" in buffer_reason:
        return "mpc_buffer"
    if fallback:
        return "fallback"
    return "mpc"
