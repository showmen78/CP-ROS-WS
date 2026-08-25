"""Typed contracts between planning stages."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence
LANE_CHANGE_DECISIONS = {"lane_change_left", "lane_change_right"}


@dataclass(frozen=True)
class ManeuverCommitment:
    """Behavior-to-candidate contract for an executing maneuver."""

    state: str = "IDLE"
    decision: str = ""
    source_lane_id: int = 0
    target_lane_id: int = 0
    progress: float = 0.0
    reference_locked: bool = False

    @property
    def active(self) -> bool:
        return bool(
            str(self.state).upper() in {"COMMITTED", "STABILIZING"}
            and str(self.decision) in LANE_CHANGE_DECISIONS
            and int(self.target_lane_id) != 0
            # A self-referencing commit (target == source) has no real lateral
            # offset to execute and can only ever fail its own reference
            # contract -- treat it as never having been active instead of
            # locking in an unwinnable maneuver.
            and int(self.target_lane_id) != int(self.source_lane_id)
            and bool(self.reference_locked)
        )

    def accepts(self, *, decision: str, target_lane_id: int) -> bool:
        if not self.active:
            return True
        return bool(
            str(decision) == str(self.decision)
            and int(target_lane_id) == int(self.target_lane_id)
        )

    def as_debug_fields(self) -> dict[str, object]:
        return {
            "maneuver_commitment_state": str(self.state),
            "maneuver_commitment_decision": str(self.decision),
            "maneuver_commitment_source_lane_id": int(self.source_lane_id),
            "maneuver_commitment_target_lane_id": int(self.target_lane_id),
            "maneuver_commitment_progress": float(self.progress),
            "maneuver_commitment_reference_locked": bool(self.reference_locked),
            "maneuver_commitment_active": bool(self.active),
        }


@dataclass(frozen=True)
class LaneChangeCompletion:
    """Geometric completion contract for a committed lane change."""

    complete: bool
    stable_frames: int
    progress: float
    target_lateral_error_m: float
    target_heading_error_rad: float
    target_lane_matches: bool
    footprint_clearance_m: float
    reason: str


def evaluate_lane_change_completion(
    *,
    reference_samples: Sequence[Mapping[str, object]],
    ego_x_m: float,
    ego_y_m: float,
    ego_heading_rad: float,
    progress: float,
    previous_stable_frames: int,
    target_lane_matches: bool = True,
    footprint_clearance_m: float = float("inf"),
    min_footprint_clearance_m: float = 0.0,
    min_progress: float = 0.92,
    max_lateral_error_m: float = 0.35,
    max_heading_error_rad: float = math.radians(8.0),
    required_stable_frames: int = 5,
) -> LaneChangeCompletion:
    """Require convergence to the locked target geometry.

    ``target_lane_matches`` is semantic evidence/debug information only.  A
    map matcher may change ID before the vehicle has converged, or re-anchor
    to a different ID namespace across a road boundary.  The committed
    reference, progress, footprint clearance, lateral error and heading are
    the authoritative execution/completion signals.
    """

    terminal_samples = [
        sample
        for sample in list(reference_samples or [])
        if float(sample.get("lane_change_progress", 0.0) or 0.0) >= 0.90
    ]
    if not terminal_samples:
        terminal_samples = list(reference_samples or [])[-10:]
    if not terminal_samples:
        return LaneChangeCompletion(
            complete=False,
            stable_frames=0,
            progress=float(progress),
            target_lateral_error_m=float("inf"),
            target_heading_error_rad=float("inf"),
            target_lane_matches=bool(target_lane_matches),
            footprint_clearance_m=float(footprint_clearance_m),
            reason="lane_change_completion_missing_target_geometry",
        )

    nearest = min(
        terminal_samples,
        key=lambda sample: (
            float(sample.get("x_ref_m", sample.get("x", 0.0))) - float(ego_x_m)
        ) ** 2
        + (
            float(sample.get("y_ref_m", sample.get("y", 0.0))) - float(ego_y_m)
        ) ** 2,
    )
    target_x_m = float(nearest.get("x_ref_m", nearest.get("x", ego_x_m)))
    target_y_m = float(nearest.get("y_ref_m", nearest.get("y", ego_y_m)))
    target_heading_rad = float(nearest.get("heading_rad", ego_heading_rad))
    dx_m = float(ego_x_m) - target_x_m
    dy_m = float(ego_y_m) - target_y_m
    lateral_error_m = (
        -math.sin(target_heading_rad) * dx_m
        + math.cos(target_heading_rad) * dy_m
    )
    heading_error_rad = _wrap_angle(float(ego_heading_rad) - target_heading_rad)
    converged = bool(
        float(progress) >= float(min_progress)
        and abs(float(lateral_error_m)) <= float(max_lateral_error_m)
        and abs(float(heading_error_rad)) <= float(max_heading_error_rad)
        and float(footprint_clearance_m)
        >= float(min_footprint_clearance_m)
    )
    stable_frames = int(previous_stable_frames) + 1 if converged else 0
    required_frames = max(1, int(required_stable_frames))
    complete = bool(stable_frames >= required_frames)
    reason = (
        (
            "lane_change_geometrically_complete"
            if bool(target_lane_matches)
            else "lane_change_geometrically_complete_lane_id_mismatch"
        )
        if complete
        else "lane_change_completion_converging"
        if converged
        else "lane_change_completion_not_converged"
    )
    return LaneChangeCompletion(
        complete=bool(complete),
        stable_frames=int(stable_frames),
        progress=float(progress),
        target_lateral_error_m=float(lateral_error_m),
        target_heading_error_rad=float(heading_error_rad),
        target_lane_matches=bool(target_lane_matches),
        footprint_clearance_m=float(footprint_clearance_m),
        reason=str(reason),
    )


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class MPCEntryAuthorization:
    """Reference-to-MPC contract evaluated once per planning tick."""

    allowed: bool
    status: str
    reason: str = ""

    def as_debug_fields(self) -> dict[str, object]:
        return {
            "mpc_entry_allowed": bool(self.allowed),
            "mpc_entry_status": str(self.status),
            "mpc_entry_reason": str(self.reason),
        }


def authorize_mpc_entry(
    *,
    candidate_status: object,
    candidate_name: object,
    candidate_reason: object,
    final_reference_accepted: bool,
    final_reference_reason: object,
    behavior_decision: object,
) -> MPCEntryAuthorization:
    """Authorize MPC only from structured upstream stage outcomes.

    Human-readable reason strings are retained for diagnostics, but never
    parsed to decide whether an invalid reference may reach the solver.
    """

    behavior = str(behavior_decision or "").strip().lower()
    selected_status = str(candidate_status or "").strip().lower()
    selected_name = str(candidate_name or "candidate").strip() or "candidate"
    if behavior == "emergency_brake":
        return MPCEntryAuthorization(
            allowed=False,
            status="direct_emergency_control",
            reason=f"{selected_name}:emergency_brake_direct_control",
        )
    if not bool(final_reference_accepted):
        reason = str(final_reference_reason or "reference_contract_rejected")
        return MPCEntryAuthorization(
            allowed=False,
            status="reference_rejected",
            reason=f"{selected_name}:final_reference_gate:{reason}",
        )
    if selected_status in {
        "infeasible",
        "infeasible_hard_stop",
        "mpc_probe_infeasible",
        "committed_reference_required",
        "no_candidates",
    }:
        reason = str(candidate_reason or selected_status)
        return MPCEntryAuthorization(
            allowed=False,
            status="candidate_rejected",
            reason=f"{selected_name}:{reason}",
        )
    return MPCEntryAuthorization(
        allowed=True,
        status="authorized",
        reason="candidate_and_reference_accepted",
    )
