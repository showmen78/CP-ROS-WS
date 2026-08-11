"""Mandatory final reference gate immediately before MPC."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from .reference_contract import (
    ReferenceValidationResult,
    contract_from_config,
    validate_reference_contract,
)


@dataclass(frozen=True)
class FinalReferenceGateResult:
    accepted: bool
    mode: str
    reason: str
    validation: ReferenceValidationResult

    def as_debug_fields(self) -> dict[str, object]:
        return {
            "final_reference_gate_valid": bool(self.accepted),
            "final_reference_gate_mode": str(self.mode),
            "final_reference_gate_reason": str(self.reason),
            "reference_max_curvature_1pm": float(
                self.validation.max_curvature_1pm
            ),
            "reference_contract_max_curvature_1pm": float(
                self.validation.contract_max_curvature_1pm
            ),
            "reference_curvature_margin_1pm": float(
                self.validation.curvature_margin_1pm
            ),
        }


class FinalReferenceGate:
    """Validate the exact destination/reference pair handed to MPC."""

    def __init__(self, config: Mapping[str, object]) -> None:
        self.config = dict(config or {})

    def validate(
        self,
        *,
        reference_samples: Sequence[Mapping[str, object]],
        destination_state: Sequence[float] | None,
        ego_state: Sequence[float],
        behavior_decision: str,
        behavior_fsm_state: str,
        current_lane_id: int,
        target_lane_id: int,
        stop_goal_active: bool,
        horizon_steps: int,
        default_speed_mps: float,
    ) -> FinalReferenceGateResult:
        mode = self._contract_mode(
            behavior_decision=behavior_decision,
            behavior_fsm_state=behavior_fsm_state,
            stop_goal_active=stop_goal_active,
        )
        if mode == "lane_change" and bool(
            self.config.get(
                "route_tracking_lane_change_direct_target_tracking_enabled",
                False,
            )
        ):
            # Under direct target-lane tracking, MPC is handed the target
            # lane's own (unblended) centerline -- its first sample
            # legitimately sits close to a full lane width from ego at lock
            # time, which the standard "lane_change" mode's tighter limit
            # (sized for an already-ramping blend) would veto.
            mode = "lane_change_direct"
        expected_lane_id = (
            int(target_lane_id)
            if mode in ("lane_change", "lane_change_direct") and int(target_lane_id) != 0
            else int(current_lane_id)
        )
        contract = contract_from_config(
            mode=mode,
            expected_lane_id=expected_lane_id,
            horizon_steps=int(horizon_steps),
            config=self.config,
            default_speed_mps=max(0.1, float(default_speed_mps)),
        )
        validation = validate_reference_contract(
            reference_samples=reference_samples,
            destination_state=destination_state,
            ego_state=ego_state,
            contract=contract,
            check_destination_body_lateral=self.check_destination_body_lateral(
                mode=mode,
                reference_samples=reference_samples,
            ),
        )
        return FinalReferenceGateResult(
            accepted=bool(validation.valid),
            mode=str(mode),
            reason="" if validation.valid else validation.reason(),
            validation=validation,
        )

    def check_destination_body_lateral(
        self,
        *,
        mode: str,
        reference_samples: Sequence[Mapping[str, object]],
    ) -> bool:
        """Use ego-body destination lateral only for approximately straight paths."""

        if str(mode) in {"stop", "emergency_stop"}:
            return True
        if str(mode) != "lane_follow":
            return False
        max_heading_change_rad = max(
            0.0,
            float(
                self.config.get(
                    "reference_contract_lane_follow_body_lateral_max_heading_change_rad",
                    0.20,
                )
            ),
        )
        headings = self._reference_headings(reference_samples)
        if len(headings) < 2:
            return True
        accumulated_change_rad = sum(
            abs(math.atan2(
                math.sin(float(second) - float(first)),
                math.cos(float(second) - float(first)),
            ))
            for first, second in zip(headings[:-1], headings[1:])
        )
        return float(accumulated_change_rad) <= float(max_heading_change_rad)

    @staticmethod
    def _reference_headings(
        reference_samples: Sequence[Mapping[str, object]],
    ) -> list[float]:
        samples = list(reference_samples or [])
        explicit = []
        for sample in samples:
            try:
                heading = float(sample.get("heading_rad", ""))
            except Exception:
                explicit = []
                break
            if not math.isfinite(heading):
                explicit = []
                break
            explicit.append(heading)
        if len(explicit) >= 2:
            return explicit

        headings = []
        points = []
        for sample in samples:
            try:
                points.append((
                    float(sample.get("x_ref_m", sample.get("x", ""))),
                    float(sample.get("y_ref_m", sample.get("y", ""))),
                ))
            except Exception:
                continue
        for first, second in zip(points[:-1], points[1:]):
            dx_m = float(second[0]) - float(first[0])
            dy_m = float(second[1]) - float(first[1])
            if math.hypot(dx_m, dy_m) > 1.0e-6:
                headings.append(math.atan2(dy_m, dx_m))
        return headings

    @staticmethod
    def _contract_mode(
        *,
        behavior_decision: str,
        behavior_fsm_state: str,
        stop_goal_active: bool,
    ) -> str:
        behavior = str(behavior_decision or "").strip().lower()
        fsm = str(behavior_fsm_state or "").strip().upper()
        if behavior == "emergency_brake":
            return "emergency_stop"
        if bool(stop_goal_active) or behavior in {
            "stop_at_intersection",
            "stop_sign",
        }:
            return "stop"
        if (
            behavior in {"lane_change_left", "lane_change_right"}
            or fsm.startswith("EXECUTE_LANE_CHANGE")
            or fsm == "TARGET_LANE_STABILIZATION"
        ):
            return "lane_change"
        if behavior in {
            "intersection_turn_left",
            "intersection_turn_right",
            "route_recovery",
        } or fsm in {"INTERSECTION_TURN", "ROUTE_TRACKING", "ROUTE_RECOVERY"}:
            return "intersection_turn"
        return "lane_follow"
