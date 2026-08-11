"""Single orchestration boundary between reference generation and MPC."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .reference_contract import (
    ReferenceValidationResult,
    contract_from_config,
    validate_reference_contract,
)
from .reference_gate import FinalReferenceGate, FinalReferenceGateResult
from .reference_generator import ReferenceGenerator


@dataclass(frozen=True)
class ReferencePipelineRequest:
    destination_state: Sequence[float]
    reference_samples: Sequence[Mapping[str, object]]
    current_state: Sequence[float]
    ego_location: Any
    ego_yaw_rad: float
    ego_speed_mps: float
    target_speed_mps: float
    behavior_decision: str
    behavior_fsm_state: str
    current_lane_id: int
    target_lane_id: int
    stop_goal_active: bool
    stop_target: Mapping[str, object] | None = None
    route_points: Sequence[Sequence[float]] = ()


@dataclass(frozen=True)
class ConditionedReference:
    destination_state: list[float]
    reference_samples: list[dict[str, object]]
    mode: str
    reason: str
    validation: ReferenceValidationResult


@dataclass(frozen=True)
class ReferencePipelineResult:
    destination_state: list[float]
    reference_samples: list[dict[str, object]]
    mode: str
    conditioning_reason: str
    gate: FinalReferenceGateResult

    @property
    def accepted(self) -> bool:
        return bool(self.gate.accepted)

    def as_debug_fields(self) -> dict[str, object]:
        fields = {
            "reference_pipeline_conditioning_reason": str(
                self.conditioning_reason
            ),
            "reference_pipeline_mode": str(self.mode),
        }
        fields.update(self.gate.as_debug_fields())
        return fields


class ReferencePipeline:
    """Own normalization, one mode-specific recovery, and the final gate.

    ReferenceGenerator owns geometry. This class owns the lifecycle of that
    geometry after generation. It never changes behavior intent.
    """

    def __init__(
        self,
        *,
        config: Mapping[str, object],
        generator: ReferenceGenerator,
        final_gate: FinalReferenceGate,
        horizon_steps: int,
        dt_s: float,
        default_speed_mps: float,
    ) -> None:
        self.config = dict(config or {})
        self.generator = generator
        self.final_gate = final_gate
        self.horizon_steps = int(horizon_steps)
        self.dt_s = float(dt_s)
        self.default_speed_mps = float(default_speed_mps)
        self._lane_follow_recovery_active = False
        self._lane_follow_valid_streak = 0

    def condition(self, request: ReferencePipelineRequest) -> ConditionedReference:
        mode = self._mode(request)
        boundary_recovery = str(
            request.behavior_fsm_state or ""
        ).upper().startswith("BOUNDARY_RECOVERY")
        destination = list(request.destination_state or [])
        reference = [
            dict(sample) for sample in list(request.reference_samples or [])
        ]
        reasons: list[str] = []

        if mode == "emergency_stop":
            generated = self.generator.emergency_stop_reference(
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
                current_lane_id=int(request.current_lane_id),
                horizon_steps=int(self.horizon_steps),
                step_distance_m=max(0.5, float(self.dt_s) * 0.8),
            )
            reference = generated.samples
            destination = generated.destination_state
            reasons.append(generated.reason)
        elif mode == "stop":
            generated = self.generator.stop_reference(
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
                current_state=request.current_state,
                current_lane_id=int(request.current_lane_id),
                stop_target=request.stop_target,
                fallback_destination_state=destination,
                ego_speed_mps=float(request.ego_speed_mps),
            )
            reference = generated.samples
            destination = generated.destination_state
            reasons.append(generated.reason)

        reference, clean_reasons = self._clean_reference(
            reference=reference,
            request=request,
            mode=mode,
        )
        reasons.extend(clean_reasons)
        if mode in {"lane_follow", "lane_change", "intersection_turn"}:
            curvature_limit_1pm = self._contract_curvature_limit(
                request=request,
                mode=mode,
            )
            reference, curvature_reason = (
                self.generator.curvature_feasible_samples(
                    reference_samples=reference,
                    ego_location=request.ego_location,
                    ego_heading_rad=float(request.ego_yaw_rad),
                    max_curvature_1pm=float(curvature_limit_1pm),
                    mode=str(mode),
                )
            )
            if curvature_reason:
                reasons.append(curvature_reason)
                destination = self._aligned_destination(
                    mode=mode,
                    destination=destination,
                    reference=reference,
                    current_state=request.current_state,
                )
        turn_footprint_validation = None
        boundary_recovery_validation = None
        if mode == "intersection_turn" and bool(boundary_recovery):
            boundary_recovery_validation = (
                self.generator.validate_boundary_recovery_progress(
                    reference_samples=reference,
                    ego_half_width_m=max(
                        0.1,
                        float(
                            self.config.get(
                                "reference_vehicle_half_width_m",
                                1.0,
                            )
                        ),
                    ),
                    ego_half_length_m=max(
                        0.1,
                        float(
                            self.config.get(
                                "reference_vehicle_half_length_m",
                                2.4,
                            )
                        ),
                    ),
                    safety_margin_m=max(
                        0.0,
                        float(
                            self.config.get(
                                "reference_contract_turn_boundary_margin_m",
                                0.15,
                            )
                        ),
                    ),
                    max_worsening_m=max(
                        0.0,
                        float(
                            self.config.get(
                                "boundary_recovery_max_worsening_m",
                                0.08,
                            )
                        ),
                    ),
                    min_terminal_improvement_m=max(
                        0.0,
                        float(
                            self.config.get(
                                "boundary_recovery_min_terminal_improvement_m",
                                0.03,
                            )
                        ),
                    ),
                )
            )
            reasons.append(str(boundary_recovery_validation.reason))
        elif mode == "intersection_turn":
            (
                reference,
                turn_footprint_validation,
                footprint_reason,
            ) = self._condition_turn_swept_footprint(
                request=request,
                reference=reference,
            )
            if (
                "corrected" in str(footprint_reason)
                or not bool(turn_footprint_validation.valid)
            ):
                reasons.append(str(footprint_reason))
                destination = self._aligned_destination(
                    mode=mode,
                    destination=destination,
                    reference=reference,
                    current_state=request.current_state,
                )
        validation = self._validate(
            request=request,
            mode=mode,
            destination=destination,
            reference=reference,
        )
        if (
            turn_footprint_validation is not None
            and not bool(turn_footprint_validation.valid)
        ):
            validation.valid = False
            validation.violations.append(
                str(turn_footprint_validation.reason)
            )
        if (
            boundary_recovery_validation is not None
            and not bool(boundary_recovery_validation.valid)
        ):
            validation.valid = False
            validation.violations.append(
                str(boundary_recovery_validation.reason)
            )
        if not validation.valid:
            reasons.append("contract_violation:" + validation.reason())
            # Even when already inside the boundary-recovery path, still try
            # _recover_once once: for intersection_turn it builds the
            # reference from route_aligned_samples (a different geometry
            # source than whatever produced the out-of-corridor reference in
            # the first place), and it does not recurse back into boundary
            # recovery, so this cannot loop forever. Skipping it here used to
            # mean a failed boundary-recovery validation had no fallback at
            # all -- validate_boundary_recovery_progress's worsening check
            # (reference_generator.py) then vetoes every subsequent tick's
            # attempt identically, and the vehicle stops permanently with no
            # path back to a valid reference.
            recovered_reference, recovered_destination, recovery_reason = (
                self._recover_once(
                    request=request,
                    mode=mode,
                    destination=destination,
                )
            )
            if recovered_reference:
                reference = recovered_reference
                destination = recovered_destination
                reasons.append(recovery_reason)
                if mode in {"lane_follow", "lane_change", "intersection_turn"}:
                    reference, recovery_curvature_reason = (
                        self.generator.curvature_feasible_samples(
                            reference_samples=reference,
                            ego_location=request.ego_location,
                            ego_heading_rad=float(request.ego_yaw_rad),
                            max_curvature_1pm=float(
                                self._contract_curvature_limit(
                                    request=request,
                                    mode=mode,
                                )
                            ),
                            mode=f"{str(mode)}_recovery",
                        )
                    )
                    if recovery_curvature_reason:
                        reasons.append(recovery_curvature_reason)
                        destination = self._aligned_destination(
                            mode=mode,
                            destination=destination,
                            reference=reference,
                            current_state=request.current_state,
                        )
                turn_footprint_validation = None
                if mode == "intersection_turn":
                    (
                        reference,
                        turn_footprint_validation,
                        footprint_reason,
                    ) = self._condition_turn_swept_footprint(
                        request=request,
                        reference=reference,
                    )
                    if (
                        "corrected" in str(footprint_reason)
                        or not bool(turn_footprint_validation.valid)
                    ):
                        reasons.append(str(footprint_reason))
                        destination = self._aligned_destination(
                            mode=mode,
                            destination=destination,
                            reference=reference,
                            current_state=request.current_state,
                        )
                validation = self._validate(
                    request=request,
                    mode=mode,
                    destination=destination,
                    reference=reference,
                    lane_recovery=mode == "lane_follow",
                )
                if (
                    turn_footprint_validation is not None
                    and not bool(turn_footprint_validation.valid)
                ):
                    validation.valid = False
                    validation.violations.append(
                        str(turn_footprint_validation.reason)
                    )
                if not validation.valid:
                    reasons.append(
                        "recovery_contract_violation:" + validation.reason()
                    )

        if bool(boundary_recovery):
            anchor_violations = self._boundary_recovery_anchor_violations(
                request=request,
                reference=reference,
            )
            if anchor_violations:
                validation.valid = False
                validation.violations.extend(anchor_violations)
                validation.violations = list(
                    dict.fromkeys(validation.violations)
                )
                reasons.append(
                    "boundary_recovery_anchor_violation:"
                    + ",".join(anchor_violations)
                )

        if mode in {"stop", "emergency_stop"}:
            if len(destination) >= 3:
                destination[2] = 0.0
            for sample in reference:
                sample["v_ref_mps"] = 0.0
                sample["speed_ref_mps"] = 0.0
                sample["speed_mps"] = 0.0

        return ConditionedReference(
            destination_state=list(destination),
            reference_samples=[dict(sample) for sample in reference],
            mode=str(mode),
            reason=self._compact_reasons(reasons),
            validation=validation,
        )

    def finalize(self, request: ReferencePipelineRequest) -> ReferencePipelineResult:
        conditioned = self.condition(request)
        conditioned = self._apply_final_continuity_policy(
            request=request,
            conditioned=conditioned,
        )
        gate = self.final_gate.validate(
            reference_samples=conditioned.reference_samples,
            destination_state=conditioned.destination_state,
            ego_state=request.current_state,
            behavior_decision=request.behavior_decision,
            behavior_fsm_state=request.behavior_fsm_state,
            current_lane_id=int(request.current_lane_id),
            target_lane_id=int(request.target_lane_id),
            stop_goal_active=bool(request.stop_goal_active),
            horizon_steps=int(self.horizon_steps),
            default_speed_mps=max(
                0.1,
                float(self.default_speed_mps),
                float(request.target_speed_mps),
            ),
        )
        if not bool(conditioned.validation.valid):
            gate = FinalReferenceGateResult(
                accepted=False,
                mode=str(conditioned.mode),
                reason=str(conditioned.validation.reason()),
                validation=conditioned.validation,
            )
        return ReferencePipelineResult(
            destination_state=conditioned.destination_state,
            reference_samples=conditioned.reference_samples,
            mode=conditioned.mode,
            conditioning_reason=conditioned.reason,
            gate=gate,
        )

    def _condition_turn_swept_footprint(
        self,
        *,
        request: ReferencePipelineRequest,
        reference: Sequence[Mapping[str, object]],
    ):
        """Validate and, once, correct the full turn footprint corridor."""

        step_distance_m = max(
            0.1,
            float(self.dt_s)
            * max(0.5, float(request.target_speed_mps)),
        )
        return self.generator.ensure_turn_swept_footprint(
            reference_samples=reference,
            horizon_steps=int(self.horizon_steps),
            step_distance_m=float(step_distance_m),
            fallback_heading_rad=float(request.ego_yaw_rad),
            ego_half_width_m=max(
                0.1,
                float(
                    self.config.get(
                        "reference_vehicle_half_width_m",
                        self.config.get("metrics_ego_half_width_m", 1.0),
                    )
                ),
            ),
            ego_half_length_m=max(
                0.1,
                float(
                    self.config.get(
                        "reference_vehicle_half_length_m",
                        2.40,
                    )
                ),
            ),
            safety_margin_m=max(
                0.0,
                float(
                    self.config.get(
                        "reference_contract_turn_boundary_margin_m",
                        0.15,
                    )
                ),
            ),
            max_violations=max(
                0,
                int(
                    self.config.get(
                        "reference_contract_turn_max_boundary_failures",
                        1,
                    )
                ),
            ),
        )

    def _apply_final_continuity_policy(
        self,
        *,
        request: ReferencePipelineRequest,
        conditioned: ConditionedReference,
    ) -> ConditionedReference:
        """Keep lane-follow recovery geometry stable across short valid gaps.

        Candidate conditioning remains side-effect free. Only the one final
        reference selected for MPC updates this lifecycle state.
        """

        if str(conditioned.mode) != "lane_follow":
            self._lane_follow_recovery_active = False
            self._lane_follow_valid_streak = 0
            return conditioned

        recovery_selected = "lane_recovery_reference" in str(
            conditioned.reason
        )
        if recovery_selected:
            self._lane_follow_recovery_active = True
            self._lane_follow_valid_streak = 0
            return conditioned
        if not bool(self._lane_follow_recovery_active):
            return conditioned

        self._lane_follow_valid_streak += 1
        release_frames = max(
            1,
            int(
                self.config.get(
                    "reference_pipeline_lane_follow_recovery_release_frames",
                    5,
                )
            ),
        )
        if int(self._lane_follow_valid_streak) >= int(release_frames):
            self._lane_follow_recovery_active = False
            self._lane_follow_valid_streak = 0
            return conditioned

        reference, destination, _ = self._recover_once(
            request=request,
            mode="lane_follow",
            destination=conditioned.destination_state,
        )
        if not reference:
            return conditioned
        validation = self._validate(
            request=request,
            mode="lane_follow",
            destination=destination,
            reference=reference,
            lane_recovery=True,
        )
        if not validation.valid:
            return conditioned
        return ConditionedReference(
            destination_state=list(destination),
            reference_samples=[dict(sample) for sample in reference],
            mode="lane_follow",
            reason=self._compact_reasons((
                conditioned.reason,
                "lane_recovery_reference_hysteresis",
            )),
            validation=validation,
        )

    def _clean_reference(
        self,
        *,
        reference: Sequence[Mapping[str, object]],
        request: ReferencePipelineRequest,
        mode: str,
    ) -> tuple[list[dict[str, object]], list[str]]:
        reasons: list[str] = []
        cleaned: list[dict[str, object]] = []
        previous_xy = None
        min_forward_defaults = {
            "lane_follow": 0.5,
            "stop": 0.2,
            "lane_change": 0.2,
            "intersection_turn": 0.2,
            "emergency_stop": 0.1,
        }
        min_forward_m = float(
            self.config.get(
                "reference_contract_" + str(mode) + "_min_first_forward_m",
                min_forward_defaults.get(mode, 0.5),
            )
        )
        min_spacing_m = max(
            1.0e-3,
            float(self.config.get("full_reference_stabilizer_min_spacing_m", 0.35)),
        )
        if mode in {"lane_change", "intersection_turn"}:
            key = (
                "full_lane_change_reference_duplicate_spacing_m"
                if mode == "lane_change"
                else "full_turn_reference_duplicate_spacing_m"
            )
            min_spacing_m = min(
                min_spacing_m,
                max(1.0e-3, float(self.config.get(key, 0.05))),
            )

        for sample in list(reference or []):
            try:
                x_m = float(sample.get("x_ref_m", sample.get("x", "")))
                y_m = float(sample.get("y_ref_m", sample.get("y", "")))
            except Exception:
                reasons.append("drop_non_finite_sample")
                continue
            if not math.isfinite(x_m) or not math.isfinite(y_m):
                reasons.append("drop_non_finite_sample")
                continue
            forward_m = self._forward_m(
                current_state=request.current_state,
                x_m=x_m,
                y_m=y_m,
            )
            if forward_m < min_forward_m:
                reasons.append("drop_behind_sample")
                continue
            if (
                previous_xy is not None
                and math.hypot(x_m - previous_xy[0], y_m - previous_xy[1])
                < min_spacing_m
            ):
                reasons.append("drop_duplicate_sample")
                continue
            row = dict(sample)
            row.update({"x_ref_m": x_m, "y_ref_m": y_m, "x": x_m, "y": y_m})
            cleaned.append(row)
            previous_xy = (x_m, y_m)
        if len(cleaned) < 2:
            reasons.append("too_few_forward_samples")
        return cleaned, reasons

    def _recover_once(
        self,
        *,
        request: ReferencePipelineRequest,
        mode: str,
        destination: Sequence[float],
    ) -> tuple[list[dict[str, object]], list[float], str]:
        step_distance_m = max(
            0.5,
            float(self.dt_s)
            * max(
                1.0,
                min(float(request.target_speed_mps), self.default_speed_mps),
            ),
        )
        if mode in {"stop", "emergency_stop"}:
            generated = self.generator.emergency_stop_reference(
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
                current_lane_id=int(request.current_lane_id),
                horizon_steps=int(self.horizon_steps),
                step_distance_m=float(step_distance_m),
            )
            return (
                generated.samples,
                generated.destination_state,
                "emergency_stop_recovery",
            )
        if mode == "lane_follow":
            reference = self.generator.lane_recovery_samples(
                ego_location=request.ego_location,
                ego_yaw_rad=float(request.ego_yaw_rad),
                start_waypoint=self.generator.map_waypoint(request.ego_location),
                current_lane_id=int(request.current_lane_id),
                horizon_steps=int(self.horizon_steps),
                step_distance_m=float(step_distance_m),
                route_points=request.route_points,
            )
            aligned = self._aligned_destination(
                mode=mode,
                destination=destination,
                reference=reference,
                current_state=request.current_state,
            )
            return reference, aligned, "lane_recovery_reference"
        if mode == "intersection_turn":
            reference = self.generator.route_aligned_samples(
                ego_location=request.ego_location,
                ego_heading_rad=float(request.ego_yaw_rad),
                current_lane_id=int(request.current_lane_id),
                horizon_steps=int(self.horizon_steps),
                step_distance_m=float(step_distance_m),
                route_points=request.route_points,
            )
            aligned = self._aligned_destination(
                mode=mode,
                destination=destination,
                reference=reference,
                current_state=request.current_state,
            )
            return reference, aligned, "route_aligned_turn_recovery"
        # A committed lane change may only continue its locked trajectory or
        # be rejected by the caller. It must never be rebuilt as lane follow.
        return [], list(destination), "lane_change_recovery_forbidden"

    def _aligned_destination(
        self,
        *,
        mode: str,
        destination: Sequence[float],
        reference: Sequence[Mapping[str, object]],
        current_state: Sequence[float],
    ) -> list[float]:
        if not reference:
            return list(destination)
        from cpx_planning.behavior_planner.reference_pipeline import (
            lane_center_destination_from_reference,
            lane_center_destination_from_reference_arc_length,
        )

        if mode == "intersection_turn":
            aligned = lane_center_destination_from_reference_arc_length(
                destination_state=destination,
                lane_center_reference=reference,
                target_arc_length_m=float(
                    self.config.get("carla_waypoint_turn_destination_arc_m", 4.5)
                ),
            )
        else:
            aligned = lane_center_destination_from_reference(
                destination_state=destination,
                lane_center_reference=reference,
                ego_state=current_state,
                target_forward_m=float(
                    self.config.get(
                        "full_lane_follow_guard_destination_forward_m", 8.0
                    )
                ),
            )
        return list(aligned if aligned is not None else destination)

    def _validate(
        self,
        *,
        request: ReferencePipelineRequest,
        mode: str,
        destination: Sequence[float],
        reference: Sequence[Mapping[str, object]],
        lane_recovery: bool = False,
    ) -> ReferenceValidationResult:
        contract_mode = mode
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
            # (sized for an already-ramping blend) would veto. Only the
            # contract-checking mode widens here; `mode` itself stays
            # "lane_change" for the reference-shaping branches elsewhere in
            # condition() that key off that exact string.
            contract_mode = "lane_change_direct"
        expected_lane_id = (
            int(request.target_lane_id)
            if contract_mode in ("lane_change", "lane_change_direct")
            and int(request.target_lane_id) != 0
            else int(request.current_lane_id)
        )
        contract = contract_from_config(
            mode=contract_mode,
            expected_lane_id=expected_lane_id,
            horizon_steps=int(self.horizon_steps),
            config=self.config,
            default_speed_mps=max(
                0.1,
                float(self.default_speed_mps),
                float(request.target_speed_mps),
            ),
        )
        return validate_reference_contract(
            reference_samples=reference,
            destination_state=destination,
            ego_state=request.current_state,
            contract=contract,
            check_destination_body_lateral=(
                self.final_gate.check_destination_body_lateral(
                    mode=contract_mode,
                    reference_samples=reference,
                )
                and not bool(lane_recovery)
            ),
        )

    def _contract_curvature_limit(
        self,
        *,
        request: ReferencePipelineRequest,
        mode: str,
    ) -> float:
        contract = contract_from_config(
            mode=mode,
            expected_lane_id=int(request.current_lane_id),
            horizon_steps=int(self.horizon_steps),
            config=self.config,
            default_speed_mps=max(
                0.1,
                float(self.default_speed_mps),
                float(request.target_speed_mps),
            ),
        )
        return float(contract.max_curvature_1pm)

    def _boundary_recovery_anchor_violations(
        self,
        *,
        request: ReferencePipelineRequest,
        reference: Sequence[Mapping[str, object]],
    ) -> list[str]:
        samples = list(reference or [])
        if len(samples) < 2:
            return ["boundary_recovery_missing_anchor_samples"]
        ego_x_m = float(request.current_state[0])
        ego_y_m = float(request.current_state[1])
        ego_heading_rad = float(request.current_state[3])
        try:
            first_x_m = float(
                samples[0].get("x_ref_m", samples[0].get("x", ""))
            )
            first_y_m = float(
                samples[0].get("y_ref_m", samples[0].get("y", ""))
            )
            second_x_m = float(
                samples[1].get("x_ref_m", samples[1].get("x", ""))
            )
            second_y_m = float(
                samples[1].get("y_ref_m", samples[1].get("y", ""))
            )
            first_heading_rad = float(
                samples[0].get(
                    "heading_rad",
                    math.atan2(
                        second_y_m - first_y_m,
                        second_x_m - first_x_m,
                    ),
                )
            )
        except (TypeError, ValueError):
            return ["boundary_recovery_non_finite_anchor"]
        ego_to_first_heading_rad = math.atan2(
            float(first_y_m) - float(ego_y_m),
            float(first_x_m) - float(ego_x_m),
        )
        first_segment_heading_rad = math.atan2(
            float(second_y_m) - float(first_y_m),
            float(second_x_m) - float(first_x_m),
        )
        max_error_rad = math.radians(
            max(
                1.0,
                float(
                    self.config.get(
                        "boundary_recovery_max_anchor_heading_error_deg",
                        12.0,
                    )
                ),
            )
        )
        violations = []
        for name, heading in (
            ("ego_to_first", ego_to_first_heading_rad),
            ("first_sample", first_heading_rad),
            ("first_segment", first_segment_heading_rad),
        ):
            error_rad = math.atan2(
                math.sin(float(heading) - float(ego_heading_rad)),
                math.cos(float(heading) - float(ego_heading_rad)),
            )
            if abs(float(error_rad)) > float(max_error_rad):
                violations.append(
                    f"boundary_recovery_{name}_heading_jump"
                )
        return violations

    @staticmethod
    def _mode(request: ReferencePipelineRequest) -> str:
        return FinalReferenceGate._contract_mode(
            behavior_decision=request.behavior_decision,
            behavior_fsm_state=request.behavior_fsm_state,
            stop_goal_active=request.stop_goal_active,
        )

    @staticmethod
    def _forward_m(
        *,
        current_state: Sequence[float],
        x_m: float,
        y_m: float,
    ) -> float:
        dx_m = float(x_m) - float(current_state[0])
        dy_m = float(y_m) - float(current_state[1])
        heading_rad = float(current_state[3])
        return (
            math.cos(heading_rad) * dx_m
            + math.sin(heading_rad) * dy_m
        )

    @staticmethod
    def _compact_reasons(reasons: Sequence[object]) -> str:
        compact = []
        for reason in reasons:
            text = str(reason or "")
            if text and text not in compact:
                compact.append(text)
        return ";".join(compact)
