"""Diagnostics and invariants for the behavior-to-MPC reference boundary.

This module intentionally does not call CARLA APIs or build waypoints.  The
runner still owns the heavy reference generation path, while this module
defines a small, testable contract for explaining what reference the MPC was
asked to track and whether that request violates layer boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from cpx_planning.MPC import build_route_reference_samples

from .planner import is_fixed_stop_decision, normalize_behavior_decision
from .reference_generator import ReferenceIntent
from .temp_destination import build_reference_samples


def _is_lane_change_behavior(behavior_decision: object, fsm_state: object) -> bool:
    normalized_decision = str(normalize_behavior_decision(str(behavior_decision)))
    normalized_fsm = str(fsm_state or "").strip().upper()
    return normalized_decision in {"lane_change_left", "lane_change_right"} or normalized_fsm in {
        "PREPARE_LANE_CHANGE_LEFT",
        "PREPARE_LANE_CHANGE_RIGHT",
        "EXECUTE_LANE_CHANGE_LEFT",
        "EXECUTE_LANE_CHANGE_RIGHT",
    }


def _join_reasons(values: Iterable[object]) -> str:
    return "|".join(str(value) for value in values if str(value))


def reference_sample_forward_lateral_m(
    *,
    reference_sample: Mapping[str, object] | None,
    ego_state: Sequence[float],
) -> tuple[float | None, float | None]:
    """Project a reference sample into ego forward/lateral coordinates."""
    if not isinstance(reference_sample, Mapping) or len(ego_state) < 4:
        return None, None
    try:
        ref_x_m = float(
            reference_sample.get(
                "x_ref_m",
                reference_sample.get("x_m", reference_sample.get("x", "")),
            )
        )
        ref_y_m = float(
            reference_sample.get(
                "y_ref_m",
                reference_sample.get("y_m", reference_sample.get("y", "")),
            )
        )
    except Exception:
        return None, None
    ego_x_m = float(ego_state[0])
    ego_y_m = float(ego_state[1])
    ego_heading_rad = float(ego_state[3])
    dx_m = float(ref_x_m) - float(ego_x_m)
    dy_m = float(ref_y_m) - float(ego_y_m)
    forward_m = math.cos(ego_heading_rad) * dx_m + math.sin(ego_heading_rad) * dy_m
    lateral_m = -math.sin(ego_heading_rad) * dx_m + math.cos(ego_heading_rad) * dy_m
    return float(forward_m), float(lateral_m)


def lane_center_destination_from_reference(
    *,
    destination_state: Sequence[float] | None,
    lane_center_reference: Sequence[Mapping[str, object]] | None,
    ego_state: Sequence[float],
    target_forward_m: float,
) -> List[float] | None:
    """Align the displayed local goal with the MPC lane reference."""
    if destination_state is None:
        return None
    destination = list(destination_state)
    if len(destination) < 4 or not lane_center_reference:
        return destination

    target_forward_m = max(1.0, float(target_forward_m))
    chosen_sample: Mapping[str, object] | None = None
    chosen_forward_m: float | None = None
    for sample in list(lane_center_reference or []):
        forward_m, _ = reference_sample_forward_lateral_m(
            reference_sample=sample,
            ego_state=ego_state,
        )
        if forward_m is None:
            continue
        chosen_sample = sample
        chosen_forward_m = float(forward_m)
        if float(forward_m) >= float(target_forward_m):
            break
    if chosen_sample is None:
        return destination

    try:
        destination[0] = float(chosen_sample.get("x_ref_m", chosen_sample.get("x", destination[0])))
        destination[1] = float(chosen_sample.get("y_ref_m", chosen_sample.get("y", destination[1])))
        destination[3] = float(chosen_sample.get("heading_rad", destination[3]))
        if len(destination) >= 5:
            destination[4] = float(chosen_sample.get("lane_id", destination[4]))
        if chosen_forward_m is not None and float(chosen_forward_m) < 0.5:
            ego_x_m = float(ego_state[0])
            ego_y_m = float(ego_state[1])
            ego_heading_rad = float(ego_state[3])
            destination[0] = ego_x_m + target_forward_m * math.cos(ego_heading_rad)
            destination[1] = ego_y_m + target_forward_m * math.sin(ego_heading_rad)
            destination[3] = ego_heading_rad
    except Exception:
        return list(destination_state)
    return destination


def is_lane_change_reference_decision(decision: object) -> bool:
    normalized = str(decision or "").strip().lower()
    return normalized in {
        "lane_change_left",
        "lane_change_right",
        "reroute",
        "prepare_lane_change_left",
        "prepare_lane_change_right",
        "execute_lane_change_left",
        "execute_lane_change_right",
    }


def reference_first_sample_jump_m(
    previous_reference: Sequence[Mapping[str, object]] | None,
    current_reference: Sequence[Mapping[str, object]] | None,
) -> float:
    if not previous_reference or not current_reference:
        return 0.0
    try:
        previous_sample = previous_reference[0]
        current_sample = current_reference[0]
        return float(math.hypot(
            float(current_sample.get("x_ref_m", current_sample.get("x", 0.0)))
            - float(previous_sample.get("x_ref_m", previous_sample.get("x", 0.0))),
            float(current_sample.get("y_ref_m", current_sample.get("y", 0.0)))
            - float(previous_sample.get("y_ref_m", previous_sample.get("y", 0.0))),
        ))
    except Exception:
        return 0.0


def extrapolate_reference_sample(
    sample: Mapping[str, object],
    *,
    step_distance_m: float,
) -> Dict[str, object]:
    next_sample = dict(sample)
    try:
        heading_rad = float(next_sample.get("heading_rad", 0.0))
        x_m = float(next_sample.get("x_ref_m", next_sample.get("x", 0.0)))
        y_m = float(next_sample.get("y_ref_m", next_sample.get("y", 0.0)))
        next_sample["x_ref_m"] = float(x_m) + float(step_distance_m) * math.cos(heading_rad)
        next_sample["y_ref_m"] = float(y_m) + float(step_distance_m) * math.sin(heading_rad)
    except Exception:
        pass
    return next_sample


def lane_follow_reference_forward_trim(
    reference_samples: Sequence[Mapping[str, object]] | None,
    *,
    ego_state: Sequence[float],
    min_first_forward_m: float,
    step_distance_m: float,
) -> List[Dict[str, object]]:
    """Drop near/behind first samples so lane-follow tracking starts ahead."""
    samples = [dict(sample) for sample in list(reference_samples or [])]
    if len(samples) <= 1 or len(ego_state) < 4 or float(min_first_forward_m) <= 0.0:
        return samples

    first_valid_index = 0
    for index, sample in enumerate(samples):
        forward_m, _ = reference_sample_forward_lateral_m(
            reference_sample=sample,
            ego_state=ego_state,
        )
        if forward_m is not None and float(forward_m) >= float(min_first_forward_m):
            first_valid_index = int(index)
            break
    else:
        first_valid_index = min(len(samples) - 1, 1)

    if int(first_valid_index) <= 0:
        return samples

    trimmed = [dict(sample) for sample in samples[int(first_valid_index):]]
    estimated_step_m = max(0.25, float(step_distance_m))
    if len(samples) >= 2:
        try:
            a = samples[-2]
            b = samples[-1]
            estimated_step_m = max(
                0.25,
                math.hypot(
                    float(b.get("x_ref_m", b.get("x", 0.0))) - float(a.get("x_ref_m", a.get("x", 0.0))),
                    float(b.get("y_ref_m", b.get("y", 0.0))) - float(a.get("y_ref_m", a.get("y", 0.0))),
                ),
            )
        except Exception:
            estimated_step_m = max(0.25, float(step_distance_m))
    while len(trimmed) < len(samples) and trimmed:
        trimmed.append(
            extrapolate_reference_sample(
                trimmed[-1],
                step_distance_m=float(estimated_step_m),
            )
        )
    return trimmed[: len(samples)]


def blend_reference_samples_with_previous(
    current_reference: Sequence[Mapping[str, object]] | None,
    previous_reference: Sequence[Mapping[str, object]] | None,
    *,
    alpha_current: float,
    blend_when_jump_above_m: float,
) -> tuple[List[Dict[str, object]], bool]:
    """Low-pass filter the reference horizon when the first sample jitters."""
    current_samples = [dict(sample) for sample in list(current_reference or [])]
    previous_samples = [dict(sample) for sample in list(previous_reference or [])]
    if not current_samples or not previous_samples:
        return current_samples, False
    jump_m = reference_first_sample_jump_m(previous_samples, current_samples)
    if float(jump_m) < max(0.0, float(blend_when_jump_above_m)):
        return current_samples, False

    alpha = min(1.0, max(0.0, float(alpha_current)))
    count = min(len(current_samples), len(previous_samples))
    blended: List[Dict[str, object]] = []
    for index in range(count):
        current = dict(current_samples[index])
        previous = dict(previous_samples[index])
        out = dict(current)
        for x_key in ("x_ref_m", "x"):
            if x_key in current or x_key in previous:
                try:
                    out[x_key] = (
                        alpha * float(current.get(x_key, current.get("x_ref_m", current.get("x", 0.0))))
                        + (1.0 - alpha) * float(previous.get(x_key, previous.get("x_ref_m", previous.get("x", 0.0))))
                    )
                except Exception:
                    pass
        for y_key in ("y_ref_m", "y"):
            if y_key in current or y_key in previous:
                try:
                    out[y_key] = (
                        alpha * float(current.get(y_key, current.get("y_ref_m", current.get("y", 0.0))))
                        + (1.0 - alpha) * float(previous.get(y_key, previous.get("y_ref_m", previous.get("y", 0.0))))
                    )
                except Exception:
                    pass
        try:
            c_heading = float(current.get("heading_rad", 0.0))
            p_heading = float(previous.get("heading_rad", c_heading))
            delta = math.atan2(math.sin(c_heading - p_heading), math.cos(c_heading - p_heading))
            out["heading_rad"] = p_heading + alpha * delta
        except Exception:
            pass
        blended.append(out)
    if len(current_samples) > count:
        blended.extend([dict(sample) for sample in current_samples[count:]])
    return blended, True


def stabilize_lane_reference_samples(
    current_reference: Sequence[Mapping[str, object]] | None,
    previous_reference: Sequence[Mapping[str, object]] | None,
    *,
    decision: object,
    max_non_lc_first_sample_jump_m: float = 2.25,
    freeze_on_jump: bool = True,
) -> tuple[List[Dict[str, object]], bool, float]:
    """Keep the MPC horizon continuous when a non-lane-change reference jumps."""
    current_samples = [dict(sample) for sample in list(current_reference or [])]
    previous_samples = [dict(sample) for sample in list(previous_reference or [])]
    if not current_samples:
        return current_samples, False, 0.0
    jump_m = reference_first_sample_jump_m(previous_samples, current_samples)
    jump_detected = (
        bool(previous_samples)
        and not is_lane_change_reference_decision(decision)
        and float(jump_m) > float(max_non_lc_first_sample_jump_m)
    )
    if bool(jump_detected) and bool(freeze_on_jump) and previous_samples:
        return previous_samples, True, float(jump_m)
    return current_samples, bool(jump_detected), float(jump_m)


def _angle_diff_abs_rad(a_rad: float, b_rad: float) -> float:
    return abs(math.atan2(math.sin(float(a_rad) - float(b_rad)), math.cos(float(a_rad) - float(b_rad))))


def reference_first_sample_invalid_reason(
    *,
    ego_state: Sequence[float],
    current_reference: Sequence[Mapping[str, object]] | None,
    previous_reference: Sequence[Mapping[str, object]] | None,
    decision: object,
    expected_lane_id: int | None = None,
    max_non_lc_jump_m: float = 4.0,
    max_heading_error_rad: float = 2.2,
    max_backward_m: float = 1.0,
) -> str:
    samples = list(current_reference or [])
    if not samples:
        return "empty_reference"
    if len(ego_state) < 4:
        return ""
    first = dict(samples[0])
    try:
        ref_x = float(first.get("x_ref_m", first.get("x", 0.0)))
        ref_y = float(first.get("y_ref_m", first.get("y", 0.0)))
        ref_heading = float(first.get("heading_rad", float(ego_state[3])))
        ref_lane_id = int(first.get("lane_id", 0))
        ego_x = float(ego_state[0])
        ego_y = float(ego_state[1])
        ego_yaw = float(ego_state[3])
    except Exception:
        return "invalid_reference_values"

    if (
        expected_lane_id is not None
        and int(expected_lane_id) != 0
        and int(ref_lane_id) != 0
        and int(ref_lane_id) != int(expected_lane_id)
        and not is_lane_change_reference_decision(decision)
    ):
        return "first_sample_lane_mismatch"

    dx_m = ref_x - ego_x
    dy_m = ref_y - ego_y
    forward_m = math.cos(ego_yaw) * dx_m + math.sin(ego_yaw) * dy_m
    lateral_m = -math.sin(ego_yaw) * dx_m + math.cos(ego_yaw) * dy_m
    if float(forward_m) < -float(max_backward_m) and math.hypot(dx_m, dy_m) > 3.0:
        return "first_sample_behind_ego"
    if abs(float(lateral_m)) > 4.5 and float(forward_m) < 2.0:
        return "first_sample_lateral_jump"
    if _angle_diff_abs_rad(ref_heading, ego_yaw) > float(max_heading_error_rad):
        return "first_sample_heading_reversed"

    jump_m = reference_first_sample_jump_m(previous_reference, current_reference)
    if (
        bool(previous_reference)
        and not is_lane_change_reference_decision(decision)
        and not bool(is_fixed_stop_decision(normalize_behavior_decision(decision)))
        and float(jump_m) > float(max_non_lc_jump_m)
    ):
        return "first_sample_discontinuous"
    return ""


def route_reference_fallback_samples(
    *,
    ego_state: Sequence[float],
    global_route_points: Sequence[Sequence[float]],
    horizon_steps: int,
    step_distance_m: float,
    target_lane_id: int,
) -> List[Dict[str, object]]:
    if len(global_route_points or []) < 2 or len(ego_state) < 4:
        return []
    ego_snapshot = {
        "x": float(ego_state[0]),
        "y": float(ego_state[1]),
        "psi": float(ego_state[3]),
    }
    route_samples = build_route_reference_samples(
        ego_snapshot=ego_snapshot,
        route_points=global_route_points,
        horizon_steps=int(horizon_steps),
        step_distance_m=float(step_distance_m),
        target_lane_id=int(target_lane_id),
    )
    return [dict(sample) for sample in route_samples]


def heading_reference_fallback_samples(
    *,
    ego_state: Sequence[float],
    horizon_steps: int,
    step_distance_m: float,
    target_lane_id: int,
) -> List[Dict[str, object]]:
    if len(ego_state) < 4:
        return []
    ego_x_m = float(ego_state[0])
    ego_y_m = float(ego_state[1])
    ego_heading_rad = float(ego_state[3])
    step_m = max(0.5, float(step_distance_m))
    samples: List[Dict[str, object]] = []
    for index in range(max(1, int(horizon_steps))):
        distance_m = float(index + 1) * float(step_m)
        samples.append({
            "x_ref_m": float(ego_x_m) + float(distance_m) * math.cos(ego_heading_rad),
            "y_ref_m": float(ego_y_m) + float(distance_m) * math.sin(ego_heading_rad),
            "heading_rad": float(ego_heading_rad),
            "lane_id": int(target_lane_id),
        })
    return samples


def reference_with_route_fallback(
    *,
    ego_state: Sequence[float],
    current_reference: Sequence[Mapping[str, object]] | None,
    previous_reference: Sequence[Mapping[str, object]] | None,
    decision: object,
    global_route_points: Sequence[Sequence[float]],
    horizon_steps: int,
    step_distance_m: float,
    target_lane_id: int,
    allow_route_fallback: bool = True,
    expected_lane_id: int | None = None,
) -> tuple[List[Dict[str, object]], str]:
    reason = reference_first_sample_invalid_reason(
        ego_state=ego_state,
        current_reference=current_reference,
        previous_reference=previous_reference,
        decision=decision,
        expected_lane_id=expected_lane_id,
    )
    if not reason:
        return [dict(sample) for sample in list(current_reference or [])], ""
    if bool(allow_route_fallback):
        fallback_samples = route_reference_fallback_samples(
            ego_state=ego_state,
            global_route_points=global_route_points,
            horizon_steps=int(horizon_steps),
            step_distance_m=float(step_distance_m),
            target_lane_id=int(target_lane_id),
        )
        if fallback_samples:
            fallback_reason = reference_first_sample_invalid_reason(
                ego_state=ego_state,
                current_reference=fallback_samples,
                previous_reference=None,
                decision=decision,
                expected_lane_id=expected_lane_id,
            )
            if not fallback_reason:
                return fallback_samples, str(reason)
    heading_fallback_samples = heading_reference_fallback_samples(
        ego_state=ego_state,
        horizon_steps=int(horizon_steps),
        step_distance_m=float(step_distance_m),
        target_lane_id=int(target_lane_id),
    )
    if heading_fallback_samples:
        fallback_kind = "heading_fallback_no_route" if not bool(allow_route_fallback) else "heading_fallback"
        return heading_fallback_samples, f"{reason}:{fallback_kind}"
    return [dict(sample) for sample in list(current_reference or [])], ""


@dataclass(frozen=True)
class ReferencePipelineTrace:
    """Compact per-tick explanation of the reference handed to MPC."""

    stage: str
    intent_mode: str
    lateral_reference_source: str
    longitudinal_target_kind: str
    stop_target_role: str
    route_role: str
    target_lane_id: int
    output_lane_id: int
    follow_global_route_lane: bool
    fallback_reason: str = ""
    stabilized: bool = False
    jump_m: float = 0.0
    first_forward_m: float | None = None
    first_lateral_m: float | None = None
    violations: List[str] = field(default_factory=list)

    def as_trace_fields(self) -> Dict[str, object]:
        """Flatten trace fields for CSV/HUD diagnostics."""
        return {
            "reference_pipeline_stage": str(self.stage),
            "reference_pipeline_intent_mode": str(self.intent_mode),
            "reference_pipeline_lateral_source": str(self.lateral_reference_source),
            "reference_pipeline_longitudinal_target_kind": str(self.longitudinal_target_kind),
            "reference_pipeline_stop_target_role": str(self.stop_target_role),
            "reference_pipeline_route_role": str(self.route_role),
            "reference_pipeline_target_lane_id": int(self.target_lane_id),
            "reference_pipeline_output_lane_id": int(self.output_lane_id),
            "reference_pipeline_follow_global_route_lane": int(bool(self.follow_global_route_lane)),
            "reference_pipeline_fallback_reason": str(self.fallback_reason),
            "reference_pipeline_stabilized": int(bool(self.stabilized)),
            "reference_pipeline_jump_m": float(self.jump_m),
            "reference_pipeline_first_forward_m": (
                "" if self.first_forward_m is None else float(self.first_forward_m)
            ),
            "reference_pipeline_first_lateral_m": (
                "" if self.first_lateral_m is None else float(self.first_lateral_m)
            ),
            "reference_pipeline_violation_count": int(len(self.violations)),
            "reference_pipeline_violations": _join_reasons(self.violations),
        }


@dataclass(frozen=True)
class MpcReferenceResult:
    """Final lateral reference contract consumed by MPC and diagnostics."""

    samples: List[Dict[str, object]]
    trace: ReferencePipelineTrace
    target_lane_id: int
    first_sample: Dict[str, object] | None = None
    first_forward_m: float | None = None
    first_lateral_m: float | None = None
    fallback_reason: str = ""
    stabilized: bool = False
    jump_m: float = 0.0

    @property
    def has_samples(self) -> bool:
        return bool(self.samples)

    def trace_fields(self) -> Dict[str, object]:
        return self.trace.as_trace_fields()


@dataclass(frozen=True, init=False)
class MpcReferenceGenerationContext:
    """Inputs required to generate the MPC lateral reference for one tick."""

    map_planner: Any
    ego_pose: Mapping[str, object]
    ego_state: Sequence[float]
    active_global_route_points: Sequence[Sequence[float]]
    previous_lane_center_reference: Sequence[Mapping[str, object]] | None
    behavior_runtime_cfg: Mapping[str, object]
    reference_intent: ReferenceIntent
    current_applied_behavior: str
    cached_planner_lc_state: str
    reference_target_lane_id: int
    current_lane_id: int
    global_route_reference_allowed: bool
    global_route_reference_gate_reason: str
    should_follow_global_route_lane_for_reference: bool
    traffic_control_lane_lock_active: bool
    final_goal_stop_active: bool
    stop_target_state: Sequence[float] | None
    follow_target_state: Sequence[float] | None
    current_temp_reference_xy: Sequence[float] | None
    current_temp_mode_value: float
    current_temp_road_id: int | None
    current_temp_entered_intersection: bool
    active_reference_maneuver: str
    current_temp_mode_str: str
    lane_reference_speed_mps: float
    lane_reference_step_distance_m: float
    mpc_horizon_steps: int
    mpc_dt_s: float
    temporary_destination_state: Sequence[float] | None
    lane_reference_freeze_count: int
    sim_time_s: float
    stop_release_temp_smooth_until_sim_time_s: float

    def __init__(self, **kwargs: object) -> None:
        field_names = set(self.__dataclass_fields__.keys())
        unknown_keys = set(kwargs.keys()) - field_names

        if unknown_keys:
            unknown = ", ".join(sorted(str(key) for key in unknown_keys))
            raise TypeError(
                f"Unexpected MpcReferenceGenerationContext argument(s): {unknown}"
            )

        for name in self.__dataclass_fields__:
            if name not in kwargs:
                raise TypeError(
                    f"Missing MpcReferenceGenerationContext argument: {name}"
                )
            object.__setattr__(self, name, kwargs[name])


@dataclass(frozen=True)
class MpcReferenceGenerationOutput:
    """Complete output of one behavior-to-MPC reference generation step.

    The custom map planner owns waypoint generation; this object formalizes
    the handoff boundary to MPC.
    """

    mpc_reference_result: MpcReferenceResult
    local_lane_center_reference: List[Dict[str, object]]
    temporary_destination_state: Sequence[float] | None
    reference_target_lane_id: int
    should_follow_global_route_lane_for_reference: bool
    lane_reference_freeze_count: int
    last_reference_fallback_reason: str
    last_reference_stabilized: bool
    last_reference_jump_m: float
    first_reference_forward_m: float | None
    first_reference_lateral_m: float | None
    reference_geometry_guard_active: bool
    reference_geometry_guard_reason: str


def build_mpc_reference_result(
    *,
    samples: Sequence[Mapping[str, object]],
    intent: ReferenceIntent,
    behavior_decision: object,
    fsm_state: object,
    target_lane_id: int,
    follow_global_route_lane: bool,
    fallback_reason: str,
    stabilized: bool,
    jump_m: float,
    first_forward_m: float | None,
    first_lateral_m: float | None,
    max_non_lc_lateral_m: float = 3.0,
) -> MpcReferenceResult:
    """Package the final MPC reference samples and diagnostics.

    Heavy waypoint generation still happens upstream.  This function marks the
    explicit handoff point: after this, the runner and MPC should consume a
    named reference result instead of a loose collection of local variables.
    """

    normalized_samples = [dict(sample) for sample in list(samples or [])]
    first_sample = dict(normalized_samples[0]) if normalized_samples else None
    trace = trace_reference_pipeline(
        intent=intent,
        behavior_decision=behavior_decision,
        fsm_state=fsm_state,
        target_lane_id=int(target_lane_id),
        output_reference_sample=first_sample,
        follow_global_route_lane=bool(follow_global_route_lane),
        fallback_reason=str(fallback_reason),
        stabilized=bool(stabilized),
        jump_m=float(jump_m),
        first_forward_m=first_forward_m,
        first_lateral_m=first_lateral_m,
        max_non_lc_lateral_m=float(max_non_lc_lateral_m),
    )
    return MpcReferenceResult(
        samples=normalized_samples,
        trace=trace,
        target_lane_id=int(target_lane_id),
        first_sample=first_sample,
        first_forward_m=first_forward_m,
        first_lateral_m=first_lateral_m,
        fallback_reason=str(fallback_reason),
        stabilized=bool(stabilized),
        jump_m=float(jump_m),
    )


def trace_reference_pipeline(
    *,
    intent: ReferenceIntent,
    behavior_decision: object,
    fsm_state: object,
    target_lane_id: int,
    output_reference_sample: Mapping[str, object] | None,
    follow_global_route_lane: bool,
    fallback_reason: str,
    stabilized: bool,
    jump_m: float,
    first_forward_m: float | None,
    first_lateral_m: float | None,
    max_non_lc_lateral_m: float = 3.0,
) -> ReferencePipelineTrace:
    """Build an invariant-checked trace for the MPC reference handoff."""

    normalized_decision = str(normalize_behavior_decision(str(behavior_decision)))
    output_lane_id = int(target_lane_id)
    if isinstance(output_reference_sample, Mapping):
        try:
            output_lane_id = int(output_reference_sample.get("lane_id", output_lane_id))
        except Exception:
            output_lane_id = int(target_lane_id)

    lane_change_active = _is_lane_change_behavior(normalized_decision, fsm_state)
    fixed_stop_active = bool(is_fixed_stop_decision(normalized_decision))
    violations: List[str] = []

    if normalized_decision == "lane_follow" and not lane_change_active:
        if str(intent.lateral_reference_source) != "lane_center":
            violations.append("lane_follow_lateral_source_not_lane_center")
        if str(intent.longitudinal_target_kind) != "speed_profile":
            violations.append("lane_follow_longitudinal_target_not_speed_profile")
        if bool(follow_global_route_lane):
            violations.append("lane_follow_direct_global_route_tracking")

    if fixed_stop_active:
        if str(intent.stop_target_role) != "longitudinal_speed_target":
            violations.append("stop_target_not_longitudinal")
        if str(intent.lateral_reference_source) != "lane_center":
            violations.append("stop_lateral_source_not_lane_center")

    if lane_change_active and str(intent.lateral_reference_source) != "lane_change_blend":
        violations.append("lane_change_lateral_source_not_blend")

    if (
        first_lateral_m is not None
        and not lane_change_active
        and not fixed_stop_active
        and abs(float(first_lateral_m)) > max(0.0, float(max_non_lc_lateral_m))
    ):
        violations.append("non_lc_reference_lateral_jump")

    stage_parts = ["intent", "raw", "validated"]
    if str(fallback_reason):
        stage_parts.append("fallback")
    if bool(stabilized):
        stage_parts.append("stabilized")

    return ReferencePipelineTrace(
        stage=">".join(stage_parts),
        intent_mode=str(intent.mode),
        lateral_reference_source=str(intent.lateral_reference_source),
        longitudinal_target_kind=str(intent.longitudinal_target_kind),
        stop_target_role=str(intent.stop_target_role),
        route_role=str(intent.route_role),
        target_lane_id=int(target_lane_id),
        output_lane_id=int(output_lane_id),
        follow_global_route_lane=bool(follow_global_route_lane),
        fallback_reason=str(fallback_reason),
        stabilized=bool(stabilized),
        jump_m=float(jump_m),
        first_forward_m=first_forward_m,
        first_lateral_m=first_lateral_m,
        violations=violations,
    )


def generate_mpc_reference(
    context: MpcReferenceGenerationContext,
) -> MpcReferenceGenerationOutput:
    """Generate the final MPC reference and package its diagnostics."""

    map_planner = context.map_planner
    ego_pose = context.ego_pose
    ego_state = context.ego_state
    active_global_route_points = context.active_global_route_points
    previous_lane_center_reference = context.previous_lane_center_reference
    behavior_runtime_cfg = context.behavior_runtime_cfg
    reference_intent = context.reference_intent
    current_applied_behavior = context.current_applied_behavior
    cached_planner_lc_state = context.cached_planner_lc_state
    reference_target_lane_id = context.reference_target_lane_id
    current_lane_id = context.current_lane_id
    global_route_reference_allowed = context.global_route_reference_allowed
    global_route_reference_gate_reason = context.global_route_reference_gate_reason
    should_follow_global_route_lane_for_reference = (
        context.should_follow_global_route_lane_for_reference
    )
    traffic_control_lane_lock_active = context.traffic_control_lane_lock_active
    final_goal_stop_active = context.final_goal_stop_active
    stop_target_state = context.stop_target_state
    follow_target_state = context.follow_target_state
    current_temp_reference_xy = context.current_temp_reference_xy
    current_temp_mode_value = context.current_temp_mode_value
    current_temp_road_id = context.current_temp_road_id
    current_temp_entered_intersection = context.current_temp_entered_intersection
    active_reference_maneuver = context.active_reference_maneuver
    current_temp_mode_str = context.current_temp_mode_str
    lane_reference_speed_mps = context.lane_reference_speed_mps
    lane_reference_step_distance_m = context.lane_reference_step_distance_m
    mpc_horizon_steps = context.mpc_horizon_steps
    mpc_dt_s = context.mpc_dt_s
    temporary_destination_state = context.temporary_destination_state
    lane_reference_freeze_count = context.lane_reference_freeze_count
    sim_time_s = context.sim_time_s
    stop_release_temp_smooth_until_sim_time_s = (
        context.stop_release_temp_smooth_until_sim_time_s
    )

    reference_route_points = (
        active_global_route_points
        if (
            bool(reference_intent.follow_global_route_lane)
            and not bool(is_fixed_stop_decision(current_applied_behavior))
            and not bool(traffic_control_lane_lock_active)
        )
        else []
    )
    reference_stop_target_state = stop_target_state if bool(final_goal_stop_active) else None
    raw_reference = build_reference_samples(
        map_planner=map_planner,
        ego_pose=ego_pose,
        target_lane_id=int(reference_target_lane_id),
        decision=str(current_applied_behavior),
        horizon_steps=int(mpc_horizon_steps),
        step_distance_m=float(lane_reference_step_distance_m),
        global_route_points=reference_route_points,
        mode_reference_xy=current_temp_reference_xy,
        prev_mode=current_temp_mode_value,
        prev_road_id=current_temp_road_id,
        prev_entered_intersection=bool(current_temp_entered_intersection),
        next_macro_maneuver=str(active_reference_maneuver),
        mode_override=str(current_temp_mode_str),
        stop_target_state=reference_stop_target_state,
        follow_target_state=follow_target_state,
        follow_global_route_lane=bool(reference_intent.follow_global_route_lane),
        force_stop_reference=False,
    )
    reference_samples, fallback_reason = reference_with_route_fallback(
        ego_state=ego_state,
        current_reference=raw_reference,
        previous_reference=previous_lane_center_reference,
        decision=str(current_applied_behavior),
        global_route_points=active_global_route_points,
        horizon_steps=int(mpc_horizon_steps),
        step_distance_m=float(lane_reference_step_distance_m),
        target_lane_id=int(reference_target_lane_id),
        allow_route_fallback=(
            bool(reference_intent.follow_global_route_lane)
            and not bool(is_fixed_stop_decision(current_applied_behavior))
        ),
        expected_lane_id=(
            0 if str(reference_intent.mode) == "route_branch_follow" else int(reference_target_lane_id)
        ),
    )
    if (
        str(fallback_reason)
        and not bool(global_route_reference_allowed)
        and str(global_route_reference_gate_reason)
    ):
        fallback_reason = f"{fallback_reason}:route_reference_{global_route_reference_gate_reason}"

    lane_follow_filter = (
        str(reference_intent.mode) == "lane_follow"
        and str(normalize_behavior_decision(current_applied_behavior)) == "lane_follow"
        and not bool(is_fixed_stop_decision(current_applied_behavior))
        and str(cached_planner_lc_state or "").upper() in {"IDLE", "LANE_KEEP"}
    )
    route_branch_filter = (
        str(reference_intent.mode) == "route_branch_follow"
        and str(normalize_behavior_decision(current_applied_behavior)) == "lane_follow"
        and not bool(is_fixed_stop_decision(current_applied_behavior))
        and str(cached_planner_lc_state or "").upper() in {"IDLE", "LANE_KEEP"}
    )
    min_first_forward_m = float(
        behavior_runtime_cfg.get("lane_follow_reference_min_first_forward_m", 1.0)
    )
    forward_filter = bool(lane_follow_filter) or bool(route_branch_filter)
    if bool(forward_filter):
        reference_samples = lane_follow_reference_forward_trim(
            reference_samples,
            ego_state=ego_state,
            min_first_forward_m=float(min_first_forward_m),
            step_distance_m=float(lane_reference_step_distance_m),
        )

    previous_for_stabilization = previous_lane_center_reference
    if bool(forward_filter) and previous_lane_center_reference:
        previous_forward_m, _ = reference_sample_forward_lateral_m(
            reference_sample=previous_lane_center_reference[0],
            ego_state=ego_state,
        )
        if previous_forward_m is not None and float(previous_forward_m) < 0.5 * float(min_first_forward_m):
            previous_for_stabilization = []

    reference_samples, stabilized, jump_m = stabilize_lane_reference_samples(
        reference_samples,
        previous_for_stabilization,
        decision=str(current_applied_behavior),
        max_non_lc_first_sample_jump_m=max(
            0.0,
            float(
                behavior_runtime_cfg.get(
                    (
                        "lane_follow_reference_freeze_jump_threshold_m"
                        if bool(lane_follow_filter)
                        else "reference_stabilization_jump_threshold_m"
                    ),
                    1.0 if bool(lane_follow_filter) else 2.25,
                )
            ),
        ),
        freeze_on_jump=bool(behavior_runtime_cfg.get("reference_stabilization_freeze_on_jump", True)),
    )
    if bool(lane_follow_filter) and not bool(stabilized) and previous_for_stabilization:
        reference_samples, blended = blend_reference_samples_with_previous(
            reference_samples,
            previous_for_stabilization,
            alpha_current=float(behavior_runtime_cfg.get("lane_follow_reference_blend_alpha", 0.55)),
            blend_when_jump_above_m=float(
                behavior_runtime_cfg.get("lane_follow_reference_blend_jump_threshold_m", 0.75)
            ),
        )
        if bool(blended):
            stabilized = True
            jump_m = reference_first_sample_jump_m(previous_for_stabilization, reference_samples)
            fallback_reason = (f"{fallback_reason}:" if str(fallback_reason) else "") + "lane_follow_reference_blend"

    first_forward_m = None
    first_lateral_m = None
    if reference_samples:
        first_forward_m, first_lateral_m = reference_sample_forward_lateral_m(
            reference_sample=reference_samples[0],
            ego_state=ego_state,
        )

    geometry_guard_active = False
    geometry_guard_reason = ""
    if (
        reference_samples
        and str(reference_intent.mode) != "route_branch_follow"
        and not bool(is_fixed_stop_decision(current_applied_behavior))
        and str(normalize_behavior_decision(current_applied_behavior)) not in {"lane_change_left", "lane_change_right"}
    ):
        min_guard_forward_m = float(
            behavior_runtime_cfg.get(
                (
                    "lane_follow_reference_guard_min_forward_m"
                    if bool(lane_follow_filter)
                    else "reference_first_sample_min_forward_m"
                ),
                0.75 if bool(lane_follow_filter) else -0.5,
            )
        )
        max_guard_lateral_m = float(
            behavior_runtime_cfg.get(
                (
                    "lane_follow_reference_guard_max_lateral_m"
                    if bool(lane_follow_filter)
                    else "reference_first_sample_max_lateral_m"
                ),
                2.5 if bool(lane_follow_filter) else 4.5,
            )
        )
        if first_forward_m is not None and float(first_forward_m) < float(min_guard_forward_m):
            geometry_guard_active = True
            geometry_guard_reason = "first_sample_behind"
        elif first_lateral_m is not None and abs(float(first_lateral_m)) > max(0.0, float(max_guard_lateral_m)):
            geometry_guard_active = True
            geometry_guard_reason = "first_sample_lateral_too_far"

    if bool(geometry_guard_active):
        reanchored_reference = build_reference_samples(
            map_planner=map_planner,
            ego_pose=ego_pose,
            target_lane_id=int(current_lane_id),
            decision="lane_follow",
            horizon_steps=int(mpc_horizon_steps),
            step_distance_m=max(0.5, float(lane_reference_speed_mps) * float(mpc_dt_s)),
            global_route_points=[],
            mode_reference_xy=None,
            prev_mode=0.0,
            prev_road_id=None,
            prev_entered_intersection=False,
            next_macro_maneuver="straight",
            mode_override="NORMAL",
            stop_target_state=None,
            follow_target_state=None,
            follow_global_route_lane=False,
            force_stop_reference=False,
        )
        reanchor_reason = "lane_center_reanchor"
        reanchor_invalid_reason = reference_first_sample_invalid_reason(
            ego_state=ego_state,
            current_reference=reanchored_reference,
            previous_reference=None,
            decision="lane_follow",
            expected_lane_id=int(current_lane_id),
            max_non_lc_jump_m=max(
                0.0,
                float(behavior_runtime_cfg.get("reference_stabilization_jump_threshold_m", 2.25)),
            ),
        )
        if reanchor_invalid_reason:
            reanchored_reference = heading_reference_fallback_samples(
                ego_state=ego_state,
                horizon_steps=int(mpc_horizon_steps),
                step_distance_m=max(0.5, float(lane_reference_speed_mps) * float(mpc_dt_s)),
                target_lane_id=int(current_lane_id),
            )
            reanchor_reason = "heading_reanchor"
        if bool(lane_follow_filter):
            reanchored_reference = lane_follow_reference_forward_trim(
                reanchored_reference,
                ego_state=ego_state,
                min_first_forward_m=float(min_first_forward_m),
                step_distance_m=float(lane_reference_step_distance_m),
            )
        reference_samples = [dict(sample) for sample in reanchored_reference]
        reference_target_lane_id = int(current_lane_id)
        should_follow_global_route_lane_for_reference = False
        stabilized = True
        fallback_reason = (
            f"{fallback_reason}:" if str(fallback_reason) else ""
        ) + f"{geometry_guard_reason}:{reanchor_reason}"
        jump_m = reference_first_sample_jump_m(previous_for_stabilization, reference_samples)

    lane_reference_freeze_count = int(lane_reference_freeze_count) + 1 if bool(stabilized) else 0
    freeze_reanchor_after_replans = max(
        0,
        int(behavior_runtime_cfg.get("reference_freeze_reanchor_after_replans", 4)),
    )
    if (
        bool(stabilized)
        and str(reference_intent.mode) != "route_branch_follow"
        and int(freeze_reanchor_after_replans) > 0
        and int(lane_reference_freeze_count) > int(freeze_reanchor_after_replans)
    ):
        allow_heading_reanchor = (
            not bool(is_fixed_stop_decision(current_applied_behavior))
            and float(sim_time_s) >= float(stop_release_temp_smooth_until_sim_time_s)
        )
        reanchored_reference = []
        if bool(is_fixed_stop_decision(current_applied_behavior)):
            reanchored_reference = build_reference_samples(
                map_planner=map_planner,
                ego_pose=ego_pose,
                target_lane_id=int(reference_target_lane_id),
                decision=str(current_applied_behavior),
                horizon_steps=int(mpc_horizon_steps),
                step_distance_m=max(0.5, float(lane_reference_speed_mps) * float(mpc_dt_s)),
                global_route_points=reference_route_points,
                mode_reference_xy=current_temp_reference_xy,
                prev_mode=current_temp_mode_value,
                prev_road_id=current_temp_road_id,
                prev_entered_intersection=bool(current_temp_entered_intersection),
                next_macro_maneuver=str(active_reference_maneuver),
                mode_override=str(current_temp_mode_str),
                stop_target_state=reference_stop_target_state,
                follow_target_state=follow_target_state,
                follow_global_route_lane=bool(reference_intent.follow_global_route_lane),
                force_stop_reference=True,
            )
        elif bool(allow_heading_reanchor):
            reanchored_reference = heading_reference_fallback_samples(
                ego_state=ego_state,
                horizon_steps=int(mpc_horizon_steps),
                step_distance_m=max(0.5, float(lane_reference_speed_mps) * float(mpc_dt_s)),
                target_lane_id=int(reference_target_lane_id),
            )
        if reanchored_reference:
            reference_samples = [dict(sample) for sample in reanchored_reference]
            reanchor_kind = "stop_target_reanchor" if bool(is_fixed_stop_decision(current_applied_behavior)) else "heading_reanchor"
            fallback_reason = (
                f"{fallback_reason}:" if str(fallback_reason) else ""
            ) + f"reference_freeze_timeout:{reanchor_kind}"
            lane_reference_freeze_count = 0

    if (
        temporary_destination_state is not None
        and reference_samples
        and str(reference_intent.mode) == "lane_follow"
        and str(normalize_behavior_decision(current_applied_behavior)) == "lane_follow"
        and str(cached_planner_lc_state or "").upper() in {"IDLE", "LANE_KEEP"}
        and not bool(is_fixed_stop_decision(current_applied_behavior))
        and not bool(final_goal_stop_active)
    ):
        temporary_destination_state = lane_center_destination_from_reference(
            destination_state=temporary_destination_state,
            lane_center_reference=reference_samples,
            ego_state=ego_state,
            target_forward_m=float(
                behavior_runtime_cfg.get("lane_follow_destination_reference_forward_m", 6.0)
            ),
        )

    first_forward_m = None
    first_lateral_m = None
    if reference_samples:
        first_forward_m, first_lateral_m = reference_sample_forward_lateral_m(
            reference_sample=reference_samples[0],
            ego_state=ego_state,
        )
    mpc_reference_result = build_mpc_reference_result(
        samples=reference_samples,
        intent=reference_intent,
        behavior_decision=str(current_applied_behavior),
        fsm_state=str(cached_planner_lc_state),
        target_lane_id=int(reference_target_lane_id),
        follow_global_route_lane=bool(should_follow_global_route_lane_for_reference),
        fallback_reason=str(fallback_reason),
        stabilized=bool(stabilized),
        jump_m=float(jump_m),
        first_forward_m=first_forward_m,
        first_lateral_m=first_lateral_m,
        max_non_lc_lateral_m=float(
            behavior_runtime_cfg.get("reference_pipeline_non_lc_max_lateral_m", 3.0)
        ),
    )
    return MpcReferenceGenerationOutput(
        mpc_reference_result=mpc_reference_result,
        local_lane_center_reference=[dict(sample) for sample in mpc_reference_result.samples],
        temporary_destination_state=temporary_destination_state,
        reference_target_lane_id=int(reference_target_lane_id),
        should_follow_global_route_lane_for_reference=bool(should_follow_global_route_lane_for_reference),
        lane_reference_freeze_count=int(lane_reference_freeze_count),
        last_reference_fallback_reason=str(fallback_reason),
        last_reference_stabilized=bool(stabilized),
        last_reference_jump_m=float(jump_m),
        first_reference_forward_m=first_forward_m,
        first_reference_lateral_m=first_lateral_m,
        reference_geometry_guard_active=bool(geometry_guard_active),
        reference_geometry_guard_reason=str(geometry_guard_reason),
    )


def summarize_reference_pipeline_history(
    history: Iterable[Mapping[str, object]],
) -> Dict[str, object]:
    """Aggregate lane-reference trace rows into a compact run summary."""

    rows = [dict(row) for row in list(history or [])]
    violation_counter: Counter[str] = Counter()
    stage_counter: Counter[str] = Counter()
    lateral_source_counter: Counter[str] = Counter()
    intent_mode_counter: Counter[str] = Counter()
    fallback_counter: Counter[str] = Counter()
    stabilized_count = 0
    max_reference_jump_m = 0.0
    max_first_lateral_abs_m = 0.0
    max_first_forward_m = 0.0
    min_first_forward_m = None

    for row in rows:
        stage_counter[str(row.get("reference_pipeline_stage", ""))] += 1
        lateral_source_counter[str(row.get("reference_pipeline_lateral_source", ""))] += 1
        intent_mode_counter[str(row.get("reference_pipeline_intent_mode", ""))] += 1
        fallback_reason = str(row.get("reference_pipeline_fallback_reason", ""))
        if fallback_reason:
            fallback_counter[fallback_reason] += 1
        try:
            stabilized_count += int(bool(int(row.get("reference_pipeline_stabilized", 0) or 0)))
        except Exception:
            stabilized_count += int(bool(row.get("reference_pipeline_stabilized", False)))
        try:
            max_reference_jump_m = max(
                max_reference_jump_m,
                abs(float(row.get("reference_pipeline_jump_m", 0.0) or 0.0)),
            )
        except Exception:
            pass
        try:
            first_lateral_m = float(row.get("reference_pipeline_first_lateral_m", 0.0) or 0.0)
            max_first_lateral_abs_m = max(max_first_lateral_abs_m, abs(first_lateral_m))
        except Exception:
            pass
        try:
            first_forward_m = float(row.get("reference_pipeline_first_forward_m", 0.0) or 0.0)
            max_first_forward_m = max(max_first_forward_m, first_forward_m)
            min_first_forward_m = (
                first_forward_m
                if min_first_forward_m is None
                else min(float(min_first_forward_m), first_forward_m)
            )
        except Exception:
            pass
        for violation in str(row.get("reference_pipeline_violations", "")).split("|"):
            if violation:
                violation_counter[violation] += 1

    total_rows = len(rows)
    violation_total = int(sum(violation_counter.values()))
    return {
        "samples": int(total_rows),
        "violation_samples": int(
            sum(1 for row in rows if int(row.get("reference_pipeline_violation_count", 0) or 0) > 0)
        ),
        "violation_total": int(violation_total),
        "violation_rate": 0.0 if total_rows <= 0 else float(violation_total) / float(total_rows),
        "stabilized_samples": int(stabilized_count),
        "stabilized_rate": 0.0 if total_rows <= 0 else float(stabilized_count) / float(total_rows),
        "max_reference_jump_m": float(max_reference_jump_m),
        "max_first_lateral_abs_m": float(max_first_lateral_abs_m),
        "min_first_forward_m": "" if min_first_forward_m is None else float(min_first_forward_m),
        "max_first_forward_m": float(max_first_forward_m),
        "violations_by_type": dict(violation_counter.most_common()),
        "stages": dict(stage_counter.most_common()),
        "lateral_sources": dict(lateral_source_counter.most_common()),
        "intent_modes": dict(intent_mode_counter.most_common()),
        "fallback_reasons": dict(fallback_counter.most_common()),
    }
