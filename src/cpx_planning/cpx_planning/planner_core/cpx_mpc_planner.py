"""CARLA- and OpenCDA-independent form of the active CP-X MPC planner bridge."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from cpx_planning.behavior_planner.reroute import reroute_from_lane_closure_messages
from cpx_planning.pipeline.candidate_evaluation import evaluate_behavior_candidates
from cpx_planning.pipeline.route_authorization import authorize_route_lane_change
from cpx_planning.pipeline.scenario_manager import CPXScenarioManager
from cpx_planning.pipeline.speed_planner import build_speed_plan
from cpx_planning.behavior_planner import MpcReferenceGenerationContext
from cpx_planning.behavior_planner import compute_temp_destination
from cpx_planning.behavior_planner import generate_mpc_reference
from cpx_planning.behavior_planner import is_emergency_brake_decision
from cpx_planning.behavior_planner import is_fixed_stop_decision
from cpx_planning.behavior_planner import normalize_behavior_decision
from cpx_planning.behavior_planner import select_reference_intent
from cpx_planning.behavior_planner.reference_pipeline import lane_center_destination_from_reference
from cpx_planning.pipeline.output import BehaviorCommand
from cpx_planning.pipeline.output import PlannerControl, PlannerDiagnostics, PlannerOutput
from cpx_planning.pipeline.decision_record import build_decision_record
from cpx_planning.pipeline.safety_supervisor import SafetySupervisor
from cpx_planning.pipeline.spline import Spline2D
from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint, canonical_lane_waypoint_for_lane_id, world_heading_rad

from cpx_planning.pipeline.candidate_pipeline import CandidateReferenceResult
from cpx_planning.pipeline.candidate_pipeline import build_candidate_intents
from cpx_planning.pipeline.candidate_pipeline import evaluate_candidate_reference
from cpx_planning.pipeline.candidate_pipeline import select_best_candidate
from cpx_planning.pipeline.candidate_pipeline import summarize_candidate_results
from cpx_planning.pipeline.reference_contract import contract_from_config, validate_reference_contract


@dataclass(frozen=True)
class _PlannerLocation:
    """Hold numeric position values where the copied planner previously received a CARLA Location."""

    x: float
    y: float
    z: float = 0.0

    @classmethod
    def from_mapping(cls, value):
        return cls(x=float(value.get("x", 0.0)), y=float(value.get("y", 0.0)), z=float(value.get("z", 0.0)))


@dataclass(frozen=True)
class _PlannerRotation:
    """Hold the numeric yaw angle needed by copied control fallback logic."""

    yaw: float


@dataclass(frozen=True)
class _PlannerTransform:
    """Group the primitive position and yaw values without importing CARLA."""

    location: _PlannerLocation
    rotation: _PlannerRotation

    @classmethod
    def from_pose(cls, value):
        return cls(location=_PlannerLocation.from_mapping(value), rotation=_PlannerRotation(yaw=math.degrees(float(value.get("heading_rad", 0.0)))))


class _Mode2TrafficLightMemory:
    """Temporal debounce for traffic controls in OpenCDA-reference MPC mode."""

    def __init__(
        self,
        *,
        hold_unknown_s: float = 0.8,
        green_confirm_s: float = 0.0,
    ) -> None:
        self.hold_unknown_s = max(0.0, float(hold_unknown_s))
        self.green_confirm_s = max(0.0, float(green_confirm_s))
        self._last_stop_state = "unknown"
        self._last_stop_target: dict[str, object] | None = None
        self._hold_until_s = -float("inf")
        self._green_since_s: float | None = None

    def update(
        self,
        *,
        state: str,
        stop_target: Mapping[str, object] | None,
        sim_time_s: float,
    ) -> tuple[str, dict[str, object] | None, str]:
        normalized_state = str(state or "unknown").strip().lower()
        reason = ""
        if normalized_state in {"red", "yellow"}:
            self._green_since_s = None
            self._last_stop_state = str(normalized_state)
            self._last_stop_target = dict(stop_target or {}) if isinstance(stop_target, Mapping) else None
            self._hold_until_s = float(sim_time_s) + float(self.hold_unknown_s)
            return str(normalized_state), self._last_stop_target, "raw_stop"
        if normalized_state == "green":
            if self._green_since_s is None:
                self._green_since_s = float(sim_time_s)
            if (
                self._last_stop_state in {"red", "yellow"}
                and float(sim_time_s) - float(self._green_since_s) < float(self.green_confirm_s)
            ):
                return str(self._last_stop_state), self._last_stop_target, "traffic_memory_wait_green_confirm"
            self._last_stop_state = "green"
            self._last_stop_target = None
            self._hold_until_s = -float("inf")
            return "green", None, "traffic_memory_green_release" if float(self.green_confirm_s) > 0.0 else ""
        if (
            normalized_state == "unknown"
            and float(sim_time_s) <= float(self._hold_until_s)
            and self._last_stop_state in {"red", "yellow"}
        ):
            reason = f"traffic_memory_hold_{self._last_stop_state}"
            return str(self._last_stop_state), self._last_stop_target, reason
        if normalized_state == "unknown":
            self._green_since_s = None
        return str(normalized_state), None, reason


class _Mode2ReferenceMemory:
    """Keep the accepted OpenCDA reference temporally consistent."""

    def __init__(
        self,
        *,
        max_first_point_jump_m: float = 2.0,
        max_destination_jump_m: float = 4.0,
        max_reuse_age_s: float = 1.0,
    ) -> None:
        self.max_first_point_jump_m = max(0.0, float(max_first_point_jump_m))
        self.max_destination_jump_m = max(0.0, float(max_destination_jump_m))
        self.max_reuse_age_s = max(0.0, float(max_reuse_age_s))
        self._last_reference: list[dict[str, object]] = []
        self._last_destination: list[float] | None = None
        self._last_time_s = -float("inf")

    def stabilize(
        self,
        *,
        reference: Sequence[Mapping[str, object]],
        destination_state: Sequence[float] | None,
        ego_location: _PlannerLocation,
        ego_yaw_rad: float,
        stop_goal_active: bool,
        sim_time_s: float,
    ) -> tuple[list[dict[str, object]], list[float] | None, str]:
        current_reference = [dict(sample) for sample in list(reference or [])]
        current_destination = list(destination_state) if destination_state is not None else None
        if bool(stop_goal_active) or not self._last_reference or self._last_destination is None:
            self._accept(current_reference, current_destination, sim_time_s)
            return current_reference, current_destination, "reference_memory_accept"
        if not current_reference or current_destination is None:
            reused_reference, reused_destination = self._reusable_previous(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                sim_time_s=float(sim_time_s),
            )
            if reused_reference and reused_destination is not None:
                return reused_reference, reused_destination, "reference_memory_reuse_missing"
            return current_reference, current_destination, "reference_memory_missing"

        first_jump_m = self._point_jump_m(current_reference[0], self._last_reference[0])
        destination_jump_m = math.hypot(
            float(current_destination[0]) - float(self._last_destination[0]),
            float(current_destination[1]) - float(self._last_destination[1]),
        )
        if (
            first_jump_m > float(self.max_first_point_jump_m)
            or destination_jump_m > float(self.max_destination_jump_m)
        ):
            reused_reference, reused_destination = self._reusable_previous(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                sim_time_s=float(sim_time_s),
            )
            if reused_reference and reused_destination is not None:
                return reused_reference, reused_destination, (
                    "reference_memory_reuse_jump"
                    f":first={first_jump_m:.2f}:dest={destination_jump_m:.2f}"
                )
        self._accept(current_reference, current_destination, sim_time_s)
        return current_reference, current_destination, "reference_memory_accept"

    def _accept(
        self,
        reference: Sequence[Mapping[str, object]],
        destination_state: Sequence[float] | None,
        sim_time_s: float,
    ) -> None:
        self._last_reference = [dict(sample) for sample in list(reference or [])]
        self._last_destination = list(destination_state) if destination_state is not None else None
        self._last_time_s = float(sim_time_s)

    def reset(self) -> None:
        self._last_reference = []
        self._last_destination = None
        self._last_time_s = -float("inf")

    def _reusable_previous(
        self,
        *,
        ego_location: _PlannerLocation,
        ego_yaw_rad: float,
        sim_time_s: float,
    ) -> tuple[list[dict[str, object]], list[float] | None]:
        if float(sim_time_s) - float(self._last_time_s) > float(self.max_reuse_age_s):
            return [], None
        cos_h = math.cos(float(ego_yaw_rad))
        sin_h = math.sin(float(ego_yaw_rad))
        kept: list[dict[str, object]] = []
        for sample in list(self._last_reference or []):
            x_m = float(sample.get("x_ref_m", sample.get("x", ego_location.x)))
            y_m = float(sample.get("y_ref_m", sample.get("y", ego_location.y)))
            dx_m = x_m - float(ego_location.x)
            dy_m = y_m - float(ego_location.y)
            forward_m = dx_m * cos_h + dy_m * sin_h
            if forward_m >= -0.25:
                kept.append(dict(sample))
        if len(kept) < 2 or self._last_destination is None:
            return [], None
        return kept, list(self._last_destination)

    @staticmethod
    def _point_jump_m(first: Mapping[str, object], second: Mapping[str, object]) -> float:
        return math.hypot(
            float(first.get("x_ref_m", first.get("x", 0.0))) - float(second.get("x_ref_m", second.get("x", 0.0))),
            float(first.get("y_ref_m", first.get("y", 0.0))) - float(second.get("y_ref_m", second.get("y", 0.0))),
        )


class _Mode2TrajectoryMemory:
    """Continuity gate for MPC outputs in OpenCDA-reference tracking mode."""

    def __init__(
        self,
        *,
        max_accel_jump_mps2: float = 1.2,
        max_steer_jump_rad: float = 0.12,
        blend_alpha: float = 0.45,
        max_reuse_age_s: float = 0.5,
    ) -> None:
        self.max_accel_jump_mps2 = max(0.0, float(max_accel_jump_mps2))
        self.max_steer_jump_rad = max(0.0, float(max_steer_jump_rad))
        self.blend_alpha = min(1.0, max(0.0, float(blend_alpha)))
        self.max_reuse_age_s = max(0.0, float(max_reuse_age_s))
        self._last_control: PlannerControl | None = None
        self._last_accel_mps2 = 0.0
        self._last_steer_rad = 0.0
        self._last_time_s = -float("inf")

    def accept_or_blend(
        self,
        *,
        control: PlannerControl,
        accel_mps2: float,
        steer_rad: float,
        control_factory: Any,
        sim_time_s: float,
    ) -> tuple[PlannerControl, float, float, str]:
        if self._last_control is None:
            self._accept(control, accel_mps2, steer_rad, sim_time_s)
            return control, float(accel_mps2), float(steer_rad), "trajectory_memory_accept"
        accel_jump = abs(float(accel_mps2) - float(self._last_accel_mps2))
        steer_jump = abs(float(steer_rad) - float(self._last_steer_rad))
        if accel_jump > float(self.max_accel_jump_mps2) or steer_jump > float(self.max_steer_jump_rad):
            alpha = float(self.blend_alpha)
            blended_accel = (1.0 - alpha) * float(self._last_accel_mps2) + alpha * float(accel_mps2)
            blended_steer = (1.0 - alpha) * float(self._last_steer_rad) + alpha * float(steer_rad)
            blended_control = control_factory(float(blended_accel), float(blended_steer))
            self._accept(blended_control, blended_accel, blended_steer, sim_time_s)
            return (
                blended_control,
                float(blended_accel),
                float(blended_steer),
                f"trajectory_memory_blend:accel={accel_jump:.2f}:steer={steer_jump:.2f}",
            )
        self._accept(control, accel_mps2, steer_rad, sim_time_s)
        return control, float(accel_mps2), float(steer_rad), "trajectory_memory_accept"

    def reuse_if_fresh(
        self,
        *,
        sim_time_s: float,
        stop_goal_active: bool,
        control_factory: Any,
    ) -> tuple[PlannerControl | None, float, float, str]:
        if self._last_control is None:
            return None, 0.0, 0.0, ""
        if float(sim_time_s) - float(self._last_time_s) > float(self.max_reuse_age_s):
            return None, 0.0, 0.0, ""
        if bool(stop_goal_active):
            accel_mps2 = min(float(self._last_accel_mps2), -0.5)
            control = control_factory(float(accel_mps2), float(self._last_steer_rad))
            self._accept(control, accel_mps2, self._last_steer_rad, sim_time_s)
            return control, float(accel_mps2), float(self._last_steer_rad), "trajectory_memory_reuse_stop"
        return (
            self._last_control,
            float(self._last_accel_mps2),
            float(self._last_steer_rad),
            "trajectory_memory_reuse",
        )

    def _accept(
        self,
        control: PlannerControl,
        accel_mps2: float,
        steer_rad: float,
        sim_time_s: float,
    ) -> None:
        self._last_control = control
        self._last_accel_mps2 = float(accel_mps2)
        self._last_steer_rad = float(steer_rad)
        self._last_time_s = float(sim_time_s)

    def reset(self) -> None:
        self._last_control = None
        self._last_accel_mps2 = 0.0
        self._last_steer_rad = 0.0
        self._last_time_s = -float("inf")


class _OpenCDAStyleReferenceConditioner:
    """OpenCDA LocalPlanner-inspired conditioning before MPC tracking."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_node_spacing_m: float = 0.45,
        ego_anchor_forward_m: float = 0.35,
        max_lateral_accel_mps2: float = 3.0,
        min_speed_mps: float = 0.6,
        turn_min_speed_mps: float = 0.45,
        turn_speed_cap_mps: float = 1.4,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_node_spacing_m = max(0.05, float(min_node_spacing_m))
        self.ego_anchor_forward_m = max(0.0, float(ego_anchor_forward_m))
        self.max_lateral_accel_mps2 = max(0.1, float(max_lateral_accel_mps2))
        self.min_speed_mps = max(0.0, float(min_speed_mps))
        self.turn_min_speed_mps = max(0.0, float(turn_min_speed_mps))
        self.turn_speed_cap_mps = max(0.1, float(turn_speed_cap_mps))
        self._history = deque(maxlen=3)
        self._last_mode_key = ""

    def reset(self) -> None:
        self._history.clear()
        self._last_mode_key = ""

    def condition(
        self,
        *,
        reference: Sequence[Mapping[str, object]],
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        current_lane_id: int,
        horizon_steps: int,
        step_distance_m: float,
        target_speed_mps: float,
        decision: str,
        stop_goal_active: bool,
    ) -> tuple[list[dict[str, object]], str]:
        if not bool(self.enabled):
            return [dict(sample) for sample in list(reference or [])], ""
        normalized_decision = str(decision or "").strip().lower()
        stop_like = bool(stop_goal_active) or normalized_decision in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }
        if bool(stop_like):
            return [dict(sample) for sample in list(reference or [])], "opencda_style_conditioner_skipped_stop"
        mode_key = str(normalized_decision or "lane_follow")
        if self._last_mode_key and self._last_mode_key != str(mode_key):
            self._history.clear()
        self._last_mode_key = str(mode_key)

        raw = [dict(sample) for sample in list(reference or [])]
        if len(raw) < 2:
            return raw, "opencda_style_conditioner_too_few_samples"

        reasons: list[str] = []
        nodes: list[dict[str, object]] = []
        anchor_x = float(ego_x_m) + float(self.ego_anchor_forward_m) * math.cos(float(ego_heading_rad))
        anchor_y = float(ego_y_m) + float(self.ego_anchor_forward_m) * math.sin(float(ego_heading_rad))
        nodes.append({
            "x_ref_m": float(anchor_x),
            "y_ref_m": float(anchor_y),
            "x": float(anchor_x),
            "y": float(anchor_y),
            "heading_rad": float(ego_heading_rad),
            "lane_id": int(current_lane_id),
            "lane_width_m": 3.5,
            "speed_ref_mps": float(target_speed_mps),
            "v_ref_mps": float(target_speed_mps),
            "speed_mps": float(target_speed_mps),
        })

        for sample in raw:
            try:
                x_m = float(sample.get("x_ref_m", sample.get("x", 0.0)))
                y_m = float(sample.get("y_ref_m", sample.get("y", 0.0)))
            except Exception:
                reasons.append("drop_bad_conditioner_sample")
                continue
            if not math.isfinite(x_m) or not math.isfinite(y_m):
                reasons.append("drop_nonfinite_conditioner_sample")
                continue
            forward_m = (
                math.cos(float(ego_heading_rad)) * (float(x_m) - float(ego_x_m))
                + math.sin(float(ego_heading_rad)) * (float(y_m) - float(ego_y_m))
            )
            if forward_m < -0.2:
                reasons.append("drop_behind_conditioner_sample")
                continue
            if nodes:
                prev = nodes[-1]
                prev_x = float(prev.get("x_ref_m", prev.get("x", x_m)))
                prev_y = float(prev.get("y_ref_m", prev.get("y", y_m)))
                if math.hypot(float(x_m) - prev_x, float(y_m) - prev_y) < float(self.min_node_spacing_m):
                    reasons.append("drop_duplicate_conditioner_sample")
                    continue
            cleaned = dict(sample)
            cleaned["x_ref_m"] = float(x_m)
            cleaned["y_ref_m"] = float(y_m)
            cleaned["x"] = float(x_m)
            cleaned["y"] = float(y_m)
            nodes.append(cleaned)

        if len(nodes) < 3:
            return raw, "opencda_style_conditioner_insufficient_nodes"

        try:
            x_nodes = [float(node.get("x_ref_m", node.get("x", 0.0))) for node in nodes]
            y_nodes = [float(node.get("y_ref_m", node.get("y", 0.0))) for node in nodes]
            spline = Spline2D(x_nodes, y_nodes)
        except Exception as exc:
            return raw, f"opencda_style_conditioner_spline_failed:{exc}"

        total_s = float(spline.s[-1]) if getattr(spline, "s", None) else 0.0
        if total_s <= 1.0e-3:
            return raw, "opencda_style_conditioner_zero_length"
        step_m = max(0.25, float(step_distance_m))
        turn_like = str(normalized_decision) in {
            "intersection_turn_left",
            "intersection_turn_right",
        }
        shaped: list[dict[str, object]] = []
        max_curvature = 0.0
        for index in range(max(1, int(horizon_steps))):
            s_m = min(float(total_s), float(index + 1) * float(step_m))
            try:
                x_m, y_m = spline.calc_position(float(s_m))
                yaw_rad = spline.calc_yaw(float(s_m))
                curvature = float(spline.calc_curvature(float(s_m)))
            except Exception:
                reasons.append("conditioner_sample_failed")
                break
            if x_m is None or y_m is None:
                reasons.append("conditioner_sample_out_of_range")
                break
            max_curvature = max(float(max_curvature), abs(float(curvature)))
            curvature_speed = math.sqrt(
                float(self.max_lateral_accel_mps2) / (abs(float(curvature)) + 1.0e-3)
            )
            raw_speed = self._speed_for_index(raw, index, float(target_speed_mps))
            cap = min(float(raw_speed), float(target_speed_mps), float(curvature_speed))
            if bool(turn_like):
                cap = min(float(cap), float(self.turn_speed_cap_mps))
                speed_mps = max(float(self.turn_min_speed_mps), float(cap))
            else:
                speed_mps = max(float(self.min_speed_mps), float(cap))
            lane_id = int(self._sample_lane_id(raw, index, int(current_lane_id)))
            lane_width = float(self._sample_lane_width(raw, index, 3.5))
            shaped.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(yaw_rad),
                "lane_id": int(lane_id),
                "lane_width_m": float(lane_width),
                "speed_ref_mps": float(speed_mps),
                "v_ref_mps": float(speed_mps),
                "speed_mps": float(speed_mps),
            })

        if len(shaped) < 2:
            return raw, "opencda_style_conditioner_empty_shaped"
        self._history.append(dict(shaped[0]))
        reasons.append(f"opencda_style_spline_curvature_speed:max_k={max_curvature:.3f}")
        return shaped, ";".join(dict.fromkeys(reasons))

    @staticmethod
    def _speed_for_index(
        reference: Sequence[Mapping[str, object]],
        index: int,
        default_speed_mps: float,
    ) -> float:
        if not reference:
            return float(default_speed_mps)
        sample = dict(reference[min(int(index), len(reference) - 1)])
        for key in ("speed_ref_mps", "v_ref_mps", "speed_mps", "v"):
            if key not in sample:
                continue
            try:
                value = float(sample[key])
            except Exception:
                continue
            if math.isfinite(value):
                return max(0.0, float(value))
        return float(default_speed_mps)

    @staticmethod
    def _sample_lane_id(
        reference: Sequence[Mapping[str, object]],
        index: int,
        default_lane_id: int,
    ) -> int:
        if not reference:
            return int(default_lane_id)
        sample = dict(reference[min(int(index), len(reference) - 1)])
        try:
            return int(float(sample.get("lane_id", default_lane_id)))
        except Exception:
            return int(default_lane_id)

    @staticmethod
    def _sample_lane_width(
        reference: Sequence[Mapping[str, object]],
        index: int,
        default_lane_width_m: float,
    ) -> float:
        if not reference:
            return float(default_lane_width_m)
        sample = dict(reference[min(int(index), len(reference) - 1)])
        try:
            return float(sample.get("lane_width_m", default_lane_width_m))
        except Exception:
            return float(default_lane_width_m)


class CPXMPCPlannerBridge:
    """Run the active CP-X planner with ROS input and custom-map boundaries."""

    def __init__(self, *, behavior_planner, route_manager, global_planner, mpc, control_buffer, mpc_feedback, behavior_runtime_cfg=None, config=None):
        """Receive the independent behavior and global-planning components and keep state between planning cycles."""
        self.behavior_planner = behavior_planner
        self.route_manager = route_manager
        self.global_planner = global_planner
        self.map_planner = global_planner
        self.reference_map = global_planner

        self.mpc = mpc
        self.control_buffer = control_buffer
        self.mpc_feedback = mpc_feedback
        self.behavior_runtime_cfg = dict(behavior_runtime_cfg or {})
        self._temporary_destination_state = None
        self._previous_lane_center_reference = []
        self._lane_reference_freeze_count = 0
        self._stop_release_temp_smooth_until_sim_time_s = 0.0
        self._last_accel_mps2 = 0.0
        self._last_steer_rad = 0.0
        self._warned = False
        self._last_planned_trajectory = []
        self.last_output = None
        self.last_debug = {}
        self.last_destination_state = None
        self.last_reference_debug = {}
        self.last_behavior_command = {}
        self.ego_z_m = 0.0
        self.active_mpc_cost_profile = "lane_follow"
        self.requested_mpc_cost_profile = "lane_follow"
        self.mpc_cost_profile_active_since_s = 0.0
        self.mpc_cost_profile_switch_reason = "initial"

        self.config = dict(config or {})
        self.target_speed_mps = float(self.config.get("target_speed_mps", 8.0))
        self.lookahead_m = float(self.config.get("lookahead_m", 18.0))
        self.min_front_gap_m = float(self.config.get("min_front_gap_m", 8.0))
        self.max_mpc_obstacles = max(0, int(self.config.get("max_mpc_obstacles", 4)))
        self.debug = bool(self.config.get("debug", True))
        self.fallback_policy = str(self.config.get("fallback_policy", "emergency_stop")).strip().lower()
        self.fallback_policy_warning = ""
        self.global_planner_backend = "custom_admap_dijkstra"
        self.global_planner_backend_warning = ""
        self._scenario_manager = CPXScenarioManager(self.config)
        self._full_traffic_memory = _Mode2TrafficLightMemory(
            hold_unknown_s=float(self.config.get("full_traffic_unknown_hold_s", 0.25)),
            green_confirm_s=float(self.config.get("full_traffic_green_confirm_s", 0.5)),
        )
        self._full_reference_memory = _Mode2ReferenceMemory(max_first_point_jump_m=float(self.config.get("full_reference_memory_max_first_jump_m", 0.85)), max_destination_jump_m=float(self.config.get("full_reference_memory_max_destination_jump_m", 2.0)), max_reuse_age_s=float(self.config.get("full_reference_memory_max_reuse_age_s", 0.8)))
        self._full_trajectory_memory = _Mode2TrajectoryMemory(max_accel_jump_mps2=float(self.config.get("full_trajectory_memory_max_accel_jump_mps2", 0.9)), max_steer_jump_rad=float(self.config.get("full_trajectory_memory_max_steer_jump_rad", 0.08)), blend_alpha=float(self.config.get("full_trajectory_memory_blend_alpha", 0.35)), max_reuse_age_s=float(self.config.get("full_trajectory_memory_max_reuse_age_s", 0.5)))
        self._opencda_style_reference_conditioner = _OpenCDAStyleReferenceConditioner(enabled=bool(self.config.get("opencda_style_reference_conditioning_enabled", True)), min_node_spacing_m=float(self.config.get("opencda_style_reference_min_node_spacing_m", 0.45)), ego_anchor_forward_m=float(self.config.get("opencda_style_reference_ego_anchor_forward_m", 0.35)), max_lateral_accel_mps2=float(self.config.get("opencda_style_reference_max_lateral_accel_mps2", 3.0)), min_speed_mps=float(self.config.get("opencda_style_reference_min_speed_mps", 0.6)), turn_min_speed_mps=float(self.config.get("opencda_style_turn_min_speed_mps", 0.45)), turn_speed_cap_mps=float(self.config.get("opencda_style_turn_speed_cap_mps", 1.35)))
        self.safety_supervisor = SafetySupervisor(enabled=bool(self.config.get("safety_supervisor_enabled", True)), max_steer_delta=float(self.config.get("safety_max_steer_delta", 0.25)), max_throttle_delta=float(self.config.get("safety_max_throttle_delta", 0.45)), max_brake_delta=float(self.config.get("safety_max_brake_delta", 0.60)))
        self.full_lane_change_start_lock_s = max(0.0, float(self.config.get("full_lane_change_start_lock_s", 8.0)))
        self.full_dense_traffic_lane_change_lock_enabled = bool(self.config.get("full_dense_traffic_lane_change_lock_enabled", True))
        self.full_dense_traffic_object_count = max(0, int(self.config.get("full_dense_traffic_object_count", 8)))
        self.full_dense_traffic_risky_lane_count = max(0, int(self.config.get("full_dense_traffic_risky_lane_count", 2)))
        self.full_prepare_lane_change_reference_lock = bool(self.config.get("full_prepare_lane_change_reference_lock", True))
        self.full_allow_opportunistic_lane_change = bool(self.config.get("full_allow_opportunistic_lane_change", False))
        self.full_lane_follow_max_destination_lateral_m = max(0.0, float(self.config.get("full_lane_follow_max_destination_lateral_m", 1.2)))
        self.full_lane_follow_max_reference_first_lateral_m = max(0.0, float(self.config.get("full_lane_follow_max_reference_first_lateral_m", 0.65)))
        self.full_stop_max_destination_lateral_m = max(0.0, float(self.config.get("full_stop_max_destination_lateral_m", 1.0)))
        self.full_stop_max_reference_first_lateral_m = max(0.0, float(self.config.get("full_stop_max_reference_first_lateral_m", 0.55)))
        self.full_candidate_pipeline_enabled = bool(self.config.get("full_candidate_pipeline_enabled", True))
        self.full_candidate_reference_min_object_distance_m = max(0.0, float(self.config.get("full_candidate_reference_min_object_distance_m", 2.0)))
        self.strict_decision_ownership_enabled = bool(self.config.get("strict_decision_ownership_enabled", True))
        self.strict_explicit_fallback_speed_mps = max(0.0, float(self.config.get("strict_explicit_fallback_speed_mps", 0.8)))
        self.full_reference_stabilizer_min_forward_m = float(self.config.get("full_reference_stabilizer_min_forward_m", -0.25))
        self.full_reference_stabilizer_min_spacing_m = max(0.0, float(self.config.get("full_reference_stabilizer_min_spacing_m", 0.35)))
        self.full_reference_stabilizer_max_heading_step_rad = max(0.0, float(self.config.get("full_reference_stabilizer_max_heading_step_rad", 0.75)))
        self.strict_lane_follow_reference = bool(self.config.get("strict_lane_follow_reference", False))
        self.full_mpc_reference_stabilizer_enabled = bool(self.config.get("full_mpc_reference_stabilizer_enabled", True))
        self.strict_reference_validator_veto_enabled = bool(self.config.get("strict_reference_validator_veto_enabled", True))
        self.overspeed_guard_enabled = bool(self.config.get("overspeed_guard_enabled", False))
        self.overspeed_margin_mps = float(self.config.get("overspeed_margin_mps", 0.75))
        self.overspeed_brake_gain = float(self.config.get("overspeed_brake_gain", 0.10))
        self.overspeed_min_brake = float(self.config.get("overspeed_min_brake", 0.15))
        self.overspeed_max_brake = float(self.config.get("overspeed_max_brake", 0.55))
        self.low_speed_lateral_recovery_enabled = bool(self.config.get("low_speed_lateral_recovery_enabled", False))
        self.low_speed_lateral_recovery_speed_mps = float(self.config.get("low_speed_lateral_recovery_speed_mps", 0.6))
        self.low_speed_lateral_recovery_threshold_m = float(self.config.get("low_speed_lateral_recovery_threshold_m", 1.5))
        self.low_speed_lateral_recovery_target_speed_mps = float(self.config.get("low_speed_lateral_recovery_target_speed_mps", 1.2))
        self.low_speed_lateral_recovery_max_steer_rad = float(self.config.get("low_speed_lateral_recovery_max_steer_rad", 0.14))
        self.full_low_speed_launch_enabled = bool(self.config.get("full_low_speed_launch_enabled", True))
        self.full_low_speed_launch_speed_mps = float(self.config.get("full_low_speed_launch_speed_mps", 0.35))
        self.full_low_speed_launch_min_accel_mps2 = float(self.config.get("full_low_speed_launch_min_accel_mps2", 0.8))
        self._full_launch_start_s = None
        self._full_launch_start_xy = None
        self._full_last_behavior_mode_key = ""
        self._full_latched_stop_target = None
        self._full_latched_stop_state = "unknown"
        self._turn_latch_decision = ""
        self._turn_latch_until_sim_time_s = -float("inf")

        from cpx_planning.pipeline.planner_pipeline import CPXPlanningPipeline

        self.planning_pipeline = CPXPlanningPipeline(self)
        self._build_decision_record = build_decision_record

    def run_step(self, adapter_output):
        """Use the same bridge entry point as OpenCDA while receiving the adapter output from ROS."""
        self._latest_adapter_output = adapter_output
        planner_output = self.planning_pipeline.run_step()
        self.last_output = planner_output
        self.last_debug = planner_output.diagnostics_dict()
        return planner_output


    def _plan_behavior_and_reference(
        self,
        *,
        ego_location: _PlannerLocation,
        ego_yaw_rad: float,
        ego_speed_mps: float,
        speed_ref_mps: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        stop_goal_active: bool,
        cp_payload: Mapping[str, Any] | None = None,
    ):
        from cpx_planning.behavior_planner import (
            MpcReferenceGenerationContext,
            compute_temp_destination,
            generate_mpc_reference,
            select_reference_intent,
        )
        from cpx_planning.pipeline.candidate_evaluation import (
            evaluate_behavior_candidates,
        )
        from cpx_planning.pipeline.candidate_pipeline import (
            build_candidate_intents,
        )
        from cpx_planning.pipeline.speed_planner import build_speed_plan

        adapter_output = self._latest_adapter_output
        planner_input_frame = adapter_output.frame
        sim_time_s = float(planner_input_frame.planning.sim_time_s)
        self._current_sim_time_s = float(sim_time_s)
        self.last_adapter_output = adapter_output
        ego_pose = adapter_output.ego_pose
        current_state = adapter_output.current_state
        current_lane_id = int(adapter_output.current_lane_id)
        ego_waypoint = adapter_output.ego_waypoint
        lane_safety_scores = dict(adapter_output.lane_safety_scores)
        front_dist_by_lane = dict(adapter_output.front_distance_by_lane)
        route_points = list(adapter_output.route_points)
        route_context = planner_input_frame.planning.route
        route_optimal_lane_id = int(adapter_output.route_optimal_lane_id)
        route_reference_allowed = bool(adapter_output.route_reference_allowed)
        route_reference_gate_reason = str(adapter_output.route_reference_gate_reason)
        route_lane_change_allowed = bool(route_reference_allowed) and (
            "direct_fallback" not in str(route_reference_gate_reason)
        )
        from cpx_planning.pipeline.route_authorization import (
            authorize_route_lane_change,
        )

        lane_change_authorization = authorize_route_lane_change(
            route_lane_change_allowed=bool(route_lane_change_allowed),
            current_lane_id=int(current_lane_id),
            route_required_lane_id=int(route_optimal_lane_id),
            next_macro_maneuver=str(route_context.next_macro_maneuver),
            current_road_option=str(route_context.current_road_option),
            remaining_distance_m=float(route_context.remaining_distance_m),
            available_lane_ids=list(planner_input_frame.map_lane.allowed_lane_ids),
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
            preparation_start_distance_m=float(
                self.config.get("route_lane_change_preparation_start_distance_m", 45.0)
            ),
            latest_start_distance_m=float(
                self.config.get("route_lane_change_latest_start_distance_m", 12.0)
            ),
            target_safety_threshold=float(
                self.config.get("route_lane_change_target_safety_threshold", 0.65)
            ),
            require_adjacent=bool(self.config.get("route_lane_change_require_adjacent", True)),
        )
        route_lane_change_required = bool(lane_change_authorization.required_by_route)
        prediction_risky_lane_count = 0
        for risk in dict(planner_input_frame.prediction.lane_prediction_risks).values():
            risk_mapping = dict(risk) if isinstance(risk, Mapping) else {}
            if bool(risk_mapping.get("risk", False)):
                prediction_risky_lane_count += 1
        dense_traffic_active = (
            bool(self.full_dense_traffic_lane_change_lock_enabled)
            and (
                len(list(object_snapshots or [])) >= int(self.full_dense_traffic_object_count)
                or int(prediction_risky_lane_count) >= int(self.full_dense_traffic_risky_lane_count)
            )
        )
        start_lane_change_lock_active = (
            float(sim_time_s) <= float(self.full_lane_change_start_lock_s)
        )
        lane_change_authorized = bool(lane_change_authorization.allowed)
        opportunistic_lane_change_allowed = bool(route_lane_change_allowed) and (
            bool(lane_change_authorized)
            or (
                bool(self.full_allow_opportunistic_lane_change)
                and not bool(start_lane_change_lock_active)
                and not bool(dense_traffic_active)
            )
        )
        lane_change_gate_reason = ""
        if bool(route_lane_change_allowed) and not bool(opportunistic_lane_change_allowed):
            reasons = []
            if bool(start_lane_change_lock_active):
                reasons.append("start_lock")
            if bool(dense_traffic_active):
                reasons.append("dense_traffic")
            if not bool(lane_change_authorized):
                reasons.append(str(lane_change_authorization.reason))
            lane_change_gate_reason = "opportunistic_lane_change_suppressed:" + "+".join(reasons)
        signal_context = dict(adapter_output.signal_context)
        source_quality = dict(adapter_output.source_quality)
        raw_stop_target = (
            planner_input_frame.planning.traffic_control.stop_target.as_dict()
            if planner_input_frame.planning.traffic_control.stop_target.active
            else None
        )
        filtered_traffic_state, filtered_stop_target, full_traffic_memory_reason = (
            self._full_traffic_memory.update(
                state=str(planner_input_frame.planning.traffic_control.signal_state),
                stop_target=raw_stop_target,
                sim_time_s=float(sim_time_s),
            )
        )
        filtered_stop_target, stop_latch_reason = self._full_latched_stop_target_for_signal(
            traffic_state=str(filtered_traffic_state),
            stop_target=(
                dict(filtered_stop_target)
                if isinstance(filtered_stop_target, Mapping)
                else None
            ),
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            current_lane_id=int(current_lane_id),
        )
        if str(stop_latch_reason):
            full_traffic_memory_reason = (
                f"{full_traffic_memory_reason};{stop_latch_reason}"
                if str(full_traffic_memory_reason)
                else str(stop_latch_reason)
            )
        traffic_stop_forward_m, traffic_stop_target_reliable = self._stop_target_forward_m(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            stop_target=(
                dict(filtered_stop_target)
                if isinstance(filtered_stop_target, Mapping)
                else None
            ),
            fallback_destination_state=[],
        )
        scenario_decision = self._scenario_manager.update(
            traffic_state=str(filtered_traffic_state),
            stop_target=(
                dict(filtered_stop_target)
                if isinstance(filtered_stop_target, Mapping)
                else None
            ),
            stop_forward_m=float(traffic_stop_forward_m),
            stop_target_reliable=bool(traffic_stop_target_reliable),
            ego_speed_mps=float(ego_speed_mps),
            ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
            current_road_option=str(route_context.current_road_option),
            next_macro_maneuver=str(route_context.next_macro_maneuver),
            sim_time_s=float(sim_time_s),
        )
        behavior_traffic_state = str(scenario_decision.behavior_signal_state)
        behavior_stop_target = (
            dict(scenario_decision.behavior_stop_target)
            if isinstance(scenario_decision.behavior_stop_target, Mapping)
            else None
        )
        traffic_stop_commit_distance_m = float(
            scenario_decision.traffic_stop_commit_distance_m
        )
        traffic_stop_approach_speed_cap_mps = float(
            scenario_decision.speed_cap_mps
            if scenario_decision.speed_cap_mps is not None
            else self.target_speed_mps
        )
        traffic_stop_approach_reason = str(scenario_decision.reason)
        filtered_signal_context = dict(signal_context or {})
        filtered_signal_context["raw_signal_state"] = str(
            planner_input_frame.planning.traffic_control.signal_state
        )
        filtered_signal_context["signal_state"] = str(filtered_traffic_state)
        filtered_signal_context["behavior_signal_state"] = str(behavior_traffic_state)
        filtered_signal_context["traffic_stop_forward_m"] = float(traffic_stop_forward_m)
        filtered_signal_context["traffic_stop_commit_distance_m"] = float(traffic_stop_commit_distance_m)
        if str(traffic_stop_approach_reason):
            filtered_signal_context["traffic_stop_approach_reason"] = str(traffic_stop_approach_reason)
        if str(full_traffic_memory_reason):
            filtered_signal_context["traffic_memory_reason"] = str(full_traffic_memory_reason)
        mpc_feedback = self.mpc_feedback.candidate_feedback(
            current_time_s=float(sim_time_s)
        )
        if bool(lane_change_authorized):
            candidate_lane_ids = [
                int(current_lane_id),
                int(lane_change_authorization.target_lane_id),
            ]
        elif bool(opportunistic_lane_change_allowed):
            candidate_lane_ids = list(planner_input_frame.map_lane.allowed_lane_ids)
        else:
            candidate_lane_ids = [int(current_lane_id)]
        candidate_frame = evaluate_behavior_candidates(
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
            ego_lane_id=int(current_lane_id),
            selected_lane_id=int(current_lane_id),
            available_lane_ids=list(candidate_lane_ids),
            route_optimal_lane_id=int(route_optimal_lane_id),
            mode="INTERSECTION" if bool(planner_input_frame.map_lane.in_junction) else "NORMAL",
            mpc_feedback_blocked_lane_ids=list(
                mpc_feedback.get("blocked_lane_ids", []) or []
            ),
            mpc_feedback_weight=float(self.config.get("mpc_feedback_candidate_weight", 80.0)),
        )
        preferred_target_lane_id = (
            int(lane_change_authorization.target_lane_id)
            if bool(lane_change_authorized)
            else int(candidate_frame.selected.target_lane_id)
            if bool(opportunistic_lane_change_allowed)
            else int(current_lane_id)
        )

        command = self.behavior_planner.update(
            lane_safety_scores=lane_safety_scores,
            ego_lane_id=int(current_lane_id),
            selected_lane_id=int(current_lane_id),
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION" if bool(planner_input_frame.map_lane.in_junction) else "NORMAL",
            route_optimal_lane_id=int(route_optimal_lane_id),
            next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            front_obstacle_distance_by_lane=front_dist_by_lane,
            current_time_s=float(sim_time_s),
            wall_time_s=float(sim_time_s),
            traffic_signal_state=str(behavior_traffic_state),
            traffic_stop_target=(
                dict(behavior_stop_target)
                if isinstance(behavior_stop_target, Mapping)
                else None
            ),
            traffic_signal_context=dict(filtered_signal_context or {}),
            ego_speed_mps=float(ego_speed_mps),
            ego_max_deceleration_mps2=abs(float(self.mpc.constraints.min_acceleration_mps2)),
            ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
            ego_position_xy=(float(ego_location.x), float(ego_location.y)),
            global_route_points=route_points,
            nearest_front_obstacles_by_lane={},
            lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
            preferred_target_lane_id=int(preferred_target_lane_id),
            lane_closure_messages=list(planner_input_frame.cp_messages.lane_closures),
        )
        decision = str(command.get("decision", "lane_follow"))
        target_lane_id = int(command.get("target_lane_id", current_lane_id) or current_lane_id)
        lc_state = str(command.get("lc_state", "LANE_KEEP"))
        behavior_override_reason = ""
        scenario_speed_cap_active = (
            scenario_decision.speed_cap_mps is not None
            and float(scenario_decision.speed_cap_mps) < float(self.target_speed_mps)
        )
        if (
            bool(scenario_speed_cap_active)
            and str(decision) not in {"stop_at_intersection", "stop_sign", "emergency_brake"}
        ):
            speed_ref_mps = min(float(speed_ref_mps), float(traffic_stop_approach_speed_cap_mps))
            behavior_override_reason = str(traffic_stop_approach_reason)
        route_turn_decision = self._route_option_turn_decision(
            current_road_option=str(route_context.current_road_option),
            next_macro_maneuver=str(route_context.next_macro_maneuver),
        )
        route_turn_prepare_decision = ""
        if (
            not str(route_turn_decision)
            and bool(self.config.get("full_intersection_turn_prepare_enabled", False))
        ):
            route_turn_prepare_decision = self._route_lookahead_turn_decision(
                ego_location=ego_location,
                ego_heading_rad=float(ego_yaw_rad),
                route_points=route_points,
            )
        if (
            not bool(opportunistic_lane_change_allowed)
            and str(decision) in {"lane_change_left", "lane_change_right"}
        ):
            decision = "lane_follow"
            target_lane_id = int(current_lane_id)
            lc_state = "LANE_KEEP"
            behavior_override_reason = (
                str(lane_change_gate_reason)
                if str(lane_change_gate_reason)
                else "lane_change_suppressed_without_valid_route"
            )
            reset_lane_change = getattr(self.behavior_planner, "_reset_lane_change_state", None)
            if callable(reset_lane_change):
                reset_lane_change(reason=str(behavior_override_reason))
        if (
            bool(self.full_prepare_lane_change_reference_lock)
            and str(lc_state).upper().startswith("PREPARE_LANE_CHANGE")
        ):
            decision = "lane_follow"
            target_lane_id = int(current_lane_id)
            lc_state = "LANE_KEEP"
            behavior_override_reason = (
                str(behavior_override_reason) + ";"
                if str(behavior_override_reason)
                else ""
            ) + "prepare_lane_change_reference_locked_to_current_lane"
            reset_lane_change = getattr(self.behavior_planner, "_reset_lane_change_state", None)
            if callable(reset_lane_change):
                reset_lane_change(reason="prepare_lane_change_reference_locked")
        if (
            str(decision) in {"lane_change_left", "lane_change_right"}
            and not bool(lane_change_authorized)
        ):
            decision = "lane_follow"
            target_lane_id = int(current_lane_id)
            lc_state = "LANE_KEEP"
            behavior_override_reason = (
                str(behavior_override_reason) + ";"
                if str(behavior_override_reason)
                else ""
            ) + f"lane_change_without_authorization:{lane_change_authorization.reason}"
            reset_lane_change = getattr(self.behavior_planner, "_reset_lane_change_state", None)
            if callable(reset_lane_change):
                reset_lane_change(reason="lane_change_without_authorization")
        scenario_behavior_override = str(scenario_decision.behavior_override_decision or "")
        if str(scenario_behavior_override):
            decision = str(scenario_behavior_override)
            target_lane_id = int(current_lane_id)
            lc_state = (
                str(scenario_decision.behavior_override_lc_state)
                if str(scenario_decision.behavior_override_lc_state)
                else "LANE_KEEP"
            )
            if scenario_decision.speed_cap_mps is not None:
                speed_ref_mps = min(float(speed_ref_mps), float(scenario_decision.speed_cap_mps))
            behavior_override_reason = (
                str(behavior_override_reason) + ";"
                if str(behavior_override_reason)
                else ""
            ) + str(scenario_decision.reason)
        stop_goal_active = bool(stop_goal_active or scenario_decision.stop_goal_active)
        if (
            (str(route_turn_decision) or str(route_turn_prepare_decision))
            and not str(scenario_behavior_override)
            and str(decision) not in {"stop_at_intersection", "stop_sign", "emergency_brake"}
        ):
            decision = str(route_turn_decision or route_turn_prepare_decision)
            target_lane_id = int(current_lane_id)
            lc_state = (
                "INTERSECTION_TURN_LEFT"
                if str(decision).endswith("_left")
                else "INTERSECTION_TURN_RIGHT"
            )
            speed_ref_mps = min(
                float(speed_ref_mps),
                float(
                    self.config.get(
                        "full_intersection_turn_prepare_speed_cap_mps",
                        self.config.get("full_intersection_turn_speed_cap_mps", 2.2),
                    )
                ),
            )
            behavior_override_reason = (
                str(behavior_override_reason) + ";"
                if str(behavior_override_reason)
                else ""
            ) + (
                f"route_option_driven_behavior:{route_context.current_road_option}"
                if str(route_turn_decision)
                else "route_lookahead_prepare_turn"
            )
        turn_latch_reason = ""
        if (
            str(decision) not in {"stop_at_intersection", "stop_sign", "emergency_brake"}
            and not bool(scenario_decision.turn_latched)
        ):
            decision, lc_state, speed_ref_mps, turn_latch_reason = self._apply_turn_direction_latch(
                decision=str(decision),
                lc_state=str(lc_state),
                speed_ref_mps=float(speed_ref_mps),
                current_road_option=str(route_context.current_road_option),
                next_macro_maneuver=str(route_context.next_macro_maneuver),
                ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
                sim_time_s=float(sim_time_s),
            )
            if str(turn_latch_reason):
                target_lane_id = int(current_lane_id)
                behavior_override_reason = (
                    str(behavior_override_reason) + ";"
                    if str(behavior_override_reason)
                    else ""
                ) + str(turn_latch_reason)
        speed_plan = build_speed_plan(
            scenario_decision=scenario_decision,
            behavior_decision=str(decision),
            requested_speed_mps=float(speed_ref_mps),
            ego_speed_mps=float(ego_speed_mps),
            config=dict(self.config),
        )
        speed_ref_mps = float(speed_plan.target_speed_mps)
        stop_goal_active = bool(stop_goal_active or speed_plan.stop_goal_active)
        planner_mode = "INTERSECTION" if bool(planner_input_frame.map_lane.in_junction) else "NORMAL"
        self._apply_mpc_cost_profile(
            behavior=str(decision),
            planner_lc_state=str(lc_state),
            planner_mode=str(planner_mode),
            next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            sim_time_s=float(sim_time_s),
        )

        base_temporary_destination_state = (
            list(self._temporary_destination_state)
            if self._temporary_destination_state is not None
            else None
        )
        self._temporary_destination_state = compute_temp_destination(
            map_planner=self.reference_map,
            ego_pose=ego_pose,
            target_lane_id=int(target_lane_id),
            decision=str(decision),
            lookahead_m=float(self.lookahead_m),
            target_v_mps=float(speed_ref_mps),
            global_route_points=route_points,
            mode_reference_xy=(
                None
                if self._temporary_destination_state is None
                else (
                    float(self._temporary_destination_state[0]),
                    float(self._temporary_destination_state[1]),
                )
            ),
            prev_mode=(
                None
                if self._temporary_destination_state is None or len(self._temporary_destination_state) < 6
                else float(self._temporary_destination_state[5])
            ),
            prev_road_id=(
                None
                if self._temporary_destination_state is None or len(self._temporary_destination_state) < 7
                else int(self._temporary_destination_state[6])
            ),
            prev_entered_intersection=(
                False
                if self._temporary_destination_state is None or len(self._temporary_destination_state) < 8
                else bool(float(self._temporary_destination_state[7]) > 0.5)
            ),
            next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            mode_override=str(planner_mode),
            follow_global_route_lane=bool(
                route_reference_allowed and planner_input_frame.map_lane.in_junction
            ),
        )

        reference_intent = select_reference_intent(
            behavior_decision=str(decision),
            planner_fsm_state=str(lc_state),
            ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
            reference_target_lane_id=int(target_lane_id),
            current_lane_id=int(current_lane_id),
            route_optimal_lane_id=int(route_optimal_lane_id),
            global_route_reference_allowed=bool(route_reference_allowed),
            traffic_control_lane_lock_active=False,
        )
        ref_context = MpcReferenceGenerationContext(
            map_planner=self.reference_map,
            ego_pose=ego_pose,
            ego_state=current_state,
            active_global_route_points=route_points,
            previous_lane_center_reference=self._previous_lane_center_reference,
            behavior_runtime_cfg=self.behavior_runtime_cfg,
            reference_intent=reference_intent,
            current_applied_behavior=str(decision),
            cached_planner_lc_state=str(lc_state),
            reference_target_lane_id=int(target_lane_id),
            current_lane_id=int(current_lane_id),
            global_route_reference_allowed=bool(route_reference_allowed),
            global_route_reference_gate_reason=str(route_reference_gate_reason),
            should_follow_global_route_lane_for_reference=bool(
                reference_intent.follow_global_route_lane
            ),
            traffic_control_lane_lock_active=False,
            final_goal_stop_active=False,
            stop_target_state=None,
            follow_target_state=None,
            current_temp_reference_xy=(
                float(self._temporary_destination_state[0]),
                float(self._temporary_destination_state[1]),
            ),
            current_temp_mode_value=(
                float(self._temporary_destination_state[5])
                if len(self._temporary_destination_state) >= 6 else 0.0
            ),
            current_temp_road_id=(
                int(self._temporary_destination_state[6])
                if len(self._temporary_destination_state) >= 7 else None
            ),
            current_temp_entered_intersection=(
                bool(float(self._temporary_destination_state[7]) > 0.5)
                if len(self._temporary_destination_state) >= 8 else False
            ),
            active_reference_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            current_temp_mode_str=str(planner_mode),
            lane_reference_speed_mps=max(
                1.0,
                float(ego_speed_mps),
                abs(float(speed_ref_mps)),
            ),
            lane_reference_step_distance_m=max(
                0.5,
                float(self.mpc.dt_s)
                * max(1.0, float(ego_speed_mps), abs(float(speed_ref_mps))),
            ),
            mpc_horizon_steps=int(self.mpc.horizon_steps),
            mpc_dt_s=float(self.mpc.dt_s),
            temporary_destination_state=self._temporary_destination_state,
            lane_reference_freeze_count=int(self._lane_reference_freeze_count),
            sim_time_s=float(sim_time_s),
            stop_release_temp_smooth_until_sim_time_s=float(self._stop_release_temp_smooth_until_sim_time_s),
        )
        ref_output = generate_mpc_reference(ref_context)
        local_lane_center_reference = [
            dict(sample) for sample in list(ref_output.local_lane_center_reference or [])
        ]
        self._temporary_destination_state = list(ref_output.temporary_destination_state or self._temporary_destination_state)
        self._lane_reference_freeze_count = int(ref_output.lane_reference_freeze_count)
        reference_debug = dict(ref_output.mpc_reference_result.trace.as_trace_fields())
        reference_debug.update(planner_input_frame.trace_fields())
        reference_debug.update({
            "stage": reference_debug.get("reference_pipeline_stage", ""),
            "intent_mode": reference_debug.get("reference_pipeline_intent_mode", ""),
            "fallback_reason": str(ref_output.last_reference_fallback_reason),
            "reference_source": "behavior_reference_pipeline",
            "route_reference_allowed": bool(route_reference_allowed),
            "route_reference_gate_reason": str(route_reference_gate_reason),
            "route_lane_change_allowed": bool(route_lane_change_allowed),
            "opportunistic_lane_change_allowed": bool(opportunistic_lane_change_allowed),
            "lane_change_gate_reason": str(lane_change_gate_reason),
            "route_lane_change_required": bool(route_lane_change_required),
            **dict(lane_change_authorization.as_debug_fields()),
            "behavior_override_reason": str(behavior_override_reason),
            "turn_latch_reason": str(turn_latch_reason),
            "route_current_road_option": str(route_context.current_road_option),
            "route_next_macro_maneuver": str(route_context.next_macro_maneuver),
            "traffic_memory_reason": str(full_traffic_memory_reason),
            "traffic_stop_forward_m": float(traffic_stop_forward_m),
            "traffic_stop_commit_distance_m": float(traffic_stop_commit_distance_m),
            "traffic_stop_approach_reason": str(traffic_stop_approach_reason),
            **dict(speed_plan.as_debug_fields()),
            "traffic_signal_state_raw": str(planner_input_frame.planning.traffic_control.signal_state),
            "traffic_signal_state_filtered": str(filtered_traffic_state),
            "candidate_evaluation_summary": str(candidate_frame.summary()),
            "candidate_selected_decision": str(candidate_frame.selected.decision),
            "candidate_selected_lane_id": int(candidate_frame.selected.target_lane_id),
            "candidate_selected_cost": float(candidate_frame.selected.total_cost),
            "mpc_feedback_summary": str(mpc_feedback.get("summary", "")),
            "mpc_feedback_blocked_lane_ids": json.dumps(
                list(mpc_feedback.get("blocked_lane_ids", []) or []),
                default=str,
            ),
            "prediction_trajectories": dict(
                planner_input_frame.prediction.obstacle_future_trajectories
            ),
        })
        reference_debug.update(scenario_decision.as_debug_fields())
        reference_debug.update(source_quality)
        if bool(self.full_candidate_pipeline_enabled):
            traffic_stop_active = bool(scenario_decision.stop_goal_active)
            candidate_intents = build_candidate_intents(
                selected_decision=str(decision),
                selected_target_lane_id=int(target_lane_id),
                current_lane_id=int(current_lane_id),
                target_speed_mps=float(speed_ref_mps),
                candidate_lane_ids=list(candidate_lane_ids),
                lane_safety_scores=lane_safety_scores,
                lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
                stop_goal_active=bool(stop_goal_active or scenario_decision.stop_goal_active),
                traffic_stop_active=bool(traffic_stop_active),
                lane_change_authorized=bool(lane_change_authorized),
                lane_change_authorized_target_lane_id=int(
                    lane_change_authorization.target_lane_id
                    if lane_change_authorized
                    else 0
                ),
                allow_lane_change_candidates=bool(opportunistic_lane_change_allowed),
                stop_target=(
                    dict(behavior_stop_target)
                    if isinstance(behavior_stop_target, Mapping)
                    else None
                ),
            )
            (
                decision,
                target_lane_id,
                speed_ref_mps,
                local_lane_center_reference,
                self._temporary_destination_state,
                selected_candidate_debug,
            ) = self._select_candidate_reference_for_mpc(
                candidate_intents=candidate_intents,
                baseline_decision=str(decision),
                baseline_lc_state=str(lc_state),
                baseline_target_lane_id=int(target_lane_id),
                baseline_speed_ref_mps=float(speed_ref_mps),
                baseline_destination_state=self._temporary_destination_state,
                baseline_reference=local_lane_center_reference,
                baseline_reference_debug=reference_debug,
                base_temporary_destination_state=base_temporary_destination_state,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                ego_pose=ego_pose,
                current_state=current_state,
                current_lane_id=int(current_lane_id),
                route_optimal_lane_id=int(route_optimal_lane_id),
                route_points=route_points,
                route_reference_allowed=bool(route_reference_allowed),
                route_reference_gate_reason=str(route_reference_gate_reason),
                planner_input_frame=planner_input_frame,
                planner_mode=str(planner_mode),
                object_snapshots=object_snapshots,
            )
            if str(decision) == "lane_follow":
                lc_state = "LANE_KEEP"
            if str(decision) in {"stop_at_intersection", "stop_sign", "emergency_brake"}:
                lc_state = "LANE_KEEP"
            reference_debug.update(selected_candidate_debug)
            reference_debug["candidate_pipeline_enabled"] = True
        else:
            reference_debug["candidate_pipeline_enabled"] = False
        # Keep OpenCDA aligned with the standalone planning runner: the
        # reference pipeline owns lane-follow trimming, blending, jump freezing,
        # and geometry guards. The optional strict lane-follow path below is an
        # experimental fallback and should stay disabled for normal testing.
        strict_lane_follow = (
            bool(self.strict_lane_follow_reference)
            and str(decision) == "lane_follow"
            and str(lc_state or "").upper() in {"IDLE", "LANE_KEEP"}
            and not bool(stop_goal_active)
        )
        if bool(strict_lane_follow):
            lane_follow_reference_source = str(
                self.config.get("lane_follow_reference_source", "auto")
            ).strip().lower()
            step_distance_m = max(
                1.0,
                float(self.mpc.dt_s) * max(float(speed_ref_mps), 3.0),
            )
            strict_reference = []
            strict_reference_source = "opencda_current_lane_center_strict"
            if lane_follow_reference_source in {"global_route", "route", "literal_route"}:
                strict_reference = self._route_aligned_reference_samples(
                    ego_location=ego_location,
                    ego_heading_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                    route_points=route_points,
                )
                strict_reference_source = "global_route_aligned_lane_follow"
            if not strict_reference:
                strict_reference = self._current_lane_center_reference_samples(
                    start_waypoint=ego_waypoint,
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                    route_points=route_points,
                )
            if self._reference_opposes_heading(
                reference_samples=strict_reference,
                ego_heading_rad=float(ego_yaw_rad),
                max_heading_error_rad=0.5 * math.pi,
            ) or self._reference_lateral_offset_too_large(
                reference_samples=strict_reference,
                ego_state=current_state,
                max_lateral_offset_m=float(
                    self.config.get("lane_follow_reference_max_initial_lateral_m", 1.75)
                ),
            ):
                route_reference = self._route_aligned_reference_samples(
                    ego_location=ego_location,
                    ego_heading_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=max(
                        1.0,
                        float(self.mpc.dt_s) * max(float(speed_ref_mps), 3.0),
                    ),
                    route_points=route_points,
                )
                if route_reference:
                    strict_reference = route_reference
                    strict_reference_source = "global_route_aligned_lane_follow"
            if strict_reference:
                from cpx_planning.behavior_planner.reference_pipeline import (
                    lane_center_destination_from_reference,
                )

                local_lane_center_reference = strict_reference
                self._temporary_destination_state = lane_center_destination_from_reference(
                    destination_state=self._temporary_destination_state,
                    lane_center_reference=local_lane_center_reference,
                    ego_state=current_state,
                    target_forward_m=float(
                        self.behavior_runtime_cfg.get(
                            "lane_follow_destination_reference_forward_m", 6.0
                        )
                    ),
                )
                self._lane_reference_freeze_count = 0
                reference_debug["reference_source"] = str(strict_reference_source)
                reference_debug["fallback_reason"] = (
                    f"{reference_debug.get('fallback_reason', '')}:"
                    if str(reference_debug.get("fallback_reason", ""))
                    else ""
                ) + str(strict_reference_source)
                reference_debug["stage"] = "strict_lane_follow"
        lateral_guard_reason = self._full_reference_lateral_guard_reason(
            decision=str(decision),
            lc_state=str(lc_state),
            stop_goal_active=bool(stop_goal_active),
            destination_state=self._temporary_destination_state,
            lane_center_reference=local_lane_center_reference,
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
        )
        if str(lateral_guard_reason):
            if bool(self.strict_reference_validator_veto_enabled):
                reference_debug["candidate_pipeline_selected_status"] = "infeasible"
                reference_debug["candidate_pipeline_selected_reason"] = (
                    str(reference_debug.get("candidate_pipeline_selected_reason", "")) + ";"
                    if str(reference_debug.get("candidate_pipeline_selected_reason", ""))
                    else ""
                ) + "strict_reference_veto:" + str(lateral_guard_reason)
                reference_debug["fallback_reason"] = (
                    f"{reference_debug.get('fallback_reason', '')}:"
                    if str(reference_debug.get("fallback_reason", ""))
                    else ""
                ) + "strict_reference_veto:" + str(lateral_guard_reason)
            else:
                step_distance_m = max(
                    0.5,
                    float(self.mpc.dt_s)
                    * max(1.0, min(float(speed_ref_mps), float(self.target_speed_mps))),
                )
                guarded_reference = self._current_lane_center_reference_samples(
                    start_waypoint=ego_waypoint,
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                    route_points=route_points,
                )
                if guarded_reference:
                    from cpx_planning.behavior_planner.reference_pipeline import (
                        lane_center_destination_from_reference,
                    )

                    local_lane_center_reference = list(guarded_reference)
                    target_forward_m = float(
                        self.config.get(
                            "full_stop_guard_destination_forward_m",
                            6.0 if bool(stop_goal_active) else 8.0,
                        )
                    )
                    self._temporary_destination_state = lane_center_destination_from_reference(
                        destination_state=self._temporary_destination_state,
                        lane_center_reference=local_lane_center_reference,
                        ego_state=current_state,
                        target_forward_m=float(target_forward_m),
                    )
                    self._lane_reference_freeze_count = 0
                    reference_debug["reference_source"] = "current_lane_center_lateral_guard"
                    reference_debug["fallback_reason"] = (
                        f"{reference_debug.get('fallback_reason', '')}:"
                        if str(reference_debug.get("fallback_reason", ""))
                        else ""
                    ) + str(lateral_guard_reason)
                    reference_debug["stage"] = "lateral_guard"
                    reference_debug["reference_pipeline_follow_global_route_lane"] = 0
        reference_debug["reference_lateral_guard_reason"] = str(lateral_guard_reason)
        opencda_conditioning_reason = ""
        if local_lane_center_reference:
            step_distance_m = max(
                0.35,
                float(self.mpc.dt_s)
                * max(0.6, min(float(speed_ref_mps), float(self.target_speed_mps))),
            )
            conditioned_reference, opencda_conditioning_reason = (
                self._opencda_style_reference_conditioner.condition(
                    reference=local_lane_center_reference,
                    ego_x_m=float(ego_location.x),
                    ego_y_m=float(ego_location.y),
                    ego_heading_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                    target_speed_mps=float(speed_ref_mps),
                    decision=str(decision),
                    stop_goal_active=bool(stop_goal_active),
                )
            )
            if conditioned_reference:
                from cpx_planning.behavior_planner.reference_pipeline import (
                    lane_center_destination_from_reference,
                )

                local_lane_center_reference = [dict(sample) for sample in conditioned_reference]
                target_forward_m = float(
                    self.config.get(
                        "opencda_style_reference_destination_forward_m",
                        7.0 if str(decision) in {"intersection_turn_left", "intersection_turn_right"} else 8.0,
                    )
                )
                conditioned_destination = lane_center_destination_from_reference(
                    destination_state=self._temporary_destination_state,
                    lane_center_reference=local_lane_center_reference,
                    ego_state=current_state,
                    target_forward_m=float(target_forward_m),
                )
                if conditioned_destination is not None:
                    self._temporary_destination_state = list(conditioned_destination)
        reference_debug["opencda_style_reference_conditioning_reason"] = str(
            opencda_conditioning_reason
        )
        reference_memory_reason = ""
        turn_reference_active = str(decision) in {
            "intersection_turn_left",
            "intersection_turn_right",
        }
        if (
            bool(self.config.get("full_reference_memory_enabled", True))
            and local_lane_center_reference
            and self._temporary_destination_state is not None
            and not bool(turn_reference_active)
        ):
            stabilized_reference, stabilized_destination, reference_memory_reason = (
                self._full_reference_memory.stabilize(
                    reference=local_lane_center_reference,
                    destination_state=self._temporary_destination_state,
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    stop_goal_active=bool(stop_goal_active),
                    sim_time_s=float(sim_time_s),
                )
            )
            if stabilized_reference:
                local_lane_center_reference = [
                    dict(sample) for sample in list(stabilized_reference or [])
                ]
            if stabilized_destination is not None:
                self._temporary_destination_state = list(stabilized_destination)
        elif bool(turn_reference_active):
            reset_reference_memory = getattr(self._full_reference_memory, "reset", None)
            if callable(reset_reference_memory):
                reset_reference_memory()
            reference_memory_reason = "reference_memory_disabled_for_intersection_turn"
        reference_debug["reference_memory_reason"] = str(reference_memory_reason)
        self._previous_lane_center_reference = [
            dict(sample) for sample in list(local_lane_center_reference or [])
        ]
        return (
            list(self._temporary_destination_state),
            list(local_lane_center_reference),
            {
                "decision": str(decision),
                "lc_state": str(lc_state),
                "target_lane_id": int(target_lane_id),
                "current_lane_id": int(current_lane_id),
                "lane_safety_scores": dict(lane_safety_scores),
                "traffic_signal_state": str(behavior_traffic_state),
                "traffic_control_from_cp": bool(planner_input_frame.planning.traffic_control.from_cp),
                "stop_target": (
                    dict(behavior_stop_target)
                    if isinstance(behavior_stop_target, Mapping)
                    else {}
                ),
            },
            reference_debug,
        )


    def _select_candidate_reference_for_mpc(
            self,
            *,
            candidate_intents: Sequence[object],
            baseline_decision: str,
            baseline_lc_state: str,
            baseline_target_lane_id: int,
            baseline_speed_ref_mps: float,
            baseline_destination_state: Sequence[float] | None,
            baseline_reference: Sequence[Mapping[str, object]],
            baseline_reference_debug: Mapping[str, object],
            base_temporary_destination_state: Sequence[float] | None,
            ego_location: _PlannerLocation,
            ego_yaw_rad: float,
            ego_speed_mps: float,
            ego_pose: Mapping[str, object],
            current_state: Sequence[float],
            current_lane_id: int,
            route_optimal_lane_id: int,
            route_points: Sequence[Sequence[float]],
            route_reference_allowed: bool,
            route_reference_gate_reason: str,
            planner_input_frame: Any,
            planner_mode: str,
            object_snapshots: Sequence[Mapping[str, object]],
        ) -> tuple[str, int, float, list[dict[str, object]], list[float], dict[str, object]]:
        from cpx_planning.behavior_planner import (
            MpcReferenceGenerationContext,
            compute_temp_destination,
            generate_mpc_reference,
            select_reference_intent,
        )
        from cpx_planning.pipeline.candidate_pipeline import (
            CandidateReferenceResult,
            evaluate_candidate_reference,
            select_best_candidate,
            summarize_candidate_results,
        )

        intents = list(candidate_intents or [])
        prediction_trajectories = dict(
            planner_input_frame.prediction.obstacle_future_trajectories
        )
        if not intents:
            return (
                str(baseline_decision),
                int(baseline_target_lane_id),
                float(baseline_speed_ref_mps),
                [dict(sample) for sample in list(baseline_reference or [])],
                list(baseline_destination_state or []),
                {
                    "candidate_pipeline_selected": "baseline_no_candidates",
                    "candidate_pipeline_selected_status": "feasible",
                    "candidate_pipeline_selected_reason": "",
                    "candidate_pipeline_count": 0,
                    "candidate_prediction_trajectory_count": int(len(prediction_trajectories)),
                    "candidate_pipeline_summary": "[]",
                },
            )

        candidate_results = []
        for intent in intents:
            candidate_decision = str(getattr(intent, "decision", baseline_decision))
            candidate_target_lane_id = int(getattr(intent, "target_lane_id", current_lane_id) or current_lane_id)
            candidate_speed_ref_mps = float(getattr(intent, "target_speed_mps", baseline_speed_ref_mps))
            candidate_stop_goal_active = bool(getattr(intent, "stop_goal_active", False)) or candidate_decision in {
                "stop_at_intersection",
                "stop_sign",
                "emergency_brake",
            }
            candidate_lc_state = self._candidate_lc_state(
                decision=str(candidate_decision),
                baseline_decision=str(baseline_decision),
                baseline_lc_state=str(baseline_lc_state),
            )
            same_as_baseline = (
                str(candidate_decision) == str(baseline_decision)
                and int(candidate_target_lane_id) == int(baseline_target_lane_id)
                and abs(float(candidate_speed_ref_mps) - float(baseline_speed_ref_mps)) < 1.0e-3
            )
            if bool(same_as_baseline) and baseline_destination_state is not None:
                destination_state = list(baseline_destination_state)
                reference = [dict(sample) for sample in list(baseline_reference or [])]
                candidate_reference_debug = dict(baseline_reference_debug or {})
            else:
                previous_temp = list(base_temporary_destination_state or [])
                candidate_temp_destination = compute_temp_destination(
                    map_planner=self.reference_map,
                    ego_pose=ego_pose,
                    target_lane_id=int(candidate_target_lane_id),
                    decision=str(candidate_decision),
                    lookahead_m=float(self.lookahead_m),
                    target_v_mps=float(candidate_speed_ref_mps),
                    global_route_points=route_points,
                    mode_reference_xy=(
                        None
                        if not previous_temp
                        else (float(previous_temp[0]), float(previous_temp[1]))
                    ),
                    prev_mode=(
                        None
                        if len(previous_temp) < 6
                        else float(previous_temp[5])
                    ),
                    prev_road_id=(
                        None
                        if len(previous_temp) < 7
                        else int(previous_temp[6])
                    ),
                    prev_entered_intersection=(
                        False
                        if len(previous_temp) < 8
                        else bool(float(previous_temp[7]) > 0.5)
                    ),
                    next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
                    mode_override=str(planner_mode),
                    follow_global_route_lane=bool(
                        route_reference_allowed and planner_input_frame.map_lane.in_junction
                    ),
                )
                reference_intent = select_reference_intent(
                    behavior_decision=str(candidate_decision),
                    planner_fsm_state=str(candidate_lc_state),
                    ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
                    reference_target_lane_id=int(candidate_target_lane_id),
                    current_lane_id=int(current_lane_id),
                    route_optimal_lane_id=int(route_optimal_lane_id),
                    global_route_reference_allowed=bool(route_reference_allowed),
                    traffic_control_lane_lock_active=False,
                )
                ref_context = MpcReferenceGenerationContext(
                    map_planner=self.reference_map,
                    ego_pose=ego_pose,
                    ego_state=current_state,
                    active_global_route_points=route_points,
                    previous_lane_center_reference=self._previous_lane_center_reference,
                    behavior_runtime_cfg=self.behavior_runtime_cfg,
                    reference_intent=reference_intent,
                    current_applied_behavior=str(candidate_decision),
                    cached_planner_lc_state=str(candidate_lc_state),
                    reference_target_lane_id=int(candidate_target_lane_id),
                    current_lane_id=int(current_lane_id),
                    global_route_reference_allowed=bool(route_reference_allowed),
                    global_route_reference_gate_reason=str(route_reference_gate_reason),
                    should_follow_global_route_lane_for_reference=bool(reference_intent.follow_global_route_lane),
                    traffic_control_lane_lock_active=False,
                    final_goal_stop_active=False,
                    stop_target_state=None,
                    follow_target_state=None,
                    current_temp_reference_xy=(
                        float(candidate_temp_destination[0]),
                        float(candidate_temp_destination[1]),
                    ),
                    current_temp_mode_value=(
                        float(candidate_temp_destination[5])
                        if len(candidate_temp_destination) >= 6 else 0.0
                    ),
                    current_temp_road_id=(
                        int(candidate_temp_destination[6])
                        if len(candidate_temp_destination) >= 7 else None
                    ),
                    current_temp_entered_intersection=(
                        bool(float(candidate_temp_destination[7]) > 0.5)
                        if len(candidate_temp_destination) >= 8 else False
                    ),
                    active_reference_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
                    current_temp_mode_str=str(planner_mode),
                    lane_reference_speed_mps=max(
                        1.0,
                        float(ego_speed_mps),
                        abs(float(candidate_speed_ref_mps)),
                    ),
                    lane_reference_step_distance_m=max(
                        0.5,
                        float(self.mpc.dt_s)
                        * max(1.0, float(ego_speed_mps), abs(float(candidate_speed_ref_mps))),
                    ),
                    mpc_horizon_steps=int(self.mpc.horizon_steps),
                    mpc_dt_s=float(self.mpc.dt_s),
                    temporary_destination_state=candidate_temp_destination,
                    lane_reference_freeze_count=int(self._lane_reference_freeze_count),
                    sim_time_s=float(self._sim_time_s()),
                    stop_release_temp_smooth_until_sim_time_s=float(self._stop_release_temp_smooth_until_sim_time_s),
                )
                ref_output = generate_mpc_reference(ref_context)
                destination_state = list(ref_output.temporary_destination_state or candidate_temp_destination)
                reference = [dict(sample) for sample in list(ref_output.local_lane_center_reference or [])]
                candidate_reference_debug = dict(ref_output.mpc_reference_result.trace.as_trace_fields())
                candidate_reference_debug["fallback_reason"] = str(ref_output.last_reference_fallback_reason)

            if candidate_decision in {"intersection_turn_left", "intersection_turn_right"}:
                turn_reference, turn_destination, turn_reference_reason = (
                    self._carla_waypoint_turn_reference(
                        ego_location=ego_location,
                        ego_yaw_rad=float(ego_yaw_rad),
                        current_state=current_state,
                        current_lane_id=int(current_lane_id),
                        target_lane_id=int(candidate_target_lane_id),
                        target_speed_mps=float(candidate_speed_ref_mps),
                        destination_state=destination_state,
                    )
                )
                if turn_reference:
                    reference = [dict(sample) for sample in turn_reference]
                    destination_state = list(turn_destination)
                    candidate_reference_debug.update({
                        "reference_pipeline_stage": "carla_waypoint_turn",
                        "reference_pipeline_intent": str(candidate_decision),
                        "reference_pipeline_intent_mode": "intersection_turn",
                        "reference_pipeline_follow_global_route_lane": 1,
                        "reference_source": "carla_grp_waypoint_turn",
                        "fallback_reason": "",
                        "carla_turn_reference_reason": str(turn_reference_reason),
                    })
                else:
                    candidate_reference_debug["carla_turn_reference_reason"] = str(
                        turn_reference_reason
                    )

            stabilizer_reason = ""
            if bool(self.full_mpc_reference_stabilizer_enabled):
                destination_state, reference, stabilizer_reason = self._stabilize_mpc_reference_input(
                    destination_state=destination_state,
                    lane_center_reference=reference,
                    current_state=current_state,
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    ego_speed_mps=float(ego_speed_mps),
                    speed_ref_mps=float(candidate_speed_ref_mps),
                    stop_goal_active=bool(candidate_stop_goal_active),
                    behavior_decision=str(candidate_decision),
                    behavior_fsm_state=str(candidate_lc_state),
                    current_lane_id=int(current_lane_id),
                    stop_target=(
                        getattr(intent, "stop_target", None)
                        if isinstance(getattr(intent, "stop_target", None), Mapping)
                        else None
                    ),
                )
            if (
                candidate_decision in {"intersection_turn_left", "intersection_turn_right"}
                and "drop_duplicate_sample" in str(stabilizer_reason)
            ):
                turn_direction = "left" if candidate_decision.endswith("_left") else "right"
                creep_reference = self._creep_turn_reference_samples(
                    ego_location=ego_location,
                    ego_heading_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    turn_direction=str(turn_direction),
                )
                if creep_reference:
                    from cpx_planning.behavior_planner.reference_pipeline import (
                        lane_center_destination_from_reference,
                    )

                    reference = [dict(sample) for sample in creep_reference]
                    aligned_destination = lane_center_destination_from_reference(
                        destination_state=destination_state,
                        lane_center_reference=reference,
                        ego_state=current_state,
                        target_forward_m=float(
                            self.config.get("full_turn_creep_destination_forward_m", 7.0)
                        ),
                    )
                    if aligned_destination is not None:
                        destination_state = list(aligned_destination)
                    stabilizer_reason = (
                        str(stabilizer_reason) + ";"
                        if str(stabilizer_reason)
                        else ""
                    ) + "creep_turn_reference_after_duplicate"
            contract_result = self._validate_candidate_reference_contract(
                decision=str(candidate_decision),
                lc_state=str(candidate_lc_state),
                current_lane_id=int(current_lane_id),
                speed_ref_mps=float(candidate_speed_ref_mps),
                stop_goal_active=bool(candidate_stop_goal_active),
                current_state=current_state,
                destination_state=destination_state,
                lane_center_reference=reference,
            )
            candidate_reference_debug["mpc_reference_stabilizer_reason"] = str(stabilizer_reason)
            candidate_result = CandidateReferenceResult(
                intent=intent,
                destination_state=list(destination_state),
                lane_center_reference=[dict(sample) for sample in list(reference or [])],
                reference_debug=dict(candidate_reference_debug),
                contract_result=contract_result,
            )
            candidate_results.append(evaluate_candidate_reference(
                candidate=candidate_result,
                ego_state=current_state,
                object_snapshots=object_snapshots,
                prediction_trajectories=prediction_trajectories,
                current_lane_id=int(current_lane_id),
                min_object_distance_m=float(self.full_candidate_reference_min_object_distance_m),
            ))

        if (
            bool(self.strict_decision_ownership_enabled)
            and candidate_results
            and not any(bool(candidate.feasible) for candidate in candidate_results)
        ):
            return self._explicit_fallback_candidate_for_mpc(
                candidate_results=candidate_results,
                baseline_decision=str(baseline_decision),
                baseline_target_lane_id=int(baseline_target_lane_id),
                current_lane_id=int(current_lane_id),
                current_state=current_state,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                summarize_candidate_results=summarize_candidate_results,
            )

        selected = select_best_candidate(candidate_results)
        selected_debug = dict(selected.reference_debug or {})
        selected_debug.update({
            "stage": selected_debug.get("reference_pipeline_stage", ""),
            "intent_mode": selected_debug.get("reference_pipeline_intent_mode", ""),
            "fallback_reason": selected_debug.get("fallback_reason", ""),
            "reference_source": str(
                selected_debug.get("reference_source", "candidate_reference_pipeline")
            ),
            "candidate_pipeline_selected": str(selected.intent.name),
            "candidate_pipeline_selected_status": str(selected.feasibility_status),
            "candidate_pipeline_selected_reason": str(selected.feasibility_reason),
            "candidate_pipeline_count": int(len(candidate_results)),
            "candidate_prediction_trajectory_count": int(len(prediction_trajectories)),
            "candidate_pipeline_summary": summarize_candidate_results(candidate_results),
            "candidate_selected_decision": str(selected.intent.decision),
            "candidate_selected_lane_id": int(selected.intent.target_lane_id),
            "candidate_selected_cost": float(selected.total_cost),
            "candidate_evaluation_summary": (
                f"{selected.intent.name}->{selected.intent.decision}"
                f":L{int(selected.intent.target_lane_id)}"
                f" cost={float(selected.total_cost):.2f}"
            ),
        })
        return (
            str(selected.intent.decision),
            int(selected.intent.target_lane_id),
            float(selected.intent.target_speed_mps),
            [dict(sample) for sample in list(selected.lane_center_reference or [])],
            list(selected.destination_state or []),
            selected_debug,
        )

    def _explicit_fallback_candidate_for_mpc(
            self,
            *,
            candidate_results: Sequence[object],
            baseline_decision: str,
            baseline_target_lane_id: int,
            current_lane_id: int,
            current_state: Sequence[float],
            ego_location: _PlannerLocation,
            ego_yaw_rad: float,
            summarize_candidate_results: Any,
        ) -> tuple[str, int, float, list[dict[str, object]], list[float], dict[str, object]]:
        """Return an explicit fallback candidate when every candidate is invalid."""

        stop_like = str(baseline_decision or "").strip().lower() in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }
        turn_like = str(baseline_decision or "").strip().lower() in {
            "intersection_turn_left",
            "intersection_turn_right",
        }
        selected_lane_id = int(current_lane_id)
        turn_reference_reason = ""
        if bool(stop_like):
            reference, destination = self._build_ego_heading_emergency_stop_reference(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_lane_id=int(current_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                step_distance_m=max(0.5, float(self.mpc.dt_s) * 0.8),
            )
            decision = "emergency_brake"
            speed_mps = 0.0
            selected_name = "explicit_fallback_emergency_stop"
            source = "explicit_fallback_ego_heading_stop"
        elif bool(turn_like):
            fallback_turn_speed_mps = float(
                self.config.get("strict_turn_fallback_speed_mps", 0.8)
            )
            reference, destination, turn_reason = self._carla_waypoint_turn_reference(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_state=current_state,
                current_lane_id=int(current_lane_id),
                target_lane_id=int(baseline_target_lane_id or current_lane_id),
                target_speed_mps=float(fallback_turn_speed_mps),
                destination_state=None,
            )
            turn_reference_reason = str(turn_reason)
            if reference:
                decision = str(baseline_decision)
                speed_mps = float(fallback_turn_speed_mps)
                selected_lane_id = int(baseline_target_lane_id or current_lane_id)
                selected_name = "explicit_fallback_carla_route_turn"
                source = "explicit_fallback_carla_grp_waypoint_turn"
            else:
                reference, destination = self._build_ego_heading_emergency_stop_reference(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=max(0.5, float(self.mpc.dt_s) * 0.8),
                )
                decision = "emergency_brake"
                speed_mps = 0.0
                selected_name = "explicit_fallback_turn_route_unavailable_stop"
                source = "explicit_fallback_ego_heading_stop"
                self._turn_latch_decision = str(baseline_decision)
                self._turn_latch_until_sim_time_s = max(
                    float(self._turn_latch_until_sim_time_s),
                    float(self._sim_time_s())
                    + float(self.config.get("turn_route_unavailable_latch_s", 1.0)),
                )
        else:
            start_waypoint = self._map_waypoint_from_location(ego_location)
            step_distance_m = max(
                0.5,
                float(self.mpc.dt_s)
                * max(0.5, float(self.strict_explicit_fallback_speed_mps)),
            )
            reference = self._current_lane_center_reference_samples(
                start_waypoint=start_waypoint,
                current_lane_id=int(current_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                step_distance_m=float(step_distance_m),
                route_points=self._active_global_route_points(),
            )
            if not reference:
                reference = self._straight_reference_samples(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                )
            for sample in reference:
                sample["v_ref_mps"] = float(self.strict_explicit_fallback_speed_mps)
                sample["speed_ref_mps"] = float(self.strict_explicit_fallback_speed_mps)
                sample["speed_mps"] = float(self.strict_explicit_fallback_speed_mps)
            from cpx_planning.behavior_planner.reference_pipeline import (
                lane_center_destination_from_reference,
            )

            destination = lane_center_destination_from_reference(
                destination_state=[
                    float(current_state[0]),
                    float(current_state[1]),
                    float(self.strict_explicit_fallback_speed_mps),
                    float(current_state[3]),
                    int(current_lane_id),
                ],
                lane_center_reference=reference,
                ego_state=current_state,
                target_forward_m=float(
                    self.config.get("strict_explicit_fallback_forward_m", 5.0)
                ),
            ) or [
                float(current_state[0]),
                float(current_state[1]),
                float(self.strict_explicit_fallback_speed_mps),
                float(current_state[3]),
                int(current_lane_id),
            ]
            decision = "lane_follow"
            speed_mps = float(self.strict_explicit_fallback_speed_mps)
            selected_name = "explicit_fallback_keep_lane"
            source = "explicit_fallback_current_lane"

        summary = summarize_candidate_results(candidate_results)
        reason = "all_candidates_infeasible"
        debug = {
            "stage": "explicit_fallback",
            "intent_mode": str(decision),
            "fallback_reason": str(reason),
            "reference_source": str(source),
            "candidate_pipeline_selected": str(selected_name),
            "candidate_pipeline_selected_status": "explicit_fallback",
            "candidate_pipeline_selected_reason": str(reason),
            "candidate_pipeline_count": int(len(candidate_results)),
            "candidate_prediction_trajectory_count": 0,
            "candidate_pipeline_summary": str(summary),
            "candidate_selected_decision": str(decision),
            "candidate_selected_lane_id": int(selected_lane_id),
            "candidate_selected_cost": 100000.0,
            "candidate_evaluation_summary": (
                f"{selected_name}->{decision}:L{int(selected_lane_id)} explicit_fallback"
            ),
            "mpc_reference_stabilizer_reason": str(reason),
            "carla_turn_reference_reason": str(turn_reference_reason),
        }
        return (
            str(decision),
            int(selected_lane_id),
            float(speed_mps),
            [dict(sample) for sample in list(reference or [])],
            list(destination or []),
            debug,
        )

    def _straight_reference_samples(
            self,
            *,
            ego_location: _PlannerLocation,
            ego_yaw_rad: float,
            current_lane_id: int,
            horizon_steps: int,
            step_distance_m: float,
        ) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        for index in range(max(1, int(horizon_steps)) + 1):
            distance_m = float(index + 1) * max(0.5, float(step_distance_m))
            x_m = float(ego_location.x) + distance_m * math.cos(float(ego_yaw_rad))
            y_m = float(ego_location.y) + distance_m * math.sin(float(ego_yaw_rad))
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(ego_yaw_rad),
                "lane_id": int(current_lane_id),
                "lane_width_m": 3.5,
                "road_center_offset_m": 0.0,
                "road_left_width_m": 1.75,
                "road_right_width_m": 1.75,
            })
        return samples

    def _build_ego_heading_reference_samples(
            self,
            *,
            ego_location: _PlannerLocation,
            ego_yaw_rad: float,
            current_lane_id: int,
            horizon_steps: int,
            step_distance_m: float,
            speed_mps: float,
        ) -> list[dict[str, float]]:
        samples: list[dict[str, float]] = []
        for index in range(max(2, int(horizon_steps))):
            forward_m = float(index + 1) * max(0.35, float(step_distance_m))
            x_m = float(ego_location.x) + float(forward_m) * math.cos(float(ego_yaw_rad))
            y_m = float(ego_location.y) + float(forward_m) * math.sin(float(ego_yaw_rad))
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(ego_yaw_rad),
                "lane_id": int(current_lane_id),
                "lane_width_m": 3.5,
                "speed_ref_mps": float(speed_mps),
                "v_ref_mps": float(speed_mps),
                "speed_mps": float(speed_mps),
            })
        return samples

    def _creep_turn_reference_samples(
            self,
            *,
            ego_location: _PlannerLocation,
            ego_heading_rad: float,
            current_lane_id: int,
            horizon_steps: int,
            turn_direction: str,
        ) -> list[dict[str, float]]:
        direction = str(turn_direction or "").strip().lower()
        sign = 1.0 if direction == "left" else -1.0 if direction == "right" else 0.0
        if abs(float(sign)) < 1.0e-6:
            return self._build_ego_heading_reference_samples(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_heading_rad),
                current_lane_id=int(current_lane_id),
                horizon_steps=int(horizon_steps),
                step_distance_m=float(self.config.get("full_turn_creep_step_m", 0.65)),
                speed_mps=float(self.config.get("full_turn_creep_speed_mps", 0.9)),
            )
        radius_m = max(4.0, float(self.config.get("full_turn_creep_radius_m", 10.0)))
        step_m = max(0.35, float(self.config.get("full_turn_creep_step_m", 0.65)))
        speed_mps = max(0.2, float(self.config.get("full_turn_creep_speed_mps", 0.9)))
        samples: list[dict[str, float]] = []
        for index in range(max(2, int(horizon_steps))):
            s_m = float(index + 1) * float(step_m)
            theta = min(float(s_m) / float(radius_m), float(self.config.get("full_turn_creep_max_theta_rad", 1.15)))
            local_x = float(radius_m) * math.sin(float(theta))
            local_y = float(sign) * float(radius_m) * (1.0 - math.cos(float(theta)))
            world_x = (
                float(ego_location.x)
                + math.cos(float(ego_heading_rad)) * float(local_x)
                - math.sin(float(ego_heading_rad)) * float(local_y)
            )
            world_y = (
                float(ego_location.y)
                + math.sin(float(ego_heading_rad)) * float(local_x)
                + math.cos(float(ego_heading_rad)) * float(local_y)
            )
            heading_rad = self._wrap_angle(float(ego_heading_rad) + float(sign) * float(theta))
            samples.append({
                "x_ref_m": float(world_x),
                "y_ref_m": float(world_y),
                "x": float(world_x),
                "y": float(world_y),
                "heading_rad": float(heading_rad),
                "lane_id": int(current_lane_id),
                "lane_width_m": 3.5,
                "speed_ref_mps": float(speed_mps),
                "v_ref_mps": float(speed_mps),
                "speed_mps": float(speed_mps),
            })
        return samples

    def _build_ego_heading_emergency_stop_reference(
            self,
            *,
            ego_location: _PlannerLocation,
            ego_yaw_rad: float,
            current_lane_id: int,
            horizon_steps: int,
            step_distance_m: float,
        ) -> tuple[list[dict[str, object]], list[float]]:
        samples = []
        for index in range(max(1, int(horizon_steps)) + 1):
            distance_m = float(index + 1) * max(0.5, float(step_distance_m))
            x_m = float(ego_location.x) + float(distance_m) * math.cos(float(ego_yaw_rad))
            y_m = float(ego_location.y) + float(distance_m) * math.sin(float(ego_yaw_rad))
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(ego_yaw_rad),
                "lane_id": int(current_lane_id),
                "lane_width_m": 3.5,
                "road_center_offset_m": 0.0,
                "road_left_width_m": 1.75,
                "road_right_width_m": 1.75,
                "v_ref_mps": 0.0,
                "speed_ref_mps": 0.0,
                "speed_mps": 0.0,
            })
        first = samples[0] if samples else {
            "x_ref_m": float(ego_location.x),
            "y_ref_m": float(ego_location.y),
            "heading_rad": float(ego_yaw_rad),
        }
        destination = [
            float(first.get("x_ref_m", ego_location.x)),
            float(first.get("y_ref_m", ego_location.y)),
            0.0,
            float(first.get("heading_rad", ego_yaw_rad)),
            int(current_lane_id),
        ]
        return samples, destination


    @staticmethod
    def _candidate_lc_state(
            *,
            decision: str,
            baseline_decision: str,
            baseline_lc_state: str,
        ) -> str:
        if str(decision) == str(baseline_decision):
            return str(baseline_lc_state or "LANE_KEEP")
        if str(decision) == "intersection_turn_left":
            return "INTERSECTION_TURN_LEFT"
        if str(decision) == "intersection_turn_right":
            return "INTERSECTION_TURN_RIGHT"
        if str(decision) == "lane_change_left":
            return "EXECUTE_LANE_CHANGE_LEFT"
        if str(decision) == "lane_change_right":
            return "EXECUTE_LANE_CHANGE_RIGHT"
        return "LANE_KEEP"


    def _validate_candidate_reference_contract(
            self,
            *,
            decision: str,
            lc_state: str,
            current_lane_id: int,
            speed_ref_mps: float,
            stop_goal_active: bool,
            current_state: Sequence[float],
            destination_state: Sequence[float],
            lane_center_reference: Sequence[Mapping[str, object]],
        ):
        from cpx_planning.pipeline.reference_contract import (
            contract_from_config,
            validate_reference_contract,
        )

        normalized_decision = str(decision or "").strip().lower()
        normalized_fsm = str(lc_state or "").strip().upper()
        lane_change_active = (
            normalized_decision in {"lane_change_left", "lane_change_right"}
            or normalized_fsm.startswith("EXECUTE_LANE_CHANGE")
        )
        turn_active = (
            normalized_decision in {"intersection_turn_left", "intersection_turn_right"}
            or normalized_fsm.startswith("INTERSECTION_TURN")
        )
        stop_like = bool(stop_goal_active) or normalized_decision in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }
        contract_mode = (
            "emergency_stop"
            if normalized_decision == "emergency_brake"
            else "stop"
            if bool(stop_like)
            else "lane_change"
            if bool(lane_change_active)
            else "intersection_turn"
            if bool(turn_active)
            else "lane_follow"
        )
        contract = contract_from_config(
            mode=str(contract_mode),
            expected_lane_id=int(current_lane_id),
            horizon_steps=int(self.mpc.horizon_steps),
            config=dict(self.config),
            default_speed_mps=max(float(self.target_speed_mps), float(speed_ref_mps), 0.1),
        )
        return validate_reference_contract(
            reference_samples=lane_center_reference,
            destination_state=destination_state,
            ego_state=current_state,
            contract=contract,
            check_destination_body_lateral=not bool(lane_change_active or turn_active),
        )

    def _run_full_cpx_pipeline_step(self):
        """Run the active OpenCDA full-pipeline order using the PlannerInputFrame already built from ROS."""
        adapter_output = self._latest_adapter_output
        planner_input_frame = adapter_output.frame
        sim_time_s = float(planner_input_frame.planning.sim_time_s)
        self._current_sim_time_s = float(sim_time_s)
        ego_pose = dict(adapter_output.ego_pose)
        ego_location = _PlannerLocation.from_mapping(ego_pose)
        ego_transform = _PlannerTransform.from_pose(ego_pose)
        ego_yaw_rad = float(planner_input_frame.planning.ego.heading_rad)
        ego_speed_mps = float(planner_input_frame.planning.ego.speed_mps)
        object_snapshots = [dict(item) for item in list(planner_input_frame.perception.planning_objects or []) if isinstance(item, Mapping)]
        mpc_object_snapshots = self._limit_obstacles_for_mpc(object_snapshots=object_snapshots, ego_location=ego_location)
        front_gap_m = self._front_gap_m(ego_location=ego_location, ego_yaw_rad=ego_yaw_rad, object_snapshots=object_snapshots)
        stop_goal_active = front_gap_m is not None and front_gap_m < self.min_front_gap_m
        speed_ref_mps = 0.0 if stop_goal_active else self.target_speed_mps
        current_state = [float(ego_location.x), float(ego_location.y), float(ego_speed_mps), float(ego_yaw_rad)]
        behavior_debug: dict[str, Any] = {}
        reference_debug: dict[str, Any] = {}
        try:
            destination_state, lane_center_reference, behavior_debug, reference_debug = self._plan_behavior_and_reference(ego_location=ego_location, ego_yaw_rad=ego_yaw_rad, ego_speed_mps=ego_speed_mps, speed_ref_mps=speed_ref_mps, object_snapshots=object_snapshots, stop_goal_active=stop_goal_active, cp_payload=None)
        except Exception as exc:
            if self.debug:
                print(f"[CP-X ROS Planner] behavior/reference pipeline failed: {exc}")
            destination_state, lane_center_reference = self._build_current_lane_fallback_reference(ego_location=ego_location, ego_yaw_rad=float(ego_yaw_rad), current_state=current_state, speed_ref_mps=float(speed_ref_mps))
            behavior_debug = {"decision": "lane_follow", "lc_state": "FALLBACK", "target_lane_id": "", "current_lane_id": self._lane_id_at_location(ego_location), "pipeline_error": str(exc)}
            reference_debug = {"reference_source": "current_lane_center_exception_fallback", "pipeline_error": str(exc)}
        mpc_stop_goal_active = bool(stop_goal_active) or str(behavior_debug.get("decision", "")) in {"stop_at_intersection", "stop_sign", "emergency_brake"}
        if bool(mpc_stop_goal_active):
            speed_ref_mps = 0.0
        if bool(mpc_stop_goal_active) and len(destination_state) >= 3:
            destination_state = list(destination_state)
            destination_state[2] = 0.0
        stop_target_forward_m_debug = ""
        stop_target_debug = behavior_debug.get("stop_target") if isinstance(behavior_debug.get("stop_target"), Mapping) else None
        if bool(mpc_stop_goal_active) and isinstance(stop_target_debug, Mapping):
            try:
                stop_target_forward_m_debug, _ = self._body_frame_xy(origin_x_m=float(ego_location.x), origin_y_m=float(ego_location.y), heading_rad=float(ego_yaw_rad), target_x_m=float(stop_target_debug.get("x_m", stop_target_debug.get("x", ego_location.x))), target_y_m=float(stop_target_debug.get("y_m", stop_target_debug.get("y", ego_location.y))))
            except Exception:
                stop_target_forward_m_debug = ""
        mpc_reference_stabilizer_reason = ""
        if bool(self.full_mpc_reference_stabilizer_enabled):
            destination_state, lane_center_reference, mpc_reference_stabilizer_reason = self._stabilize_mpc_reference_input(destination_state=destination_state, lane_center_reference=lane_center_reference, current_state=current_state, ego_location=ego_location, ego_yaw_rad=float(ego_yaw_rad), ego_speed_mps=float(ego_speed_mps), speed_ref_mps=float(speed_ref_mps), stop_goal_active=bool(mpc_stop_goal_active), behavior_decision=str(behavior_debug.get("decision", "")), behavior_fsm_state=str(behavior_debug.get("lc_state", "")), current_lane_id=int(behavior_debug.get("current_lane_id", 0) or 0), stop_target=behavior_debug.get("stop_target") if isinstance(behavior_debug.get("stop_target"), Mapping) else None)
            reference_debug["mpc_reference_stabilizer_reason"] = str(mpc_reference_stabilizer_reason)
        destination_forward_m, destination_lateral_m = self._body_frame_xy(origin_x_m=float(ego_location.x), origin_y_m=float(ego_location.y), heading_rad=float(ego_yaw_rad), target_x_m=float(destination_state[0]), target_y_m=float(destination_state[1]))
        reference_first_forward_m = ""
        reference_first_lateral_m = ""
        if lane_center_reference:
            first_reference = lane_center_reference[0]
            reference_first_forward_m, reference_first_lateral_m = self._body_frame_xy(origin_x_m=float(ego_location.x), origin_y_m=float(ego_location.y), heading_rad=float(ego_yaw_rad), target_x_m=float(first_reference.get("x_ref_m", first_reference.get("x", ego_location.x))), target_y_m=float(first_reference.get("y_ref_m", first_reference.get("y", ego_location.y))))
        mpc_status = str(getattr(self.mpc, "_last_status", ""))
        trajectory_memory_reason = ""
        mode_transition_guard_reason = self._apply_behavior_mode_transition_guard(decision=str(behavior_debug.get("decision", "")), lc_state=str(behavior_debug.get("lc_state", "")), target_lane_id=int(behavior_debug.get("target_lane_id", 0) or 0), stop_goal_active=bool(mpc_stop_goal_active))
        candidate_hard_gate_reason = self._candidate_hard_gate_reason(reference_debug=reference_debug, behavior_decision=str(behavior_debug.get("decision", "")), stop_goal_active=bool(mpc_stop_goal_active))
        try:
            if str(candidate_hard_gate_reason):
                raise RuntimeError(str(candidate_hard_gate_reason))
            force_replan = bool(mpc_stop_goal_active) or str(behavior_debug.get("decision", "")) in {"stop_at_intersection", "stop_sign", "emergency_brake", "intersection_turn_left", "intersection_turn_right"} or bool(mode_transition_guard_reason)
            mpc_replan_executed = bool(self.control_buffer.should_replan(sim_time_s=float(sim_time_s), force_replan=bool(force_replan)))
            if bool(mpc_replan_executed):
                self.mpc.plan_trajectory(current_state=current_state, destination_state=destination_state, object_snapshots=mpc_object_snapshots, current_acceleration_mps2=float(self._last_accel_mps2), current_steering_rad=float(self._last_steer_rad), lane_center_reference_samples=lane_center_reference, stop_goal_active=bool(mpc_stop_goal_active))
                mpc_status = str(getattr(self.mpc, "_last_status", "")).strip().lower()
                if mpc_status and mpc_status not in {"solved", "solved inaccurate"}:
                    raise RuntimeError(f"MPC status={mpc_status}")
                u_solution = getattr(self.mpc, "_last_u_solution", None)
                if u_solution is None or len(u_solution) == 0:
                    raise RuntimeError("MPC did not expose a control solution")
                self.control_buffer.update_from_solution(u_solution=u_solution, plan_time_s=float(sim_time_s), dt_s=float(self.mpc.dt_s))
                accel_mps2 = float(u_solution[0, 0])
                steer_rad = float(u_solution[0, 1])
            else:
                buffered = self.control_buffer.sample(sim_time_s=float(sim_time_s))
                if buffered is None:
                    raise RuntimeError("MPC control buffer empty")
                accel_mps2, steer_rad, _buffer_reason = buffered
                mpc_status = "buffer_reuse"
            control = self._control_from_mpc(accel_mps2, steer_rad)
            if bool(self.config.get("full_trajectory_memory_enabled", True)):
                control, accel_mps2, steer_rad, trajectory_memory_reason = self._full_trajectory_memory.accept_or_blend(control=control, accel_mps2=float(accel_mps2), steer_rad=float(steer_rad), control_factory=self._control_from_mpc, sim_time_s=float(sim_time_s))
            fallback_reason = ""
        except Exception as exc:
            mpc_replan_executed = True
            hard_gate_active = str(exc).startswith("candidate_hard_gate:")
            if bool(hard_gate_active):
                mpc_replan_executed = False
            if bool(self.config.get("full_trajectory_memory_enabled", True)) and not bool(hard_gate_active):
                memory_control, memory_accel, memory_steer, memory_reason = self._full_trajectory_memory.reuse_if_fresh(sim_time_s=float(sim_time_s), stop_goal_active=bool(mpc_stop_goal_active), control_factory=self._control_from_mpc)
                if memory_control is not None:
                    control = memory_control
                    accel_mps2 = float(memory_accel)
                    steer_rad = float(memory_steer)
                    fallback_reason = f"fallback_to_full_trajectory_memory:{exc}"
                    trajectory_memory_reason = str(memory_reason)
                else:
                    control = None
            else:
                control = None
            if control is None:
                fallback_reason = str(exc)
                trajectory_memory_reason = ""
                if bool(hard_gate_active):
                    control = self._emergency_stop_control()
                    accel_mps2 = float(self._last_accel_mps2)
                    steer_rad = 0.0
                else:
                    control = self._fallback_control(ego_transform=ego_transform, ego_speed_mps=ego_speed_mps, destination_state=destination_state, stop_goal_active=mpc_stop_goal_active)
                    accel_mps2 = self._last_accel_mps2
                    steer_rad = self._last_steer_rad
            mpc_status = "candidate_hard_gate" if bool(hard_gate_active) else str(getattr(self.mpc, "_last_status", str(exc)))
            if not self._warned:
                print(f"[CP-X ROS Planner] MPC fallback active: {fallback_reason}")
                self._warned = True
        mpc_feedback_record_reason = self.mpc_feedback.record_result(decision=str(behavior_debug.get("decision", "")), target_lane_id=int(behavior_debug.get("target_lane_id", 0) or 0), status=str(mpc_status), reason=str(fallback_reason), timestamp_s=float(sim_time_s), success=not bool(fallback_reason))
        control_guard_reason = ""
        control, accel_mps2, steer_rad, control_guard_reason = self._apply_control_safety_guards(control=control, accel_mps2=float(accel_mps2), steer_rad=float(steer_rad), ego_transform=ego_transform, ego_speed_mps=float(ego_speed_mps), speed_ref_mps=float(speed_ref_mps), destination_state=destination_state, destination_lateral_m=float(destination_lateral_m), stop_goal_active=bool(mpc_stop_goal_active), behavior_decision=str(behavior_debug.get("decision", "")), behavior_fsm_state=str(behavior_debug.get("lc_state", "")), traffic_signal_state=str(behavior_debug.get("traffic_signal_state", "")), sim_time_s=float(sim_time_s))
        pre_supervisor_accel_mps2 = float(accel_mps2)
        pre_supervisor_steer_rad = float(steer_rad)
        control, safety_supervisor_reason = self.safety_supervisor.filter_control(control=control, safety_manager=None, input_frame=planner_input_frame, behavior_decision=str(behavior_debug.get("decision", "")), traffic_signal_state=str(behavior_debug.get("traffic_signal_state", "")), stop_goal_active=bool(mpc_stop_goal_active), planner_accel_mps2=float(pre_supervisor_accel_mps2))
        post_supervisor_accel_mps2 = self._accel_from_control(control)
        post_supervisor_steer_rad = self._steer_rad_from_control(control)
        self._last_accel_mps2 = float(post_supervisor_accel_mps2)
        self._last_steer_rad = float(post_supervisor_steer_rad)
        accel_mps2 = float(post_supervisor_accel_mps2)
        steer_rad = float(post_supervisor_steer_rad)
        reference_debug.update({
            "sim_time_s": float(sim_time_s),
            "vehicle_id": -1,
            "x_m": float(ego_location.x),
            "y_m": float(ego_location.y),
            "yaw_deg": float(ego_transform.rotation.yaw),
            "speed_mps": float(ego_speed_mps),
            "planner": "cpx_mpc",
            "object_count": len(object_snapshots),
            "local_object_count": int(planner_input_frame.perception.dynamic_count),
            "v2x_nearby_count": int(planner_input_frame.cp_messages.obstacle_count),
            "cp_obstacle_count": int(planner_input_frame.cp_messages.obstacle_count),
            "cp_control_count": int(planner_input_frame.cp_messages.traffic_control_count),
            "front_gap_m": "" if front_gap_m is None else float(front_gap_m),
            "stop_goal_active": bool(mpc_stop_goal_active),
            "behavior_decision": str(behavior_debug.get("decision", "")),
            "behavior_fsm_state": str(behavior_debug.get("lc_state", "")),
            "current_lane_id": behavior_debug.get("current_lane_id", ""),
            "behavior_target_lane_id": behavior_debug.get("target_lane_id", ""),
            "traffic_signal_state": behavior_debug.get("traffic_signal_state", ""),
            "traffic_control_from_cp": behavior_debug.get("traffic_control_from_cp", ""),
            "lane_safety_scores": json.dumps(behavior_debug.get("lane_safety_scores", {}), default=str),
            "destination_x": float(destination_state[0]),
            "destination_y": float(destination_state[1]),
            "destination_forward_m": float(destination_forward_m),
            "destination_lateral_m": float(destination_lateral_m),
            "destination_lane_id": int(destination_state[4]) if len(destination_state) >= 5 else "",
            "reference_first_forward_m": reference_first_forward_m,
            "reference_first_lateral_m": reference_first_lateral_m,
            "global_route_point_count": len(self._active_global_route_points()),
            "route_manager_status": json.dumps(self.route_manager.last_status.as_dict(), default=str),
            "route_remaining_distance_m": float(self.route_manager.last_status.remaining_distance_m),
            "route_reached_destination": bool(self.route_manager.last_status.reached_destination),
            "global_planner_backend": str(self.global_planner_backend),
            "global_planner_backend_warning": str(self.global_planner_backend_warning),
            "carla_route_debug_reason": str(self.route_manager.carla_route_debug_reason),
            "carla_route_sync_reason": str(self.route_manager.carla_route_sync_reason),
            "carla_route_progress_index": int(self.route_manager.carla_route_progress_index),
            "stop_target_forward_m": stop_target_forward_m_debug,
            "mpc_trajectory_points": self._last_mpc_trajectory_points(),
            "global_route_points": self._active_global_route_points(),
            "lane_reference_points": [[float(sample.get("x_ref_m", sample.get("x", 0.0))), float(sample.get("y_ref_m", sample.get("y", 0.0)))] for sample in list(lane_center_reference or [])],
            "target_speed_mps": float(speed_ref_mps),
            "mpc_status": mpc_status,
            "mpc_feasibility_checked": bool(mpc_replan_executed),
            "mpc_feasibility_status": str(mpc_status),
            "mpc_feasibility_reason": str(fallback_reason),
            "mpc_replan_executed": mpc_replan_executed,
            "mpc_fallback_reason": fallback_reason,
            "mpc_feedback_record_reason": mpc_feedback_record_reason,
            "mpc_solve_time_ms": float(getattr(self.mpc, "_last_solve_time_ms", 0.0)),
            "mpc_cost_profile": str(self.active_mpc_cost_profile),
            "requested_mpc_cost_profile": str(self.requested_mpc_cost_profile),
            "mpc_cost_profile_switch_reason": str(self.mpc_cost_profile_switch_reason),
            "mpc_object_count": len(mpc_object_snapshots),
            "mpc_trajectory_point_count": len(self._last_mpc_trajectory_points()),
            "control_buffer_reason": self.control_buffer.last_reason,
            "mode_transition_guard_reason": mode_transition_guard_reason,
            "trajectory_memory_reason": trajectory_memory_reason,
            "control_guard_reason": control_guard_reason,
            "safety_supervisor_reason": safety_supervisor_reason,
            "pre_supervisor_accel_cmd_mps2": pre_supervisor_accel_mps2,
            "pre_supervisor_steer_cmd_rad": pre_supervisor_steer_rad,
            "post_supervisor_accel_cmd_mps2": float(accel_mps2),
            "post_supervisor_steer_cmd_rad": float(steer_rad),
            "applied_throttle": float(control.throttle),
            "applied_brake": float(control.brake),
            "applied_steer": float(control.steer),
            "accel_cmd_mps2": float(accel_mps2),
            "steer_cmd_rad": float(steer_rad),
            "fallback_reason": fallback_reason,
            "fallback_active": bool(fallback_reason),
            "fallback_policy": str(self.fallback_policy),
            "fallback_policy_warning": str(self.fallback_policy_warning),
            "planner_requested": True,
            "planner_executed": True,
        })
        reference_debug["reference_source"] = str(reference_debug.get("reference_source", "map_lane_center" if lane_center_reference else "straight_fallback"))
        decision_record = self._build_decision_record(scenario_state=reference_debug.get("scenario_fsm_state", ""), behavior_decision=str(behavior_debug.get("decision", "")), behavior_fsm_state=str(behavior_debug.get("lc_state", "")), candidate_selected_decision=reference_debug.get("candidate_selected_decision", ""), candidate_selected_status=reference_debug.get("candidate_pipeline_selected_status", ""), candidate_selected_reason=reference_debug.get("candidate_pipeline_selected_reason", ""), reference_source=reference_debug.get("reference_source", ""), reference_stage=reference_debug.get("stage", ""), reference_fallback_reason=reference_debug.get("fallback_reason", ""), reference_lateral_guard_reason=reference_debug.get("reference_lateral_guard_reason", ""), reference_stabilizer_reason=reference_debug.get("mpc_reference_stabilizer_reason", ""), lane_change_authorized=reference_debug.get("lane_change_authorized", ""), lane_change_gate_reason=reference_debug.get("lane_change_gate_reason", ""), route_lane_change_required=reference_debug.get("route_lane_change_required", ""), behavior_override_reason=reference_debug.get("behavior_override_reason", ""), mode_transition_guard_reason=mode_transition_guard_reason, mpc_status=mpc_status, mpc_fallback_reason=fallback_reason, control_guard_reason=control_guard_reason, control_buffer_reason=self.control_buffer.last_reason, trajectory_memory_reason=trajectory_memory_reason, safety_supervisor_reason=safety_supervisor_reason, applied_throttle=control.throttle, applied_brake=control.brake, applied_steer=control.steer)
        reference_debug.update(decision_record.as_debug_fields())
        self.last_destination_state = list(destination_state)
        self.last_reference_debug = dict(reference_debug)
        self.last_behavior_command = dict(behavior_debug)
        return PlannerOutput(
            control=control,
            behavior_command=BehaviorCommand.from_debug(behavior_debug=behavior_debug, target_speed_mps=float(speed_ref_mps)),
            reference_trajectory=[dict(sample) for sample in lane_center_reference],
            planned_trajectory=self._last_mpc_trajectory_points(),
            predictions=dict(reference_debug.get("prediction_trajectories", {}) or {}),
            acceleration_mps2=float(accel_mps2),
            steering_rad=float(steer_rad),
            diagnostics=PlannerDiagnostics(fields=dict(reference_debug)),
        )

    def _build_current_lane_fallback_reference(
            self,
            *,
            ego_location: _PlannerLocation,
            ego_yaw_rad: float,
            current_state: Sequence[float],
            speed_ref_mps: float,
        ):
        current_lane_id = self._lane_id_at_location(ego_location)
        start_waypoint = self._map_waypoint_from_location(ego_location)
        step_distance_m = max(
            0.5,
            float(self.mpc.dt_s) * max(1.0, min(float(speed_ref_mps), float(self.target_speed_mps))),
        )
        samples = self._current_lane_center_reference_samples(
            start_waypoint=start_waypoint,
            current_lane_id=int(current_lane_id),
            horizon_steps=int(self.mpc.horizon_steps),
            step_distance_m=float(step_distance_m),
            route_points=[],
        )
        if not samples:
            samples = self._straight_reference_samples(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_lane_id=int(current_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                step_distance_m=float(step_distance_m),
            )
        destination = None
        if samples:
            from cpx_planning.behavior_planner.reference_pipeline import (
                lane_center_destination_from_reference,
            )

            destination = lane_center_destination_from_reference(
                destination_state=[
                    float(samples[-1].get("x_ref_m", samples[-1].get("x", ego_location.x))),
                    float(samples[-1].get("y_ref_m", samples[-1].get("y", ego_location.y))),
                    float(speed_ref_mps),
                    float(samples[-1].get("heading_rad", ego_yaw_rad)),
                    int(current_lane_id),
                ],
                lane_center_reference=samples,
                ego_state=current_state,
                target_forward_m=float(
                    self.config.get("full_lane_follow_guard_destination_forward_m", 8.0)
                ),
            )
        if destination is None:
            destination = [
                float(ego_location.x) + 6.0 * math.cos(float(ego_yaw_rad)),
                float(ego_location.y) + 6.0 * math.sin(float(ego_yaw_rad)),
                float(speed_ref_mps),
                float(ego_yaw_rad),
                int(current_lane_id),
            ]
        return list(destination), list(samples)


    def _limit_obstacles_for_mpc(self, *, object_snapshots: Sequence[Mapping[str, Any]], ego_location: _PlannerLocation) -> list[dict[str, Any]]:
        fused = [dict(item) for item in list(object_snapshots or []) if isinstance(item, Mapping)]
        if self.max_mpc_obstacles > 0 and len(fused) > self.max_mpc_obstacles:
            fused.sort(key=lambda item: (float(item.get("x", 0.0)) - float(ego_location.x)) ** 2 + (float(item.get("y", 0.0)) - float(ego_location.y)) ** 2)
            fused = fused[: self.max_mpc_obstacles]
        return fused

    def _front_gap_m(
            self,
            ego_location: _PlannerLocation,
            ego_yaw_rad: float,
            object_snapshots: Sequence[Mapping[str, Any]],
        ) -> Optional[float]:
        cos_h = math.cos(ego_yaw_rad)
        sin_h = math.sin(ego_yaw_rad)
        best_gap = None
        for snapshot in object_snapshots:
            dx = float(snapshot.get("x", 0.0)) - float(ego_location.x)
            dy = float(snapshot.get("y", 0.0)) - float(ego_location.y)
            longitudinal = dx * cos_h + dy * sin_h
            lateral = -dx * sin_h + dy * cos_h
            if longitudinal <= 0.0 or abs(lateral) > 2.5:
                continue
            best_gap = longitudinal if best_gap is None else min(best_gap, longitudinal)
        return best_gap

    def _full_reference_lateral_guard_reason(
            self,
            *,
            decision: str,
            lc_state: str,
            stop_goal_active: bool,
            destination_state: Sequence[float] | None,
            lane_center_reference: Sequence[Mapping[str, object]] | None,
            ego_location: _PlannerLocation,
            ego_yaw_rad: float,
        ) -> str:
        normalized_decision = str(decision or "").strip().lower()
        normalized_lc_state = str(lc_state or "").strip().upper()
        lane_follow_like = (
            normalized_decision == "lane_follow"
            and normalized_lc_state in {"", "IDLE", "LANE_KEEP"}
        )
        stop_like = bool(stop_goal_active) or normalized_decision in {
            "stop_at_intersection",
            "stop_sign",
        }
        if not bool(lane_follow_like or stop_like):
            return ""

        max_destination_lateral_m = (
            float(self.full_stop_max_destination_lateral_m)
            if bool(stop_like)
            else float(self.full_lane_follow_max_destination_lateral_m)
        )
        max_reference_first_lateral_m = (
            float(self.full_stop_max_reference_first_lateral_m)
            if bool(stop_like)
            else float(self.full_lane_follow_max_reference_first_lateral_m)
        )
        reasons: list[str] = []
        if destination_state is not None and len(destination_state) >= 2:
            _, destination_lateral_m = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(destination_state[0]),
                target_y_m=float(destination_state[1]),
            )
            if abs(float(destination_lateral_m)) > float(max_destination_lateral_m):
                reasons.append(f"dest_lat={destination_lateral_m:.2f}")

        if lane_center_reference:
            first = dict(list(lane_center_reference)[0])
            _, reference_lateral_m = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(first.get("x_ref_m", first.get("x", ego_location.x))),
                target_y_m=float(first.get("y_ref_m", first.get("y", ego_location.y))),
            )
            if abs(float(reference_lateral_m)) > float(max_reference_first_lateral_m):
                reasons.append(f"ref_lat={reference_lateral_m:.2f}")

        if not reasons:
            return ""
        mode = "stop" if bool(stop_like) else "lane_follow"
        return f"{mode}_lateral_guard:" + ":".join(reasons)



    def _apply_behavior_mode_transition_guard(
            self,
            *,
            decision: str,
            lc_state: str,
            target_lane_id: int,
            stop_goal_active: bool,
        ) -> str:
        mode_key = self._behavior_mode_key(
            decision=str(decision),
            lc_state=str(lc_state),
            target_lane_id=int(target_lane_id),
            stop_goal_active=bool(stop_goal_active),
        )
        previous_key = str(getattr(self, "_full_last_behavior_mode_key", "") or "")
        self._full_last_behavior_mode_key = str(mode_key)
        if not previous_key or previous_key == str(mode_key):
            return ""
        reset_control_buffer = getattr(self.control_buffer, "reset", None)
        if callable(reset_control_buffer):
            reset_control_buffer(reason="control_buffer_reset_mode_transition")
        reset_trajectory_memory = getattr(self._full_trajectory_memory, "reset", None)
        if callable(reset_trajectory_memory):
            reset_trajectory_memory()
        reset_reference_memory = getattr(self._full_reference_memory, "reset", None)
        if callable(reset_reference_memory):
            reset_reference_memory()
        reset_conditioner = getattr(self._opencda_style_reference_conditioner, "reset", None)
        if callable(reset_conditioner):
            reset_conditioner()
        return f"mode_transition:{previous_key}->{mode_key}:reset_buffer_memory"

    @staticmethod
    def _behavior_mode_key(
            *,
            decision: str,
            lc_state: str,
            target_lane_id: int,
            stop_goal_active: bool,
        ) -> str:
        normalized_decision = str(decision or "").strip().lower()
        normalized_fsm = str(lc_state or "").strip().upper()
        if bool(stop_goal_active) or normalized_decision in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }:
            return "stop"
        if normalized_decision in {"intersection_turn_left", "intersection_turn_right"}:
            return str(normalized_decision)
        if normalized_decision in {"lane_change_left", "lane_change_right"}:
            return f"{normalized_decision}:{int(target_lane_id)}"
        if normalized_fsm.startswith("EXECUTE_LANE_CHANGE"):
            return f"{normalized_fsm.lower()}:{int(target_lane_id)}"
        return f"lane_follow:{int(target_lane_id)}"

    @staticmethod
    def _candidate_hard_gate_reason(
            *,
            reference_debug: Mapping[str, object],
            behavior_decision: str,
            stop_goal_active: bool,
        ) -> str:
        status = str(reference_debug.get("candidate_pipeline_selected_status", "")).strip().lower()
        selected = str(reference_debug.get("candidate_pipeline_selected", "candidate"))
        reason = str(reference_debug.get("candidate_pipeline_selected_reason", "infeasible"))
        stabilizer_reason = str(reference_debug.get("mpc_reference_stabilizer_reason", ""))
        combined_reason = ";".join(
            item for item in [reason, stabilizer_reason] if str(item)
        )
        if "stop_missing_target_hard_lock" in combined_reason:
            return f"candidate_hard_gate:{selected}:{combined_reason}"
        if status != "infeasible":
            return ""
        normalized_behavior = str(behavior_decision or "").strip().lower()
        stop_like = bool(stop_goal_active) or normalized_behavior in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }
        turn_like = normalized_behavior in {
            "intersection_turn_left",
            "intersection_turn_right",
        }
        hard_reason_tokens = {
            "candidate_prediction_collision_risk",
            "candidate_reference_collision_risk",
            "stop_missing_target_hard_lock",
            "strict_reference_veto",
        }
        turn_hard_reason_tokens = {
            "first_lateral_out_of_contract",
            "destination_body_lateral_out_of_contract",
            "destination_lane_error_out_of_contract",
            "turn_route_rebuild_failed",
            "creep_turn_reference_contract_violation",
        }
        if bool(stop_like) or any(token in combined_reason for token in hard_reason_tokens):
            return f"candidate_hard_gate:{selected}:{combined_reason}"
        if bool(turn_like) and any(token in combined_reason for token in turn_hard_reason_tokens):
            return f"candidate_hard_gate:{selected}:{combined_reason}"
        return ""

    def _stabilize_mpc_reference_input(
            self,
            *,
            destination_state: Sequence[float],
            lane_center_reference: Sequence[Mapping[str, object]],
            current_state: Sequence[float],
            ego_location: _PlannerLocation,
            ego_yaw_rad: float,
            ego_speed_mps: float,
            speed_ref_mps: float,
            stop_goal_active: bool,
            behavior_decision: str,
            behavior_fsm_state: str,
            current_lane_id: int,
            stop_target: Mapping[str, object] | None,
        ) -> tuple[list[float], list[dict[str, object]], str]:
        """Normalize the final reference contract before MPC sees it.

        Behavior and route generation can legitimately be noisy near junctions.
        MPC should still receive a forward, smooth, lane-consistent tracking
        target unless the planner is explicitly executing a lane change.
        """

        destination = list(destination_state or [])
        reference = [dict(sample) for sample in list(lane_center_reference or [])]
        reasons: list[str] = []

        normalized_behavior = str(behavior_decision or "").strip().lower()
        normalized_fsm = str(behavior_fsm_state or "").strip().upper()
        lane_change_active = (
            normalized_behavior in {"lane_change_left", "lane_change_right"}
            or normalized_fsm.startswith("EXECUTE_LANE_CHANGE")
        )
        turn_active = (
            normalized_behavior in {"intersection_turn_left", "intersection_turn_right"}
            or normalized_fsm.startswith("INTERSECTION_TURN")
        )
        stop_like = bool(stop_goal_active) or normalized_behavior in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }
        lane_follow_like = (
            normalized_behavior == "lane_follow"
            and normalized_fsm in {"", "IDLE", "LANE_KEEP"}
        )
        contract_mode = (
            "emergency_stop"
            if normalized_behavior == "emergency_brake"
            else "stop"
            if bool(stop_like)
            else "lane_change"
            if bool(lane_change_active)
            else "intersection_turn"
            if bool(turn_active)
            else "lane_follow"
        )
        if bool(stop_like):
            stop_reference, stop_destination, stop_reason = self._build_independent_stop_reference(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_state=current_state,
                current_lane_id=int(current_lane_id),
                stop_target=stop_target,
                fallback_destination_state=destination,
                ego_speed_mps=float(ego_speed_mps),
            )
            if stop_reference and stop_destination:
                reference = stop_reference
                destination = stop_destination
                reasons.append(str(stop_reason))

        filtered_reference: list[dict[str, object]] = []
        last_xy: tuple[float, float] | None = None
        min_forward_m = float(self.full_reference_stabilizer_min_forward_m)
        min_spacing_m = float(self.full_reference_stabilizer_min_spacing_m)
        for sample in reference:
            try:
                x_m = float(sample.get("x_ref_m", sample.get("x", ego_location.x)))
                y_m = float(sample.get("y_ref_m", sample.get("y", ego_location.y)))
            except Exception:
                reasons.append("drop_bad_sample")
                continue
            forward_m, _ = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(x_m),
                target_y_m=float(y_m),
            )
            if float(forward_m) < float(min_forward_m):
                reasons.append("drop_behind_sample")
                continue
            if (
                last_xy is not None
                and math.hypot(float(x_m) - last_xy[0], float(y_m) - last_xy[1])
                < float(min_spacing_m)
            ):
                reasons.append("drop_duplicate_sample")
                continue
            clean_sample = dict(sample)
            clean_sample["x_ref_m"] = float(x_m)
            clean_sample["y_ref_m"] = float(y_m)
            clean_sample["x"] = float(x_m)
            clean_sample["y"] = float(y_m)
            filtered_reference.append(clean_sample)
            last_xy = (float(x_m), float(y_m))
        reference = filtered_reference

        if len(reference) < 2:
            reasons.append("too_few_forward_samples")
        if self._reference_opposes_heading(
            reference_samples=reference,
            ego_heading_rad=float(ego_yaw_rad),
            max_heading_error_rad=0.5 * math.pi,
        ):
            reasons.append("reference_opposes_heading")
        if self._reference_has_heading_jump(
            reference_samples=reference,
            max_heading_step_rad=float(self.full_reference_stabilizer_max_heading_step_rad),
        ):
            reasons.append("reference_heading_jump")

        if destination and len(destination) >= 2:
            _, destination_lateral_m = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(destination[0]),
                target_y_m=float(destination[1]),
            )
            destination_threshold_m = None
            if bool(stop_like):
                destination_threshold_m = float(self.full_stop_max_destination_lateral_m)
            elif bool(lane_follow_like):
                destination_threshold_m = float(self.full_lane_follow_max_destination_lateral_m)
            if (
                not bool(lane_change_active)
                and bool(stop_like or lane_follow_like)
                and destination_threshold_m is not None
                and abs(float(destination_lateral_m)) > float(destination_threshold_m)
            ):
                reasons.append(f"destination_lateral_out_of_contract:{destination_lateral_m:.2f}")

        if reference:
            first = dict(reference[0])
            _, first_lateral_m = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(first.get("x_ref_m", first.get("x", ego_location.x))),
                target_y_m=float(first.get("y_ref_m", first.get("y", ego_location.y))),
            )
            first_threshold_m = (
                float(self.full_stop_max_reference_first_lateral_m)
                if bool(stop_like)
                else float(self.full_lane_follow_max_reference_first_lateral_m)
            )
            if (
                not bool(lane_change_active)
                and bool(stop_like or lane_follow_like)
                and abs(float(first_lateral_m)) > float(first_threshold_m)
            ):
                reasons.append(f"first_reference_lateral_out_of_contract:{first_lateral_m:.2f}")

        from cpx_planning.pipeline.reference_contract import (
            contract_from_config,
            validate_reference_contract,
        )

        contract = contract_from_config(
            mode=str(contract_mode),
            expected_lane_id=int(current_lane_id),
            horizon_steps=int(self.mpc.horizon_steps),
            config=dict(self.config),
            default_speed_mps=max(float(self.target_speed_mps), float(speed_ref_mps), 0.1),
        )
        validation = validate_reference_contract(
            reference_samples=reference,
            destination_state=destination,
            ego_state=current_state,
            contract=contract,
            check_destination_body_lateral=bool(stop_like or lane_follow_like),
        )
        if not bool(validation.valid):
            reasons.append("contract_violation:" + str(validation.reason()))
            if bool(stop_like):
                reference, destination = self._build_ego_heading_emergency_stop_reference(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=max(
                        0.5,
                        float(self.mpc.dt_s) * max(0.5, min(float(ego_speed_mps), 1.5)),
                    ),
                )
                reasons.append("stop_hard_lock_ego_heading_emergency_reference")
                validation = validate_reference_contract(
                    reference_samples=reference,
                    destination_state=destination,
                    ego_state=current_state,
                    contract=contract,
                    check_destination_body_lateral=True,
                )
                if not bool(validation.valid):
                    reasons.append("emergency_reference_contract_violation:" + str(validation.reason()))
            elif bool(self.strict_reference_validator_veto_enabled):
                reasons.append("strict_reference_veto")
                compact_reasons = []
                for reason in reasons:
                    if reason not in compact_reasons:
                        compact_reasons.append(reason)
                return list(destination), list(reference), ";".join(compact_reasons)
            elif bool(turn_active):
                step_distance_m = max(
                    0.5,
                    float(self.mpc.dt_s)
                    * max(1.0, min(float(speed_ref_mps), float(self.target_speed_mps))),
                )
                turn_direction = (
                    "left"
                    if normalized_behavior.endswith("_left") or normalized_fsm.endswith("_LEFT")
                    else "right"
                    if normalized_behavior.endswith("_right") or normalized_fsm.endswith("_RIGHT")
                    else ""
                )
                rebuilt_reference = self._ego_anchored_turn_reference_samples(
                    ego_location=ego_location,
                    ego_heading_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                    turn_direction=str(turn_direction),
                    route_points=self._active_global_route_points(),
                )
                rebuild_source = "rebuilt_ego_anchored_turn_reference"
                if not rebuilt_reference:
                    rebuilt_reference = self._route_aligned_reference_samples(
                        ego_location=ego_location,
                        ego_heading_rad=float(ego_yaw_rad),
                        current_lane_id=int(current_lane_id),
                        horizon_steps=int(self.mpc.horizon_steps),
                        step_distance_m=float(step_distance_m),
                        route_points=self._active_global_route_points(),
                    )
                    rebuild_source = "rebuilt_global_route_turn_reference"
                if rebuilt_reference:
                    from cpx_planning.behavior_planner.reference_pipeline import (
                        lane_center_destination_from_reference,
                    )

                    reference = [dict(sample) for sample in rebuilt_reference]
                    aligned_destination = lane_center_destination_from_reference(
                        destination_state=destination,
                        lane_center_reference=reference,
                        ego_state=current_state,
                        target_forward_m=float(
                            self.config.get("full_turn_guard_destination_forward_m", 10.0)
                        ),
                    )
                    if aligned_destination is not None:
                        destination = list(aligned_destination)
                    reasons.append(str(rebuild_source))
                    validation = validate_reference_contract(
                        reference_samples=reference,
                        destination_state=destination,
                        ego_state=current_state,
                        contract=contract,
                        check_destination_body_lateral=False,
                    )
                    if not bool(validation.valid):
                        reasons.append("turn_route_reference_contract_violation:" + str(validation.reason()))
                        creep_reference = self._creep_turn_reference_samples(
                            ego_location=ego_location,
                            ego_heading_rad=float(ego_yaw_rad),
                            current_lane_id=int(current_lane_id),
                            horizon_steps=int(self.mpc.horizon_steps),
                            turn_direction=str(turn_direction),
                        )
                        if creep_reference:
                            reference = [dict(sample) for sample in creep_reference]
                            aligned_destination = lane_center_destination_from_reference(
                                destination_state=destination,
                                lane_center_reference=reference,
                                ego_state=current_state,
                                target_forward_m=float(
                                    self.config.get("full_turn_creep_destination_forward_m", 7.0)
                                ),
                            )
                            if aligned_destination is not None:
                                destination = list(aligned_destination)
                            reasons.append("creep_turn_reference")
                            validation = validate_reference_contract(
                                reference_samples=reference,
                                destination_state=destination,
                                ego_state=current_state,
                                contract=contract,
                                check_destination_body_lateral=False,
                            )
                            if not bool(validation.valid):
                                reasons.append("creep_turn_reference_contract_violation:" + str(validation.reason()))
                else:
                    reasons.append("turn_route_rebuild_failed")
                    creep_reference = self._creep_turn_reference_samples(
                        ego_location=ego_location,
                        ego_heading_rad=float(ego_yaw_rad),
                        current_lane_id=int(current_lane_id),
                        horizon_steps=int(self.mpc.horizon_steps),
                        turn_direction=str(turn_direction),
                    )
                    if creep_reference:
                        from cpx_planning.behavior_planner.reference_pipeline import (
                            lane_center_destination_from_reference,
                        )

                        reference = [dict(sample) for sample in creep_reference]
                        aligned_destination = lane_center_destination_from_reference(
                            destination_state=destination,
                            lane_center_reference=reference,
                            ego_state=current_state,
                            target_forward_m=float(
                                self.config.get("full_turn_creep_destination_forward_m", 7.0)
                            ),
                        )
                        if aligned_destination is not None:
                            destination = list(aligned_destination)
                        reasons.append("creep_turn_reference")

        needs_rebuild = (
            bool(reasons)
            and not bool(lane_change_active)
            and not bool(stop_like)
            and not bool(turn_active)
            and not bool(self.strict_reference_validator_veto_enabled)
        )
        if bool(needs_rebuild):
            start_waypoint = self._map_waypoint_from_location(ego_location)
            step_distance_m = max(
                0.5,
                float(self.mpc.dt_s)
                * max(1.0, min(float(speed_ref_mps), float(self.target_speed_mps))),
            )
            rebuilt_reference = self._current_lane_center_reference_samples(
                start_waypoint=start_waypoint,
                current_lane_id=int(current_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                step_distance_m=float(step_distance_m),
                route_points=self._active_global_route_points(),
            )
            if rebuilt_reference:
                from cpx_planning.behavior_planner.reference_pipeline import (
                    lane_center_destination_from_reference,
                )

                reference = [dict(sample) for sample in rebuilt_reference]
                target_forward_m = (
                    float(self.config.get("full_stop_guard_destination_forward_m", 6.0))
                    if bool(stop_like)
                    else float(
                        self.config.get(
                            "full_lane_follow_guard_destination_forward_m",
                            8.0,
                        )
                    )
                )
                aligned_destination = lane_center_destination_from_reference(
                    destination_state=destination,
                    lane_center_reference=reference,
                    ego_state=current_state,
                    target_forward_m=float(target_forward_m),
                )
                if aligned_destination is not None:
                    destination = list(aligned_destination)
                reasons.append("rebuilt_current_lane_reference")
            else:
                reasons.append("rebuild_failed")

        if destination and len(destination) >= 3 and bool(stop_like):
            destination[2] = 0.0
            for sample in reference:
                sample["v_ref_mps"] = 0.0
                sample["speed_ref_mps"] = 0.0
                sample["speed_mps"] = 0.0

        compact_reasons = []
        for reason in reasons:
            if reason not in compact_reasons:
                compact_reasons.append(reason)
        return list(destination), list(reference), ";".join(compact_reasons)

    def _build_independent_stop_reference(
            self,
            *,
            ego_location: _PlannerLocation,
            ego_yaw_rad: float,
            current_state: Sequence[float],
            current_lane_id: int,
            stop_target: Mapping[str, object] | None,
            fallback_destination_state: Sequence[float],
            ego_speed_mps: float,
        ) -> tuple[list[dict[str, object]], list[float], str]:
        stop_buffer_m = max(0.0, float(self.config.get("full_stop_reference_buffer_m", 1.5)))
        comfortable_decel_mps2 = max(
            0.1,
            float(self.config.get("full_stop_reference_decel_mps2", 2.0)),
        )
        stop_speed_cap_mps = max(
            0.1,
            float(self.config.get("full_stop_reference_speed_cap_mps", 2.0)),
        )
        stop_forward_m, stop_target_reliable = self._stop_target_forward_m(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            stop_target=stop_target,
            fallback_destination_state=fallback_destination_state,
        )
        if not bool(stop_target_reliable):
            emergency_reference, emergency_destination = self._build_ego_heading_emergency_stop_reference(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_lane_id=int(current_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                step_distance_m=max(
                    0.5,
                    float(self.mpc.dt_s) * max(0.5, min(float(ego_speed_mps), 1.5)),
                ),
            )
            return (
                emergency_reference,
                emergency_destination,
                "independent_stop_reference;stop_missing_target_hard_lock",
            )
        stop_forward_m = max(0.5, float(stop_forward_m) - float(stop_buffer_m))
        step_distance_m = max(
            0.5,
            float(self.mpc.dt_s) * max(1.0, min(float(ego_speed_mps), stop_speed_cap_mps)),
        )
        start_waypoint = self._map_waypoint_from_location(ego_location)
        lane_reference = self._current_lane_center_reference_samples(
            start_waypoint=start_waypoint,
            current_lane_id=int(current_lane_id),
            horizon_steps=int(self.mpc.horizon_steps),
            step_distance_m=float(step_distance_m),
            route_points=[],
        )
        if not lane_reference:
            lane_reference = self._straight_reference_samples(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_lane_id=int(current_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                step_distance_m=float(step_distance_m),
            )

        shaped_reference: list[dict[str, object]] = []
        chosen_destination = None
        best_distance_error_m = float("inf")
        previous_speed_mps = min(float(stop_speed_cap_mps), max(0.0, float(ego_speed_mps)))
        for sample in lane_reference:
            shaped = dict(sample)
            x_m = float(shaped.get("x_ref_m", shaped.get("x", ego_location.x)))
            y_m = float(shaped.get("y_ref_m", shaped.get("y", ego_location.y)))
            forward_m, _ = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(x_m),
                target_y_m=float(y_m),
            )
            remaining_m = max(0.0, float(stop_forward_m) - float(forward_m))
            speed_ref_mps = min(
                float(stop_speed_cap_mps),
                math.sqrt(max(0.0, 2.0 * float(comfortable_decel_mps2) * float(remaining_m))),
                float(previous_speed_mps),
            )
            if float(forward_m) >= float(stop_forward_m):
                speed_ref_mps = 0.0
            previous_speed_mps = float(speed_ref_mps)
            shaped["v_ref_mps"] = float(speed_ref_mps)
            shaped["speed_ref_mps"] = float(speed_ref_mps)
            shaped["speed_mps"] = float(speed_ref_mps)
            shaped["lane_id"] = int(current_lane_id)
            shaped_reference.append(shaped)
            distance_error_m = abs(float(forward_m) - float(stop_forward_m))
            if distance_error_m < best_distance_error_m:
                best_distance_error_m = float(distance_error_m)
                chosen_destination = dict(shaped)

        if shaped_reference:
            shaped_reference[-1]["v_ref_mps"] = 0.0
            shaped_reference[-1]["speed_ref_mps"] = 0.0
            shaped_reference[-1]["speed_mps"] = 0.0
        if chosen_destination is None and shaped_reference:
            chosen_destination = dict(shaped_reference[-1])

        if chosen_destination is not None:
            destination = [
                float(chosen_destination.get("x_ref_m", chosen_destination.get("x", ego_location.x))),
                float(chosen_destination.get("y_ref_m", chosen_destination.get("y", ego_location.y))),
                0.0,
                float(chosen_destination.get("heading_rad", ego_yaw_rad)),
                int(current_lane_id),
            ]
        else:
            destination = [
                float(ego_location.x) + float(stop_forward_m) * math.cos(float(ego_yaw_rad)),
                float(ego_location.y) + float(stop_forward_m) * math.sin(float(ego_yaw_rad)),
                0.0,
                float(ego_yaw_rad),
                int(current_lane_id),
            ]
        return shaped_reference, destination, "independent_stop_reference"

    def _current_lane_center_reference_samples(self, *, start_waypoint, current_lane_id, horizon_steps, step_distance_m, route_points=None):
        """Use the copied lane-center traversal with custom global-planner waypoints."""
        if start_waypoint is None:
            return []
        samples = []
        current = start_waypoint
        step_m = max(0.5, float(step_distance_m))
        previous_heading = float(world_heading_rad(current) or 0.0)
        first_step_m = max(step_m, float(self.config.get("lane_follow_reference_first_point_m", 2.0)))
        first_candidates = list(current.next(first_step_m) or [])
        if first_candidates:
            first_current = self._select_smooth_next_waypoint(current_waypoint=current, candidates=first_candidates, previous_heading_rad=previous_heading, route_points=route_points)
            if first_current is not None:
                current = first_current
                previous_heading = float(world_heading_rad(current) or previous_heading)
        for _ in range(max(1, int(horizon_steps)) + 1):
            xy_heading = self._waypoint_xy_heading(current)
            if xy_heading is None:
                break
            lane_width_m = self._waypoint_lane_width(current)
            heading_rad = float(world_heading_rad(current) or xy_heading[2])
            lane_id = int(canonical_lane_id_for_waypoint(current) or current_lane_id)
            samples.append({"x_ref_m": float(xy_heading[0]), "y_ref_m": float(xy_heading[1]), "x": float(xy_heading[0]), "y": float(xy_heading[1]), "heading_rad": heading_rad, "lane_id": lane_id, "lane_width_m": lane_width_m, "road_center_offset_m": 0.0, "road_left_width_m": 0.5 * lane_width_m, "road_right_width_m": 0.5 * lane_width_m})
            candidates = list(current.next(step_m) or [])
            if not candidates:
                break
            current = self._select_smooth_next_waypoint(current_waypoint=current, candidates=candidates, previous_heading_rad=previous_heading, route_points=route_points)
            if current is None:
                break
            previous_heading = float(world_heading_rad(current) or previous_heading)
        return samples

    @staticmethod
    def _reference_opposes_heading(
            *,
            reference_samples: Sequence[Mapping[str, Any]],
            ego_heading_rad: float,
            max_heading_error_rad: float,
        ) -> bool:
        points = [
            (
                float(sample.get("x_ref_m", sample.get("x", 0.0))),
                float(sample.get("y_ref_m", sample.get("y", 0.0))),
            )
            for sample in list(reference_samples or [])
        ]
        for first, second in zip(points[:-1], points[1:]):
            dx = float(second[0]) - float(first[0])
            dy = float(second[1]) - float(first[1])
            if math.hypot(dx, dy) <= 0.25:
                continue
            reference_heading = math.atan2(dy, dx)
            error = CPXMPCPlannerBridge._wrap_angle_static(
                float(ego_heading_rad) - float(reference_heading)
            )
            return abs(float(error)) > float(max_heading_error_rad)
        return False

    @staticmethod
    def _reference_has_heading_jump(
            *,
            reference_samples: Sequence[Mapping[str, object]],
            max_heading_step_rad: float,
        ) -> bool:
        points = [
            (
                float(sample.get("x_ref_m", sample.get("x", 0.0))),
                float(sample.get("y_ref_m", sample.get("y", 0.0))),
            )
            for sample in list(reference_samples or [])
        ]
        if len(points) < 3:
            return False
        previous_heading = None
        for first, second in zip(points[:-1], points[1:]):
            dx_m = float(second[0]) - float(first[0])
            dy_m = float(second[1]) - float(first[1])
            if math.hypot(dx_m, dy_m) <= 0.25:
                continue
            heading = math.atan2(dy_m, dx_m)
            if previous_heading is not None:
                delta = CPXMPCPlannerBridge._wrap_angle_static(
                    float(heading) - float(previous_heading)
                )
                if abs(float(delta)) > float(max_heading_step_rad):
                    return True
            previous_heading = float(heading)
        return False

    def _control_from_mpc(self, acceleration_mps2: float, steering_angle_rad: float) -> PlannerControl:
        max_accel = max(1e-6, float(self.mpc.constraints.max_acceleration_mps2))
        max_brake = max(1e-6, abs(float(self.mpc.constraints.min_acceleration_mps2)))
        max_steer = max(1e-6, float(self.mpc.constraints.max_steer_rad))
        throttle = min(1.0, max(0.0, float(acceleration_mps2) / max_accel))
        brake = min(1.0, max(0.0, -float(acceleration_mps2) / max_brake))
        steer = min(1.0, max(-1.0, float(steering_angle_rad) / max_steer))
        return PlannerControl(throttle=throttle, brake=brake, steer=steer)

    def _accel_from_control(self, control: PlannerControl) -> float:
        max_accel = max(1e-6, float(self.mpc.constraints.max_acceleration_mps2))
        max_brake = max(1e-6, abs(float(self.mpc.constraints.min_acceleration_mps2)))
        throttle_accel = float(getattr(control, "throttle", 0.0)) * float(max_accel)
        brake_accel = float(getattr(control, "brake", 0.0)) * float(max_brake)
        return float(throttle_accel - brake_accel)

    def _steer_rad_from_control(self, control: PlannerControl) -> float:
        max_steer = max(1e-6, float(self.mpc.constraints.max_steer_rad))
        return float(getattr(control, "steer", 0.0)) * float(max_steer)

    def _emergency_stop_control(self) -> PlannerControl:
        self._last_accel_mps2 = float(getattr(self.mpc.constraints, "min_acceleration_mps2", -3.0))
        self._last_steer_rad = 0.0
        return PlannerControl(throttle=0.0, brake=1.0, steer=0.0)

    def _last_mpc_trajectory_points(self) -> list[tuple[float, float]]:
        x_solution = getattr(self.mpc, "_last_x_solution", None)
        if x_solution is None:
            return []
        points: list[tuple[float, float]] = []
        try:
            for state in list(x_solution):
                if len(state) < 2:
                    continue
                points.append((float(state[0]), float(state[1])))
        except Exception:
            return []
        return points

    def _apply_control_safety_guards(
            self,
            *,
            control: PlannerControl,
            accel_mps2: float,
            steer_rad: float,
            ego_transform: _PlannerTransform,
            ego_speed_mps: float,
            speed_ref_mps: float,
            destination_state: Sequence[float],
            destination_lateral_m: float,
            stop_goal_active: bool,
            behavior_decision: str,
            behavior_fsm_state: str,
            traffic_signal_state: str = "",
            sim_time_s: float = 0.0,
        ) -> tuple[PlannerControl, float, float, str]:
        """Apply last-mile control guards before sending commands to CARLA.

        These guards do not change the behavior decision or MPC reference. They
        only prevent two observed failure modes in the OpenCDA bridge:
        persistent speed overshoot and saturated low-speed steering while the
        planner is nominally in lane-follow.
        """

        max_brake_accel = max(1.0e-6, abs(float(self.mpc.constraints.min_acceleration_mps2)))
        max_steer = max(1.0e-6, float(self.mpc.constraints.max_steer_rad))
        normalized_signal = str(traffic_signal_state or "").strip().lower()
        normalized_behavior = str(behavior_decision or "").strip().lower()

        if bool(stop_goal_active):
            if normalized_signal in {"red", "yellow"}:
                stop_distance_m = 1.0
                if destination_state is not None and len(destination_state) >= 2:
                    stop_forward_m, _ = self._body_frame_xy(
                        origin_x_m=float(ego_transform.location.x),
                        origin_y_m=float(ego_transform.location.y),
                        heading_rad=math.radians(float(ego_transform.rotation.yaw)),
                        target_x_m=float(destination_state[0]),
                        target_y_m=float(destination_state[1]),
                    )
                    stop_distance_m = max(1.0, float(stop_forward_m))
                stop_buffer_m = max(
                    0.0,
                    float(self.config.get("red_yellow_stop_guard_buffer_m", 1.5)),
                )
                remaining_m = max(0.0, float(stop_distance_m) - float(stop_buffer_m))
                comfortable_decel_mps2 = max(
                    0.1,
                    float(self.config.get("red_yellow_stop_guard_comfort_decel_mps2", 1.6)),
                )
                approach_speed_cap_mps = max(
                    0.1,
                    float(self.config.get("red_yellow_stop_guard_approach_speed_mps", 1.5)),
                )
                target_speed_mps = min(
                    float(approach_speed_cap_mps),
                    math.sqrt(max(0.0, 2.0 * float(comfortable_decel_mps2) * float(remaining_m))),
                )
                if float(remaining_m) <= 0.05:
                    target_speed_mps = 0.0
                speed_error_mps = float(target_speed_mps) - float(ego_speed_mps)
                desired_accel = float(
                    self.config.get("red_yellow_stop_guard_speed_kp", 0.8)
                ) * float(speed_error_mps)
                max_approach_accel_mps2 = float(
                    self.config.get("red_yellow_stop_guard_approach_max_accel_mps2", 0.45)
                )
                min_decel_mps2 = -min(
                    float(max_brake_accel),
                    float(self.config.get("red_yellow_stop_guard_max_decel_mps2", 2.2)),
                )
                guarded_accel = min(
                    float(max_approach_accel_mps2),
                    max(float(min_decel_mps2), float(desired_accel)),
                )
                reason = (
                    "red_yellow_stop_final_brake_envelope"
                    if float(target_speed_mps) <= 0.05
                    else "red_yellow_stop_braking_envelope"
                )
                return (
                    self._control_from_mpc(float(guarded_accel), float(steer_rad)),
                    float(guarded_accel),
                    float(steer_rad),
                    str(reason),
                )
            return control, float(accel_mps2), float(steer_rad), ""

        if bool(self.overspeed_guard_enabled) and float(speed_ref_mps) > 0.1:
            overspeed_mps = float(ego_speed_mps) - float(speed_ref_mps)
            if overspeed_mps > float(self.overspeed_margin_mps):
                brake = min(
                    float(self.overspeed_max_brake),
                    max(
                        float(self.overspeed_min_brake),
                        float(self.overspeed_min_brake)
                        + float(self.overspeed_brake_gain)
                        * (overspeed_mps - float(self.overspeed_margin_mps)),
                    ),
                )
                guarded_steer_rad = min(
                    max_steer,
                    max(-max_steer, float(steer_rad)),
                )
                guarded_control = self._control_from_mpc(
                    -float(brake) * max_brake_accel,
                    guarded_steer_rad,
                )
                return (
                    guarded_control,
                    -float(brake) * max_brake_accel,
                    float(guarded_steer_rad),
                    "overspeed_guard",
                )

        normalized_fsm = str(behavior_fsm_state or "").strip().upper()
        lane_follow_like = (
            normalized_behavior == "lane_follow"
            and normalized_fsm in {"", "IDLE", "LANE_KEEP"}
        )
        launch_guard_reason = self._low_speed_launch_ramp_reason(
            lane_follow_like=bool(lane_follow_like),
            ego_transform=ego_transform,
            ego_speed_mps=float(ego_speed_mps),
            speed_ref_mps=float(speed_ref_mps),
            accel_mps2=float(accel_mps2),
            sim_time_s=float(sim_time_s),
        )
        if str(launch_guard_reason):
            max_launch_accel = min(
                float(self.mpc.constraints.max_acceleration_mps2),
                float(self.config.get("full_low_speed_launch_max_accel_mps2", 1.0)),
            )
            guarded_accel = min(float(accel_mps2), float(max_launch_accel))
            guarded_accel = max(
                float(self.config.get("full_low_speed_launch_min_accel_mps2", 0.45)),
                float(guarded_accel),
            )
            return (
                self._control_from_mpc(float(guarded_accel), float(steer_rad)),
                float(guarded_accel),
                float(steer_rad),
                str(launch_guard_reason),
            )
        if (
            bool(self.full_low_speed_launch_enabled)
            and bool(lane_follow_like)
            and float(speed_ref_mps) > float(ego_speed_mps) + 0.2
            and float(ego_speed_mps) < float(self.full_low_speed_launch_speed_mps)
            and float(accel_mps2) < float(self.full_low_speed_launch_min_accel_mps2)
            and float(accel_mps2) > -0.05
        ):
            launch_accel = min(
                float(self.mpc.constraints.max_acceleration_mps2),
                float(self.full_low_speed_launch_min_accel_mps2),
            )
            return (
                self._control_from_mpc(float(launch_accel), float(steer_rad)),
                float(launch_accel),
                float(steer_rad),
                "low_speed_launch_guard",
            )
        if (
            bool(self.low_speed_lateral_recovery_enabled)
            and bool(lane_follow_like)
            and float(ego_speed_mps) < float(self.low_speed_lateral_recovery_speed_mps)
            and abs(float(destination_lateral_m))
            > float(self.low_speed_lateral_recovery_threshold_m)
            and len(destination_state) >= 2
        ):
            dx = float(destination_state[0]) - float(ego_transform.location.x)
            dy = float(destination_state[1]) - float(ego_transform.location.y)
            target_yaw = math.atan2(dy, dx)
            yaw_error = self._wrap_angle(
                target_yaw - math.radians(float(ego_transform.rotation.yaw))
            )
            max_recovery_steer = min(
                max_steer,
                max(0.01, float(self.low_speed_lateral_recovery_max_steer_rad)),
            )
            guarded_steer_rad = min(
                max_recovery_steer,
                max(-max_recovery_steer, 0.45 * float(yaw_error)),
            )
            target_speed = max(0.1, float(self.low_speed_lateral_recovery_target_speed_mps))
            accel = min(
                float(self.mpc.constraints.max_acceleration_mps2),
                max(
                    0.15,
                    0.55 * (target_speed - float(ego_speed_mps)),
                ),
            )
            return (
                self._control_from_mpc(float(accel), float(guarded_steer_rad)),
                float(accel),
                float(guarded_steer_rad),
                "low_speed_lateral_recovery",
            )

        return control, float(accel_mps2), float(steer_rad), ""

    def _low_speed_launch_ramp_reason(
            self,
            *,
            lane_follow_like: bool,
            ego_transform: _PlannerTransform,
            ego_speed_mps: float,
            speed_ref_mps: float,
            accel_mps2: float,
            sim_time_s: float,
        ) -> str:
        if not bool(self.config.get("full_low_speed_launch_ramp_enabled", True)):
            self._full_launch_start_s = None
            self._full_launch_start_xy = None
            return ""
        if (
            not bool(lane_follow_like)
            or float(speed_ref_mps) <= 0.2
            or float(ego_speed_mps) >= float(self.config.get("full_low_speed_launch_ramp_speed_mps", 0.25))
            or float(accel_mps2) <= float(self.config.get("full_low_speed_launch_ramp_trigger_accel_mps2", 0.6))
        ):
            self._full_launch_start_s = None
            self._full_launch_start_xy = None
            return ""
        x_m = float(ego_transform.location.x)
        y_m = float(ego_transform.location.y)
        if self._full_launch_start_s is None or self._full_launch_start_xy is None:
            self._full_launch_start_s = float(sim_time_s)
            self._full_launch_start_xy = (float(x_m), float(y_m))
            return "low_speed_launch_ramp"
        elapsed_s = max(0.0, float(sim_time_s) - float(self._full_launch_start_s))
        moved_m = math.hypot(
            float(x_m) - float(self._full_launch_start_xy[0]),
            float(y_m) - float(self._full_launch_start_xy[1]),
        )
        if (
            elapsed_s >= float(self.config.get("full_low_speed_launch_stuck_s", 0.8))
            and moved_m <= float(self.config.get("full_low_speed_launch_stuck_distance_m", 0.15))
        ):
            return "low_speed_launch_stuck_ramp"
        return "low_speed_launch_ramp"


    def _apply_mpc_cost_profile(
            self,
            *,
            behavior: str,
            planner_lc_state: str,
            planner_mode: str,
            next_macro_maneuver: str,
            sim_time_s: float,
        ) -> None:
        requested = _mpc_cost_profile_for_behavior(
            behavior=behavior,
            planner_lc_state=planner_lc_state,
            planner_mode=planner_mode,
            next_macro_maneuver=next_macro_maneuver,
        )
        (
            self.active_mpc_cost_profile,
            self.mpc_cost_profile_active_since_s,
            self.mpc_cost_profile_switch_reason,
        ) = _select_mpc_cost_profile_with_hysteresis(
            requested_profile=str(requested),
            active_profile=str(self.active_mpc_cost_profile),
            sim_time_s=float(sim_time_s),
            active_since_s=float(self.mpc_cost_profile_active_since_s),
            min_hold_s=float(self.behavior_runtime_cfg.get("mpc_cost_profile_min_hold_s", 1.5)),
        )
        self.requested_mpc_cost_profile = str(requested)
        if hasattr(self.mpc, "apply_mode_cost_profile"):
            self.active_mpc_cost_profile = str(
                self.mpc.apply_mode_cost_profile(str(self.active_mpc_cost_profile))
            )



    def _full_latched_stop_target_for_signal(
        self,
        *,
        traffic_state: str,
        stop_target: Mapping[str, object] | None,
        ego_location: _PlannerLocation,
        ego_yaw_rad: float,
        current_lane_id: int,
    ) -> tuple[dict[str, object] | None, str]:
        state = str(traffic_state or "unknown").strip().lower()

        if state not in {"red", "yellow"}:
            if self._full_latched_stop_target is not None:
                self._full_latched_stop_target = None
                self._full_latched_stop_state = str(state)
                return None, "stop_target_latch_release"

            self._full_latched_stop_state = str(state)
            return None, ""

        if self._full_latched_stop_target is not None and self._full_latched_stop_state in {"red", "yellow"}:
            return dict(self._full_latched_stop_target), "stop_target_latch_reuse"

        latched: dict[str, object] | None = None
        if isinstance(stop_target, Mapping):
            try:
                x_value = stop_target.get("x_m", stop_target.get("x", None))
                y_value = stop_target.get("y_m", stop_target.get("y", None))

                if x_value is not None and y_value is not None:
                    latched = dict(stop_target)
                    latched["x_m"] = float(x_value)
                    latched["y_m"] = float(y_value)
                    latched["x"] = float(x_value)
                    latched["y"] = float(y_value)
                    latched["source"] = str(latched.get("source", "")) + ":latched_world_stop_target"
            except Exception:
                latched = None

        if latched is None:
            distance_m = max(2.0, float(self.config.get("full_latched_virtual_stop_distance_m", 12.0)))
            x_m = float(ego_location.x) + float(distance_m) * math.cos(float(ego_yaw_rad))
            y_m = float(ego_location.y) + float(distance_m) * math.sin(float(ego_yaw_rad))
            latched = {
                "x_m": float(x_m),
                "y_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(ego_yaw_rad),
                "lane_id": int(current_lane_id),
                "distance_m": float(distance_m),
                "source": "latched_virtual_stop_target",
            }

        self._full_latched_stop_target = dict(latched)
        self._full_latched_stop_state = str(state)
        return dict(latched), "stop_target_latch_create"

    def _stop_target_forward_m(
            self,
            *,
            ego_location: _PlannerLocation,
            ego_yaw_rad: float,
            stop_target: Mapping[str, object] | None,
            fallback_destination_state: Sequence[float],
        ) -> tuple[float, bool]:
        if isinstance(stop_target, Mapping):
            try:
                has_x = "x_m" in stop_target or "x" in stop_target
                has_y = "y_m" in stop_target or "y" in stop_target
                if not bool(has_x and has_y) and "distance_m" in stop_target:
                    return max(0.0, float(stop_target.get("distance_m", 0.0))), True
                target_x = float(stop_target.get("x_m", stop_target.get("x", ego_location.x)))
                target_y = float(stop_target.get("y_m", stop_target.get("y", ego_location.y)))
                forward_m, _ = self._body_frame_xy(
                    origin_x_m=float(ego_location.x),
                    origin_y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    target_x_m=float(target_x),
                    target_y_m=float(target_y),
                )
                return max(0.0, float(forward_m)), True
            except Exception:
                pass
        if fallback_destination_state is not None and len(fallback_destination_state) >= 2:
            try:
                forward_m, _ = self._body_frame_xy(
                    origin_x_m=float(ego_location.x),
                    origin_y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    target_x_m=float(fallback_destination_state[0]),
                    target_y_m=float(fallback_destination_state[1]),
                )
                return max(0.0, float(forward_m)), False
            except Exception:
                pass
        return max(2.0, float(self.config.get("full_stop_guard_destination_forward_m", 6.0))), False

    @staticmethod
    def _route_option_turn_decision(*, current_road_option: str, next_macro_maneuver: str) -> str:
        route_option = str(current_road_option or "").strip().upper()
        macro = str(next_macro_maneuver or "").strip().lower()
        if route_option == "LEFT" or macro == "turn left":
            return "intersection_turn_left"
        if route_option == "RIGHT" or macro == "turn right":
            return "intersection_turn_right"
        return ""

    def _apply_turn_direction_latch(
            self,
            *,
            decision: str,
            lc_state: str,
            speed_ref_mps: float,
            current_road_option: str,
            next_macro_maneuver: str,
            ego_in_junction: bool,
            sim_time_s: float,
        ) -> tuple[str, str, float, str]:
        if not bool(self.config.get("full_intersection_turn_latch_enabled", True)):
            self._turn_latch_decision = ""
            self._turn_latch_until_sim_time_s = -float("inf")
            return str(decision), str(lc_state), float(speed_ref_mps), ""
        explicit_turn = self._route_option_turn_decision(
            current_road_option=str(current_road_option),
            next_macro_maneuver=str(next_macro_maneuver),
        )
        normalized_decision = str(decision or "").strip().lower()
        if str(explicit_turn):
            self._turn_latch_decision = str(explicit_turn)
            self._turn_latch_until_sim_time_s = float(sim_time_s) + float(
                self.config.get("full_intersection_turn_latch_hold_s", 6.0)
            )
            latched_lc_state = (
                "INTERSECTION_TURN_LEFT"
                if str(explicit_turn).endswith("_left")
                else "INTERSECTION_TURN_RIGHT"
            )
            capped_speed = min(
                float(speed_ref_mps),
                float(self.config.get("full_intersection_turn_speed_cap_mps", 2.0)),
            )
            return str(explicit_turn), str(latched_lc_state), float(capped_speed), "turn_latch_set:" + str(explicit_turn)
        latch_active = (
            str(self._turn_latch_decision)
            and float(sim_time_s) <= float(self._turn_latch_until_sim_time_s)
            and (bool(ego_in_junction) or normalized_decision.startswith("intersection_turn"))
        )
        if bool(latch_active):
            latched = str(self._turn_latch_decision)
            latched_lc_state = (
                "INTERSECTION_TURN_LEFT"
                if str(latched).endswith("_left")
                else "INTERSECTION_TURN_RIGHT"
            )
            capped_speed = min(
                float(speed_ref_mps),
                float(self.config.get("full_intersection_turn_speed_cap_mps", 2.0)),
            )
            return str(latched), str(latched_lc_state), float(capped_speed), "turn_latch_hold:" + str(latched)
        if str(self._turn_latch_decision):
            self._turn_latch_decision = ""
            self._turn_latch_until_sim_time_s = -float("inf")
            return str(decision), str(lc_state), float(speed_ref_mps), "turn_latch_release"
        return str(decision), str(lc_state), float(speed_ref_mps), ""

    def _route_lookahead_turn_decision(
            self,
            *,
            ego_location: _PlannerLocation,
            ego_heading_rad: float,
            route_points: Sequence[Sequence[float]],
        ) -> str:
        route_xy = [
            (float(point[0]), float(point[1]))
            for point in list(route_points or [])
            if len(point) >= 2
        ]
        if len(route_xy) < 4:
            return ""
        route_progress = [0.0]
        for first, second in zip(route_xy[:-1], route_xy[1:]):
            route_progress.append(
                float(route_progress[-1])
                + math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))
            )
        base_s = self._project_point_to_polyline_s(
            route_xy=route_xy,
            route_progress=route_progress,
            point_xy=(float(ego_location.x), float(ego_location.y)),
        )
        lookahead_m = float(self.config.get("full_intersection_turn_prepare_lookahead_m", 28.0))
        min_prepare_m = float(self.config.get("full_intersection_turn_prepare_min_m", 6.0))
        start_heading = float(ego_heading_rad)
        near_heading = self._route_heading_at_s(
            route_xy=route_xy,
            route_progress=route_progress,
            target_s=float(base_s) + float(min_prepare_m),
            fallback_heading_rad=float(start_heading),
        )
        far_heading = self._route_heading_at_s(
            route_xy=route_xy,
            route_progress=route_progress,
            target_s=float(base_s) + float(lookahead_m),
            fallback_heading_rad=float(near_heading),
        )
        delta = self._wrap_angle(float(far_heading) - float(near_heading))
        threshold = float(self.config.get("full_intersection_turn_prepare_heading_delta_rad", 0.45))
        if delta > float(threshold):
            return "intersection_turn_left"
        if delta < -float(threshold):
            return "intersection_turn_right"
        return ""

    @staticmethod
    def _route_heading_at_s(
            *,
            route_xy: Sequence[tuple[float, float]],
            route_progress: Sequence[float],
            target_s: float,
            fallback_heading_rad: float,
        ) -> float:
        if len(route_xy) < 2:
            return float(fallback_heading_rad)
        clamped_s = min(max(float(target_s), float(route_progress[0])), float(route_progress[-1]))
        for index in range(len(route_progress) - 1):
            if float(route_progress[index + 1]) + 1.0e-6 < float(clamped_s):
                continue
            first = route_xy[index]
            second = route_xy[index + 1]
            dx = float(second[0]) - float(first[0])
            dy = float(second[1]) - float(first[1])
            if math.hypot(dx, dy) > 1.0e-6:
                return math.atan2(dy, dx)
        first = route_xy[-2]
        second = route_xy[-1]
        dx = float(second[0]) - float(first[0])
        dy = float(second[1]) - float(first[1])
        if math.hypot(dx, dy) > 1.0e-6:
            return math.atan2(dy, dx)
        return float(fallback_heading_rad)

    @staticmethod
    def _project_point_to_polyline_s(
            *,
            route_xy: Sequence[tuple[float, float]],
            route_progress: Sequence[float],
            point_xy: tuple[float, float],
        ) -> float:
        best_distance_m = float("inf")
        best_s = float(route_progress[0]) if route_progress else 0.0
        px, py = float(point_xy[0]), float(point_xy[1])
        for idx, (first, second) in enumerate(zip(route_xy[:-1], route_xy[1:])):
            ax, ay = float(first[0]), float(first[1])
            bx, by = float(second[0]), float(second[1])
            dx = bx - ax
            dy = by - ay
            segment_len_sq = dx * dx + dy * dy
            if segment_len_sq <= 1.0e-9:
                continue
            ratio = ((px - ax) * dx + (py - ay) * dy) / segment_len_sq
            ratio = min(1.0, max(0.0, float(ratio)))
            proj_x = ax + ratio * dx
            proj_y = ay + ratio * dy
            distance_m = math.hypot(px - proj_x, py - proj_y)
            if distance_m < best_distance_m:
                best_distance_m = float(distance_m)
                segment_len_m = math.sqrt(segment_len_sq)
                best_s = float(route_progress[idx]) + ratio * segment_len_m
        return float(best_s)

    @staticmethod
    def _body_frame_xy(
            *,
            origin_x_m: float,
            origin_y_m: float,
            heading_rad: float,
            target_x_m: float,
            target_y_m: float,
        ) -> tuple[float, float]:
        dx_m = float(target_x_m) - float(origin_x_m)
        dy_m = float(target_y_m) - float(origin_y_m)
        cos_h = math.cos(float(heading_rad))
        sin_h = math.sin(float(heading_rad))
        forward_m = dx_m * cos_h + dy_m * sin_h
        lateral_m = -dx_m * sin_h + dy_m * cos_h
        return float(forward_m), float(lateral_m)

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _reference_lateral_offset_too_large(
            *,
            reference_samples: Sequence[Mapping[str, Any]],
            ego_state: Sequence[float],
            max_lateral_offset_m: float,
        ) -> bool:
        if not reference_samples or len(ego_state) < 4:
            return False
        sample = dict(list(reference_samples)[0])
        dx_m = float(sample.get("x_ref_m", sample.get("x", 0.0))) - float(ego_state[0])
        dy_m = float(sample.get("y_ref_m", sample.get("y", 0.0))) - float(ego_state[1])
        heading_rad = float(ego_state[3])
        lateral_m = -math.sin(heading_rad) * dx_m + math.cos(heading_rad) * dy_m
        return abs(float(lateral_m)) > max(0.0, float(max_lateral_offset_m))

    def _ego_anchored_turn_reference_samples(
            self,
            *,
            ego_location: _PlannerLocation,
            ego_heading_rad: float,
            current_lane_id: int,
            horizon_steps: int,
            step_distance_m: float,
            turn_direction: str,
            route_points: Sequence[Sequence[float]] | None,
        ) -> list[dict[str, float]]:
        start_waypoint = self._map_waypoint_from_location(ego_location)
        if start_waypoint is None:
            return []

        from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint, world_heading_rad

        step_m = max(0.5, float(step_distance_m))
        raw_points: list[tuple[float, float, int, float]] = []
        anchor_forward_m = max(
            0.6,
            float(self.config.get("turn_reference_anchor_forward_m", 1.0)),
        )
        raw_points.append((
            float(ego_location.x) + float(anchor_forward_m) * math.cos(float(ego_heading_rad)),
            float(ego_location.y) + float(anchor_forward_m) * math.sin(float(ego_heading_rad)),
            int(current_lane_id),
            3.5,
        ))

        current = start_waypoint
        previous_heading = float(world_heading_rad(current) or ego_heading_rad)
        first_step_m = max(
            step_m,
            float(self.config.get("turn_reference_first_waypoint_step_m", 1.5)),
        )
        for index in range(max(2, int(horizon_steps) + 3)):
            candidates = list(current.next(first_step_m if index == 0 else step_m) or [])
            if not candidates:
                break
            selected = self._select_turn_next_waypoint(
                current_waypoint=current,
                candidates=candidates,
                previous_heading_rad=float(previous_heading),
                turn_direction=str(turn_direction),
                route_points=route_points,
            )
            if selected is None:
                break
            current = selected
            xy_heading = self._waypoint_xy_heading(current)
            if xy_heading is None:
                break
            lane_width_m = self._waypoint_lane_width(current)
            raw_points.append((
                float(xy_heading[0]),
                float(xy_heading[1]),
                int(canonical_lane_id_for_waypoint(current) or current_lane_id),
                float(lane_width_m),
            ))
            previous_heading = float(world_heading_rad(current) or xy_heading[2])

        if len(raw_points) < 2:
            return []
        raw_samples = []
        for x_m, y_m, lane_id, lane_width_m in raw_points:
            raw_samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(ego_heading_rad),
                "lane_id": int(lane_id),
                "lane_width_m": float(lane_width_m),
            })
        return self._smooth_reference_polyline_samples(
            raw_samples=raw_samples,
            horizon_steps=int(horizon_steps),
            step_distance_m=float(step_m),
            fallback_heading_rad=float(ego_heading_rad),
        )

    def _route_aligned_reference_samples(
            self,
            *,
            ego_location: _PlannerLocation,
            ego_heading_rad: float,
            current_lane_id: int,
            horizon_steps: int,
            step_distance_m: float,
            route_points: Sequence[Sequence[float]] | None,
        ) -> list[dict[str, float]]:
        route_xy = [
            (float(point[0]), float(point[1]))
            for point in list(route_points or [])
            if len(point) >= 2
        ]
        if len(route_xy) < 2:
            return []
        ego_xy = (float(ego_location.x), float(ego_location.y))
        route_progress = [0.0]
        for first, second in zip(route_xy[:-1], route_xy[1:]):
            route_progress.append(
                float(route_progress[-1])
                + math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))
            )
        base_s = self._project_point_to_polyline_s(
            route_xy=route_xy,
            route_progress=route_progress,
            point_xy=ego_xy,
        )
        preview_m = max(
            0.5,
            float(self.config.get("route_aligned_reference_first_forward_m", 1.0)),
        )
        base_s = float(base_s) + float(preview_m)
        samples: list[dict[str, float]] = []
        lane_width_m = self._waypoint_lane_width(self._map_waypoint_from_location(ego_location))
        step_m = max(0.5, float(step_distance_m))
        oversample_step_m = max(
            0.25,
            min(float(step_m), float(self.config.get("route_aligned_reference_oversample_step_m", 0.5))),
        )
        raw_samples: list[dict[str, float]] = []
        raw_count = max(4, (max(1, int(horizon_steps)) + 1) * 2)
        for step_index in range(raw_count):
            target_s = float(base_s) + float(step_index) * float(oversample_step_m)
            x_m, y_m, heading_rad = self._sample_polyline_at_s(
                route_xy=route_xy,
                route_progress=route_progress,
                target_s=target_s,
                fallback_heading_rad=float(ego_heading_rad),
            )
            forward_m, _ = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_heading_rad),
                target_x_m=float(x_m),
                target_y_m=float(y_m),
            )
            if float(forward_m) < float(self.config.get("route_aligned_reference_min_forward_m", 0.25)):
                continue
            raw_samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(heading_rad),
                "lane_id": int(current_lane_id),
                "lane_width_m": float(lane_width_m),
                "road_center_offset_m": 0.0,
                "road_left_width_m": 0.5 * float(lane_width_m),
                "road_right_width_m": 0.5 * float(lane_width_m),
            })
        if not raw_samples:
            return []
        samples = self._smooth_reference_polyline_samples(
            raw_samples=raw_samples,
            horizon_steps=int(horizon_steps),
            step_distance_m=float(step_m),
            fallback_heading_rad=float(ego_heading_rad),
        )
        return samples

    def _smooth_reference_polyline_samples(
            self,
            *,
            raw_samples: Sequence[Mapping[str, object]],
            horizon_steps: int,
            step_distance_m: float,
            fallback_heading_rad: float,
        ) -> list[dict[str, float]]:
        points = [
            (
                float(sample.get("x_ref_m", sample.get("x", 0.0))),
                float(sample.get("y_ref_m", sample.get("y", 0.0))),
                int(sample.get("lane_id", 0) or 0),
                float(sample.get("lane_width_m", 3.5) or 3.5),
            )
            for sample in list(raw_samples or [])
        ]
        if len(points) < 2:
            return [dict(sample) for sample in list(raw_samples or [])]
        progress = [0.0]
        for first, second in zip(points[:-1], points[1:]):
            progress.append(
                progress[-1] + math.hypot(second[0] - first[0], second[1] - first[1])
            )
        samples: list[dict[str, float]] = []
        previous_heading = None
        for index in range(max(1, int(horizon_steps)) + 1):
            target_s = min(float(progress[-1]), float(index) * max(0.5, float(step_distance_m)))
            x_m, y_m, heading_rad = self._sample_polyline_at_s(
                route_xy=[(p[0], p[1]) for p in points],
                route_progress=progress,
                target_s=float(target_s),
                fallback_heading_rad=float(fallback_heading_rad),
            )
            if previous_heading is not None:
                while heading_rad - previous_heading > math.pi:
                    heading_rad -= 2.0 * math.pi
                while heading_rad - previous_heading < -math.pi:
                    heading_rad += 2.0 * math.pi
            previous_heading = float(heading_rad)
            nearest = min(
                points,
                key=lambda point: math.hypot(float(point[0]) - float(x_m), float(point[1]) - float(y_m)),
            )
            lane_id = int(nearest[2])
            lane_width_m = max(0.1, float(nearest[3]))
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(heading_rad),
                "lane_id": int(lane_id),
                "lane_width_m": float(lane_width_m),
                "road_center_offset_m": 0.0,
                "road_left_width_m": 0.5 * float(lane_width_m),
                "road_right_width_m": 0.5 * float(lane_width_m),
            })
        return samples

    @staticmethod
    def _sample_polyline_at_s(
            *,
            route_xy: Sequence[tuple[float, float]],
            route_progress: Sequence[float],
            target_s: float,
            fallback_heading_rad: float,
        ) -> tuple[float, float, float]:
        if len(route_xy) == 0:
            return 0.0, 0.0, float(fallback_heading_rad)
        if len(route_xy) == 1:
            return float(route_xy[0][0]), float(route_xy[0][1]), float(fallback_heading_rad)
        if float(target_s) <= float(route_progress[0]):
            first, second = route_xy[0], route_xy[1]
            heading = math.atan2(float(second[1]) - float(first[1]), float(second[0]) - float(first[0]))
            return float(first[0]), float(first[1]), float(heading)
        for idx in range(len(route_xy) - 1):
            s0 = float(route_progress[idx])
            s1 = float(route_progress[idx + 1])
            if float(target_s) > s1:
                continue
            first = route_xy[idx]
            second = route_xy[idx + 1]
            denom = max(1.0e-6, float(s1) - float(s0))
            ratio = min(1.0, max(0.0, (float(target_s) - float(s0)) / denom))
            x_m = float(first[0]) + ratio * (float(second[0]) - float(first[0]))
            y_m = float(first[1]) + ratio * (float(second[1]) - float(first[1]))
            heading = math.atan2(float(second[1]) - float(first[1]), float(second[0]) - float(first[0]))
            return float(x_m), float(y_m), float(heading)
        before = route_xy[-2]
        last = route_xy[-1]
        heading = math.atan2(float(last[1]) - float(before[1]), float(last[0]) - float(before[0]))
        return float(last[0]), float(last[1]), float(heading)


    def _fallback_control(
            self,
            ego_transform: _PlannerTransform,
            ego_speed_mps: float,
            destination_state: Sequence[float],
            stop_goal_active: bool,
        ) -> PlannerControl:
        if stop_goal_active:
            self._last_accel_mps2 = float(self.mpc.constraints.min_acceleration_mps2)
            self._last_steer_rad = 0.0
            return PlannerControl(throttle=0.0, brake=0.8, steer=0.0)

        dx = float(destination_state[0]) - float(ego_transform.location.x)
        dy = float(destination_state[1]) - float(ego_transform.location.y)
        target_yaw = math.atan2(dy, dx)
        yaw_error = self._wrap_angle(target_yaw - math.radians(float(ego_transform.rotation.yaw)))
        max_steer = max(1e-6, float(self.mpc.constraints.max_steer_rad))
        steer_rad = min(max_steer, max(-max_steer, 0.7 * yaw_error))
        speed_error = float(self.target_speed_mps) - float(ego_speed_mps)
        accel = min(
            float(self.mpc.constraints.max_acceleration_mps2),
            max(float(self.mpc.constraints.min_acceleration_mps2), 0.6 * speed_error),
        )
        self._last_accel_mps2 = float(accel)
        self._last_steer_rad = float(steer_rad)
        return self._control_from_mpc(accel, steer_rad)

    @staticmethod
    def _wrap_angle_static(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    def _sim_time_s(self) -> float:
        """Return the current ROS planning-frame time through the original helper name."""
        return float(getattr(self, "_current_sim_time_s", 0.0))

    def _active_global_route_points(self):
        """Return the custom planner route through the original bridge helper name."""
        return self.route_manager.route_points(query_key="cpx_mpc_bridge")

    def _map_waypoint_from_location(self, location):
        """Replace the CARLA map lookup with the custom global planner lookup."""
        if location is None:
            return None
        try:
            return self.reference_map.get_waypoint({"x": float(location.x), "y": float(location.y), "z": float(getattr(location, "z", 0.0))})
        except Exception:
            return None

    def _lane_id_at_location(self, location) -> int:
        """Return the canonical behavior lane ID at a primitive position."""
        return int(canonical_lane_id_for_waypoint(self._map_waypoint_from_location(location)) or 0)

    @staticmethod
    def _waypoint_xy_heading(waypoint):
        """Read position and world heading from one custom waypoint."""
        if waypoint is None:
            return None
        position = getattr(waypoint, "position", None)
        if not isinstance(position, Mapping):
            return None
        heading_rad = world_heading_rad(waypoint)
        return float(position.get("x", 0.0)), float(position.get("y", 0.0)), float(heading_rad or 0.0)

    @staticmethod
    def _waypoint_lane_width(waypoint) -> float:
        """Read lane width from one custom waypoint."""
        if waypoint is None:
            return 3.5
        return max(0.1, float(getattr(waypoint, "lane_width_m", getattr(waypoint, "lane_width", 3.5)) or 3.5))

    @staticmethod
    def _select_smooth_next_waypoint(*, current_waypoint, candidates, previous_heading_rad, route_points=None):
        """Use OpenCDA's branch-selection order with custom waypoint fields."""
        if not candidates:
            return None
        route_candidate = CPXMPCPlannerBridge._select_route_aligned_candidate(candidates=candidates, route_points=route_points)
        if route_candidate is not None:
            return route_candidate
        current_road_id = int(getattr(current_waypoint, "road_id", 0) or 0)
        current_lane_id = int(getattr(current_waypoint, "lane_id", 0) or 0)

        def heading_cost(waypoint):
            heading = float(world_heading_rad(waypoint) or previous_heading_rad)
            return abs(CPXMPCPlannerBridge._wrap_angle_static(heading - float(previous_heading_rad)))

        same_lane = [candidate for candidate in candidates if int(getattr(candidate, "road_id", 0) or 0) == current_road_id and int(getattr(candidate, "lane_id", 0) or 0) == current_lane_id]
        if same_lane:
            return min(same_lane, key=heading_cost)
        same_raw_lane = [candidate for candidate in candidates if int(getattr(candidate, "lane_id", 0) or 0) == current_lane_id]
        if same_raw_lane:
            return min(same_raw_lane, key=heading_cost)
        return min(candidates, key=heading_cost)

    @staticmethod
    def _select_turn_next_waypoint(*, current_waypoint, candidates, previous_heading_rad, turn_direction, route_points=None):
        """Use OpenCDA's turn-branch score with custom waypoint fields."""
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        direction = str(turn_direction or "").strip().lower()
        route_xy = [(float(point[0]), float(point[1])) for point in list(route_points or []) if len(point) >= 2]

        def route_distance(waypoint):
            position = getattr(waypoint, "position", None)
            if not isinstance(position, Mapping) or not route_xy:
                return 0.0
            return min(math.hypot(float(position.get("x", 0.0)) - x_m, float(position.get("y", 0.0)) - y_m) for x_m, y_m in route_xy)

        def score(waypoint):
            heading = float(world_heading_rad(waypoint) or previous_heading_rad)
            delta = CPXMPCPlannerBridge._wrap_angle_static(heading - float(previous_heading_rad))
            direction_bonus = -max(0.0, delta) if direction == "left" else -max(0.0, -delta) if direction == "right" else 0.0
            return float(route_distance(waypoint)) + 0.35 * abs(delta) + direction_bonus

        return min(candidates, key=score)

    @staticmethod
    def _select_route_aligned_candidate(*, candidates, route_points=None):
        """Choose the custom waypoint closest to the active route, as OpenCDA does for CARLA waypoints."""
        route_xy = [(float(point[0]), float(point[1])) for point in list(route_points or []) if len(point) >= 2]
        if not candidates or len(route_xy) < 2:
            return None
        scored = []
        for candidate in candidates:
            position = getattr(candidate, "position", None)
            if not isinstance(position, Mapping):
                continue
            distance_m = min(math.hypot(float(position.get("x", 0.0)) - route_x, float(position.get("y", 0.0)) - route_y) for route_x, route_y in route_xy)
            scored.append((distance_m, candidate))
        if not scored:
            return None
        best_distance_m, best_candidate = min(scored, key=lambda item: item[0])
        return best_candidate if float(best_distance_m) <= 5.0 else None

    def _carla_waypoint_turn_reference(self, *, ego_location, ego_yaw_rad, current_state, current_lane_id, target_lane_id, target_speed_mps, destination_state):
        """Keep OpenCDA's method name while replacing its CARLA route query with the custom route manager."""
        turn_speed_mps = min(max(0.4, float(target_speed_mps)), float(self.config.get("carla_waypoint_turn_speed_cap_mps", 2.2)))
        step_distance_m = max(float(self.config.get("carla_waypoint_turn_min_step_m", 0.35)), float(self.mpc.dt_s) * max(0.8, turn_speed_mps))
        reference, reason = self.route_manager.carla_waypoint_reference(ego_x_m=float(ego_location.x), ego_y_m=float(ego_location.y), ego_heading_rad=float(ego_yaw_rad), horizon_steps=int(self.mpc.horizon_steps), step_distance_m=step_distance_m, target_speed_mps=turn_speed_mps, fallback_lane_id=int(target_lane_id or current_lane_id))
        if not reference:
            return [], list(destination_state or []), str(reason)
        seed_destination = list(destination_state or [])
        if len(seed_destination) < 5:
            seed_destination = [float(current_state[0]), float(current_state[1]), turn_speed_mps, float(current_state[3]), int(target_lane_id or current_lane_id)]
        seed_destination[2] = turn_speed_mps
        destination = lane_center_destination_from_reference(destination_state=seed_destination, lane_center_reference=reference, ego_state=current_state, target_forward_m=float(self.config.get("carla_waypoint_turn_destination_forward_m", 6.0))) or seed_destination
        return [dict(sample) for sample in reference], list(destination), str(reason)



def _mpc_cost_profile_for_behavior(*, behavior: str, planner_lc_state: str, planner_mode: str, next_macro_maneuver: str) -> str:
    raw_behavior = str(behavior or "").strip().lower()
    normalized_behavior = str(normalize_behavior_decision(behavior))
    normalized_lc_state = str(planner_lc_state or "").strip().upper()
    normalized_mode = str(planner_mode or "").strip().upper()
    normalized_maneuver = str(next_macro_maneuver or "straight").strip().lower()

    if bool(is_fixed_stop_decision(normalized_behavior)):
        return "stop"
    if bool(is_emergency_brake_decision(normalized_behavior)):
        return "recovery"
    if normalized_lc_state.startswith("PREPARE_LANE_CHANGE"):
        return "prepare_lane_change"
    if raw_behavior in {"intersection_turn_left", "intersection_turn_right"}:
        return "intersection_turn"
    if normalized_lc_state.startswith("EXECUTE_LANE_CHANGE") or normalized_behavior in {"lane_change_left", "lane_change_right"}:
        return "execute_lane_change"
    if normalized_mode == "INTERSECTION" and normalized_maneuver in {"left", "right"}:
        return "intersection_turn"
    return "lane_follow"


def _select_mpc_cost_profile_with_hysteresis(*, requested_profile: str, active_profile: str, sim_time_s: float, active_since_s: float, min_hold_s: float) -> tuple[str, float, str]:
    requested = str(requested_profile or "lane_follow").strip() or "lane_follow"
    active = str(active_profile or "lane_follow").strip() or "lane_follow"
    elapsed_s = max(0.0, float(sim_time_s) - float(active_since_s))
    min_hold_s = max(0.0, float(min_hold_s))

    if requested == active:
        return active, float(active_since_s), "same_profile"
    if requested in {"stop", "recovery"}:
        return requested, float(sim_time_s), "safety_preempt"
    if active in {"stop", "recovery"} and elapsed_s < min_hold_s:
        return active, float(active_since_s), "hold_safety_profile"
    if elapsed_s < min_hold_s:
        return active, float(active_since_s), "min_hold"
    return requested, float(sim_time_s), "switch"
