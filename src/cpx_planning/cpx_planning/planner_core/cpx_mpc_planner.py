"""CP-X MPC planner copied from the active OpenCDA bridge.

ROS owns only the input/output boundary. The planning order and calculations
remain the same, while map queries use the custom AD-map planner and the
returned control is a simulator-independent numeric object.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from cpx_planning.component_interfaces import PlannerControl, PlannerLocation, PlannerRuntime, PlannerTransform
import yaml

from cpx_planning.pipeline.traffic_light_memory import (
    TrafficLightMemory,
)
from cpx_planning.pipeline.map_matching import (
    DiagnosticHDMapMatcher,
    LaneProjectionCandidate,
    local_lane_frame_invariants,
    topology_relation,
)
from cpx_planning.utility.speed_profile import (
    curvature_speed_cap_mps,
)
from cpx_planning.utility.global_planner import world_heading_rad


class _CarlaMapPlannerAdapter:
    """Small adapter so planning-module helpers can query a CARLA map with dict poses."""

    def __init__(self, map_planner: Any):
        self._map_planner = map_planner

    def get_waypoint(self, point: Any):
        get_waypoint = getattr(self._map_planner, "get_waypoint", None)
        if not callable(get_waypoint):
            return None
        if isinstance(point, Mapping):
            location = PlannerLocation(
                x=float(point.get("x", 0.0)),
                y=float(point.get("y", 0.0)),
                z=float(point.get("z", 0.0)),
            )
            try:
                return get_waypoint(location)
            except Exception:
                return None
        try:
            return get_waypoint(point)
        except Exception:
            return None


def _static_obstacle_cooldown_policy(
    *,
    failed_latched: bool,
    route_transition_pending: bool,
) -> tuple[str, bool]:
    """Return debug status and stop ownership during a replan cooldown."""

    if bool(failed_latched):
        return "cooldown_stop", True
    if bool(route_transition_pending):
        return "cooldown_route_transition", False
    return "cooldown_stop", True


def _lane_change_execution_active(
        *, reference_locked: bool, phase: object) -> bool:
    """Return whether a committed lane change still owns route execution.

    The route progress tracker is allowed to observe a temporarily lapsed lane
    change requirement while ego follows the locked lateral trajectory.  That
    lapse must not be interpreted as a missed maneuver until the trajectory is
    released.  The phase check also protects the stabilization hand-off, where
    the semantic route instruction may already have advanced.
    """
    normalized_phase = str(phase or "").strip().lower()
    return bool(reference_locked) or normalized_phase in {
        "executing",
        "target_lane_stabilization",
    }


def _select_static_obstacle_local_avoidance_lane(
    *,
    current_lane_id: int,
    available_lane_ids: Sequence[int],
    lane_safety_scores: Mapping[int, float],
    lane_prediction_risks: Mapping[int, Mapping[str, object]],
    minimum_safety_score: float,
) -> int | None:
    """Select one adjacent, prediction-safe lane for local obstacle bypass.

    This helper deliberately does not alter the global route or the map.  It
    only authorizes the existing behavior/reference candidate pipeline to
    evaluate a local lane-borrow trajectory.  The downstream FSM, reference
    contract, MPC probe and safety supervisor retain veto authority.
    """

    current = int(current_lane_id)
    alternatives = sorted(
        {
            int(lane_id)
            for lane_id in list(available_lane_ids or [])
            if int(lane_id) != 0 and int(lane_id) != current
        },
        key=lambda lane_id: abs(int(lane_id) - current),
    )
    if not alternatives:
        return None

    nearest_delta = abs(int(alternatives[0]) - current)
    adjacent = [
        int(lane_id)
        for lane_id in alternatives
        if abs(int(lane_id) - current) == int(nearest_delta)
    ]
    safe = []
    for lane_id in adjacent:
        score = float(lane_safety_scores.get(int(lane_id), 0.0))
        risk = dict(lane_prediction_risks.get(int(lane_id), {}) or {})
        if score <= float(minimum_safety_score) or bool(risk.get("risk", False)):
            continue
        safe.append((float(score), int(lane_id)))
    if not safe:
        return None
    safe.sort(key=lambda row: (-float(row[0]), abs(int(row[1]) - current), -int(row[1])))
    return int(safe[0][1])


class CPXMPCPlannerBridge:
    """Direct-control planner used inside ``VehicleManager.run_step``."""

    def __init__(
        self,
        vehicle_manager: Any = None,
        config: Optional[Mapping[str, Any]] = None,
        *,
        map_planner: Any = None,
        mpc_instance: Any = None,
        route_manager_instance: Any = None,
        behavior_components_enabled: bool = True,
    ):
        self._ensure_planning_module_import_path()
        del vehicle_manager
        self.vehicle_manager = None
        from cpx_planning.pipeline.architecture_profile import (
            normalize_architecture_config,
        )

        self.config, self.architecture_profile = normalize_architecture_config(config)
        self.carla = PlannerRuntime
        self.map_planner = map_planner
        if self.map_planner is None:
            raise TypeError("ROS CP-X requires CustomGlobalPlannerAdapter as map_planner.")
        self.enabled = bool(self.config.get("enabled", True))
        self.mode = str(self.config.get("mode", "full_cpx_mpc")).strip().lower()
        self.fallback_policy = str(
            self.config.get("fallback_policy", "emergency_stop")
        ).strip().lower()
        self.fallback_policy_warning = ""
        if self.fallback_policy == "opencda":
            self.fallback_policy = "emergency_stop"
            self.fallback_policy_warning = "opencda_fallback_disabled_in_full_cpx_mpc"
        self.use_opencda_global_route = bool(
            self.config.get("use_opencda_global_route", True)
        )
        self.opencda_global_route_reference_allowed = bool(
            self.config.get("opencda_global_route_reference_allowed", True)
        )
        self.target_speed_mps = float(self.config.get("target_speed_mps", 8.0))
        self.lookahead_m = float(self.config.get("lookahead_m", 18.0))
        self.min_front_gap_m = float(self.config.get("min_front_gap_m", 8.0))
        # min_front_gap_m alone is a flat distance that doesn't scale with
        # cruise speed: at 11.18 m/s the default 8.0m gave several
        # seconds of reaction margin before target_lane_prediction_risk
        # would trip, but at 20 m/s the same 8.0m is covered in half the
        # time -- confirmed via telemetry (Interactive_Lane_Change's
        # queued lane change missed again at 20 m/s cruise, tripping this
        # exact check, after being fixed at 11.18 m/s). min_front_gap_time_s
        # defaults to 8.0/11.18 so today's calibrated distance is
        # reproduced exactly at 11.18 m/s, while the effective floor grows
        # proportionally with whatever cruise speed is configured.
        self.min_front_gap_time_s = float(
            self.config.get("min_front_gap_time_s", 8.0 / 11.18)
        )
        self.max_mpc_obstacles = max(0, int(self.config.get("max_mpc_obstacles", 4)))
        self.debug = bool(self.config.get("debug", True))
        self.last_debug: dict[str, Any] = {}
        self._last_accel_mps2 = 0.0
        self._last_steer_rad = 0.0
        self._actuator_ego_speed_mps = 0.0
        self._actuator_target_speed_mps = 0.0
        self._actuator_stop_goal_active = False
        self._warned = False
        self._previous_lane_center_reference: list[dict[str, object]] = []
        from cpx_planning.utility.lane_graph import StableLaneIdTracker

        self._lane_id_tracker = StableLaneIdTracker()
        self._diagnostic_hd_map_matcher = DiagnosticHDMapMatcher()
        self._diagnostic_local_lane_frame: dict[str, object] = {}
        self._diagnostic_map_matching: dict[str, object] = {}
        self._lane_id_discontinuity_log_file = None
        self._temporary_destination_state: list[float] | None = None
        self._lane_reference_freeze_count = 0
        self._stop_release_temp_smooth_until_sim_time_s = 0.0
        self._full_latched_stop_target: dict[str, object] | None = None
        self._full_latched_stop_state = "unknown"
        self._full_signal_actor_id = ""
        self._full_traffic_memory = TrafficLightMemory(
            hold_unknown_s=float(self.config.get("full_traffic_unknown_hold_s", 1.5)),
            hold_green_unknown_s=float(
                self.config.get("full_traffic_green_unknown_hold_s", 0.25)
            ),
            green_confirm_s=float(self.config.get("full_traffic_green_confirm_s", 0.15)),
            hold_stop_unknown_until_green=bool(
                self.config.get(
                    "full_traffic_hold_stop_unknown_until_green",
                    False,
                )
            ),
        ) if bool(behavior_components_enabled) else None
        from cpx_planning.pipeline.scenario_manager import (
            BoundaryRecoveryRequest,
            CPXScenarioManager,
        )

        self._scenario_manager = CPXScenarioManager(self.config) if bool(behavior_components_enabled) else None
        self._boundary_recovery_request = BoundaryRecoveryRequest()
        self._boundary_recovery_trigger_frames = 0
        self._boundary_recovery_infeasible_frames = 0
        self._boundary_recovery_cooldown_until_s = -float("inf")
        self._full_last_behavior_mode_key = ""
        self._turn_latch_decision = ""
        self._turn_latch_until_sim_time_s = -float("inf")
        self._route_replan_last_attempt_s = -float("inf")
        self._route_replan_attempt_count = 0
        self._route_replan_last_reason = "route_replan_not_requested"
        self._static_obstacle_replan_last_attempt_s = -float("inf")
        self._static_obstacle_replan_failed_latched = False
        self._static_obstacle_replan_status = "idle"
        self._static_obstacle_candidate_id = ""
        self._static_obstacle_candidate_since_s = -float("inf")
        self._static_obstacle_route_transition_pending = False
        self._static_obstacle_blocked_lane_id: object = ""
        self._static_obstacle_replan_reason = "not_requested"
        self._static_obstacle_local_target_lane_id: Optional[int] = None
        self._last_required_lane_change_target_lane_id: Optional[int] = None
        self._last_required_lane_change_target_ad_lane_id: Optional[int] = None
        self._route_tracking_lane_change_option = ""
        self._route_tracking_lane_change_progress = 0.0
        self._route_tracking_lane_change_reference: list[dict[str, object]] = []
        self._route_tracking_lane_change_progress_pairs: list[
            tuple[dict[str, object], dict[str, object]]
        ] = []
        self._route_tracking_lane_change_envelope_blocks = None
        self._route_tracking_lane_change_envelope_epsilon0 = 0.0
        self._route_tracking_lane_change_commitment_invalid_frames = 0
        self._route_tracking_lane_change_duration_comfort_reason = ""
        self._route_tracking_lane_change_resolved_duration_s = 0.0
        self._route_tracking_lane_change_progress_index = 0
        self._route_tracking_lane_change_source_lane_id = 0
        self._route_tracking_lane_change_target_lane_id = 0
        self._route_tracking_lane_change_target_speed_mps = 0.0
        self._route_tracking_lane_change_phase = "idle"
        self._route_tracking_lane_change_stabilization_frames = 0
        self._route_tracking_lane_change_completion_stable_frames = 0
        self._route_tracking_lane_change_completion_debug: dict[str, object] = {}
        self._route_tracking_lane_change_committed_at_s = -float("inf")
        self._lane_id_discontinuity_since_lock = False
        self._route_tracking_lane_change_completed_option = ""
        self.strict_lane_follow_reference = bool(
            self.config.get("strict_lane_follow_reference", False)
        )
        self.draw_world_debug = bool(self.config.get("draw_world_debug", False))
        self.draw_world_debug_destination = bool(
            self.config.get("draw_world_debug_destination", False)
        )
        self.world_debug_life_time_s = float(self.config.get("world_debug_life_time_s", 0.15))
        self.full_control_buffer_min_speed_mps = max(
            0.0,
            float(self.config.get("full_control_buffer_min_speed_mps", 1.5)),
        )
        self.full_lane_change_start_lock_s = max(
            0.0,
            float(self.config.get("full_lane_change_start_lock_s", 8.0)),
        )
        self.full_dense_traffic_lane_change_lock_enabled = bool(
            self.config.get("full_dense_traffic_lane_change_lock_enabled", True)
        )
        self.full_dense_traffic_object_count = max(
            0,
            int(self.config.get("full_dense_traffic_object_count", 8)),
        )
        self.full_dense_traffic_risky_lane_count = max(
            0,
            int(self.config.get("full_dense_traffic_risky_lane_count", 2)),
        )
        self.full_prepare_lane_change_reference_lock = bool(
            self.config.get("full_prepare_lane_change_reference_lock", True)
        )
        self.full_allow_opportunistic_lane_change = bool(
            self.config.get("full_allow_opportunistic_lane_change", False)
        )
        self.full_lane_follow_max_destination_lateral_m = max(
            0.0,
            float(self.config.get("full_lane_follow_max_destination_lateral_m", 1.2)),
        )
        self.full_lane_follow_max_reference_first_lateral_m = max(
            0.0,
            float(self.config.get("full_lane_follow_max_reference_first_lateral_m", 0.65)),
        )
        self.full_stop_max_destination_lateral_m = max(
            0.0,
            float(self.config.get("full_stop_max_destination_lateral_m", 1.0)),
        )
        self.full_stop_max_reference_first_lateral_m = max(
            0.0,
            float(self.config.get("full_stop_max_reference_first_lateral_m", 0.55)),
        )
        # The lateral-only guard below lets a persistent physical heading
        # bias go uncorrected for many ticks: each tick's lateral offset is
        # individually small enough to stay under the lateral thresholds
        # above, but a several-degree heading error against the true lane
        # tangent (from compute_ego_lane_offset) integrates into lateral
        # drift at low speed (v*sin(heading_error)) over a few seconds,
        # eventually crossing the lateral threshold anyway -- just late,
        # after the vehicle has drifted toward an adjacent lane and often
        # after the maneuver window (e.g. an approaching intersection stop)
        # has already closed. Checking heading directly forces the same
        # already-working true-waypoint rebuild before that drift compounds.
        self.full_lane_follow_max_heading_error_deg = max(
            0.0,
            float(self.config.get("full_lane_follow_max_heading_error_deg", 4.0)),
        )
        self.full_stop_max_heading_error_deg = max(
            0.0,
            float(self.config.get("full_stop_max_heading_error_deg", 4.0)),
        )
        self.full_mpc_reference_stabilizer_enabled = bool(
            self.config.get("full_mpc_reference_stabilizer_enabled", True)
        )
        self.full_candidate_pipeline_enabled = bool(
            self.config.get("full_candidate_pipeline_enabled", True)
        )
        self.full_candidate_reference_min_object_distance_m = max(
            0.0,
            float(self.config.get("full_candidate_reference_min_object_distance_m", 2.0)),
        )
        # The generic clearance above (default 2.0m, configured to 3.5m here)
        # sizes lateral gaps for negotiating with *moving* traffic. Applied
        # unmodified to a static-obstacle local-avoidance candidate it is
        # self-defeating: the whole point of that candidate is to pass close
        # to the very obstacle it is routing around, in a lane only ~3.5m
        # wide, so it always scores infeasible and the vehicle never moves
        # (confirmed via decision_veto_chain: all three lane-change variants
        # rejected on candidate_prediction_collision_risk ~1.1-1.3m, the
        # ego's own predicted clearance from the blocking obstacle, static
        # across assertive/normal/conservative timing since the obstacle
        # isn't moving). Use a tighter, still-conservative clearance just for
        # the candidate whose target lane matches the selected local-
        # avoidance lane; every other candidate keeps the full margin above.
        self.static_obstacle_local_avoidance_min_object_distance_m = max(
            0.0,
            float(
                self.config.get(
                    "static_obstacle_local_avoidance_min_object_distance_m",
                    0.8,
                )
            ),
        )
        # Two same-lane candidates (e.g. full-speed "keep_lane" vs slowed
        # "yield_slow_down") build references at different speeds/extents, so
        # their own predicted-obstacle-distance estimates can differ by more
        # than this margin purely from that shape difference, not real
        # obstacle motion -- flipping which discrete risk bucket (and thus
        # which candidate) wins every other tick and reading to MPC as a
        # discontinuous reference. Keyed by candidate name so each logical
        # candidate slot keeps its own hysteresis state across ticks.
        self.candidate_risk_hysteresis_margin_m = max(
            0.0,
            float(self.config.get("candidate_risk_hysteresis_margin_m", 1.5)),
        )
        self._candidate_risk_bucket_state: dict[str, str] = {}
        self.candidate_mpc_probe_enabled = bool(
            self.config.get("candidate_mpc_probe_enabled", True)
        )
        self.candidate_mpc_probe_top_k = max(
            2,
            int(self.config.get("candidate_mpc_probe_top_k", 2)),
        )
        self.candidate_mpc_probe_interval_s = max(
            0.05,
            float(self.config.get("candidate_mpc_probe_interval_s", 0.2)),
        )
        self._candidate_mpc_probe_last_time_s = -float("inf")
        self._candidate_mpc_probe_cache: dict[tuple[object, ...], dict[str, object]] = {}
        self.candidate_lane_change_assertive_duration_s = max(
            0.1,
            float(self.config.get("candidate_lane_change_assertive_duration_s", 3.2)),
        )
        self.candidate_lane_change_normal_duration_s = max(
            0.1,
            float(self.config.get("candidate_lane_change_normal_duration_s", 4.0)),
        )
        self.candidate_lane_change_conservative_duration_s = max(
            0.1,
            float(self.config.get("candidate_lane_change_conservative_duration_s", 5.5)),
        )
        self.candidate_lane_change_assertive_speed_scale = max(
            0.1,
            float(self.config.get("candidate_lane_change_assertive_speed_scale", 1.0)),
        )
        self.candidate_lane_change_normal_speed_scale = max(
            0.1,
            float(self.config.get("candidate_lane_change_normal_speed_scale", 0.9)),
        )
        self.candidate_lane_change_conservative_speed_scale = max(
            0.1,
            float(self.config.get("candidate_lane_change_conservative_speed_scale", 0.7)),
        )
        self.strict_decision_ownership_enabled = bool(
            self.config.get("strict_decision_ownership_enabled", True)
        )
        self.strict_reference_validator_veto_enabled = bool(
            self.config.get("strict_reference_validator_veto_enabled", True)
        )
        self.strict_explicit_fallback_speed_mps = max(
            0.0,
            float(self.config.get("strict_explicit_fallback_speed_mps", 0.8)),
        )
        self.full_reference_stabilizer_min_forward_m = float(
            self.config.get("full_reference_stabilizer_min_forward_m", -0.25)
        )
        self.full_reference_stabilizer_min_spacing_m = max(
            0.0,
            float(self.config.get("full_reference_stabilizer_min_spacing_m", 0.35)),
        )
        self.full_reference_stabilizer_max_heading_step_rad = max(
            0.0,
            float(self.config.get("full_reference_stabilizer_max_heading_step_rad", 0.75)),
        )
        self._debug_writer = None
        self._debug_csv_file = None
        self._debug_jsonl_file = None
        self._debug_fieldnames = [
            "sim_time_s",
            "vehicle_id",
            "x_m",
            "y_m",
            "yaw_deg",
            "speed_mps",
            "measured_accel_mps2",
            "mpc_jerk_seed_accel_mps2",
            "target_speed_mps",
            "speed_plan_target_mps",
            "speed_plan_front_gap_m",
            "speed_plan_desired_follow_gap_m",
            "speed_plan_continuous_following_active",
            "speed_plan_idm_acceleration_mps2",
            "speed_plan_reason",
            "speed_owner_requested_mps",
            "speed_owner_scenario_cap_mps",
            "speed_owner_turn_cap_mps",
            "speed_owner_lane_change_cap_mps",
            "speed_owner_following_cap_mps",
            "speed_owner_turn_approach_cap_mps",
            "speed_owner_upcoming_turn_distance_m",
            "speed_owner_selected_target_mps",
            "speed_owner_limiting_owner",
            "speed_owner_active_constraints",
            "speed_owner_mpc_entry_target_mps",
            "speed_owner_post_plan_delta_mps",
            "speed_owner_target_overridden_after_plan",
            "speed_owner_proposed_post_plan_target_mps",
            "speed_owner_ceiling_applied",
            "speed_owner_ceiling_reduction_mps",
            "behavior_decision",
            "static_obstacle_stop_active_input",
            "static_obstacle_replan_status",
            "static_obstacle_replan_reason",
            "static_obstacle_candidate_id",
            "static_obstacle_blocked_lane_id",
            "static_obstacle_route_transition_pending",
            "behavior_fsm_state",
            "current_lane_id",
            "behavior_target_lane_id",
            "map_match_valid",
            "map_match_ad_lane_id",
            "map_match_road_id",
            "map_match_section_id",
            "map_match_raw_lane_id",
            "map_match_center_x_m",
            "map_match_center_y_m",
            "map_match_lane_width_m",
            "map_match_lateral_offset_m",
            "map_match_heading_error_rad",
            "map_match_score",
            "map_match_confidence",
            "map_match_reason",
            "map_match_candidate_count",
            "local_lane_frame_cache_reused",
            "local_lane_frame_generation_reason",
            "local_lane_frame_ego_ad_lane_id",
            "local_lane_frame_forward_distance_m",
            "local_lane_frame_backward_distance_m",
            "local_lane_frame_corridors",
            "local_lane_frame_lane_to_offset",
            "local_lane_frame_route_target_ad_lane_id",
            "local_lane_frame_target_in_frame",
            "local_lane_frame_target_offset",
            "local_lane_frame_invariant_violations",
            "stop_goal_active",
            "normal_stop_requested",
            "emergency_brake_requested",
            "emergency_brake_control_active",
            "normal_stop_mpc_suspended",
            "normal_stop_mpc_suspend_speed_mps",
            "normal_stop_mpc_suspend_brake",
            "front_gap_m",
            "object_count",
            "mpc_object_count",
            "cp_provider_source",
            "native_opencda_available",
            "cp_obstacle_count",
            "cp_control_count",
            "v2x_nearby_count",
            "cp_observer_cav_count",
            "cp_observer_cav_ids",
            "cp_multi_observer_obstacle_count",
            "cp_blind_spot_shared_count",
            "cp_blind_spot_shared_actor_ids",
            "cp_actor_provenance",
            "cp_pedestrian_count",
            "cp_blind_spot_pedestrian_count",
            "cp_prediction_used_actor_ids",
            "cp_prediction_used_pedestrian_ids",
            "cp_candidate_relevant_actor_ids",
            "cp_candidate_relevant_pedestrian_ids",
            "cp_actor_evidence",
            "cp_visibility_filter_enabled",
            "cp_visibility_backend",
            "reference_source",
            "final_reference_geometry_source",
            "reference_pipeline_stage",
            "reference_pipeline_intent",
            "reference_pipeline_fallback",
            "destination_x",
            "destination_y",
            "destination_forward_m",
            "destination_lateral_m",
            "destination_lane_id",
            "reference_first_forward_m",
            "reference_first_lateral_m",
            "mpc_trajectory_point_count",
            "global_route_point_count",
            "route_reference_allowed",
            "route_reference_gate_reason",
            "route_lane_change_allowed",
            "opportunistic_lane_change_allowed",
            "lane_change_gate_reason",
            "route_lane_change_required",
            "lane_change_authorized",
            "lane_change_authorization_direction",
            "lane_change_authorization_reason",
            "lane_change_required_by_route",
            "lane_change_distance_to_maneuver_m",
            "lane_change_authorized_target_lane_id",
            "route_maneuver_normalized",
            "behavior_override_reason",
            "reference_follow_global_route_lane",
            "route_current_road_option",
            "route_next_macro_maneuver",
            "mpc_status",
            "mpc_feasibility_checked",
            "mpc_feasibility_status",
            "mpc_feasibility_reason",
            "mpc_solve_time_ms",
            "mpc_cost_profile",
            "requested_mpc_cost_profile",
            "mpc_cost_profile_switch_reason",
            "mpc_fallback_reason",
            "control_guard_reason",
            "safety_supervisor_reason",
            "accel_cmd_mps2",
            "steer_cmd_rad",
            "pre_supervisor_accel_cmd_mps2",
            "pre_supervisor_steer_cmd_rad",
            "post_supervisor_accel_cmd_mps2",
            "post_supervisor_steer_cmd_rad",
            "applied_throttle",
            "applied_brake",
            "applied_steer",
            "control_interface",
            "platform_target_speed_mps",
            "platform_target_steer_rad",
            "platform_adapter_steer_rad",
            "platform_adapter_throttle",
            "platform_adapter_brake",
            "platform_adapter_steer",
            "platform_applied_steer_rad",
            "platform_actual_speed_mps",
            "platform_adapter_reason",
            "planner_input_cp_traffic_control_count",
            "planner_input_prediction_risky_lane_count",
            "planner_input_perception_planning_count",
            "perception_mode",
            "perception_ml_active",
            "perception_camera_count",
            "planner_input_cp_obstacle_count",
            "planner_input_frame_timestamp_s",
            "cp_message_timestamp_s",
            "cp_message_age_s",
            "cp_message_valid",
            "planner_requested",
            "planner_executed",
            "fallback_active",
            "local_object_count",
            "traffic_signal_state",
            "traffic_signal_raw_state",
            "traffic_signal_resolved_state",
            "traffic_signal_filtered_state",
            "traffic_signal_behavior_state",
            "traffic_control_from_cp",
            "candidate_evaluation_summary",
            "candidate_selected_decision",
            "candidate_selected_lane_id",
            "candidate_selected_cost",
            "candidate_pipeline_enabled",
            "candidate_pipeline_selected",
            "candidate_pipeline_selected_status",
            "candidate_pipeline_selected_reason",
            "candidate_selected_stop_goal_active",
            "candidate_pipeline_count",
            "candidate_prediction_trajectory_count",
            "candidate_pipeline_summary",
            "candidate_mpc_probe_summary",
            "candidate_selected_trajectory_variant",
            "candidate_selected_lane_change_duration_s",
            "candidate_selected_lane_change_duration_comfort_reason",
            "candidate_selected_lane_change_planning_average_speed_mps",
            "candidate_selected_lane_change_authorization_source",
            "candidate_selected_lane_change_initial_progress",
            "candidate_selected_lane_change_terminal_progress",
            "candidate_selection_status",
            "candidate_selection_reason",
            "maneuver_commitment_state",
            "maneuver_commitment_decision",
            "maneuver_commitment_source_lane_id",
            "maneuver_commitment_target_lane_id",
            "maneuver_commitment_progress",
            "maneuver_commitment_reference_locked",
            "maneuver_commitment_active",
            "maneuver_geometry_active",
            "maneuver_geometry_id",
            "maneuver_geometry_type",
            "maneuver_geometry_direction",
            "maneuver_geometry_phase",
            "maneuver_geometry_revision",
            "maneuver_geometry_source_changed",
            "maneuver_geometry_owner",
            "maneuver_geometry_point_count",
            "maneuver_first_point_jump_m",
            "maneuver_first_heading_jump_deg",
            "maneuver_geometry_release_reason",
            "lane_change_commitment_release_reason",
            "lane_change_phase",
            "lane_change_stabilization_frames",
            "lane_change_completion_reason",
            "lane_change_completion_stable_frames",
            "lane_change_completion_lateral_error_m",
            "lane_change_completion_heading_error_deg",
            "lane_change_stabilization_entry_lateral_error_m",
            "lane_change_stabilization_geometry_ready",
            "behavior_lane_lateral_error_m",
            "behavior_lane_heading_error_deg",
            "behavior_lane_alignment_valid",
            "behavior_lane_change_completion_allowed",
            "lane_change_completion_target_lane_matches",
            "lane_change_completion_footprint_clearance_m",
            "route_tracking_lane_change_locked",
            "route_tracking_lane_change_progress_index",
            "route_tracking_lane_change_source_lane_id",
            "route_tracking_lane_change_target_lane_id",
            "route_tracking_recovery_active",
            "route_tracking_recovery_reason",
            "mpc_feedback_summary",
            "mpc_feedback_record_reason",
            "mpc_feedback_blocked_lane_ids",
            "mode_transition_guard_reason",
            "control_buffer_reason",
            "control_buffered_step_count",
            "mpc_replan_executed",
            "route_manager_status",
            "route_replan_attempted",
            "route_replan_succeeded",
            "route_replan_attempt_count",
            "route_replan_reason",
            "route_remaining_distance_m",
            "route_reached_destination",
            "global_planner_backend",
            "global_planner_backend_warning",
            "tracker_active_count",
            "tracker_stale_count",
            "prediction_validity_reason",
            "object_memory_reason",
            "traffic_memory_reason",
            "decision_scenario_state",
            "decision_behavior",
            "decision_behavior_fsm",
            "decision_candidate",
            "decision_reference_source",
            "decision_reference_stage",
            "decision_mpc_status",
            "decision_final_action",
            "decision_control_source",
            "decision_veto_count",
            "decision_veto_chain",
            "decision_veto_chain_text",
            "decision_owner_summary",
            "architecture_profile",
            "architecture_behavior_owner",
            "architecture_speed_owner",
            "architecture_reference_owner",
            "architecture_control_memory_owner",
            "architecture_safety_owner",
            "architecture_normalized_overrides",
            "scenario_fsm_state",
            "scenario_fsm_reason",
            "scenario_behavior_signal_state",
            "scenario_behavior_override_decision",
            "scenario_speed_cap_mps",
            "scenario_stop_goal_active",
            "scenario_turn_direction",
            "scenario_turn_latched",
            "scenario_boundary_recovery_active",
            "scenario_boundary_clearance_m",
            "scenario_boundary_lateral_offset_m",
            "scenario_boundary_heading_error_rad",
            "boundary_recovery_generation_reason",
            "boundary_recovery_conditioning_reason",
            "traffic_stop_forward_m",
            "traffic_stop_commit_distance_m",
            "traffic_stop_approach_reason",
            "speed_plan_target_mps",
            "speed_plan_cap_mps",
            "speed_plan_stop_goal_active",
            "speed_plan_reason",
            "carla_turn_reference_reason",
            "carla_route_debug_reason",
            "carla_route_sync_reason",
            "carla_route_progress_index",
            "carla_upcoming_turn_direction",
            "carla_upcoming_turn_distance_m",
            "carla_upcoming_turn_reason",
            "turn_latch_reason",
            "opencda_style_reference_conditioning_reason",
            "reference_lateral_guard_reason",
            "mpc_reference_stabilizer_reason",
            "final_reference_gate_valid",
            "final_reference_gate_mode",
            "final_reference_gate_reason",
            "reference_max_curvature_1pm",
            "reference_contract_max_curvature_1pm",
            "reference_curvature_margin_1pm",
            "reference_pipeline_conditioning_reason",
            "reference_pipeline_mode",
            "mpc_entry_allowed",
            "mpc_entry_status",
            "mpc_entry_reason",
            "pipeline_error",
            "stop_target_forward_m",
            "stop_approach_speed_mps",
            "green_release_reference_active",
            "lane_safety_scores",
            "evaluation_metrics_available",
            "collision_sensor_available",
            "collision_sensor_error",
            "collision_event_this_frame",
            "collision_count",
            "collision_rate_per_km",
            "last_collision_actor_type",
            "last_collision_impulse",
            "nearest_ttc_s",
            "min_ttc_s",
            "nearest_ttc_obstacle_id",
            "nearest_ttc_reason",
            "nearest_ttc_longitudinal_gap_m",
            "nearest_ttc_lateral_gap_m",
            "nearest_ttc_bumper_gap_m",
            "nearest_ttc_closing_speed_mps",
            "tick_max_drac_mps2",
            "max_drac_mps2",
            "min_pet_s",
            "distance_traveled_m",
            "road_boundary_sample_valid",
            "road_boundary_lateral_offset_m",
            "road_boundary_lane_width_m",
            "road_boundary_ego_half_width_m",
            "road_boundary_heading_error_rad",
            "road_boundary_clearance_m",
            "road_boundary_breach",
            "road_boundary_projection_segment_index",
            "road_boundary_projection_segment_ratio",
            "road_boundary_projection_raw_heading_rad",
            "road_boundary_projection_conditioned_heading_rad",
            "road_boundary_projection_continuity_limited",
            "road_boundary_projection_reason",
            "road_boundary_geometry_source",
            "road_boundary_drivable_inside",
            "road_boundary_breach_count",
            "road_boundary_sample_count",
            "road_boundary_breach_rate",
            "Cost_RoadBoundary",
            "Cost_Repulsive",
            "Cost_Repulsive_Safe",
            "Cost_Repulsive_Collision",
            "Cost_Repulsive_LogBarrier",
            "Cost_ref",
            "Cost_LaneCenter",
            "Cost_Control",
            "Cost_VelocitySlack",
            "prediction_lane_step_resolved_count",
            "prediction_lane_step_none_count",
            "turn_boundary_recovery_active",
            "turn_boundary_recovery_phase",
        ]

        self._ensure_planning_module_import_path()
        from cpx_planning.MPC.mpc import MPC
        from cpx_planning.behavior_planner import LaneSafetyScorer, RuleBasedBehaviorPlanner
        from cpx_planning.pipeline.control_buffer import MPCControlBuffer
        from cpx_planning.pipeline.decision_record import build_decision_record
        from cpx_planning.pipeline.mpc_feedback import BehaviorMPCFeedback
        from cpx_planning.pipeline.planner_pipeline import CPXPlanningPipeline
        from cpx_planning.pipeline.route_manager import CPXRouteManager
        from cpx_planning.pipeline.reference_gate import FinalReferenceGate
        from cpx_planning.pipeline.reference_generator import ReferenceGenerator
        from cpx_planning.pipeline.reference_pipeline import (
            ReferencePipeline,
        )
        from cpx_planning.pipeline.safety_supervisor import SafetySupervisor
        from cpx_planning.pipeline.velocity_steering_adapter import (
            CarlaVelocitySteeringAdapter,
        )
        from cpx_planning.pipeline.stage_contracts import (
            authorize_mpc_entry,
        )
        from cpx_planning.pipeline.tracker import CPXObstacleTracker
        from cpx_planning.utility.evaluation_metrics import (
            EvaluationMetricsRecorder,
            write_planning_metrics_artifacts,
        )

        mpc_cfg, road_cfg = self._load_mpc_config()
        self.mpc = mpc_instance if mpc_instance is not None else MPC(mpc_cfg=mpc_cfg, road_cfg=road_cfg)
        from cpx_planning.pipeline.actuator_mapper import CarlaActuatorMapper
        self.actuator_mapper = CarlaActuatorMapper(self.config)
        vehicle_curvature_margin = min(
            1.0,
            max(
                0.1,
                float(
                    self.config.get(
                        "reference_vehicle_curvature_safety_factor",
                        0.90,
                    )
                ),
            ),
        )
        vehicle_max_curvature_1pm = (
            math.tan(float(self.mpc.constraints.max_steer_rad))
            / max(1.0e-6, float(self.mpc.wheelbase_m))
        )
        self.config["reference_vehicle_max_curvature_1pm"] = (
            float(vehicle_curvature_margin)
            * float(vehicle_max_curvature_1pm)
        )
        self.reference_generator = ReferenceGenerator(
            config=self.config,
            mpc=self.mpc,
            map_planner=self.map_planner,
            map_waypoint_from_location=self._map_waypoint_from_location,
            lane_id_at_location=self._lane_id_at_location,
            body_frame_xy=self._body_frame_xy,
            target_speed_mps=float(self.target_speed_mps),
            lookahead_m=float(self.lookahead_m),
            drivable_waypoint_from_location=(
                self._drivable_waypoint_from_location
            ),
        )
        self.behavior_runtime_cfg = dict(mpc_cfg.get("behavior_planner_runtime", {}))
        self.lane_safety_scorer = LaneSafetyScorer()
        self.reference_map = _CarlaMapPlannerAdapter(self.map_planner)
        self.tracker = CPXObstacleTracker(
            max_stale_s=float(self.config.get("tracker_max_stale_s", 0.5)),
            max_speed_mps=float(self.config.get("tracker_max_speed_mps", 45.0)),
            max_acceleration_mps2=float(
                self.config.get("tracker_max_acceleration_mps2", 12.0)
            ),
            max_position_jump_m=float(self.config.get("tracker_max_position_jump_m", 12.0)),
        )
        self.planning_pipeline = CPXPlanningPipeline(self)
        self.final_reference_gate = FinalReferenceGate(self.config)
        self.reference_pipeline = ReferencePipeline(
            config=self.config,
            generator=self.reference_generator,
            final_gate=self.final_reference_gate,
            horizon_steps=int(self.mpc.horizon_steps),
            dt_s=float(self.mpc.dt_s),
            default_speed_mps=float(self.target_speed_mps),
        )
        from cpx_planning.pipeline.maneuver_manager import ManeuverManager
        self.maneuver_manager = ManeuverManager(self.config) if bool(behavior_components_enabled) else None
        self._authorize_mpc_entry = authorize_mpc_entry
        self._build_decision_record = build_decision_record
        self.safety_supervisor = SafetySupervisor(
            enabled=bool(self.config.get("safety_supervisor_enabled", True)),
            max_steer_delta=float(self.config.get("safety_max_steer_delta", 0.25)),
            max_throttle_delta=float(self.config.get("safety_max_throttle_delta", 0.45)),
            max_brake_delta=float(self.config.get("safety_max_brake_delta", 0.60)),
            stuck_release_min_accel_mps2=float(
                self.config.get("safety_stuck_release_min_accel_mps2", 0.01)
            ),
        )
        self.velocity_steering_interface_enabled = bool(
            self.config.get("velocity_steering_interface_enabled", False)
        )
        self.velocity_steering_adapter = CarlaVelocitySteeringAdapter(self.config)
        route_sample_distance_m = float(self.config.get("route_sample_distance_m", 2.0))
        if self.map_planner is None or not callable(getattr(self.map_planner, "plan_route_from_locations", None)):
            raise TypeError("ROS CP-X requires CustomGlobalPlannerAdapter as map_planner.")
        self.global_planner = self.map_planner
        self.global_planner_backend = "custom_admap_dijkstra"
        self.global_planner_backend_warning = ""
        road_cfg_from_map = {"lane_count": 1, "lane_width_m": 3.5}
        self.route_manager = route_manager_instance if route_manager_instance is not None else CPXRouteManager(
            global_planner=self.global_planner,
            carla_map=None,
            carla_api=None,
            carla_route_sampling_resolution_m=float(self.config.get("carla_route_sampling_resolution_m", 1.0)),
            carla_reference_smoothing_passes=int(self.config.get("carla_reference_smoothing_passes", 3)),
            carla_turn_connector_smoothing_passes=int(self.config.get("carla_turn_connector_smoothing_passes", 16)),
            carla_reference_boundary_aware=bool(self.config.get("carla_reference_boundary_aware", True)),
            carla_reference_vehicle_half_width_m=float(self.config.get("reference_vehicle_half_width_m", 1.0)),
            carla_reference_boundary_margin_m=float(self.config.get("reference_contract_turn_boundary_margin_m", 0.15)),
            carla_reference_tracking_reserve_m=float(self.config.get("carla_reference_tracking_reserve_m", 0.20)),
            carla_rejoin_min_lateral_m=float(self.config.get("carla_rejoin_min_lateral_m", 0.35)),
            carla_rejoin_max_lateral_m=float(self.config.get("carla_rejoin_max_lateral_m", 3.0)),
            carla_rejoin_distance_m=float(self.config.get("carla_rejoin_distance_m", 8.0)),
            reached_distance_m=float(self.config.get("route_reached_distance_m", 3.0)),
            stale_route_lateral_m=float(self.config.get("route_stale_lateral_m", 12.0)),
        )
        self.road_cfg_from_map = dict(road_cfg_from_map or {})
        self._active_route_summary = None
        self.mpc_feedback = BehaviorMPCFeedback(
            enabled=bool(self.config.get("mpc_feedback_enabled", True)),
            hold_s=float(self.config.get("mpc_feedback_hold_s", 1.5)),
            min_failures=int(self.config.get("mpc_feedback_min_failures", 1)),
        )
        self.control_buffer = MPCControlBuffer(
            enabled=bool(self.config.get("control_buffer_enabled", True)),
            replan_period_s=float(
                self.config.get(
                    "mpc_replan_period_s",
                    getattr(self.mpc, "trajectory_generation_period_s", 0.25),
                )
            ),
            max_reuse_s=float(self.config.get("control_buffer_max_reuse_s", 0.35)),
            max_reference_anchor_jump_m=float(
                self.config.get(
                    "control_buffer_max_reference_anchor_jump_m",
                    0.75,
                )
            ),
            max_predicted_speed_error_mps=float(
                self.config.get(
                    "control_buffer_max_predicted_speed_error_mps",
                    0.75,
                )
            ),
            max_target_speed_jump_mps=float(
                self.config.get(
                    "control_buffer_max_target_speed_jump_mps",
                    1.0,
                )
            ),
        )
        self.cp_message_path = str(
            self.config.get(
                "cp_message_path",
                Path(__file__).resolve().parents[1] / "behavior_planner" / "cp_message.json",
            )
        )
        self.behavior_planner = RuleBasedBehaviorPlanner(
            cp_message_path=str(self.cp_message_path),
            cooperative_message_check_frequency_hz=float(
                self.config.get("cooperative_message_check_frequency_hz", 5.0)
            ),
        ) if bool(behavior_components_enabled) else None
        self.cp_provider = None
        self.active_mpc_cost_profile = "lane_follow"
        self.requested_mpc_cost_profile = "lane_follow"
        self.mpc_cost_profile_active_since_s = 0.0
        self.mpc_cost_profile_switch_reason = "initial"
        self._latest_opencda_update: dict[str, Any] = {}
        self.last_output = None
        self._write_planning_metrics_artifacts = write_planning_metrics_artifacts
        self.evaluation_metrics = EvaluationMetricsRecorder(
            ego_length_m=float(self.config.get("metrics_ego_length_m", 4.5)),
            lateral_conflict_width_m=float(
                self.config.get("metrics_lateral_conflict_width_m", 2.5)
            ),
            min_ego_speed_for_ttc_mps=float(
                self.config.get("metrics_min_ego_speed_for_ttc_mps", 0.5)
            ),
            pet_conflict_radius_m=float(
                self.config.get("metrics_pet_conflict_radius_m", 3.0)
            ),
            pet_bin_size_m=float(self.config.get("metrics_pet_bin_size_m", 3.0)),
        )
        self._metrics_collision_sensor = None
        self._metrics_collision_sensor_available = False
        self._metrics_collision_sensor_error = ""
        self._metrics_last_emitted_collision_count = 0
        self._metrics_boundary_breach_count = 0
        self._metrics_boundary_sample_count = 0
        self._metrics_last_collision_actor_type = ""
        self._metrics_last_collision_impulse = ""
        self._prediction_lane_step_resolved_count = 0
        self._prediction_lane_step_none_count = 0
        self.input_adapter = None
        self.last_adapter_output = None
        self._spawn_metrics_collision_sensor()

    @staticmethod
    def _ensure_planning_module_import_path() -> None:
        """Expose planning_module-local imports used by legacy MPC modules.

        The standalone planning runner is usually launched from
        ``opencda/planning_module``, so imports like ``from cpx_planning.utility...`` work.
        Native OpenCDA scenarios are launched from the repository root, where
        that directory is not on ``sys.path``.  Add it only when the bridge is
        constructed so the default OpenCDA path stays untouched.
        """

        planning_module_root = str(Path(__file__).resolve().parents[1])
        if planning_module_root not in sys.path:
            sys.path.insert(0, planning_module_root)

    def _resolve_global_planner_xodr_path(self) -> Path:
        raw_path = str(
            self.config.get(
                "global_planner_xodr_path",
                self.config.get("xodr_path", ""),
            )
            or ""
        ).strip()
        planning_module_root = Path(__file__).resolve().parents[1]
        if raw_path:
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = planning_module_root / path
            if path.exists():
                return path
            raise FileNotFoundError(f"Global planner xodr_path not found: {path}")

        map_name = str(self.config.get("global_planner_map_name", "") or "").strip()
        if not map_name:
            try:
                map_name = str(self.map_planner.name).split("/")[-1]
            except Exception:
                map_name = ""
        if not map_name:
            map_name = "Town06"
        candidates = []
        if map_name.endswith(".xodr"):
            candidates.append(planning_module_root / "Global_Planner" / "maps" / map_name)
        else:
            candidates.extend([
                planning_module_root / "Global_Planner" / "maps" / f"{map_name}.xodr",
                planning_module_root / "Global_Planner" / "maps" / f"{map_name}_Opt.xodr",
            ])
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not resolve custom global planner .xodr path; checked: "
            + ", ".join(str(candidate) for candidate in candidates)
        )

    def set_destination(
        self,
        *,
        start_location: Any,
        end_location: Any,
        clean: bool = False,
        end_reset: bool = True,
    ) -> None:
        """Set the CP-X global route without using OpenCDA BehaviorAgent."""

        del clean, end_reset
        start_point = self._location_to_point(start_location)
        goal_point = self._location_to_point(end_location)
        self._active_route_summary = self.route_manager.set_destination(
            start_point=start_point,
            goal_point=goal_point,
        )
        self._temporary_destination_state = None
        self._previous_lane_center_reference = []
        self._lane_reference_freeze_count = 0
        self._lane_id_tracker.reset()
        self.control_buffer.reset(reason="destination_updated")
        maneuver_manager = getattr(self, "maneuver_manager", None)
        if maneuver_manager is not None:
            maneuver_manager.reset(reason="destination_updated")
        self._turn_latch_decision = ""
        self._turn_latch_until_sim_time_s = 0.0

    def set_external_global_plan(self, world_plan: Sequence[Any]) -> None:
        """Install a CARLA/Leaderboard route without replanning its topology.

        The leaderboard harness (route_scenario.py) already ran its own
        GlobalRoutePlanner.trace_route() between the scenario's XML
        waypoints before handing us this dense, lane-accurate path via
        set_global_plan() -- by default we just consume it as given, so our
        own route_manager.replan_from() (backed by an independent
        GlobalRoutePlanner over the same OpenDRIVE topology, normally only
        exercised on an obstacle-triggered reroute) never actually runs a
        search for the everyday case. When use_own_global_planner is set,
        immediately re-plan from the same start/goal through our own
        planner instead of keeping the leaderboard-supplied path -- same
        topology and same search algorithm family, so this should trace an
        equivalent route (verify: route completion should not regress) while
        genuinely exercising our own routing code instead of only its
        obstacle-triggered branch.
        """

        self.route_manager.set_external_carla_route(world_plan)
        if bool(self.config.get("use_own_global_planner", False)):
            own_plan_start = dict(self.route_manager._start_point or {})
            if own_plan_start:
                replan_result = self.route_manager.replan_from(
                    start_point=own_plan_start,
                    trigger_reason="use_own_global_planner_initial",
                )
                self._route_replan_last_reason = str(replan_result.reason)
            else:
                self._route_replan_last_reason = (
                    "use_own_global_planner_start_point_unavailable"
                )
        self._active_route_summary = None
        self._temporary_destination_state = None
        self._previous_lane_center_reference = []
        self._lane_reference_freeze_count = 0
        self.control_buffer.reset(reason="external_global_plan_installed")
        maneuver_manager = getattr(self, "maneuver_manager", None)
        if maneuver_manager is not None:
            maneuver_manager.reset(reason="external_global_plan_installed")

    def update_information(
        self,
        *,
        ego_transform: Any,
        ego_speed_kmh: float,
        detected_objects: Any = None,
        v2x_manager: Any = None,
        safety_manager: Any = None,
        map_manager: Any = None,
    ) -> None:
        """Receive the current OpenCDA tick snapshot from VehicleManager.update_info."""

        self._latest_opencda_update = {
            "ego_transform": ego_transform,
            "ego_speed_kmh": float(ego_speed_kmh),
            "detected_objects": detected_objects,
            "v2x_manager": v2x_manager,
            "safety_manager": safety_manager,
            "map_manager": map_manager,
            "sim_time_s": float(self._sim_time_s()),
        }

    def run_step(self):
        """Run one planning tick from the raw inputs stored by the ROS input adapter."""

        if self.input_adapter is None:
            raise RuntimeError("ROSInputAdapter must be attached before the planner runs.")
        try:
            planner_output = self.planning_pipeline.run_step()
        except Exception as exc:
            if self.fallback_policy == "raise":
                raise
            if self.fallback_policy == "opencda":
                raise
            control = self._emergency_stop_control()
            self.last_debug = {
                "sim_time_s": float(self._sim_time_s()),
                "vehicle_id": -1,
                "planner": "cpx_mpc",
                "planner_requested": True,
                "planner_executed": False,
                "fallback_active": True,
                "fallback_reason": str(exc),
                "mpc_fallback_reason": str(exc),
                "control_guard_reason": "fallback_policy_emergency_stop",
                "accel_cmd_mps2": float(getattr(self.mpc.constraints, "min_acceleration_mps2", -3.0)),
                "steer_cmd_rad": 0.0,
            }
            self._record_debug(self.last_debug)
            from cpx_planning.pipeline.output import BehaviorCommand, PlannerDiagnostics, PlannerOutput
            return PlannerOutput(control=control, behavior_command=BehaviorCommand(decision="emergency_brake", target_speed_mps=0.0, stop_requested=True, emergency_brake=True, fsm_state="FALLBACK", debug_reason=str(exc)), acceleration_mps2=float(getattr(self.mpc.constraints, "min_acceleration_mps2", -3.0)), steering_rad=0.0, diagnostics=PlannerDiagnostics(fields=dict(self.last_debug)))
        self.last_output = planner_output
        self.last_debug = planner_output.diagnostics_dict()
        self._record_debug(self.last_debug)
        return planner_output

    def _apply_velocity_steering_interface(
        self,
        *,
        target_speed_mps: float,
        target_steering_rad: float,
        actual_speed_mps: float,
        stop_goal_active: bool,
        emergency_stop: bool,
        sim_time_s: float,
    ):
        """Map planner-owned speed/steering before final safety supervision."""

        from cpx_planning.pipeline.velocity_steering_adapter import (
            VelocitySteeringCommand,
        )

        control, adapter_reason = self.velocity_steering_adapter.run_step(
            command=VelocitySteeringCommand(
                target_speed_mps=float(target_speed_mps),
                target_steering_rad=float(target_steering_rad),
                emergency_stop=bool(emergency_stop),
                stop_goal_active=bool(stop_goal_active),
            ),
            actual_speed_mps=float(actual_speed_mps),
            sim_time_s=float(sim_time_s),
            max_steering_rad=float(self.mpc.constraints.max_steer_rad),
            carla_module=PlannerRuntime,
        )
        applied_steer_rad = (
            float(getattr(control, "steer", 0.0))
            * float(self.mpc.constraints.max_steer_rad)
        )
        debug = {
            "control_interface": "planner_velocity_steering",
            "platform_target_speed_mps": float(target_speed_mps),
            "platform_target_steer_rad": float(target_steering_rad),
            "platform_adapter_steer_rad": float(applied_steer_rad),
            "platform_actual_speed_mps": float(actual_speed_mps),
            "platform_adapter_reason": str(adapter_reason),
            "platform_adapter_throttle": float(
                getattr(control, "throttle", 0.0)
            ),
            "platform_adapter_brake": float(getattr(control, "brake", 0.0)),
            "platform_adapter_steer": float(getattr(control, "steer", 0.0)),
        }
        return (
            control,
            float(self._accel_from_control(control)),
            float(applied_steer_rad),
            debug,
        )

    def execute_planning_pipeline(self):
        """Public OpenCDA bridge port for one full CP-X planning tick."""

        return self._run_full_cpx_pipeline_step()

    def _run_full_cpx_pipeline_step(self):

        from cpx_planning.pipeline.output import (
            BehaviorCommand,
            PlannerDiagnostics,
            PlannerOutput,
        )
        from cpx_planning.pipeline.reference_pipeline import (
            ReferencePipelineRequest,
        )

        runtime_inputs = self.input_adapter.runtime_inputs()
        latest_update: dict[str, Any] = {
            "safety_manager": runtime_inputs.get("safety_manager"),
            "v2x_nearby_count": int(runtime_inputs.get("v2x_nearby_count", 0) or 0),
        }
        sim_time_s = float(runtime_inputs["sim_time_s"])
        ego_pose = dict(runtime_inputs["ego_pose"])
        ego_location = PlannerLocation(x=float(ego_pose["x"]), y=float(ego_pose["y"]), z=float(ego_pose.get("z", 0.0)))
        ego_yaw_rad = float(ego_pose["heading_rad"])
        ego_speed_mps = float(runtime_inputs["ego_speed_mps"])
        ego_transform = PlannerTransform(location=ego_location, rotation=PlannerRuntime.Rotation(yaw=math.degrees(float(ego_yaw_rad)))) if hasattr(PlannerRuntime, "Rotation") else PlannerTransform(location=ego_location)
        measured_accel_mps2 = self.actuator_mapper.update_measurement(speed_mps=float(ego_speed_mps), timestamp_s=float(sim_time_s))
        local_object_snapshots = [dict(item) for item in list(runtime_inputs["local_object_snapshots"] or [])]
        cp_payload = dict(runtime_inputs["cp_payload"] or {})
        object_snapshots = self._fused_planning_object_snapshots(local_object_snapshots=local_object_snapshots, cp_obstacles=list(cp_payload.get("obstacles", []) or []), ego_location=ego_location, sim_time_s=float(sim_time_s))
        mpc_object_snapshots = self._limit_obstacles_for_mpc(object_snapshots=object_snapshots, ego_location=ego_location)
        front_gap_m, front_gap_actor_id_early = self._front_gap_m(ego_location=ego_location, ego_yaw_rad=ego_yaw_rad, object_snapshots=object_snapshots, return_actor_id=True)
        front_gap_obstacle_speed_mps_early = None
        if front_gap_actor_id_early:
            for _snapshot in object_snapshots:
                if str(self._object_track_id(_snapshot)) == str(front_gap_actor_id_early):
                    front_gap_obstacle_speed_mps_early = max(0.0, float(_snapshot.get("v", _snapshot.get("speed_mps", 0.0)) or 0.0))
                    break
        from cpx_planning.pipeline.speed_planner import effective_emergency_gap_m as _effective_emergency_gap_m

        emergency_front_gap_m = _effective_emergency_gap_m(base_emergency_gap_m=max(0.5, float(self.config.get("following_emergency_gap_m", 3.0))), ego_speed_mps=float(ego_speed_mps), front_obstacle_speed_mps=front_gap_obstacle_speed_mps_early, standstill_buffer_m=max(0.0, float(self.config.get("following_emergency_standstill_buffer_m", 1.0))), time_headway_s=max(0.1, float(self.config.get("following_time_headway_s", 1.5))))
        front_gap_at_emergency_threshold = front_gap_m is not None and float(front_gap_m) <= float(emergency_front_gap_m)
        lane_change_already_committed = bool(self._route_tracking_lane_change_reference)
        stop_goal_active = bool(front_gap_at_emergency_threshold and not lane_change_already_committed)
        if front_gap_at_emergency_threshold and lane_change_already_committed:
            speed_ref_mps = max(0.0, float(self.config.get("following_emergency_lane_change_crawl_speed_mps", 0.5)))
        else:
            speed_ref_mps = 0.0 if stop_goal_active else self.target_speed_mps
        current_state = [float(ego_location.x), float(ego_location.y), float(ego_speed_mps), float(ego_yaw_rad)]

        behavior_debug: dict[str, Any] = {}
        reference_debug: dict[str, Any] = {}
        try:
            destination_state, lane_center_reference, behavior_debug, reference_debug = (
                self._plan_behavior_and_reference(
                    ego_location=ego_location,
                    ego_yaw_rad=ego_yaw_rad,
                    ego_speed_mps=ego_speed_mps,
                    speed_ref_mps=speed_ref_mps,
                    object_snapshots=object_snapshots,
                    stop_goal_active=stop_goal_active,
                    cp_payload=cp_payload,
                )
            )
        except Exception as exc:
            if self.debug:
                print(f"[CP-X OpenCDA Bridge] behavior/reference pipeline failed: {exc}")
            generated_fallback = self.reference_generator.build_lane_fallback(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_state=current_state,
                speed_ref_mps=float(speed_ref_mps),
            )
            destination_state = generated_fallback.destination_state
            lane_center_reference = generated_fallback.samples
            behavior_debug = {
                "decision": "lane_follow",
                "lc_state": "FALLBACK",
                "target_lane_id": "",
                "current_lane_id": self._lane_id_at_location(ego_location),
                "pipeline_error": str(exc),
            }
            reference_debug = {
                "reference_source": "current_lane_center_exception_fallback",
                "pipeline_error": str(exc),
            }

        # The behavior/reference plan owns the final stop state. The raw
        # front-gap threshold is only an input proposal and must not re-latch
        # stop after candidate evaluation has selected a safe route maneuver.
        selected_stop_goal_active = behavior_debug.get(
            "stop_goal_active",
            stop_goal_active,
        )
        speed_ref_mps = max(
            0.0,
            float(behavior_debug.get("target_speed_mps", speed_ref_mps)),
        )
        proposed_post_plan_speed_mps = float(speed_ref_mps)
        speed_ceiling_applied = False
        speed_ceiling_reduction_mps = 0.0
        speed_plan_ceiling = reference_debug.get("speed_plan_target_mps")
        if speed_plan_ceiling not in (None, ""):
            from cpx_planning.pipeline.speed_planner import (
                enforce_speed_ceiling,
            )

            ceiling_result = enforce_speed_ceiling(
                proposed_target_mps=float(speed_ref_mps),
                ceiling_mps=float(speed_plan_ceiling),
                destination_state=destination_state,
                reference_samples=lane_center_reference,
            )
            speed_ref_mps = float(ceiling_result.target_speed_mps)
            destination_state = list(ceiling_result.destination_state)
            lane_center_reference = list(ceiling_result.reference_samples)
            speed_ceiling_applied = bool(ceiling_result.applied)
            speed_ceiling_reduction_mps = float(ceiling_result.reduction_mps)
        reference_debug.update({
            "speed_owner_proposed_post_plan_target_mps": float(
                proposed_post_plan_speed_mps
            ),
            "speed_owner_ceiling_applied": bool(speed_ceiling_applied),
            "speed_owner_ceiling_reduction_mps": float(
                speed_ceiling_reduction_mps
            ),
        })
        mpc_stop_goal_active = bool(selected_stop_goal_active) or str(
            behavior_debug.get("decision", "")
        ) in {"stop_at_intersection", "stop_sign", "emergency_brake"}
        behavior_decision_normalized = str(
            behavior_debug.get("decision", "")
        ).strip().lower()
        normal_stop_requested = behavior_decision_normalized in {
            "stop_at_intersection",
            "stop_sign",
        }
        emergency_brake_requested = (
            behavior_decision_normalized == "emergency_brake"
        )
        if bool(mpc_stop_goal_active):
            speed_ref_mps = 0.0
        if bool(mpc_stop_goal_active) and len(destination_state) >= 3:
            destination_state = list(destination_state)
            destination_state[2] = 0.0
        stop_target_forward_m_debug = ""
        stop_target_debug = (
            behavior_debug.get("stop_target")
            if isinstance(behavior_debug.get("stop_target"), Mapping)
            else None
        )
        if bool(mpc_stop_goal_active) and isinstance(stop_target_debug, Mapping):
            try:
                stop_target_forward_m_debug, _ = self._body_frame_xy(
                    origin_x_m=float(ego_location.x),
                    origin_y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    target_x_m=float(stop_target_debug.get("x_m", stop_target_debug.get("x", ego_location.x))),
                    target_y_m=float(stop_target_debug.get("y_m", stop_target_debug.get("y", ego_location.y))),
                )
            except Exception:
                stop_target_forward_m_debug = ""

        reference_pipeline_result = self.reference_pipeline.finalize(
            ReferencePipelineRequest(
                destination_state=destination_state,
                reference_samples=lane_center_reference,
                current_state=current_state,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(speed_ref_mps),
                behavior_decision=str(behavior_debug.get("decision", "")),
                behavior_fsm_state=str(behavior_debug.get("lc_state", "")),
                current_lane_id=int(
                    behavior_debug.get("current_lane_id", 0) or 0
                ),
                target_lane_id=int(
                    behavior_debug.get("target_lane_id", 0) or 0
                ),
                stop_goal_active=bool(mpc_stop_goal_active),
                stop_target=(
                    behavior_debug.get("stop_target")
                    if isinstance(behavior_debug.get("stop_target"), Mapping)
                    else None
                ),
                route_points=self._active_global_route_points(),
            )
        )
        destination_state = list(reference_pipeline_result.destination_state)
        lane_center_reference = [
            dict(sample)
            for sample in reference_pipeline_result.reference_samples
        ]
        mpc_reference_stabilizer_reason = str(
            reference_pipeline_result.conditioning_reason
        )
        reference_debug["mpc_reference_stabilizer_reason"] = str(
            mpc_reference_stabilizer_reason
        )
        reference_debug.update(reference_pipeline_result.as_debug_fields())
        final_reference_gate = reference_pipeline_result.gate
        if not bool(final_reference_gate.accepted):
            gate_reason = "final_reference_gate:" + str(
                final_reference_gate.reason
            )
            mpc_reference_stabilizer_reason = ";".join(
                reason
                for reason in (
                    str(mpc_reference_stabilizer_reason),
                    str(gate_reason),
                )
                if reason
            )
            reference_debug["mpc_reference_stabilizer_reason"] = str(
                mpc_reference_stabilizer_reason
            )
            reference_debug["candidate_pipeline_selected_status"] = "infeasible"
            reference_debug["candidate_pipeline_selected_reason"] = ";".join(
                reason
                for reason in (
                    str(
                        reference_debug.get(
                            "candidate_pipeline_selected_reason", ""
                        )
                    ),
                    str(gate_reason),
                )
                if reason
            )
        reference_debug["final_reference_geometry_source"] = str(
            reference_debug.get("reference_source", "unknown")
        )
        destination_forward_m, destination_lateral_m = self._body_frame_xy(
            origin_x_m=float(ego_location.x),
            origin_y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad),
            target_x_m=float(destination_state[0]),
            target_y_m=float(destination_state[1]),
        )
        reference_first_forward_m = ""
        reference_first_lateral_m = ""
        if lane_center_reference:
            first_reference = lane_center_reference[0]
            reference_first_forward_m, reference_first_lateral_m = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(first_reference.get("x_ref_m", first_reference.get("x", ego_location.x))),
                target_y_m=float(first_reference.get("y_ref_m", first_reference.get("y", ego_location.y))),
            )

        mpc_status = str(getattr(self.mpc, "_last_status", ""))
        mode_transition_guard_reason = self._apply_behavior_mode_transition_guard(
            decision=str(behavior_debug.get("decision", "")),
            lc_state=str(behavior_debug.get("lc_state", "")),
            target_lane_id=int(behavior_debug.get("target_lane_id", 0) or 0),
            stop_goal_active=bool(mpc_stop_goal_active),
        )
        mpc_entry_authorization = self._authorize_mpc_entry(
            candidate_status=reference_debug.get(
                "candidate_pipeline_selected_status", ""
            ),
            candidate_name=reference_debug.get(
                "candidate_pipeline_selected", ""
            ),
            candidate_reason=reference_debug.get(
                "candidate_pipeline_selected_reason", ""
            ),
            final_reference_accepted=bool(final_reference_gate.accepted),
            final_reference_reason=str(final_reference_gate.reason),
            behavior_decision=str(behavior_debug.get("decision", "")),
        )
        reference_debug.update(mpc_entry_authorization.as_debug_fields())
        candidate_hard_gate_reason = (
            ""
            if bool(mpc_entry_authorization.allowed)
            else "candidate_hard_gate:" + str(mpc_entry_authorization.reason)
        )
        stationary_traffic_stop_hold = _should_suspend_mpc_for_normal_stop(
            candidate_hard_gate_active=bool(candidate_hard_gate_reason),
            stop_goal_active=bool(mpc_stop_goal_active),
            behavior_decision=str(behavior_debug.get("decision", "")),
            ego_speed_mps=float(ego_speed_mps),
            suspend_speed_mps=max(
                0.0,
                float(
                    self.config.get(
                        "normal_stop_mpc_suspend_speed_mps",
                        0.30,
                    )
                ),
            ),
        )
        control_context_key = "|".join((
            str(behavior_debug.get("decision", "")),
            str(behavior_debug.get("lc_state", "")),
            str(behavior_debug.get("target_lane_id", "")),
            str(reference_debug.get("reference_source", "")),
            str(bool(mpc_stop_goal_active)),
            str(behavior_debug.get("traffic_signal_state", "")),
            # A buffered control sequence was optimized against whichever
            # vehicle _front_gap_m() picked as "ahead of me" -- if that
            # identity changes (e.g. the source-lane vehicle a lane change
            # was following drops out of the gate and a different, target-
            # lane vehicle takes over), the old sequence's braking/following
            # intent no longer means what it did when it was solved, even
            # though decision/lc_state/target_lane haven't changed yet.
            str(reference_debug.get("front_gap_actor_id", "")),
        ))
        reference_anchor_xy = (
            (
                float(
                    lane_center_reference[0].get(
                        "x_ref_m",
                        lane_center_reference[0].get("x", ego_location.x),
                    )
                ),
                float(
                    lane_center_reference[0].get(
                        "y_ref_m",
                        lane_center_reference[0].get("y", ego_location.y),
                    )
                ),
            )
            if lane_center_reference
            else None
        )
        # MPC constrains jerk between the previous control input and the new
        # acceleration sequence. Seed that constraint with the acceleration
        # command actually sent last tick, not the measured vehicle response.
        # The latter contains actuator lag and can stay strongly negative
        # after the speed target has recovered, otherwise forcing every new
        # solve to continue braking until the vehicle is almost stationary.
        mpc_jerk_seed_accel_mps2 = float(self._last_accel_mps2)
        if str(candidate_hard_gate_reason):
            self.control_buffer.reset(reason="control_buffer_reference_hard_veto")
        elif bool(stationary_traffic_stop_hold):
            self.control_buffer.update_from_solution(
                u_solution=[[0.0, 0.0]],
                plan_time_s=float(sim_time_s),
                dt_s=float(self.mpc.dt_s),
                context_key=str(control_context_key),
                reference_anchor_xy=reference_anchor_xy,
            )
        failed_replan_buffer_reused = False
        failed_replan_maneuver_steer_held = False
        try:
            if str(candidate_hard_gate_reason):
                raise RuntimeError(str(candidate_hard_gate_reason))
            force_replan = (
                not bool(stationary_traffic_stop_hold)
                and (
                    bool(mpc_stop_goal_active)
                    or str(behavior_debug.get("decision", ""))
                    in {
                        "stop_at_intersection",
                        "stop_sign",
                        "emergency_brake",
                        "intersection_turn_left",
                        "intersection_turn_right",
                    }
                    or bool(mode_transition_guard_reason)
                )
            )
            low_speed_buffer_replan = self._low_speed_control_buffer_force_replan(
                ego_speed_mps=float(ego_speed_mps),
                behavior_decision=str(behavior_debug.get("decision", "")),
                behavior_fsm_state=str(behavior_debug.get("lc_state", "")),
                stop_goal_active=bool(mpc_stop_goal_active),
            )
            force_replan = bool(force_replan) or bool(low_speed_buffer_replan)
            mpc_replan_executed = bool(
                self.control_buffer.should_replan(
                    sim_time_s=float(sim_time_s),
                    force_replan=bool(force_replan),
                    context_key=str(control_context_key),
                    reference_anchor_xy=reference_anchor_xy,
                    ego_speed_mps=float(ego_speed_mps),
                    target_speed_mps=float(speed_ref_mps),
                    speed_error_crossing_deadband_mps=float(
                        self.config.get(
                            "control_buffer_speed_crossing_deadband_mps",
                            0.15,
                        )
                    ),
                )
            )
            if bool(mpc_replan_executed):
                road_envelope_payload_world = (
                    self._current_route_tracking_lane_change_envelope_payload_world()
                )
                if road_envelope_payload_world is None:
                    road_envelope_payload_world = (
                        self._rolling_turn_envelope_payload_world(
                            behavior_decision=str(
                                behavior_debug.get("decision", "")
                            ),
                            reference_samples=lane_center_reference,
                        )
                    )
                self.mpc.plan_trajectory(
                    current_state=current_state,
                    destination_state=destination_state,
                    object_snapshots=self._mpc_object_snapshots_with_prediction(
                        mpc_object_snapshots,
                        prediction_trajectories=reference_debug.get(
                            "prediction_trajectories", {}
                        ),
                    ),
                    current_acceleration_mps2=float(mpc_jerk_seed_accel_mps2),
                    current_steering_rad=float(self._last_steer_rad),
                    lane_center_reference_samples=lane_center_reference,
                    stop_goal_active=bool(mpc_stop_goal_active),
                    road_envelope_payload_world=road_envelope_payload_world,
                )
                mpc_status = str(getattr(self.mpc, "_last_status", "")).strip().lower()
                if mpc_status and mpc_status not in {"solved", "solved inaccurate"}:
                    raise RuntimeError(f"MPC status={mpc_status}")
                u_solution = getattr(self.mpc, "_last_u_solution", None)
                if u_solution is None or len(u_solution) == 0:
                    raise RuntimeError("MPC did not expose a control solution")
                x_solution = getattr(self.mpc, "_last_x_solution", None)
                predicted_speed_sequence_mps = (
                    None
                    if x_solution is None or len(x_solution) == 0
                    # Column 2 is speed; the (x, y) world-origin offset
                    # baked into _last_x_solution doesn't touch it.
                    else [float(state[2]) for state in x_solution]
                )
                self.control_buffer.update_from_solution(
                    u_solution=u_solution,
                    plan_time_s=float(sim_time_s),
                    dt_s=float(self.mpc.dt_s),
                    context_key=str(control_context_key),
                    reference_anchor_xy=reference_anchor_xy,
                    predicted_speed_sequence_mps=predicted_speed_sequence_mps,
                    target_speed_mps=float(speed_ref_mps),
                )
                accel_mps2 = float(u_solution[0, 0])
                steer_rad = float(u_solution[0, 1])
            else:
                buffered = self.control_buffer.sample(
                    sim_time_s=float(sim_time_s),
                    context_key=str(control_context_key),
                    reference_anchor_xy=reference_anchor_xy,
                )
                if buffered is None:
                    raise RuntimeError("MPC control buffer empty")
                accel_mps2, steer_rad, _buffer_reason = buffered
                mpc_status = (
                    "stop_hold_direct"
                    if bool(stationary_traffic_stop_hold)
                    else "buffer_reuse"
                )
            self._set_actuator_context(
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(speed_ref_mps),
                stop_goal_active=bool(mpc_stop_goal_active),
            )
            control = self._control_from_mpc(accel_mps2, steer_rad)
            if bool(stationary_traffic_stop_hold):
                hold_brake = min(
                    1.0,
                    max(
                        0.0,
                        float(
                            self.config.get(
                                "normal_stop_mpc_suspend_brake",
                                0.08,
                            )
                        ),
                    ),
                )
                control = PlannerControl(
                    throttle=0.0,
                    brake=float(hold_brake),
                    steer=float(getattr(control, "steer", 0.0)),
                )
                accel_mps2 = float(self._accel_from_control(control))
                steer_rad = float(self._steer_rad_from_control(control))
            fallback_reason = ""
        except Exception as exc:
            mpc_replan_executed = True
            hard_gate_active = str(exc).startswith("candidate_hard_gate:")
            hard_gate_emergency_stop = _hard_gate_requires_emergency_stop(
                fallback_reason=str(exc),
                behavior_decision=str(behavior_debug.get("decision", "")),
                stop_goal_active=bool(mpc_stop_goal_active),
            )
            if bool(hard_gate_active):
                mpc_replan_executed = False
            fallback_reason = str(exc)
            if bool(hard_gate_emergency_stop):
                control = self._emergency_stop_control()
                accel_mps2 = float(self._last_accel_mps2)
                steer_rad = 0.0
            else:
                normalized_behavior = str(
                    behavior_debug.get("decision", "")
                ).strip().lower()
                maneuver_tracking_active = normalized_behavior in {
                    "intersection_turn_left",
                    "intersection_turn_right",
                    "lane_change_left",
                    "lane_change_right",
                }
                buffered_after_failure = (
                    self.control_buffer.sample(
                        sim_time_s=float(sim_time_s),
                        context_key=str(control_context_key),
                        reference_anchor_xy=reference_anchor_xy,
                    )
                    if (
                        bool(maneuver_tracking_active)
                        and not bool(mpc_stop_goal_active)
                    )
                    else None
                )
                if buffered_after_failure is not None:
                    (
                        accel_mps2,
                        steer_rad,
                        _failed_replan_buffer_reason,
                    ) = buffered_after_failure
                    self._set_actuator_context(
                        ego_speed_mps=float(ego_speed_mps),
                        target_speed_mps=float(speed_ref_mps),
                        stop_goal_active=False,
                    )
                    control = self._control_from_mpc(
                        float(accel_mps2), float(steer_rad)
                    )
                    self._last_accel_mps2 = float(accel_mps2)
                    self._last_steer_rad = float(steer_rad)
                    failed_replan_buffer_reused = True
                else:
                    previous_valid_steer_rad = float(self._last_steer_rad)
                    control = self._fallback_control(
                        ego_transform=ego_transform,
                        ego_speed_mps=ego_speed_mps,
                        destination_state=destination_state,
                        stop_goal_active=mpc_stop_goal_active,
                    )
                    accel_mps2 = self._last_accel_mps2
                    steer_rad = self._last_steer_rad
                    if bool(maneuver_tracking_active):
                        # A failed maneuver solve must not transfer lateral
                        # ownership to the destination-point fallback.  Hold
                        # the last accepted steering direction only briefly.
                        # Once the optimized buffer has already expired,
                        # repeatedly holding the full turn command can drive
                        # the vehicle off-road forever (the diagnosed Town06
                        # vegetation collision). Decay it toward neutral so a
                        # prolonged solver outage is fail-passive laterally.
                        failed_steer_decay = (
                            max(
                                0.0,
                                min(
                                    1.0,
                                    float(
                                        self.config.get(
                                            "turn_failed_replan_steer_decay",
                                            0.65,
                                        )
                                    ),
                                ),
                            )
                            if normalized_behavior in {
                                "intersection_turn_left",
                                "intersection_turn_right",
                            }
                            else 1.0
                        )
                        steer_rad = (
                            float(previous_valid_steer_rad)
                            * float(failed_steer_decay)
                        )
                        self._set_actuator_context(
                            ego_speed_mps=float(ego_speed_mps),
                            target_speed_mps=float(speed_ref_mps),
                            stop_goal_active=False,
                        )
                        control = self._control_from_mpc(
                            float(accel_mps2), float(steer_rad)
                        )
                        self._last_steer_rad = float(steer_rad)
                        failed_replan_maneuver_steer_held = True
            if bool(hard_gate_active) and str(
                behavior_debug.get("decision", "")
            ) == "emergency_brake":
                mpc_status = "emergency_brake_direct"
            else:
                mpc_status = (
                    "candidate_hard_gate"
                    if bool(hard_gate_active)
                    else "buffer_reuse_after_failed_replan"
                    if bool(failed_replan_buffer_reused)
                    else "maneuver_steer_hold_after_failed_replan"
                    if bool(failed_replan_maneuver_steer_held)
                    else str(getattr(self.mpc, "_last_status", str(exc)))
                )
            if not self._warned:
                print(f"[CP-X OpenCDA Bridge] MPC fallback active: {fallback_reason}")
                self._warned = True
        mpc_feedback_record_reason = self.mpc_feedback.record_result(
            decision=str(behavior_debug.get("decision", "")),
            target_lane_id=int(behavior_debug.get("target_lane_id", 0) or 0),
            status=str(mpc_status),
            reason=str(fallback_reason),
            timestamp_s=float(sim_time_s),
            success=not bool(fallback_reason),
        )

        platform_adapter_debug: dict[str, object] = {
            "control_interface": "mpc_acceleration_steering",
        }
        hard_gate_active = str(fallback_reason).startswith(
            "candidate_hard_gate:"
        )
        # The platform adapter is part of actuation, not a post-processing
        # owner. Run it before every safety guard so signal, boundary and
        # collision decisions remain authoritative at apply_control(). Keep
        # non-gate MPC fallback controls intact instead of converting them
        # back into a routine target-speed command.
        if bool(self.velocity_steering_interface_enabled) and (
            not str(fallback_reason) or bool(hard_gate_active)
        ):
            behavior_decision = str(
                behavior_debug.get("decision", "")
            ).strip().lower()
            (
                control,
                accel_mps2,
                steer_rad,
                platform_adapter_debug,
            ) = self._apply_velocity_steering_interface(
                target_speed_mps=float(speed_ref_mps),
                target_steering_rad=float(steer_rad),
                actual_speed_mps=float(ego_speed_mps),
                stop_goal_active=bool(mpc_stop_goal_active),
                emergency_stop=bool(
                    _hard_gate_requires_emergency_stop(
                        fallback_reason=str(fallback_reason),
                        behavior_decision=str(behavior_decision),
                        stop_goal_active=bool(mpc_stop_goal_active),
                    )
                    or behavior_decision == "emergency_brake"
                ),
                sim_time_s=float(sim_time_s),
            )

        control, accel_mps2, steer_rad, control_guard_reason = (
            self.safety_supervisor.enforce_signal_stop(
                control=control,
                carla_module=self.carla,
                accel_mps2=float(accel_mps2),
                steer_rad=float(steer_rad),
                ego_transform=ego_transform,
                ego_speed_mps=float(ego_speed_mps),
                destination_state=destination_state,
                stop_goal_active=bool(mpc_stop_goal_active),
                traffic_signal_state=str(
                    behavior_debug.get("traffic_signal_state", "")
                ),
                min_acceleration_mps2=float(
                    self.mpc.constraints.min_acceleration_mps2
                ),
                control_factory=self._control_from_mpc,
                config=self.config,
                stop_target_forward_m=stop_target_forward_m_debug,
            )
        )
        boundary_guard_reason = ""
        boundary_snapshot = None
        turn_behavior_active = str(behavior_debug.get("decision", "")) in {
            "intersection_turn_left",
            "intersection_turn_right",
        }
        if bool(turn_behavior_active):
            boundary_snapshot = self._road_boundary_metrics(
                ego_location,
                record_sample=True,
                ego_yaw_rad=float(ego_yaw_rad),
                reference_samples=lane_center_reference,
            )
            if bool(self.config.get("boundary_recovery_enabled", False)):
                self._update_boundary_recovery_request(
                    boundary_snapshot=boundary_snapshot,
                    behavior_decision=str(
                        behavior_debug.get("decision", "")
                    ),
                    sim_time_s=float(sim_time_s),
                    recovery_planned=bool(
                        behavior_debug.get(
                            "boundary_recovery_active",
                            False,
                        )
                    ),
                    recovery_reference_feasible=bool(
                        str(
                            reference_debug.get(
                                "candidate_pipeline_selected_status",
                                "",
                            )
                        ).strip().lower()
                        != "infeasible"
                        and bool(final_reference_gate.accepted)
                    ),
                )
            else:
                self._reset_boundary_recovery_request()
            (
                control,
                accel_mps2,
                steer_rad,
                boundary_guard_reason,
            ) = self.safety_supervisor.enforce_turn_boundary(
                control=control,
                carla_module=self.carla,
                accel_mps2=float(accel_mps2),
                steer_rad=float(steer_rad),
                ego_speed_mps=float(ego_speed_mps),
                behavior_decision=str(behavior_debug.get("decision", "")),
                boundary_clearance_m=boundary_snapshot.get(
                    "road_boundary_clearance_m", ""
                ),
                min_acceleration_mps2=float(
                    self.mpc.constraints.min_acceleration_mps2
                ),
                config=self.config,
                control_factory=self._control_from_mpc,
                boundary_recovery_planned=bool(
                    behavior_debug.get(
                        "boundary_recovery_active",
                        False,
                    )
                ),
            )
            control_guard_reason = ";".join(
                reason
                for reason in (
                    str(control_guard_reason),
                    str(boundary_guard_reason),
                )
                if reason
            )
        else:
            self._reset_boundary_recovery_request()
        pre_supervisor_accel_mps2 = float(accel_mps2)
        pre_supervisor_steer_rad = float(steer_rad)
        control, safety_supervisor_reason = self.safety_supervisor.filter_control(
            control=control,
            carla_module=self.carla,
            safety_manager=latest_update.get("safety_manager"),
            behavior_decision=str(behavior_debug.get("decision", "")),
            traffic_signal_state=str(behavior_debug.get("traffic_signal_state", "")),
            stop_goal_active=bool(mpc_stop_goal_active),
            planner_accel_mps2=float(pre_supervisor_accel_mps2),
            sim_time_s=float(sim_time_s),
        )
        post_supervisor_accel_mps2 = self._accel_from_control(control)
        post_supervisor_steer_rad = self._steer_rad_from_control(control)
        self._last_accel_mps2 = float(post_supervisor_accel_mps2)
        self._last_steer_rad = float(post_supervisor_steer_rad)
        cp_summary = dict(getattr(self.cp_provider, "last_publish_summary", {}) or {})
        diagnostics = {
            "sim_time_s": float(self._sim_time_s()),
            "vehicle_id": -1,
            "x_m": float(ego_location.x),
            "y_m": float(ego_location.y),
            "yaw_deg": float(ego_transform.rotation.yaw),
            "speed_mps": float(ego_speed_mps),
            "measured_accel_mps2": float(measured_accel_mps2),
            "mpc_jerk_seed_accel_mps2": float(mpc_jerk_seed_accel_mps2),
            "planner": "cpx_mpc",
            **platform_adapter_debug,
            "object_count": len(object_snapshots),
            "mpc_object_count": len(mpc_object_snapshots),
            "local_object_count": len(local_object_snapshots),
            **self._perception_diagnostics(),
            "v2x_nearby_count": int(latest_update.get("v2x_nearby_count", 0) or 0),
            "cp_provider_summary": dict(cp_summary),
            "cp_provider_source": str(cp_summary.get("provider_source", "")),
            "native_opencda_required": bool(cp_summary.get("native_opencda_required", False)),
            "native_opencda_available": bool(cp_summary.get("native_opencda_available", False)),
            "cp_obstacle_count": int(cp_summary.get("obstacle_count", 0) or 0),
            "cp_control_count": int(cp_summary.get("control_count", 0) or 0),
            "cp_observer_cav_count": int(
                cp_summary.get("observer_cav_count", 0) or 0
            ),
            "cp_observer_cav_ids": ",".join(
                str(item)
                for item in list(cp_summary.get("observer_cav_ids", []) or [])
            ),
            "cp_multi_observer_obstacle_count": int(
                cp_summary.get("multi_observer_obstacle_count", 0) or 0
            ),
            "cp_blind_spot_shared_count": int(
                cp_summary.get("blind_spot_shared_count", 0) or 0
            ),
            "cp_blind_spot_shared_actor_ids": ",".join(
                str(item)
                for item in list(
                    cp_summary.get("blind_spot_shared_actor_ids", []) or []
                )
            ),
            **self._cooperative_actor_evidence(
                cp_summary=cp_summary,
                prediction_trajectories=dict(
                    reference_debug.get("prediction_trajectories", {}) or {}
                ),
                selected_reference=lane_center_reference,
            ),
            "cp_visibility_filter_enabled": bool(
                cp_summary.get("visibility_filter_enabled", False)
            ),
            "cp_visibility_backend": str(
                cp_summary.get("visibility_backend", "")
            ),
            "front_gap_m": "" if front_gap_m is None else float(front_gap_m),
            "stop_goal_active": bool(mpc_stop_goal_active),
            "normal_stop_requested": bool(normal_stop_requested),
            "emergency_brake_requested": bool(emergency_brake_requested),
            "emergency_brake_control_active": bool(
                emergency_brake_requested
                or hard_gate_active
                or str(safety_supervisor_reason).startswith(
                    "safety_supervisor_emergency_stop:"
                )
            ),
            "normal_stop_mpc_suspended": bool(stationary_traffic_stop_hold),
            "normal_stop_mpc_suspend_speed_mps": float(
                self.config.get("normal_stop_mpc_suspend_speed_mps", 0.30)
            ),
            "normal_stop_mpc_suspend_brake": float(
                self.config.get("normal_stop_mpc_suspend_brake", 0.08)
            ),
            "behavior_decision": str(behavior_debug.get("decision", "")),
            "static_obstacle_stop_active_input": bool(
                getattr(self, "_last_static_obstacle_stop_active_input", False)
            ),
            "static_obstacle_replan_status": str(
                getattr(self, "_static_obstacle_replan_status", "idle")
            ),
            "static_obstacle_replan_reason": str(
                getattr(self, "_static_obstacle_replan_reason", "not_requested")
            ),
            "static_obstacle_candidate_id": str(
                getattr(self, "_static_obstacle_candidate_id", "")
            ),
            "static_obstacle_blocked_lane_id": getattr(
                self, "_static_obstacle_blocked_lane_id", ""
            ),
            "static_obstacle_route_transition_pending": bool(
                getattr(
                    self,
                    "_static_obstacle_route_transition_pending",
                    False,
                )
            ),
            "behavior_fsm_state": str(behavior_debug.get("lc_state", "")),
            "current_lane_id": behavior_debug.get("current_lane_id", ""),
            "behavior_target_lane_id": behavior_debug.get("target_lane_id", ""),
            "traffic_signal_state": behavior_debug.get("traffic_signal_state", ""),
            "traffic_signal_raw_state": behavior_debug.get(
                "traffic_signal_raw_state", ""
            ),
            "traffic_signal_resolved_state": behavior_debug.get(
                "traffic_signal_resolved_state", ""
            ),
            "traffic_signal_filtered_state": behavior_debug.get(
                "traffic_signal_filtered_state", ""
            ),
            "traffic_signal_behavior_state": behavior_debug.get(
                "traffic_signal_behavior_state",
                behavior_debug.get("traffic_signal_state", ""),
            ),
            "traffic_control_from_cp": behavior_debug.get("traffic_control_from_cp", ""),
            "lane_safety_scores": json.dumps(behavior_debug.get("lane_safety_scores", {}), default=str),
            "reference_pipeline_stage": str(reference_debug.get("stage", "")),
            "reference_pipeline_intent": str(reference_debug.get("intent_mode", "")),
            "reference_pipeline_fallback": str(reference_debug.get("fallback_reason", "")),
            "planner_input_cp_traffic_control_count": reference_debug.get("planner_input_cp_traffic_control_count", ""),
            "planner_input_prediction_risky_lane_count": reference_debug.get("planner_input_prediction_risky_lane_count", ""),
            "planner_input_perception_planning_count": reference_debug.get("planner_input_perception_planning_count", ""),
            "planner_input_cp_obstacle_count": reference_debug.get("planner_input_cp_obstacle_count", ""),
            "planner_input_frame_timestamp_s": reference_debug.get("planner_input_frame_timestamp_s", ""),
            "cp_message_timestamp_s": reference_debug.get("cp_message_timestamp_s", ""),
            "cp_message_age_s": reference_debug.get("cp_message_age_s", ""),
            "cp_message_valid": reference_debug.get("cp_message_valid", ""),
            "destination_x": float(destination_state[0]),
            "destination_y": float(destination_state[1]),
            "destination_forward_m": float(destination_forward_m),
            "destination_lateral_m": float(destination_lateral_m),
            "destination_lane_id": (
                int(destination_state[4]) if len(destination_state) >= 5 else ""
            ),
            "reference_first_forward_m": reference_first_forward_m,
            "reference_first_lateral_m": reference_first_lateral_m,
            "mpc_trajectory_point_count": len(self._last_mpc_trajectory_points()),
            "global_route_point_count": len(self._active_global_route_points()),
            "map_match_valid": bool(
                self._diagnostic_map_matching.get("valid", False)
            ),
            "map_match_ad_lane_id": self._diagnostic_map_matching.get(
                "ad_lane_id", ""
            ),
            "map_match_road_id": self._diagnostic_map_matching.get("road_id", ""),
            "map_match_section_id": self._diagnostic_map_matching.get(
                "section_id", ""
            ),
            "map_match_raw_lane_id": self._diagnostic_map_matching.get(
                "raw_lane_id", ""
            ),
            "map_match_center_x_m": self._diagnostic_map_matching.get(
                "center_x_m", ""
            ),
            "map_match_center_y_m": self._diagnostic_map_matching.get(
                "center_y_m", ""
            ),
            "map_match_lane_width_m": self._diagnostic_map_matching.get(
                "lane_width_m", ""
            ),
            "map_match_lateral_offset_m": self._diagnostic_map_matching.get(
                "lateral_offset_m", ""
            ),
            "map_match_heading_error_rad": self._diagnostic_map_matching.get(
                "heading_error_rad", ""
            ),
            "map_match_score": self._diagnostic_map_matching.get("score", ""),
            "map_match_confidence": self._diagnostic_map_matching.get(
                "confidence", ""
            ),
            "map_match_reason": self._diagnostic_map_matching.get(
                "match_reason", ""
            ),
            "map_match_candidate_count": self._diagnostic_map_matching.get(
                "candidate_count", ""
            ),
            "local_lane_frame_cache_reused": bool(
                self._diagnostic_local_lane_frame.get("cache_reused", False)
            ),
            "local_lane_frame_generation_reason": self._diagnostic_local_lane_frame.get(
                "generation_reason", ""
            ),
            "local_lane_frame_ego_ad_lane_id": self._diagnostic_local_lane_frame.get(
                "ego_ad_lane_id", ""
            ),
            "local_lane_frame_forward_distance_m": self._diagnostic_local_lane_frame.get(
                "forward_distance_m", ""
            ),
            "local_lane_frame_backward_distance_m": self._diagnostic_local_lane_frame.get(
                "backward_distance_m", ""
            ),
            "local_lane_frame_corridors": json.dumps(
                self._diagnostic_local_lane_frame.get("corridors", {}),
                sort_keys=True,
            ),
            "local_lane_frame_lane_to_offset": json.dumps(
                self._diagnostic_local_lane_frame.get("lane_to_offset", {}),
                sort_keys=True,
            ),
            "local_lane_frame_route_target_ad_lane_id": self._diagnostic_local_lane_frame.get(
                "route_target_ad_lane_id", ""
            ),
            "local_lane_frame_target_in_frame": self._diagnostic_local_lane_frame.get(
                "route_target_in_frame", ""
            ),
            "local_lane_frame_target_offset": self._diagnostic_local_lane_frame.get(
                "route_target_offset", ""
            ),
            "local_lane_frame_invariant_violations": ";".join(
                str(value)
                for value in list(
                    self._diagnostic_local_lane_frame.get(
                        "invariant_violations", []
                    )
                    or []
                )
            ),
            "route_reference_allowed": reference_debug.get("route_reference_allowed", ""),
            "route_reference_gate_reason": reference_debug.get("route_reference_gate_reason", ""),
            "route_lane_change_allowed": reference_debug.get("route_lane_change_allowed", ""),
            "opportunistic_lane_change_allowed": reference_debug.get("opportunistic_lane_change_allowed", ""),
            "lane_change_gate_reason": reference_debug.get("lane_change_gate_reason", ""),
            "route_lane_change_required": reference_debug.get("route_lane_change_required", ""),
            "lane_change_authorized": reference_debug.get("lane_change_authorized", ""),
            "lane_change_authorization_direction": reference_debug.get("lane_change_authorization_direction", ""),
            "lane_change_authorization_reason": reference_debug.get("lane_change_authorization_reason", ""),
            "lane_change_required_by_route": reference_debug.get("lane_change_required_by_route", ""),
            "lane_change_distance_to_maneuver_m": reference_debug.get("lane_change_distance_to_maneuver_m", ""),
            "lane_change_authorized_target_lane_id": reference_debug.get("lane_change_authorized_target_lane_id", ""),
            "route_maneuver_normalized": reference_debug.get("route_maneuver_normalized", ""),
            "behavior_override_reason": reference_debug.get("behavior_override_reason", ""),
            "reference_follow_global_route_lane": reference_debug.get("reference_pipeline_follow_global_route_lane", ""),
            "route_current_road_option": reference_debug.get("route_current_road_option", ""),
            "route_next_macro_maneuver": reference_debug.get("route_next_macro_maneuver", ""),
            "candidate_evaluation_summary": reference_debug.get("candidate_evaluation_summary", ""),
            "candidate_selected_decision": reference_debug.get("candidate_selected_decision", ""),
            "candidate_selected_lane_id": reference_debug.get("candidate_selected_lane_id", ""),
            "candidate_selected_cost": reference_debug.get("candidate_selected_cost", ""),
            "candidate_pipeline_enabled": reference_debug.get("candidate_pipeline_enabled", ""),
            "candidate_pipeline_selected": reference_debug.get("candidate_pipeline_selected", ""),
            "candidate_pipeline_selected_status": reference_debug.get("candidate_pipeline_selected_status", ""),
            "candidate_pipeline_selected_reason": reference_debug.get("candidate_pipeline_selected_reason", ""),
            "candidate_selected_stop_goal_active": reference_debug.get(
                "candidate_selected_stop_goal_active",
                "",
            ),
            "candidate_pipeline_count": reference_debug.get("candidate_pipeline_count", ""),
            "candidate_prediction_trajectory_count": reference_debug.get("candidate_prediction_trajectory_count", ""),
            "candidate_pipeline_summary": reference_debug.get("candidate_pipeline_summary", ""),
            "candidate_mpc_probe_summary": reference_debug.get("candidate_mpc_probe_summary", ""),
            "candidate_selected_trajectory_variant": reference_debug.get("lane_change_trajectory_variant", ""),
            "candidate_selected_lane_change_duration_s": reference_debug.get("lane_change_duration_s", ""),
            "candidate_selected_lane_change_duration_comfort_reason": reference_debug.get(
                "lane_change_duration_comfort_reason", ""
            ),
            "candidate_selected_lane_change_planning_average_speed_mps": reference_debug.get(
                "lane_change_planning_average_speed_mps", ""
            ),
            "candidate_selected_lane_change_authorization_source": reference_debug.get(
                "lane_change_authorization_source", ""
            ),
            "candidate_selected_lane_change_initial_progress": reference_debug.get(
                "lane_change_initial_progress", ""
            ),
            "candidate_selected_lane_change_terminal_progress": reference_debug.get(
                "lane_change_terminal_progress", ""
            ),
            "route_tracking_lane_change_locked": reference_debug.get(
                "route_tracking_lane_change_locked", ""
            ),
            "route_tracking_lane_change_progress_index": reference_debug.get(
                "route_tracking_lane_change_progress_index", ""
            ),
            "route_tracking_lane_change_source_lane_id": reference_debug.get(
                "route_tracking_lane_change_source_lane_id", ""
            ),
            "route_tracking_lane_change_target_lane_id": reference_debug.get(
                "route_tracking_lane_change_target_lane_id", ""
            ),
            "lane_change_commitment_release_reason": reference_debug.get(
                "lane_change_commitment_release_reason", ""
            ),
            "lane_change_completion_reason": reference_debug.get(
                "lane_change_completion_reason", ""
            ),
            "lane_change_completion_stable_frames": reference_debug.get(
                "lane_change_completion_stable_frames", ""
            ),
            "lane_change_completion_lateral_error_m": reference_debug.get(
                "lane_change_completion_lateral_error_m", ""
            ),
            "lane_change_completion_heading_error_deg": reference_debug.get(
                "lane_change_completion_heading_error_deg", ""
            ),
            "lane_change_stabilization_entry_lateral_error_m": reference_debug.get(
                "lane_change_stabilization_entry_lateral_error_m", ""
            ),
            "lane_change_stabilization_geometry_ready": reference_debug.get(
                "lane_change_stabilization_geometry_ready", ""
            ),
            "behavior_lane_lateral_error_m": reference_debug.get(
                "behavior_lane_lateral_error_m", ""
            ),
            "behavior_lane_heading_error_deg": reference_debug.get(
                "behavior_lane_heading_error_deg", ""
            ),
            "behavior_lane_alignment_valid": reference_debug.get(
                "behavior_lane_alignment_valid", ""
            ),
            "behavior_lane_change_completion_allowed": reference_debug.get(
                "behavior_lane_change_completion_allowed", ""
            ),
            "lane_change_completion_target_lane_matches": reference_debug.get(
                "lane_change_completion_target_lane_matches", ""
            ),
            "lane_change_completion_footprint_clearance_m": reference_debug.get(
                "lane_change_completion_footprint_clearance_m", ""
            ),
            "lane_change_phase": reference_debug.get("lane_change_phase", ""),
            "lane_change_stabilization_frames": reference_debug.get(
                "lane_change_stabilization_frames", ""
            ),
            "maneuver_commitment_state": reference_debug.get(
                "maneuver_commitment_state", ""
            ),
            "maneuver_commitment_decision": reference_debug.get(
                "maneuver_commitment_decision", ""
            ),
            "maneuver_commitment_source_lane_id": reference_debug.get(
                "maneuver_commitment_source_lane_id", ""
            ),
            "maneuver_commitment_target_lane_id": reference_debug.get(
                "maneuver_commitment_target_lane_id", ""
            ),
            "maneuver_commitment_progress": reference_debug.get(
                "maneuver_commitment_progress", ""
            ),
            "maneuver_commitment_reference_locked": reference_debug.get(
                "maneuver_commitment_reference_locked", ""
            ),
            "maneuver_commitment_active": reference_debug.get(
                "maneuver_commitment_active", ""
            ),
            "maneuver_commitment_committed_at_s": float(
                self._route_tracking_lane_change_committed_at_s
            ),
            "maneuver_geometry_active": reference_debug.get(
                "maneuver_geometry_active", ""
            ),
            "maneuver_geometry_id": reference_debug.get(
                "maneuver_geometry_id", ""
            ),
            "maneuver_geometry_type": reference_debug.get(
                "maneuver_geometry_type", ""
            ),
            "maneuver_geometry_direction": reference_debug.get(
                "maneuver_geometry_direction", ""
            ),
            "maneuver_geometry_phase": reference_debug.get(
                "maneuver_geometry_phase", ""
            ),
            "maneuver_geometry_revision": reference_debug.get(
                "maneuver_geometry_revision", ""
            ),
            "maneuver_geometry_source_changed": reference_debug.get(
                "maneuver_geometry_source_changed", ""
            ),
            "maneuver_geometry_owner": reference_debug.get(
                "maneuver_geometry_owner", ""
            ),
            "maneuver_geometry_point_count": reference_debug.get(
                "maneuver_geometry_point_count", ""
            ),
            "maneuver_first_point_jump_m": reference_debug.get(
                "maneuver_first_point_jump_m", ""
            ),
            "maneuver_first_heading_jump_deg": reference_debug.get(
                "maneuver_first_heading_jump_deg", ""
            ),
            "maneuver_geometry_release_reason": reference_debug.get(
                "maneuver_geometry_release_reason", ""
            ),
            "route_tracking_recovery_active": reference_debug.get(
                "route_tracking_recovery_active", ""
            ),
            "route_tracking_recovery_reason": reference_debug.get(
                "route_tracking_recovery_reason", ""
            ),
            "mpc_feedback_summary": reference_debug.get("mpc_feedback_summary", ""),
            "mpc_feedback_record_reason": str(mpc_feedback_record_reason),
            "mpc_feedback_blocked_lane_ids": reference_debug.get("mpc_feedback_blocked_lane_ids", ""),
            "mode_transition_guard_reason": str(mode_transition_guard_reason),
            "control_buffer_reason": str(self.control_buffer.last_reason),
            "control_buffered_step_count": int(self.control_buffer.buffered_step_count),
            "mpc_replan_executed": bool(mpc_replan_executed),
            "route_manager_status": json.dumps(
                self.route_manager.last_status.as_dict(),
                default=str,
            ),
            "route_replan_attempted": reference_debug.get(
                "route_replan_attempted", False
            ),
            "route_replan_succeeded": reference_debug.get(
                "route_replan_succeeded", False
            ),
            "route_replan_attempt_count": int(
                self._route_replan_attempt_count
            ),
            "route_replan_reason": str(self._route_replan_last_reason),
            "route_remaining_distance_m": float(
                self.route_manager.last_status.remaining_distance_m
            ),
            "route_reached_destination": bool(self.route_manager.last_status.reached_destination),
            "global_planner_backend": str(self.global_planner_backend),
            "global_planner_backend_warning": str(self.global_planner_backend_warning),
            "tracker_active_count": reference_debug.get("tracker_active_count", ""),
            "tracker_stale_count": reference_debug.get("tracker_stale_count", ""),
            "prediction_validity_reason": reference_debug.get("prediction_validity_reason", ""),
            "scenario_fsm_state": reference_debug.get("scenario_fsm_state", ""),
            "scenario_fsm_reason": reference_debug.get("scenario_fsm_reason", ""),
            "scenario_behavior_signal_state": reference_debug.get("scenario_behavior_signal_state", ""),
            "scenario_behavior_override_decision": reference_debug.get("scenario_behavior_override_decision", ""),
            "scenario_speed_cap_mps": reference_debug.get("scenario_speed_cap_mps", ""),
            "scenario_stop_goal_active": reference_debug.get("scenario_stop_goal_active", ""),
            "scenario_turn_direction": reference_debug.get("scenario_turn_direction", ""),
            "scenario_turn_latched": reference_debug.get("scenario_turn_latched", ""),
            "scenario_boundary_recovery_active": reference_debug.get(
                "scenario_boundary_recovery_active", ""
            ),
            "scenario_boundary_clearance_m": reference_debug.get(
                "scenario_boundary_clearance_m", ""
            ),
            "scenario_boundary_lateral_offset_m": reference_debug.get(
                "scenario_boundary_lateral_offset_m", ""
            ),
            "scenario_boundary_heading_error_rad": reference_debug.get(
                "scenario_boundary_heading_error_rad", ""
            ),
            "boundary_recovery_generation_reason": reference_debug.get(
                "boundary_recovery_generation_reason", ""
            ),
            "boundary_recovery_conditioning_reason": reference_debug.get(
                "boundary_recovery_conditioning_reason", ""
            ),
            "speed_plan_target_mps": reference_debug.get("speed_plan_target_mps", ""),
            "speed_plan_cap_mps": reference_debug.get("speed_plan_cap_mps", ""),
            "speed_plan_stop_goal_active": reference_debug.get("speed_plan_stop_goal_active", ""),
            "speed_plan_reason": reference_debug.get("speed_plan_reason", ""),
            "speed_plan_front_gap_m": reference_debug.get(
                "speed_plan_front_gap_m", ""
            ),
            "speed_plan_desired_follow_gap_m": reference_debug.get(
                "speed_plan_desired_follow_gap_m", ""
            ),
            "speed_plan_continuous_following_active": reference_debug.get(
                "speed_plan_continuous_following_active", ""
            ),
            "speed_plan_idm_acceleration_mps2": reference_debug.get(
                "speed_plan_idm_acceleration_mps2", ""
            ),
            "front_gap_actor_id": reference_debug.get("front_gap_actor_id", ""),
            "front_gap_obstacle_speed_mps": reference_debug.get(
                "front_gap_obstacle_speed_mps", ""
            ),
            "snapshot_repr_diag": reference_debug.get("snapshot_repr_diag", ""),
            "speed_owner_requested_mps": reference_debug.get(
                "speed_owner_requested_mps", ""
            ),
            "speed_owner_scenario_cap_mps": reference_debug.get(
                "speed_owner_scenario_cap_mps", ""
            ),
            "speed_owner_turn_cap_mps": reference_debug.get(
                "speed_owner_turn_cap_mps", ""
            ),
            "speed_owner_lane_change_cap_mps": reference_debug.get(
                "speed_owner_lane_change_cap_mps", ""
            ),
            "speed_owner_following_cap_mps": reference_debug.get(
                "speed_owner_following_cap_mps", ""
            ),
            "speed_owner_turn_approach_cap_mps": reference_debug.get(
                "speed_owner_turn_approach_cap_mps", ""
            ),
            "speed_owner_upcoming_turn_distance_m": reference_debug.get(
                "speed_owner_upcoming_turn_distance_m", ""
            ),
            "speed_owner_selected_target_mps": reference_debug.get(
                "speed_owner_selected_target_mps", ""
            ),
            "speed_owner_limiting_owner": reference_debug.get(
                "speed_owner_limiting_owner", ""
            ),
            "speed_owner_active_constraints": reference_debug.get(
                "speed_owner_active_constraints", ""
            ),
            "speed_owner_mpc_entry_target_mps": float(speed_ref_mps),
            "speed_owner_post_plan_delta_mps": (
                float(speed_ref_mps)
                - float(reference_debug.get("speed_plan_target_mps") or speed_ref_mps)
            ),
            "speed_owner_target_overridden_after_plan": abs(
                float(speed_ref_mps)
                - float(reference_debug.get("speed_plan_target_mps") or speed_ref_mps)
            ) > 1.0e-6,
            "speed_owner_proposed_post_plan_target_mps": reference_debug.get(
                "speed_owner_proposed_post_plan_target_mps", ""
            ),
            "speed_owner_ceiling_applied": reference_debug.get(
                "speed_owner_ceiling_applied", ""
            ),
            "speed_owner_ceiling_reduction_mps": reference_debug.get(
                "speed_owner_ceiling_reduction_mps", ""
            ),
            "carla_turn_reference_reason": str(
                reference_debug.get("carla_turn_reference_reason", "")
            ),
            "carla_route_debug_reason": str(
                self.route_manager.carla_route_debug_reason
            ),
            "carla_route_sync_reason": str(
                self.route_manager.carla_route_sync_reason
            ),
            "carla_route_progress_index": int(
                self.route_manager.carla_route_progress_index
            ),
            "carla_upcoming_turn_direction": str(
                reference_debug.get("carla_upcoming_turn_direction", "")
            ),
            "carla_upcoming_turn_distance_m": reference_debug.get(
                "carla_upcoming_turn_distance_m", ""
            ),
            "carla_upcoming_turn_reason": str(
                reference_debug.get("carla_upcoming_turn_reason", "")
            ),
            "reference_lateral_guard_reason": str(reference_debug.get("reference_lateral_guard_reason", "")),
            "mpc_reference_stabilizer_reason": str(reference_debug.get("mpc_reference_stabilizer_reason", "")),
            "final_reference_gate_valid": reference_debug.get(
                "final_reference_gate_valid", ""
            ),
            "final_reference_gate_mode": reference_debug.get(
                "final_reference_gate_mode", ""
            ),
            "final_reference_gate_reason": reference_debug.get(
                "final_reference_gate_reason", ""
            ),
            "reference_max_curvature_1pm": reference_debug.get(
                "reference_max_curvature_1pm", ""
            ),
            "reference_contract_max_curvature_1pm": reference_debug.get(
                "reference_contract_max_curvature_1pm", ""
            ),
            "reference_curvature_margin_1pm": reference_debug.get(
                "reference_curvature_margin_1pm", ""
            ),
            "reference_pipeline_conditioning_reason": reference_debug.get(
                "reference_pipeline_conditioning_reason", ""
            ),
            "reference_pipeline_mode": reference_debug.get(
                "reference_pipeline_mode", ""
            ),
            "mpc_entry_allowed": reference_debug.get("mpc_entry_allowed", ""),
            "mpc_entry_status": reference_debug.get("mpc_entry_status", ""),
            "mpc_entry_reason": reference_debug.get("mpc_entry_reason", ""),
            "pipeline_error": str(reference_debug.get("pipeline_error", behavior_debug.get("pipeline_error", ""))),
            "stop_target_forward_m": stop_target_forward_m_debug,
            "mpc_trajectory_points": self._last_mpc_trajectory_points(),
            "global_route_points": self._active_global_route_points(),
            "lane_reference_points": [
                [
                    float(sample.get("x_ref_m", sample.get("x", 0.0))),
                    float(sample.get("y_ref_m", sample.get("y", 0.0))),
                ]
                for sample in list(lane_center_reference or [])
            ],
            "target_speed_mps": float(speed_ref_mps),
            "mpc_status": str(mpc_status),
            "mpc_feasibility_checked": bool(mpc_replan_executed),
            "mpc_feasibility_status": str(mpc_status),
            "mpc_feasibility_reason": str(fallback_reason),
            "mpc_solve_time_ms": float(getattr(self.mpc, "_last_solve_time_ms", 0.0)),
            "mpc_cost_profile": str(self.active_mpc_cost_profile),
            "requested_mpc_cost_profile": str(self.requested_mpc_cost_profile),
            "mpc_cost_profile_switch_reason": str(self.mpc_cost_profile_switch_reason),
            "reference_source": str(reference_debug.get(
                "reference_source",
                "map_lane_center" if lane_center_reference else "straight_fallback",
            )),
            "final_reference_geometry_source": str(
                reference_debug.get(
                    "final_reference_geometry_source",
                    reference_debug.get("reference_source", "unknown"),
                )
            ),
            "fallback_reason": fallback_reason,
            "mpc_fallback_reason": fallback_reason,
            "control_guard_reason": str(control_guard_reason),
            "accel_cmd_mps2": float(accel_mps2),
            "steer_cmd_rad": float(steer_rad),
            "pre_supervisor_accel_cmd_mps2": float(pre_supervisor_accel_mps2),
            "pre_supervisor_steer_cmd_rad": float(pre_supervisor_steer_rad),
            "post_supervisor_accel_cmd_mps2": float(post_supervisor_accel_mps2),
            "post_supervisor_steer_cmd_rad": float(post_supervisor_steer_rad),
            "applied_throttle": float(getattr(control, "throttle", 0.0)),
            "applied_brake": float(getattr(control, "brake", 0.0)),
            "applied_steer": float(getattr(control, "steer", 0.0)),
            "platform_applied_steer_rad": float(
                getattr(control, "steer", 0.0)
            ) * float(self.mpc.constraints.max_steer_rad),
            "planner_requested": True,
            "planner_executed": True,
            "fallback_active": bool(fallback_reason),
            "fallback_policy": str(self.fallback_policy),
            "fallback_policy_warning": str(self.fallback_policy_warning),
            "safety_supervisor_reason": str(safety_supervisor_reason),
            "turn_boundary_recovery_active": bool(
                self.safety_supervisor.turn_boundary_recovery_active
            ),
            "turn_boundary_recovery_phase": str(
                self.safety_supervisor.turn_boundary_recovery_phase
            ),
        }
        diagnostics.update(self.architecture_profile.as_debug_fields())
        diagnostics.update(
            self._update_evaluation_metrics(
                ego_location=ego_location,
                ego_speed_mps=float(ego_speed_mps),
                ego_yaw_rad=float(ego_yaw_rad),
                object_snapshots=object_snapshots,
                behavior_decision=str(behavior_debug.get("decision", "")),
                behavior_fsm_state=str(behavior_debug.get("lc_state", "")),
                mpc_replan_executed=bool(mpc_replan_executed),
                cp_summary=cp_summary,
                reference_samples=lane_center_reference,
                boundary_snapshot=boundary_snapshot,
            )
        )
        decision_record = self._build_decision_record(
            scenario_state=diagnostics.get("scenario_fsm_state", ""),
            behavior_decision=diagnostics.get("behavior_decision", ""),
            behavior_fsm_state=diagnostics.get("behavior_fsm_state", ""),
            candidate_selected_name=diagnostics.get("candidate_pipeline_selected", ""),
            candidate_selected_decision=diagnostics.get("candidate_selected_decision", ""),
            candidate_selected_status=diagnostics.get("candidate_pipeline_selected_status", ""),
            candidate_selected_reason=diagnostics.get("candidate_pipeline_selected_reason", ""),
            candidate_pipeline_summary=diagnostics.get("candidate_pipeline_summary", ""),
            candidate_mpc_probe_summary=diagnostics.get("candidate_mpc_probe_summary", ""),
            reference_source=diagnostics.get("reference_source", ""),
            reference_stage=diagnostics.get("reference_pipeline_stage", ""),
            reference_fallback_reason=diagnostics.get("reference_pipeline_fallback", ""),
            reference_lateral_guard_reason=diagnostics.get("reference_lateral_guard_reason", ""),
            reference_stabilizer_reason=diagnostics.get("mpc_reference_stabilizer_reason", ""),
            final_reference_gate_reason=diagnostics.get("final_reference_gate_reason", ""),
            lane_change_authorized=diagnostics.get("lane_change_authorized", ""),
            lane_change_gate_reason=diagnostics.get("lane_change_gate_reason", ""),
            route_lane_change_required=diagnostics.get("route_lane_change_required", ""),
            behavior_override_reason=diagnostics.get("behavior_override_reason", ""),
            mode_transition_guard_reason=diagnostics.get("mode_transition_guard_reason", ""),
            mpc_status=diagnostics.get("mpc_status", ""),
            mpc_fallback_reason=diagnostics.get("mpc_fallback_reason", ""),
            control_guard_reason=diagnostics.get("control_guard_reason", ""),
            control_buffer_reason=diagnostics.get("control_buffer_reason", ""),
            safety_supervisor_reason=diagnostics.get("safety_supervisor_reason", ""),
            applied_throttle=diagnostics.get("applied_throttle", 0.0),
            applied_brake=diagnostics.get("applied_brake", 0.0),
            applied_steer=diagnostics.get("applied_steer", 0.0),
        )
        diagnostics.update(decision_record.as_debug_fields())
        self._draw_world_debug_primitives(
            destination_state=destination_state,
            lane_center_reference=lane_center_reference,
        )
        return PlannerOutput(
            control=control,
            behavior_command=BehaviorCommand.from_debug(
                behavior_debug=behavior_debug,
                target_speed_mps=float(speed_ref_mps),
            ),
            reference_trajectory=[dict(sample) for sample in list(lane_center_reference or [])],
            planned_trajectory=self._last_mpc_trajectory_points(),
            predictions=dict(reference_debug.get("prediction_trajectories", {}) or {}),
            acceleration_mps2=float(post_supervisor_accel_mps2),
            steering_rad=float(post_supervisor_steer_rad),
            diagnostics=PlannerDiagnostics(diagnostics),
        )

    def _full_latched_stop_target_for_signal(
        self,
        *,
        traffic_state: str,
        stop_target: Mapping[str, object] | None,
        ego_location: PlannerLocation,
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
            distance_m = max(
                2.0,
                float(self.config.get("full_latched_virtual_stop_distance_m", 12.0)),
            )
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

    def _resolve_full_traffic_state_from_carla_actor(
        self,
        *,
        raw_state: str,
        signal_context: Mapping[str, object] | None,
    ) -> tuple[str, str]:
        """Use the traffic-light state received through ROS perception."""

        del signal_context
        return str(raw_state or "unknown").strip().lower(), "ros_perception_signal_state"

    def _record_lane_id_discontinuity(
        self,
        *,
        previous_waypoint: Any,
        previous_lane_id: int,
        new_waypoint: Any,
    ) -> None:
        """Log when `_lane_id_tracker` lost continuity and re-anchored.

        Also flips `_lane_id_discontinuity_since_lock`, which
        `_release_completed_lane_change_commitment` reads to stop requiring
        exact `current_lane_id == target_lane_id` equality once the id space
        is known to have shifted since the commitment was locked (this can
        otherwise strand a vehicle that has genuinely reached the target
        lane in EXECUTE_LANE_CHANGE indefinitely). Everything else here is a
        read-only diagnostic. Exists to measure, on real routes, how often
        `StableLaneIdTracker` hits a boundary `lane_hop_offset` cannot
        prove adjacency across (see its docstring -- typically a road_id
        change, since it only walks get_left_lane()/get_right_lane(), never
        next()/previous()) before deciding whether the fuller AD-map-backed
        identity migration is actually needed.
        """
        self._lane_id_discontinuity_since_lock = True
        if not bool(self.config.get("record_debug", True)):
            return
        try:
            from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint

            new_lane_id = int(canonical_lane_id_for_waypoint(new_waypoint))
        except Exception:
            new_lane_id = 0
        event = {
            "sim_time_s": float(self._sim_time_s()),
            "previous_road_id": int(getattr(previous_waypoint, "road_id", 0) or 0),
            "previous_section_id": int(getattr(previous_waypoint, "section_id", 0) or 0),
            "previous_raw_lane_id": int(getattr(previous_waypoint, "lane_id", 0) or 0),
            "previous_canonical_lane_id": int(previous_lane_id),
            "new_road_id": int(getattr(new_waypoint, "road_id", 0) or 0),
            "new_section_id": int(getattr(new_waypoint, "section_id", 0) or 0),
            "new_raw_lane_id": int(getattr(new_waypoint, "lane_id", 0) or 0),
            "new_canonical_lane_id": new_lane_id,
        }
        try:
            location = getattr(getattr(new_waypoint, "transform", None), "location", None)
            if location is not None:
                event["x_m"] = float(location.x)
                event["y_m"] = float(location.y)
        except Exception:
            pass
        if self.debug:
            print(f"[CP-X OpenCDA Bridge] lane_id_tracker discontinuity: {event}")
        try:
            debug_dir = Path(
                self.config.get(
                    "debug_output_dir",
                    Path(__file__).resolve().parent / "debug",
                )
            )
            debug_dir.mkdir(parents=True, exist_ok=True)
            if self._lane_id_discontinuity_log_file is None:
                self._lane_id_discontinuity_log_file = open(
                    debug_dir / "lane_id_discontinuities.jsonl",
                    "w",
                    encoding="utf-8",
                )
            self._lane_id_discontinuity_log_file.write(json.dumps(event) + "\n")
            self._lane_id_discontinuity_log_file.flush()
        except Exception:
            pass

    def _record_debug(self, payload: Mapping[str, Any]) -> None:
        if not bool(self.config.get("record_debug", True)):
            return
        try:
            debug_dir = Path(
                self.config.get(
                    "debug_output_dir",
                    Path(__file__).resolve().parent / "debug",
                )
            )
            debug_dir.mkdir(parents=True, exist_ok=True)
            if self._debug_writer is None:
                self._debug_csv_file = open(
                    debug_dir / "opencda_planner_debug.csv",
                    "w",
                    newline="",
                    encoding="utf-8",
                )
                self._debug_writer = csv.DictWriter(
                    self._debug_csv_file,
                    fieldnames=self._debug_fieldnames,
                    extrasaction="ignore",
                )
                self._debug_writer.writeheader()
                self._debug_jsonl_file = open(
                    debug_dir / "opencda_planner_debug.jsonl",
                    "w",
                    encoding="utf-8",
                )
            row = {name: payload.get(name, "") for name in self._debug_fieldnames}
            self._debug_writer.writerow(row)
            self._debug_csv_file.flush()
            if self._debug_jsonl_file is not None:
                self._debug_jsonl_file.write(json.dumps(dict(payload), default=str) + "\n")
                self._debug_jsonl_file.flush()
        except Exception as exc:
            if self.debug:
                print(f"[CP-X OpenCDA Bridge] debug record failed: {exc}")

    def _spawn_metrics_collision_sensor(self) -> None:
        """Collision events are supplied by the simulator boundary, not created in ROS."""

        self._metrics_collision_sensor_error = "simulation_boundary_unavailable_in_ros"

    def _on_metrics_collision(self, event: Any) -> None:
        """Keep the source callback name; ROS has no simulator collision event here."""

        del event

    def _update_boundary_recovery_request(
        self,
        *,
        boundary_snapshot: Mapping[str, object],
        behavior_decision: str,
        sim_time_s: float,
        recovery_planned: bool = False,
        recovery_reference_feasible: bool = True,
    ) -> None:
        from cpx_planning.pipeline.scenario_manager import (
            BoundaryRecoveryRequest,
        )

        if float(sim_time_s) < float(
            getattr(
                self,
                "_boundary_recovery_cooldown_until_s",
                -float("inf"),
            )
        ):
            self._reset_boundary_recovery_request()
            return

        geometry_source = str(
            boundary_snapshot.get(
                "road_boundary_geometry_source",
                "",
            )
        )
        drivable_inside = boundary_snapshot.get(
            "road_boundary_drivable_inside",
            "",
        )
        if (
            geometry_source.startswith("drivable_footprint:")
            and drivable_inside in {True, "True", "true", "1", 1}
        ):
            # Consuming the soft boundary margin may request lower speed, but
            # it must not latch hard recovery while the complete footprint is
            # still on CARLA's driving-lane union.
            self._boundary_recovery_infeasible_frames = 0
            self._reset_boundary_recovery_request()
            return

        if bool(recovery_planned) and not bool(recovery_reference_feasible):
            self._boundary_recovery_infeasible_frames = (
                int(
                    getattr(
                        self,
                        "_boundary_recovery_infeasible_frames",
                        0,
                    )
                )
                + 1
            )
            max_failures = max(
                1,
                int(
                    self.config.get(
                        "boundary_recovery_max_infeasible_frames",
                        3,
                    )
                ),
            )
            if int(self._boundary_recovery_infeasible_frames) >= int(
                max_failures
            ):
                self._boundary_recovery_cooldown_until_s = (
                    float(sim_time_s)
                    + max(
                        0.1,
                        float(
                            self.config.get(
                                "boundary_recovery_cooldown_s",
                                2.0,
                            )
                        ),
                    )
                )
                self._reset_boundary_recovery_request()
            return
        self._boundary_recovery_infeasible_frames = 0

        try:
            valid = bool(
                boundary_snapshot.get(
                    "road_boundary_sample_valid",
                    False,
                )
            )
            clearance_m = float(
                boundary_snapshot.get("road_boundary_clearance_m", "")
            )
            lateral_offset_m = float(
                boundary_snapshot.get(
                    "road_boundary_lateral_offset_m",
                    "",
                )
            )
            heading_error_rad = float(
                boundary_snapshot.get(
                    "road_boundary_heading_error_rad",
                    "",
                )
            )
        except (TypeError, ValueError):
            valid = False
            clearance_m = float("inf")
            lateral_offset_m = 0.0
            heading_error_rad = 0.0
        if not bool(valid) or not all(
            math.isfinite(value)
            for value in (
                float(clearance_m),
                float(lateral_offset_m),
                float(heading_error_rad),
            )
        ):
            self._reset_boundary_recovery_request()
            return

        trigger_clearance_m = float(
            self.config.get(
                "boundary_recovery_trigger_clearance_m",
                -0.10,
            )
        )
        release_clearance_m = max(
            float(trigger_clearance_m),
            float(
                self.config.get(
                    "boundary_recovery_release_clearance_m",
                    0.10,
                )
            ),
        )
        if float(clearance_m) <= float(trigger_clearance_m):
            self._boundary_recovery_trigger_frames = (
                int(self._boundary_recovery_trigger_frames) + 1
            )
        else:
            self._boundary_recovery_trigger_frames = 0
        required_frames = max(
            1,
            int(
                self.config.get(
                    "boundary_recovery_trigger_frames",
                    3,
                )
            ),
        )
        previous_active = bool(
            getattr(
                getattr(self, "_boundary_recovery_request", None),
                "active",
                False,
            )
        )
        active = bool(
            (
                bool(previous_active)
                and float(clearance_m) < float(release_clearance_m)
            )
            or int(self._boundary_recovery_trigger_frames)
            >= int(required_frames)
        )
        decision = str(behavior_decision or "").strip().lower()
        turn_direction = (
            "left"
            if decision.endswith("_left")
            else "right"
            if decision.endswith("_right")
            else ""
        )
        self._boundary_recovery_request = BoundaryRecoveryRequest(
            valid=True,
            active=bool(active),
            clearance_m=float(clearance_m),
            lateral_offset_m=float(lateral_offset_m),
            heading_error_rad=float(heading_error_rad),
            turn_direction=str(turn_direction),
            timestamp_s=float(sim_time_s),
            reason=(
                "boundary_recovery_latched"
                if bool(active)
                else "boundary_recovery_monitor"
            ),
        )

    def _reset_boundary_recovery_request(self) -> None:
        from cpx_planning.pipeline.scenario_manager import (
            BoundaryRecoveryRequest,
        )

        self._boundary_recovery_trigger_frames = 0
        self._boundary_recovery_request = BoundaryRecoveryRequest()

    def _road_boundary_metrics(
        self,
        ego_location: Any,
        *,
        record_sample: bool = True,
        ego_yaw_rad: float | None = None,
        reference_samples: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, object]:
        """Measure ego with the same route-corridor footprint contract."""

        result: dict[str, object] = {
            "road_boundary_sample_valid": False,
            "road_boundary_lateral_offset_m": "",
            "road_boundary_lane_width_m": "",
            "road_boundary_ego_half_width_m": "",
            "road_boundary_clearance_m": "",
            "road_boundary_breach": "",
            "road_boundary_heading_error_rad": "",
            "road_boundary_projection_segment_index": "",
            "road_boundary_projection_segment_ratio": "",
            "road_boundary_projection_raw_heading_rad": "",
            "road_boundary_projection_conditioned_heading_rad": "",
            "road_boundary_projection_continuity_limited": "",
            "road_boundary_projection_reason": "",
            "road_boundary_geometry_source": "",
            "road_boundary_drivable_inside": "",
        }
        try:
            ego_half_width_m = float(self.config.get("metrics_ego_half_width_m", self.config.get("reference_vehicle_half_width_m", 1.0)))
            ego_half_length_m = float(self.config.get("reference_vehicle_half_length_m", 2.4))
            if ego_yaw_rad is None:
                return result
            projection = None
            if len(list(reference_samples or [])) >= 2:
                projection = self.reference_generator.project_reference_corridor(
                    reference_samples=reference_samples,
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    ego_half_width_m=float(ego_half_width_m),
                    ego_half_length_m=float(ego_half_length_m),
                    safety_margin_m=float(
                        self.config.get(
                            "reference_contract_turn_boundary_margin_m",
                            0.15,
                        )
                    ),
                    max_heading_step_rad=float(
                        self.config.get(
                            "road_boundary_projection_max_heading_step_rad",
                            0.04,
                        )
                    ),
                    continuity_reset_distance_m=float(
                        self.config.get(
                            "road_boundary_projection_reset_distance_m",
                            2.5,
                        )
                    ),
                    max_position_step_m=float(
                        self.config.get(
                            "road_boundary_projection_max_position_step_m",
                            0.5,
                        )
                    ),
                )
                occupancy = projection.occupancy
            else:
                occupancy = self.reference_generator.lane_corridor_occupancy(
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    ego_half_width_m=float(ego_half_width_m),
                    ego_half_length_m=float(ego_half_length_m),
                    safety_margin_m=float(
                        self.config.get(
                            "reference_contract_turn_boundary_margin_m",
                            0.15,
                        )
                    ),
                )
            if not bool(occupancy.valid):
                return result
            lateral_offset_m = float(occupancy.lateral_offset_m)
            lane_width_m = float(occupancy.lane_width_m)
            clearance_m = float(occupancy.footprint_clearance_m)
            geometry_source = "route_tangent_strip"
            drivable_inside: object = ""
            if bool(
                self.config.get(
                    "road_boundary_carla_drivable_footprint_enabled",
                    True,
                )
            ):
                drivable_occupancy = (
                    self.reference_generator.drivable_footprint_occupancy(
                        x_m=float(ego_location.x),
                        y_m=float(ego_location.y),
                        z_m=float(getattr(ego_location, "z", 0.0)),
                        heading_rad=float(ego_yaw_rad),
                        ego_half_width_m=float(ego_half_width_m),
                        ego_half_length_m=float(ego_half_length_m),
                        safety_margin_m=float(
                            self.config.get(
                                "reference_contract_turn_boundary_margin_m",
                                0.15,
                            )
                        ),
                    )
                )
                if bool(drivable_occupancy.valid):
                    clearance_m = float(
                        drivable_occupancy.min_clearance_m
                    )
                    geometry_source = str(
                        drivable_occupancy.reason
                    )
                    drivable_inside = bool(
                        drivable_occupancy.inside
                    )
            breach = (
                not bool(drivable_inside)
                if drivable_inside != ""
                else bool(clearance_m < 0.0)
            )
            if bool(record_sample):
                self._metrics_boundary_sample_count += 1
                if breach:
                    self._metrics_boundary_breach_count += 1
            result.update(
                {
                    "road_boundary_sample_valid": True,
                    "road_boundary_lateral_offset_m": float(lateral_offset_m),
                    "road_boundary_lane_width_m": float(lane_width_m),
                    "road_boundary_ego_half_width_m": float(ego_half_width_m),
                    "road_boundary_clearance_m": float(clearance_m),
                    "road_boundary_breach": bool(breach),
                    "road_boundary_heading_error_rad": float(
                        occupancy.heading_error_rad
                    ),
                    "road_boundary_projection_segment_index": (
                        int(projection.segment_index)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_segment_ratio": (
                        float(projection.segment_ratio)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_raw_heading_rad": (
                        float(projection.raw_heading_rad)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_conditioned_heading_rad": (
                        float(projection.conditioned_heading_rad)
                        if projection is not None
                        else ""
                    ),
                    "road_boundary_projection_continuity_limited": (
                        bool(projection.continuity_limited)
                        if projection is not None
                        else False
                    ),
                    "road_boundary_projection_reason": (
                        str(projection.reason)
                        if projection is not None
                        else "lane_corridor_occupancy:map_fallback"
                    ),
                    "road_boundary_geometry_source": str(
                        geometry_source
                    ),
                    "road_boundary_drivable_inside": (
                        drivable_inside
                    ),
                }
            )
        except Exception:
            pass
        return result

    def _update_evaluation_metrics(
        self,
        *,
        ego_location: Any,
        ego_speed_mps: float,
        ego_yaw_rad: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        behavior_decision: str,
        behavior_fsm_state: str,
        mpc_replan_executed: bool,
        cp_summary: Mapping[str, Any],
        reference_samples: Sequence[Mapping[str, Any]] = (),
        boundary_snapshot: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Update run metrics and return fields for the unified debug row."""

        if not bool(self.config.get("record_evaluation_metrics", True)):
            return {
                "evaluation_metrics_available": False,
                "collision_sensor_available": bool(
                    self._metrics_collision_sensor_available
                ),
            }
        sim_time_s = float(self._sim_time_s())
        self.evaluation_metrics.update(
            ego_state={
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "v": float(ego_speed_mps),
                "psi": float(ego_yaw_rad),
            },
            obstacle_snapshots=list(object_snapshots or []),
            sim_time_s=sim_time_s,
            behavior_decision=str(behavior_decision),
            fsm_state=str(behavior_fsm_state),
            collision_count=int(self.evaluation_metrics.collision_count),
            last_collision_actor_type=str(
                self._metrics_last_collision_actor_type
            ),
            cp_provider_source=str(cp_summary.get("provider_source", "")),
            native_opencda_available=bool(
                cp_summary.get("native_opencda_available", False)
            ),
            native_opencda_required=bool(
                cp_summary.get("native_opencda_required", False)
            ),
            cp_obstacle_count=int(cp_summary.get("obstacle_count", 0) or 0),
        )
        if bool(mpc_replan_executed):
            self.evaluation_metrics.record_mpc_status(self.mpc.get_runtime_status())
            self.evaluation_metrics.record_mpc_extras(
                lateral_offset_m=None,
                heading_error_rad=None,
                cost_terms=self.mpc.get_last_cost_terms(),
            )
        sample = dict(self.evaluation_metrics.samples[-1])
        boundary = (
            dict(boundary_snapshot)
            if boundary_snapshot is not None
            else self._road_boundary_metrics(
                ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                reference_samples=reference_samples,
            )
        )
        self.evaluation_metrics.record_road_boundary(
            sample_valid=bool(boundary["road_boundary_sample_valid"]),
            lateral_offset_m=(
                boundary["road_boundary_lateral_offset_m"]
                if boundary["road_boundary_lateral_offset_m"] != ""
                else None
            ),
            lane_width_m=(
                boundary["road_boundary_lane_width_m"]
                if boundary["road_boundary_lane_width_m"] != ""
                else None
            ),
            ego_half_width_m=(
                boundary["road_boundary_ego_half_width_m"]
                if boundary["road_boundary_ego_half_width_m"] != ""
                else None
            ),
            clearance_m=(
                boundary["road_boundary_clearance_m"]
                if boundary["road_boundary_clearance_m"] != ""
                else None
            ),
            breach=bool(boundary["road_boundary_breach"]),
        )
        summary = self.evaluation_metrics.summary()
        boundary_sample_count = int(self._metrics_boundary_sample_count)
        collision_count = int(self.evaluation_metrics.collision_count)
        collision_this_frame = (
            collision_count > int(self._metrics_last_emitted_collision_count)
        )
        self._metrics_last_emitted_collision_count = collision_count
        cost_terms = dict(self.mpc.get_last_cost_terms())
        return {
            "evaluation_metrics_available": True,
            "collision_sensor_available": bool(
                self._metrics_collision_sensor_available
            ),
            "collision_sensor_error": str(self._metrics_collision_sensor_error),
            "collision_event_this_frame": bool(collision_this_frame),
            "collision_count": collision_count,
            "collision_rate_per_km": summary.get("collision_rate_per_km", ""),
            "last_collision_actor_type": str(
                self._metrics_last_collision_actor_type
            ),
            "last_collision_impulse": self._metrics_last_collision_impulse,
            "nearest_ttc_s": sample.get("nearest_ttc_s", ""),
            "min_ttc_s": summary.get("min_ttc_s", ""),
            "nearest_ttc_obstacle_id": sample.get(
                "nearest_ttc_obstacle_id", ""
            ),
            "nearest_ttc_reason": sample.get("nearest_ttc_reason", ""),
            "nearest_ttc_longitudinal_gap_m": sample.get(
                "nearest_ttc_longitudinal_gap_m", ""
            ),
            "nearest_ttc_lateral_gap_m": sample.get(
                "nearest_ttc_lateral_gap_m", ""
            ),
            "nearest_ttc_bumper_gap_m": sample.get(
                "nearest_ttc_bumper_gap_m", ""
            ),
            "nearest_ttc_closing_speed_mps": sample.get(
                "nearest_ttc_closing_speed_mps", ""
            ),
            "tick_max_drac_mps2": sample.get("max_drac_mps2", ""),
            "max_drac_mps2": summary.get("max_drac_mps2", ""),
            "min_pet_s": summary.get("min_pet_s", ""),
            "distance_traveled_m": summary.get("distance_traveled_m", ""),
            **boundary,
            "road_boundary_breach_count": int(
                self._metrics_boundary_breach_count
            ),
            "road_boundary_sample_count": boundary_sample_count,
            "road_boundary_breach_rate": (
                float(self._metrics_boundary_breach_count)
                / float(boundary_sample_count)
                if boundary_sample_count > 0
                else ""
            ),
            "Cost_RoadBoundary": cost_terms.get("Cost_RoadBoundary", ""),
            "Cost_Repulsive": cost_terms.get("Cost_Repulsive", ""),
            "Cost_Repulsive_Safe": cost_terms.get("Cost_Repulsive_Safe", ""),
            "Cost_Repulsive_Collision": cost_terms.get(
                "Cost_Repulsive_Collision", ""
            ),
            "Cost_Repulsive_LogBarrier": cost_terms.get(
                "Cost_Repulsive_LogBarrier", ""
            ),
            "Cost_ref": cost_terms.get("Cost_ref", ""),
            "Cost_LaneCenter": cost_terms.get("Cost_LaneCenter", ""),
            "Cost_Control": cost_terms.get("Cost_Control", ""),
            "Cost_VelocitySlack": cost_terms.get("Cost_VelocitySlack", ""),
            "prediction_lane_step_resolved_count": int(
                self._prediction_lane_step_resolved_count
            ),
            "prediction_lane_step_none_count": int(
                self._prediction_lane_step_none_count
            ),
        }

    def destroy(self) -> None:
        sensor = getattr(self, "_metrics_collision_sensor", None)
        if sensor is not None:
            try:
                sensor.stop()
            except Exception:
                pass
            try:
                sensor.destroy()
            except Exception:
                pass
            self._metrics_collision_sensor = None
        if bool(self.config.get("record_evaluation_metrics", True)):
            try:
                debug_dir = Path(
                    self.config.get(
                        "debug_output_dir",
                        Path(__file__).resolve().parent / "debug",
                    )
                )
                self._write_planning_metrics_artifacts(
                    artifact_dir=str(debug_dir),
                    recorder=self.evaluation_metrics,
                    scenario_name=str(
                        self.config.get("scenario_name", "opencda_scenario")
                    ),
                )
            except Exception as exc:
                if self.debug:
                    print(
                        "[CP-X OpenCDA Bridge] metrics artifact write failed: "
                        f"{exc}"
                    )
        for handle_name in ("_debug_csv_file", "_debug_jsonl_file"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    pass
                setattr(self, handle_name, None)
        self._debug_writer = None

    def _plan_behavior_and_reference(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        ego_speed_mps: float,
        speed_ref_mps: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        stop_goal_active: bool,
        cp_payload: Mapping[str, Any] | None = None,
    ):
        from cpx_planning.behavior_planner import (
            MpcReferenceGenerationContext,
            compute_ego_lane_offset,
            compute_temp_destination,
            evaluate_intersection_obstacle_response,
            generate_mpc_reference,
            select_reference_intent,
        )
        from cpx_planning.pipeline.candidate_evaluation import (
            evaluate_behavior_candidates,
        )
        from cpx_planning.pipeline.candidate_pipeline import (
            build_candidate_intents,
            physical_adjacent_direction,
            route_lane_change_target_anchor,
        )
        from cpx_planning.pipeline.speed_planner import (
            build_speed_plan,
            turn_approach_lookahead_m,
        )

        sim_time_s = self._sim_time_s()
        adapter_output = self.input_adapter.build(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            ego_speed_mps=float(ego_speed_mps),
            object_snapshots=object_snapshots,
            cp_payload=cp_payload,
        )
        planner_input_frame = adapter_output.frame
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
            lane_change_target_reached,
            suppress_lane_change_for_lateral_owner,
        )

        adjacent_lane_directions: dict[int, str] = {}
        # Direction ownership is topology-only. ``ego_waypoint`` comes from
        # the CARLA-only reference_map and intentionally has no AD lane id.
        route_topology_direction = str(
            adapter_output.route_summary.get("lane_change_direction", "") or ""
        ).strip().lower()
        if route_topology_direction in {"left", "right"}:
            adjacent_lane_directions[int(route_optimal_lane_id)] = (
                route_topology_direction
            )
        topology_current_lane_id = int(
            adapter_output.route_summary.get("ad_current_lane_id", 0) or 0
        )
        topology_target_lane_id = int(
            adapter_output.route_summary.get("ad_target_lane_id", 0) or 0
        )
        topology_lane_offset = int(
            adapter_output.route_summary.get("lane_change_offset", 0) or 0
        )
        topology_target_in_local_frame = bool(
            adapter_output.route_summary.get("target_in_local_frame", False)
        )
        # The rolling HD-map frame owns physical left/right whenever it has a
        # target. CARLA canonical ids are point-local semantic labels and may
        # renumber across a junction; their numeric ordering must not override
        # the signed corridor offset.
        if bool(topology_target_in_local_frame) and int(topology_lane_offset) != 0:
            route_topology_direction = (
                "left" if int(topology_lane_offset) > 0 else "right"
            )
            adjacent_lane_directions[int(route_optimal_lane_id)] = str(
                route_topology_direction
            )
        physical_direction_reason = ""
        if (
            route_topology_direction not in {"left", "right"}
            and len(route_points) >= 2
        ):
            physical_target_wp, physical_target_reason = (
                route_lane_change_target_anchor(
                    map_planner=self.reference_map,
                    route_points=route_points,
                    ego_x_m=float(current_state[0]),
                    ego_y_m=float(current_state[1]),
                    z_m=float(getattr(ego_location, "z", 0.0)),
                    nominal_step_m=1.0,
                )
            )
            physical_direction, physical_direction_reason = (
                physical_adjacent_direction(
                    ego_waypoint=ego_waypoint,
                    target_waypoint=physical_target_wp,
                )
            )
            if physical_direction in {"left", "right"}:
                route_topology_direction = str(physical_direction)
                adjacent_lane_directions[int(route_optimal_lane_id)] = str(
                    physical_direction
                )
            elif physical_target_reason:
                physical_direction_reason = str(physical_target_reason)

        # These gates are meters-from-maneuver, but the time available to
        # recover from a transient block (e.g. a background vehicle briefly
        # dropping the target lane's safety score right when a route-required
        # change is authorized) is distance/speed -- at a fixed distance, a
        # faster ego has strictly less time to retry before crossing the
        # "too close" line, so the same transient block that resolves fine
        # at low speed can burn through the whole margin and permanently
        # abandon the lane change at higher speed (confirmed via debug CSV on
        # cpx_single_left_lane_turn: a ~7.6s block consumed a few meters at
        # near-zero speed post-emergency-brake, but the same block duration
        # at cruise speed would consume tens of meters instead). Scaling both
        # bounds by a minimum retry-time margin keeps that recovery window
        # roughly constant in TIME regardless of speed, instead of shrinking
        # as speed increases. prep's margin is kept larger than latest's so
        # the valid window (prep > latest) never inverts and permanently
        # denies the maneuver.
        route_lane_change_preparation_start_distance_m = max(
            float(
                self.config.get(
                    "route_lane_change_preparation_start_distance_m", 45.0
                )
            ),
            float(ego_speed_mps)
            * float(
                self.config.get(
                    "route_lane_change_preparation_time_margin_s", 15.0
                )
            ),
        )
        route_lane_change_latest_start_distance_m = max(
            float(
                self.config.get("route_lane_change_latest_start_distance_m", 12.0)
            ),
            float(ego_speed_mps)
            * float(
                self.config.get(
                    "route_lane_change_latest_retry_time_margin_s", 9.0
                )
            ),
        )
        explicit_lane_change_start_distance_m = max(
            float(self.config.get("route_lane_change_min_trigger_distance_m", 8.0)),
            float(ego_speed_mps)
            * float(self.config.get("route_tracking_lane_change_duration_s", 4.0))
            + float(self.config.get("route_lane_change_trigger_buffer_m", 3.0)),
        )
        authorization_maneuver = str(route_context.next_macro_maneuver)
        normalized_authorization_maneuver = (
            authorization_maneuver.strip().lower().replace("-", "_").replace(" ", "_")
        )
        if (
            route_topology_direction in {"left", "right"}
            and normalized_authorization_maneuver
            in {"lane_change_left", "lane_change_right", "change_lane_left", "change_lane_right"}
        ):
            authorization_maneuver = f"lane_change_{route_topology_direction}"
        lane_change_authorization = authorize_route_lane_change(
            route_lane_change_allowed=bool(route_lane_change_allowed),
            current_lane_id=int(current_lane_id),
            route_required_lane_id=int(route_optimal_lane_id),
            next_macro_maneuver=str(authorization_maneuver),
            current_road_option=str(route_context.current_road_option),
            remaining_distance_m=float(route_context.next_macro_distance_m),
            available_lane_ids=list(planner_input_frame.map_lane.allowed_lane_ids),
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
            preparation_start_distance_m=float(
                route_lane_change_preparation_start_distance_m
            ),
            latest_start_distance_m=float(
                route_lane_change_latest_start_distance_m
            ),
            target_safety_threshold=float(
                self.config.get("route_lane_change_target_safety_threshold", 0.65)
            ),
            require_adjacent=bool(self.config.get("route_lane_change_require_adjacent", True)),
            explicit_lane_change_start_distance_m=float(
                explicit_lane_change_start_distance_m
            ),
            adjacent_lane_directions=adjacent_lane_directions,
            topology_current_lane_id=int(topology_current_lane_id),
            topology_target_lane_id=int(topology_target_lane_id),
            topology_lane_offset=int(topology_lane_offset),
            topology_target_in_local_frame=bool(
                topology_target_in_local_frame
            ),
        )
        cooperative_lane_change_yield_reason = (
            self._cooperative_lane_change_yield_reason(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
            )
            if bool(lane_change_authorization.allowed)
            else ""
        )
        if cooperative_lane_change_yield_reason:
            lane_change_authorization = dataclasses.replace(
                lane_change_authorization,
                allowed=False,
                reason=cooperative_lane_change_yield_reason,
            )
            cooperative_wait_speed_cap_mps = self._cooperative_wait_speed_cap_mps(
                ego_location=ego_location,
                ego_speed_mps=float(ego_speed_mps),
                cooperative_lane_change_yield_reason=(
                    cooperative_lane_change_yield_reason
                ),
            )
            if cooperative_wait_speed_cap_mps is not None:
                speed_ref_mps = min(
                    float(speed_ref_mps), float(cooperative_wait_speed_cap_mps)
                )
        route_lane_change_required = bool(lane_change_authorization.required_by_route)
        # If the route ever genuinely required a specific lane, remember it.
        # The route's own next-macro-maneuver progression advances on
        # arc-length along the original polyline regardless of which lane
        # ego actually occupies, so once the requirement lapses (denied as
        # "too close", or the route just quietly stops asking for it --
        # "already_in_required_lane"/"route_maneuver_does_not_require_lane_
        # change" can appear even though ego's own lane never changed,
        # because route_optimal_lane_id drifted to match current_lane_id
        # instead of the other way around) while ego is STILL in the lane it
        # started in, the lane change was missed, not completed. Left alone,
        # ego just keeps lane_follow-ing straight through and past the
        # junction the plan needed it to turn at, off the swept/tested route
        # corridor entirely (confirmed via debug CSV on a deterministic
        # lane-blocking-vehicle test: ego sailed ~85m past its own
        # destination with no lane change ever attempted and no replan, and
        # the actor was later destroyed off-route). Treat a lapsed
        # requirement the same as turn_reference_unavailable: request a
        # fresh route from wherever ego actually is instead of continuing to
        # chase a plan that assumed a lane change that never happened.
        if bool(lane_change_authorization.required_by_route):
            self._last_required_lane_change_target_lane_id = int(
                lane_change_authorization.target_lane_id
            )
            self._last_required_lane_change_target_ad_lane_id = (
                int(topology_target_lane_id)
                if int(topology_target_lane_id or 0) != 0
                else None
            )
        elif self._last_required_lane_change_target_lane_id is not None:
            if lane_change_target_reached(
                current_lane_id=int(current_lane_id),
                remembered_target_lane_id=int(
                    self._last_required_lane_change_target_lane_id
                ),
                current_ad_lane_id=int(topology_current_lane_id or 0),
                remembered_target_ad_lane_id=int(
                    self._last_required_lane_change_target_ad_lane_id or 0
                ),
                target_in_local_frame=bool(topology_target_in_local_frame),
                target_lane_offset=int(topology_lane_offset),
            ):
                self._last_required_lane_change_target_lane_id = None
                self._last_required_lane_change_target_ad_lane_id = None
            elif _lane_change_execution_active(
                reference_locked=bool(
                    self._route_tracking_lane_change_reference
                ),
                phase=self._route_tracking_lane_change_phase,
            ):
                # The route instruction can advance before the locked
                # trajectory has physically reached its target lane.  Keep
                # the remembered requirement and let the committed geometry
                # finish; otherwise a mid-maneuver route replan resets the
                # reference and produces a one-frame lane-follow interruption.
                pass
            elif bool(self.config.get("missed_lane_change_route_replan_enabled", True)):
                self._attempt_turn_route_replan(
                    ego_location=ego_location,
                    trigger_reason="lane_change_missed_route_unreachable",
                )
                self._last_required_lane_change_target_lane_id = None
                self._last_required_lane_change_target_ad_lane_id = None
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
        raw_traffic_state = str(
            planner_input_frame.planning.traffic_control.signal_state
        )
        resolved_traffic_state, signal_actor_resolution_reason = (
            self._resolve_full_traffic_state_from_carla_actor(
                raw_state=str(raw_traffic_state),
                signal_context=signal_context,
            )
        )
        raw_stop_target = (
            planner_input_frame.planning.traffic_control.stop_target.as_dict()
            if planner_input_frame.planning.traffic_control.stop_target.active
            else None
        )
        filtered_traffic_state, filtered_stop_target, full_traffic_memory_reason = (
            self._full_traffic_memory.update(
                state=str(resolved_traffic_state),
                stop_target=raw_stop_target,
                sim_time_s=float(sim_time_s),
            )
        )
        if str(signal_actor_resolution_reason):
            full_traffic_memory_reason = (
                f"{signal_actor_resolution_reason};{full_traffic_memory_reason}"
                if str(full_traffic_memory_reason)
                else str(signal_actor_resolution_reason)
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
        traffic_stop_forward_m, traffic_stop_target_reliable = self.reference_generator.stop_target_forward(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            stop_target=(
                dict(filtered_stop_target)
                if isinstance(filtered_stop_target, Mapping)
                else None
            ),
            fallback_destination_state=[],
        )
        (
            upcoming_turn_direction,
            upcoming_turn_distance_m,
            upcoming_turn_reason,
        ) = self.route_manager.upcoming_turn(
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
            lookahead_m=float(
                turn_approach_lookahead_m(
                    cruise_speed_mps=float(self.target_speed_mps),
                    config=dict(self.config),
                )
            ),
        )
        route_macro_text = str(route_context.next_macro_maneuver or "").strip().lower()
        route_macro_normalized = route_macro_text.replace("-", "_").replace(" ", "_")
        route_advanced_to_lane_change = route_macro_normalized in {
            "lane_change_left",
            "lane_change_right",
            "change_lane_left",
            "change_lane_right",
        }
        route_macro_direction = (
            "left" if "turn left" in route_macro_text
            else "right" if "turn right" in route_macro_text
            else ""
        )
        if route_macro_direction:
            upcoming_turn_direction = str(route_macro_direction)
            upcoming_turn_distance_m = float(route_context.next_macro_distance_m)
            upcoming_turn_reason = "admap_route_macro_direction"
        (
            turn_exit_heading_error_rad,
            turn_exit_lateral_m,
            turn_exit_alignment_reason,
        ) = self.route_manager.carla_route_alignment(
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
            heading_lookahead_m=float(
                self.config.get("scenario_turn_exit_heading_lookahead_m", 5.0)
            ),
        )
        turn_exit_alignment_valid = bool(
            math.isfinite(float(turn_exit_heading_error_rad))
            and math.isfinite(float(turn_exit_lateral_m))
        )
        turn_exit_aligned = bool(
            turn_exit_alignment_valid
            and abs(float(turn_exit_heading_error_rad))
            <= float(
                self.config.get(
                    "scenario_turn_exit_max_heading_error_rad",
                    0.15,
                )
            )
            and float(turn_exit_lateral_m)
            <= float(
                self.config.get(
                    "scenario_turn_exit_max_lateral_m",
                    0.75,
                )
            )
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
            upcoming_turn_direction=str(upcoming_turn_direction),
            upcoming_turn_distance_m=float(upcoming_turn_distance_m),
            turn_exit_alignment_valid=bool(turn_exit_alignment_valid),
            turn_exit_aligned=bool(turn_exit_aligned),
            turn_exit_heading_error_rad=float(turn_exit_heading_error_rad),
            turn_exit_lateral_m=float(turn_exit_lateral_m),
            boundary_recovery_request=(
                getattr(self, "_boundary_recovery_request", None)
                if bool(
                    self.config.get(
                        "boundary_recovery_enabled",
                        False,
                    )
                )
                else None
            ),
        )
        lane_change_authorization = suppress_lane_change_for_lateral_owner(
            lane_change_authorization,
            owner_state=scenario_decision.state,
        )
        lane_change_authorized = bool(lane_change_authorization.allowed)
        if str(lane_change_authorization.reason).startswith(
            "scenario_lateral_owner:"
        ):
            opportunistic_lane_change_allowed = False
            lane_change_gate_reason = (
                "opportunistic_lane_change_suppressed:"
                + str(lane_change_authorization.reason)
            )
        if bool(self.config.get("route_tracking_baseline_enabled", False)):
            return self._build_route_tracking_baseline_plan(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                ego_speed_mps=float(ego_speed_mps),
                current_state=current_state,
                current_lane_id=int(current_lane_id),
                object_snapshots=object_snapshots,
                raw_front_stop_active=bool(stop_goal_active),
                scenario_decision=scenario_decision,
                raw_traffic_state=str(raw_traffic_state),
                resolved_traffic_state=str(resolved_traffic_state),
                filtered_traffic_state=str(filtered_traffic_state),
                traffic_control_from_cp=bool(
                    planner_input_frame.planning.traffic_control.from_cp
                ),
                behavior_stop_target=(
                    dict(scenario_decision.behavior_stop_target)
                    if isinstance(
                        scenario_decision.behavior_stop_target,
                        Mapping,
                    )
                    else None
                ),
                route_context=route_context,
                route_reference_allowed=bool(route_reference_allowed),
                route_reference_gate_reason=str(route_reference_gate_reason),
                upcoming_turn_direction=str(upcoming_turn_direction),
                upcoming_turn_distance_m=float(upcoming_turn_distance_m),
                adapter_output=adapter_output,
                planner_input_frame=planner_input_frame,
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
        filtered_signal_context["raw_signal_state"] = str(raw_traffic_state)
        filtered_signal_context["resolved_signal_state"] = str(
            resolved_traffic_state
        )
        filtered_signal_context["signal_state"] = str(filtered_traffic_state)
        filtered_signal_context["behavior_signal_state"] = str(behavior_traffic_state)
        filtered_signal_context["scenario_owns_traffic_control"] = True
        filtered_signal_context["scenario_fsm_state"] = str(
            scenario_decision.state
        )
        filtered_signal_context["traffic_stop_forward_m"] = float(traffic_stop_forward_m)
        filtered_signal_context["traffic_stop_commit_distance_m"] = float(traffic_stop_commit_distance_m)
        filtered_signal_context["carla_upcoming_turn_direction"] = str(
            upcoming_turn_direction
        )
        filtered_signal_context["carla_upcoming_turn_distance_m"] = (
            ""
            if not math.isfinite(float(upcoming_turn_distance_m))
            else float(upcoming_turn_distance_m)
        )
        filtered_signal_context["carla_upcoming_turn_reason"] = str(
            upcoming_turn_reason
        )
        filtered_signal_context["turn_exit_heading_error_rad"] = (
            ""
            if not bool(turn_exit_alignment_valid)
            else float(turn_exit_heading_error_rad)
        )
        filtered_signal_context["turn_exit_lateral_m"] = (
            ""
            if not bool(turn_exit_alignment_valid)
            else float(turn_exit_lateral_m)
        )
        filtered_signal_context["turn_exit_aligned"] = bool(turn_exit_aligned)
        filtered_signal_context["turn_exit_alignment_reason"] = str(
            turn_exit_alignment_reason
        )
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

        try:
            behavior_lane_alignment = compute_ego_lane_offset(
                self.reference_map,
                ego_pose,
            )
        except Exception:
            behavior_lane_alignment = {
                "lane_id": 0,
                "lateral_offset_m": float("inf"),
                "heading_error_rad": float("inf"),
            }
        behavior_lane_alignment_valid = bool(
            int(behavior_lane_alignment.get("lane_id", 0) or 0) != 0
            and math.isfinite(
                float(behavior_lane_alignment.get("lateral_offset_m", float("nan")))
            )
            and math.isfinite(
                float(behavior_lane_alignment.get("heading_error_rad", float("nan")))
            )
        )
        behavior_lane_lateral_error_m = float(
            behavior_lane_alignment.get("lateral_offset_m", 0.0)
        )
        behavior_lane_heading_error_rad = float(
            behavior_lane_alignment.get("heading_error_rad", 0.0)
        )
        if not bool(behavior_lane_alignment_valid):
            behavior_lane_lateral_error_m = float("inf")
            behavior_lane_heading_error_rad = float("inf")

        nearest_front_obstacles_by_lane = self._nearest_front_obstacle_by_lane(
            ego_snapshot={
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "psi": float(ego_yaw_rad),
            },
            obstacle_snapshots=object_snapshots,
            lane_assignments=dict(
                planner_input_frame.prediction.lane_assignments or {}
            ),
            available_lane_ids=list(planner_input_frame.map_lane.allowed_lane_ids),
        )
        static_front_obstacle = nearest_front_obstacles_by_lane.get(
            int(current_lane_id)
        )
        actual_obstacle_mode = (
            "INTERSECTION"
            if bool(planner_input_frame.map_lane.in_junction)
            else "NORMAL"
        )
        obstacle_evaluation_mode = str(actual_obstacle_mode)
        if (
            obstacle_evaluation_mode == "NORMAL"
            and bool(
                self.config.get(
                    "static_obstacle_replan_normal_mode_enabled",
                    self.behavior_runtime_cfg.get(
                        "static_obstacle_replan_normal_mode_enabled",
                        True,
                    ),
                )
            )
        ):
            # Reuse the same conservative two-condition classifier on normal
            # roads when explicitly enabled. The classifier itself remains
            # intersection-scoped for backward compatibility.
            obstacle_evaluation_mode = "INTERSECTION"
        static_obstacle_response = evaluate_intersection_obstacle_response(
            mode=str(obstacle_evaluation_mode),
            front_obstacle_speed_mps=(
                None
                if static_front_obstacle is None
                else float(static_front_obstacle.get("v", 0.0))
            ),
            original_max_velocity_mps=float(self.target_speed_mps),
            moving_obstacle_speed_threshold_mps=float(
                self.config.get(
                    "static_obstacle_speed_threshold_mps",
                    self.behavior_runtime_cfg.get(
                        "static_obstacle_speed_threshold_mps",
                        self.behavior_runtime_cfg.get(
                            "intersection_obstacle_moving_speed_threshold_mps",
                            0.5,
                        ),
                    ),
                )
            ),
            route_lane_safety_score=float(
                lane_safety_scores.get(int(current_lane_id), 1.0)
            ),
            static_obstacle_replan_lane_safety_threshold=float(
                self.config.get(
                    "static_obstacle_replan_lane_safety_threshold",
                    self.behavior_runtime_cfg.get(
                        "static_obstacle_replan_lane_safety_threshold",
                        self.behavior_runtime_cfg.get(
                            "intersection_static_obstacle_replan_lane_safety_threshold",
                            0.5,
                        ),
                    ),
                )
            ),
        )
        traffic_control_stop_active = bool(
            scenario_decision.stop_goal_active
            or str(behavior_traffic_state).strip().lower()
            in {"red", "yellow", "stop"}
        )
        static_replan_requested = bool(
            self.config.get(
                "static_obstacle_replan_enabled",
                self.behavior_runtime_cfg.get(
                    "static_obstacle_replan_enabled",
                    True,
                ),
            )
            and static_obstacle_response.get(
                "request_static_obstacle_replan", False
            )
            and not bool(traffic_control_stop_active)
        )
        static_obstacle_transition_hold = False
        static_obstacle_cooldown_hold = False
        latched_local_target_lane_id = getattr(
            self, "_static_obstacle_local_target_lane_id", None
        )
        if (
            latched_local_target_lane_id is not None
            and int(current_lane_id) == int(latched_local_target_lane_id)
            and not bool(self._route_tracking_lane_change_reference)
        ):
            # The local lane-borrow maneuver has geometrically converged.
            self._static_obstacle_local_target_lane_id = None
            latched_local_target_lane_id = None
        static_obstacle_local_avoidance_active = bool(
            latched_local_target_lane_id is not None
        )
        static_obstacle_local_target_lane_id: int | None = (
            None
            if latched_local_target_lane_id is None
            else int(latched_local_target_lane_id)
        )
        static_obstacle_id = (
            ""
            if static_front_obstacle is None
            else self._object_track_id(static_front_obstacle)
        )
        if not bool(static_replan_requested):
            self._static_obstacle_candidate_id = ""
            self._static_obstacle_candidate_since_s = -float("inf")
            # Seeing a clear frame closes the previous encounter. If the same
            # object blocks the route again during cooldown, it is a new
            # encounter and must hold stop until a retry is allowed.
            self._static_obstacle_route_transition_pending = False
            self._static_obstacle_replan_failed_latched = False
            self._static_obstacle_replan_status = (
                "local_avoidance_executing"
                if bool(static_obstacle_local_avoidance_active)
                else "traffic_control_excluded"
                if bool(traffic_control_stop_active)
                and bool(
                    static_obstacle_response.get(
                        "request_static_obstacle_replan", False
                    )
                )
                else "idle"
            )
        else:
            if str(static_obstacle_id) != str(self._static_obstacle_candidate_id):
                self._static_obstacle_candidate_id = str(static_obstacle_id)
                self._static_obstacle_candidate_since_s = float(sim_time_s)
            blocked_confirm_s = max(
                0.0,
                float(
                    self.config.get(
                        "static_obstacle_blocked_confirm_s",
                        self.behavior_runtime_cfg.get(
                            "static_obstacle_blocked_confirm_s",
                            1.0,
                        ),
                    )
                ),
            )
            blocked_elapsed_s = max(
                0.0,
                float(sim_time_s)
                - float(self._static_obstacle_candidate_since_s),
            )
            if blocked_elapsed_s < blocked_confirm_s:
                self._static_obstacle_replan_status = "confirming"
            else:
                local_avoidance_enabled = bool(
                    self.config.get(
                        "static_obstacle_local_avoidance_enabled",
                        self.behavior_runtime_cfg.get(
                            "static_obstacle_local_avoidance_enabled", True
                        ),
                    )
                )
                local_target_lane_id = (
                    _select_static_obstacle_local_avoidance_lane(
                        current_lane_id=int(current_lane_id),
                        available_lane_ids=list(
                            planner_input_frame.map_lane.allowed_lane_ids
                        ),
                        lane_safety_scores=lane_safety_scores,
                        lane_prediction_risks=dict(
                            planner_input_frame.prediction.lane_prediction_risks
                        ),
                        minimum_safety_score=float(
                            self.config.get(
                                "static_obstacle_local_lane_min_safety_score",
                                self.behavior_runtime_cfg.get(
                                    "static_obstacle_local_lane_min_safety_score",
                                    0.55,
                                ),
                            )
                        ),
                    )
                    if bool(local_avoidance_enabled)
                    and str(actual_obstacle_mode) == "NORMAL"
                    else None
                )
                avoidance_lane_yield_reason = ""
                if local_target_lane_id is not None:
                    avoidance_lane_yield_reason = (
                        self._cooperative_avoidance_lane_yield_reason(
                            target_lane_id=int(local_target_lane_id),
                            ego_location=ego_location,
                            ego_yaw_rad=float(ego_yaw_rad),
                        )
                    )
                    if avoidance_lane_yield_reason:
                        local_target_lane_id = None
                if local_target_lane_id is not None:
                    static_obstacle_local_avoidance_active = True
                    static_obstacle_local_target_lane_id = int(local_target_lane_id)
                    self._static_obstacle_local_target_lane_id = int(
                        local_target_lane_id
                    )
                    self._static_obstacle_replan_failed_latched = False
                    self._static_obstacle_route_transition_pending = False
                    self._static_obstacle_replan_status = "local_avoidance_ready"
                    self._static_obstacle_replan_reason = (
                        "static_obstacle_local_lane_borrow:"
                        f"target_lane={int(local_target_lane_id)}"
                    )
                elif bool(
                    self.config.get(
                        "static_obstacle_global_replan_enabled",
                        self.behavior_runtime_cfg.get(
                            "static_obstacle_global_replan_enabled", False
                        ),
                    )
                ):
                    attempted, succeeded, replan_reason = (
                        self._attempt_static_obstacle_route_replan(
                            ego_location=ego_location,
                            obstacle=dict(static_front_obstacle or {}),
                        )
                    )
                    self._static_obstacle_replan_reason = str(replan_reason)
                    if bool(succeeded):
                        self._static_obstacle_replan_failed_latched = False
                        self._static_obstacle_replan_status = "succeeded"
                        self._static_obstacle_route_transition_pending = True
                        static_obstacle_transition_hold = True
                    elif bool(attempted):
                        self._static_obstacle_replan_failed_latched = True
                        self._static_obstacle_replan_status = "failed_stop"
                    else:
                        (
                            self._static_obstacle_replan_status,
                            static_obstacle_cooldown_hold,
                        ) = _static_obstacle_cooldown_policy(
                            failed_latched=bool(
                                self._static_obstacle_replan_failed_latched
                            ),
                            route_transition_pending=bool(
                                self._static_obstacle_route_transition_pending
                            ),
                        )
                else:
                    # Local avoidance is unavailable or unsafe.  Preserve the
                    # active global route and stop behind the obstacle; a
                    # cooperative road-closure event may request rerouting via
                    # the separate BehaviorPlanner reroute-message path. Retry
                    # continues every tick (this whole branch re-runs
                    # unconditionally next step), so a cooperative-yield hold
                    # self-clears as soon as the peer's claim does.
                    self._static_obstacle_replan_failed_latched = True
                    self._static_obstacle_route_transition_pending = False
                    self._static_obstacle_replan_status = (
                        "local_avoidance_yield_to_peer_cav"
                        if avoidance_lane_yield_reason
                        else "local_avoidance_unavailable_stop"
                    )
                    self._static_obstacle_replan_reason = (
                        avoidance_lane_yield_reason
                        or "static_obstacle_local_avoidance_unavailable"
                    )

        if bool(static_obstacle_local_avoidance_active):
            # A confirmed blocker is an explicit behavior-level reason to
            # consider an adjacent lane.  It bypasses only the route-demand
            # gate; prediction, lane-safety, reference and MPC safety gates
            # remain unchanged.
            opportunistic_lane_change_allowed = True
            candidate_lane_ids = [
                int(current_lane_id),
                int(static_obstacle_local_target_lane_id),
            ]
            preferred_target_lane_id = int(static_obstacle_local_target_lane_id)
            candidate_frame = evaluate_behavior_candidates(
                lane_safety_scores=lane_safety_scores,
                lane_prediction_risks=dict(
                    planner_input_frame.prediction.lane_prediction_risks
                ),
                ego_lane_id=int(current_lane_id),
                selected_lane_id=int(current_lane_id),
                available_lane_ids=list(candidate_lane_ids),
                route_optimal_lane_id=int(route_optimal_lane_id),
                mode=str(actual_obstacle_mode),
                mpc_feedback_blocked_lane_ids=list(
                    mpc_feedback.get("blocked_lane_ids", []) or []
                ),
                mpc_feedback_weight=float(
                    self.config.get("mpc_feedback_candidate_weight", 80.0)
                ),
            )
        static_obstacle_stop_active = bool(
            self._static_obstacle_replan_failed_latched
            or static_obstacle_transition_hold
            or static_obstacle_cooldown_hold
        )
        self._last_static_obstacle_stop_active_input = bool(static_obstacle_stop_active)

        command = self.behavior_planner.update(
            static_obstacle_stop_active=bool(static_obstacle_stop_active),
            lane_safety_scores=lane_safety_scores,
            ego_lane_id=int(current_lane_id),
            selected_lane_id=int(current_lane_id),
            ego_lateral_offset_m=float(behavior_lane_lateral_error_m),
            ego_heading_error_rad=float(behavior_lane_heading_error_rad),
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
            nearest_front_obstacles_by_lane=nearest_front_obstacles_by_lane,
            lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
            preferred_target_lane_id=int(preferred_target_lane_id),
            local_avoidance_target_lane_id=(
                int(static_obstacle_local_target_lane_id)
                if bool(static_obstacle_local_avoidance_active)
                and static_obstacle_local_target_lane_id is not None
                else None
            ),
            lane_change_completion_allowed=not bool(
                self._route_tracking_lane_change_reference
            ),
        )
        decision = str(command.get("decision", "lane_follow"))
        target_lane_id = int(command.get("target_lane_id", current_lane_id) or current_lane_id)
        lc_state = str(command.get("lc_state", "LANE_KEEP"))
        if bool(static_obstacle_local_avoidance_active):
            if str(decision) in {"lane_change_left", "lane_change_right"}:
                self._static_obstacle_replan_status = "local_avoidance_executing"
            elif bool(traffic_control_stop_active) and str(decision) in {
                "stop_at_intersection",
                "stop_sign",
            }:
                self._static_obstacle_replan_status = (
                    "local_avoidance_preempted_by_traffic_control"
                )
        behavior_override_reason = ""
        lane_change_commitment_pending_stabilization = bool(
            self._route_tracking_lane_change_reference
        )
        turn_prepare_speed_suppressed_by_lane_change = bool(
            lane_change_commitment_pending_stabilization
            and str(scenario_decision.state).strip().upper() == "PREPARE_TURN"
            and not bool(scenario_decision.stop_goal_active)
        )
        scenario_speed_cap_active = (
            scenario_decision.speed_cap_mps is not None
            and float(scenario_decision.speed_cap_mps) < float(self.target_speed_mps)
            and not bool(turn_prepare_speed_suppressed_by_lane_change)
        )
        if (
            bool(scenario_speed_cap_active)
            and str(decision) not in {"stop_at_intersection", "stop_sign", "emergency_brake"}
        ):
            speed_ref_mps = min(float(speed_ref_mps), float(traffic_stop_approach_speed_cap_mps))
            behavior_override_reason = str(traffic_stop_approach_reason)
        route_turn_decision = self._route_option_turn_decision(
            current_road_option=str(route_context.current_road_option),
            next_macro_maneuver="",
        )
        if bool(route_advanced_to_lane_change):
            # The AD route has consumed the connector. A stale CARLA road
            # option must not recreate the turn after ScenarioManager released
            # it, even while CARLA still reports ego inside the junction.
            route_turn_decision = ""
            self._turn_latch_decision = ""
            self._turn_latch_until_sim_time_s = -float("inf")
        route_turn_prepare_decision = ""
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
            # PREPARE intentionally keeps tracking the source lane, but the
            # Behavior FSM must remain latched long enough to advance to
            # EXECUTE. Resetting it here caused a permanent prepare/reset loop.
            decision = "lane_follow"
            target_lane_id = int(current_lane_id)
            behavior_override_reason = (
                str(behavior_override_reason) + ";"
                if str(behavior_override_reason)
                else ""
            ) + "prepare_lane_change_reference_locked_to_current_lane"
        if (
            str(decision) in {"lane_change_left", "lane_change_right"}
            and not bool(
                lane_change_authorized or opportunistic_lane_change_allowed
            )
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
        scenario_override_is_mandatory_stop = str(scenario_behavior_override) in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }
        suppress_turn_override_for_local_avoidance = bool(
            static_obstacle_local_avoidance_active
            and str(decision) in {"lane_change_left", "lane_change_right"}
            and not bool(planner_input_frame.map_lane.in_junction)
            and not bool(scenario_override_is_mandatory_stop)
        )
        if str(scenario_behavior_override) and not bool(
            suppress_turn_override_for_local_avoidance
        ):
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
        # The route's current_road_option flips to LEFT/RIGHT purely off
        # distance travelled along the GRP, with no awareness of whether a
        # locked route-tracking lane change has actually finished converging
        # (lane id matched, lateral/heading error settled). Handing off to
        # the turn-reference generator before that happens lets it sample a
        # waypoint far enough from ego's still-not-centered position to trip
        # its own lateral contract check, with no recovery path -- keep
        # tracking the locked lane-change reference a little longer instead;
        # _route_tracking_lane_change_reference clears itself once the
        # commitment is genuinely released (or abandoned as stale), so this
        # naturally falls through on its own.
        if (
            (str(route_turn_decision) or str(route_turn_prepare_decision))
            and not str(scenario_behavior_override)
            and not bool(lane_change_commitment_pending_stabilization)
            and not bool(static_obstacle_local_avoidance_active)
            and str(decision) not in {"stop_at_intersection", "stop_sign", "emergency_brake"}
        ):
            decision = str(route_turn_decision or route_turn_prepare_decision)
            target_lane_id = int(current_lane_id)
            lc_state = (
                "INTERSECTION_TURN_LEFT"
                if str(decision).endswith("_left")
                else "INTERSECTION_TURN_RIGHT"
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
            and not bool(static_obstacle_local_avoidance_active)
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
        front_gap_m, front_gap_actor_id = self._front_gap_m(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            object_snapshots=object_snapshots,
            current_lane_id=int(current_lane_id),
            lane_assignments=dict(adapter_output.lane_assignments),
            lane_change_direction=(
                "left" if str(decision) == "lane_change_left"
                else "right" if str(decision) == "lane_change_right"
                else ""
            ),
            lane_change_progress=float(
                getattr(self, "_route_tracking_lane_change_progress", 0.0) or 0.0
            ),
            return_actor_id=True,
        )
        front_gap_obstacle_speed_mps = None
        if front_gap_actor_id:
            for _snapshot in object_snapshots:
                if str(self._object_track_id(_snapshot)) == str(front_gap_actor_id):
                    front_gap_obstacle_speed_mps = max(
                        0.0,
                        float(
                            _snapshot.get(
                                "v", _snapshot.get("speed_mps", 0.0)
                            )
                            or 0.0
                        ),
                    )
                    break
        speed_plan = build_speed_plan(
            scenario_decision=scenario_decision,
            behavior_decision=str(decision),
            requested_speed_mps=float(speed_ref_mps),
            ego_speed_mps=float(ego_speed_mps),
            config=dict(self.config),
            front_gap_m=front_gap_m,
            front_obstacle_speed_mps=front_gap_obstacle_speed_mps,
            upcoming_turn_direction=str(upcoming_turn_direction),
            upcoming_turn_distance_m=(
                None
                if not math.isfinite(float(upcoming_turn_distance_m))
                else float(upcoming_turn_distance_m)
            ),
            lane_change_commitment_active=bool(
                lane_change_commitment_pending_stabilization
            ),
            previous_idm_acceleration_mps2=getattr(
                self, "_previous_following_idm_acceleration_mps2", None
            ),
        )
        self._previous_following_idm_acceleration_mps2 = (
            None
            if speed_plan.idm_acceleration_mps2 is None
            else float(speed_plan.idm_acceleration_mps2)
        )
        speed_ref_mps = float(speed_plan.target_speed_mps)
        stop_goal_active = bool(stop_goal_active or speed_plan.stop_goal_active)
        planner_mode = "INTERSECTION" if bool(planner_input_frame.map_lane.in_junction) else "NORMAL"

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
            "front_gap_actor_id": str(front_gap_actor_id or ""),
            "front_gap_obstacle_speed_mps": (
                ""
                if front_gap_obstacle_speed_mps is None
                else float(front_gap_obstacle_speed_mps)
            ),
            "snapshot_repr_diag": str(
                [
                    {
                        k: v
                        for k, v in dict(snap).items()
                        if k in (
                            "track_id", "object_id", "vehicle_id",
                            "actor_id", "id", "v", "speed_mps", "x", "y",
                        )
                    }
                    for snap in list(object_snapshots or [])
                ]
            ),
            "route_reference_allowed": bool(route_reference_allowed),
            "route_reference_gate_reason": str(route_reference_gate_reason),
            "route_lane_change_allowed": bool(route_lane_change_allowed),
            "opportunistic_lane_change_allowed": bool(opportunistic_lane_change_allowed),
            "lane_change_gate_reason": str(lane_change_gate_reason),
            "static_obstacle_local_avoidance_active": bool(
                static_obstacle_local_avoidance_active
            ),
            "static_obstacle_local_target_lane_id": (
                ""
                if static_obstacle_local_target_lane_id is None
                else int(static_obstacle_local_target_lane_id)
            ),
            "static_obstacle_candidate_since_s": float(
                self._static_obstacle_candidate_since_s
            ),
            "static_obstacle_global_replan_enabled": bool(
                self.config.get(
                    "static_obstacle_global_replan_enabled",
                    self.behavior_runtime_cfg.get(
                        "static_obstacle_global_replan_enabled", False
                    ),
                )
            ),
            "route_lane_change_required": bool(route_lane_change_required),
            "behavior_lane_lateral_error_m": float(
                behavior_lane_lateral_error_m
            ),
            "behavior_lane_heading_error_deg": math.degrees(
                float(behavior_lane_heading_error_rad)
            ),
            "behavior_lane_alignment_valid": bool(
                behavior_lane_alignment_valid
            ),
            "behavior_lane_change_completion_allowed": not bool(
                self._route_tracking_lane_change_reference
            ),
            **dict(lane_change_authorization.as_debug_fields()),
            "behavior_override_reason": str(behavior_override_reason),
            "turn_latch_reason": str(turn_latch_reason),
            "route_current_road_option": str(route_context.current_road_option),
            "route_next_macro_maneuver": str(route_context.next_macro_maneuver),
            "traffic_memory_reason": str(full_traffic_memory_reason),
            "traffic_signal_raw_state": str(
                planner_input_frame.planning.traffic_control.signal_state
            ),
            "traffic_signal_resolved_state": str(resolved_traffic_state),
            "traffic_signal_filtered_state": str(filtered_traffic_state),
            "traffic_signal_behavior_state": str(behavior_traffic_state),
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
        reference_debug.update({
            "carla_upcoming_turn_direction": str(upcoming_turn_direction),
            "carla_upcoming_turn_distance_m": (
                ""
                if not math.isfinite(float(upcoming_turn_distance_m))
                else float(upcoming_turn_distance_m)
            ),
            "carla_upcoming_turn_reason": str(upcoming_turn_reason),
        })
        reference_debug.update(source_quality)
        if bool(self.full_candidate_pipeline_enabled):
            traffic_stop_active = bool(scenario_decision.stop_goal_active)
            behavior_lane_change_proposed = str(decision) in {
                "lane_change_left",
                "lane_change_right",
            }
            opportunistic_lane_change_authorized = bool(
                behavior_lane_change_proposed
                and opportunistic_lane_change_allowed
                and not lane_change_authorized
            )
            candidate_lane_change_authorized = bool(
                lane_change_authorized or opportunistic_lane_change_authorized
            )
            candidate_lane_change_target_lane_id = int(
                lane_change_authorization.target_lane_id
                if lane_change_authorized
                else target_lane_id
            )
            candidate_lane_change_authorization_source = (
                "route" if lane_change_authorized else "opportunistic"
            )
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
                lane_change_authorized=bool(candidate_lane_change_authorized),
                lane_change_authorized_target_lane_id=int(
                    candidate_lane_change_target_lane_id
                ),
                # Route-required and explicitly proposed opportunistic changes
                # share one downstream candidate/reference/MPC gate.
                allow_lane_change_candidates=bool(
                    candidate_lane_change_authorized
                ),
                stop_target=(
                    dict(behavior_stop_target)
                    if isinstance(behavior_stop_target, Mapping)
                    else None
                ),
                lane_change_assertive_duration_s=float(
                    self.candidate_lane_change_assertive_duration_s
                ),
                lane_change_normal_duration_s=float(
                    self.candidate_lane_change_normal_duration_s
                ),
                lane_change_conservative_duration_s=float(
                    self.candidate_lane_change_conservative_duration_s
                ),
                lane_change_assertive_speed_scale=float(
                    self.candidate_lane_change_assertive_speed_scale
                ),
                lane_change_normal_speed_scale=float(
                    self.candidate_lane_change_normal_speed_scale
                ),
                lane_change_conservative_speed_scale=float(
                    self.candidate_lane_change_conservative_speed_scale
                ),
                lane_change_authorization_source=str(
                    candidate_lane_change_authorization_source
                ),
                lane_change_authorization_direction=(
                    str(lane_change_authorization.direction or "")
                    if lane_change_authorized
                    else "left" if str(decision) == "lane_change_left"
                    else "right" if str(decision) == "lane_change_right"
                    else ""
                ),
                lane_change_defer_cost=float(
                    self.config.get("candidate_lane_change_defer_cost", 10.0)
                ),
                turn_obstacle_stop_defer_cost=float(
                    self.config.get("candidate_turn_obstacle_stop_defer_cost", 90.0)
                ),
                local_obstacle_avoidance_active=bool(
                    static_obstacle_local_avoidance_active
                ),
                local_obstacle_stop_defer_cost=float(
                    self.config.get(
                        "candidate_local_obstacle_stop_defer_cost", 25.0
                    )
                ),
                human_like_lane_change_enabled=bool(
                    self.config.get("human_like_lane_change_enabled", True)
                ),
                ego_speed_mps=float(ego_speed_mps),
                lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                lane_change_available_distance_m=(
                    lane_change_authorization.distance_to_maneuver_m
                    if bool(candidate_lane_change_authorized)
                    else None
                ),
                human_lane_change_min_duration_s=float(
                    self.config.get("human_lane_change_min_duration_s", 3.0)
                ),
                human_lane_change_max_duration_s=float(
                    self.config.get("human_lane_change_max_duration_s", 6.5)
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
                required_lane_change_decision=(
                    "lane_change_left"
                    if bool(route_lane_change_required)
                    and bool(lane_change_authorized)
                    and str(lane_change_authorization.direction).strip().lower() == "left"
                    else "lane_change_right"
                    if bool(route_lane_change_required)
                    and bool(lane_change_authorized)
                    and str(lane_change_authorization.direction).strip().lower() == "right"
                    else ""
                ),
                required_lane_change_target_lane_id=(
                    int(lane_change_authorization.target_lane_id)
                    if bool(route_lane_change_required) and bool(lane_change_authorized)
                    else 0
                ),
            )
            # Candidate selection owns maneuver geometry, never longitudinal
            # authority.  In particular, do not re-cap a lane-change speed
            # here after SpeedPlanner has selected it.  The curvature-derived
            # value remains diagnostic so a future unified SpeedPlanner can
            # consume it explicitly, but it must not silently rewrite the MPC
            # entry target.  Turns retain their safety cap below because the
            # final turn decision is not known when the earlier speed plan is
            # built; moving that input upstream is a separate change.
            #
            # For turns this is the only place the configured
            # full_intersection_turn_speed_cap_mps actually reaches the
            # winning candidate at all -- confirmed via debug CSV at 35mph:
            # speed climbed past 4 m/s through an entire intersection_turn_left
            # with the cap doing nothing, because only build_speed_plan's
            # (bypassed) turn_cap_mps was ever computed against it.
            if str(decision) in {"lane_change_left", "lane_change_right"}:
                lane_change_curvature_1pm = float(
                    self.reference_generator.discrete_curvature_1pm(
                        local_lane_center_reference
                    )
                )
                lane_change_curvature_cap_mps = curvature_speed_cap_mps(
                    curve_curvature_abs=float(lane_change_curvature_1pm),
                    curve_min_curvature=max(
                        0.0,
                        float(
                            self.config.get(
                                "full_lane_change_curvature_min_curvature_1pm",
                                0.002,
                            )
                        ),
                    ),
                    current_speed_mps=float(ego_speed_mps),
                    curve_lateral_accel_limit_mps2=max(
                        0.1,
                        float(
                            self.config.get(
                                "route_tracking_lane_change_lateral_accel_limit_mps2",
                                1.3,
                            )
                        ),
                    ),
                    speed_enable_threshold_mps=0.0,
                )
                reference_debug.update({
                    "lane_change_reference_curvature_1pm": float(
                        lane_change_curvature_1pm
                    ),
                    "lane_change_curvature_speed_advisory_mps": (
                        ""
                        if lane_change_curvature_cap_mps is None
                        else float(lane_change_curvature_cap_mps)
                    ),
                    "lane_change_longitudinal_authority": "SpeedPlanner",
                })
            elif str(decision) in {"intersection_turn_left", "intersection_turn_right"}:
                # full_intersection_turn_speed_cap_mps is a per-fleet ceiling,
                # not a per-turn comfort speed: two turns at different
                # intersections can have very different connector curvature
                # (confirmed via debug CSV -- this route's right turn measured
                # ~0.145 1/m vs. the left turn's ~0.091 1/m), so a single
                # static cap that is comfortable for a gentle turn can still
                # be too fast for a tighter one, causing the turn's swept
                # vehicle envelope to exceed the drivable corridor and the
                # candidate to be permanently rejected with no fallback.
                # Derive this turn's own cap from its actual winning-candidate
                # curvature and take the tighter of that and the static
                # ceiling.
                turn_ceiling_mps = max(
                    0.1,
                    float(
                        self.config.get("full_intersection_turn_speed_cap_mps", 2.2)
                    ),
                )
                turn_curvature_1pm = float(
                    self.reference_generator.discrete_curvature_1pm(
                        local_lane_center_reference
                    )
                )
                turn_curvature_cap_mps = curvature_speed_cap_mps(
                    curve_curvature_abs=float(turn_curvature_1pm),
                    curve_min_curvature=max(
                        0.0,
                        float(
                            self.config.get(
                                "full_intersection_turn_curvature_min_curvature_1pm",
                                0.01,
                            )
                        ),
                    ),
                    current_speed_mps=float(ego_speed_mps),
                    curve_lateral_accel_limit_mps2=max(
                        0.1,
                        float(
                            self.config.get(
                                "full_intersection_turn_lateral_accel_comfort_mps2",
                                2.5,
                            )
                        ),
                    ),
                    speed_enable_threshold_mps=0.0,
                )
                turn_cap_mps = float(turn_ceiling_mps)
                if turn_curvature_cap_mps is not None:
                    turn_cap_mps = min(turn_cap_mps, float(turn_curvature_cap_mps))
                speed_ref_mps = min(float(speed_ref_mps), float(turn_cap_mps))
            # The upstream front-gap flag proposes an obstacle-stop candidate;
            # it must not remain a global stop latch after a safe lane-change
            # candidate wins. Traffic-control stops remain hard and exclusive.
            stop_goal_active = bool(
                scenario_decision.stop_goal_active
                or selected_candidate_debug.get(
                    "candidate_selected_stop_goal_active",
                    False,
                )
                or str(decision)
                in {"stop_at_intersection", "stop_sign", "emergency_brake"}
            )
            if str(decision) == "lane_follow":
                lc_state = "LANE_KEEP"
            if str(decision) in {"stop_at_intersection", "stop_sign", "emergency_brake"}:
                lc_state = "LANE_KEEP"
            reference_debug.update(selected_candidate_debug)
            # Candidate commitment can restore a lane-change decision after
            # an upstream authorization gate temporarily set the baseline
            # back to lane-follow.  Normalize the FSM from the FINAL decision
            # and locked maneuver phase; otherwise diagnostics and downstream
            # control context can become lane_change_right + LANE_KEEP.
            if str(decision) in {"lane_change_left", "lane_change_right"}:
                selected_phase = str(
                    selected_candidate_debug.get("lane_change_phase", "")
                ).strip().lower()
                lc_state = self._normalized_final_lc_state(
                    decision=str(decision),
                    lc_state=str(lc_state),
                    lane_change_phase=str(selected_phase),
                )
            reference_debug["candidate_pipeline_enabled"] = True
            reference_debug["turn_prepare_speed_suppressed_by_lane_change"] = bool(
                turn_prepare_speed_suppressed_by_lane_change
            )
        else:
            reference_debug["candidate_pipeline_enabled"] = False

        boundary_recovery_active = bool(
            self.config.get("boundary_recovery_enabled", False)
        ) and bool(
            getattr(
                scenario_decision,
                "boundary_recovery_active",
                False,
            )
        )
        if bool(boundary_recovery_active):
            generated_recovery = self.reference_generator.build_boundary_recovery(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_lane_id=int(current_lane_id),
                base_reference_samples=local_lane_center_reference,
                target_speed_mps=float(speed_plan.target_speed_mps),
                horizon_steps=int(self.mpc.horizon_steps),
                dt_s=float(self.mpc.dt_s),
            )
            recovery_reference = [
                dict(sample)
                for sample in list(generated_recovery.samples or [])
            ]
            recovery_conditioning_reason = ""
            if recovery_reference:
                (
                    recovery_reference,
                    recovery_conditioning_reason,
                ) = self.reference_generator.curvature_feasible_samples(
                    reference_samples=recovery_reference,
                    ego_location=ego_location,
                    ego_heading_rad=float(ego_yaw_rad),
                    max_curvature_1pm=float(
                        self.config.get(
                            "boundary_recovery_max_curvature_1pm",
                            self.config.get(
                                "reference_vehicle_max_curvature_1pm",
                                0.22,
                            ),
                        )
                    ),
                    mode="boundary_recovery",
                )
            if recovery_reference:
                speed_ref_mps = float(speed_plan.target_speed_mps)
                for sample in recovery_reference:
                    sample["speed_ref_mps"] = float(speed_ref_mps)
                    sample["v_ref_mps"] = float(speed_ref_mps)
                    sample["speed_mps"] = float(speed_ref_mps)
                    sample["reference_mode"] = "boundary_recovery"
                local_lane_center_reference = list(recovery_reference)
                terminal = local_lane_center_reference[-1]
                self._temporary_destination_state = [
                    float(terminal.get("x_ref_m", terminal.get("x", ego_location.x))),
                    float(terminal.get("y_ref_m", terminal.get("y", ego_location.y))),
                    float(speed_ref_mps),
                    float(terminal.get("heading_rad", ego_yaw_rad)),
                    int(terminal.get("lane_id", current_lane_id) or current_lane_id),
                ]
                reference_debug.update({
                    "reference_source": "ego_anchored_boundary_recovery",
                    "final_reference_geometry_source": (
                        "ego_anchored_boundary_recovery"
                    ),
                    "stage": "boundary_recovery_reference",
                    "intent_mode": "boundary_recovery",
                    "boundary_recovery_active": True,
                    "boundary_recovery_generation_reason": str(
                        generated_recovery.reason
                    ),
                    "boundary_recovery_conditioning_reason": str(
                        recovery_conditioning_reason
                    ),
                })
            else:
                reference_debug.update({
                    "boundary_recovery_active": True,
                    "boundary_recovery_generation_reason": str(
                        generated_recovery.reason
                    ),
                    "candidate_pipeline_selected_status": "infeasible",
                    "candidate_pipeline_selected_reason": (
                        "boundary_recovery_reference_generation_failed:"
                        + str(generated_recovery.reason)
                    ),
                })
        # MPC is downstream of candidate selection. Its objective profile must
        # describe the maneuver that will actually execute, not the behavior
        # proposal that existed before candidate arbitration.
        self._apply_mpc_cost_profile(
            behavior=str(decision),
            planner_lc_state=str(lc_state),
            planner_mode=str(planner_mode),
            next_macro_maneuver=str(
                planner_input_frame.planning.route.next_macro_maneuver
            ),
            sim_time_s=float(sim_time_s),
            nearest_obstacle_distance_m=(
                float(front_gap_m)
                if front_gap_m is not None and math.isfinite(float(front_gap_m))
                else None
            ),
            ego_speed_mps=float(ego_speed_mps),
        )
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
                strict_reference = self.reference_generator.route_aligned_samples(
                    ego_location=ego_location,
                    ego_heading_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                    route_points=route_points,
                )
                strict_reference_source = "global_route_aligned_lane_follow"
            if not strict_reference:
                strict_reference = self.reference_generator.lane_center_samples(
                    start_waypoint=ego_waypoint,
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                    route_points=route_points,
                )
            if self.reference_generator.reference_opposes_heading(
                reference_samples=strict_reference,
                ego_heading_rad=float(ego_yaw_rad),
                max_heading_error_rad=0.5 * math.pi,
            ) or self.reference_generator.reference_lateral_offset_too_large(
                reference_samples=strict_reference,
                ego_state=current_state,
                max_lateral_offset_m=float(
                    self.config.get("lane_follow_reference_max_initial_lateral_m", 1.75)
                ),
            ):
                route_reference = self.reference_generator.route_aligned_samples(
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
        committed_lane_change_reference_active = bool(
            str(decision) in {"lane_change_left", "lane_change_right"}
            and self._route_tracking_lane_change_reference
        )
        lateral_guard_reason = ""
        if (
            not bool(committed_lane_change_reference_active)
            and not bool(boundary_recovery_active)
        ):
            lateral_guard_reason = self._full_reference_lateral_guard_reason(
                decision=str(decision),
                lc_state=str(lc_state),
                stop_goal_active=bool(stop_goal_active),
                destination_state=self._temporary_destination_state,
                lane_center_reference=local_lane_center_reference,
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                heading_error_rad=(
                    float(behavior_lane_heading_error_rad)
                    if bool(behavior_lane_alignment_valid)
                    else float("nan")
                ),
            )
        if str(lateral_guard_reason):
            # Rebuild from the route-owned corridor first. At a junction,
            # CARLA may renumber the lane and expose several valid next()
            # branches; independently walking the current lane center here can
            # conflict with the already-authorized GRP branch.
            step_distance_m = max(
                0.5,
                float(self.mpc.dt_s)
                * max(1.0, min(float(speed_ref_mps), float(self.target_speed_mps))),
            )
            guarded_reference, guarded_route_reason = (
                self.route_manager.carla_waypoint_reference(
                    ego_x_m=float(ego_location.x),
                    ego_y_m=float(ego_location.y),
                    ego_heading_rad=float(ego_yaw_rad),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                    target_speed_mps=float(speed_ref_mps),
                    fallback_lane_id=int(current_lane_id),
                    anchor_to_ego_heading=False,
                )
            )
            guarded_reference_source = "carla_grp_lateral_guard"
            if not guarded_reference:
                guarded_reference = self.reference_generator.lane_center_samples(
                    start_waypoint=ego_waypoint,
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(step_distance_m),
                    route_points=route_points,
                )
                guarded_reference_source = "current_lane_center_lateral_guard"
                guarded_route_reason = "carla_route_reference_unavailable"
            if guarded_reference:
                from cpx_planning.behavior_planner.reference_pipeline import (
                    lane_center_destination_from_reference,
                )

                local_lane_center_reference = list(guarded_reference)
                target_forward_m = float(
                    self.config.get(
                        "full_stop_guard_destination_forward_m"
                        if bool(stop_goal_active)
                        else "full_lane_follow_guard_destination_forward_m",
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
                reference_debug["reference_source"] = str(
                    guarded_reference_source
                )
                reference_debug["junction_connector_reason"] = str(
                    guarded_route_reason
                )
                reference_debug["fallback_reason"] = (
                    f"{reference_debug.get('fallback_reason', '')}:"
                    if str(reference_debug.get("fallback_reason", ""))
                    else ""
                ) + str(lateral_guard_reason)
                reference_debug["stage"] = "lateral_guard"
                reference_debug["reference_pipeline_follow_global_route_lane"] = int(
                    str(guarded_reference_source) == "carla_grp_lateral_guard"
                )
            elif bool(self.strict_reference_validator_veto_enabled):
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
                ) + "strict_reference_veto:lateral_guard_rebuild_failed:" + str(lateral_guard_reason)
        reference_debug["reference_lateral_guard_reason"] = str(lateral_guard_reason)
        reference_debug["opencda_style_reference_conditioning_reason"] = ""
        raw_geometry_source = str(
            reference_debug.get(
                "final_reference_geometry_source",
                reference_debug.get("reference_source", ""),
            )
        )
        maneuver_reference = self.maneuver_manager.update(
            reference_samples=local_lane_center_reference,
            destination_state=self._temporary_destination_state,
            decision=str(decision),
            behavior_fsm_state=str(lc_state),
            current_lane_id=int(current_lane_id),
            target_lane_id=int(target_lane_id),
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            reference_source=str(raw_geometry_source),
            route_current_option=str(route_context.current_road_option),
            route_next_maneuver=str(route_context.next_macro_maneuver),
            stop_goal_active=bool(stop_goal_active),
            lane_change_commitment_active=bool(
                self._route_tracking_lane_change_reference
            ),
        )
        local_lane_center_reference = list(
            maneuver_reference.reference_samples
        )
        self._temporary_destination_state = list(
            maneuver_reference.destination_state
        )
        reference_debug.update(maneuver_reference.debug)
        if bool(maneuver_reference.debug.get("maneuver_geometry_active", False)):
            reference_debug["pre_maneuver_geometry_source"] = str(
                raw_geometry_source
            )
            reference_debug["final_reference_geometry_source"] = (
                "unified_maneuver_geometry"
            )
            reference_debug["reference_source"] = (
                "unified_maneuver_geometry"
            )
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
                "traffic_signal_raw_state": str(
                    planner_input_frame.planning.traffic_control.signal_state
                ),
                "traffic_signal_resolved_state": str(resolved_traffic_state),
                "traffic_signal_filtered_state": str(filtered_traffic_state),
                "traffic_signal_behavior_state": str(behavior_traffic_state),
                "traffic_control_from_cp": bool(planner_input_frame.planning.traffic_control.from_cp),
                "stop_goal_active": bool(stop_goal_active),
                "target_speed_mps": float(speed_ref_mps),
                "boundary_recovery_active": bool(
                    boundary_recovery_active
                ),
                "boundary_recovery_scenario_state": str(
                    scenario_decision.state
                ),
                "stop_target": (
                    dict(behavior_stop_target)
                    if isinstance(behavior_stop_target, Mapping)
                    else {}
                ),
            },
            reference_debug,
        )

    def _build_route_tracking_baseline_plan(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        ego_speed_mps: float,
        current_state: Sequence[float],
        current_lane_id: int,
        object_snapshots: Sequence[Mapping[str, object]],
        raw_front_stop_active: bool,
        scenario_decision: Any,
        raw_traffic_state: str,
        resolved_traffic_state: str,
        filtered_traffic_state: str,
        traffic_control_from_cp: bool,
        behavior_stop_target: Mapping[str, object] | None,
        route_context: Any,
        route_reference_allowed: bool,
        route_reference_gate_reason: str,
        upcoming_turn_direction: str,
        upcoming_turn_distance_m: float,
        adapter_output: Any,
        planner_input_frame: Any,
    ):
        """Build the minimal route -> speed profile -> MPC reference chain."""

        requested_speed_mps = max(0.0, float(self.target_speed_mps))
        if scenario_decision.speed_cap_mps is not None:
            requested_speed_mps = min(
                float(requested_speed_mps),
                max(0.0, float(scenario_decision.speed_cap_mps)),
            )

        traffic_stop_active = bool(scenario_decision.stop_goal_active)
        obstacle_stop_active = bool(
            raw_front_stop_active and not traffic_stop_active
        )
        stop_goal_active = bool(traffic_stop_active or obstacle_stop_active)
        stop_target = (
            dict(behavior_stop_target)
            if isinstance(behavior_stop_target, Mapping)
            else None
        )
        stop_reason = ""
        if bool(traffic_stop_active):
            stop_reason = "traffic_control_stop"
        elif bool(obstacle_stop_active):
            front_gap_m = self._front_gap_m(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                object_snapshots=object_snapshots,
            )
            if front_gap_m is not None:
                additional_buffer_m = max(
                    0.0,
                    float(
                        self.config.get(
                            "route_tracking_obstacle_additional_buffer_m",
                            3.0,
                        )
                    ),
                )
                target_forward_m = max(
                    0.5,
                    float(front_gap_m) - float(additional_buffer_m),
                )
                stop_target = {
                    "active": True,
                    "x_m": float(ego_location.x)
                    + float(target_forward_m) * math.cos(float(ego_yaw_rad)),
                    "y_m": float(ego_location.y)
                    + float(target_forward_m) * math.sin(float(ego_yaw_rad)),
                    "source": "route_tracking_front_obstacle",
                }
            stop_reason = "front_obstacle_stop"

        speed_ref_mps = 0.0 if bool(stop_goal_active) else float(requested_speed_mps)
        step_distance_m = max(
            float(self.config.get("route_tracking_min_step_m", 0.10)),
            float(self.mpc.dt_s)
            * max(0.5, float(ego_speed_mps), float(requested_speed_mps)),
        )
        reference, route_reference_reason = self.route_manager.carla_waypoint_reference(
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
            horizon_steps=int(self.mpc.horizon_steps),
            step_distance_m=float(step_distance_m),
            target_speed_mps=float(speed_ref_mps),
            fallback_lane_id=int(current_lane_id),
            # The GRP chain is connected once when the route is built. Keep
            # that immutable corridor through junction lane-id transitions;
            # small offsets use the bounded route-rejoin path below.
            anchor_to_ego_heading=False,
        )
        normalized_route_option = str(
            route_context.current_road_option or ""
        ).strip().upper().replace("_", "")
        if (
            self._route_tracking_lane_change_completed_option
            and normalized_route_option
            != self._route_tracking_lane_change_completed_option
        ):
            self._route_tracking_lane_change_completed_option = ""
        completion_reason = self._release_completed_lane_change_commitment(
            current_lane_id=int(current_lane_id),
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
        )
        if str(completion_reason):
            route_reference_reason += ";" + str(completion_reason)
        route_lane_change_requested = bool(
            normalized_route_option
            in {"CHANGELANELEFT", "CHANGELANERIGHT"}
            and normalized_route_option
            != self._route_tracking_lane_change_completed_option
        )
        locked_lane_change_incomplete = bool(
            self._route_tracking_lane_change_reference
        )
        route_lane_change_active = bool(
            not bool(stop_goal_active)
            and (
                bool(route_lane_change_requested)
                or bool(locked_lane_change_incomplete)
            )
        )
        active_route_option = (
            str(normalized_route_option)
            if bool(route_lane_change_requested)
            else str(self._route_tracking_lane_change_option)
        )
        lane_change_direction = (
            "left"
            if active_route_option == "CHANGELANELEFT"
            else "right"
            if active_route_option == "CHANGELANERIGHT"
            else ""
        )
        route_recovery_active = False
        route_recovery_reason = ""
        if bool(route_lane_change_active):
            if (
                bool(route_lane_change_requested)
                and str(self._route_tracking_lane_change_option)
                != str(normalized_route_option)
                or not self._route_tracking_lane_change_reference
            ):
                build_reason = self._lock_route_tracking_lane_change_reference(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    target_lane_id=int(route_context.optimal_lane_id),
                    route_option=str(active_route_option),
                    target_speed_mps=float(speed_ref_mps),
                    step_distance_m=float(step_distance_m),
                )
                route_reference_reason += ";" + str(build_reason)
            locked_reference, locked_reason = (
                self._route_tracking_lane_change_window(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    target_speed_mps=float(speed_ref_mps),
                    step_distance_m=float(step_distance_m),
                )
            )
            if locked_reference:
                reference = [dict(sample) for sample in locked_reference]
            route_reference_reason += ";" + str(locked_reason)
            valid, validation_reason = (
                self._validate_route_tracking_lane_change_reference(
                    reference=reference,
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                )
            )
            locked_reference_available = bool(
                locked_reference
                and self._route_tracking_lane_change_reference
            )
            if not bool(locked_reference_available):
                validation_reason = (
                    "lane_change_validation:missing_locked_trajectory;"
                    + str(validation_reason)
                )
            if not bool(valid) or not bool(locked_reference_available):
                route_recovery_active = True
                route_recovery_reason = str(validation_reason)
                recovery_reference, recovery_reason = (
                    self.route_manager.carla_waypoint_reference(
                        ego_x_m=float(ego_location.x),
                        ego_y_m=float(ego_location.y),
                        ego_heading_rad=float(ego_yaw_rad),
                        horizon_steps=int(self.mpc.horizon_steps),
                        step_distance_m=float(step_distance_m),
                        target_speed_mps=min(
                            float(speed_ref_mps),
                            float(
                                self.config.get(
                                    "route_tracking_recovery_speed_mps",
                                    1.0,
                                )
                            ),
                        ),
                        fallback_lane_id=int(current_lane_id),
                        anchor_to_ego_heading=True,
                        ego_anchor_distance_m=float(
                            self.config.get(
                                "route_tracking_recovery_rejoin_distance_m",
                                10.0,
                            )
                        ),
                    )
                )
                if recovery_reference:
                    reference = [
                        dict(sample) for sample in recovery_reference
                    ]
                else:
                    generated_fallback = self.reference_generator.build_lane_fallback(
                        ego_location=ego_location,
                        ego_yaw_rad=float(ego_yaw_rad),
                        current_state=current_state,
                        speed_ref_mps=float(
                            self.config.get(
                                "route_tracking_recovery_speed_mps",
                                1.0,
                            )
                        ),
                    )
                    reference = generated_fallback.samples
                speed_ref_mps = min(
                    float(speed_ref_mps),
                    float(
                        self.config.get(
                            "route_tracking_recovery_speed_mps",
                            1.0,
                        )
                    ),
                )
                route_reference_reason += (
                    ";route_tracking_recovery:"
                    + str(validation_reason)
                    + ":"
                    + str(recovery_reason)
                )
        elif not bool(stop_goal_active):
            self._reset_route_tracking_lane_change_reference()
        if not reference:
            generated_fallback = self.reference_generator.build_lane_fallback(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_state=current_state,
                speed_ref_mps=float(speed_ref_mps),
            )
            reference = generated_fallback.samples
            destination = generated_fallback.destination_state
            route_reference_reason = (
                "route_tracking_current_lane_fallback:"
                + str(route_reference_reason)
            )
        else:
            from cpx_planning.behavior_planner.reference_pipeline import (
                lane_center_destination_from_reference_arc_length,
            )

            terminal = dict(reference[-1])
            initial_destination = [
                float(terminal.get("x_ref_m", terminal.get("x", ego_location.x))),
                float(terminal.get("y_ref_m", terminal.get("y", ego_location.y))),
                float(speed_ref_mps),
                float(terminal.get("heading_rad", ego_yaw_rad)),
                int(terminal.get("lane_id", current_lane_id) or current_lane_id),
            ]
            reachable_arc_m = max(
                0.5,
                min(
                    float(
                        self.config.get(
                            "route_tracking_destination_arc_m",
                            8.0,
                        )
                    ),
                    float(self.mpc.horizon_s)
                    * max(0.5, float(speed_ref_mps))
                    * float(
                        self.config.get(
                            "route_tracking_destination_horizon_fraction",
                            0.9,
                        )
                    ),
                ),
            )
            destination = lane_center_destination_from_reference_arc_length(
                destination_state=initial_destination,
                lane_center_reference=reference,
                target_arc_length_m=float(reachable_arc_m),
            )
            if destination is None:
                destination = list(initial_destination)

        if bool(stop_goal_active):
            generated_stop = self.reference_generator.stop_reference(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_state=current_state,
                current_lane_id=int(current_lane_id),
                stop_target=stop_target,
                fallback_destination_state=destination,
                ego_speed_mps=float(ego_speed_mps),
            )
            reference = generated_stop.samples
            destination = generated_stop.destination_state
            route_reference_reason += ";" + str(generated_stop.reason)

        turn_activation_distance_m = float(
            self.config.get("route_tracking_turn_activation_distance_m", 18.0)
        )
        turn_active = bool(
            str(upcoming_turn_direction) in {"left", "right"}
            and math.isfinite(float(upcoming_turn_distance_m))
            and float(upcoming_turn_distance_m) <= float(turn_activation_distance_m)
        )
        decision = (
            "route_recovery"
            if bool(route_recovery_active)
            else f"lane_change_{str(lane_change_direction)}"
            if bool(route_lane_change_active)
            else f"intersection_turn_{str(upcoming_turn_direction)}"
            if bool(turn_active) and not bool(stop_goal_active)
            else "stop_at_intersection"
            if bool(stop_goal_active)
            else "lane_follow"
        )
        lc_state = (
            "LANE_KEEP"
            if bool(stop_goal_active)
            else "ROUTE_RECOVERY"
            if bool(route_recovery_active)
            else "TARGET_LANE_STABILIZATION"
            if (
                bool(route_lane_change_active)
                and str(
                    getattr(
                        self,
                        "_route_tracking_lane_change_phase",
                        "executing",
                    )
                )
                == "target_lane_stabilization"
            )
            else "ROUTE_TRACKING_LANE_CHANGE"
            if bool(route_lane_change_active)
            else "ROUTE_TRACKING"
        )
        planner_mode = "INTERSECTION" if bool(turn_active) else "NORMAL"
        self._apply_mpc_cost_profile(
            behavior=str(decision),
            planner_lc_state=str(lc_state),
            planner_mode=str(planner_mode),
            next_macro_maneuver=str(route_context.next_macro_maneuver),
            sim_time_s=float(self._sim_time_s()),
        )

        behavior_debug = {
            "decision": str(decision),
            "lc_state": str(lc_state),
            "target_lane_id": (
                int(self._route_tracking_lane_change_target_lane_id)
                if bool(route_lane_change_active)
                else int(current_lane_id)
            ),
            "current_lane_id": int(current_lane_id),
            "lane_safety_scores": dict(adapter_output.lane_safety_scores),
            "traffic_signal_state": str(
                scenario_decision.behavior_signal_state
            ),
            "traffic_signal_raw_state": str(raw_traffic_state),
            "traffic_signal_resolved_state": str(resolved_traffic_state),
            "traffic_signal_filtered_state": str(filtered_traffic_state),
            "traffic_signal_behavior_state": str(
                scenario_decision.behavior_signal_state
            ),
            "traffic_control_from_cp": bool(traffic_control_from_cp),
            "stop_goal_active": bool(stop_goal_active),
            "target_speed_mps": float(speed_ref_mps),
            "stop_target": dict(stop_target or {}),
        }
        reference_debug = {
            "stage": "route_tracking_baseline",
            "intent_mode": "route_tracking",
            "reference_source": (
                "carla_route_recovery_reference"
                if bool(route_recovery_active)
                else "target_lane_stabilization_reference"
                if (
                    bool(route_lane_change_active)
                    and str(
                        getattr(
                            self,
                            "_route_tracking_lane_change_phase",
                            "executing",
                        )
                    )
                    == "target_lane_stabilization"
                )
                else "locked_quintic_lane_change_reference"
                if bool(route_lane_change_active)
                else "carla_grp_waypoint_route_baseline"
            ),
            "final_reference_geometry_source": (
                "carla_route_recovery_reference"
                if bool(route_recovery_active)
                else "target_lane_stabilization_reference"
                if (
                    bool(route_lane_change_active)
                    and str(
                        getattr(
                            self,
                            "_route_tracking_lane_change_phase",
                            "executing",
                        )
                    )
                    == "target_lane_stabilization"
                )
                else "locked_quintic_lane_change_reference"
                if bool(route_lane_change_active)
                else "carla_grp_waypoint_route_baseline"
            ),
            "fallback_reason": str(route_reference_reason),
            "behavior_override_reason": str(stop_reason),
            "route_reference_allowed": bool(route_reference_allowed),
            "route_reference_gate_reason": str(route_reference_gate_reason),
            "route_lane_change_allowed": bool(
                route_lane_change_active and not route_recovery_active
            ),
            "opportunistic_lane_change_allowed": False,
            "lane_change_gate_reason": (
                "route_recovery:" + str(route_recovery_reason)
                if bool(route_recovery_active)
                else "route_triggered_locked_trajectory"
                if bool(route_lane_change_active)
                else "no_active_route_lane_change"
            ),
            "route_lane_change_required": bool(route_lane_change_active),
            "lane_change_authorized": bool(
                route_lane_change_active and not route_recovery_active
            ),
            "lane_change_authorization_direction": str(
                lane_change_direction
            ),
            "lane_change_authorization_reason": (
                "global_route_trigger_then_locked_quintic"
                if bool(route_lane_change_active)
                else ""
            ),
            "lane_change_required_by_route": bool(route_lane_change_active),
            "lane_change_authorized_target_lane_id": (
                int(self._route_tracking_lane_change_target_lane_id)
                if bool(route_lane_change_active)
                else int(current_lane_id)
            ),
            "route_current_road_option": str(route_context.current_road_option),
            "route_next_macro_maneuver": str(route_context.next_macro_maneuver),
            "candidate_pipeline_enabled": False,
            "candidate_pipeline_selected": "route_tracking_baseline",
            "candidate_pipeline_selected_status": (
                "route_recovery"
                if bool(route_recovery_active)
                else "feasible"
            ),
            "candidate_pipeline_selected_reason": (
                str(route_recovery_reason)
                if bool(route_recovery_active)
                else str(stop_reason)
            ),
            "candidate_selected_stop_goal_active": bool(stop_goal_active),
            "lane_change_trajectory_variant": (
                "route_recovery"
                if bool(route_recovery_active)
                else "quintic_route_tracking"
                if bool(route_lane_change_active)
                else ""
            ),
            "lane_change_duration_s": (
                float(
                    self._route_tracking_lane_change_resolved_duration_s
                    or self.config.get(
                        "route_tracking_lane_change_duration_s",
                        4.0,
                    )
                )
                if bool(route_lane_change_active)
                else 0.0
            ),
            "lane_change_duration_comfort_reason": (
                str(self._route_tracking_lane_change_duration_comfort_reason)
                if bool(route_lane_change_active)
                else ""
            ),
            "lane_change_initial_progress": (
                float(self._route_tracking_lane_change_progress)
                if bool(route_lane_change_active)
                else 0.0
            ),
            "lane_change_terminal_progress": (
                float(reference[-1].get("lane_change_progress", 0.0))
                if bool(route_lane_change_active) and reference
                else 0.0
            ),
            "route_tracking_lane_change_locked": bool(
                self._route_tracking_lane_change_reference
            ),
            "route_tracking_lane_change_progress_index": int(
                self._route_tracking_lane_change_progress_index
            ),
            "route_tracking_lane_change_source_lane_id": int(
                self._route_tracking_lane_change_source_lane_id
            ),
            "route_tracking_lane_change_target_lane_id": int(
                self._route_tracking_lane_change_target_lane_id
            ),
            "route_tracking_recovery_active": bool(route_recovery_active),
            "route_tracking_recovery_reason": str(route_recovery_reason),
            "planner_input_cp_traffic_control_count": len(
                list(planner_input_frame.cp_messages.traffic_controls)
            ),
            "planner_input_prediction_risky_lane_count": sum(
                bool(dict(risk).get("risk", False))
                for risk in dict(
                    planner_input_frame.prediction.lane_prediction_risks
                ).values()
                if isinstance(risk, Mapping)
            ),
            "planner_input_perception_planning_count": len(
                list(planner_input_frame.perception.planning_objects)
            ),
            "planner_input_cp_obstacle_count": len(
                list(planner_input_frame.cp_messages.obstacles)
            ),
            "planner_input_frame_timestamp_s": float(
                planner_input_frame.planning.sim_time_s
            ),
            "reference_pipeline_follow_global_route_lane": (
                0
                if bool(route_lane_change_active)
                or bool(route_recovery_active)
                else 1
            ),
        }
        return (
            list(destination),
            [dict(sample) for sample in list(reference or [])],
            behavior_debug,
            reference_debug,
        )

    def _cooperative_lane_change_yield_reason(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
    ) -> str:
        """Hold in lane if a nearby CPX-controlled peer is already mid-lane-change.

        Both CAVs independently deciding to change lanes at the same moment
        near each other is exactly the situation that produced the
        multi-CAV mutual-interference gridlock diagnosed in Construction_Zone
        testing. Serialize on physical order instead: whichever CAV is
        already committed to a lane change goes first; a trailing peer holds
        lane_follow until that commitment clears (state resets to IDLE, which
        stops being broadcast as active -- see ManeuverCommitment.active).

        Thin wrapper around the shared cooperative_arbitration module: any
        active peer lane-change conflicts with ego's own (resource_id is a
        constant, not the specific lane, since two CAVs changing lanes near
        each other at the same time is the thing being serialized,
        regardless of which lanes are involved).
        """
        if not bool(
            self.config.get("cooperative_lane_change_yield_enabled", True)
        ):
            return ""
        v2x_manager = getattr(self.vehicle_manager, "v2x_manager", None)
        cav_intents = dict(getattr(v2x_manager, "cav_intents", {}) or {})
        cav_nearby = dict(getattr(v2x_manager, "cav_nearby", {}) or {})
        if not cav_intents or not cav_nearby:
            return ""
        from cpx_planning.pipeline.cooperative_arbitration import (
            ResourceClaim,
            should_yield,
        )

        peers: list[tuple[int, ResourceClaim, tuple[float, float]]] = []
        for peer_id, message in cav_intents.items():
            if not isinstance(message, Mapping):
                continue
            decision = str(message.get("maneuver_commitment_decision", ""))
            if decision not in ("lane_change_left", "lane_change_right"):
                continue
            peer_manager = cav_nearby.get(str(peer_id))
            peer_vehicle = getattr(peer_manager, "vehicle", None)
            if peer_vehicle is None:
                continue
            try:
                peer_location = peer_vehicle.get_location()
            except Exception:
                continue
            try:
                peer_actor_id = int(peer_id)
            except (TypeError, ValueError):
                continue
            peers.append((
                peer_actor_id,
                ResourceClaim(
                    kind="lane_change",
                    resource_id="lane_change",
                    committed_at_s=float(
                        message.get("maneuver_commitment_committed_at_s", 0.0) or 0.0
                    ),
                    active=bool(message.get("maneuver_commitment_active", False)),
                ),
                (float(peer_location.x), float(peer_location.y)),
            ))
        if not peers:
            return ""
        my_claim = ResourceClaim(
            kind="lane_change",
            resource_id="lane_change",
            committed_at_s=float(self._sim_time_s()),
            active=True,
        )
        my_actor_id = int(self.config.get("ego_actor_id", -1))
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=my_actor_id,
            my_position_xy=(float(ego_location.x), float(ego_location.y)),
            my_heading_rad=float(ego_yaw_rad),
            peers=peers,
            range_m=float(
                self.config.get("cooperative_lane_change_yield_range_m", 40.0)
            ),
        )
        return str(reason) if reason else ""

    def _cooperative_wait_speed_cap_mps(
        self,
        *,
        ego_location: PlannerLocation,
        ego_speed_mps: float,
        cooperative_lane_change_yield_reason: str,
    ) -> Optional[float]:
        """Cap speed while queued behind a peer's lane change.

        ``_cooperative_lane_change_yield_reason`` already holds ego's own
        lane change back until the peer clears -- necessary but not
        sufficient. That peer is normally in an ADJACENT lane, outside
        ego's own-lane ``_front_gap_m`` search cone, so the ordinary
        following-cap in speed_planner.py never sees it and has no reason
        to slow down for it. Left unconstrained, ego keeps accelerating
        toward its full cruise target while waiting, closes the real gap
        to the peer it intends to merge behind, and by the time its own
        turn opens up the gap has fallen under trajectory_risk.py's
        min_front_gap_m -- so the now-authorized lane change gets denied
        by target_lane_prediction_risk and is missed once the route's own
        lane-change requirement lapses (diagnosed via Interactive_Lane_
        Change telemetry: gap fell from ~8.3m to ~6.5m across the wait
        window). This does not touch that prediction-risk check at all;
        it just stops ego from closing the gap in the first place while
        it has nowhere to go yet.

        The trigger distance is deliberately larger than trajectory_risk.
        py's own min_front_gap_m (8.0m default): reusing that exact value
        here gave this cap zero lead time -- telemetry showed the yield
        reason (and therefore this function) only ever starts firing once
        the gap has *already* dropped to ~7.9m, one tick past the hard
        floor, so there was never a tick left where capping ego's speed
        could still have prevented the gap sliding on down to ~6.5-6.9m
        and tripping target_lane_prediction_risk. A separate, wider
        trigger (cooperative_wait_trigger_gap_m, default 15.0m) gives the
        cap several seconds of runway to hold ego at the peer's speed
        before the hard threshold is anywhere close.
        """
        if not cooperative_lane_change_yield_reason:
            return None
        match = re.search(r"peer=(-?\d+)", cooperative_lane_change_yield_reason)
        if match is None:
            return None
        peer_id = match.group(1)
        v2x_manager = getattr(self.vehicle_manager, "v2x_manager", None)
        cav_nearby = dict(getattr(v2x_manager, "cav_nearby", {}) or {})
        peer_manager = cav_nearby.get(str(peer_id))
        peer_vehicle = getattr(peer_manager, "vehicle", None)
        if peer_vehicle is None:
            return None
        try:
            peer_location = peer_vehicle.get_location()
            peer_velocity = peer_vehicle.get_velocity()
        except Exception:
            return None
        peer_speed_mps = math.sqrt(
            float(peer_velocity.x) ** 2
            + float(peer_velocity.y) ** 2
            + float(peer_velocity.z) ** 2
        )
        distance_m = math.hypot(
            float(peer_location.x) - float(ego_location.x),
            float(peer_location.y) - float(ego_location.y),
        )
        # Both floors below are flat distances that don't scale with
        # cruise speed -- also give them the same reaction-time margin
        # regardless of configured cruise speed, matching min_front_gap_m's
        # own speed scaling in planner_input_adapter.py. Scaled off the
        # *configured* cruise target (self.target_speed_mps), not ego's
        # live instantaneous speed -- this wait window happens while ego
        # is still mid-acceleration toward that target, so scaling off
        # the live speed barely moved either floor at the moment it
        # mattered (confirmed via telemetry: identical denial, identical
        # distances down to the decimal, before and after that version).
        min_gap_m = max(
            0.5,
            float(self.target_speed_mps) * float(self.min_front_gap_time_s),
            float(self.config.get("cooperative_wait_min_gap_m", 8.0)),
        )
        trigger_gap_m = max(
            float(min_gap_m),
            float(self.target_speed_mps)
            * float(self.config.get("cooperative_wait_trigger_time_s", 15.0 / 11.18)),
            float(self.config.get("cooperative_wait_trigger_gap_m", 15.0)),
        )
        if float(distance_m) >= float(trigger_gap_m):
            return None
        # min(ego_speed, peer_speed) was the original cap here, but it does
        # nothing when both CAVs are ramping up toward the same cruise
        # target in near lockstep from a similar start (confirmed via
        # telemetry at 20 m/s cruise: peer's speed tracked ego's own climb
        # tick-for-tick, ~7->11 m/s over the same 2s window, so "cap at
        # peer's speed" never actually differed from where ego was already
        # headed -- three separate threshold-tuning attempts on the
        # trigger/min-gap distances above produced bit-identical
        # trajectories because the actual constraining value never
        # changed). Reuse the same IDM model used for ordinary front-
        # vehicle following instead: it reacts to the actual gap being
        # smaller than the desired safe spacing even when closing speed is
        # ~0, which a plain speed-match can't express.
        from cpx_planning.behavior_planner.car_follow import (
            idm_acceleration as _cooperative_wait_idm_acceleration,
        )

        idm_accel = _cooperative_wait_idm_acceleration(
            v=float(ego_speed_mps),
            v_lead=max(0.0, float(peer_speed_mps)),
            gap_m=max(0.1, float(distance_m)),
            v_desired=max(0.1, float(self.target_speed_mps)),
            a_max=max(
                0.05,
                float(self.config.get("following_idm_max_acceleration_mps2", 2.0)),
            ),
            b_comfort=max(
                0.05,
                float(
                    self.config.get(
                        "following_idm_comfort_deceleration_mps2", 3.0
                    )
                ),
            ),
            time_headway_s=max(
                0.05, float(self.config.get("following_time_headway_s", 1.5))
            ),
            min_gap_m=float(min_gap_m),
            delta=max(
                1.0, float(self.config.get("following_idm_acceleration_exponent", 4.0))
            ),
        )
        cap_horizon_s = max(
            0.05, float(self.config.get("cooperative_wait_cap_horizon_s", 1.0))
        )
        return max(0.0, float(ego_speed_mps) + float(idm_accel) * float(cap_horizon_s))

    def _cooperative_avoidance_lane_yield_reason(
        self,
        *,
        target_lane_id: int,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
    ) -> str:
        """Hold back if a peer CAV already claimed this exact avoidance lane.

        Construction_Zone testing with every CAV controlled surfaced a
        multi-CAV gridlock: several CAVs converge on the same one or two
        usable bypass lanes at once, so each one's lane_safety_scores for
        that lane stays low (correctly -- a peer really is right there) and
        nobody ever moves, forever, since nothing breaks the symmetry.
        Rather than blind the safety scorer to peer CAVs (a peer stopped in
        your target lane is a real hazard, CAV or not), arbitrate who is
        even allowed to attempt this specific lane: whichever CAV has been
        blocked by its obstacle the longest goes first (a reasonable stand-in
        for "committed first", since local-avoidance commitment itself is
        decided in the same step this reads); the rest hold in place and
        re-check every tick, so as soon as the leader clears the lane (moves
        through, or its own commitment resets) the next one in line takes
        its turn instead of everyone staying wedged forever.
        """
        if not bool(
            self.config.get("cooperative_avoidance_lane_yield_enabled", True)
        ):
            return ""
        v2x_manager = getattr(self.vehicle_manager, "v2x_manager", None)
        cav_intents = dict(getattr(v2x_manager, "cav_intents", {}) or {})
        cav_nearby = dict(getattr(v2x_manager, "cav_nearby", {}) or {})
        if not cav_intents or not cav_nearby:
            return ""
        from cpx_planning.pipeline.cooperative_arbitration import (
            ResourceClaim,
            should_yield,
        )

        resource_id = str(int(target_lane_id))
        peers: list[tuple[int, ResourceClaim, tuple[float, float]]] = []
        for peer_id, message in cav_intents.items():
            if not isinstance(message, Mapping):
                continue
            if not bool(message.get("static_obstacle_local_avoidance_active", False)):
                continue
            peer_target_lane_id = message.get("static_obstacle_local_target_lane_id", "")
            if str(peer_target_lane_id) != resource_id:
                continue
            peer_manager = cav_nearby.get(str(peer_id))
            peer_vehicle = getattr(peer_manager, "vehicle", None)
            if peer_vehicle is None:
                continue
            try:
                peer_location = peer_vehicle.get_location()
            except Exception:
                continue
            try:
                peer_actor_id = int(peer_id)
            except (TypeError, ValueError):
                continue
            peers.append((
                peer_actor_id,
                ResourceClaim(
                    kind="avoidance_lane",
                    resource_id=resource_id,
                    committed_at_s=float(
                        message.get("static_obstacle_candidate_since_s", 0.0) or 0.0
                    ),
                    active=True,
                    require_ahead=False,
                ),
                (float(peer_location.x), float(peer_location.y)),
            ))
        if not peers:
            return ""
        my_claim = ResourceClaim(
            kind="avoidance_lane",
            resource_id=resource_id,
            committed_at_s=float(self._static_obstacle_candidate_since_s),
            active=True,
            require_ahead=False,
        )
        my_actor_id = int(self.config.get("ego_actor_id", -1))
        reason = should_yield(
            my_claim=my_claim,
            my_actor_id=my_actor_id,
            my_position_xy=(float(ego_location.x), float(ego_location.y)),
            my_heading_rad=float(ego_yaw_rad),
            peers=peers,
            range_m=float(
                self.config.get("cooperative_avoidance_lane_yield_range_m", 40.0)
            ),
        )
        return str(reason) if reason else ""

    def _lock_route_tracking_lane_change_reference(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        current_lane_id: int,
        target_lane_id: int,
        route_option: str,
        target_speed_mps: float,
        step_distance_m: float,
        duration_s: Optional[float] = None,
        target_waypoint_override: Any = None,
    ) -> str:
        """Generate one fixed source-to-target trajectory for a route lane change."""

        self._reset_route_tracking_lane_change_reference()
        start_waypoint = self._map_waypoint_from_location(ego_location)
        if start_waypoint is None:
            return "lane_change_lock_failed:no_source_waypoint"
        normalized_option = str(route_option or "").strip().upper().replace("_", "")
        adjacent_method = "left" if normalized_option == "CHANGELANELEFT" else "right"
        target_waypoint = target_waypoint_override
        target_reason = (
            "global_route_physical_target_lane"
            if target_waypoint_override is not None
            else "adjacent_lane_center"
        )
        if target_waypoint is None:
            adjacent = getattr(start_waypoint, adjacent_method, None)
            if callable(adjacent):
                try:
                    target_waypoint = adjacent()
                except Exception:
                    target_waypoint = None
        resolved_target_lane_id = int(target_lane_id)
        if target_waypoint is not None:
            try:
                from cpx_planning.utility.global_planner import (
                    canonical_lane_id_for_waypoint,
                )

                resolved_target_lane_id = int(
                    canonical_lane_id_for_waypoint(target_waypoint)
                    or getattr(target_waypoint, "lane_id", target_lane_id)
                )
            except (TypeError, ValueError):
                resolved_target_lane_id = int(target_lane_id)
        if resolved_target_lane_id == int(current_lane_id):
            # The waypoint-canonicalized target collapsed onto ego's current
            # lane -- CARLA's canonical lane numbering is known-lossy right
            # at a lane boundary (see the comment on this near
            # `_route_required_carla_lane_id` usage). Trust the caller's
            # already-resolved target_lane_id over this re-derivation rather
            # than silently locking a self-referencing (target == source)
            # commitment that can never build a real lateral offset and is
            # guaranteed to fail its own reference contract downstream.
            return "lane_change_lock_failed:resolved_target_matches_source"
        resolved_duration_s = max(
            float(self.mpc.dt_s),
            float(duration_s) if duration_s is not None else float(
                self.config.get(
                    "route_tracking_lane_change_duration_s",
                    4.0,
                )
            ),
        )
        from cpx_planning.pipeline.candidate_pipeline import (
            lane_change_geometry_requirements,
        )

        geometry_speed_mps, geometry_length_m, geometry_step_m = (
            lane_change_geometry_requirements(
                ego_speed_mps=float(getattr(self, "_actuator_ego_speed_mps", 0.0)),
                target_speed_mps=float(target_speed_mps),
                duration_s=float(resolved_duration_s),
                dt_s=float(self.mpc.dt_s),
                lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                max_curvature_1pm=float(
                    self.config.get("reference_vehicle_max_curvature_1pm", 0.35)
                ),
                minimum_geometry_speed_mps=float(
                    self.config.get(
                        "lane_change_min_geometry_speed_mps", 2.0
                    )
                ),
                minimum_length_m=float(
                    self.config.get("lane_change_min_length_m", 10.0)
                ),
                acceleration_limit_mps2=float(
                    self.config.get(
                        "lane_change_planning_acceleration_limit_mps2", 2.0
                    )
                ),
            )
        )
        # The caller may request a larger step on a high-speed maneuver, but
        # never allow a low-speed request to compress the spatial quintic
        # below its curvature/minimum-length requirement.
        step_distance_m = max(float(step_distance_m), float(geometry_step_m))
        duration_comfort_check_enabled = bool(
            self.config.get(
                "route_tracking_lane_change_duration_comfort_check_enabled",
                False,
            )
        )
        duration_max_s = max(
            float(resolved_duration_s),
            float(
                self.config.get(
                    "route_tracking_lane_change_duration_max_s",
                    8.0,
                )
            ),
        )
        # Size the fetched source/target arrays for whatever duration the
        # comfort search might settle on -- shape_lane_change_reference can
        # only blend as far as the arrays it's given.
        master_steps_duration_s = (
            float(duration_max_s)
            if bool(duration_comfort_check_enabled)
            else float(resolved_duration_s)
        )
        master_steps = max(
            int(self.mpc.horizon_steps),
            int(math.ceil(float(master_steps_duration_s) / float(self.mpc.dt_s)))
            + int(self.mpc.horizon_steps),
        )
        source_reference = self.reference_generator.lane_center_samples(
            start_waypoint=start_waypoint,
            current_lane_id=int(current_lane_id),
            horizon_steps=int(master_steps),
            step_distance_m=float(step_distance_m),
            route_points=[],
            minimum_step_m=0.05,
            first_point_distance_m=float(step_distance_m),
        )
        target_reference = []
        if target_waypoint is not None:
            target_reference = self.reference_generator.lane_center_samples(
                start_waypoint=target_waypoint,
                current_lane_id=int(resolved_target_lane_id),
                horizon_steps=int(master_steps),
                step_distance_m=float(step_distance_m),
                route_points=[],
                minimum_step_m=0.05,
                first_point_distance_m=float(step_distance_m),
            )
        if not target_reference:
            target_reference, target_reason = (
                self.route_manager.carla_waypoint_reference(
                    ego_x_m=float(ego_location.x),
                    ego_y_m=float(ego_location.y),
                    ego_heading_rad=float(ego_yaw_rad),
                    horizon_steps=int(master_steps),
                    step_distance_m=float(step_distance_m),
                    target_speed_mps=float(target_speed_mps),
                    fallback_lane_id=int(resolved_target_lane_id),
                    anchor_to_ego_heading=False,
                )
            )
        if not source_reference or not target_reference:
            return (
                "lane_change_lock_failed:"
                f"source={len(source_reference)}:"
                f"target={len(target_reference)}:{target_reason}"
            )
        from cpx_planning.pipeline.candidate_pipeline import (
            _align_target_reference_to_source,
            select_comfortable_lane_change_duration_s,
            shape_lane_change_reference,
        )

        direct_target_tracking_enabled = bool(
            self.config.get(
                "route_tracking_lane_change_direct_target_tracking_enabled",
                False,
            )
        )
        road_envelope_enabled = bool(
            self.config.get(
                "route_tracking_lane_change_road_envelope_enabled",
                False,
            )
        )
        if road_envelope_enabled:
            from cpx_planning.pipeline.candidate_pipeline import (
                build_route_tracking_lane_change_envelope_blocks,
            )
            from cpx_planning.MPC.lane_keep import (
                road_envelope_conservativeness_correction,
            )

            self._route_tracking_lane_change_envelope_blocks = (
                build_route_tracking_lane_change_envelope_blocks(
                    source_reference=source_reference,
                    target_reference=target_reference,
                    master_step_count=int(master_steps),
                    step_distance_m=float(step_distance_m),
                    road_boundary_margin_m=float(
                        getattr(self.mpc, "road_boundary_margin_m", 0.5)
                    ),
                    default_lane_width_m=float(
                        getattr(self.mpc, "lane_width_m", 3.5)
                    ),
                )
            )
            self._route_tracking_lane_change_envelope_epsilon0 = (
                road_envelope_conservativeness_correction(
                    self._route_tracking_lane_change_envelope_blocks,
                    rho=float(getattr(self.mpc, "road_envelope_rho", -8.0)),
                )
                if self._route_tracking_lane_change_envelope_blocks
                else 0.0
            )
        else:
            self._route_tracking_lane_change_envelope_blocks = None
            self._route_tracking_lane_change_envelope_epsilon0 = 0.0
        duration_comfort_reason = ""
        if bool(duration_comfort_check_enabled) and not bool(direct_target_tracking_enabled):
            # The comfort loop below only self-checks v^2*curvature against a
            # lateral-accel limit; at low speed that stays "comfortable"
            # well past the raw curvature the hard reference contract will
            # separately enforce for mode="lane_change" (v^2 masks a large
            # curvature). Fetch that same contract limit so the loop can
            # require both, instead of reporting success on a path that
            # then fails downstream as curvature_out_of_contract.
            from cpx_planning.pipeline.reference_contract import (
                contract_from_config,
            )

            lane_change_contract = contract_from_config(
                mode="lane_change",
                expected_lane_id=int(resolved_target_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                config=dict(self.config),
                default_speed_mps=max(float(self.target_speed_mps), float(target_speed_mps), 0.1),
            )
            resolved_duration_s, locked, duration_comfort_reason = (
                select_comfortable_lane_change_duration_s(
                    target_reference=target_reference,
                    source_reference=source_reference,
                    initial_duration_s=float(resolved_duration_s),
                    duration_max_s=float(duration_max_s),
                    dt_s=float(self.mpc.dt_s),
                    current_lane_id=int(current_lane_id),
                    target_lane_id=int(resolved_target_lane_id),
                    target_speed_mps=float(target_speed_mps),
                    lateral_accel_limit_mps2=float(
                        self.config.get(
                            "route_tracking_lane_change_lateral_accel_limit_mps2",
                            1.3,
                        )
                    ),
                    curvature_fn=self.reference_generator.discrete_curvature_1pm,
                    ego_x_m=float(ego_location.x),
                    ego_y_m=float(ego_location.y),
                    initial_progress_floor=0.0,
                    duration_growth_factor=float(
                        self.config.get(
                            "route_tracking_lane_change_duration_growth_factor",
                            1.3,
                        )
                    ),
                    max_curvature_1pm=float(lane_change_contract.max_curvature_1pm),
                )
            )
        else:
            locked = shape_lane_change_reference(
                target_reference=target_reference,
                source_reference=source_reference,
                duration_s=float(resolved_duration_s),
                dt_s=float(self.mpc.dt_s),
                current_lane_id=int(current_lane_id),
                target_lane_id=int(resolved_target_lane_id),
                target_speed_mps=float(target_speed_mps),
                ego_x_m=float(ego_location.x),
                ego_y_m=float(ego_location.y),
                initial_progress_floor=0.0,
                blend_geometry=not bool(direct_target_tracking_enabled),
            )
        self._route_tracking_lane_change_duration_comfort_reason = str(
            (
                str(duration_comfort_reason) + ";"
                if str(duration_comfort_reason)
                else ""
            )
            + "geometry_floor:"
            f"speed={float(geometry_speed_mps):.2f}:"
            f"length={float(geometry_length_m):.2f}:"
            f"step={float(step_distance_m):.3f}"
        )
        self._route_tracking_lane_change_resolved_duration_s = float(
            resolved_duration_s
        )
        if len(locked) < int(self.mpc.horizon_steps):
            return f"lane_change_lock_failed:short_reference:{len(locked)}"
        self._route_tracking_lane_change_option = str(normalized_option)
        self._route_tracking_lane_change_reference = [
            dict(sample) for sample in locked
        ]
        if bool(direct_target_tracking_enabled):
            # Under direct target-lane tracking, MPC's own QP -- not a
            # pre-shaped geometric blend -- determines the transient path,
            # so the per-sample "lane_change_progress" tag (a time schedule)
            # no longer reflects genuine lateral crossing. Keep the aligned
            # source/target pairs so the window search below can measure
            # ego's real geometric progress each tick instead.
            aligned_target = _align_target_reference_to_source(
                source_reference=source_reference,
                target_reference=target_reference,
            )
            pair_count = min(len(source_reference), len(aligned_target))
            self._route_tracking_lane_change_progress_pairs = [
                (dict(source_reference[index]), dict(aligned_target[index]))
                for index in range(pair_count)
            ]
        else:
            self._route_tracking_lane_change_progress_pairs = []
        self._route_tracking_lane_change_progress_index = 0
        self._route_tracking_lane_change_source_lane_id = int(current_lane_id)
        self._route_tracking_lane_change_target_lane_id = int(
            resolved_target_lane_id
        )
        self._route_tracking_lane_change_target_speed_mps = max(
            0.0,
            float(target_speed_mps),
        )
        self._route_tracking_lane_change_phase = "executing"
        self._route_tracking_lane_change_stabilization_frames = 0
        self._route_tracking_lane_change_completion_stable_frames = 0
        self._route_tracking_lane_change_completion_debug = {}
        # The moment this commitment locked, for cross-CAV arbitration
        # ("earliest commitment wins" -- see cooperative_arbitration.py).
        self._route_tracking_lane_change_committed_at_s = float(self._sim_time_s())
        # A freshly locked commitment starts with no known ID discontinuity;
        # see `_release_completed_lane_change_commitment` for why this flag
        # exists.
        self._lane_id_discontinuity_since_lock = False
        self._route_tracking_lane_change_progress = float(
            locked[0].get("lane_change_initial_progress", 0.0)
        )
        return (
            "lane_change_reference_locked:"
            f"{normalized_option}:N={len(locked)}:{target_reason}"
        )

    def _route_tracking_lane_change_window(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        target_speed_mps: float,
        step_distance_m: float,
    ) -> tuple[list[dict[str, object]], str]:
        """Advance monotonically over the locked lane-change trajectory."""

        master = [
            dict(sample)
            for sample in self._route_tracking_lane_change_reference
        ]
        if not master:
            return [], "lane_change_window_missing_locked_reference"
        start = min(
            max(0, int(self._route_tracking_lane_change_progress_index)),
            len(master) - 1,
        )
        search_end = min(
            len(master),
            start + max(10, int(self.mpc.horizon_steps)),
        )
        best_index = min(
            range(start, search_end),
            key=lambda index: math.hypot(
                float(master[index].get("x_ref_m", master[index].get("x", 0.0)))
                - float(ego_location.x),
                float(master[index].get("y_ref_m", master[index].get("y", 0.0)))
                - float(ego_location.y),
            ),
        )
        min_first_forward_m = float(
            self.config.get(
                "reference_contract_lane_change_min_first_forward_m",
                0.2,
            )
        )
        while best_index + 1 < len(master):
            sample = master[best_index]
            forward_m, _ = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(
                    sample.get("x_ref_m", sample.get("x", ego_location.x))
                ),
                target_y_m=float(
                    sample.get("y_ref_m", sample.get("y", ego_location.y))
                ),
            )
            if float(forward_m) >= float(min_first_forward_m):
                break
            best_index += 1
        self._route_tracking_lane_change_progress_index = max(
            int(self._route_tracking_lane_change_progress_index),
            int(best_index),
        )
        window = [
            dict(sample)
            for sample in master[
                best_index : best_index + int(self.mpc.horizon_steps)
            ]
        ]
        progress_pairs = getattr(
            self, "_route_tracking_lane_change_progress_pairs", []
        )
        if window and not progress_pairs:
            self._route_tracking_lane_change_progress = max(
                float(self._route_tracking_lane_change_progress),
                float(window[0].get("lane_change_progress", 0.0)),
            )
        if progress_pairs:
            # Under direct target-lane tracking (progress_pairs populated),
            # the per-sample "lane_change_progress" tag is a stale
            # time-schedule, not a genuine crossing measurement -- taking
            # its max against a live geometric read would let one bad
            # (over-reported) schedule value permanently inflate progress,
            # since this accumulator is monotonic. Measure live progress
            # instead, monotonic only against its own prior value.
            from cpx_planning.pipeline.candidate_pipeline import (
                _lane_change_initial_progress,
            )

            pair_index = min(int(best_index), len(progress_pairs) - 1)
            source_sample, target_sample = progress_pairs[pair_index]
            live_progress = _lane_change_initial_progress(
                source_sample=source_sample,
                target_sample=target_sample,
                ego_x_m=float(ego_location.x),
                ego_y_m=float(ego_location.y),
            )
            self._route_tracking_lane_change_progress = max(
                float(self._route_tracking_lane_change_progress),
                float(live_progress),
            )
        while window and len(window) < int(self.mpc.horizon_steps):
            previous = dict(window[-1])
            heading_rad = float(previous.get("heading_rad", ego_yaw_rad))
            x_m = float(previous.get("x_ref_m", previous.get("x", ego_location.x)))
            y_m = float(previous.get("y_ref_m", previous.get("y", ego_location.y)))
            padded = dict(previous)
            padded["x_ref_m"] = x_m + float(step_distance_m) * math.cos(heading_rad)
            padded["y_ref_m"] = y_m + float(step_distance_m) * math.sin(heading_rad)
            padded["x"] = float(padded["x_ref_m"])
            padded["y"] = float(padded["y_ref_m"])
            padded["lane_id"] = int(
                self._route_tracking_lane_change_target_lane_id
            )
            padded["lane_change_progress"] = 1.0
            window.append(padded)
        for sample in window:
            sample["speed_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["v_ref_mps"] = max(0.0, float(target_speed_mps))
            sample["speed_mps"] = max(0.0, float(target_speed_mps))
        return (
            window,
            "lane_change_locked_window:"
            f"phase={str(getattr(self, '_route_tracking_lane_change_phase', 'executing'))}:"
            f"index={int(best_index)}:"
            f"progress={float(self._route_tracking_lane_change_progress):.3f}",
        )

    def _validate_route_tracking_lane_change_reference(
        self,
        *,
        reference: Sequence[Mapping[str, object]],
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
    ) -> tuple[bool, str]:
        """Validate heading, curvature, spacing, and road-corridor reachability."""

        samples = [dict(sample) for sample in list(reference or [])]
        if len(samples) < 2:
            return False, "lane_change_validation:too_few_points"
        points: list[tuple[float, float]] = []
        headings: list[float] = []
        distances: list[float] = []
        boundary_failures = 0
        max_lane_fraction = float(
            self.config.get(
                "route_tracking_lane_change_max_lane_center_fraction",
                0.70,
            )
        )
        for sample in samples:
            try:
                x_m = float(sample.get("x_ref_m", sample.get("x", "")))
                y_m = float(sample.get("y_ref_m", sample.get("y", "")))
            except Exception:
                return False, "lane_change_validation:non_finite_point"
            if not math.isfinite(x_m) or not math.isfinite(y_m):
                return False, "lane_change_validation:non_finite_point"
            points.append((x_m, y_m))
            try:
                waypoint = self._map_waypoint_from_location(
                    PlannerLocation(x=float(x_m), y=float(y_m), z=0.0)
                )
            except Exception:
                waypoint = None
            waypoint_xyh = self.reference_generator.waypoint_geometry(waypoint)
            if waypoint is None or waypoint_xyh is None:
                boundary_failures += 1
            else:
                lane_width_m = self.reference_generator.waypoint_lane_width(waypoint)
                lane_distance_m = math.hypot(
                    float(x_m) - float(waypoint_xyh[0]),
                    float(y_m) - float(waypoint_xyh[1]),
                )
                if float(lane_distance_m) > (
                    float(max_lane_fraction) * float(lane_width_m)
                ):
                    boundary_failures += 1
        for first, second in zip(points[:-1], points[1:]):
            dx_m = float(second[0]) - float(first[0])
            dy_m = float(second[1]) - float(first[1])
            distance_m = math.hypot(dx_m, dy_m)
            if distance_m <= 1.0e-4:
                return False, "lane_change_validation:duplicate_point"
            distances.append(float(distance_m))
            headings.append(math.atan2(dy_m, dx_m))
        first_heading_error_rad = abs(
            self._wrap_angle(float(headings[0]) - float(ego_yaw_rad))
        )
        max_heading_error_rad = math.radians(
            float(
                self.config.get(
                    "route_tracking_recovery_heading_error_deg",
                    25.0,
                )
            )
        )
        if float(first_heading_error_rad) > float(max_heading_error_rad):
            return (
                False,
                "lane_change_validation:heading_error:"
                f"{math.degrees(first_heading_error_rad):.1f}deg",
            )
        max_curvature_1pm = 0.0
        max_heading_jump_rad = 0.0
        for index, (first, second) in enumerate(
            zip(headings[:-1], headings[1:])
        ):
            heading_jump_rad = abs(self._wrap_angle(second - first))
            max_heading_jump_rad = max(
                float(max_heading_jump_rad),
                float(heading_jump_rad),
            )
            max_curvature_1pm = max(
                float(max_curvature_1pm),
                float(heading_jump_rad)
                / max(1.0e-3, float(distances[index + 1])),
            )
        curvature_limit = float(
            self.config.get(
                "route_tracking_lane_change_max_curvature_1pm",
                0.35,
            )
        )
        if float(max_curvature_1pm) > float(curvature_limit):
            return (
                False,
                "lane_change_validation:curvature:"
                f"{float(max_curvature_1pm):.3f}",
            )
        heading_jump_limit = float(
            self.config.get(
                "route_tracking_lane_change_max_heading_jump_rad",
                0.35,
            )
        )
        if float(max_heading_jump_rad) > float(heading_jump_limit):
            return (
                False,
                "lane_change_validation:heading_jump:"
                f"{float(max_heading_jump_rad):.3f}",
            )
        max_boundary_failures = int(
            self.config.get(
                "route_tracking_lane_change_max_boundary_failures",
                1,
            )
        )
        if int(boundary_failures) > int(max_boundary_failures):
            return (
                False,
                "lane_change_validation:boundary:"
                f"{int(boundary_failures)}",
            )
        return (
            True,
            "lane_change_validation:valid:"
            f"heading={math.degrees(first_heading_error_rad):.1f}deg:"
            f"curvature={float(max_curvature_1pm):.3f}:"
            f"boundary_failures={int(boundary_failures)}",
        )

    def _reset_route_tracking_lane_change_reference(self) -> None:
        self._route_tracking_lane_change_option = ""
        self._route_tracking_lane_change_progress = 0.0
        self._route_tracking_lane_change_reference = []
        self._route_tracking_lane_change_progress_pairs = []
        self._route_tracking_lane_change_envelope_blocks = None
        self._route_tracking_lane_change_envelope_epsilon0 = 0.0
        self._route_tracking_lane_change_commitment_invalid_frames = 0
        self._route_tracking_lane_change_duration_comfort_reason = ""
        self._route_tracking_lane_change_resolved_duration_s = 0.0
        self._route_tracking_lane_change_progress_index = 0
        self._route_tracking_lane_change_source_lane_id = 0
        self._route_tracking_lane_change_target_lane_id = 0
        self._route_tracking_lane_change_target_speed_mps = 0.0
        self._route_tracking_lane_change_phase = "idle"
        self._route_tracking_lane_change_stabilization_frames = 0
        self._route_tracking_lane_change_completion_stable_frames = 0
        self._route_tracking_lane_change_committed_at_s = -float("inf")

    def _attempt_turn_route_replan(
        self,
        *,
        ego_location: Any,
        trigger_reason: str = "turn_reference_unavailable",
    ) -> tuple[bool, bool, str]:
        """Request a bounded route rebuild while preserving stop-on-failure."""

        now_s = float(self._sim_time_s())
        cooldown_s = max(
            0.1,
            float(self.config.get("turn_route_replan_cooldown_s", 2.0)),
        )
        elapsed_s = float(now_s) - float(self._route_replan_last_attempt_s)
        if elapsed_s < cooldown_s:
            reason = (
                "route_replan_cooldown:"
                f"remaining={float(cooldown_s - elapsed_s):.2f}"
            )
            self._route_replan_last_reason = str(reason)
            return False, False, str(reason)

        self._route_replan_last_attempt_s = float(now_s)
        self._route_replan_attempt_count += 1
        result = self.route_manager.replan_from(
            start_point={
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "z": float(getattr(ego_location, "z", 0.0)),
            },
            trigger_reason=str(trigger_reason),
        )
        self._route_replan_last_reason = str(result.reason)
        if not bool(result.success):
            return True, False, str(result.reason)

        self._active_route_summary = self.route_manager.active_route_summary
        self._temporary_destination_state = None
        self._previous_lane_center_reference = []
        self._lane_reference_freeze_count = 0
        self._reset_route_tracking_lane_change_reference()
        maneuver_manager = getattr(self, "maneuver_manager", None)
        if maneuver_manager is not None:
            maneuver_manager.reset(reason="turn_route_replanned")
        self.control_buffer.reset(reason="turn_route_replanned")
        return True, True, str(result.reason)

    def _current_route_tracking_lane_change_envelope_payload_world(
        self,
    ) -> Optional[Mapping[str, object]]:
        if (
            not self._route_tracking_lane_change_envelope_blocks
            or str(getattr(self, "_route_tracking_lane_change_phase", "idle"))
            != "executing"
        ):
            return None
        return {
            "blocks": self._route_tracking_lane_change_envelope_blocks,
            "epsilon0": self._route_tracking_lane_change_envelope_epsilon0,
            "rho": float(getattr(self.mpc, "road_envelope_rho", -8.0)),
        }

    def _rolling_turn_envelope_payload_world(
        self,
        *,
        behavior_decision: str,
        reference_samples: Sequence[Mapping[str, object]],
    ) -> Optional[Mapping[str, object]]:
        """Build an MPC road envelope for only the current turn horizon."""

        if str(behavior_decision or "").strip().lower() not in {
            "intersection_turn_left",
            "intersection_turn_right",
        }:
            return None
        if not bool(self.config.get("turn_mpc_road_envelope_enabled", True)):
            return None
        from cpx_planning.pipeline.candidate_pipeline import (
            build_turn_reference_envelope_blocks,
        )
        from cpx_planning.MPC.lane_keep import (
            road_envelope_conservativeness_correction,
        )

        vehicle = getattr(getattr(self, "vehicle_manager", None), "vehicle", None)
        extent = getattr(getattr(vehicle, "bounding_box", None), "extent", None)
        ego_half_width_m = max(
            0.1,
            float(
                getattr(
                    extent,
                    "y",
                    self.config.get("metrics_ego_half_width_m", 1.0),
                )
            ),
        )
        blocks = build_turn_reference_envelope_blocks(
            reference_samples=reference_samples,
            ego_half_width_m=float(ego_half_width_m),
            safety_margin_m=max(
                0.0,
                float(
                    self.config.get(
                        "turn_mpc_road_envelope_safety_margin_m",
                        self.config.get(
                            "reference_contract_turn_boundary_margin_m",
                            0.15,
                        ),
                    )
                ),
            ),
            default_lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
            longitudinal_overlap_m=max(
                0.0,
                float(self.config.get("turn_mpc_road_envelope_overlap_m", 0.75)),
            ),
        )
        if not blocks:
            return None
        rho = float(getattr(self.mpc, "road_envelope_rho", -8.0))
        return {
            "blocks": blocks,
            "epsilon0": road_envelope_conservativeness_correction(
                blocks,
                rho=float(rho),
            ),
            "rho": float(rho),
            # This is recovery slack, not extra drivable width.  Keeping the
            # 10k envelope penalty means MPC still prefers the body-safe tube,
            # while the larger ceiling prevents a small tracking error at the
            # turn apex from making the entire QP mathematically infeasible.
            "max_slack_m": max(
                0.10,
                float(
                    self.config.get(
                        "turn_mpc_road_envelope_recovery_slack_m",
                        1.5,
                    )
                ),
            ),
        }

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
        ego_location: PlannerLocation,
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
        required_lane_change_decision: str = "",
        required_lane_change_target_lane_id: int = 0,
    ) -> tuple[str, int, float, list[dict[str, object]], list[float], dict[str, object]]:
        from cpx_planning.behavior_planner import (
            MpcReferenceGenerationContext,
            compute_temp_destination,
            generate_mpc_reference,
            select_reference_intent,
        )
        from cpx_planning.pipeline.candidate_pipeline import (
            CandidateBehaviorIntent,
            CandidateReferenceResult,
            _align_target_reference_to_source,
            apply_mpc_probe_result,
            evaluate_candidate_reference,
            mark_mpc_probe_skipped,
            predicted_lane_change_average_speed_mps,
            lane_change_geometry_requirements,
            select_best_candidate,
            select_candidate_with_commitment,
            route_lane_change_target_anchor,
            shape_lane_change_reference,
            summarize_candidate_results,
        )
        from cpx_planning.pipeline.stage_contracts import (
            ManeuverCommitment,
        )
        from cpx_planning.pipeline.reference_pipeline import (
            ReferencePipelineRequest,
        )

        lane_change_commitment_release_reason = (
            self._release_completed_lane_change_commitment(
                current_lane_id=int(current_lane_id),
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
            )
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
        keep_lane_reference: list[dict[str, object]] = []
        for intent in intents:
            candidate_decision = str(getattr(intent, "decision", baseline_decision))
            candidate_target_lane_id = int(getattr(intent, "target_lane_id", current_lane_id) or current_lane_id)
            candidate_speed_ref_mps = float(getattr(intent, "target_speed_mps", baseline_speed_ref_mps))
            route_required_candidate = bool(
                str(required_lane_change_decision)
                and str(candidate_decision) == str(required_lane_change_decision)
                and int(candidate_target_lane_id)
                == int(required_lane_change_target_lane_id)
            )
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
            candidate_lane_reference_step_m = max(
                float(self.config.get("route_tracking_min_step_m", 0.10)),
                float(self.mpc.dt_s)
                * max(
                    0.5,
                    float(ego_speed_mps),
                    abs(float(candidate_speed_ref_mps)),
                ),
            )
            candidate_geometry_length_m = 0.0
            candidate_geometry_speed_mps = 0.0
            if str(candidate_decision) in {"lane_change_left", "lane_change_right"}:
                (
                    candidate_geometry_speed_mps,
                    candidate_geometry_length_m,
                    candidate_geometry_step_m,
                ) = lane_change_geometry_requirements(
                    ego_speed_mps=float(ego_speed_mps),
                    target_speed_mps=float(candidate_speed_ref_mps),
                    duration_s=float(
                        getattr(intent, "lane_change_duration_s", 4.0) or 4.0
                    ),
                    dt_s=float(self.mpc.dt_s),
                    lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                    max_curvature_1pm=float(
                        self.config.get("reference_vehicle_max_curvature_1pm", 0.35)
                    ),
                    minimum_geometry_speed_mps=float(
                        self.config.get("lane_change_min_geometry_speed_mps", 2.0)
                    ),
                    minimum_length_m=float(
                        self.config.get("lane_change_min_length_m", 10.0)
                    ),
                    acceleration_limit_mps2=float(
                        self.config.get(
                            "lane_change_planning_acceleration_limit_mps2", 2.0
                        )
                    ),
                )
                candidate_lane_reference_step_m = max(
                    float(candidate_lane_reference_step_m),
                    float(candidate_geometry_step_m),
                )
            same_as_baseline = (
                str(candidate_decision) == str(baseline_decision)
                and int(candidate_target_lane_id) == int(baseline_target_lane_id)
                and abs(float(candidate_speed_ref_mps) - float(baseline_speed_ref_mps)) < 1.0e-3
                # Lane-change candidates require the geometry floor computed
                # below. The generic baseline may have been sampled from the
                # near-zero controller speed and is therefore not an
                # equivalent reference even when decision/target/speed match.
                and str(candidate_decision)
                not in {"lane_change_left", "lane_change_right"}
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
                        0.05,
                        float(candidate_lane_reference_step_m),
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

            if bool(route_required_candidate) and len(route_points) >= 2:
                # A GRP lane change contains a single lateral edge.  Use the
                # point after that edge to identify the physical target lane,
                # then rebuild a continuous lane centerline from the ego
                # station.  Feeding the raw lateral edge to MPC makes the
                # locked maneuver infeasible; selecting by canonical lane id
                # alone is ambiguous on Town06 where adjacent lanes can share
                # the same id.
                from cpx_planning.behavior_planner.temp_destination import (
                    _build_forward_reference_samples,
                )
                lane_reference_step_m = max(
                    0.05,
                    float(candidate_lane_reference_step_m),
                )
                target_anchor_wp, target_anchor_reason = (
                    route_lane_change_target_anchor(
                        map_planner=self.reference_map,
                        route_points=route_points,
                        ego_x_m=float(current_state[0]),
                        ego_y_m=float(current_state[1]),
                        z_m=float(getattr(ego_location, "z", 0.0)),
                        # Detection belongs to route geometry, not controller
                        # sampling.  A speed-dependent value would stop
                        # recognizing the same 3.5 m lane edge at high speed.
                        nominal_step_m=1.0,
                    )
                )
                continuous_target_reference = _build_forward_reference_samples(
                    target_anchor_wp,
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=float(lane_reference_step_m),
                    route_points=route_points,
                    fallback_lane_id=int(candidate_target_lane_id),
                ) if target_anchor_wp is not None else []
                if continuous_target_reference:
                    reference = [dict(sample) for sample in continuous_target_reference]
                    candidate_reference_debug.update({
                        "reference_source": "continuous_route_target_lane_center",
                        "route_lane_change_target_anchor": True,
                        "route_lane_change_target_anchor_reason": str(target_anchor_reason),
                    })
                else:
                    candidate_reference_debug.update({
                        "route_lane_change_target_anchor": False,
                        "route_lane_change_target_anchor_reason": str(target_anchor_reason),
                    })

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

            if (
                candidate_decision in {"lane_change_left", "lane_change_right"}
                and keep_lane_reference
            ):
                lane_change_duration_s = max(
                    float(self.mpc.dt_s),
                    float(getattr(intent, "lane_change_duration_s", 4.0) or 4.0),
                )
                # Use the geometry-sized target stations as the longitudinal
                # backbone. The generic keep-lane candidate may have been
                # sampled at the near-zero controller speed; using its
                # stations would silently compress the lateral quintic again.
                lane_change_source_reference = _align_target_reference_to_source(
                    source_reference=reference,
                    target_reference=keep_lane_reference,
                )
                reference = shape_lane_change_reference(
                    target_reference=reference,
                    source_reference=lane_change_source_reference,
                    duration_s=float(lane_change_duration_s),
                    dt_s=float(self.mpc.dt_s),
                    current_lane_id=int(current_lane_id),
                    target_lane_id=int(candidate_target_lane_id),
                    target_speed_mps=float(candidate_speed_ref_mps),
                    ego_x_m=float(current_state[0]),
                    ego_y_m=float(current_state[1]),
                )
                if reference and len(destination_state) >= 4:
                    terminal = reference[-1]
                    destination_state = list(destination_state)
                    destination_state[0] = float(
                        terminal.get("x_ref_m", terminal.get("x", destination_state[0]))
                    )
                    destination_state[1] = float(
                        terminal.get("y_ref_m", terminal.get("y", destination_state[1]))
                    )
                    destination_state[2] = float(candidate_speed_ref_mps)
                    destination_state[3] = float(
                        terminal.get("heading_rad", destination_state[3])
                    )
                    if len(destination_state) >= 5:
                        destination_state[4] = int(candidate_target_lane_id)
                candidate_reference_debug.update({
                    "lane_change_trajectory_variant": str(
                        getattr(intent, "trajectory_variant", "normal")
                    ),
                    "lane_change_duration_s": float(
                        self._route_tracking_lane_change_resolved_duration_s
                        or lane_change_duration_s
                    ),
                    "lane_change_duration_comfort_reason": str(
                        self._route_tracking_lane_change_duration_comfort_reason
                    ),
                    "lane_change_reference_profile": "quintic_time_blend",
                    "lane_change_geometry_speed_mps": float(
                        candidate_geometry_speed_mps
                    ),
                    "lane_change_geometry_length_m": float(
                        candidate_geometry_length_m
                    ),
                    "lane_change_geometry_step_m": float(
                        candidate_lane_reference_step_m
                    ),
                    "lane_change_authorization_source": (
                        "opportunistic"
                        if str(getattr(intent, "reason", "")).startswith("opportunistic_")
                        else "route"
                    ),
                    "lane_change_initial_progress": (
                        float(reference[0].get("lane_change_initial_progress", 0.0))
                        if reference else 0.0
                    ),
                    "lane_change_terminal_progress": (
                        float(reference[-1].get("lane_change_progress", 0.0))
                        if reference else 0.0
                    ),
                })

            conditioned = self.reference_pipeline.condition(
                ReferencePipelineRequest(
                    destination_state=destination_state,
                    reference_samples=reference,
                    current_state=current_state,
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    ego_speed_mps=float(ego_speed_mps),
                    target_speed_mps=float(candidate_speed_ref_mps),
                    behavior_decision=str(candidate_decision),
                    behavior_fsm_state=str(candidate_lc_state),
                    current_lane_id=int(current_lane_id),
                    target_lane_id=int(candidate_target_lane_id),
                    stop_goal_active=bool(candidate_stop_goal_active),
                    stop_target=(
                        getattr(intent, "stop_target", None)
                        if isinstance(
                            getattr(intent, "stop_target", None), Mapping
                        )
                        else None
                    ),
                    route_points=route_points,
                )
            )
            destination_state = list(conditioned.destination_state)
            reference = [
                dict(sample) for sample in conditioned.reference_samples
            ]
            stabilizer_reason = str(conditioned.reason)
            contract_result = conditioned.validation
            candidate_reference_debug["mpc_reference_stabilizer_reason"] = str(stabilizer_reason)
            candidate_result = CandidateReferenceResult(
                intent=intent,
                destination_state=list(destination_state),
                lane_center_reference=[dict(sample) for sample in list(reference or [])],
                reference_debug=dict(candidate_reference_debug),
                contract_result=contract_result,
            )
            candidate_is_static_obstacle_local_avoidance = bool(
                self._static_obstacle_local_target_lane_id is not None
                and int(candidate_target_lane_id)
                == int(self._static_obstacle_local_target_lane_id)
                and int(candidate_target_lane_id) != int(current_lane_id)
            )
            evaluated_candidate_result = evaluate_candidate_reference(
                candidate=candidate_result,
                ego_state=current_state,
                object_snapshots=object_snapshots,
                prediction_trajectories=prediction_trajectories,
                current_lane_id=int(current_lane_id),
                min_object_distance_m=float(
                    self.static_obstacle_local_avoidance_min_object_distance_m
                    if candidate_is_static_obstacle_local_avoidance
                    else self.full_candidate_reference_min_object_distance_m
                ),
                previous_risk_bucket=str(
                    self._candidate_risk_bucket_state.get(str(intent.name), "")
                ),
                risk_hysteresis_margin_m=float(
                    self.candidate_risk_hysteresis_margin_m
                ),
            )
            self._candidate_risk_bucket_state[str(intent.name)] = str(
                evaluated_candidate_result.risk_bucket
            )
            candidate_results.append(evaluated_candidate_result)
            if (
                str(candidate_decision) == "lane_follow"
                and int(candidate_target_lane_id) == int(current_lane_id)
                and not keep_lane_reference
                and reference
            ):
                keep_lane_reference = [
                    dict(sample) for sample in list(reference or [])
                ]

        # A committed maneuver owns one immutable master trajectory. Replanned
        # lane-change variants remain useful before commitment, but they must
        # not replace the executing trajectory after commitment has started.
        if bool(self._route_tracking_lane_change_reference):
            commitment_phase = str(
                getattr(
                    self,
                    "_route_tracking_lane_change_phase",
                    "executing",
                )
            )
            stabilization_active = bool(
                commitment_phase == "target_lane_stabilization"
            )
            committed_decision = (
                "lane_change_left"
                if str(self._route_tracking_lane_change_option)
                == "CHANGELANELEFT"
                else "lane_change_right"
            )
            committed_speed_mps = max(
                0.5,
                float(
                    getattr(
                        self,
                        "_route_tracking_lane_change_target_speed_mps",
                        0.0,
                    )
                    or baseline_speed_ref_mps
                ),
            )
            committed_step_m = max(
                0.1,
                float(self.mpc.dt_s) * float(committed_speed_mps),
            )
            committed_reference, committed_window_reason = (
                self._route_tracking_lane_change_window(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    target_speed_mps=float(committed_speed_mps),
                    step_distance_m=float(committed_step_m),
                )
            )
            # Windowing the locked master path (nearest-point search plus
            # low-speed tail padding) can transiently read a higher raw
            # curvature than the master path was built for, even though the
            # locked path itself was shaped to satisfy the contract at lock
            # time. reference_pipeline.py already repairs exactly this case
            # for mode="lane_change" via curvature_feasible_samples before
            # validating; _validate_candidate_reference_contract below had no
            # equivalent repair, so a one-tick windowing spike hard-rejected
            # the committed candidate outright and forced an emergency-brake
            # fallback mid-maneuver. Apply the same repair here so the
            # candidate is judged on the same shaped geometry the final
            # reference pipeline would have produced anyway.
            if committed_reference:
                from cpx_planning.pipeline.reference_contract import (
                    contract_from_config,
                )

                committed_lane_change_contract = contract_from_config(
                    mode="lane_change",
                    expected_lane_id=int(
                        self._route_tracking_lane_change_target_lane_id
                    ),
                    horizon_steps=int(self.mpc.horizon_steps),
                    config=dict(self.config),
                    default_speed_mps=max(
                        float(self.target_speed_mps),
                        float(committed_speed_mps),
                        0.1,
                    ),
                )
                committed_reference, committed_curvature_reason = (
                    self.reference_generator.curvature_feasible_samples(
                        reference_samples=committed_reference,
                        ego_location=ego_location,
                        ego_heading_rad=float(ego_yaw_rad),
                        max_curvature_1pm=float(
                            committed_lane_change_contract.max_curvature_1pm
                        ),
                        mode="committed_lane_change",
                    )
                )
                if committed_curvature_reason:
                    committed_window_reason = (
                        str(committed_window_reason)
                        + ";"
                        + str(committed_curvature_reason)
                    )
            committed_destination: list[float] = []
            if committed_reference:
                terminal = dict(committed_reference[-1])
                committed_destination = [
                    float(
                        terminal.get(
                            "x_ref_m",
                            terminal.get("x", current_state[0]),
                        )
                    ),
                    float(
                        terminal.get(
                            "y_ref_m",
                            terminal.get("y", current_state[1]),
                        )
                    ),
                    float(committed_speed_mps),
                    float(terminal.get("heading_rad", current_state[3])),
                    int(self._route_tracking_lane_change_target_lane_id),
                ]
            committed_contract = self._validate_candidate_reference_contract(
                decision=str(committed_decision),
                lc_state=(
                    "TARGET_LANE_STABILIZATION"
                    if bool(stabilization_active)
                    else "EXECUTE_LANE_CHANGE_LEFT"
                    if committed_decision == "lane_change_left"
                    else "EXECUTE_LANE_CHANGE_RIGHT"
                ),
                current_lane_id=int(current_lane_id),
                speed_ref_mps=float(committed_speed_mps),
                stop_goal_active=False,
                current_state=current_state,
                destination_state=committed_destination,
                lane_center_reference=committed_reference,
            )
            committed_intent = CandidateBehaviorIntent(
                name="committed_lane_change_continuation",
                decision=str(committed_decision),
                target_lane_id=int(
                    self._route_tracking_lane_change_target_lane_id
                ),
                target_speed_mps=float(committed_speed_mps),
                base_cost=-100.0,
                reason=(
                    "target_lane_stabilization"
                    if bool(stabilization_active)
                    else "locked_maneuver_execution"
                ),
                trajectory_variant=(
                    "target_lane_stabilization"
                    if bool(stabilization_active)
                    else "locked"
                ),
            )
            committed_result = CandidateReferenceResult(
                intent=committed_intent,
                destination_state=list(committed_destination),
                lane_center_reference=[
                    dict(sample) for sample in committed_reference
                ],
                reference_debug={
                    "reference_source": (
                        "target_lane_stabilization_reference"
                        if bool(stabilization_active)
                        else "locked_quintic_lane_change_reference"
                    ),
                    "candidate_lane_change_window_reason": str(
                        committed_window_reason
                    ),
                    "lane_change_phase": str(commitment_phase),
                    "lane_change_stabilization_frames": int(
                        getattr(
                            self,
                            "_route_tracking_lane_change_stabilization_frames",
                            0,
                        )
                    ),
                    "route_tracking_lane_change_locked": True,
                    "route_tracking_lane_change_progress_index": int(
                        self._route_tracking_lane_change_progress_index
                    ),
                    "route_tracking_lane_change_source_lane_id": int(
                        self._route_tracking_lane_change_source_lane_id
                    ),
                    "route_tracking_lane_change_target_lane_id": int(
                        self._route_tracking_lane_change_target_lane_id
                    ),
                    "lane_change_duration_s": float(
                        self._route_tracking_lane_change_resolved_duration_s
                    ),
                    "lane_change_duration_comfort_reason": str(
                        self._route_tracking_lane_change_duration_comfort_reason
                    ),
                },
                contract_result=committed_contract,
            )
            committed_is_static_obstacle_local_avoidance = bool(
                self._static_obstacle_local_target_lane_id is not None
                and int(self._route_tracking_lane_change_target_lane_id)
                == int(self._static_obstacle_local_target_lane_id)
                and int(self._route_tracking_lane_change_target_lane_id)
                != int(current_lane_id)
            )
            evaluated_committed_result = evaluate_candidate_reference(
                candidate=committed_result,
                ego_state=current_state,
                object_snapshots=object_snapshots,
                prediction_trajectories=prediction_trajectories,
                current_lane_id=int(current_lane_id),
                min_object_distance_m=float(
                    self.static_obstacle_local_avoidance_min_object_distance_m
                    if committed_is_static_obstacle_local_avoidance
                    else self.full_candidate_reference_min_object_distance_m
                ),
                previous_risk_bucket=str(
                    self._candidate_risk_bucket_state.get(
                        str(committed_intent.name), ""
                    )
                ),
                risk_hysteresis_margin_m=float(
                    self.candidate_risk_hysteresis_margin_m
                ),
            )
            self._candidate_risk_bucket_state[str(committed_intent.name)] = str(
                evaluated_committed_result.risk_bucket
            )
            candidate_results.append(
                evaluated_committed_result
            )

        probe_summary = self._probe_candidate_results_for_mpc(
            candidate_results=candidate_results,
            current_state=current_state,
            object_snapshots=object_snapshots,
            apply_mpc_probe_result=apply_mpc_probe_result,
            mark_mpc_probe_skipped=mark_mpc_probe_skipped,
            road_envelope_payload_world=(
                self._current_route_tracking_lane_change_envelope_payload_world()
            ),
        )
        committed_decision = (
            "lane_change_left"
            if str(self._route_tracking_lane_change_option) == "CHANGELANELEFT"
            else "lane_change_right"
            if str(self._route_tracking_lane_change_option) == "CHANGELANERIGHT"
            else ""
        )
        maneuver_commitment = ManeuverCommitment(
            state=(
                "STABILIZING"
                if str(
                    getattr(
                        self,
                        "_route_tracking_lane_change_phase",
                        "",
                    )
                )
                == "target_lane_stabilization"
                else "COMMITTED"
                if bool(self._route_tracking_lane_change_reference)
                else "IDLE"
            ),
            decision=str(committed_decision),
            source_lane_id=int(self._route_tracking_lane_change_source_lane_id),
            target_lane_id=int(self._route_tracking_lane_change_target_lane_id),
            progress=float(self._route_tracking_lane_change_progress),
            reference_locked=bool(self._route_tracking_lane_change_reference),
        )
        selection_outcome = select_candidate_with_commitment(
            candidate_results,
            commitment=maneuver_commitment,
            required_decision=str(required_lane_change_decision),
            required_target_lane_id=int(required_lane_change_target_lane_id),
        )

        if (
            bool(self.strict_decision_ownership_enabled)
            and candidate_results
            and (
                not any(bool(candidate.feasible) for candidate in candidate_results)
                or selection_outcome.selected is None
            )
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
                maneuver_commitment=maneuver_commitment,
                selection_reason=str(selection_outcome.reason),
            )

        selected = (
            selection_outcome.selected
            if selection_outcome.selected is not None
            else select_best_candidate(candidate_results)
        )
        selected_debug = dict(selected.reference_debug or {})
        selected_reference = [
            dict(sample)
            for sample in list(selected.lane_center_reference or [])
        ]
        selected_destination = list(selected.destination_state or [])
        selected_decision = str(selected.intent.decision)
        if selected_decision in {"lane_change_left", "lane_change_right"}:
            selected_target_lane_id = int(selected.intent.target_lane_id)
            route_option = (
                "CHANGELANELEFT"
                if selected_decision == "lane_change_left"
                else "CHANGELANERIGHT"
            )
            planned_lane_change_speed_mps = predicted_lane_change_average_speed_mps(
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(selected.intent.target_speed_mps),
                duration_s=float(selected.intent.lane_change_duration_s or 4.0),
                acceleration_limit_mps2=float(
                    self.config.get(
                        "lane_change_planning_acceleration_limit_mps2",
                        2.0,
                    )
                ),
            )
            (
                lane_change_geometry_speed_mps,
                lane_change_geometry_length_m,
                geometry_step_distance_m,
            ) = lane_change_geometry_requirements(
                ego_speed_mps=float(ego_speed_mps),
                target_speed_mps=float(selected.intent.target_speed_mps),
                duration_s=float(selected.intent.lane_change_duration_s or 4.0),
                dt_s=float(self.mpc.dt_s),
                lane_width_m=float(getattr(self.mpc, "lane_width_m", 3.5)),
                max_curvature_1pm=float(
                    self.config.get("reference_vehicle_max_curvature_1pm", 0.35)
                ),
                minimum_geometry_speed_mps=float(
                    self.config.get("lane_change_min_geometry_speed_mps", 2.0)
                ),
                minimum_length_m=float(
                    self.config.get("lane_change_min_length_m", 10.0)
                ),
                acceleration_limit_mps2=float(
                    self.config.get(
                        "lane_change_planning_acceleration_limit_mps2", 2.0
                    )
                ),
            )
            step_distance_m = max(0.1, float(geometry_step_distance_m))
            selected_is_committed_continuation = bool(
                str(selected.intent.name)
                == "committed_lane_change_continuation"
            )
            lock_matches = bool(
                self._route_tracking_lane_change_reference
                and int(self._route_tracking_lane_change_target_lane_id)
                == int(selected_target_lane_id)
                and str(self._route_tracking_lane_change_option)
                == str(route_option)
                and (
                    bool(selected_is_committed_continuation)
                    or int(self._route_tracking_lane_change_source_lane_id)
                    == int(current_lane_id)
                )
            )
            lock_reason = "candidate_lane_change_lock_reused"
            if not bool(lock_matches):
                lock_target_waypoint = None
                selected_is_route_required = bool(
                    str(required_lane_change_decision)
                    and str(selected_decision) == str(required_lane_change_decision)
                    and int(selected_target_lane_id)
                    == int(required_lane_change_target_lane_id)
                )
                if bool(selected_is_route_required) and len(route_points) >= 2:
                    lock_target_waypoint, _ = route_lane_change_target_anchor(
                        map_planner=self.reference_map,
                        route_points=route_points,
                        ego_x_m=float(current_state[0]),
                        ego_y_m=float(current_state[1]),
                        z_m=float(getattr(ego_location, "z", 0.0)),
                        nominal_step_m=1.0,
                    )
                lock_reason = self._lock_route_tracking_lane_change_reference(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    target_lane_id=int(selected_target_lane_id),
                    route_option=str(route_option),
                    target_speed_mps=float(selected.intent.target_speed_mps),
                    step_distance_m=float(step_distance_m),
                    duration_s=float(
                        selected.intent.lane_change_duration_s or 4.0
                    ),
                    target_waypoint_override=lock_target_waypoint,
                )
            locked_window, window_reason = (
                self._route_tracking_lane_change_window(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    target_speed_mps=float(selected.intent.target_speed_mps),
                    step_distance_m=float(step_distance_m),
                )
            )
            locked_valid, locked_validation_reason = (
                self._validate_route_tracking_lane_change_reference(
                    reference=locked_window,
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                )
            )
            if locked_window and bool(locked_valid):
                selected_reference = [
                    dict(sample) for sample in locked_window
                ]
                terminal = selected_reference[-1]
                if len(selected_destination) >= 4:
                    selected_destination[0] = float(
                        terminal.get("x_ref_m", terminal.get("x", selected_destination[0]))
                    )
                    selected_destination[1] = float(
                        terminal.get("y_ref_m", terminal.get("y", selected_destination[1]))
                    )
                    selected_destination[2] = float(
                        selected.intent.target_speed_mps
                    )
                    selected_destination[3] = float(
                        terminal.get("heading_rad", selected_destination[3])
                    )
                    if len(selected_destination) >= 5:
                        selected_destination[4] = int(selected_target_lane_id)
            selected_debug.update({
                "lane_change_planning_average_speed_mps": float(
                    planned_lane_change_speed_mps
                ),
                "lane_change_geometry_speed_mps": float(
                    lane_change_geometry_speed_mps
                ),
                "lane_change_geometry_length_m": float(
                    lane_change_geometry_length_m
                ),
                "lane_change_geometry_step_m": float(step_distance_m),
                "route_tracking_lane_change_locked": bool(
                    self._route_tracking_lane_change_reference
                ),
                "route_tracking_lane_change_progress_index": int(
                    self._route_tracking_lane_change_progress_index
                ),
                "route_tracking_lane_change_source_lane_id": int(
                    self._route_tracking_lane_change_source_lane_id
                ),
                "route_tracking_lane_change_target_lane_id": int(
                    self._route_tracking_lane_change_target_lane_id
                ),
                "candidate_lane_change_lock_reason": str(lock_reason),
                "candidate_lane_change_window_reason": str(window_reason),
                "candidate_lane_change_lock_validation_reason": str(
                    locked_validation_reason
                ),
            })
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
            "candidate_selected_stop_goal_active": bool(
                selected.intent.stop_goal_active
            ),
            "candidate_pipeline_count": int(len(candidate_results)),
            "candidate_prediction_trajectory_count": int(len(prediction_trajectories)),
            "candidate_pipeline_summary": summarize_candidate_results(candidate_results),
            "candidate_mpc_probe_summary": str(probe_summary),
            "candidate_selected_decision": str(selected.intent.decision),
            "candidate_selected_lane_id": int(selected.intent.target_lane_id),
            "candidate_selected_cost": float(selected.total_cost),
            "candidate_evaluation_summary": (
                f"{selected.intent.name}->{selected.intent.decision}"
                f":L{int(selected.intent.target_lane_id)}"
                f" cost={float(selected.total_cost):.2f}"
            ),
            "candidate_selection_status": str(selection_outcome.status),
            "candidate_selection_reason": str(selection_outcome.reason),
            "lane_change_commitment_release_reason": str(
                lane_change_commitment_release_reason
            ),
            "lane_change_phase": str(
                getattr(
                    self,
                    "_route_tracking_lane_change_phase",
                    "idle",
                )
            ),
            "lane_change_stabilization_frames": int(
                getattr(
                    self,
                    "_route_tracking_lane_change_stabilization_frames",
                    0,
                )
            ),
        })
        selected_debug.update(maneuver_commitment.as_debug_fields())
        selected_debug.update(
            dict(
                getattr(
                    self,
                    "_route_tracking_lane_change_completion_debug",
                    {},
                )
            )
        )
        return (
            str(selected_decision),
            int(selected.intent.target_lane_id),
            float(selected.intent.target_speed_mps),
            list(selected_reference),
            list(selected_destination),
            selected_debug,
        )

    def _release_completed_lane_change_commitment(
        self,
        *,
        current_lane_id: int,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
    ) -> str:
        """Release only after the ego converges to the locked target path."""

        if not self._route_tracking_lane_change_reference:
            return ""
        phase = str(
            getattr(
                self,
                "_route_tracking_lane_change_phase",
                "executing",
            )
        )
        target_lane_id = int(self._route_tracking_lane_change_target_lane_id)
        entry_min_progress = float(
            self.config.get(
                "lane_change_target_lane_entry_min_progress",
                0.50,
            )
        )
        terminal_samples = [
            dict(sample)
            for sample in self._route_tracking_lane_change_reference
            if float(sample.get("lane_change_progress", 0.0) or 0.0) >= 0.9
        ]
        target_corridor_sample = (
            min(
                terminal_samples,
                key=lambda sample: (
                    float(sample.get("x_ref_m", sample.get("x", ego_location.x)))
                    - float(ego_location.x)
                ) ** 2
                + (
                    float(sample.get("y_ref_m", sample.get("y", ego_location.y)))
                    - float(ego_location.y)
                ) ** 2,
            )
            if terminal_samples
            else None
        )
        stabilization_entry_lateral_error_m = float("inf")
        if target_corridor_sample is not None:
            target_x_m = float(
                target_corridor_sample.get(
                    "x_ref_m", target_corridor_sample.get("x", ego_location.x)
                )
            )
            target_y_m = float(
                target_corridor_sample.get(
                    "y_ref_m", target_corridor_sample.get("y", ego_location.y)
                )
            )
            target_heading_rad = float(
                target_corridor_sample.get("heading_rad", ego_yaw_rad)
            )
            stabilization_entry_lateral_error_m = (
                -math.sin(float(target_heading_rad))
                * (float(ego_location.x) - float(target_x_m))
                + math.cos(float(target_heading_rad))
                * (float(ego_location.y) - float(target_y_m))
            )
        stabilization_entry_max_lateral_error_m = max(
            0.1,
            float(
                self.config.get(
                    "lane_change_stabilization_entry_max_lateral_error_m",
                    0.75,
                )
            ),
        )
        stabilization_geometry_ready = bool(
            math.isfinite(float(stabilization_entry_lateral_error_m))
            and abs(float(stabilization_entry_lateral_error_m))
            <= float(stabilization_entry_max_lateral_error_m)
        )
        # Lane IDs identify the source/target topology but do not own motion
        # phase transitions.  Enter stabilization only from continuous
        # progress and convergence to the locked target corridor.  This is
        # robust both when the map ID flips early and when a road-boundary
        # re-anchor changes the ID namespace during the maneuver.
        geometry_and_progress_ready = bool(
            str(phase) != "target_lane_stabilization"
            and float(self._route_tracking_lane_change_progress)
            >= float(entry_min_progress)
            and bool(stabilization_geometry_ready)
        )
        heading_ready = True
        if geometry_and_progress_ready:
            # current_lane_id/progress alone only capture that the ego has
            # crossed into the target lane's lateral extent -- the ego's
            # heading can still be mid-turn at that instant. Stabilization
            # hands off to a one-shot quintic connector whose start
            # derivative is locked to whatever heading the ego has *right
            # now*; starting it while heading is still far from the target
            # lane's own direction forces that connector to reconcile a
            # large heading gap over a short, fixed distance, producing a
            # sharp, sustained steering correction that overshoots the
            # centerline before settling. Require heading to have mostly
            # caught up first; otherwise keep tracking the locked
            # lane-change reference a little longer and re-check next tick.
            target_waypoint = (
                self.reference_generator._map_waypoint_from_location(
                    ego_location
                )
            )
            target_heading_rad = world_heading_rad(target_waypoint)
            if target_heading_rad is not None:
                heading_error_deg = abs(
                    math.degrees(
                        math.atan2(
                            math.sin(
                                float(ego_yaw_rad) - float(target_heading_rad)
                            ),
                            math.cos(
                                float(ego_yaw_rad) - float(target_heading_rad)
                            ),
                        )
                    )
                )
                max_heading_error_deg = float(
                    self.config.get(
                        "lane_change_target_lane_entry_max_heading_error_deg",
                        10.0,
                    )
                )
                heading_ready = bool(
                    float(heading_error_deg) <= float(max_heading_error_deg)
                )
        if geometry_and_progress_ready and bool(heading_ready):
            start_reason = self._start_target_lane_stabilization(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
            )
            if str(start_reason).startswith(
                "target_lane_stabilization_started"
            ):
                return str(start_reason)
            completed_option = str(
                self._route_tracking_lane_change_option
            )
            self._route_tracking_lane_change_completed_option = (
                completed_option
            )
            self._reset_route_tracking_lane_change_reference()
            return (
                "lane_change_stabilization_unavailable_to_lane_follow_recovery:"
                f"target_lane={int(target_lane_id)}:"
                f"{str(start_reason)}"
            )

        phase = str(
            getattr(
                self,
                "_route_tracking_lane_change_phase",
                "executing",
            )
        )
        if str(phase) == "target_lane_stabilization":
            self._route_tracking_lane_change_stabilization_frames = (
                int(
                    getattr(
                        self,
                        "_route_tracking_lane_change_stabilization_frames",
                        0,
                    )
                )
                + 1
            )
            timeout_frames = max(
                1,
                int(
                    self.config.get(
                        "lane_change_stabilization_timeout_frames",
                        100,
                    )
                ),
            )
            if (
                int(self._route_tracking_lane_change_stabilization_frames)
                > int(timeout_frames)
            ):
                completed_option = str(
                    self._route_tracking_lane_change_option
                )
                self._route_tracking_lane_change_completed_option = (
                    completed_option
                )
                self._reset_route_tracking_lane_change_reference()
                return (
                    "lane_change_stabilization_timeout_to_lane_follow_recovery:"
                    f"target_lane={int(target_lane_id)}:"
                    f"map_lane={int(current_lane_id)}"
                )
        from cpx_planning.pipeline.stage_contracts import (
            evaluate_lane_change_completion,
        )

        occupancy = self.reference_generator.lane_corridor_occupancy(
            x_m=float(ego_location.x),
            y_m=float(ego_location.y),
            heading_rad=float(ego_yaw_rad),
            ego_half_width_m=float(self.config.get("reference_vehicle_half_width_m", 1.0)),
            ego_half_length_m=float(self.config.get("reference_vehicle_half_length_m", 2.4)),
            safety_margin_m=float(
                self.config.get(
                    "lane_change_completion_footprint_margin_m",
                    0.05,
                )
            ),
            corridor_sample=target_corridor_sample,
            prefer_tracking_point=True,
        )
        completion = evaluate_lane_change_completion(
            reference_samples=self._route_tracking_lane_change_reference,
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
            progress=float(self._route_tracking_lane_change_progress),
            previous_stable_frames=int(
                self._route_tracking_lane_change_completion_stable_frames
            ),
            target_lane_matches=bool(
                int(current_lane_id) == int(target_lane_id)
            ),
            footprint_clearance_m=(
                float(occupancy.footprint_clearance_m)
                if bool(occupancy.valid)
                else float("-inf")
            ),
            min_footprint_clearance_m=0.0,
            min_progress=float(
                self.config.get("lane_change_completion_min_progress", 0.92)
            ),
            max_lateral_error_m=float(
                self.config.get("lane_change_completion_max_lateral_error_m", 0.35)
            ),
            max_heading_error_rad=math.radians(
                float(
                    self.config.get(
                        "lane_change_completion_max_heading_error_deg",
                        8.0,
                    )
                )
            ),
            required_stable_frames=int(
                self.config.get("lane_change_completion_stable_frames", 5)
            ),
        )
        self._route_tracking_lane_change_completion_stable_frames = int(
            completion.stable_frames
        )
        self._route_tracking_lane_change_completion_debug = {
            "lane_change_stabilization_entry_lateral_error_m": float(
                stabilization_entry_lateral_error_m
            ),
            "lane_change_stabilization_geometry_ready": bool(
                stabilization_geometry_ready
            ),
            "lane_change_completion_reason": str(completion.reason),
            "lane_change_completion_stable_frames": int(
                completion.stable_frames
            ),
            "lane_change_completion_lateral_error_m": float(
                completion.target_lateral_error_m
            ),
            "lane_change_completion_heading_error_deg": math.degrees(
                float(completion.target_heading_error_rad)
            ),
            "lane_change_completion_target_lane_matches": bool(
                completion.target_lane_matches
            ),
            "lane_change_completion_footprint_clearance_m": float(
                completion.footprint_clearance_m
            ),
        }
        if not bool(completion.complete):
            return ""
        completed_option = str(self._route_tracking_lane_change_option)
        self._route_tracking_lane_change_completed_option = completed_option
        self._reset_route_tracking_lane_change_reference()
        return (
            "lane_change_commitment_released:"
            f"target_lane={int(target_lane_id)}:"
            f"map_lane={int(current_lane_id)}:"
            f"progress={float(completion.progress):.3f}:"
            f"lateral_error={float(completion.target_lateral_error_m):.3f}:"
            f"heading_error_deg="
            f"{math.degrees(float(completion.target_heading_error_rad)):.2f}"
        )

    def _start_target_lane_stabilization(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
    ) -> str:
        """Replace the completed lateral crossing with a target-lane handoff."""

        target_lane_id = int(self._route_tracking_lane_change_target_lane_id)
        duration_s = max(
            float(self.mpc.dt_s),
            float(
                self.config.get(
                    "lane_change_stabilization_duration_s",
                    2.5,
                )
            ),
        )
        master_steps = max(
            int(self.mpc.horizon_steps),
            int(math.ceil(float(duration_s) / float(self.mpc.dt_s))),
        )
        # Stabilization is a geometric phase, not a separate low-speed
        # behavior. Preserve the committed maneuver speed; curvature and
        # traffic-control constraints are applied by the unified speed path.
        committed_speed_mps = float(
            getattr(
                self,
                "_route_tracking_lane_change_target_speed_mps",
                0.0,
            )
            or self.target_speed_mps
        )
        speed_mps = max(
            0.5,
            min(
                float(getattr(self, "target_speed_mps", committed_speed_mps)),
                float(committed_speed_mps),
            ),
        )
        step_distance_m = max(
            0.10,
            float(self.mpc.dt_s) * float(speed_mps),
        )
        reference = self.reference_generator.target_lane_stabilization_samples(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            target_lane_id=int(target_lane_id),
            horizon_steps=int(master_steps),
            step_distance_m=float(step_distance_m),
            route_points=[],
        )
        reference, curvature_reason = (
            self.reference_generator.curvature_feasible_samples(
                reference_samples=reference,
                ego_location=ego_location,
                ego_heading_rad=float(ego_yaw_rad),
                max_curvature_1pm=float(
                    self.config.get(
                        "reference_vehicle_max_curvature_1pm",
                        0.20,
                    )
                ),
                mode="target_lane_stabilization",
            )
        )
        if len(reference) < int(self.mpc.horizon_steps):
            return (
                "target_lane_stabilization_failed:"
                f"short_reference={len(reference)}"
            )
        for sample in reference:
            sample["lane_id"] = int(target_lane_id)
            sample["lane_change_progress"] = 1.0
            sample["lane_transition_kind"] = "target_lane_stabilization"
            sample["speed_ref_mps"] = float(speed_mps)
            sample["v_ref_mps"] = float(speed_mps)
            sample["speed_mps"] = float(speed_mps)
        self._route_tracking_lane_change_reference = [
            dict(sample) for sample in reference
        ]
        self._route_tracking_lane_change_progress_index = 0
        self._route_tracking_lane_change_progress = 1.0
        self._route_tracking_lane_change_target_speed_mps = float(speed_mps)
        self._route_tracking_lane_change_phase = (
            "target_lane_stabilization"
        )
        self._route_tracking_lane_change_stabilization_frames = 0
        self._route_tracking_lane_change_completion_stable_frames = 0
        return (
            "target_lane_stabilization_started:"
            f"target_lane={int(target_lane_id)}:"
            f"N={len(reference)}:"
            f"speed={float(speed_mps):.2f}:"
            f"curvature_conditioning={str(curvature_reason or 'not_required')}"
        )

    def _probe_candidate_results_for_mpc(
        self,
        *,
        candidate_results: Sequence[object],
        current_state: Sequence[float],
        object_snapshots: Sequence[Mapping[str, object]],
        apply_mpc_probe_result: Any,
        mark_mpc_probe_skipped: Any,
        road_envelope_payload_world: Optional[Mapping[str, object]] = None,
    ) -> str:
        """Run side-effect-free MPC probes for the best keep/lane-change paths."""

        rows = list(candidate_results or [])
        lane_change_rows = [
            row
            for row in rows
            if str(getattr(getattr(row, "intent", None), "decision", "")).startswith(
                "lane_change"
            )
            and bool(getattr(row, "feasible", False))
        ]
        if not bool(self.candidate_mpc_probe_enabled) or not lane_change_rows:
            return "mpc_probe_not_applicable"

        feasible_rows = [
            row for row in rows if bool(getattr(row, "feasible", False))
        ]
        feasible_rows.sort(key=lambda row: float(getattr(row, "total_cost", float("inf"))))
        keep_rows = [
            row
            for row in feasible_rows
            if str(getattr(getattr(row, "intent", None), "decision", ""))
            == "lane_follow"
        ]
        selected_for_probe = []
        if keep_rows:
            selected_for_probe.append(keep_rows[0])
        if lane_change_rows:
            lane_change_rows.sort(
                key=lambda row: float(getattr(row, "total_cost", float("inf")))
            )
            selected_for_probe.append(lane_change_rows[0])
        for row in feasible_rows:
            if row in selected_for_probe:
                continue
            if len(selected_for_probe) >= int(self.candidate_mpc_probe_top_k):
                break
            selected_for_probe.append(row)
        selected_for_probe = selected_for_probe[: int(self.candidate_mpc_probe_top_k)]

        sim_time_s = float(self._sim_time_s())
        refresh_cache = (
            float(sim_time_s) - float(self._candidate_mpc_probe_last_time_s)
            >= float(self.candidate_mpc_probe_interval_s)
        )
        if bool(refresh_cache):
            self._candidate_mpc_probe_cache = {}
            self._candidate_mpc_probe_last_time_s = float(sim_time_s)

        probe_rows = []
        selected_ids = {id(row) for row in selected_for_probe}
        previous_profile = str(
            getattr(self.mpc, "active_cost_profile_name", "lane_follow")
        )
        for row in feasible_rows:
            if id(row) not in selected_ids:
                mark_mpc_probe_skipped(row)
                continue
            intent = getattr(row, "intent", None)
            destination = list(getattr(row, "destination_state", []) or [])
            reference = [
                dict(sample)
                for sample in list(getattr(row, "lane_center_reference", []) or [])
            ]
            cache_key = (
                str(getattr(intent, "name", "")),
                str(getattr(intent, "decision", "")),
                int(getattr(intent, "target_lane_id", 0) or 0),
                str(getattr(intent, "trajectory_variant", "")),
                round(float(getattr(intent, "lane_change_duration_s", 0.0) or 0.0), 2),
            )
            probe = self._candidate_mpc_probe_cache.get(cache_key)
            if probe is None:
                probe_profile = _mpc_cost_profile_for_behavior(
                    behavior=str(getattr(intent, "decision", "")),
                    planner_lc_state=(
                        "EXECUTE_LANE_CHANGE"
                        if str(getattr(intent, "decision", "")).startswith(
                            "lane_change"
                        )
                        else "LANE_KEEP"
                    ),
                    planner_mode="NORMAL",
                    next_macro_maneuver="straight",
                )
                if hasattr(self.mpc, "apply_mode_cost_profile"):
                    self.mpc.apply_mode_cost_profile(
                        str(probe_profile),
                        blend_alpha=1.0,
                    )
                probe = self.mpc.probe_trajectory_feasibility(
                    current_state=current_state,
                    destination_state=destination,
                    object_snapshots=object_snapshots,
                    current_acceleration_mps2=float(self._last_accel_mps2),
                    current_steering_rad=float(self._last_steer_rad),
                    lane_center_reference_samples=reference,
                    stop_goal_active=bool(
                        getattr(intent, "stop_goal_active", False)
                    ),
                    road_envelope_payload_world=(
                        road_envelope_payload_world
                        if str(getattr(intent, "name", ""))
                        == "committed_lane_change_continuation"
                        else None
                    ),
                )
                self._candidate_mpc_probe_cache[cache_key] = dict(probe)
            apply_mpc_probe_result(
                candidate=row,
                solved=bool(probe.get("solved", False)),
                status=str(probe.get("status", "")),
                solve_time_ms=float(probe.get("solve_time_ms", 0.0) or 0.0),
                dynamic_cost=float(probe.get("dynamic_cost", 0.0) or 0.0),
            )
            probe_rows.append(
                "%s:%s"
                % (
                    str(getattr(intent, "name", "")),
                    str(probe.get("status", "")),
                )
            )
        if hasattr(self.mpc, "apply_mode_cost_profile"):
            self.mpc.apply_mode_cost_profile(
                str(previous_profile),
                blend_alpha=1.0,
            )
        return "|".join(probe_rows) if probe_rows else "mpc_probe_no_feasible_top_k"

    def _carla_waypoint_turn_reference(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        current_state: Sequence[float],
        current_lane_id: int,
        target_lane_id: int,
        target_speed_mps: float,
        destination_state: Sequence[float] | None,
    ) -> tuple[list[dict[str, object]], list[float], str]:
        """Build a turn horizon from the CARLA GRP-selected connector."""

        turn_speed_mps = min(
            max(0.4, float(target_speed_mps)),
            float(self.config.get("carla_waypoint_turn_speed_cap_mps", 2.2)),
        )
        step_distance_m = max(
            float(self.config.get("carla_waypoint_turn_min_step_m", 0.35)),
            float(self.mpc.dt_s) * max(0.8, float(turn_speed_mps)),
        )
        reference, reason = self.route_manager.carla_waypoint_reference(
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
            horizon_steps=int(self.mpc.horizon_steps),
            step_distance_m=float(step_distance_m),
            target_speed_mps=float(turn_speed_mps),
            fallback_lane_id=int(target_lane_id or current_lane_id),
            # The GRP junction connector is the turn path. Re-anchoring it to
            # ego every frame feeds tracking error back into the next
            # reference and can make the vehicle cut across the lane boundary.
            anchor_to_ego_heading=bool(
                self.config.get(
                    "carla_waypoint_turn_anchor_to_ego_heading",
                    False,
                )
            ),
            ego_anchor_distance_m=float(
                self.config.get("carla_waypoint_turn_connector_distance_m", 7.0)
            ),
            allow_route_rejoin=False,
            # Turn connectors are short curved segments. A long MPC horizon
            # can request more lookahead distance than the real connector
            # has, and carla_waypoint_reference falls back to extrapolating
            # straight ahead past the real curve when that happens -- bound
            # how far that fallback is allowed to drift so MPC never chases
            # a fictional straight tail instead of the real turn.
            max_extrapolation_m=float(
                self.config.get("carla_waypoint_turn_max_extrapolation_m", 2.0)
            ),
            # The parked tail has no real geometry behind it, so don't force
            # a normal-width corridor onto it -- a road-boundary constraint
            # built from a wrong-for-that-point corridor is what turned into
            # a genuine QP infeasibility once the vehicle got far enough
            # into the turn for its horizon to reach this tail.
            extrapolated_lane_width_m=float(
                self.config.get("carla_waypoint_turn_extrapolated_lane_width_m", 8.0)
            ),
        )
        if not reference:
            return [], list(destination_state or []), str(reason)
        reference, curvature_reason = (
            self.reference_generator.curvature_feasible_turn_samples(
                reference_samples=reference,
                ego_location=ego_location,
                ego_heading_rad=float(ego_yaw_rad),
                max_curvature_1pm=float(
                    self.config.get(
                        "reference_vehicle_max_curvature_1pm",
                        0.20,
                    )
                ),
            )
        )
        if curvature_reason:
            reason = ";".join(
                item for item in (str(reason), str(curvature_reason)) if item
            )

        from cpx_planning.behavior_planner.reference_pipeline import (
            lane_center_destination_from_reference_arc_length,
        )

        seed_destination = list(destination_state or [])
        if len(seed_destination) < 5:
            seed_destination = [
                float(current_state[0]),
                float(current_state[1]),
                float(turn_speed_mps),
                float(current_state[3]),
                int(target_lane_id or current_lane_id),
            ]
        seed_destination[2] = float(turn_speed_mps)
        destination = lane_center_destination_from_reference_arc_length(
            destination_state=seed_destination,
            lane_center_reference=reference,
            target_arc_length_m=float(
                self.config.get("carla_waypoint_turn_destination_arc_m", 4.5)
            ),
        ) or seed_destination
        return [dict(sample) for sample in reference], list(destination), str(reason)

    def _explicit_fallback_candidate_for_mpc(
        self,
        *,
        candidate_results: Sequence[object],
        baseline_decision: str,
        baseline_target_lane_id: int,
        current_lane_id: int,
        current_state: Sequence[float],
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        summarize_candidate_results: Any,
        maneuver_commitment: Any = None,
        selection_reason: str = "",
    ) -> tuple[str, int, float, list[dict[str, object]], list[float], dict[str, object]]:
        """Return an explicit fallback candidate when every candidate is invalid."""

        route_replan_attempted = False
        route_replan_succeeded = False

        stop_like = str(baseline_decision or "").strip().lower() in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }
        turn_like = str(baseline_decision or "").strip().lower() in {
            "intersection_turn_left",
            "intersection_turn_right",
        }
        committed_lane_change = bool(
            maneuver_commitment is not None
            and bool(getattr(maneuver_commitment, "active", False))
        )
        hard_safety_veto = any(
            "collision_risk" in str(getattr(candidate, "feasibility_reason", ""))
            for candidate in list(candidate_results or [])
        )
        selected_lane_id = int(current_lane_id)
        turn_reference_reason = ""
        if bool(stop_like):
            generated_fallback = self.reference_generator.emergency_stop_reference(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_lane_id=int(current_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                step_distance_m=max(0.5, float(self.mpc.dt_s) * 0.8),
            )
            reference = generated_fallback.samples
            destination = generated_fallback.destination_state
            decision = "emergency_brake"
            speed_mps = 0.0
            selected_name = "explicit_fallback_emergency_stop"
            source = "explicit_fallback_ego_heading_stop"
        elif bool(turn_like):
            fallback_turn_speed_mps = float(
                self.config.get("strict_turn_fallback_speed_mps", 0.8)
            )
            turn_exit_stabilization_active = (
                str(getattr(getattr(self, "_scenario_manager", None), "state", ""))
                .strip()
                .upper()
                == "TURN_EXIT_STABILIZATION"
            )
            retained_turn_reference = (
                self.maneuver_manager.retained_turn_continuation(
                    ego_x_m=float(ego_location.x),
                    ego_y_m=float(ego_location.y),
                    target_speed_mps=float(
                        max(
                            fallback_turn_speed_mps,
                            float(
                                getattr(
                                    getattr(self, "_scenario_manager", None),
                                    "turn_speed_cap_mps",
                                    fallback_turn_speed_mps,
                                )
                            ),
                        )
                    ),
                    count=int(self.mpc.horizon_steps),
                )
                if bool(turn_exit_stabilization_active)
                and not bool(hard_safety_veto)
                else []
            )
            if retained_turn_reference:
                terminal = dict(retained_turn_reference[-1])
                speed_mps = float(
                    retained_turn_reference[0].get(
                        "speed_ref_mps",
                        getattr(
                            getattr(self, "_scenario_manager", None),
                            "turn_speed_cap_mps",
                            fallback_turn_speed_mps,
                        ),
                    )
                )
                reference = [dict(sample) for sample in retained_turn_reference]
                destination = [
                    float(terminal.get("x_ref_m", terminal.get("x", current_state[0]))),
                    float(terminal.get("y_ref_m", terminal.get("y", current_state[1]))),
                    float(speed_mps),
                    float(terminal.get("heading_rad", current_state[3])),
                    int(baseline_target_lane_id or current_lane_id),
                ]
                decision = str(baseline_decision)
                selected_lane_id = int(baseline_target_lane_id or current_lane_id)
                selected_name = "retained_turn_exit_continuation"
                source = "unified_maneuver_turn_exit_continuation"
                turn_reference_reason = "retained_turn_exit_geometry"
            else:
                reference = []
                destination = []
            reference, destination, turn_reason = self._carla_waypoint_turn_reference(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_state=current_state,
                current_lane_id=int(current_lane_id),
                target_lane_id=int(baseline_target_lane_id or current_lane_id),
                target_speed_mps=float(fallback_turn_speed_mps),
                destination_state=None,
            ) if not retained_turn_reference else (reference, destination, "")
            if not retained_turn_reference:
                turn_reference_reason = str(turn_reason)
            turn_contract = None
            if reference and not retained_turn_reference:
                turn_contract = self._validate_candidate_reference_contract(
                    decision=str(baseline_decision),
                    lc_state=(
                        "INTERSECTION_TURN_LEFT"
                        if str(baseline_decision).endswith("_left")
                        else "INTERSECTION_TURN_RIGHT"
                    ),
                    current_lane_id=int(current_lane_id),
                    speed_ref_mps=float(fallback_turn_speed_mps),
                    stop_goal_active=False,
                    current_state=current_state,
                    destination_state=destination,
                    lane_center_reference=reference,
                )
            if (
                not retained_turn_reference
                and (
                    not reference
                    or turn_contract is None
                    or not turn_contract.valid
                )
                and bool(
                self.config.get("turn_route_replan_enabled", True)
                )
            ):
                (
                    route_replan_attempted,
                    route_replan_succeeded,
                    route_replan_reason,
                ) = self._attempt_turn_route_replan(
                    ego_location=ego_location,
                )
                turn_reference_reason = ";".join(
                    reason
                    for reason in (
                        str(turn_reference_reason),
                        str(route_replan_reason),
                    )
                    if reason
                )
                if bool(route_replan_succeeded):
                    reference, destination, retry_reason = (
                        self._carla_waypoint_turn_reference(
                            ego_location=ego_location,
                            ego_yaw_rad=float(ego_yaw_rad),
                            current_state=current_state,
                            current_lane_id=int(current_lane_id),
                            target_lane_id=int(
                                baseline_target_lane_id or current_lane_id
                            ),
                            target_speed_mps=float(fallback_turn_speed_mps),
                            destination_state=None,
                        )
                    )
                    turn_reference_reason = ";".join(
                        reason
                        for reason in (
                            str(turn_reference_reason),
                            f"route_replan_retry:{str(retry_reason)}",
                        )
                        if reason
                    )
                    turn_contract = None
                    if reference:
                        turn_contract = self._validate_candidate_reference_contract(
                            decision=str(baseline_decision),
                            lc_state=(
                                "INTERSECTION_TURN_LEFT"
                                if str(baseline_decision).endswith("_left")
                                else "INTERSECTION_TURN_RIGHT"
                            ),
                            current_lane_id=int(current_lane_id),
                            speed_ref_mps=float(fallback_turn_speed_mps),
                            stop_goal_active=False,
                            current_state=current_state,
                            destination_state=destination,
                            lane_center_reference=reference,
                        )
            if (
                bool(retained_turn_reference)
                or (
                    reference
                    and turn_contract is not None
                    and bool(turn_contract.valid)
                    and not bool(hard_safety_veto)
                )
            ):
                decision = str(baseline_decision)
                if not retained_turn_reference:
                    speed_mps = float(fallback_turn_speed_mps)
                selected_lane_id = int(baseline_target_lane_id or current_lane_id)
                if not retained_turn_reference:
                    selected_name = "explicit_fallback_carla_route_turn"
                    source = "explicit_fallback_carla_grp_waypoint_turn"
            else:
                if bool(hard_safety_veto):
                    turn_reference_reason = (
                        str(turn_reference_reason) + ":candidate_collision_risk_veto"
                    )
                if turn_contract is not None and not bool(turn_contract.valid):
                    turn_reference_reason = (
                        str(turn_reference_reason)
                        + ":contract_invalid:"
                        + str(turn_contract.reason())
                    )
                generated_fallback = self.reference_generator.emergency_stop_reference(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    current_lane_id=int(current_lane_id),
                    horizon_steps=int(self.mpc.horizon_steps),
                    step_distance_m=max(0.5, float(self.mpc.dt_s) * 0.8),
                )
                reference = generated_fallback.samples
                destination = generated_fallback.destination_state
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
        elif bool(committed_lane_change):
            committed_speed_mps = max(
                0.5,
                float(
                    getattr(
                        self,
                        "_route_tracking_lane_change_target_speed_mps",
                        0.0,
                    )
                    or self.strict_explicit_fallback_speed_mps
                ),
            )
            step_distance_m = max(
                0.1,
                float(self.mpc.dt_s) * float(committed_speed_mps),
            )
            reference, commitment_window_reason = (
                self._route_tracking_lane_change_window(
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                    target_speed_mps=float(committed_speed_mps),
                    step_distance_m=float(step_distance_m),
                )
            )
            reference_valid, commitment_validation_reason = (
                self._validate_route_tracking_lane_change_reference(
                    reference=reference,
                    ego_location=ego_location,
                    ego_yaw_rad=float(ego_yaw_rad),
                )
            )
            turn_reference_reason = ";".join(
                reason
                for reason in (
                    str(commitment_window_reason),
                    str(commitment_validation_reason),
                )
                if reason
            )
            # Candidate feasibility is what brought us into this fallback.
            # Requiring that same candidate to be feasible here made the
            # independently rebuilt and validated locked window unreachable,
            # so a one-frame contract/transient failure always became a full
            # emergency stop.  The locked window validation and hard safety
            # veto below are the authoritative continuation gates.
            if (
                reference
                and bool(reference_valid)
                and not bool(hard_safety_veto)
            ):
                self._route_tracking_lane_change_commitment_invalid_frames = 0
                terminal = dict(reference[-1])
                selected_lane_id = int(
                    getattr(maneuver_commitment, "target_lane_id", current_lane_id)
                    or current_lane_id
                )
                committed_speed_mps = max(
                    0.5,
                    float(
                        reference[0].get(
                            "speed_ref_mps",
                            reference[0].get("v_ref_mps", committed_speed_mps),
                        )
                    ),
                )
                destination = [
                    float(terminal.get("x_ref_m", terminal.get("x", current_state[0]))),
                    float(terminal.get("y_ref_m", terminal.get("y", current_state[1]))),
                    float(committed_speed_mps),
                    float(terminal.get("heading_rad", current_state[3])),
                    int(selected_lane_id),
                ]
                decision = str(getattr(maneuver_commitment, "decision", ""))
                speed_mps = float(committed_speed_mps)
                selected_name = "committed_lane_change_continuation"
                source = (
                    "target_lane_stabilization_reference"
                    if str(
                        getattr(
                            self,
                            "_route_tracking_lane_change_phase",
                            "executing",
                        )
                    )
                    == "target_lane_stabilization"
                    else "locked_quintic_lane_change_reference"
                )
            else:
                turn_reference_reason = (
                    str(turn_reference_reason) + ";"
                    if str(turn_reference_reason)
                    else ""
                ) + "committed_candidate_hard_gate"
                self._route_tracking_lane_change_commitment_invalid_frames = (
                    int(
                        getattr(
                            self,
                            "_route_tracking_lane_change_commitment_invalid_frames",
                            0,
                        )
                    )
                    + 1
                )
                commitment_invalid_timeout_frames = max(
                    1,
                    int(
                        self.config.get(
                            "lane_change_commitment_invalid_timeout_frames",
                            40,
                        )
                    ),
                )
                # A commitment stuck contract-invalid tick after tick (e.g. the
                # locked window's progress index drifts ahead of ego while
                # blocked by traffic) has no other release path -- without this
                # timeout it falls into the emergency-stop branch below forever,
                # since _release_completed_lane_change_commitment only fires once
                # ego actually reaches the target lane. Abandon it like a
                # stabilization timeout: reset now so the next tick's candidate
                # generation stops re-proposing this stale commitment even if
                # the recovery reference below happens to fail this tick too.
                stale_commitment_abandoned = bool(
                    int(self._route_tracking_lane_change_commitment_invalid_frames)
                    > int(commitment_invalid_timeout_frames)
                )
                if bool(stale_commitment_abandoned):
                    turn_reference_reason += ":commitment_invalid_timeout_abandoned"
                    self._reset_route_tracking_lane_change_reference()
                stabilization_recovery = bool(
                    bool(stale_commitment_abandoned)
                    or (
                        str(
                            getattr(
                                self,
                                "_route_tracking_lane_change_phase",
                                "",
                            )
                        )
                        == "target_lane_stabilization"
                        and int(current_lane_id)
                        == int(
                            getattr(
                                self,
                                "_route_tracking_lane_change_target_lane_id",
                                0,
                            )
                        )
                    )
                )
                recovery_reference = []
                if bool(stabilization_recovery):
                    recovery_reference = (
                        self.reference_generator.lane_recovery_samples(
                            ego_location=ego_location,
                            ego_yaw_rad=float(ego_yaw_rad),
                            start_waypoint=self.reference_generator.map_waypoint(
                                ego_location
                            ),
                            current_lane_id=int(current_lane_id),
                            horizon_steps=int(self.mpc.horizon_steps),
                            step_distance_m=max(
                                0.25,
                                float(self.mpc.dt_s) * 1.0,
                            ),
                            route_points=[],
                        )
                    )
                    recovery_reference, recovery_curvature_reason = (
                        self.reference_generator.curvature_feasible_samples(
                            reference_samples=recovery_reference,
                            ego_location=ego_location,
                            ego_heading_rad=float(ego_yaw_rad),
                            max_curvature_1pm=float(
                                self.config.get(
                                    "reference_vehicle_max_curvature_1pm",
                                    0.20,
                                )
                            ),
                            mode="target_lane_stabilization_recovery",
                        )
                    )
                    if recovery_reference:
                        terminal = dict(recovery_reference[-1])
                        speed_mps = min(
                            1.0,
                            max(0.5, float(committed_speed_mps)),
                        )
                        destination = [
                            float(
                                terminal.get(
                                    "x_ref_m",
                                    terminal.get("x", current_state[0]),
                                )
                            ),
                            float(
                                terminal.get(
                                    "y_ref_m",
                                    terminal.get("y", current_state[1]),
                                )
                            ),
                            float(speed_mps),
                            float(
                                terminal.get(
                                    "heading_rad",
                                    current_state[3],
                                )
                            ),
                            int(current_lane_id),
                        ]
                        reference = [
                            dict(sample) for sample in recovery_reference
                        ]
                        decision = "lane_follow"
                        selected_name = (
                            "target_lane_stabilization_recovery"
                        )
                        source = "target_lane_center_recovery_reference"
                        turn_reference_reason += (
                            ";target_lane_recovery:"
                            + str(
                                recovery_curvature_reason
                                or "curvature_valid"
                            )
                        )
                        completed_option = str(
                            self._route_tracking_lane_change_option
                        )
                        self._route_tracking_lane_change_completed_option = (
                            completed_option
                        )
                        self._reset_route_tracking_lane_change_reference()
                if not recovery_reference:
                    generated_fallback = (
                        self.reference_generator.emergency_stop_reference(
                            ego_location=ego_location,
                            ego_yaw_rad=float(ego_yaw_rad),
                            current_lane_id=int(current_lane_id),
                            horizon_steps=int(self.mpc.horizon_steps),
                            step_distance_m=max(
                                0.5,
                                float(self.mpc.dt_s) * 0.8,
                            ),
                        )
                    )
                    reference = generated_fallback.samples
                    destination = generated_fallback.destination_state
                    decision = "emergency_brake"
                    speed_mps = 0.0
                    selected_name = (
                        "committed_lane_change_invalid_emergency_stop"
                    )
                    source = "explicit_fallback_ego_heading_stop"
        else:
            start_waypoint = self._map_waypoint_from_location(ego_location)
            step_distance_m = max(
                0.5,
                float(self.mpc.dt_s)
                * max(0.5, float(self.strict_explicit_fallback_speed_mps)),
            )
            reference = self.reference_generator.lane_center_samples(
                start_waypoint=start_waypoint,
                current_lane_id=int(current_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                step_distance_m=float(step_distance_m),
                route_points=self._active_global_route_points(),
            )
            if not reference:
                reference = self.reference_generator.straight_samples(
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
        reason = str(selection_reason or "all_candidates_infeasible")
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
            "route_replan_attempted": bool(route_replan_attempted),
            "route_replan_succeeded": bool(route_replan_succeeded),
        }
        debug.update(
            dict(
                getattr(
                    self,
                    "_route_tracking_lane_change_completion_debug",
                    {},
                )
            )
        )
        if maneuver_commitment is not None:
            debug.update(maneuver_commitment.as_debug_fields())
        return (
            str(decision),
            int(selected_lane_id),
            float(speed_mps),
            [dict(sample) for sample in list(reference or [])],
            list(destination or []),
            debug,
        )

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

    @staticmethod
    def _normalized_final_lc_state(
        *, decision: str, lc_state: str, lane_change_phase: str = ""
    ) -> str:
        """Keep the public FSM consistent with the final selected action."""

        normalized_decision = str(decision or "").strip().lower()
        normalized_phase = str(lane_change_phase or "").strip().lower()
        if normalized_decision in {"lane_change_left", "lane_change_right"}:
            if normalized_phase == "target_lane_stabilization":
                return "TARGET_LANE_STABILIZATION"
            return (
                "EXECUTE_LANE_CHANGE_LEFT"
                if normalized_decision == "lane_change_left"
                else "EXECUTE_LANE_CHANGE_RIGHT"
            )
        return str(lc_state or "LANE_KEEP")

    @staticmethod
    def _route_option_turn_decision(*, current_road_option: str, next_macro_maneuver: str) -> str:
        route_option = str(current_road_option or "").strip().upper()
        macro = (
            str(next_macro_maneuver or "")
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )
        macro_tokens = set(macro.split())
        if route_option == "LEFT" or {"left", "turn"}.issubset(macro_tokens):
            return "intersection_turn_left"
        if route_option == "RIGHT" or {"right", "turn"}.issubset(macro_tokens):
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
            next_macro_maneuver="",
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
        ego_location: PlannerLocation,
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
        base_s = self.reference_generator.project_to_route_s(
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
        return f"mode_transition:{previous_key}->{mode_key}:reset_control_buffer"

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
        if normalized_decision == "route_recovery":
            return "route_recovery"
        if normalized_decision in {"lane_change_left", "lane_change_right"}:
            return f"{normalized_decision}:{int(target_lane_id)}"
        if normalized_fsm.startswith("EXECUTE_LANE_CHANGE"):
            return f"{normalized_fsm.lower()}:{int(target_lane_id)}"
        return f"lane_follow:{int(target_lane_id)}"

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
        direct_target_tracking_enabled = bool(
            self.config.get(
                "route_tracking_lane_change_direct_target_tracking_enabled",
                False,
            )
        )
        contract_mode = (
            "emergency_stop"
            if normalized_decision == "emergency_brake"
            else "stop"
            if bool(stop_like)
            else "lane_change_direct"
            if bool(lane_change_active) and bool(direct_target_tracking_enabled)
            else "lane_change"
            if bool(lane_change_active)
            else "intersection_turn"
            if bool(turn_active)
            else "lane_follow"
        )
        expected_lane_id = int(current_lane_id)
        if bool(lane_change_active) and len(destination_state or []) >= 5:
            try:
                expected_lane_id = int(float(destination_state[4]))
            except (TypeError, ValueError):
                expected_lane_id = int(current_lane_id)
        contract = contract_from_config(
            mode=str(contract_mode),
            expected_lane_id=int(expected_lane_id),
            horizon_steps=int(self.mpc.horizon_steps),
            config=dict(self.config),
            default_speed_mps=max(float(self.target_speed_mps), float(speed_ref_mps), 0.1),
        )
        recovery_reference_active = any(
            str(sample.get("lane_transition_kind", ""))
            == "ego_anchored_lane_recovery"
            for sample in list(lane_center_reference or [])[:2]
        )
        validation = validate_reference_contract(
            reference_samples=lane_center_reference,
            destination_state=destination_state,
            ego_state=current_state,
            contract=contract,
            check_destination_body_lateral=not bool(
                lane_change_active or turn_active or recovery_reference_active
            ),
        )
        if (
            bool(validation.valid)
            and bool(turn_active)
            and bool(
                self.config.get(
                    "reference_contract_turn_vehicle_footprint_enabled",
                    True,
                )
            )
        ):
            boundary_valid, boundary_reason = (
                self._reference_vehicle_footprint_boundary_valid(
                    reference_samples=lane_center_reference,
                )
            )
            if not bool(boundary_valid):
                validation.valid = False
                validation.violations.append(str(boundary_reason))
        return validation

    def _reference_vehicle_footprint_boundary_valid(
        self,
        *,
        reference_samples: Sequence[Mapping[str, object]],
    ) -> tuple[bool, str]:
        """Delegate candidate turn-corridor validation to ReferenceGenerator."""

        ego_half_width_m = float(self.config.get("metrics_ego_half_width_m", self.config.get("reference_vehicle_half_width_m", 1.0)))
        ego_half_length_m = float(self.config.get("reference_vehicle_half_length_m", 2.4))
        safety_margin_m = max(
            0.0,
            float(
                self.config.get(
                    "reference_contract_turn_boundary_margin_m",
                    0.15,
                )
            ),
        )
        max_failures = max(
            0,
            int(
                self.config.get(
                    "reference_contract_turn_max_boundary_failures",
                    1,
                )
            ),
        )
        validation = self.reference_generator.validate_turn_swept_footprint(
            reference_samples=reference_samples,
            ego_half_width_m=max(0.1, float(ego_half_width_m)),
            ego_half_length_m=max(0.1, float(ego_half_length_m)),
            safety_margin_m=float(safety_margin_m),
            max_violations=int(max_failures),
        )
        return bool(validation.valid), (
            "" if bool(validation.valid) else str(validation.reason)
        )

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
        normalized_behavior = str(behavior_decision or "").strip().lower()
        if normalized_behavior == "emergency_brake":
            return f"candidate_hard_gate:{selected}:emergency_brake_direct_control"
        if "stop_missing_target_hard_lock" in combined_reason:
            return f"candidate_hard_gate:{selected}:{combined_reason}"
        if "strict_reference_veto" in combined_reason:
            return f"candidate_hard_gate:{selected}:{combined_reason}"
        if "final_reference_gate:" in combined_reason:
            return f"candidate_hard_gate:{selected}:{combined_reason}"
        if status != "infeasible":
            return ""
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
            "final_reference_gate:",
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

    def _sim_time_s(self) -> float:
        output = getattr(self, "last_adapter_output", None)
        if output is not None:
            return float(output.frame.planning.sim_time_s)
        adapter = getattr(self, "input_adapter", None)
        latest_timestamp_s = getattr(adapter, "latest_timestamp_s", None)
        return 0.0 if not callable(latest_timestamp_s) else float(latest_timestamp_s())

    def _obstacle_lane_step_fn(self):
        """Return a ``(x, y, distance_m) -> (x, y, heading_rad) | None``
        closure for lane-curve-aware obstacle prediction, or None to keep the
        old straight-line-only fallback.

        ``obstacle_future_trajectory`` (behavior_planner/trajectory_risk.py)
        only follows the lane centerline when given this closure; without
        it, every obstacle without a CP-supplied ``predicted_trajectory``
        keeps being extrapolated as a straight line at its current heading,
        which is wrong for a vehicle following a curved lane (e.g. mid-turn
        at an intersection).
        """

        if not bool(self.config.get("prediction_lane_following_enabled", True)):
            return None
        from cpx_planning.utility.global_planner import lane_step_xy_heading

        get_waypoint_fn = self.reference_map.get_waypoint

        def _step(x_m: float, y_m: float, distance_m: float):
            result = lane_step_xy_heading(
                float(x_m),
                float(y_m),
                float(distance_m),
                get_waypoint_fn=get_waypoint_fn,
            )
            if result is None:
                self._prediction_lane_step_none_count += 1
            else:
                self._prediction_lane_step_resolved_count += 1
            return result

        return _step

    def _assign_obstacles_to_lanes(
        self,
        object_snapshots: Sequence[Mapping[str, Any]],
        *,
        ego_waypoint: Any = None,
        ego_lane_id: int = 0,
    ) -> dict[str, int]:
        from cpx_planning.utility.lane_graph import lane_hop_offset
        from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint

        assignments: dict[str, int] = {}
        for snapshot in list(object_snapshots or []):
            obstacle_id = self._object_track_id(snapshot)
            if not obstacle_id:
                continue
            waypoint = self.reference_map.get_waypoint({
                "x": float(snapshot.get("x", 0.0)),
                "y": float(snapshot.get("y", 0.0)),
                "z": float(snapshot.get("z", 0.0)),
            })
            # A fresh per-tick canonical recount can disagree with ego's own
            # StableLaneIdTracker-held id even when the obstacle physically
            # shares ego's lane (e.g. just past a road/section boundary the
            # tracker bridged but a bare recount has not re-settled on) --
            # front-gap/lane-safety lookups key off ego's lane id, so a
            # same-lane obstacle assigned a different number silently drops
            # out of them. Prefer proven real adjacency to ego over the
            # recount whenever it is available.
            lane_id = 0
            if ego_waypoint is not None and int(ego_lane_id) != 0:
                if lane_hop_offset(ego_waypoint, waypoint) == 0:
                    lane_id = int(ego_lane_id)
            if lane_id == 0:
                lane_id = int(canonical_lane_id_for_waypoint(waypoint))
            if int(lane_id) != 0:
                assignments[obstacle_id] = int(lane_id)
        return assignments

    @staticmethod
    def _object_track_id(snapshot: Mapping[str, Any]) -> str:
        for key in ("track_id", "object_id", "vehicle_id", "actor_id", "id"):
            value = snapshot.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        try:
            return "xy:{:.1f}:{:.1f}".format(
                float(snapshot.get("x", snapshot.get("x_m", 0.0))),
                float(snapshot.get("y", snapshot.get("y_m", 0.0))),
            )
        except Exception:
            return ""

    @staticmethod
    def _nearest_front_distance_by_lane(
        *,
        ego_snapshot: Mapping[str, object],
        obstacle_snapshots: Sequence[Mapping[str, Any]],
        lane_assignments: Mapping[str, int],
        available_lane_ids: Sequence[int],
    ) -> dict[int, float]:
        ego_x = float(ego_snapshot.get("x", 0.0))
        ego_y = float(ego_snapshot.get("y", 0.0))
        ego_psi = float(ego_snapshot.get("psi", 0.0))
        cos_h = math.cos(ego_psi)
        sin_h = math.sin(ego_psi)
        nearest: dict[int, float] = {}
        allowed = {int(lane_id) for lane_id in list(available_lane_ids or [])}
        for snapshot in list(obstacle_snapshots or []):
            obstacle_id = str(snapshot.get("vehicle_id", snapshot.get("id", ""))).strip()
            lane_id = int(lane_assignments.get(obstacle_id, 0))
            if lane_id not in allowed:
                continue
            dx = float(snapshot.get("x", 0.0)) - ego_x
            dy = float(snapshot.get("y", 0.0)) - ego_y
            longitudinal = dx * cos_h + dy * sin_h
            if longitudinal <= 0.0:
                continue
            nearest[lane_id] = min(float(nearest.get(lane_id, float("inf"))), float(longitudinal))
        return {
            int(lane_id): float(distance)
            for lane_id, distance in nearest.items()
            if math.isfinite(float(distance))
        }

    @classmethod
    def _nearest_front_obstacle_by_lane(
        cls,
        *,
        ego_snapshot: Mapping[str, object],
        obstacle_snapshots: Sequence[Mapping[str, Any]],
        lane_assignments: Mapping[str, int],
        available_lane_ids: Sequence[int],
    ) -> dict[int, dict[str, Any]]:
        """Return the nearest complete front-obstacle record per lane."""

        ego_x = float(ego_snapshot.get("x", 0.0))
        ego_y = float(ego_snapshot.get("y", 0.0))
        ego_psi = float(ego_snapshot.get("psi", 0.0))
        cos_h = math.cos(ego_psi)
        sin_h = math.sin(ego_psi)
        allowed = {int(lane_id) for lane_id in list(available_lane_ids or [])}
        nearest: dict[int, dict[str, Any]] = {}
        for raw_snapshot in list(obstacle_snapshots or []):
            snapshot = dict(raw_snapshot)
            obstacle_id = cls._object_track_id(snapshot)
            lane_id = int(lane_assignments.get(str(obstacle_id), 0))
            if lane_id not in allowed:
                continue
            obstacle_x = float(snapshot.get("x", snapshot.get("x_m", 0.0)))
            obstacle_y = float(snapshot.get("y", snapshot.get("y_m", 0.0)))
            longitudinal_m = (
                (obstacle_x - ego_x) * cos_h
                + (obstacle_y - ego_y) * sin_h
            )
            if longitudinal_m <= 0.0:
                continue
            previous = nearest.get(int(lane_id))
            if previous is not None and float(
                previous.get("front_distance_m", float("inf"))
            ) <= float(longitudinal_m):
                continue
            snapshot["vehicle_id"] = str(obstacle_id)
            snapshot["x"] = float(obstacle_x)
            snapshot["y"] = float(obstacle_y)
            snapshot["v"] = max(
                0.0,
                float(snapshot.get("v", snapshot.get("speed_mps", 0.0))),
            )
            snapshot["front_distance_m"] = float(longitudinal_m)
            nearest[int(lane_id)] = snapshot
        return nearest

    def _attempt_static_obstacle_route_replan(
        self,
        *,
        ego_location: Any,
        obstacle: Mapping[str, object],
    ) -> tuple[bool, bool, str]:
        """Block the obstacle lane and atomically rebuild the active route."""

        now_s = float(self._sim_time_s())
        cooldown_s = max(
            0.1,
            float(
                self.config.get(
                    "static_obstacle_replan_cooldown_s",
                    self.behavior_runtime_cfg.get(
                        "static_obstacle_replan_cooldown_s",
                        2.0,
                    ),
                )
            ),
        )
        elapsed_s = now_s - float(self._static_obstacle_replan_last_attempt_s)
        if elapsed_s < cooldown_s:
            reason = "static_obstacle_replan_cooldown:remaining={:.2f}".format(
                cooldown_s - elapsed_s
            )
            self._static_obstacle_replan_reason = str(reason)
            return False, False, str(reason)

        self._static_obstacle_replan_last_attempt_s = float(now_s)
        block_fn = getattr(self.global_planner, "block_lane_at_position", None)
        if not callable(block_fn):
            reason = "static_obstacle_block_lane_unsupported"
            self._static_obstacle_replan_reason = str(reason)
            return True, False, str(reason)
        blocked_lane_id = block_fn({
            "x": float(obstacle.get("x", obstacle.get("x_m", 0.0))),
            "y": float(obstacle.get("y", obstacle.get("y_m", 0.0))),
            "z": float(obstacle.get("z", obstacle.get("z_m", 0.0))),
        })
        if blocked_lane_id is None:
            reason = "static_obstacle_lane_mapping_failed"
            self._static_obstacle_replan_reason = str(reason)
            return True, False, str(reason)
        self._static_obstacle_blocked_lane_id = blocked_lane_id

        result = self.route_manager.replan_from(
            start_point={
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "z": float(getattr(ego_location, "z", 0.0)),
            },
            trigger_reason="static_obstacle",
        )
        self._static_obstacle_replan_reason = str(result.reason)
        if not bool(result.success):
            return True, False, str(result.reason)

        self._active_route_summary = self.route_manager.active_route_summary
        self._temporary_destination_state = None
        self._previous_lane_center_reference = []
        self._lane_reference_freeze_count = 0
        self._reset_route_tracking_lane_change_reference()
        self._last_required_lane_change_target_lane_id = None
        self._last_required_lane_change_target_ad_lane_id = None
        self._lane_id_tracker.reset()
        maneuver_manager = getattr(self, "maneuver_manager", None)
        if maneuver_manager is not None:
            maneuver_manager.reset(reason="static_obstacle_route_replanned")
        self.control_buffer.reset(reason="static_obstacle_route_replanned")
        if hasattr(self.mpc, "clear_previous_solution_seed"):
            self.mpc.clear_previous_solution_seed()
        return True, True, str(result.reason)

    def _load_cp_message_payload(self) -> dict[str, Any]:
        try:
            with open(self.cp_message_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return dict(payload or {})
        except Exception:
            return {}

    @staticmethod
    def _traffic_context_from_cp_control(
        *,
        selected_control: Mapping[str, object] | None,
        ego_location: PlannerLocation,
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        if not isinstance(selected_control, Mapping):
            return {"signal_state": "unknown", "from_cp": False}, None
        state = str(
            selected_control.get(
                "signal_state",
                selected_control.get("state", "unknown"),
            )
            or "unknown"
        ).strip().lower()
        stop_line = selected_control.get("stop_line_position", selected_control.get("stop_line", None))
        stop_target = None
        if isinstance(stop_line, Mapping):
            x_value = stop_line.get("x", stop_line.get("x_m", None))
            y_value = stop_line.get("y", stop_line.get("y_m", None))
            if x_value is not None and y_value is not None:
                distance_m = math.hypot(float(x_value) - float(ego_location.x), float(y_value) - float(ego_location.y))
                stop_target = {
                    "x_m": float(x_value),
                    "y_m": float(y_value),
                    "lane_id": int(float(
                        selected_control.get("lane_id", stop_line.get("lane_id", 0)) or 0
                    )),
                    "road_id": int(float(
                        selected_control.get("road_id", stop_line.get("road_id", 0)) or 0
                    )),
                    "distance_m": float(distance_m),
                    "source": "opencda_cp_control",
                }
        context = {
            "signal_state": str(state),
            "signal_source": str(selected_control.get("source", "opencda_cp")),
            "source": str(selected_control.get("source", "opencda_cp")),
            "cp_control_id": str(selected_control.get("control_id", selected_control.get("id", ""))),
            "control_id": str(selected_control.get("control_id", selected_control.get("id", ""))),
            "signal_actor_id": str(
                selected_control.get(
                    "signal_actor_id",
                    selected_control.get(
                        "control_id",
                        selected_control.get("id", ""),
                    ),
                )
            ),
            "cp_provider_source": str(selected_control.get("provider_source", "")),
            "provider_source": str(selected_control.get("provider_source", "")),
            "from_cp": True,
            "traffic_control_from_cp": True,
            "confidence": float(selected_control.get("confidence", 1.0) or 0.0),
            "ego_passed_stop_line": bool(selected_control.get("ego_passed_stop_line", False)),
        }
        return context, stop_target

    def _select_relevant_traffic_control(
        self,
        *,
        traffic_controls: Sequence[Mapping[str, object]],
        ego_location: PlannerLocation,
        ego_heading_rad: float,
        current_lane_id: int,
        current_road_id: int,
        sim_time_s: float,
    ) -> Mapping[str, object] | None:
        best_control: Mapping[str, object] | None = None
        best_score: tuple[float, float, float] | None = None
        cos_h = math.cos(float(ego_heading_rad))
        sin_h = math.sin(float(ego_heading_rad))
        for control in list(traffic_controls or []):
            if not isinstance(control, Mapping):
                continue
            if not self._cp_message_is_fresh(control, sim_time_s=float(sim_time_s)):
                continue
            stop_line = control.get("stop_line_position", control.get("stop_line", None))
            if not isinstance(stop_line, Mapping):
                continue
            x_value = stop_line.get("x", stop_line.get("x_m", None))
            y_value = stop_line.get("y", stop_line.get("y_m", None))
            if x_value is None or y_value is None:
                continue
            dx_m = float(x_value) - float(ego_location.x)
            dy_m = float(y_value) - float(ego_location.y)
            forward_m = cos_h * dx_m + sin_h * dy_m
            lateral_m = -sin_h * dx_m + cos_h * dy_m
            if bool(control.get("ego_passed_stop_line", False)) or float(forward_m) < -1.0:
                continue
            lane_id = int(float(
                control.get(
                    "lane_id",
                    stop_line.get("lane_id", 0),
                )
                or 0
            ))
            road_id = int(float(
                control.get(
                    "road_id",
                    stop_line.get("road_id", 0),
                )
                or 0
            ))
            road_mismatch = 1.0 if road_id and current_road_id and road_id != current_road_id else 0.0
            lane_mismatch = 1.0 if lane_id and current_lane_id and lane_id != current_lane_id else 0.0
            score = (road_mismatch, lane_mismatch, abs(float(lateral_m)) + 0.01 * float(forward_m))
            if best_score is None or score < best_score:
                best_control = control
                best_score = score
        return best_control

    @staticmethod
    def _cp_message_is_fresh(message: Mapping[str, object], *, sim_time_s: float) -> bool:
        try:
            valid_until_s = float(message.get("valid_until_s", "nan"))
            if math.isfinite(valid_until_s):
                return float(sim_time_s) <= valid_until_s
        except Exception:
            pass
        try:
            timestamp_s = float(message.get("timestamp_s", sim_time_s))
            ttl_s = float(message.get("ttl_s", 0.0))
        except Exception:
            return True
        if float(ttl_s) <= 0.0:
            return True
        return float(sim_time_s) <= float(timestamp_s) + float(ttl_s)

    def _planning_module_global_route_summary(
        self,
        *,
        ego_location: PlannerLocation,
        ego_heading_rad: float,
        fallback_lane_id: int,
        ego_waypoint: Any = None,
    ) -> dict[str, object]:
        if getattr(self, "global_planner_backend", "") == "custom_admap_dijkstra":
            matched_waypoint = None
            try:
                raw_candidates = list(
                    self.global_planner.get_waypoint_candidates(
                        {
                            "x": float(ego_location.x),
                            "y": float(ego_location.y),
                            "z": float(getattr(ego_location, "z", 0.0)),
                        }
                    )
                    or []
                )
                previous_frame = dict(self._diagnostic_local_lane_frame or {})
                previous_corridors = {
                    int(key): list(value or [])
                    for key, value in dict(
                        previous_frame.get("corridors", {}) or {}
                    ).items()
                }
                previous_lane_id = int(
                    getattr(self._diagnostic_hd_map_matcher.previous, "ad_lane_id", 0)
                    or 0
                )
                candidates = []
                waypoint_by_lane: dict[int, object] = {}
                from cpx_planning.utility.global_planner import world_heading_rad

                for item in raw_candidates:
                    waypoint = item.get("waypoint")
                    if waypoint is None:
                        continue
                    position = dict(getattr(waypoint, "position", {}) or {})
                    ad_lane_id = int(item.get("ad_lane_id", 0) or 0)
                    waypoint_by_lane.setdefault(ad_lane_id, waypoint)
                    candidates.append(LaneProjectionCandidate(
                        ad_lane_id=ad_lane_id,
                        road_id=int(getattr(waypoint, "road_id", 0) or 0),
                        section_id=int(getattr(waypoint, "section_id", 0) or 0),
                        raw_lane_id=int(getattr(waypoint, "lane_id", 0) or 0),
                        center_x_m=float(position.get("x", ego_location.x)),
                        center_y_m=float(position.get("y", ego_location.y)),
                        heading_rad=float(world_heading_rad(waypoint) or 0.0),
                        lane_width_m=max(
                            0.1, float(getattr(waypoint, "lane_width_m", 3.5) or 3.5)
                        ),
                        snap_distance_m=float(item.get("snap_distance_m", 0.0)),
                        is_in_lane=bool(item.get("is_in_lane", False)),
                        probability=float(item.get("probability", 0.0)),
                        topology_relation=topology_relation(
                            candidate_lane_id=ad_lane_id,
                            previous_lane_id=previous_lane_id,
                            previous_corridors=previous_corridors,
                        ),
                    ))
                matched = self._diagnostic_hd_map_matcher.update(
                    ego_x_m=float(ego_location.x),
                    ego_y_m=float(ego_location.y),
                    ego_heading_rad=float(ego_heading_rad),
                    candidates=candidates,
                )
                matched_waypoint = waypoint_by_lane.get(int(matched.ad_lane_id))
                self._diagnostic_map_matching = {
                    **matched.as_dict(),
                    "candidate_count": len(candidates),
                }
            except Exception as exc:
                self._diagnostic_map_matching = {
                    "valid": False,
                    "match_reason": f"diagnostic_map_match_failed:{exc}",
                    "candidate_count": 0,
                }
            try:
                summary = self.global_planner.get_current_route_info(
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    query_key=str(self.config.get("route_query_key", "ego")),
                )
            except Exception as exc:
                return {
                    "route_found": False,
                    "optimal_lane_id": int(fallback_lane_id),
                    "current_road_option": "",
                    "next_macro_maneuver": "Continue Straight",
                    "next_macro_distance_m": float("inf"),
                    "remaining_distance_m": 0.0,
                    "debug_reason": f"admap_route_query_failed:{exc}",
                }
            ad_target_lane_id = int(
                getattr(summary, "optimal_lane_id", 0) or 0
            )
            local_target_lane_id = int(fallback_lane_id)
            ad_current_lane_id = 0
            local_direction = ""
            local_offset = 0
            target_in_local_frame = False
            try:
                local_graph = self.global_planner.get_local_lane_graph(
                    float(ego_location.x),
                    float(ego_location.y),
                    z_m=float(getattr(ego_location, "z", 0.0)),
                    forward_distance_m=100.0,
                    backward_distance_m=100.0,
                    ego_waypoint=matched_waypoint,
                )
                self._diagnostic_local_lane_frame = dict(local_graph)
                ad_current_lane_id = int(local_graph.get("ego_ad_lane_id", 0) or 0)
                lane_to_offset = dict(local_graph.get("lane_to_offset", {}) or {})
                target_in_local_frame = int(ad_target_lane_id) in {
                    int(lane_id) for lane_id in lane_to_offset
                }
                offset = int(lane_to_offset.get(int(ad_target_lane_id), 0))
                local_offset = int(offset)
                local_direction = "left" if offset > 0 else "right" if offset < 0 else ""
                if local_direction and ego_waypoint is not None:
                    from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint

                    accessor_name = "left" if local_direction == "left" else "right"
                    candidates = [ego_waypoint]
                    next_fn = getattr(ego_waypoint, "next", None)
                    if callable(next_fn):
                        for distance_m in range(5, 101, 5):
                            candidates.extend(list(next_fn(float(distance_m)) or []))
                    for candidate in candidates:
                        accessor = getattr(candidate, accessor_name, None)
                        adjacent = accessor() if callable(accessor) else None
                        if adjacent is None:
                            continue
                        resolved = int(canonical_lane_id_for_waypoint(adjacent) or 0)
                        if resolved != 0 and resolved != int(fallback_lane_id):
                            local_target_lane_id = int(resolved)
                            break
            except Exception:
                # Route information remains usable even if this tick's
                # topology-to-local-lane projection cannot be resolved.
                local_target_lane_id = int(fallback_lane_id)
            violations = local_lane_frame_invariants(
                matched_lane_id=int(
                    self._diagnostic_map_matching.get("ad_lane_id", 0) or 0
                ),
                corridors={
                    int(key): list(value or [])
                    for key, value in dict(
                        self._diagnostic_local_lane_frame.get("corridors", {}) or {}
                    ).items()
                },
                target_lane_id=int(ad_target_lane_id),
                reported_target_offset=int(local_offset),
            )
            self._diagnostic_local_lane_frame["invariant_violations"] = list(
                violations
            )
            self._diagnostic_local_lane_frame["route_target_offset"] = int(
                local_offset
            )
            self._diagnostic_local_lane_frame["route_target_ad_lane_id"] = int(
                ad_target_lane_id
            )
            self._diagnostic_local_lane_frame["route_target_in_frame"] = bool(
                target_in_local_frame
            )
            next_macro_maneuver = str(
                getattr(summary, "next_macro_maneuver", "Continue Straight")
            )
            normalized_macro = (
                next_macro_maneuver.strip().lower().replace("-", "_").replace(" ", "_")
            )
            if (
                normalized_macro in {"lane_change_left", "lane_change_right"}
                and int(ad_current_lane_id) != 0
                and int(ad_current_lane_id) == int(ad_target_lane_id)
                and int(local_offset) == 0
            ):
                # The route backend can keep reporting the consumed edge for
                # a few progress samples. Expose completion immediately so
                # behavior and ManeuverManager do not restart/retain it.
                next_macro_maneuver = "Lane Follow"
            route_result = {
                "route_found": bool(getattr(summary, "route_found", False)),
                "optimal_lane_id": int(local_target_lane_id),
                "ad_current_lane_id": int(ad_current_lane_id),
                "ad_target_lane_id": int(ad_target_lane_id),
                "lane_change_direction": str(local_direction),
                "lane_change_offset": int(local_offset),
                "target_in_local_frame": bool(target_in_local_frame),
                "diagnostic_map_matching": dict(self._diagnostic_map_matching),
                "diagnostic_local_lane_frame": dict(
                    self._diagnostic_local_lane_frame
                ),
                "current_road_option": str(getattr(summary, "current_road_option", "")),
                "next_macro_maneuver": str(next_macro_maneuver),
                "next_macro_distance_m": float(
                    getattr(summary, "next_macro_distance_m", float("inf"))
                ),
                "remaining_distance_m": float(
                    getattr(summary, "distance_to_destination_m", 0.0) or 0.0
                ),
                "debug_reason": "admap_topology_carla_geometry_active",
            }
            # AD-map owns route identity, maneuver semantics, progress, and
            # destination completion in this backend.  Publish the exact
            # per-tick query consumed by behavior so RouteManagerStatus/CSV
            # cannot remain frozen at the last CARLA route rebuild.  CARLA's
            # independently synchronized index is geometry-only.
            route_manager = getattr(self, "route_manager", None)
            publish = getattr(
                route_manager,
                "accept_authoritative_route_summary",
                None,
            )
            if callable(publish):
                publish(
                    summary,
                    debug_reason="admap_authoritative_route_active",
                )
            return route_result
        if not hasattr(self, "route_manager"):
            try:
                summary = self.global_planner.get_current_route_info(
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    query_key="ros_planner",
                )
            except Exception as exc:
                return {
                    "route_found": False,
                    "optimal_lane_id": int(fallback_lane_id),
                    "current_road_option": "",
                    "next_macro_maneuver": "Continue Straight",
                    "debug_reason": f"planning_module_global_route_missing:{exc}",
                }
            lane_id = int(getattr(summary, "optimal_lane_id", fallback_lane_id) or fallback_lane_id)
            if int(lane_id) == 0:
                lane_id = int(fallback_lane_id)
            return {
                "route_found": bool(getattr(summary, "route_found", False)),
                "optimal_lane_id": int(lane_id),
                "current_road_option": str(getattr(summary, "current_road_option", "")),
                "next_macro_maneuver": str(
                    getattr(summary, "next_macro_maneuver", "Continue Straight")
                ),
                "debug_reason": str(
                    getattr(summary, "debug_reason", "planning_module_global_route")
                ),
            }
        return self.route_manager.get_route_info(
            x_m=float(ego_location.x),
            y_m=float(ego_location.y),
            query_key="ros_planner",
            fallback_lane_id=int(fallback_lane_id),
            ego_waypoint=ego_waypoint,
        )

    def _route_optimal_lane_id(self, *, ego_location: PlannerLocation, fallback_lane_id: int) -> int:
        return int(
            self._planning_module_global_route_summary(
                ego_location=ego_location,
                ego_heading_rad=0.0,
                fallback_lane_id=int(fallback_lane_id),
            ).get("optimal_lane_id", fallback_lane_id)
        )

    @staticmethod
    def _nearest_route_index_ahead(
        *,
        route_entries: Sequence[Any],
        ego_location: PlannerLocation,
        ego_heading_rad: float,
    ) -> Optional[int]:
        cos_h = math.cos(float(ego_heading_rad))
        sin_h = math.sin(float(ego_heading_rad))
        best_index = None
        best_score = None
        for index, (waypoint, _) in enumerate(list(route_entries or [])):
            transform = getattr(waypoint, "transform", None)
            location = getattr(transform, "location", None)
            if location is None:
                continue
            dx_m = float(location.x) - float(ego_location.x)
            dy_m = float(location.y) - float(ego_location.y)
            forward_m = cos_h * dx_m + sin_h * dy_m
            lateral_m = -sin_h * dx_m + cos_h * dy_m
            distance_m = math.hypot(dx_m, dy_m)
            behind_penalty = 20.0 if float(forward_m) < -2.0 else 0.0
            score = (
                float(behind_penalty),
                abs(float(lateral_m)) + 0.15 * max(0.0, -float(forward_m)),
                float(distance_m),
                int(index),
            )
            if best_score is None or score < best_score:
                best_index = int(index)
                best_score = score
        return best_index

    @staticmethod
    def _road_option_name(option: object) -> str:
        if option is None:
            return ""
        name = getattr(option, "name", None)
        if name is not None:
            return str(name).strip().upper()
        text = str(option).strip()
        if "." in text:
            text = text.rsplit(".", 1)[-1]
        return text.strip().upper()

    @staticmethod
    def _next_macro_maneuver_from_road_options(options: Sequence[str]) -> str:
        normalized = [str(option).strip().upper() for option in list(options or [])]
        for option in normalized:
            if option == "CHANGELANELEFT":
                return "Lane Change Left"
            if option == "CHANGELANERIGHT":
                return "Lane Change Right"
            if option == "LEFT":
                return "Left Turn"
            if option == "RIGHT":
                return "Right Turn"
            if option == "STRAIGHT":
                return "Continue Straight"
        return "Continue Straight"

    def _legacy_route_optimal_lane_id(self, *, ego_location: PlannerLocation, fallback_lane_id: int) -> int:
        return self._route_optimal_lane_id(ego_location=ego_location, fallback_lane_id=fallback_lane_id)

    def _apply_mpc_cost_profile(
        self,
        *,
        behavior: str,
        planner_lc_state: str,
        planner_mode: str,
        next_macro_maneuver: str,
        sim_time_s: float,
        nearest_obstacle_distance_m: Optional[float] = None,
        ego_speed_mps: float = 0.0,
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
        if bool(
            getattr(self.mpc, "adaptive_horizon_enabled", False)
        ) and hasattr(self.mpc, "blend_toward_horizon_s"):
            # adaptive_horizon_enabled/min_s/max_s live on the MPC instance
            # (parsed from MPC/mpc.yaml, the same file horizon_s itself comes
            # from) -- self.config here is the bridge/scenario config, a
            # separate namespace that was never going to have that key.
            profile_horizon_s = (
                dict(self.config.get("adaptive_horizon_profile_s", {}))
                or _DEFAULT_ADAPTIVE_HORIZON_PROFILE_S
            )
            self.mpc.blend_toward_horizon_s(
                _adaptive_target_horizon_s(
                    mpc_cost_profile=str(self.active_mpc_cost_profile),
                    nearest_obstacle_distance_m=nearest_obstacle_distance_m,
                    ego_speed_mps=float(ego_speed_mps),
                    profile_horizon_s=profile_horizon_s,
                    obstacle_reference_speed_mps=float(
                        self.config.get(
                            "adaptive_horizon_obstacle_reference_speed_mps", 2.0
                        )
                    ),
                    obstacle_comfortable_decel_mps2=float(
                        self.config.get(
                            "adaptive_horizon_obstacle_comfortable_decel_mps2", 2.0
                        )
                    ),
                )
            )

    def _load_mpc_config(self) -> tuple[dict[str, Any], dict[str, Any]]:
        cfg_path = self.config.get("mpc_config_path")
        if not cfg_path:
            cfg_path = Path(__file__).resolve().parents[1] / "MPC" / "mpc.yaml"
        with open(cfg_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        mpc_cfg = dict(payload.get("mpc", payload))
        behavior_cfg_path = Path(__file__).resolve().parents[1] / "behavior_planner" / "behavior_planner.yaml"
        with open(behavior_cfg_path, "r", encoding="utf-8") as f:
            behavior_payload = yaml.safe_load(f) or {}
        mpc_cfg["behavior_planner_runtime"] = dict(behavior_payload.get("behavior_planner_runtime", behavior_payload))
        road_cfg = dict(payload.get("road", {}))
        road_cfg.setdefault("lane_count", int(self.config.get("lane_count", 3)))
        road_cfg.setdefault("lane_width_m", float(self.config.get("lane_width_m", 3.5)))
        return mpc_cfg, road_cfg

    def _collect_object_snapshots(self, detected_objects: Any = None) -> list[dict[str, Any]]:
        objects = detected_objects or {}
        if not isinstance(objects, Mapping):
            objects = getattr(objects, "objects", {}) or {}
        vehicles = list(objects.get("vehicles", []) or [])
        snapshots: list[dict[str, Any]] = []
        for index, obj in enumerate(vehicles):
            if isinstance(obj, Mapping):
                normalized = self._normalize_local_object_snapshot(obj)
                if normalized is not None:
                    snapshots.append(dict(normalized))
                continue
            actor = getattr(obj, "carla_actor", None) or getattr(obj, "vehicle", None) or obj
            if actor is None:
                continue
            try:
                get_transform = getattr(actor, "get_transform", None)
                transform = get_transform() if callable(get_transform) else None
                location = getattr(transform, "location", None)
                if location is None:
                    get_location = getattr(actor, "get_location", None)
                    location = (
                        get_location()
                        if callable(get_location)
                        else getattr(actor, "location", None)
                    )
                if location is None:
                    continue

                get_velocity = getattr(actor, "get_velocity", None)
                velocity = (
                    get_velocity()
                    if callable(get_velocity)
                    else getattr(actor, "velocity", None)
                )
                velocity_x = float(getattr(velocity, "x", 0.0))
                velocity_y = float(getattr(velocity, "y", 0.0))
                velocity_z = float(getattr(velocity, "z", 0.0))
                bbox = getattr(actor, "bounding_box", None)
                extent = getattr(bbox, "extent", None)
                speed_mps = math.sqrt(
                    velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2
                )
                rotation = getattr(transform, "rotation", None)
                if rotation is not None:
                    heading_rad = math.radians(float(getattr(rotation, "yaw", 0.0)))
                elif speed_mps > 0.05:
                    heading_rad = math.atan2(velocity_y, velocity_x)
                else:
                    heading_rad = 0.0

                raw_actor_id = getattr(
                    actor, "id", getattr(actor, "carla_id", None))
                try:
                    has_stable_actor_id = int(raw_actor_id) >= 0
                except (TypeError, ValueError):
                    has_stable_actor_id = bool(str(raw_actor_id or "").strip())
                actor_id = (
                    str(raw_actor_id)
                    if has_stable_actor_id
                    else "opencda_detection:%d" % int(index)
                )
                length_m = 2.0 * float(getattr(extent, "x", 2.2))
                width_m = 2.0 * float(getattr(extent, "y", 0.9))
                if not math.isfinite(length_m) or length_m <= 0.1:
                    length_m = 4.5
                if not math.isfinite(width_m) or width_m <= 0.1:
                    width_m = 2.0
                snapshots.append({
                    "vehicle_id": actor_id,
                    "id": actor_id,
                    "x": float(location.x),
                    "y": float(location.y),
                    "v": float(speed_mps),
                    "psi": float(heading_rad),
                    "length_m": float(length_m),
                    "width_m": float(width_m),
                    "source": (
                        "opencda_perception"
                        if transform is not None
                        else "opencda_ml_lidar_fusion"
                    ),
                    "provider_source": "native_opencda_perception",
                    "confidence": float(getattr(actor, "confidence", 1.0)),
                })
            except (AttributeError, RuntimeError, TypeError, ValueError):
                continue
        return snapshots

    def _perception_diagnostics(self) -> dict[str, object]:
        return {
            "perception_mode": "ros_topic",
            "perception_ml_active": False,
            "perception_camera_count": 0,
        }

    def _fused_planning_object_snapshots(
        self,
        *,
        local_object_snapshots: Sequence[Mapping[str, Any]],
        cp_obstacles: Sequence[Mapping[str, Any]],
        ego_location: PlannerLocation,
        sim_time_s: float,
    ) -> list[dict[str, Any]]:
        fused_by_key: dict[str, dict[str, Any]] = {}
        priorities_by_key: dict[str, int] = {}

        for snapshot in list(local_object_snapshots or []):
            normalized = self._normalize_local_object_snapshot(snapshot)
            if normalized is not None:
                self._upsert_fused_obstacle(
                    fused_by_key=fused_by_key,
                    priorities_by_key=priorities_by_key,
                    snapshot=normalized,
                    priority=self._obstacle_source_priority(normalized),
                )

        for obstacle in list(cp_obstacles or []):
            if not isinstance(obstacle, Mapping):
                continue
            if not self._cp_message_is_fresh(obstacle, sim_time_s=float(sim_time_s)):
                continue
            normalized = self._normalize_cp_obstacle_snapshot(obstacle)
            if normalized is not None:
                if self._is_duplicate_native_perception_cp_obstacle(
                    cp_snapshot=normalized,
                    fused_snapshots=fused_by_key.values(),
                ):
                    continue
                self._upsert_fused_obstacle(
                    fused_by_key=fused_by_key,
                    priorities_by_key=priorities_by_key,
                    snapshot=normalized,
                    priority=self._obstacle_source_priority(normalized),
                )

        return list(fused_by_key.values())

    @staticmethod
    def _is_duplicate_native_perception_cp_obstacle(
        *,
        cp_snapshot: Mapping[str, Any],
        fused_snapshots: Sequence[Mapping[str, Any]],
        max_position_delta_m: float = 1.0,
    ) -> bool:
        provider_source = str(cp_snapshot.get("provider_source", "")).strip().lower()
        source = str(cp_snapshot.get("source", "")).strip().lower()
        if "perception" not in provider_source and "perception" not in source:
            return False
        try:
            cp_x = float(cp_snapshot.get("x", 0.0))
            cp_y = float(cp_snapshot.get("y", 0.0))
        except Exception:
            return False
        for existing in list(fused_snapshots or []):
            existing_provider = str(existing.get("provider_source", "")).strip().lower()
            existing_source = str(existing.get("source", "")).strip().lower()
            if "perception" not in existing_provider and "perception" not in existing_source:
                continue
            try:
                dx = cp_x - float(existing.get("x", 0.0))
                dy = cp_y - float(existing.get("y", 0.0))
            except Exception:
                continue
            if math.hypot(dx, dy) <= float(max_position_delta_m):
                return True
        return False

    def _limit_obstacles_for_mpc(
        self,
        *,
        object_snapshots: Sequence[Mapping[str, Any]],
        ego_location: PlannerLocation,
    ) -> list[dict[str, Any]]:
        fused = [dict(item) for item in list(object_snapshots or []) if isinstance(item, Mapping)]
        if self.max_mpc_obstacles > 0 and len(fused) > self.max_mpc_obstacles:
            fused.sort(
                key=lambda item: (
                    float(item.get("x", 0.0)) - float(ego_location.x)
                ) ** 2
                + (
                    float(item.get("y", 0.0)) - float(ego_location.y)
                ) ** 2
            )
            fused = fused[: self.max_mpc_obstacles]
        return fused

    def _mpc_object_snapshots_with_prediction(
        self,
        object_snapshots: Sequence[Mapping[str, Any]],
        *,
        prediction_trajectories: Mapping[str, Sequence[Mapping[str, object]]],
    ) -> list[dict[str, Any]]:
        """Attach this tick's predicted trajectory to each MPC obstacle.

        Without this, ``MPC._get_object_state_at_stage`` (MPC/mpc.py) never
        sees ``planner_input_frame.prediction.obstacle_future_trajectories``
        at all -- it only recognizes a ``predicted_trajectory`` already
        shaped as one ``[x, y, v, psi]`` entry per stage, so it silently
        falls back to its own constant-velocity extrapolation for every
        obstacle, independent of (and less accurate than) the
        constant-acceleration/CP-supplied prediction the rest of the
        pipeline already computed. ``bridge.tracker.predict()`` is already
        called with ``horizon_s=self.mpc.horizon_s, dt_s=self.mpc.dt_s`` (see
        planner_input_adapter.py), so each trajectory here already has one
        point per MPC stage -- this only needs to convert the shape and
        attach it, not resample it.
        """

        from cpx_planning.pipeline.prediction import (
            mpc_stage_trajectory,
            obstacle_track_id,
        )

        if not prediction_trajectories:
            return [dict(snapshot) for snapshot in list(object_snapshots or [])]
        horizon_steps = int(self.mpc.horizon_steps)
        dt_s = float(self.mpc.dt_s)
        annotated: list[dict[str, Any]] = []
        for snapshot in list(object_snapshots or []):
            if not isinstance(snapshot, Mapping):
                continue
            updated = dict(snapshot)
            points = prediction_trajectories.get(obstacle_track_id(snapshot))
            if points:
                updated["predicted_trajectory"] = mpc_stage_trajectory(
                    list(points),
                    fallback_heading_rad=float(snapshot.get("psi", snapshot.get("heading_rad", 0.0))),
                    horizon_steps=horizon_steps,
                    dt_s=dt_s,
                )
            annotated.append(updated)
        return annotated

    @staticmethod
    def _normalize_local_object_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
        try:
            obstacle_id = str(snapshot.get("vehicle_id", snapshot.get("id", ""))).strip()
            if not obstacle_id:
                return None
            return {
                "vehicle_id": obstacle_id,
                "id": obstacle_id,
                "x": float(snapshot.get("x", 0.0)),
                "y": float(snapshot.get("y", 0.0)),
                "v": float(snapshot.get("v", 0.0)),
                "psi": float(snapshot.get("psi", 0.0)),
                "length_m": float(snapshot.get("length_m", 4.5)),
                "width_m": float(snapshot.get("width_m", 2.0)),
                "source": str(snapshot.get("source", "opencda_perception")),
                "provider_source": str(snapshot.get("provider_source", "native_opencda_perception")),
                "confidence": float(snapshot.get("confidence", 1.0)),
            }
        except Exception:
            return None

    @staticmethod
    def _normalize_cp_obstacle_snapshot(obstacle: Mapping[str, Any]) -> dict[str, Any] | None:
        try:
            raw_id = str(obstacle.get("id", obstacle.get("vehicle_id", ""))).strip()
            if not raw_id:
                return None
            state = obstacle.get("state", [])
            if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray)):
                state_values = list(state)
            else:
                state_values = []
            x_m = obstacle.get("x", obstacle.get("x_m", state_values[0] if len(state_values) >= 1 else None))
            y_m = obstacle.get("y", obstacle.get("y_m", state_values[1] if len(state_values) >= 2 else None))
            speed_mps = obstacle.get("v", obstacle.get("speed_mps", state_values[2] if len(state_values) >= 3 else 0.0))
            heading_rad = obstacle.get("psi", obstacle.get("heading_rad", state_values[3] if len(state_values) >= 4 else 0.0))
            if x_m is None or y_m is None:
                return None
            shape = obstacle.get("shape", {})
            shape = dict(shape) if isinstance(shape, Mapping) else {}
            obstacle_id = raw_id.rsplit(":", 1)[-1] if ":" in raw_id else raw_id
            provider_source = str(obstacle.get("provider_source", "opencda_cp"))
            source = str(obstacle.get("source", "opencda_cp"))
            return {
                "vehicle_id": obstacle_id,
                "id": obstacle_id,
                "cp_message_id": raw_id,
                "x": float(x_m),
                "y": float(y_m),
                "v": float(speed_mps),
                "psi": float(heading_rad),
                "length_m": float(shape.get("length_m", obstacle.get("length_m", 4.5))),
                "width_m": float(shape.get("width_m", obstacle.get("width_m", 2.0))),
                "source": source,
                "provider_source": provider_source,
                "confidence": float(obstacle.get("confidence", 0.5)),
                "lane_id": int(float(obstacle.get("lane_id", 0) or 0)),
                "road_id": int(float(obstacle.get("road_id", 0) or 0)),
                "object_type": str(obstacle.get("type", "unknown")),
                "observed_by_cav_ids": list(
                    obstacle.get("observed_by_cav_ids", []) or []
                ),
                "not_observed_by_cav_ids": list(
                    obstacle.get("not_observed_by_cav_ids", []) or []
                ),
                "blind_spot_shared": bool(
                    obstacle.get("blind_spot_shared", False)
                ),
            }
        except Exception:
            return None

    @staticmethod
    def _cooperative_actor_evidence(
        *,
        cp_summary: Mapping[str, Any],
        prediction_trajectories: Mapping[
            str, Sequence[Mapping[str, Any]]
        ],
        selected_reference: Sequence[Mapping[str, Any]],
        candidate_proximity_m: float = 3.0,
    ) -> dict[str, Any]:
        """Build an auditable CP-to-prediction-to-candidate evidence chain.

        ``candidate_relevant`` means that a predicted actor position enters
        the selected reference's spatial safety envelope at a corresponding
        horizon step. It deliberately does not claim that the actor changed
        the selected decision; proving that stronger counterfactual requires
        evaluating the same candidate set with that actor removed.
        """

        provenance = [
            dict(item)
            for item in list(cp_summary.get("actor_provenance", []) or [])
            if isinstance(item, Mapping)
        ]
        prediction_by_actor = {
            str(key).rsplit(":", 1)[-1]: list(points or [])
            for key, points in dict(prediction_trajectories or {}).items()
        }
        reference = list(selected_reference or [])
        prediction_used_ids: list[str] = []
        prediction_used_pedestrian_ids: list[str] = []
        candidate_relevant_ids: list[str] = []
        candidate_relevant_pedestrian_ids: list[str] = []
        evidence: list[dict[str, Any]] = []

        for actor in provenance:
            message_id = str(actor.get("actor_id", ""))
            actor_id = message_id.rsplit(":", 1)[-1]
            actor_type = str(actor.get("actor_type", "unknown"))
            predicted_points = prediction_by_actor.get(actor_id, [])
            used_by_prediction = bool(predicted_points)
            min_distance_m: float | None = None
            if used_by_prediction and reference:
                for index in range(min(len(reference), len(predicted_points))):
                    try:
                        ref = reference[index]
                        point = predicted_points[index]
                        rx = float(ref.get("x_ref_m", ref.get("x", 0.0)))
                        ry = float(ref.get("y_ref_m", ref.get("y", 0.0)))
                        px = float(point.get("x", point.get("x_m", 0.0)))
                        py = float(point.get("y", point.get("y_m", 0.0)))
                    except (TypeError, ValueError):
                        continue
                    distance_m = math.hypot(rx - px, ry - py)
                    min_distance_m = (
                        distance_m
                        if min_distance_m is None
                        else min(min_distance_m, distance_m)
                    )
            candidate_relevant = bool(
                min_distance_m is not None
                and min_distance_m <= float(candidate_proximity_m)
            )
            if used_by_prediction:
                prediction_used_ids.append(message_id)
                if actor_type == "pedestrian":
                    prediction_used_pedestrian_ids.append(message_id)
            if candidate_relevant:
                candidate_relevant_ids.append(message_id)
                if actor_type == "pedestrian":
                    candidate_relevant_pedestrian_ids.append(message_id)
            evidence.append({
                **actor,
                "used_by_prediction": bool(used_by_prediction),
                "candidate_relevant": bool(candidate_relevant),
                "candidate_min_predicted_distance_m": (
                    None if min_distance_m is None else float(min_distance_m)
                ),
            })

        pedestrian_count = sum(
            str(item.get("actor_type", "")) == "pedestrian"
            for item in provenance
        )
        blind_pedestrian_count = sum(
            str(item.get("actor_type", "")) == "pedestrian"
            and bool(item.get("blind_spot_shared", False))
            for item in provenance
        )
        return {
            "cp_actor_provenance": json.dumps(provenance, default=str),
            "cp_pedestrian_count": int(pedestrian_count),
            "cp_blind_spot_pedestrian_count": int(blind_pedestrian_count),
            "cp_prediction_used_actor_ids": ",".join(prediction_used_ids),
            "cp_prediction_used_pedestrian_ids": ",".join(
                prediction_used_pedestrian_ids
            ),
            "cp_candidate_relevant_actor_ids": ",".join(
                candidate_relevant_ids
            ),
            "cp_candidate_relevant_pedestrian_ids": ",".join(
                candidate_relevant_pedestrian_ids
            ),
            "cp_actor_evidence": json.dumps(evidence, default=str),
        }

    @staticmethod
    def _obstacle_source_priority(snapshot: Mapping[str, Any]) -> int:
        provider_source = str(snapshot.get("provider_source", "")).lower()
        source = str(snapshot.get("source", "")).lower()
        if "perception" in provider_source or "perception" in source:
            return 100
        if "v2x" in provider_source or "v2x" in source:
            return 80
        if "fallback" in provider_source or "fallback" in source or "carla" in source:
            return 40
        return 60

    @staticmethod
    def _fused_obstacle_key(snapshot: Mapping[str, Any]) -> str:
        obstacle_id = str(snapshot.get("vehicle_id", snapshot.get("id", ""))).strip()
        return obstacle_id.rsplit(":", 1)[-1] if ":" in obstacle_id else obstacle_id

    @classmethod
    def _upsert_fused_obstacle(
        cls,
        *,
        fused_by_key: dict[str, dict[str, Any]],
        priorities_by_key: dict[str, int],
        snapshot: Mapping[str, Any],
        priority: int,
    ) -> None:
        key = cls._fused_obstacle_key(snapshot)
        if not key:
            return
        previous_priority = int(priorities_by_key.get(key, -1))
        previous = fused_by_key.get(key)
        previous_confidence = float(previous.get("confidence", 0.0)) if isinstance(previous, Mapping) else -1.0
        confidence = float(snapshot.get("confidence", 0.0))
        if int(priority) > previous_priority or (
            int(priority) == previous_priority and float(confidence) >= previous_confidence
        ):
            fused_by_key[key] = dict(snapshot)
            priorities_by_key[key] = int(priority)

    def _draw_world_debug_primitives(
        self,
        *,
        destination_state: Sequence[float],
        lane_center_reference: Sequence[Mapping[str, Any]],
    ) -> None:
        """Leave simulator drawing to the OpenCDA boundary process."""

        del destination_state, lane_center_reference

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

    def _draw_debug_polyline(
        self,
        *,
        debug: Any,
        points_xy: Sequence[Sequence[float]],
        z_m: float,
        color: Any,
        thickness: float,
        life_time_s: float,
        max_segments: int,
    ) -> None:
        points = [
            (float(point[0]), float(point[1]))
            for point in list(points_xy or [])
            if len(point) >= 2
        ]
        if len(points) < 2:
            return
        stride = max(1, int(len(points) / max(1, int(max_segments))))
        sampled = points[::stride]
        if sampled[-1] != points[-1]:
            sampled.append(points[-1])
        for first, second in zip(sampled[:-1], sampled[1:]):
            if math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1])) < 1.0e-3:
                continue
            debug.draw_line(
                PlannerLocation(x=float(first[0]), y=float(first[1]), z=float(z_m)),
                PlannerLocation(x=float(second[0]), y=float(second[1]), z=float(z_m)),
                thickness=float(thickness),
                color=color,
                life_time=float(life_time_s),
                persistent_lines=False,
            )

    def _active_global_route_points(self) -> list[list[float]]:
        """Return the geometry used by both local references and visualization."""

        latest_update = dict(getattr(self, "_latest_opencda_update", {}) or {})
        ego_transform = latest_update.get("ego_transform")
        if ego_transform is not None:
            loc = ego_transform.location
            return self.route_manager.geometry_route_points(
                x_m=float(loc.x),
                y_m=float(loc.y),
                query_key="ros_planner_polyline",
            )
        route_points = self.route_manager.geometry_route_points()
        if route_points:
            return route_points

        summary = None
        try:
            summary = self._active_route_summary
        except Exception:
            summary = None
        route_waypoints = list(getattr(summary, "route_waypoints", []) or [])
        points: list[list[float]] = []
        for index, raw_point in enumerate(route_waypoints):
            try:
                x_m = float(raw_point[0])
                y_m = float(raw_point[1])
                z_m = float(raw_point[2]) if len(raw_point) >= 3 else 0.0
            except Exception:
                continue
            if index < len(route_waypoints) - 1:
                try:
                    nx_m = float(route_waypoints[index + 1][0])
                    ny_m = float(route_waypoints[index + 1][1])
                    heading_rad = math.atan2(ny_m - y_m, nx_m - x_m)
                except Exception:
                    heading_rad = points[-1][3] if points else 0.0
            else:
                heading_rad = points[-1][3] if points else 0.0
            if points and math.hypot(
                float(points[-1][0]) - float(x_m),
                float(points[-1][1]) - float(y_m),
            ) < 1.0e-6:
                continue
            points.append([
                float(x_m),
                float(y_m),
                float(z_m),
                float(heading_rad),
            ])
        return points

    def _map_waypoint_from_location(self, location: PlannerLocation):
        if self.map_planner is None:
            return None
        point = {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        }
        get_waypoint = getattr(self.map_planner, "get_waypoint", None)
        if not callable(get_waypoint):
            return None
        try:
            return get_waypoint(point)
        except Exception:
            pass
        try:
            return get_waypoint(
                PlannerLocation(
                    x=float(location.x),
                    y=float(location.y),
                    z=float(location.z),
                )
            )
        except Exception:
            return None

    def _drivable_waypoint_from_location(
        self,
        location: PlannerLocation,
    ):
        """Query drivable geometry through the custom AD-map planner."""

        return self._map_waypoint_from_location(location)

    def _lane_id_at_location(self, location: PlannerLocation) -> int:
        waypoint = self._map_waypoint_from_location(location)
        if waypoint is None:
            return 1
        try:
            from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint

            lane_id = int(canonical_lane_id_for_waypoint(waypoint) or 0)
            return lane_id if lane_id != 0 else 1
        except Exception:
            return int(getattr(waypoint, "lane_id", 1) or 1)

    @staticmethod
    def _location_to_point(location: Any) -> dict[str, float]:
        return {
            "x": float(getattr(location, "x", 0.0)),
            "y": float(getattr(location, "y", 0.0)),
            "z": float(getattr(location, "z", 0.0)),
        }

    def _front_gap_m(
        self,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        *,
        lane_change_direction: str = "",
        lane_change_progress: float = 0.0,
        current_lane_id: Optional[int] = None,
        lane_assignments: Optional[Mapping[str, int]] = None,
        return_actor_id: bool = False,
    ):
        """Nearest-ahead gap in ego's body frame.

        Outside an active lane change (``lane_change_direction == ""``),
        this is a plain nearest-ahead search within a +/-2.5 m lateral
        gate -- unchanged from before.

        During an active lane change ("left"/"right"), the source lane's
        front vehicle must not be dropped the instant the maneuver starts
        (ego hasn't moved yet -- it's still physically in the source lane),
        but also must not keep braking ego once ego's body has actually
        cleared it. This computes the source-lane gap and target-lane gap
        *separately* (split at ego's current heading, not by lane_id) and
        blends between them as a smooth function of ``lane_change_progress``
        (alpha in [0, 1], 0 = still at the source lane center, 1 = at the
        target lane center -- pass self._route_tracking_lane_change_progress):

          - alpha <= alpha_clear: fully the source-lane gap. alpha_clear is
            the progress at which ego's own body -- not just its center --
            has crossed the source/target lane boundary, derived from
            vehicle width and lane width, not a fixed distance or a
            lane_id switch: alpha_clear = 0.5 + vehicle_width_m / (2 *
            lane_width_m).
          - alpha_clear < alpha < 1: smoothstep blend toward the
            target-lane gap.
          - alpha >= 1: fully the target-lane gap.

        Sign convention (lateral = -dx*sin_h + dy*cos_h): validated against
        the cpx_lane_change_speed_* scenarios -- positive lateral is the
        left-hand side of ego's current heading.

        ``return_actor_id=True`` also returns the id of whichever object
        dominates the blended gap (None if neither side has one), so a
        caller can detect "the object being used as my front-vehicle
        reference just changed" even when the gap distance itself moves
        smoothly -- see control_context_key in _run_full_cpx_pipeline_step.
        """

        cos_h = math.cos(ego_yaw_rad)
        sin_h = math.sin(ego_yaw_rad)
        ego_half_length_m = max(0.0, float(self.config.get("reference_vehicle_half_length_m", 2.25)))

        def _nearest_gap(
            *, min_lateral_m: float, max_lateral_m: float
        ) -> tuple[Optional[float], Optional[str]]:
            best_gap = None
            best_actor_id = None
            for snapshot in object_snapshots:
                if current_lane_id is not None and lane_assignments is not None:
                    obstacle_id = self._object_track_id(snapshot)
                    assigned_lane_id = int(
                        lane_assignments.get(str(obstacle_id), 0) or 0
                    )
                    if assigned_lane_id != int(current_lane_id):
                        continue
                dx = float(snapshot.get("x", 0.0)) - float(ego_location.x)
                dy = float(snapshot.get("y", 0.0)) - float(ego_location.y)
                longitudinal = dx * cos_h + dy * sin_h
                lateral = -dx * sin_h + dy * cos_h
                if (
                    longitudinal <= 0.0
                    or lateral < float(min_lateral_m)
                    or lateral > float(max_lateral_m)
                ):
                    continue
                object_half_length_m = max(
                    0.0,
                    0.5 * float(snapshot.get("length_m", 4.5) or 4.5),
                )
                clearance_m = max(
                    0.0,
                    float(longitudinal)
                    - float(ego_half_length_m)
                    - float(object_half_length_m),
                )
                if best_gap is None or float(clearance_m) < float(best_gap):
                    best_gap = float(clearance_m)
                    best_actor_id = self._object_track_id(snapshot)
            return best_gap, best_actor_id

        # When stable map assignments are available, "own lane" is the
        # currently map-matched ego lane. Adjacent-lane actors never enter
        # longitudinal following, including during a lane change; once ego's
        # map match moves to the target lane, that lane naturally becomes its
        # own lane on the next tick.
        strict_current_lane = bool(
            current_lane_id is not None and lane_assignments is not None
        )
        direction = (
            ""
            if strict_current_lane
            else str(lane_change_direction or "").strip().lower()
        )
        if direction not in {"left", "right"}:
            best_gap, best_actor_id = _nearest_gap(
                min_lateral_m=-2.5, max_lateral_m=2.5
            )
            if bool(return_actor_id):
                return best_gap, (
                    None if best_gap is None else str(best_actor_id)
                )
            return best_gap

        # A small overlap around the ego-heading split line keeps an object
        # sitting right at the boundary visible to both searches, instead
        # of a strict 0.0 cutoff creating a blind seam between them.
        boundary_overlap_m = max(
            0.0, float(self.config.get("lane_change_boundary_overlap_m", 0.75))
        )
        if direction == "left":
            source_gap, source_actor_id = _nearest_gap(
                min_lateral_m=-2.5, max_lateral_m=boundary_overlap_m
            )
            target_gap, target_actor_id = _nearest_gap(
                min_lateral_m=-boundary_overlap_m, max_lateral_m=2.5
            )
        else:
            source_gap, source_actor_id = _nearest_gap(
                min_lateral_m=-boundary_overlap_m, max_lateral_m=2.5
            )
            target_gap, target_actor_id = _nearest_gap(
                min_lateral_m=-2.5, max_lateral_m=boundary_overlap_m
            )

        vehicle_width_m = max(0.5, 2.0 * float(self.config.get("reference_vehicle_half_width_m", 1.0)))
        lane_width_m = max(1.0, float(getattr(self.mpc, "lane_width_m", 3.5)))
        alpha_clear = min(
            0.95, 0.5 + float(vehicle_width_m) / (2.0 * float(lane_width_m))
        )
        alpha = max(0.0, min(1.0, float(lane_change_progress)))
        if alpha <= alpha_clear:
            blend_weight = 0.0
        else:
            span = max(1.0e-6, 1.0 - float(alpha_clear))
            ramp = min(1.0, (float(alpha) - float(alpha_clear)) / float(span))
            blend_weight = float(ramp) * float(ramp) * (3.0 - 2.0 * float(ramp))

        _no_constraint_gap_m = 1.0e6
        source_value = (
            _no_constraint_gap_m if source_gap is None else float(source_gap)
        )
        target_value = (
            _no_constraint_gap_m if target_gap is None else float(target_gap)
        )
        blended_gap = (
            (1.0 - blend_weight) * source_value + blend_weight * target_value
        )
        best_gap = (
            None if blended_gap >= 0.5 * _no_constraint_gap_m else float(blended_gap)
        )
        best_actor_id = (
            target_actor_id if blend_weight >= 0.5 else source_actor_id
        )
        if bool(return_actor_id):
            return best_gap, (None if best_gap is None else str(best_actor_id))
        return best_gap

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

    def _set_actuator_context(
        self,
        *,
        ego_speed_mps: float,
        target_speed_mps: float,
        stop_goal_active: bool,
    ) -> None:
        self._actuator_ego_speed_mps = float(ego_speed_mps)
        self._actuator_target_speed_mps = float(target_speed_mps)
        self._actuator_stop_goal_active = bool(stop_goal_active)

    def _control_from_mpc(self, acceleration_mps2: float, steering_angle_rad: float) -> PlannerControl:
        max_accel = max(1e-6, float(self.mpc.constraints.max_acceleration_mps2))
        max_brake = max(1e-6, abs(float(self.mpc.constraints.min_acceleration_mps2)))
        max_steer = max(1e-6, float(self.mpc.constraints.max_steer_rad))
        pedals = self.actuator_mapper.map_acceleration(
            acceleration_mps2=float(acceleration_mps2),
            max_acceleration_mps2=float(max_accel),
            min_acceleration_mps2=-float(max_brake),
            ego_speed_mps=float(self._actuator_ego_speed_mps),
            target_speed_mps=float(self._actuator_target_speed_mps),
            stop_goal_active=bool(self._actuator_stop_goal_active),
            timestamp_s=float(self._sim_time_s()),
        )
        steer = min(1.0, max(-1.0, float(steering_angle_rad) / max_steer))
        return PlannerControl(
            throttle=float(pedals.throttle),
            brake=float(pedals.brake),
            steer=steer,
        )

    def _accel_from_control(self, control: PlannerControl) -> float:
        max_accel = max(1e-6, float(self.mpc.constraints.max_acceleration_mps2))
        max_brake = max(1e-6, abs(float(self.mpc.constraints.min_acceleration_mps2)))
        return self.actuator_mapper.acceleration_from_command(
            throttle=float(getattr(control, "throttle", 0.0)),
            brake=float(getattr(control, "brake", 0.0)),
            max_acceleration_mps2=float(max_accel),
            min_acceleration_mps2=-float(max_brake),
            ego_speed_mps=float(self._actuator_ego_speed_mps),
            target_speed_mps=float(self._actuator_target_speed_mps),
            stop_goal_active=bool(self._actuator_stop_goal_active),
        )

    def _steer_rad_from_control(self, control: PlannerControl) -> float:
        max_steer = max(1e-6, float(self.mpc.constraints.max_steer_rad))
        return float(getattr(control, "steer", 0.0)) * float(max_steer)

    def _emergency_stop_control(self) -> PlannerControl:
        self._last_accel_mps2 = float(getattr(self.mpc.constraints, "min_acceleration_mps2", -3.0))
        self._last_steer_rad = 0.0
        return PlannerControl(throttle=0.0, brake=1.0, steer=0.0)

    def _full_reference_lateral_guard_reason(
        self,
        *,
        decision: str,
        lc_state: str,
        stop_goal_active: bool,
        destination_state: Sequence[float] | None,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        heading_error_rad: float = float("nan"),
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

        if math.isfinite(float(heading_error_rad)):
            max_heading_error_deg = (
                float(self.full_stop_max_heading_error_deg)
                if bool(stop_like)
                else float(self.full_lane_follow_max_heading_error_deg)
            )
            heading_error_deg = math.degrees(float(heading_error_rad))
            if abs(float(heading_error_deg)) > float(max_heading_error_deg):
                reasons.append(f"heading={heading_error_deg:.2f}")

        if not reasons:
            return ""
        mode = "stop" if bool(stop_like) else "lane_follow"
        return f"{mode}_lateral_guard:" + ":".join(reasons)


    def _traffic_control_stop_gate(
        self,
        *,
        traffic_state: str,
        stop_target: Mapping[str, object] | None,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        ego_speed_mps: float,
        ego_in_junction: bool,
    ) -> tuple[str, Mapping[str, object] | None, float, float, float, str]:
        normalized_state = str(traffic_state or "unknown").strip().lower()
        if normalized_state not in {"red", "yellow"}:
            return str(normalized_state), None, 0.0, 0.0, float(self.target_speed_mps), ""
        stop_forward_m, target_reliable = self.reference_generator.stop_target_forward(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            stop_target=stop_target,
            fallback_destination_state=[],
        )
        comfortable_decel_mps2 = max(
            0.1,
            float(self.config.get("traffic_stop_commit_decel_mps2", 2.0)),
        )
        stop_buffer_m = max(
            0.0,
            float(self.config.get("traffic_stop_commit_buffer_m", 4.0)),
        )
        min_commit_distance_m = max(
            0.0,
            float(self.config.get("traffic_stop_min_commit_distance_m", 10.0)),
        )
        commit_distance_m = max(
            float(min_commit_distance_m),
            (float(ego_speed_mps) ** 2) / (2.0 * float(comfortable_decel_mps2)) + float(stop_buffer_m),
        )
        if (
            not bool(target_reliable)
            or bool(ego_in_junction)
            or float(stop_forward_m) <= float(commit_distance_m)
        ):
            return (
                str(normalized_state),
                dict(stop_target or {}) if isinstance(stop_target, Mapping) else None,
                float(stop_forward_m),
                float(commit_distance_m),
                0.0,
                "",
            )
        far_speed_cap_mps = max(
            0.1,
            float(self.config.get("traffic_stop_approach_far_speed_cap_mps", self.target_speed_mps)),
        )
        near_speed_cap_mps = max(
            0.1,
            float(self.config.get("traffic_stop_approach_near_speed_cap_mps", 2.5)),
        )
        slow_distance_m = max(
            float(commit_distance_m),
            float(self.config.get("traffic_stop_approach_slow_distance_m", 22.0)),
        )
        if float(stop_forward_m) <= float(slow_distance_m):
            speed_cap_mps = min(float(far_speed_cap_mps), float(near_speed_cap_mps))
        else:
            speed_cap_mps = float(far_speed_cap_mps)
        speed_cap_mps = min(float(self.target_speed_mps), float(speed_cap_mps))
        return (
            "unknown",
            None,
            float(stop_forward_m),
            float(commit_distance_m),
            float(speed_cap_mps),
            (
                "traffic_stop_far_approach:"
                f"state={normalized_state}:"
                f"stop_f={float(stop_forward_m):.2f}:"
                f"commit={float(commit_distance_m):.2f}:"
                f"cap={float(speed_cap_mps):.2f}"
            ),
        )

    def _low_speed_control_buffer_force_replan(
        self,
        *,
        ego_speed_mps: float,
        behavior_decision: str,
        behavior_fsm_state: str,
        stop_goal_active: bool,
    ) -> bool:
        """Keep low-speed control closed-loop until the vehicle is moving."""

        normalized_behavior = str(behavior_decision or "").strip().lower()
        normalized_fsm = str(behavior_fsm_state or "").strip().upper()
        return bool(
            not bool(stop_goal_active)
            and normalized_behavior == "lane_follow"
            and normalized_fsm in {"", "IDLE", "LANE_KEEP"}
            and float(ego_speed_mps)
            < float(self.full_control_buffer_min_speed_mps)
        )

    def _fallback_control(
        self,
        ego_transform: PlannerTransform,
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
    def _wrap_angle(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _wrap_angle_static(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def _should_suspend_mpc_for_normal_stop(
    *,
    candidate_hard_gate_active: bool,
    stop_goal_active: bool,
    behavior_decision: str,
    ego_speed_mps: float,
    suspend_speed_mps: float,
) -> bool:
    """Return whether a committed normal stop should use deterministic hold."""

    decision = str(behavior_decision or "").strip().lower()
    return bool(
        not bool(candidate_hard_gate_active)
        and bool(stop_goal_active)
        and decision in {"stop_at_intersection", "stop_sign"}
        and float(ego_speed_mps) <= max(0.0, float(suspend_speed_mps))
    )


def _hard_gate_requires_emergency_stop(
    *,
    fallback_reason: str,
    behavior_decision: str,
    stop_goal_active: bool,
) -> bool:
    """Reserve full braking for hard gates that represent a stop hazard.

    A geometry/continuity contract veto means MPC must not consume that
    reference, but it is not evidence of an imminent collision.  Those
    failures use the bounded tracking fallback and remain subject to the
    downstream safety supervisor.  Collision, explicit stop, and emergency
    behavior retain deterministic full braking.
    """

    reason = str(fallback_reason or "").strip().lower()
    decision = str(behavior_decision or "").strip().lower()
    if not reason.startswith("candidate_hard_gate:"):
        return False
    if bool(stop_goal_active) or decision in {
        "emergency_brake",
        "stop_at_intersection",
        "stop_sign",
    }:
        return True
    hazard_tokens = (
        "collision_risk",
        "emergency_brake_direct_control",
        "stop_missing_target_hard_lock",
    )
    return any(token in reason for token in hazard_tokens)


def _mpc_cost_profile_for_behavior(
    *,
    behavior: str,
    planner_lc_state: str,
    planner_mode: str,
    next_macro_maneuver: str,
) -> str:
    from cpx_planning.behavior_planner import (
        is_emergency_brake_decision,
        is_fixed_stop_decision,
        normalize_behavior_decision,
    )

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
    if normalized_lc_state.startswith("EXECUTE_LANE_CHANGE") or normalized_behavior in {
        "lane_change_left",
        "lane_change_right",
    }:
        return "execute_lane_change"
    if normalized_mode == "INTERSECTION" and normalized_maneuver in {"left", "right"}:
        return "intersection_turn"
    return "lane_follow"


_DEFAULT_ADAPTIVE_HORIZON_PROFILE_S: dict[str, float] = {
    "lane_follow": 3.0,
    "prepare_lane_change": 4.5,
    "execute_lane_change": 4.5,
    "intersection_turn": 2.2,
    "stop": 2.0,
    "recovery": 1.5,
}


def _adaptive_target_horizon_s(
    *,
    mpc_cost_profile: str,
    nearest_obstacle_distance_m: Optional[float],
    ego_speed_mps: float,
    profile_horizon_s: Mapping[str, float],
    obstacle_reference_speed_mps: float = 2.0,
    obstacle_comfortable_decel_mps2: float = 2.0,
) -> float:
    """Pick a prediction-horizon target from the active behavior mode, then
    shorten it further if a nearby obstacle needs quicker reaction -- a
    human driver looks less far ahead through a tight turn than down an open
    lane, and less still when something close needs immediate attention.

    The obstacle term is the MAX of two independent estimates, not a single
    distance/current_speed ratio: that ratio blows up as ego comfortably
    decelerates toward a stop behind a closing lead vehicle (the exact
    "shouldn't horizon keep shrinking here?" case this was built for) --
    dividing by ego's own shrinking speed makes the estimate grow right when
    it should keep shrinking. distance_reaction_s (distance over a fixed
    reference speed, not ego's live one) shrinks monotonically as the gap
    closes regardless of ego's speed; stopping_time_s (ego's own speed over a
    comfortable deceleration) shrinks to 0 as ego actually comes to a stop.
    Taking the max avoids either term alone causing a premature shrink (e.g.
    ego already slow with an unrelated, still-distant obstacle ahead).
    """

    base = float(
        profile_horizon_s.get(
            str(mpc_cost_profile),
            profile_horizon_s.get("lane_follow", 3.0),
        )
    )
    if nearest_obstacle_distance_m is not None and math.isfinite(
        float(nearest_obstacle_distance_m)
    ):
        distance_reaction_s = float(nearest_obstacle_distance_m) / max(
            1.0e-3, float(obstacle_reference_speed_mps)
        )
        stopping_time_s = float(ego_speed_mps) / max(
            1.0e-3, float(obstacle_comfortable_decel_mps2)
        )
        reaction_s = max(distance_reaction_s, stopping_time_s)
        base = min(base, max(1.0, reaction_s))
    return float(base)


def _select_mpc_cost_profile_with_hysteresis(
    *,
    requested_profile: str,
    active_profile: str,
    sim_time_s: float,
    active_since_s: float,
    min_hold_s: float,
) -> tuple[str, float, str]:
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


def cpx_planner_enabled(config: Mapping[str, Any]) -> bool:
    """Return whether a vehicle config requests the CP-X planner bridge."""

    planner_cfg = dict(config.get("planner", {}) or {})
    if planner_cfg and not bool(planner_cfg.get("enabled", True)):
        return False
    planner_type = str(planner_cfg.get("type", "")).strip().lower()
    env_type = str(os.environ.get("OPENCDA_PLANNER", "")).strip().lower()
    if planner_type:
        return planner_type in {"cpx_mpc", "cp_x_mpc"}
    return env_type in {"cpx_mpc", "cp_x_mpc"}
