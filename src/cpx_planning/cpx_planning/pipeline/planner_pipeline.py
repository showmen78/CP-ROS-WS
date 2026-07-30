"""Run the CP-X behavior-planning stage without OpenCDA or CARLA."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

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
from cpx_planning.pipeline.output import BehaviorCommand as PlannerBehaviorCommand
from cpx_planning.pipeline.output import PlannerDiagnostics, PlannerOutput

from cpx_planning.pipeline.candidate_pipeline import CandidateReferenceResult
from cpx_planning.pipeline.candidate_pipeline import build_candidate_intents
from cpx_planning.pipeline.candidate_pipeline import evaluate_candidate_reference
from cpx_planning.pipeline.candidate_pipeline import select_best_candidate
from cpx_planning.pipeline.candidate_pipeline import summarize_candidate_results
from cpx_planning.pipeline.reference_contract import contract_from_config, validate_reference_contract


class _Mode2TrafficLightMemory:
    """Keep a traffic-light state briefly when one ROS message is missing or changes too quickly."""

    def __init__(self, *, hold_unknown_s: float = 0.8, green_confirm_s: float = 0.0) -> None:
        """Save how long an unknown state is held and how long green must remain before releasing a stop."""
        self.hold_unknown_s = max(0.0, float(hold_unknown_s))
        self.green_confirm_s = max(0.0, float(green_confirm_s))
        self._last_stop_state = "unknown"
        self._last_stop_target = None
        self._hold_until_s = -float("inf")
        self._green_since_s = None

    def update(self, *, state: str, stop_target: Mapping[str, object] | None, sim_time_s: float):
        """Filter one traffic-light reading and return its state, stop target, and filtering reason."""
        normalized_state = str(state or "unknown").strip().lower()
        reason = ""

        if normalized_state in {"red", "yellow"}:
            self._green_since_s = None
            self._last_stop_state = normalized_state
            self._last_stop_target = dict(stop_target or {}) if isinstance(stop_target, Mapping) else None
            self._hold_until_s = float(sim_time_s) + self.hold_unknown_s
            return normalized_state, self._last_stop_target, "raw_stop"

        if normalized_state == "green":
            if self._green_since_s is None:
                self._green_since_s = float(sim_time_s)

            if self._last_stop_state in {"red", "yellow"} and float(sim_time_s) - float(self._green_since_s) < self.green_confirm_s:
                return self._last_stop_state, self._last_stop_target, "traffic_memory_wait_green_confirm"

            self._last_stop_state = "green"
            self._last_stop_target = None
            self._hold_until_s = -float("inf")
            reason = "traffic_memory_green_release" if self.green_confirm_s > 0.0 else ""
            return "green", None, reason

        if normalized_state == "unknown" and float(sim_time_s) <= self._hold_until_s and self._last_stop_state in {"red", "yellow"}:
            return self._last_stop_state, self._last_stop_target, "traffic_memory_hold_{}".format(self._last_stop_state)

        if normalized_state == "unknown":
            self._green_since_s = None

        return normalized_state, None, reason


class CPXPlanningPipeline:
    """Run the same behavior-planning order used by the OpenCDA CP-X planner."""

    def __init__(self, *, behavior_planner, route_manager, global_planner, mpc, control_buffer, mpc_feedback, behavior_runtime_cfg=None, config=None):
        """Receive the independent behavior and global-planning components and keep state between planning cycles."""
        self.behavior_planner = behavior_planner
        self.route_manager = route_manager
        self.global_planner = global_planner
        
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
        self._last_planned_trajectory = []
        self.last_destination_state = None
        self.last_reference_debug = {}
        self.last_behavior_command = {}
        self.active_mpc_cost_profile = "lane_follow"
        self.requested_mpc_cost_profile = "lane_follow"
        self.mpc_cost_profile_active_since_s = 0.0
        self.mpc_cost_profile_switch_reason = "initial"
        
        self.config = dict(config or {})
        self.scenario_manager = CPXScenarioManager(self.config)
        self.traffic_memory = _Mode2TrafficLightMemory(
            hold_unknown_s=float(self.config.get("full_traffic_unknown_hold_s", 0.25)),
            green_confirm_s=float(self.config.get("full_traffic_green_confirm_s", 0.5)),
        )
        self._latched_stop_target = None
        self._latched_stop_state = "unknown"
        self._turn_latch_decision = ""
        self._turn_latch_until_sim_time_s = -float("inf")

    def run_behavior(self, adapter_output) -> Dict[str, object]:
        """Use one PlannerInputFrame to produce the same behavior decision used before reference generation in OpenCDA."""
        planner_input_frame = adapter_output.frame
        route_context = planner_input_frame.planning.route
        sim_time_s = float(planner_input_frame.planning.sim_time_s)
        current_lane_id = int(adapter_output.current_lane_id)
        ego_x_m = float(planner_input_frame.planning.ego.x_m)
        ego_y_m = float(planner_input_frame.planning.ego.y_m)
        ego_heading_rad = float(planner_input_frame.planning.ego.heading_rad)
        ego_speed_mps = float(planner_input_frame.planning.ego.speed_mps)
        lane_safety_scores = dict(adapter_output.lane_safety_scores)
        lane_prediction_risks = dict(planner_input_frame.prediction.lane_prediction_risks)
        front_dist_by_lane = dict(adapter_output.front_distance_by_lane)
        route_points = list(adapter_output.route_points)
        route_optimal_lane_id = int(adapter_output.route_optimal_lane_id)
        route_reference_allowed = bool(adapter_output.route_reference_allowed)
        route_reference_gate_reason = str(adapter_output.route_reference_gate_reason)
        route_lane_change_allowed = route_reference_allowed and "direct_fallback" not in route_reference_gate_reason

        lane_change_authorization = authorize_route_lane_change(
            route_lane_change_allowed=route_lane_change_allowed,
            current_lane_id=current_lane_id,
            route_required_lane_id=route_optimal_lane_id,
            next_macro_maneuver=str(route_context.next_macro_maneuver),
            current_road_option=str(route_context.current_road_option),
            remaining_distance_m=float(route_context.remaining_distance_m),
            available_lane_ids=list(planner_input_frame.map_lane.allowed_lane_ids),
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=lane_prediction_risks,
            preparation_start_distance_m=float(self.config.get("route_lane_change_preparation_start_distance_m", 45.0)),
            latest_start_distance_m=float(self.config.get("route_lane_change_latest_start_distance_m", 12.0)),
            target_safety_threshold=float(self.config.get("route_lane_change_target_safety_threshold", 0.65)),
            require_adjacent=bool(self.config.get("route_lane_change_require_adjacent", True)),
        )

        route_lane_change_required = bool(lane_change_authorization.required_by_route)
        prediction_risky_lane_count = sum(1 for risk in lane_prediction_risks.values() if bool(dict(risk or {}).get("risk", False)))
        dense_traffic_active = bool(self.config.get("full_dense_traffic_lane_change_lock_enabled", True)) and (
            planner_input_frame.perception.planning_count >= int(self.config.get("full_dense_traffic_object_count", 8))
            or prediction_risky_lane_count >= int(self.config.get("full_dense_traffic_risky_lane_count", 2))
        )
        start_lane_change_lock_active = sim_time_s <= float(self.config.get("full_lane_change_start_lock_s", 8.0))
        lane_change_authorized = bool(lane_change_authorization.allowed)
        opportunistic_lane_change_allowed = route_lane_change_allowed and (
            lane_change_authorized
            or (
                bool(self.config.get("full_allow_opportunistic_lane_change", False))
                and not start_lane_change_lock_active
                and not dense_traffic_active
            )
        )

        lane_change_gate_reason = ""
        if route_lane_change_allowed and not opportunistic_lane_change_allowed:
            reasons = []
            if start_lane_change_lock_active:
                reasons.append("start_lock")
            if dense_traffic_active:
                reasons.append("dense_traffic")
            if not lane_change_authorized:
                reasons.append(str(lane_change_authorization.reason))
            lane_change_gate_reason = "opportunistic_lane_change_suppressed:" + "+".join(reasons)

        raw_stop_target = None
        if planner_input_frame.planning.traffic_control.stop_target.active:
            raw_stop_target = planner_input_frame.planning.traffic_control.stop_target.as_dict()

        filtered_traffic_state, filtered_stop_target, traffic_memory_reason = self.traffic_memory.update(
            state=str(planner_input_frame.planning.traffic_control.signal_state),
            stop_target=raw_stop_target,
            sim_time_s=sim_time_s,
        )

        filtered_stop_target, stop_latch_reason = self._latched_stop_target_for_signal(
            traffic_state=filtered_traffic_state,
            stop_target=filtered_stop_target,
            ego_x_m=ego_x_m,
            ego_y_m=ego_y_m,
            ego_heading_rad=ego_heading_rad,
            current_lane_id=current_lane_id,
        )

        if stop_latch_reason:
            traffic_memory_reason = "{};{}".format(traffic_memory_reason, stop_latch_reason) if traffic_memory_reason else stop_latch_reason

        traffic_stop_forward_m, traffic_stop_target_reliable = self._stop_target_forward_m(
            ego_x_m=ego_x_m,
            ego_y_m=ego_y_m,
            ego_heading_rad=ego_heading_rad,
            stop_target=filtered_stop_target,
        )

        scenario_decision = self.scenario_manager.update(
            traffic_state=filtered_traffic_state,
            stop_target=filtered_stop_target,
            stop_forward_m=traffic_stop_forward_m,
            stop_target_reliable=traffic_stop_target_reliable,
            ego_speed_mps=ego_speed_mps,
            ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
            current_road_option=str(route_context.current_road_option),
            next_macro_maneuver=str(route_context.next_macro_maneuver),
            sim_time_s=sim_time_s,
        )

        behavior_traffic_state = str(scenario_decision.behavior_signal_state)
        behavior_stop_target = dict(scenario_decision.behavior_stop_target) if isinstance(scenario_decision.behavior_stop_target, Mapping) else None
        signal_context = dict(adapter_output.signal_context)
        signal_context["raw_signal_state"] = str(planner_input_frame.planning.traffic_control.signal_state)
        signal_context["signal_state"] = str(filtered_traffic_state)
        signal_context["behavior_signal_state"] = behavior_traffic_state
        signal_context["traffic_stop_forward_m"] = traffic_stop_forward_m
        signal_context["traffic_stop_commit_distance_m"] = float(scenario_decision.traffic_stop_commit_distance_m)

        if scenario_decision.reason:
            signal_context["traffic_stop_approach_reason"] = str(scenario_decision.reason)

        if traffic_memory_reason:
            signal_context["traffic_memory_reason"] = traffic_memory_reason

        if lane_change_authorized:
            candidate_lane_ids = [current_lane_id, int(lane_change_authorization.target_lane_id)]
        elif opportunistic_lane_change_allowed:
            candidate_lane_ids = list(planner_input_frame.map_lane.allowed_lane_ids)
        else:
            candidate_lane_ids = [current_lane_id]


        mpc_feedback = self.mpc_feedback.candidate_feedback(current_time_s=sim_time_s)
        candidate_frame = evaluate_behavior_candidates(
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=lane_prediction_risks,
            ego_lane_id=current_lane_id,
            selected_lane_id=current_lane_id,
            available_lane_ids=candidate_lane_ids,
            route_optimal_lane_id=route_optimal_lane_id,
            mode="INTERSECTION" if planner_input_frame.map_lane.in_junction else "NORMAL",
            mpc_feedback_blocked_lane_ids=list(mpc_feedback.get("blocked_lane_ids", []) or []),
            mpc_feedback_weight=float(self.config.get("mpc_feedback_candidate_weight", 80.0)),
        )

        if lane_change_authorized:
            preferred_target_lane_id = int(lane_change_authorization.target_lane_id)
        elif opportunistic_lane_change_allowed:
            preferred_target_lane_id = int(candidate_frame.selected.target_lane_id)
        else:
            preferred_target_lane_id = current_lane_id

        command = self.behavior_planner.update(
            lane_safety_scores=lane_safety_scores,
            ego_lane_id=current_lane_id,
            selected_lane_id=current_lane_id,
            ego_lateral_offset_m=0.0,
            ego_heading_error_rad=0.0,
            mode="INTERSECTION" if planner_input_frame.map_lane.in_junction else "NORMAL",
            route_optimal_lane_id=route_optimal_lane_id,
            next_macro_maneuver=str(route_context.next_macro_maneuver),
            front_obstacle_distance_by_lane=front_dist_by_lane,
            current_time_s=sim_time_s,
            wall_time_s=sim_time_s,
            traffic_signal_state=behavior_traffic_state,
            traffic_stop_target=behavior_stop_target,
            traffic_signal_context=signal_context,
            ego_speed_mps=ego_speed_mps,
            ego_max_deceleration_mps2=float(self.config.get("ego_max_deceleration_mps2", 3.0)),
            ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
            ego_position_xy=(ego_x_m, ego_y_m),
            global_route_points=route_points,
            nearest_front_obstacles_by_lane={},
            lane_prediction_risks=lane_prediction_risks,
            preferred_target_lane_id=preferred_target_lane_id,
            lane_closure_messages=list(planner_input_frame.cp_messages.lane_closures),
        )

        decision = str(command.get("decision", "lane_follow"))
        target_lane_id = int(command.get("target_lane_id", current_lane_id) or current_lane_id)
        lc_state = str(command.get("lc_state", "LANE_KEEP"))
        speed_ref_mps = float(self.config.get("target_speed_mps", 10.0))
        behavior_override_reasons = []

        if scenario_decision.speed_cap_mps is not None and float(scenario_decision.speed_cap_mps) < speed_ref_mps:
            if decision not in {"stop_at_intersection", "stop_sign", "emergency_brake"}:
                speed_ref_mps = min(speed_ref_mps, float(scenario_decision.speed_cap_mps))
                behavior_override_reasons.append(str(scenario_decision.reason))

        route_turn_decision = self._route_option_turn_decision(
            current_road_option=str(route_context.current_road_option),
            next_macro_maneuver=str(route_context.next_macro_maneuver),
        )

        route_turn_prepare_decision = ""
        if not route_turn_decision and bool(self.config.get("full_intersection_turn_prepare_enabled", False)):
            route_turn_prepare_decision = self._route_lookahead_turn_decision(
                ego_x_m=ego_x_m,
                ego_y_m=ego_y_m,
                ego_heading_rad=ego_heading_rad,
                route_points=route_points,
            )

        if not opportunistic_lane_change_allowed and decision in {"lane_change_left", "lane_change_right"}:
            decision = "lane_follow"
            target_lane_id = current_lane_id
            lc_state = "LANE_KEEP"
            reason = lane_change_gate_reason or "lane_change_suppressed_without_valid_route"
            behavior_override_reasons.append(reason)
            self.behavior_planner._reset_lane_change_state(reason=reason)

        if bool(self.config.get("full_prepare_lane_change_reference_lock", True)) and lc_state.upper().startswith("PREPARE_LANE_CHANGE"):
            decision = "lane_follow"
            target_lane_id = current_lane_id
            lc_state = "LANE_KEEP"
            behavior_override_reasons.append("prepare_lane_change_reference_locked_to_current_lane")
            self.behavior_planner._reset_lane_change_state(reason="prepare_lane_change_reference_locked")

        if decision in {"lane_change_left", "lane_change_right"} and not lane_change_authorized:
            decision = "lane_follow"
            target_lane_id = current_lane_id
            lc_state = "LANE_KEEP"
            behavior_override_reasons.append("lane_change_without_authorization:{}".format(lane_change_authorization.reason))
            self.behavior_planner._reset_lane_change_state(reason="lane_change_without_authorization")

        scenario_behavior_override = str(scenario_decision.behavior_override_decision or "")
        if scenario_behavior_override:
            decision = scenario_behavior_override
            target_lane_id = current_lane_id
            lc_state = str(scenario_decision.behavior_override_lc_state or "LANE_KEEP")
            if scenario_decision.speed_cap_mps is not None:
                speed_ref_mps = min(speed_ref_mps, float(scenario_decision.speed_cap_mps))
            behavior_override_reasons.append(str(scenario_decision.reason))

        stop_goal_active = bool(command.get("stop", False) or scenario_decision.stop_goal_active)

        if (route_turn_decision or route_turn_prepare_decision) and not scenario_behavior_override:
            if decision not in {"stop_at_intersection", "stop_sign", "emergency_brake"}:
                decision = route_turn_decision or route_turn_prepare_decision
                target_lane_id = current_lane_id
                lc_state = "INTERSECTION_TURN_LEFT" if decision.endswith("_left") else "INTERSECTION_TURN_RIGHT"
                speed_ref_mps = min(speed_ref_mps, float(self.config.get("full_intersection_turn_prepare_speed_cap_mps", 2.2)))
                reason = "route_option_driven_behavior:{}".format(route_context.current_road_option) if route_turn_decision else "route_lookahead_prepare_turn"
                behavior_override_reasons.append(reason)

        turn_latch_reason = ""
        if decision not in {"stop_at_intersection", "stop_sign", "emergency_brake"} and not scenario_decision.turn_latched:
            decision, lc_state, speed_ref_mps, turn_latch_reason = self._apply_turn_direction_latch(
                decision=decision,
                lc_state=lc_state,
                speed_ref_mps=speed_ref_mps,
                current_road_option=str(route_context.current_road_option),
                next_macro_maneuver=str(route_context.next_macro_maneuver),
                ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
                sim_time_s=sim_time_s,
            )

            if turn_latch_reason:
                target_lane_id = current_lane_id
                behavior_override_reasons.append(turn_latch_reason)

        speed_plan = build_speed_plan(
            scenario_decision=scenario_decision,
            behavior_decision=decision,
            requested_speed_mps=speed_ref_mps,
            ego_speed_mps=ego_speed_mps,
            config=self.config,
        )

        speed_ref_mps = float(speed_plan.target_speed_mps)
        stop_goal_active = bool(stop_goal_active or speed_plan.stop_goal_active)
        reroute_debug_reason = ""

        if decision == "reroute":
            reroute_debug_reason = self._perform_reroute(
                command=command,
                planner_input_frame=planner_input_frame,
                adapter_output=adapter_output,
                current_route_points=route_points,
            )

        command["decision"] = decision
        command["target_lane_id"] = target_lane_id
        command["lc_state"] = lc_state
        command["fsm_state"] = lc_state
        command["target_speed_mps"] = speed_ref_mps
        command["stop_goal_active"] = stop_goal_active
        command["behavior_override_reason"] = ";".join(reason for reason in behavior_override_reasons if reason)
        command["route_lane_change_allowed"] = route_lane_change_allowed
        command["route_lane_change_required"] = route_lane_change_required
        command["opportunistic_lane_change_allowed"] = opportunistic_lane_change_allowed
        command["lane_change_gate_reason"] = lane_change_gate_reason
        command["lane_change_authorized"] = lane_change_authorized
        command["lane_change_authorization_reason"] = str(lane_change_authorization.reason)
        command["candidate_evaluation_summary"] = candidate_frame.summary()
        command["candidate_selected_decision"] = str(candidate_frame.selected.decision)
        command["candidate_selected_lane_id"] = int(candidate_frame.selected.target_lane_id)
        command["scenario_state"] = str(scenario_decision.state)
        command["scenario_reason"] = str(scenario_decision.reason)
        command["traffic_memory_reason"] = traffic_memory_reason
        command["traffic_stop_forward_m"] = traffic_stop_forward_m
        command["turn_latch_reason"] = turn_latch_reason
        command["reroute_debug_reason"] = reroute_debug_reason
        command["mpc_feedback_summary"] = str(mpc_feedback.get("summary", ""))
        command["mpc_feedback_blocked_lane_ids"] = list(mpc_feedback.get("blocked_lane_ids", []) or [])
        
        command["candidate_lane_ids"] = list(candidate_lane_ids)
        command["lane_change_authorized_target_lane_id"] = int(lane_change_authorization.target_lane_id) if lane_change_authorized else 0
        command["traffic_stop_active"] = bool(scenario_decision.stop_goal_active)
        command["behavior_stop_target"] = dict(behavior_stop_target) if isinstance(behavior_stop_target, Mapping) else None
                
                
        return command
    
    def _plan_behavior_and_reference(self, adapter_output):
        """Run behavior planning and generate the temporary destination and MPC reference for the same planning frame."""
        planner_input_frame = adapter_output.frame
        behavior_command = self.run_behavior(adapter_output)
        ego_pose = dict(adapter_output.ego_pose)
        current_state = list(adapter_output.current_state)
        current_lane_id = int(adapter_output.current_lane_id)
        route_points = list(adapter_output.route_points)
        route_optimal_lane_id = int(adapter_output.route_optimal_lane_id)
        route_reference_allowed = bool(adapter_output.route_reference_allowed)
        route_reference_gate_reason = str(adapter_output.route_reference_gate_reason)
        sim_time_s = float(planner_input_frame.planning.sim_time_s)
        ego_speed_mps = float(planner_input_frame.planning.ego.speed_mps)
        decision = str(behavior_command.get("decision", "lane_follow"))
        target_lane_id = int(behavior_command.get("target_lane_id", current_lane_id) or current_lane_id)
        lc_state = str(behavior_command.get("lc_state", "LANE_KEEP"))
        speed_ref_mps = float(behavior_command.get("target_speed_mps", self.config.get("target_speed_mps", 10.0)))
        stop_goal_active = bool(behavior_command.get("stop_goal_active", False))
        planner_mode = "INTERSECTION" if bool(planner_input_frame.map_lane.in_junction) else "NORMAL"

        self._apply_mpc_cost_profile(
            behavior=decision,
            planner_lc_state=lc_state,
            planner_mode=planner_mode,
            next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            sim_time_s=sim_time_s,
        )

        base_temporary_destination_state = list(self._temporary_destination_state) if self._temporary_destination_state is not None else None

        self._temporary_destination_state = compute_temp_destination(
            map_planner=self.global_planner,
            ego_pose=ego_pose,
            target_lane_id=target_lane_id,
            decision=decision,
            lookahead_m=float(self.config.get("lookahead_m", 18.0)),
            target_v_mps=speed_ref_mps,
            global_route_points=route_points,
            mode_reference_xy=None if base_temporary_destination_state is None else (float(base_temporary_destination_state[0]), float(base_temporary_destination_state[1])),
            prev_mode=None if base_temporary_destination_state is None or len(base_temporary_destination_state) < 6 else float(base_temporary_destination_state[5]),
            prev_road_id=None if base_temporary_destination_state is None or len(base_temporary_destination_state) < 7 else int(base_temporary_destination_state[6]),
            prev_entered_intersection=False if base_temporary_destination_state is None or len(base_temporary_destination_state) < 8 else bool(float(base_temporary_destination_state[7]) > 0.5),
            next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            mode_override=planner_mode,
            follow_global_route_lane=bool(route_reference_allowed and planner_input_frame.map_lane.in_junction),
        )

        reference_intent = select_reference_intent(
            behavior_decision=decision,
            planner_fsm_state=lc_state,
            ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
            reference_target_lane_id=target_lane_id,
            current_lane_id=current_lane_id,
            route_optimal_lane_id=route_optimal_lane_id,
            global_route_reference_allowed=route_reference_allowed,
            traffic_control_lane_lock_active=False,
        )

        reference_context = MpcReferenceGenerationContext(
            map_planner=self.global_planner,
            ego_pose=ego_pose,
            ego_state=current_state,
            active_global_route_points=route_points,
            previous_lane_center_reference=self._previous_lane_center_reference,
            behavior_runtime_cfg=self.behavior_runtime_cfg,
            reference_intent=reference_intent,
            current_applied_behavior=decision,
            cached_planner_lc_state=lc_state,
            reference_target_lane_id=target_lane_id,
            current_lane_id=current_lane_id,
            global_route_reference_allowed=route_reference_allowed,
            global_route_reference_gate_reason=route_reference_gate_reason,
            should_follow_global_route_lane_for_reference=bool(reference_intent.follow_global_route_lane),
            traffic_control_lane_lock_active=False,
            final_goal_stop_active=False,
            stop_target_state=None,
            follow_target_state=None,
            current_temp_reference_xy=(float(self._temporary_destination_state[0]), float(self._temporary_destination_state[1])),
            current_temp_mode_value=float(self._temporary_destination_state[5]) if len(self._temporary_destination_state) >= 6 else 0.0,
            current_temp_road_id=int(self._temporary_destination_state[6]) if len(self._temporary_destination_state) >= 7 else None,
            current_temp_entered_intersection=bool(float(self._temporary_destination_state[7]) > 0.5) if len(self._temporary_destination_state) >= 8 else False,
            active_reference_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
            current_temp_mode_str=planner_mode,
            lane_reference_speed_mps=max(1.0, ego_speed_mps, abs(speed_ref_mps)),
            lane_reference_step_distance_m=max(0.5, float(self.mpc.dt_s) * max(1.0, ego_speed_mps, abs(speed_ref_mps))),
            mpc_horizon_steps=int(self.mpc.horizon_steps),
            mpc_dt_s=float(self.mpc.dt_s),
            temporary_destination_state=self._temporary_destination_state,
            lane_reference_freeze_count=int(self._lane_reference_freeze_count),
            sim_time_s=sim_time_s,
            stop_release_temp_smooth_until_sim_time_s=float(self._stop_release_temp_smooth_until_sim_time_s),
        )

        reference_output = generate_mpc_reference(reference_context)
        lane_center_reference = [dict(sample) for sample in list(reference_output.local_lane_center_reference or [])]
        destination_state = list(reference_output.temporary_destination_state or self._temporary_destination_state)
        candidate_freeze_count = int(reference_output.lane_reference_freeze_count)

        reference_debug = dict(reference_output.mpc_reference_result.trace.as_trace_fields())
        reference_debug.update(planner_input_frame.trace_fields())
        reference_debug.update({
            "stage": reference_debug.get("reference_pipeline_stage", ""),
            "intent_mode": reference_debug.get("reference_pipeline_intent_mode", ""),
            "fallback_reason": str(reference_output.last_reference_fallback_reason),
            "reference_source": "behavior_reference_pipeline",
            "route_reference_allowed": route_reference_allowed,
            "route_reference_gate_reason": route_reference_gate_reason,
            "current_lane_id": current_lane_id,
            "target_lane_id": target_lane_id,
            "target_speed_mps": speed_ref_mps,
            "stop_goal_active": stop_goal_active,
            "mpc_cost_profile": self.active_mpc_cost_profile,
            "requested_mpc_cost_profile": self.requested_mpc_cost_profile,
            "mpc_cost_profile_switch_reason": self.mpc_cost_profile_switch_reason,
        })
        reference_debug.update(dict(adapter_output.source_quality))

        baseline_decision = str(decision)
        baseline_lc_state = str(lc_state)

        if bool(self.config.get("full_candidate_pipeline_enabled", True)):
            candidate_intents = build_candidate_intents(
                selected_decision=decision,
                selected_target_lane_id=target_lane_id,
                current_lane_id=current_lane_id,
                target_speed_mps=speed_ref_mps,
                candidate_lane_ids=list(behavior_command.get("candidate_lane_ids", [current_lane_id])),
                lane_safety_scores=dict(adapter_output.lane_safety_scores),
                lane_prediction_risks=dict(planner_input_frame.prediction.lane_prediction_risks),
                stop_goal_active=stop_goal_active,
                traffic_stop_active=bool(behavior_command.get("traffic_stop_active", False)),
                lane_change_authorized=bool(behavior_command.get("lane_change_authorized", False)),
                lane_change_authorized_target_lane_id=int(behavior_command.get("lane_change_authorized_target_lane_id", 0)),
                allow_lane_change_candidates=bool(behavior_command.get("opportunistic_lane_change_allowed", False)),
                stop_target=behavior_command.get("behavior_stop_target"),
            )

            decision, target_lane_id, speed_ref_mps, lane_center_reference, destination_state, candidate_debug = self._select_candidate_reference_for_mpc(
                candidate_intents=candidate_intents,
                baseline_decision=baseline_decision,
                baseline_lc_state=baseline_lc_state,
                baseline_target_lane_id=target_lane_id,
                baseline_speed_ref_mps=speed_ref_mps,
                baseline_destination_state=destination_state,
                baseline_reference=lane_center_reference,
                baseline_reference_debug=reference_debug,
                base_temporary_destination_state=base_temporary_destination_state,
                adapter_output=adapter_output,
                planner_mode=planner_mode,
            )

            reference_debug.update(candidate_debug)
            reference_debug["candidate_pipeline_enabled"] = True
        else:
            reference_debug["candidate_pipeline_enabled"] = False

        lc_state = self._candidate_lc_state(decision=decision, baseline_decision=baseline_decision, baseline_lc_state=baseline_lc_state)
        stop_goal_active = bool(stop_goal_active or decision in {"stop_at_intersection", "stop_sign", "emergency_brake"})

        reference_debug.update({
            "current_lane_id": current_lane_id,
            "target_lane_id": target_lane_id,
            "target_speed_mps": speed_ref_mps,
            "stop_goal_active": stop_goal_active,
        })

        self._temporary_destination_state = list(destination_state)
        self._lane_reference_freeze_count = candidate_freeze_count
        self._previous_lane_center_reference = [dict(sample) for sample in lane_center_reference]

        behavior_command["decision"] = decision
        behavior_command["current_lane_id"] = current_lane_id
        behavior_command["target_lane_id"] = target_lane_id
        behavior_command["target_speed_mps"] = speed_ref_mps
        behavior_command["lc_state"] = lc_state
        behavior_command["fsm_state"] = lc_state
        behavior_command["stop_goal_active"] = stop_goal_active

        self.last_destination_state = list(destination_state)
        self.last_reference_debug = dict(reference_debug)
        self.last_behavior_command = dict(behavior_command)

        return list(destination_state), lane_center_reference, behavior_command, reference_debug


    def _select_candidate_reference_for_mpc(
        self,
        *,
        candidate_intents,
        baseline_decision,
        baseline_lc_state,
        baseline_target_lane_id,
        baseline_speed_ref_mps,
        baseline_destination_state,
        baseline_reference,
        baseline_reference_debug,
        base_temporary_destination_state,
        adapter_output,
        planner_mode,
    ):
        """Generate and compare a reference for every allowed behavior candidate, then return the safest valid candidate."""
        planner_input_frame = adapter_output.frame
        ego_pose = dict(adapter_output.ego_pose)
        current_state = list(adapter_output.current_state)
        current_lane_id = int(adapter_output.current_lane_id)
        route_optimal_lane_id = int(adapter_output.route_optimal_lane_id)
        route_points = list(adapter_output.route_points)
        route_reference_allowed = bool(adapter_output.route_reference_allowed)
        route_reference_gate_reason = str(adapter_output.route_reference_gate_reason)
        ego_speed_mps = float(planner_input_frame.planning.ego.speed_mps)
        sim_time_s = float(planner_input_frame.planning.sim_time_s)
        object_snapshots = [dict(item) for item in list(planner_input_frame.perception.planning_objects or []) if isinstance(item, Mapping)]
        prediction_trajectories = dict(planner_input_frame.prediction.obstacle_future_trajectories)
        intents = list(candidate_intents or [])

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
                    "candidate_prediction_trajectory_count": len(prediction_trajectories),
                    "candidate_pipeline_summary": "[]",
                },
            )

        candidate_results = []

        for intent in intents:
            candidate_decision = str(getattr(intent, "decision", baseline_decision))
            candidate_target_lane_id = int(getattr(intent, "target_lane_id", current_lane_id) or current_lane_id)
            candidate_speed_ref_mps = float(getattr(intent, "target_speed_mps", baseline_speed_ref_mps))
            candidate_stop_goal_active = bool(getattr(intent, "stop_goal_active", False)) or candidate_decision in {"stop_at_intersection", "stop_sign", "emergency_brake"}
            candidate_lc_state = self._candidate_lc_state(decision=candidate_decision, baseline_decision=baseline_decision, baseline_lc_state=baseline_lc_state)

            same_as_baseline = candidate_decision == str(baseline_decision) and candidate_target_lane_id == int(baseline_target_lane_id) and abs(candidate_speed_ref_mps - float(baseline_speed_ref_mps)) < 1.0e-3

            if same_as_baseline:
                destination_state = list(baseline_destination_state or [])
                reference = [dict(sample) for sample in list(baseline_reference or [])]
                candidate_reference_debug = dict(baseline_reference_debug or {})
            else:
                previous_temp = list(base_temporary_destination_state or [])

                candidate_temp_destination = compute_temp_destination(
                    map_planner=self.global_planner,
                    ego_pose=ego_pose,
                    target_lane_id=candidate_target_lane_id,
                    decision=candidate_decision,
                    lookahead_m=float(self.config.get("lookahead_m", 18.0)),
                    target_v_mps=candidate_speed_ref_mps,
                    global_route_points=route_points,
                    mode_reference_xy=None if not previous_temp else (float(previous_temp[0]), float(previous_temp[1])),
                    prev_mode=None if len(previous_temp) < 6 else float(previous_temp[5]),
                    prev_road_id=None if len(previous_temp) < 7 else int(previous_temp[6]),
                    prev_entered_intersection=False if len(previous_temp) < 8 else bool(float(previous_temp[7]) > 0.5),
                    next_macro_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
                    mode_override=str(planner_mode),
                    follow_global_route_lane=bool(route_reference_allowed and planner_input_frame.map_lane.in_junction),
                )

                candidate_reference_intent = select_reference_intent(
                    behavior_decision=candidate_decision,
                    planner_fsm_state=candidate_lc_state,
                    ego_in_junction=bool(planner_input_frame.map_lane.in_junction),
                    reference_target_lane_id=candidate_target_lane_id,
                    current_lane_id=current_lane_id,
                    route_optimal_lane_id=route_optimal_lane_id,
                    global_route_reference_allowed=route_reference_allowed,
                    traffic_control_lane_lock_active=False,
                )

                candidate_reference_context = MpcReferenceGenerationContext(
                    map_planner=self.global_planner,
                    ego_pose=ego_pose,
                    ego_state=current_state,
                    active_global_route_points=route_points,
                    previous_lane_center_reference=self._previous_lane_center_reference,
                    behavior_runtime_cfg=self.behavior_runtime_cfg,
                    reference_intent=candidate_reference_intent,
                    current_applied_behavior=candidate_decision,
                    cached_planner_lc_state=candidate_lc_state,
                    reference_target_lane_id=candidate_target_lane_id,
                    current_lane_id=current_lane_id,
                    global_route_reference_allowed=route_reference_allowed,
                    global_route_reference_gate_reason=route_reference_gate_reason,
                    should_follow_global_route_lane_for_reference=bool(candidate_reference_intent.follow_global_route_lane),
                    traffic_control_lane_lock_active=False,
                    final_goal_stop_active=False,
                    stop_target_state=None,
                    follow_target_state=None,
                    current_temp_reference_xy=(float(candidate_temp_destination[0]), float(candidate_temp_destination[1])),
                    current_temp_mode_value=float(candidate_temp_destination[5]) if len(candidate_temp_destination) >= 6 else 0.0,
                    current_temp_road_id=int(candidate_temp_destination[6]) if len(candidate_temp_destination) >= 7 else None,
                    current_temp_entered_intersection=bool(float(candidate_temp_destination[7]) > 0.5) if len(candidate_temp_destination) >= 8 else False,
                    active_reference_maneuver=str(planner_input_frame.planning.route.next_macro_maneuver),
                    current_temp_mode_str=str(planner_mode),
                    lane_reference_speed_mps=max(1.0, ego_speed_mps, abs(candidate_speed_ref_mps)),
                    lane_reference_step_distance_m=max(0.5, float(self.mpc.dt_s) * max(1.0, ego_speed_mps, abs(candidate_speed_ref_mps))),
                    mpc_horizon_steps=int(self.mpc.horizon_steps),
                    mpc_dt_s=float(self.mpc.dt_s),
                    temporary_destination_state=candidate_temp_destination,
                    lane_reference_freeze_count=int(self._lane_reference_freeze_count),
                    sim_time_s=sim_time_s,
                    stop_release_temp_smooth_until_sim_time_s=float(self._stop_release_temp_smooth_until_sim_time_s),
                )

                candidate_reference_output = generate_mpc_reference(candidate_reference_context)
                destination_state = list(candidate_reference_output.temporary_destination_state or candidate_temp_destination)
                reference = [dict(sample) for sample in list(candidate_reference_output.local_lane_center_reference or [])]
                candidate_reference_debug = dict(candidate_reference_output.mpc_reference_result.trace.as_trace_fields())
                candidate_reference_debug["fallback_reason"] = str(candidate_reference_output.last_reference_fallback_reason)

            contract_result = self._validate_candidate_reference_contract(
                decision=candidate_decision,
                lc_state=candidate_lc_state,
                current_lane_id=current_lane_id,
                speed_ref_mps=candidate_speed_ref_mps,
                stop_goal_active=candidate_stop_goal_active,
                current_state=current_state,
                destination_state=destination_state,
                lane_center_reference=reference,
            )

            candidate_result = CandidateReferenceResult(
                intent=intent,
                destination_state=list(destination_state),
                lane_center_reference=[dict(sample) for sample in reference],
                reference_debug=dict(candidate_reference_debug),
                contract_result=contract_result,
            )

            candidate_results.append(evaluate_candidate_reference(
                candidate=candidate_result,
                ego_state=current_state,
                object_snapshots=object_snapshots,
                prediction_trajectories=prediction_trajectories,
                current_lane_id=current_lane_id,
                min_object_distance_m=float(self.config.get("full_candidate_reference_min_object_distance_m", 2.0)),
            ))

        selected = select_best_candidate(candidate_results)

        if bool(self.config.get("strict_decision_ownership_enabled", True)) and not any(candidate.feasible for candidate in candidate_results):
            keep_lane_candidate = next((candidate for candidate in candidate_results if candidate.intent.decision == "lane_follow" and int(candidate.intent.target_lane_id) == current_lane_id), selected)
            stop_or_turn = baseline_decision in {"stop_at_intersection", "stop_sign", "emergency_brake", "intersection_turn_left", "intersection_turn_right"}
            fallback_decision = "emergency_brake" if stop_or_turn else "lane_follow"
            fallback_speed_mps = 0.0 if stop_or_turn else min(0.8, max(0.0, float(baseline_speed_ref_mps)))
            fallback_reference = [dict(sample) for sample in list(keep_lane_candidate.lane_center_reference or [])]

            for sample in fallback_reference:
                sample["v_ref_mps"] = fallback_speed_mps
                sample["speed_ref_mps"] = fallback_speed_mps
                sample["speed_mps"] = fallback_speed_mps

            fallback_destination = list(keep_lane_candidate.destination_state or [])
            if len(fallback_destination) < 5:
                fallback_destination = [float(current_state[0]), float(current_state[1]), fallback_speed_mps, float(current_state[3]), current_lane_id]
            else:
                fallback_destination[2] = fallback_speed_mps
                fallback_destination[4] = current_lane_id

            fallback_debug = dict(keep_lane_candidate.reference_debug or {})
            fallback_debug.update({
                "candidate_pipeline_selected": "explicit_fallback",
                "candidate_pipeline_selected_status": "explicit_fallback",
                "candidate_pipeline_selected_reason": "all_candidates_infeasible",
                "candidate_pipeline_count": len(candidate_results),
                "candidate_prediction_trajectory_count": len(prediction_trajectories),
                "candidate_pipeline_summary": summarize_candidate_results(candidate_results),
                "candidate_selected_decision": fallback_decision,
                "candidate_selected_lane_id": current_lane_id,
                "candidate_selected_cost": 100000.0,
                "reference_source": "custom_planner_explicit_fallback",
                "fallback_reason": "all_candidates_infeasible",
            })

            return fallback_decision, current_lane_id, fallback_speed_mps, fallback_reference, fallback_destination, fallback_debug

        selected_debug = dict(selected.reference_debug or {})
        selected_debug.update({
            "stage": selected_debug.get("reference_pipeline_stage", ""),
            "intent_mode": selected_debug.get("reference_pipeline_intent_mode", ""),
            "fallback_reason": selected_debug.get("fallback_reason", ""),
            "reference_source": str(selected_debug.get("reference_source", "candidate_reference_pipeline")),
            "candidate_pipeline_selected": str(selected.intent.name),
            "candidate_pipeline_selected_status": str(selected.feasibility_status),
            "candidate_pipeline_selected_reason": str(selected.feasibility_reason),
            "candidate_pipeline_count": len(candidate_results),
            "candidate_prediction_trajectory_count": len(prediction_trajectories),
            "candidate_pipeline_summary": summarize_candidate_results(candidate_results),
            "candidate_selected_decision": str(selected.intent.decision),
            "candidate_selected_lane_id": int(selected.intent.target_lane_id),
            "candidate_selected_cost": float(selected.total_cost),
        })

        return (
            str(selected.intent.decision),
            int(selected.intent.target_lane_id),
            float(selected.intent.target_speed_mps),
            [dict(sample) for sample in list(selected.lane_center_reference or [])],
            list(selected.destination_state or []),
            selected_debug,
        )


    @staticmethod
    def _candidate_lc_state(*, decision, baseline_decision, baseline_lc_state):
        """Return the behavior state that matches a candidate decision."""
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
        decision,
        lc_state,
        current_lane_id,
        speed_ref_mps,
        stop_goal_active,
        current_state,
        destination_state,
        lane_center_reference,
    ):
        """Check that a candidate reference has safe geometry and the correct lane, speed, and horizon."""
        normalized_decision = str(decision or "").strip().lower()
        normalized_fsm = str(lc_state or "").strip().upper()
        lane_change_active = normalized_decision in {"lane_change_left", "lane_change_right"} or normalized_fsm.startswith("EXECUTE_LANE_CHANGE")
        turn_active = normalized_decision in {"intersection_turn_left", "intersection_turn_right"} or normalized_fsm.startswith("INTERSECTION_TURN")
        stop_like = bool(stop_goal_active) or normalized_decision in {"stop_at_intersection", "stop_sign", "emergency_brake"}

        if normalized_decision == "emergency_brake":
            contract_mode = "emergency_stop"
        elif stop_like:
            contract_mode = "stop"
        elif lane_change_active:
            contract_mode = "lane_change"
        elif turn_active:
            contract_mode = "intersection_turn"
        else:
            contract_mode = "lane_follow"

        contract = contract_from_config(
            mode=contract_mode,
            expected_lane_id=int(current_lane_id),
            horizon_steps=int(self.mpc.horizon_steps),
            config=dict(self.config),
            default_speed_mps=max(float(self.config.get("target_speed_mps", 10.0)), float(speed_ref_mps), 0.1),
        )

        return validate_reference_contract(
            reference_samples=lane_center_reference,
            destination_state=destination_state,
            ego_state=current_state,
            contract=contract,
            check_destination_body_lateral=not bool(lane_change_active or turn_active),
        )

    def run_planning_cycle(self, adapter_output):
        """Run behavior planning, reference generation, and MPC for one PlannerInputFrame."""
        planner_input_frame = adapter_output.frame
        sim_time_s = float(planner_input_frame.planning.sim_time_s)
        current_state = list(adapter_output.current_state)

        destination_state, lane_center_reference, behavior_command, reference_debug = self._plan_behavior_and_reference(adapter_output)

        decision = str(behavior_command.get("decision", "lane_follow"))
        target_lane_id = int(behavior_command.get("target_lane_id", adapter_output.current_lane_id) or adapter_output.current_lane_id)
        target_speed_mps = float(behavior_command.get("target_speed_mps", self.config.get("target_speed_mps", 10.0)))
        mpc_stop_goal_active = bool(behavior_command.get("stop_goal_active", False)) or decision in {"stop_at_intersection", "stop_sign", "emergency_brake"}

        if mpc_stop_goal_active and len(destination_state) >= 3:
            destination_state = list(destination_state)
            destination_state[2] = 0.0

        object_snapshots = self._limit_obstacles_for_mpc(planner_input_frame=planner_input_frame, current_state=current_state)
        force_replan = mpc_stop_goal_active or decision in {"stop_at_intersection", "stop_sign", "emergency_brake", "intersection_turn_left", "intersection_turn_right"}
        mpc_replan_executed = False
        mpc_status = str(getattr(self.mpc, "_last_status", "not_solved"))
        fallback_reason = ""
        planned_trajectory = list(self._last_planned_trajectory)

        try:
            mpc_replan_executed = bool(self.control_buffer.should_replan(sim_time_s=sim_time_s, force_replan=force_replan))

            if mpc_replan_executed:
                planned_trajectory = self.mpc.plan_trajectory(
                    current_state=current_state,
                    destination_state=destination_state,
                    object_snapshots=object_snapshots,
                    current_acceleration_mps2=float(self._last_accel_mps2),
                    current_steering_rad=float(self._last_steer_rad),
                    lane_center_reference_samples=lane_center_reference,
                    stop_goal_active=mpc_stop_goal_active,
                )

                mpc_status = str(getattr(self.mpc, "_last_status", "")).strip().lower()
                if mpc_status and mpc_status not in {"solved", "solved inaccurate"}:
                    raise RuntimeError("MPC status={}".format(mpc_status))

                control_solution = getattr(self.mpc, "_last_u_solution", None)
                if control_solution is None or len(control_solution) == 0:
                    raise RuntimeError("MPC did not expose a control solution.")

                self.control_buffer.update_from_solution(u_solution=control_solution, plan_time_s=sim_time_s, dt_s=float(self.mpc.dt_s))
                acceleration_mps2 = float(control_solution[0, 0])
                steering_rad = float(control_solution[0, 1])
                self._last_planned_trajectory = [list(state) for state in planned_trajectory]
            else:
                buffered_control = self.control_buffer.sample(sim_time_s=sim_time_s)
                if buffered_control is None:
                    raise RuntimeError("MPC control buffer is empty.")

                acceleration_mps2, steering_rad, _ = buffered_control
                mpc_status = "buffer_reuse"

        except Exception as error:
            fallback_reason = str(error)
            acceleration_mps2, steering_rad = self._fallback_numeric_control(current_state=current_state, destination_state=destination_state, stop_goal_active=mpc_stop_goal_active)
            planned_trajectory = []
            self.control_buffer.reset(reason="mpc_failure")
            mpc_status = "fallback"

        self._last_accel_mps2 = float(acceleration_mps2)
        self._last_steer_rad = float(steering_rad)

        mpc_feedback_reason = self.mpc_feedback.record_result(
            decision=decision,
            target_lane_id=target_lane_id,
            status=mpc_status,
            reason=fallback_reason,
            timestamp_s=sim_time_s,
            success=not bool(fallback_reason),
        )

        behavior_command["reroute_requested"] = decision == "reroute"
        reference_debug.update({
            "mpc_status": mpc_status,
            "mpc_replan_executed": mpc_replan_executed,
            "mpc_fallback_reason": fallback_reason,
            "mpc_feedback_record_reason": mpc_feedback_reason,
            "mpc_solve_time_ms": float(getattr(self.mpc, "_last_solve_time_ms", 0.0)),
            "mpc_object_count": len(object_snapshots),
            "mpc_trajectory_point_count": len(planned_trajectory),
            "control_buffer_reason": self.control_buffer.last_reason,
            "accel_cmd_mps2": float(acceleration_mps2),
            "steer_cmd_rad": float(steering_rad),
        })

        self.last_destination_state = list(destination_state)
        self.last_reference_debug = dict(reference_debug)
        self.last_behavior_command = dict(behavior_command)

        return PlannerOutput(
            control=None,
            behavior_command=PlannerBehaviorCommand.from_debug(behavior_debug=behavior_command, target_speed_mps=target_speed_mps),
            reference_trajectory=[dict(sample) for sample in lane_center_reference],
            planned_trajectory=[list(state) for state in planned_trajectory],
            predictions=dict(planner_input_frame.prediction.obstacle_future_trajectories),
            acceleration_mps2=float(acceleration_mps2),
            steering_rad=float(steering_rad),
            diagnostics=PlannerDiagnostics(fields=dict(reference_debug)),
        )


    def _limit_obstacles_for_mpc(self, *, planner_input_frame, current_state):
        """Keep the closest planning objects so MPC receives the same limited obstacle list used by OpenCDA."""
        object_snapshots = [dict(item) for item in list(planner_input_frame.perception.planning_objects or []) if isinstance(item, Mapping)]
        max_mpc_obstacles = max(0, int(self.config.get("max_mpc_obstacles", 4)))

        if max_mpc_obstacles > 0 and len(object_snapshots) > max_mpc_obstacles:
            ego_x_m = float(current_state[0])
            ego_y_m = float(current_state[1])
            object_snapshots.sort(key=lambda item: (float(item.get("x", 0.0)) - ego_x_m) ** 2 + (float(item.get("y", 0.0)) - ego_y_m) ** 2)
            object_snapshots = object_snapshots[:max_mpc_obstacles]

        return object_snapshots
    
    
    def _fallback_numeric_control(self, *, current_state, destination_state, stop_goal_active):
        """Return safe numeric acceleration and steering if MPC cannot provide a valid solution."""
        if stop_goal_active:
            return float(self.mpc.constraints.min_acceleration_mps2), 0.0

        dx = float(destination_state[0]) - float(current_state[0])
        dy = float(destination_state[1]) - float(current_state[1])
        target_heading_rad = math.atan2(dy, dx)
        heading_error_rad = self._wrap_angle(target_heading_rad - float(current_state[3]))
        max_steer_rad = max(1.0e-6, float(self.mpc.constraints.max_steer_rad))
        steering_rad = min(max_steer_rad, max(-max_steer_rad, 0.7 * heading_error_rad))
        speed_error_mps = float(self.config.get("target_speed_mps", 10.0)) - float(current_state[2])
        acceleration_mps2 = min(float(self.mpc.constraints.max_acceleration_mps2), max(float(self.mpc.constraints.min_acceleration_mps2), 0.6 * speed_error_mps))
        return float(acceleration_mps2), float(steering_rad)


    def _apply_mpc_cost_profile(self, *, behavior, planner_lc_state, planner_mode, next_macro_maneuver, sim_time_s):
        """Select the MPC cost profile associated with the current behavior and apply it with the same hold logic as OpenCDA."""
        requested_profile = _mpc_cost_profile_for_behavior(behavior=behavior, planner_lc_state=planner_lc_state, planner_mode=planner_mode, next_macro_maneuver=next_macro_maneuver)
        self.active_mpc_cost_profile, self.mpc_cost_profile_active_since_s, self.mpc_cost_profile_switch_reason = _select_mpc_cost_profile_with_hysteresis(requested_profile=requested_profile, active_profile=self.active_mpc_cost_profile, sim_time_s=sim_time_s, active_since_s=self.mpc_cost_profile_active_since_s, min_hold_s=float(self.behavior_runtime_cfg.get("mpc_cost_profile_min_hold_s", 1.5)))
        self.requested_mpc_cost_profile = requested_profile

        if hasattr(self.mpc, "apply_mode_cost_profile"):
            self.active_mpc_cost_profile = str(self.mpc.apply_mode_cost_profile(self.active_mpc_cost_profile))


    def _perform_reroute(self, *, command, planner_input_frame, adapter_output, current_route_points) -> str:
        """Use ROS lane-closure messages to block AD-map lanes and replace the route only after successful rerouting."""
        final_goal = planner_input_frame.planning.targets.final_goal
        if final_goal is None or len(final_goal) < 2:
            saved_goal = self.route_manager.goal_point
            if saved_goal is None:
                return "Final destination is unavailable."
            goal_position = dict(saved_goal)
        else:
            goal_position = {
                "x": float(final_goal[0]),
                "y": float(final_goal[1]),
                "z": float(final_goal[2]) if len(final_goal) >= 3 else 0.0,
            }

        ego_position = {
            "x": float(planner_input_frame.planning.ego.x_m),
            "y": float(planner_input_frame.planning.ego.y_m),
            "z": float(adapter_output.ego_pose.get("z", 0.0)),
        }

        reroute_result = reroute_from_lane_closure_messages(
            messages=list(command.get("reroute_messages", []) or []),
            global_planner=self.global_planner,
            ego_position=ego_position,
            goal_position=goal_position,
            current_route_points=current_route_points,
        )

        route_summary = reroute_result.get("route_summary")
        if route_summary is None:
            return str(reroute_result.get("debug_reason", "Rerouting failed."))

        if not self.route_manager.accept_route_summary(route_summary):
            return "The new route was not valid."

        self.behavior_planner.acknowledge_reroute_success(reroute_result.get("handled_message_ids", []))
        return ""

    def _latched_stop_target_for_signal(
        self,
        *,
        traffic_state: str,
        stop_target: Mapping[str, object] | None,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        current_lane_id: int,
    ):
        """Keep one stable stop position while the same red or yellow signal remains active."""
        state = str(traffic_state or "unknown").strip().lower()

        if state not in {"red", "yellow"}:
            if self._latched_stop_target is not None:
                self._latched_stop_target = None
                self._latched_stop_state = state
                return None, "stop_target_latch_release"

            self._latched_stop_state = state
            return None, ""

        if self._latched_stop_target is not None and self._latched_stop_state in {"red", "yellow"}:
            return dict(self._latched_stop_target), "stop_target_latch_reuse"

        latched = None
        if isinstance(stop_target, Mapping):
            try:
                x_value = stop_target.get("x_m", stop_target.get("x"))
                y_value = stop_target.get("y_m", stop_target.get("y"))

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
            x_m = ego_x_m + distance_m * math.cos(ego_heading_rad)
            y_m = ego_y_m + distance_m * math.sin(ego_heading_rad)
            latched = {
                "x_m": x_m,
                "y_m": y_m,
                "x": x_m,
                "y": y_m,
                "heading_rad": ego_heading_rad,
                "lane_id": current_lane_id,
                "distance_m": distance_m,
                "source": "latched_virtual_stop_target",
            }

        self._latched_stop_target = dict(latched)
        self._latched_stop_state = state
        return dict(latched), "stop_target_latch_create"

    def _stop_target_forward_m(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        stop_target: Mapping[str, object] | None,
    ):
        """Measure how far the traffic stop target is in front of the ego vehicle."""
        if isinstance(stop_target, Mapping):
            try:
                has_x = "x_m" in stop_target or "x" in stop_target
                has_y = "y_m" in stop_target or "y" in stop_target

                if not (has_x and has_y) and "distance_m" in stop_target:
                    return max(0.0, float(stop_target.get("distance_m", 0.0))), True

                target_x_m = float(stop_target.get("x_m", stop_target.get("x", ego_x_m)))
                target_y_m = float(stop_target.get("y_m", stop_target.get("y", ego_y_m)))
                forward_m, _ = self._body_frame_xy(
                    origin_x_m=ego_x_m,
                    origin_y_m=ego_y_m,
                    heading_rad=ego_heading_rad,
                    target_x_m=target_x_m,
                    target_y_m=target_y_m,
                )
                return max(0.0, forward_m), True
            except Exception:
                pass

        return max(2.0, float(self.config.get("full_stop_guard_destination_forward_m", 6.0))), False

    @staticmethod
    def _route_option_turn_decision(*, current_road_option: str, next_macro_maneuver: str) -> str:
        """Convert the custom global route's next turn into a behavior decision."""
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
    ):
        """Keep a route turn active while the ego passes through the intersection."""
        if not bool(self.config.get("full_intersection_turn_latch_enabled", True)):
            self._turn_latch_decision = ""
            self._turn_latch_until_sim_time_s = -float("inf")
            return decision, lc_state, speed_ref_mps, ""

        explicit_turn = self._route_option_turn_decision(
            current_road_option=current_road_option,
            next_macro_maneuver=next_macro_maneuver,
        )
        normalized_decision = str(decision or "").strip().lower()

        if explicit_turn:
            self._turn_latch_decision = explicit_turn
            self._turn_latch_until_sim_time_s = sim_time_s + float(self.config.get("full_intersection_turn_latch_hold_s", 6.0))
            latched_lc_state = "INTERSECTION_TURN_LEFT" if explicit_turn.endswith("_left") else "INTERSECTION_TURN_RIGHT"
            capped_speed = min(speed_ref_mps, float(self.config.get("full_intersection_turn_speed_cap_mps", 2.2)))
            return explicit_turn, latched_lc_state, capped_speed, "turn_latch_set:" + explicit_turn

        latch_active = (
            bool(self._turn_latch_decision)
            and sim_time_s <= self._turn_latch_until_sim_time_s
            and (ego_in_junction or normalized_decision.startswith("intersection_turn"))
        )

        if latch_active:
            latched_lc_state = "INTERSECTION_TURN_LEFT" if self._turn_latch_decision.endswith("_left") else "INTERSECTION_TURN_RIGHT"
            capped_speed = min(speed_ref_mps, float(self.config.get("full_intersection_turn_speed_cap_mps", 2.2)))
            return self._turn_latch_decision, latched_lc_state, capped_speed, "turn_latch_hold:" + self._turn_latch_decision

        if self._turn_latch_decision:
            self._turn_latch_decision = ""
            self._turn_latch_until_sim_time_s = -float("inf")
            return decision, lc_state, speed_ref_mps, "turn_latch_release"

        return decision, lc_state, speed_ref_mps, ""

    def _route_lookahead_turn_decision(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        route_points: Sequence[Sequence[float]],
    ) -> str:
        """Look ahead on the custom global route and detect whether it curves into a left or right turn."""
        route_xy = [(float(point[0]), float(point[1])) for point in list(route_points or []) if len(point) >= 2]
        if len(route_xy) < 4:
            return ""

        route_progress = [0.0]
        for first, second in zip(route_xy[:-1], route_xy[1:]):
            route_progress.append(route_progress[-1] + math.hypot(second[0] - first[0], second[1] - first[1]))

        base_s = self._project_point_to_polyline_s(route_xy=route_xy, route_progress=route_progress, point_xy=(ego_x_m, ego_y_m))
        lookahead_m = float(self.config.get("full_intersection_turn_prepare_lookahead_m", 28.0))
        min_prepare_m = float(self.config.get("full_intersection_turn_prepare_min_m", 6.0))
        near_heading = self._route_heading_at_s(
            route_xy=route_xy,
            route_progress=route_progress,
            target_s=base_s + min_prepare_m,
            fallback_heading_rad=ego_heading_rad,
        )
        far_heading = self._route_heading_at_s(
            route_xy=route_xy,
            route_progress=route_progress,
            target_s=base_s + lookahead_m,
            fallback_heading_rad=near_heading,
        )
        heading_change = self._wrap_angle(far_heading - near_heading)
        threshold = float(self.config.get("full_intersection_turn_prepare_heading_delta_rad", 0.45))

        if heading_change > threshold:
            return "intersection_turn_left"

        if heading_change < -threshold:
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
        """Return the direction of the custom global route at a requested distance along the route."""
        if len(route_xy) < 2:
            return fallback_heading_rad

        clamped_s = min(max(target_s, route_progress[0]), route_progress[-1])
        for index in range(len(route_progress) - 1):
            if route_progress[index + 1] + 1.0e-6 < clamped_s:
                continue

            first = route_xy[index]
            second = route_xy[index + 1]
            dx_m = second[0] - first[0]
            dy_m = second[1] - first[1]

            if math.hypot(dx_m, dy_m) > 1.0e-6:
                return math.atan2(dy_m, dx_m)

        first = route_xy[-2]
        second = route_xy[-1]
        dx_m = second[0] - first[0]
        dy_m = second[1] - first[1]

        if math.hypot(dx_m, dy_m) > 1.0e-6:
            return math.atan2(dy_m, dx_m)

        return fallback_heading_rad

    @staticmethod
    def _project_point_to_polyline_s(
        *,
        route_xy: Sequence[tuple[float, float]],
        route_progress: Sequence[float],
        point_xy: tuple[float, float],
    ) -> float:
        """Find the ego vehicle's distance along the custom global route."""
        best_distance_m = float("inf")
        best_s = float(route_progress[0]) if route_progress else 0.0
        point_x_m = float(point_xy[0])
        point_y_m = float(point_xy[1])

        for index, (first, second) in enumerate(zip(route_xy[:-1], route_xy[1:])):
            first_x_m = float(first[0])
            first_y_m = float(first[1])
            second_x_m = float(second[0])
            second_y_m = float(second[1])
            dx_m = second_x_m - first_x_m
            dy_m = second_y_m - first_y_m
            segment_length_sq = dx_m * dx_m + dy_m * dy_m

            if segment_length_sq <= 1.0e-9:
                continue

            ratio = ((point_x_m - first_x_m) * dx_m + (point_y_m - first_y_m) * dy_m) / segment_length_sq
            ratio = min(1.0, max(0.0, ratio))
            projected_x_m = first_x_m + ratio * dx_m
            projected_y_m = first_y_m + ratio * dy_m
            distance_m = math.hypot(point_x_m - projected_x_m, point_y_m - projected_y_m)

            if distance_m < best_distance_m:
                best_distance_m = distance_m
                best_s = float(route_progress[index]) + ratio * math.sqrt(segment_length_sq)

        return best_s

    @staticmethod
    def _body_frame_xy(
        *,
        origin_x_m: float,
        origin_y_m: float,
        heading_rad: float,
        target_x_m: float,
        target_y_m: float,
    ):
        """Convert a world position into forward and sideways distances measured from the ego vehicle."""
        dx_m = target_x_m - origin_x_m
        dy_m = target_y_m - origin_y_m
        cos_heading = math.cos(heading_rad)
        sin_heading = math.sin(heading_rad)
        forward_m = dx_m * cos_heading + dy_m * sin_heading
        lateral_m = -dx_m * sin_heading + dy_m * cos_heading
        return forward_m, lateral_m

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        """Keep an angle between minus pi and plus pi."""
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
    
    
    
def _mpc_cost_profile_for_behavior(*, behavior, planner_lc_state, planner_mode, next_macro_maneuver):
    """Return the MPC cost-profile name used for the current behavior."""
    raw_behavior = str(behavior or "").strip().lower()
    normalized_behavior = str(normalize_behavior_decision(behavior))
    normalized_lc_state = str(planner_lc_state or "").strip().upper()
    normalized_mode = str(planner_mode or "").strip().upper()
    normalized_maneuver = str(next_macro_maneuver or "straight").strip().lower()

    if is_fixed_stop_decision(normalized_behavior):
        return "stop"
    if is_emergency_brake_decision(normalized_behavior):
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


def _select_mpc_cost_profile_with_hysteresis(*, requested_profile, active_profile, sim_time_s, active_since_s, min_hold_s):
    """Prevent the MPC cost profile from changing too frequently."""
    requested = str(requested_profile or "lane_follow").strip() or "lane_follow"
    active = str(active_profile or "lane_follow").strip() or "lane_follow"
    elapsed_s = max(0.0, float(sim_time_s) - float(active_since_s))
    minimum_hold_s = max(0.0, float(min_hold_s))

    if requested == active:
        return active, float(active_since_s), "same_profile"
    if requested in {"stop", "recovery"}:
        return requested, float(sim_time_s), "safety_preempt"
    if active in {"stop", "recovery"} and elapsed_s < minimum_hold_s:
        return active, float(active_since_s), "hold_safety_profile"
    if elapsed_s < minimum_hold_s:
        return active, float(active_since_s), "min_hold"
    return requested, float(sim_time_s), "switch"