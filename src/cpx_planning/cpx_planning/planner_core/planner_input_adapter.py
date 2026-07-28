"""OpenCDA-to-CP-X planner input adapter.

This module is the explicit boundary between OpenCDA's runtime data providers
and the CP-X planning pipeline.  OpenCDA still owns scenario execution,
VehicleManager.update_info(), localization, perception, V2X/CP publication,
and stable global-route generation.  The adapter converts those signals into
the PlannerInputFrame consumed by CP-X behavior, reference generation, and MPC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from cpx_planning.utility.planning_context import (
    CPMessageContext,
    EgoPlanningState,
    MapLaneContext,
    PerceptionContext,
    PlannerInputFrame,
    PlanningContext,
    PredictionContext,
    RouteContext,
    TargetContext,
    TrafficControlContext,
)
from cpx_planning.utility.global_planner import  (
    canonical_lane_id_for_waypoint,
    canonical_lane_waypoints,
)


@dataclass(frozen=True)
class PlannerInputAdapterOutput:
    """Planner-facing input and reusable per-tick adapter products."""

    frame: PlannerInputFrame
    ego_pose: Dict[str, float]
    current_state: List[float]
    current_lane_id: int
    lane_ids: List[int]
    ego_waypoint: Any
    ego_snapshot: Dict[str, float]
    lane_assignments: Dict[str, int]
    lane_safety_scores: Dict[int, float]
    front_distance_by_lane: Dict[int, float]
    route_points: List[List[float]]
    route_summary: Dict[str, object]
    route_optimal_lane_id: int
    route_reference_allowed: bool
    route_reference_gate_reason: str
    selected_traffic_control: Optional[Mapping[str, object]]
    signal_context: Dict[str, object]
    stop_target: Optional[Mapping[str, object]]
    source_quality: Dict[str, object]


class OpenCDAPlanningAdapter:
    """Build PlannerInputFrame from a native OpenCDA VehicleManager snapshot."""

    def __init__(self, bridge: Any):
        self.bridge = bridge

    def build(
        self,
        *,
        ego_location: Any,
        ego_yaw_rad: float,
        ego_speed_mps: float,
        object_snapshots: Sequence[Mapping[str, Any]],
        cp_payload: Optional[Mapping[str, Any]],
    ) -> PlannerInputAdapterOutput:
        bridge = self.bridge
        sim_time_s = float(bridge._sim_time_s())
        ego_pose = {
            "x": float(ego_location.x),
            "y": float(ego_location.y),
            "z": float(ego_location.z),
            "heading_rad": float(ego_yaw_rad),
        }
        current_state = [
            float(ego_location.x),
            float(ego_location.y),
            float(ego_speed_mps),
            float(ego_yaw_rad),
        ]
        ego_waypoint = bridge.reference_map.get_waypoint(ego_pose)
        current_lane_id = int(canonical_lane_id_for_waypoint(ego_waypoint))
        if current_lane_id == 0:
            current_lane_id = 1
        lane_ids = [
            int(canonical_lane_id_for_waypoint(wp))
            for wp in list(canonical_lane_waypoints(ego_waypoint) or [])
            if int(canonical_lane_id_for_waypoint(wp)) != 0
        ]
        if not lane_ids:
            lane_ids = [int(current_lane_id)]

        # Keep custom global-route progress synchronized with the current ego position.
        # This ensures that reference generation starts from the correct route segment.
        bridge.route_manager.sync_route_progress(
            ego_x_m=float(ego_location.x),
            ego_y_m=float(ego_location.y),
            ego_heading_rad=float(ego_yaw_rad),
        )

        lane_assignments = bridge._assign_obstacles_to_lanes(object_snapshots)
        ego_snapshot = {
            "x": float(ego_location.x),
            "y": float(ego_location.y),
            "v": float(ego_speed_mps),
            "psi": float(ego_yaw_rad),
        }
        lane_safety_scores = bridge.lane_safety_scorer.compute_lane_scores(
            ego_snapshot=ego_snapshot,
            obstacle_snapshots=object_snapshots,
            lane_assignments=lane_assignments,
            ego_lane_id=int(current_lane_id),
            available_lane_ids=lane_ids,
            timestamp_s=float(sim_time_s),
        )
        bridge.lane_safety_scorer.cleanup_stale_obstacles(set(lane_assignments.keys()))
        front_dist_by_lane = bridge._nearest_front_distance_by_lane(
            ego_snapshot=ego_snapshot,
            obstacle_snapshots=object_snapshots,
            lane_assignments=lane_assignments,
            available_lane_ids=lane_ids,
        )

        route_points = bridge._active_global_route_points()
        route_summary = bridge._planning_module_global_route_summary(
            ego_location=ego_location,
            ego_heading_rad=float(ego_yaw_rad),
            fallback_lane_id=int(current_lane_id),
        )
        route_optimal_lane_id = int(
            route_summary.get("optimal_lane_id", current_lane_id) or current_lane_id
        )
        route_reference_allowed = (
            bool(bridge.use_opencda_global_route)
            and bool(bridge.opencda_global_route_reference_allowed)
            and bool(route_summary.get("route_found", False))
            and len(route_points) >= 2
        )
        route_reference_gate_reason = (
            "opencda_global_route_enabled"
            if bool(route_reference_allowed)
            else str(route_summary.get("debug_reason", "opencda_global_route_unavailable"))
        )

        cp_payload = dict(cp_payload or bridge._load_cp_message_payload())
        cp_timestamp_s = _optional_float(cp_payload.get("timestamp_s", None))
        cp_age_s = (
            ""
            if cp_timestamp_s is None
            else max(0.0, float(sim_time_s) - float(cp_timestamp_s))
        )
        cp_valid = True
        if cp_age_s != "":
            cp_valid = float(cp_age_s) <= float(
                bridge.config.get("max_cp_message_age_s", 1.0)
            )
        traffic_controls = list(cp_payload.get("control", []) or [])
        lane_closures = list(
            cp_payload.get("lane_closures", cp_payload.get("lane_events", [])) or []
        )
        cp_obstacles = list(cp_payload.get("obstacles", []) or [])
        selected_control = bridge._select_relevant_traffic_control(
            traffic_controls=traffic_controls,
            ego_location=ego_location,
            ego_heading_rad=float(ego_yaw_rad),
            current_lane_id=int(current_lane_id),
            current_road_id=int(getattr(ego_waypoint, "road_id", 0) or 0),
            sim_time_s=float(sim_time_s),
        )
        signal_context, stop_target = bridge._traffic_context_from_cp_control(
            selected_control=selected_control,
            ego_location=ego_location,
        )
        if bool(bridge.config.get("ignore_traffic_control", False)):
            signal_context = {
                "signal_state": "unknown",
                "signal_source": "disabled_by_planner_config",
                "traffic_control_from_cp": False,
            }
            stop_target = None
        traffic_control_context = TrafficControlContext.from_signal_context(
            signal_context=signal_context,
            stop_target=stop_target,
        )
        tracked_obstacles = bridge.tracker.update(
            obstacle_snapshots=object_snapshots,
            timestamp_s=float(sim_time_s),
            signal_context=signal_context,
            stop_target=stop_target,
        )
        lane_assignments = bridge._assign_obstacles_to_lanes(tracked_obstacles)
        lane_safety_scores = bridge.lane_safety_scorer.compute_lane_scores(
            ego_snapshot=ego_snapshot,
            obstacle_snapshots=tracked_obstacles,
            lane_assignments=lane_assignments,
            ego_lane_id=int(current_lane_id),
            available_lane_ids=lane_ids,
            timestamp_s=float(sim_time_s),
        )
        bridge.lane_safety_scorer.cleanup_stale_obstacles(set(lane_assignments.keys()))
        front_dist_by_lane = bridge._nearest_front_distance_by_lane(
            ego_snapshot=ego_snapshot,
            obstacle_snapshots=tracked_obstacles,
            lane_assignments=lane_assignments,
            available_lane_ids=lane_ids,
        )
        prediction_frame = bridge.tracker.predict(
            ego_snapshot=ego_snapshot,
            lane_assignments=lane_assignments,
            available_lane_ids=lane_ids,
            horizon_s=float(bridge.mpc.horizon_s),
            dt_s=float(bridge.mpc.dt_s),
            min_front_gap_m=float(bridge.min_front_gap_m),
            min_rear_gap_m=float(bridge.min_front_gap_m),
            min_ttc_s=float(bridge.config.get("prediction_min_ttc_s", 2.0)),
        )
        route_context = RouteContext(
            optimal_lane_id=int(route_optimal_lane_id),
            next_macro_maneuver=str(
                route_summary.get("next_macro_maneuver", "Continue Straight")
            ),
            current_road_option=str(route_summary.get("current_road_option", "")),
            remaining_distance_m=float(route_summary.get("remaining_distance_m", 0.0) or 0.0),
            remaining_points_count=len(route_points),
            route_found=bool(route_summary.get("route_found", False)),
        )
        frame = PlannerInputFrame(
            planning=PlanningContext(
                sim_time_s=float(sim_time_s),
                ego=EgoPlanningState(
                    x_m=float(ego_location.x),
                    y_m=float(ego_location.y),
                    speed_mps=float(ego_speed_mps),
                    heading_rad=float(ego_yaw_rad),
                    lane_id=int(current_lane_id),
                    road_id=int(getattr(ego_waypoint, "road_id", 0) or 0),
                    section_id=int(getattr(ego_waypoint, "section_id", 0) or 0),
                    in_junction=bool(getattr(ego_waypoint, "is_intersection", False)),
                ),
                route=route_context,
                traffic_control=traffic_control_context,
                targets=TargetContext(stop_target=traffic_control_context.stop_target),
                global_route_reference_allowed=bool(route_reference_allowed),
                global_route_reference_gate_reason=str(route_reference_gate_reason),
            ),
            map_lane=MapLaneContext(
                lane_id=int(current_lane_id),
                road_id=int(getattr(ego_waypoint, "road_id", 0) or 0),
                section_id=int(getattr(ego_waypoint, "section_id", 0) or 0),
                lane_count=len(lane_ids),
                allowed_lane_ids=list(lane_ids),
                in_junction=bool(getattr(ego_waypoint, "is_intersection", False)),
                route_lane_id=int(route_optimal_lane_id),
                route_maneuver=str(route_context.next_macro_maneuver),
            ),
            perception=PerceptionContext(
                dynamic_objects=[dict(obj) for obj in tracked_obstacles],
                planning_objects=[dict(obj) for obj in tracked_obstacles],
                source="native_opencda",
            ),
            prediction=PredictionContext(
                lane_assignments=dict(lane_assignments),
                lane_prediction_risks=dict(prediction_frame.lane_prediction_risks),
                obstacle_future_trajectories=dict(
                    prediction_frame.obstacle_future_trajectories
                ),
                model="constant_acceleration",
                horizon_s=float(bridge.mpc.horizon_s),
                dt_s=float(bridge.mpc.dt_s),
            ),
            cp_messages=CPMessageContext(
                message_path=str(bridge.cp_message_path),
                traffic_controls=traffic_controls,
                selected_traffic_control=selected_control,
                lane_closures=lane_closures,
                obstacles=cp_obstacles,
            ),
        )
        return PlannerInputAdapterOutput(
            frame=frame,
            ego_pose=ego_pose,
            current_state=current_state,
            current_lane_id=int(current_lane_id),
            lane_ids=list(lane_ids),
            ego_waypoint=ego_waypoint,
            ego_snapshot=ego_snapshot,
            lane_assignments=dict(lane_assignments),
            lane_safety_scores=dict(lane_safety_scores),
            front_distance_by_lane=dict(front_dist_by_lane),
            route_points=list(route_points),
            route_summary=dict(route_summary),
            route_optimal_lane_id=int(route_optimal_lane_id),
            route_reference_allowed=bool(route_reference_allowed),
            route_reference_gate_reason=str(route_reference_gate_reason),
            selected_traffic_control=selected_control,
            signal_context=dict(signal_context or {}),
            stop_target=stop_target,
            source_quality={
                "planner_input_frame_timestamp_s": float(sim_time_s),
                "cp_message_timestamp_s": "" if cp_timestamp_s is None else float(cp_timestamp_s),
                "cp_message_age_s": cp_age_s,
                "cp_message_valid": bool(cp_valid),
                **dict(bridge.tracker.diagnostics),
            },
        )


def _optional_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


PlannerInputAdapter = OpenCDAPlanningAdapter
