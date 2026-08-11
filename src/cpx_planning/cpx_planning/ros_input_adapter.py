"""Convert typed ROS messages into the existing CP-X planner input."""

from __future__ import annotations

import json
import math
from typing import Mapping, Sequence

from cpx_planning.component_interfaces import PlannerLocation, PlannerSafetyManager, ROSInputSnapshot
from cpx_planning.planner_core.planner_input_adapter import PlannerInputAdapterOutput
from cpx_planning.utility.global_planner import (
    canonical_lane_id_for_waypoint,
    canonical_lane_waypoints,
)
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


OBJECT_TYPES = {
    0: "unknown",
    1: "car",
    2: "truck",
    3: "bus",
    4: "trailer",
    5: "motorcycle",
    6: "bicycle",
    7: "pedestrian",
}

LANE_EVENT_TYPES = {
    0: "unknown",
    1: "lane_closure",
    2: "road_hazard",
    3: "work_zone",
}


class ROSInputAdapter:
    """Keep the latest ROS inputs and build the same PlannerInputFrame used by the existing planner."""

    def __init__(
        self,
        *,
        bridge,
    ):
        """Store the planner components, prediction settings, and latest-message placeholders."""
        self.bridge = bridge
        self.map_planner = bridge.map_planner
        self.reference_map = bridge.reference_map
        self.route_manager = bridge.route_manager
        self.tracker = bridge.tracker
        self.lane_safety_scorer = bridge.lane_safety_scorer
        self.mpc = bridge.mpc
        self.min_front_gap_m = float(bridge.min_front_gap_m)
        self.min_ttc_s = float(bridge.config.get("prediction_min_ttc_s", 2.0))
        self.communication_range_m = max(0.0, float(bridge.config.get("communication_range_m", 80.0)))
        self.config = bridge.config
        self._lane_id_tracker = bridge._lane_id_tracker
        self._prediction_lane_step_resolved_count = 0
        self._prediction_lane_step_none_count = 0

        self._localization = None
        self._perception = None
        self._v2x = None
        self._traffic_lights = None
        self._cooperative = None
        self._cp_obstacles = None
        self._safety_status = None
        self._final_destination = None

        # The route only needs to be rebuilt when the destination changes.
        self._active_goal_signature = None

    def update_localization(self, message):
        """Save the newest ego localization message for the next planning frame."""
        self._check_frame(message)
        self._localization = message

    def update_perception(self, message):
        """Save the newest locally perceived-object message for the next planning frame."""
        self._check_frame(message)
        self._perception = message

    def update_v2x(self, message):
        """Save the newest V2X object message for the next planning frame."""
        self._check_frame(message)
        self._v2x = message

    def update_traffic_lights(self, message):
        """Save the newest perceived traffic-light message for the next planning frame."""
        self._check_frame(message)
        self._traffic_lights = message

    def update_cooperative_messages(self, message):
        """Save the newest cooperative lane-event message; this input is currently optional."""
        self._check_frame(message)
        self._cooperative = message

    def update_cp_obstacles(self, message):
        """Save the exact CP obstacle dictionaries produced by the new OpenCDA provider."""
        try:
            payload = json.loads(str(message.data or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid /cpx/cp_obstacles JSON payload.") from exc
        self._cp_obstacles = payload if isinstance(payload, Mapping) else {}

    def update_safety_status(self, message):
        """Save the current OpenCDA safety flags for the copied final safety filter."""
        try:
            payload = json.loads(str(message.data or "{}"))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid /cpx/safety_status JSON payload.") from exc
        self._safety_status = payload if isinstance(payload, Mapping) else {}

    def update_final_destination(self, message):
        """Save the latest mission destination used by the custom global planner."""
        self._check_frame(message)
        self._final_destination = message

    def ready(self):
        """Return True only when the required ROS inputs belong to the same CARLA cycle."""
        if self._localization is None or self._perception is None or self._v2x is None or self._traffic_lights is None or self._cp_obstacles is None or self._safety_status is None or self._final_destination is None:
            return False

        timestamps = [
            self._stamp_seconds(self._localization.header.stamp),
            self._stamp_seconds(self._perception.header.stamp),
            self._stamp_seconds(self._v2x.header.stamp),
            self._stamp_seconds(self._traffic_lights.header.stamp),
            self._stamp_seconds(self._final_destination.header.stamp),
            float(self._safety_status.get("timestamp_s", 0.0) or 0.0),
        ]

        return max(timestamps) - min(timestamps) <= 1.0e-6

    def latest_timestamp_s(self):
        """Return the timestamp of the latest localization input."""
        return 0.0 if self._localization is None else self._stamp_seconds(self._localization.header.stamp)

    def runtime_inputs(self):
        """Return the raw ROS values used at the start of the copied OpenCDA planning cycle."""
        if not self.ready():
            raise RuntimeError("ROS planner inputs are not ready.")
        snapshot = self._make_snapshot()
        self._update_route(ego_pose=snapshot.ego_pose, final_goal=snapshot.final_goal)
        cp_obstacles = [dict(item) for item in list(self._cp_obstacles.get("obstacles", []) or []) if isinstance(item, Mapping)]
        traffic_controls = [control for control in (self._traffic_light_to_control_message(traffic_light=traffic_light, ego_pose=snapshot.ego_pose, sim_time_s=float(snapshot.timestamp_s)) for traffic_light in snapshot.traffic_lights) if control is not None]
        cp_payload = {
            "schema_version": int(self._cp_obstacles.get("schema_version", 1) or 1),
            "timestamp_s": float(self._cp_obstacles.get("timestamp_s", snapshot.timestamp_s) or snapshot.timestamp_s),
            "obstacles": [dict(item) for item in cp_obstacles],
            "control": [dict(item) for item in traffic_controls],
            "lane_closures": [dict(item) for item in snapshot.lane_events],
        }
        return {
            "sim_time_s": float(snapshot.timestamp_s),
            "ego_pose": dict(snapshot.ego_pose),
            "ego_speed_mps": float(snapshot.ego_speed_mps),
            "local_object_snapshots": [dict(item) for item in snapshot.perception_objects],
            "cp_payload": cp_payload,
            "v2x_nearby_count": len(list(self._v2x.objects)),
            "safety_manager": PlannerSafetyManager(timestamp_s=float(snapshot.timestamp_s), status=dict(self._safety_status.get("status", {}) or {})),
        }

    def build(self, *, ego_location=None, ego_yaw_rad=None, ego_speed_mps=None, object_snapshots=None, cp_payload=None):
        """Build the same PlannerInputAdapterOutput as OpenCDA from the current ROS cycle."""
        if not self.ready():
            raise RuntimeError("ROS planner inputs are not ready.")

        snapshot = self._make_snapshot()
        if ego_location is None:
            runtime_inputs = self.runtime_inputs()
            ego_pose = dict(runtime_inputs["ego_pose"])
            ego_speed_mps = float(runtime_inputs["ego_speed_mps"])
            cp_payload = dict(runtime_inputs["cp_payload"])
            object_snapshots = self.bridge._fused_planning_object_snapshots(local_object_snapshots=runtime_inputs["local_object_snapshots"], cp_obstacles=list(cp_payload.get("obstacles", []) or []), ego_location=PlannerLocation(x=float(ego_pose["x"]), y=float(ego_pose["y"]), z=float(ego_pose.get("z", 0.0))), sim_time_s=float(snapshot.timestamp_s))
        else:
            ego_pose = {
                "x": float(ego_location.x),
                "y": float(ego_location.y),
                "z": float(getattr(ego_location, "z", 0.0)),
                "heading_rad": float(ego_yaw_rad),
            }
            ego_speed_mps = float(ego_speed_mps)
            cp_payload = dict(cp_payload or {})
            object_snapshots = [dict(item) for item in list(object_snapshots or [])]
        current_state = [
            float(ego_pose["x"]),
            float(ego_pose["y"]),
            float(ego_speed_mps),
            float(ego_pose["heading_rad"]),
        ]
        ego_snapshot = {
            "x": float(ego_pose["x"]),
            "y": float(ego_pose["y"]),
            "v": float(ego_speed_mps),
            "psi": float(ego_pose["heading_rad"]),
        }

        # This is the same map-match point used by OpenCDAPlanningAdapter; only the map backend is AD-map.
        ego_waypoint = self.reference_map.get_waypoint(ego_pose)
        if ego_waypoint is None:
            raise RuntimeError("Custom global planner could not find the ego waypoint.")

        current_lane_id = int(self._lane_id_tracker.update(ego_waypoint))
        if current_lane_id == 0:
            current_lane_id = 1

        lane_ids = [
            int(canonical_lane_id_for_waypoint(waypoint))
            for waypoint in canonical_lane_waypoints(ego_waypoint)
            if int(canonical_lane_id_for_waypoint(waypoint)) != 0
        ]
        if not lane_ids:
            lane_ids = [current_lane_id]

        self._update_route(ego_pose=ego_pose, final_goal=snapshot.final_goal)
        self.route_manager.sync_carla_route_progress(
            ego_x_m=float(ego_pose["x"]),
            ego_y_m=float(ego_pose["y"]),
            ego_heading_rad=float(ego_pose["heading_rad"]),
        )

        cp_obstacles = [dict(item) for item in list(cp_payload.get("obstacles", []) or [])]
        lane_assignments = self.bridge._assign_obstacles_to_lanes(object_snapshots)
        lane_safety_scores = self.lane_safety_scorer.compute_lane_scores(ego_snapshot=ego_snapshot, obstacle_snapshots=object_snapshots, lane_assignments=lane_assignments, ego_lane_id=int(current_lane_id), available_lane_ids=lane_ids, timestamp_s=float(snapshot.timestamp_s))
        self.lane_safety_scorer.cleanup_stale_obstacles(set(lane_assignments.keys()))
        front_dist_by_lane = self.bridge._nearest_front_distance_by_lane(ego_snapshot=ego_snapshot, obstacle_snapshots=object_snapshots, lane_assignments=lane_assignments, available_lane_ids=lane_ids)

        route_points = self.bridge._active_global_route_points()
        route_summary = self.bridge._planning_module_global_route_summary(
            ego_location=PlannerLocation(x=float(ego_pose["x"]), y=float(ego_pose["y"]), z=float(ego_pose.get("z", 0.0))),
            ego_heading_rad=float(ego_pose["heading_rad"]),
            fallback_lane_id=int(current_lane_id),
            ego_waypoint=ego_waypoint,
        )
        route_optimal_lane_id = int(route_summary.get("optimal_lane_id", current_lane_id) or current_lane_id)
        route_reference_allowed = bool(self.bridge.use_opencda_global_route) and bool(self.bridge.opencda_global_route_reference_allowed) and bool(route_summary.get("route_found", False)) and len(route_points) >= 2
        route_reference_gate_reason = (
            "opencda_global_route_enabled"
            if route_reference_allowed
            else str(route_summary.get("debug_reason", "opencda_global_route_unavailable"))
        )

        cp_timestamp_s = _optional_float(cp_payload.get("timestamp_s", None))
        cp_age_s = "" if cp_timestamp_s is None else max(0.0, float(snapshot.timestamp_s) - float(cp_timestamp_s))
        cp_valid = True
        if cp_age_s != "":
            cp_valid = float(cp_age_s) <= float(self.config.get("max_cp_message_age_s", 1.0))
        traffic_controls = [dict(item) for item in list(cp_payload.get("control", []) or [])]
        ego_location_value = PlannerLocation(x=float(ego_pose["x"]), y=float(ego_pose["y"]), z=float(ego_pose.get("z", 0.0)))
        selected_control = self.bridge._select_relevant_traffic_control(
            traffic_controls=traffic_controls,
            ego_location=ego_location_value,
            ego_heading_rad=float(ego_pose["heading_rad"]),
            current_lane_id=current_lane_id,
            current_road_id=int(getattr(ego_waypoint, "road_id", 0) or 0),
            sim_time_s=float(snapshot.timestamp_s),
        )
        signal_context, stop_target = self.bridge._traffic_context_from_cp_control(selected_control=selected_control, ego_location=ego_location_value)
        if bool(self.config.get("ignore_traffic_control", False)):
            signal_context = {"signal_state": "unknown", "signal_source": "disabled_by_planner_config", "traffic_control_from_cp": False}
            stop_target = None
        traffic_control_context = TrafficControlContext.from_signal_context(
            signal_context=signal_context,
            stop_target=stop_target,
        )

        tracked_obstacles = self.tracker.update(
            obstacle_snapshots=object_snapshots,
            timestamp_s=float(snapshot.timestamp_s),
            signal_context=signal_context,
            stop_target=stop_target,
        )
        lane_assignments = self.bridge._assign_obstacles_to_lanes(tracked_obstacles)
        lane_safety_scores = self.lane_safety_scorer.compute_lane_scores(
            ego_snapshot=ego_snapshot,
            obstacle_snapshots=tracked_obstacles,
            lane_assignments=lane_assignments,
            ego_lane_id=current_lane_id,
            available_lane_ids=lane_ids,
            timestamp_s=float(snapshot.timestamp_s),
        )
        self.lane_safety_scorer.cleanup_stale_obstacles(set(lane_assignments.keys()))
        front_dist_by_lane = self.bridge._nearest_front_distance_by_lane(
            ego_snapshot=ego_snapshot,
            obstacle_snapshots=tracked_obstacles,
            lane_assignments=lane_assignments,
            available_lane_ids=lane_ids,
        )
        prediction_frame = self.tracker.predict(
            ego_snapshot=ego_snapshot,
            lane_assignments=lane_assignments,
            available_lane_ids=lane_ids,
            horizon_s=float(self.mpc.horizon_s),
            dt_s=float(self.mpc.dt_s),
            min_front_gap_m=self.min_front_gap_m,
            min_rear_gap_m=self.min_front_gap_m,
            min_ttc_s=self.min_ttc_s,
            lane_step_fn=self.bridge._obstacle_lane_step_fn(),
        )

        route_context = RouteContext(
            optimal_lane_id=route_optimal_lane_id,
            next_macro_maneuver=str(route_summary.get("next_macro_maneuver", "Continue Straight")),
            current_road_option=str(route_summary.get("current_road_option", "")),
            remaining_distance_m=float(route_summary.get("remaining_distance_m", 0.0) or 0.0),
            remaining_points_count=len(route_points),
            route_found=bool(route_summary.get("route_found", False)),
        )
        frame = PlannerInputFrame(
            planning=PlanningContext(
                sim_time_s=float(snapshot.timestamp_s),
                ego=EgoPlanningState(
                    x_m=float(ego_pose["x"]),
                    y_m=float(ego_pose["y"]),
                    speed_mps=float(ego_speed_mps),
                    heading_rad=float(ego_pose["heading_rad"]),
                    lane_id=current_lane_id,
                    road_id=int(getattr(ego_waypoint, "road_id", 0) or 0),
                    section_id=int(getattr(ego_waypoint, "section_id", 0) or 0),
                    in_junction=_waypoint_is_junction(ego_waypoint),
                ),
                route=route_context,
                traffic_control=traffic_control_context,
                targets=TargetContext(stop_target=traffic_control_context.stop_target),
                global_route_reference_allowed=route_reference_allowed,
                global_route_reference_gate_reason=route_reference_gate_reason,
            ),
            map_lane=MapLaneContext(
                lane_id=current_lane_id,
                road_id=int(getattr(ego_waypoint, "road_id", 0) or 0),
                section_id=int(getattr(ego_waypoint, "section_id", 0) or 0),
                lane_count=len(lane_ids),
                allowed_lane_ids=list(lane_ids),
                in_junction=_waypoint_is_junction(ego_waypoint),
                route_lane_id=route_optimal_lane_id,
                route_maneuver=str(route_context.next_macro_maneuver),
            ),
            perception=PerceptionContext(
                dynamic_objects=[dict(item) for item in tracked_obstacles],
                planning_objects=[dict(item) for item in tracked_obstacles],
                source="native_opencda",
            ),
            prediction=PredictionContext(
                lane_assignments=dict(lane_assignments),
                lane_prediction_risks=dict(prediction_frame.lane_prediction_risks),
                obstacle_future_trajectories=dict(prediction_frame.obstacle_future_trajectories),
                model="constant_acceleration",
                horizon_s=float(self.mpc.horizon_s),
                dt_s=float(self.mpc.dt_s),
            ),
            cp_messages=CPMessageContext(
                message_path=str(self.bridge.cp_message_path),
                traffic_controls=[dict(item) for item in traffic_controls],
                selected_traffic_control=selected_control,
                lane_closures=[dict(item) for item in list(cp_payload.get("lane_closures", cp_payload.get("lane_events", [])) or [])],
                obstacles=[dict(item) for item in cp_obstacles],
            ),
        )

        return PlannerInputAdapterOutput(
            frame=frame,
            ego_pose=ego_pose,
            current_state=current_state,
            current_lane_id=current_lane_id,
            lane_ids=list(lane_ids),
            ego_waypoint=ego_waypoint,
            ego_snapshot=ego_snapshot,
            lane_assignments=dict(lane_assignments),
            lane_safety_scores=dict(lane_safety_scores),
            front_distance_by_lane=dict(front_dist_by_lane),
            route_points=list(route_points),
            route_summary=dict(route_summary),
            route_optimal_lane_id=route_optimal_lane_id,
            route_reference_allowed=route_reference_allowed,
            route_reference_gate_reason=route_reference_gate_reason,
            selected_traffic_control=selected_control,
            signal_context=dict(signal_context),
            stop_target=stop_target,
            source_quality={
                "planner_input_frame_timestamp_s": float(snapshot.timestamp_s),
                "cp_message_timestamp_s": "" if cp_timestamp_s is None else float(cp_timestamp_s),
                "cp_message_age_s": cp_age_s,
                "cp_message_valid": bool(cp_valid),
                **dict(self.tracker.diagnostics),
            },
        )

    def _make_snapshot(self):
        """Convert the latest typed ROS messages into one plain-data snapshot used by the planner."""
        pose = self._localization.pose.pose
        twist = self._localization.twist.twist
        ego_pose = {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
            "heading_rad": self._yaw_from_quaternion(pose.orientation),
        }
        ego_speed_mps = math.sqrt(
            float(twist.linear.x) ** 2 + float(twist.linear.y) ** 2 + float(twist.linear.z) ** 2
        )
        final_goal = {
            "x": float(self._final_destination.pose.position.x),
            "y": float(self._final_destination.pose.position.y),
            "z": float(self._final_destination.pose.position.z),
        }
        return ROSInputSnapshot(
            timestamp_s=self._stamp_seconds(self._localization.header.stamp),
            ego_pose=ego_pose,
            ego_speed_mps=ego_speed_mps,
            perception_objects=self._tracked_objects(self._perception, source="opencda_perception", provider_source="native_opencda_perception"),
            v2x_objects=self._tracked_objects(self._v2x, source="opencda_v2x", provider_source="native_opencda_v2x"),
            traffic_lights=self._traffic_light_list(self._traffic_lights),
            lane_events=self._lane_event_list(self._cooperative),
            final_goal=final_goal,
        )

    def _update_route(self, *, ego_pose, final_goal):
        """Create a new global route only when the received final destination changes."""
        goal_signature = (
            round(float(final_goal["x"]), 3),
            round(float(final_goal["y"]), 3),
            round(float(final_goal.get("z", 0.0)), 3),
        )
        if goal_signature == self._active_goal_signature:
            return

        self.route_manager.set_destination(start_point=ego_pose, goal_point=final_goal)
        self._active_goal_signature = goal_signature

    def _tracked_objects(self, message, *, source, provider_source):
        """Convert Autoware TrackedObjects into the object dictionaries expected by the current planner."""
        output = []
        for tracked in list(message.objects):
            pose = tracked.kinematics.pose_with_covariance.pose
            twist = tracked.kinematics.twist_with_covariance.twist
            dimensions = tracked.shape.dimensions
            object_id = bytes(tracked.object_id.uuid).hex()
            speed_mps = math.sqrt(
                float(twist.linear.x) ** 2 + float(twist.linear.y) ** 2 + float(twist.linear.z) ** 2
            )
            heading_rad = self._yaw_from_quaternion(pose.orientation)

            object_type = "unknown"
            if tracked.classification:
                best_classification = max(tracked.classification, key=lambda item: float(item.probability))
                object_type = OBJECT_TYPES.get(int(best_classification.label), "unknown")

            output.append(
                {
                    "id": object_id,
                    "vehicle_id": object_id,
                    "type": object_type,
                    "x": float(pose.position.x),
                    "y": float(pose.position.y),
                    "z": float(pose.position.z),
                    "v": float(speed_mps),
                    "psi": float(heading_rad),
                    "state": [
                        float(pose.position.x),
                        float(pose.position.y),
                        float(speed_mps),
                        float(heading_rad),
                    ],
                    "length_m": float(dimensions.x),
                    "width_m": float(dimensions.y),
                    "height_m": float(dimensions.z),
                    "confidence": float(tracked.existence_probability),
                    "source": source,
                    "provider_source": provider_source,
                }
            )
        return output

    def _traffic_light_list(self, message):
        """Convert received traffic-light observations into the dictionaries used by traffic planning."""
        output = []
        for observation in list(message.traffic_lights):
            output.append(
                {
                    "id": str(observation.source_id),
                    "type": str(observation.type),
                    "state": str(observation.state).strip().lower(),
                    "position": {
                        "x": float(observation.position.x),
                        "y": float(observation.position.y),
                        "z": float(observation.position.z),
                    },
                    "confidence": float(observation.confidence),
                    "source": "ros_perception",
                }
            )
        return output

    def _lane_event_list(self, message):
        """Convert optional cooperative lane events into the CP dictionaries expected by the planner."""
        if message is None:
            return []

        output = []
        for event in list(message.lane_events):
            timestamp_s = self._stamp_seconds(event.source_stamp)
            ttl_s = self._duration_seconds(event.ttl)
            output.append(
                {
                    "id": str(event.id),
                    "type": LANE_EVENT_TYPES.get(int(event.event_type), "unknown"),
                    "source": str(event.source),
                    "timestamp_s": timestamp_s,
                    "ttl_s": ttl_s,
                    "valid_until_s": timestamp_s + ttl_s,
                    "position": {
                        "x": float(event.position.x),
                        "y": float(event.position.y),
                        "z": float(event.position.z),
                    },
                    "confidence": float(event.confidence),
                }
            )
        return output

    def _cp_message_is_fresh(message: Mapping[str, object], *, sim_time_s: float) -> bool:
        """Apply OpenCDA's TTL rule when a cooperative object includes freshness information."""
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

    def _traffic_light_to_control_message(self, *, traffic_light, ego_pose, sim_time_s):
        """Build the same internal traffic-control fields as OpenCDA, using only perception data and the custom map."""
        position = dict(traffic_light.get("position", {}))
        if "x" not in position or "y" not in position:
            return None
        dx_m = float(position["x"]) - float(ego_pose["x"])
        dy_m = float(position["y"]) - float(ego_pose["y"])
        distance_m = math.hypot(dx_m, dy_m)
        if self.communication_range_m > 0.0 and distance_m > self.communication_range_m:
            return None
        ego_heading_rad = float(ego_pose["heading_rad"])
        forward_m = math.cos(ego_heading_rad) * dx_m + math.sin(ego_heading_rad) * dy_m
        lateral_m = -math.sin(ego_heading_rad) * dx_m + math.cos(ego_heading_rad) * dy_m
        waypoint = self.map_planner.get_waypoint(position)
        lane_id = int(canonical_lane_id_for_waypoint(waypoint)) if waypoint is not None else 0
        road_id = int(getattr(waypoint, "road_id", 0) or 0) if waypoint is not None else 0
        section_id = int(getattr(waypoint, "section_id", 0) or 0) if waypoint is not None else 0
        # OpenCDA uses the ego heading in this stop-line field. Keep that exact rule.
        heading_rad = float(ego_heading_rad)
        state = str(traffic_light.get("state", "unknown") or "unknown").strip().lower()
        confidence = float(traffic_light.get("confidence", 1.0 if state in {"red", "yellow", "green"} else 0.3))
        stop_line = {"x_m": float(position["x"]), "y_m": float(position["y"]), "heading_rad": heading_rad, "lane_id": lane_id, "road_id": road_id, "section_id": section_id}
        ttl_s = max(0.2, 2.0 * float(self.mpc.dt_s))
        control_id = str(traffic_light.get("id", ""))
        return {
            "type": "traffic_light",
            "id": "native_opencda_tl:{}".format(control_id),
            "state": state,
            "signal_state": state,
            "control_id": control_id,
            "timestamp_s": float(sim_time_s),
            "ttl_s": ttl_s,
            "valid_until_s": float(sim_time_s) + ttl_s,
            "confidence": confidence,
            "distance_m": distance_m,
            "stop_line": dict(stop_line),
            "stop_line_position": dict(stop_line),
            "valid_range": {
                "search_distance_m": float(self.communication_range_m),
                "forward_m": forward_m,
                "lateral_m": lateral_m,
                "signal_forward_m": forward_m,
                "signal_lateral_m": lateral_m,
                "road_id": road_id,
                "section_id": section_id,
                "lane_id": lane_id,
            },
            "ego_passed_stop_line": bool(forward_m < -1.0),
            "source": "native_opencda",
            "provider_source": "native_opencda_traffic_light",
            "signal_actor_id": control_id,
            "signal_actor_name": "traffic.traffic_light" if str(traffic_light.get("type", "")).strip().lower() == "traffic_light" else str(traffic_light.get("type", "")),
            "signal_actor_raw_state": str(traffic_light.get("state", "unknown")).strip().capitalize(),
            "signal_distance_m": distance_m,
            "signal_forward_m": forward_m,
            "signal_lateral_m": lateral_m,
        }

    @staticmethod
    def _stamp_seconds(stamp):
        """Convert a ROS Time value into one floating-point number of seconds."""
        return float(stamp.sec) + float(stamp.nanosec) / 1000000000.0

    @staticmethod
    def _duration_seconds(duration):
        """Convert a ROS Duration value into one floating-point number of seconds."""
        return float(duration.sec) + float(duration.nanosec) / 1000000000.0

    @staticmethod
    def _yaw_from_quaternion(quaternion):
        """Convert a ROS orientation quaternion into the planar heading used by the planner."""
        return math.atan2(
            2.0
            * (
                float(quaternion.w) * float(quaternion.z)
                + float(quaternion.x) * float(quaternion.y)
            ),
            1.0 - 2.0 * (float(quaternion.y) ** 2 + float(quaternion.z) ** 2),
        )

    @staticmethod
    def _check_frame(message):
        """Reject a ROS message when its coordinates are explicitly labeled as something other than map."""
        header = getattr(message, "header", None)
        frame_id = "" if header is None else str(header.frame_id).strip()
        if frame_id and frame_id != "map":
            raise ValueError("Expected ROS frame 'map', received '{}'.".format(frame_id))


def _optional_float(value):
    """Match the OpenCDA adapter's optional numeric conversion."""
    try:
        return float(value)
    except Exception:
        return None


def _waypoint_is_junction(waypoint):
    """Read the custom waypoint intersection flag through the OpenCDA input field."""
    return bool(getattr(waypoint, "is_intersection", getattr(waypoint, "is_junction", False)))
