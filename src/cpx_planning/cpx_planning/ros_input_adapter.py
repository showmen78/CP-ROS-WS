"""Convert typed ROS messages into the existing CP-X planner input."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from cpx_planning.component_interfaces import ROSInputSnapshot
from cpx_planning.planner_core.planner_input_adapter import PlannerInputAdapterOutput
from cpx_planning.utility.global_planner import (
    canonical_lane_id_for_waypoint,
    canonical_lane_waypoints,
    world_heading_rad,
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
        map_planner,
        route_manager,
        tracker,
        lane_safety_scorer,
        mpc,
        min_front_gap_m=8.0,
        min_ttc_s=2.0,
        communication_range_m=80.0,
    ):
        """Store the planner components, prediction settings, and latest-message placeholders."""
        self.map_planner = map_planner
        self.reference_map = map_planner
        self.route_manager = route_manager
        self.tracker = tracker
        self.lane_safety_scorer = lane_safety_scorer
        self.mpc = mpc
        self.min_front_gap_m = float(min_front_gap_m)
        self.min_ttc_s = float(min_ttc_s)
        self.communication_range_m = max(0.0, float(communication_range_m))

        self._localization = None
        self._perception = None
        self._v2x = None
        self._traffic_lights = None
        self._cooperative = None
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

    def update_final_destination(self, message):
        """Save the latest mission destination used by the custom global planner."""
        self._check_frame(message)
        self._final_destination = message

    def ready(self):
        """Return True only when the required ROS inputs belong to the same CARLA cycle."""
        if self._localization is None or self._perception is None or self._v2x is None or self._traffic_lights is None or self._final_destination is None:
            return False

        timestamps = [
            self._stamp_seconds(self._localization.header.stamp),
            self._stamp_seconds(self._perception.header.stamp),
            self._stamp_seconds(self._v2x.header.stamp),
            self._stamp_seconds(self._traffic_lights.header.stamp),
            self._stamp_seconds(self._final_destination.header.stamp),
        ]

        return max(timestamps) - min(timestamps) <= 1.0e-6

    def build(self):
        """Convert the latest ROS messages and build one complete PlannerInputFrame."""
        if not self.ready():
            raise RuntimeError("ROS planner inputs are not ready.")

        snapshot = self._make_snapshot()
        ego_pose = dict(snapshot.ego_pose)
        ego_speed_mps = float(snapshot.ego_speed_mps)
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

        # Map matching stays inside the custom AD-map planner.
        ego_waypoint = self.map_planner.get_waypoint(ego_pose)
        if ego_waypoint is None:
            raise RuntimeError("Custom global planner could not find the ego waypoint.")

        current_lane_id = int(canonical_lane_id_for_waypoint(ego_waypoint))
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

        cp_obstacles = self._native_opencda_messages(local_object_snapshots=snapshot.perception_objects, v2x_object_snapshots=snapshot.v2x_objects, ego_location=ego_pose, sim_time_s=float(snapshot.timestamp_s))
        object_snapshots = self._fused_planning_object_snapshots(local_object_snapshots=snapshot.perception_objects, cp_obstacles=cp_obstacles, ego_location=ego_pose, sim_time_s=float(snapshot.timestamp_s))
        lane_assignments = self._assign_obstacles_to_lanes(object_snapshots)
        lane_safety_scores = self.lane_safety_scorer.compute_lane_scores(ego_snapshot=ego_snapshot, obstacle_snapshots=object_snapshots, lane_assignments=lane_assignments, ego_lane_id=int(current_lane_id), available_lane_ids=lane_ids, timestamp_s=float(snapshot.timestamp_s))
        self.lane_safety_scorer.cleanup_stale_obstacles(set(lane_assignments.keys()))
        front_dist_by_lane = self._nearest_front_distance_by_lane(ego_snapshot=ego_snapshot, obstacle_snapshots=object_snapshots, lane_assignments=lane_assignments, available_lane_ids=lane_ids)

        route_summary = self.route_manager.get_route_info(
            x_m=float(ego_pose["x"]),
            y_m=float(ego_pose["y"]),
            query_key="ros_planner_input",
            fallback_lane_id=current_lane_id,
        )
        route_points = self.route_manager.route_points(
            x_m=float(ego_pose["x"]),
            y_m=float(ego_pose["y"]),
            query_key="ros_planner_input_points",
        )
        route_optimal_lane_id = int(route_summary.get("optimal_lane_id", current_lane_id) or current_lane_id)
        route_reference_allowed = bool(route_summary.get("route_found", False)) and len(route_points) >= 2
        route_reference_gate_reason = (
            "custom_global_route_enabled"
            if route_reference_allowed
            else str(route_summary.get("debug_reason", "custom_global_route_unavailable"))
        )

        traffic_controls = [
            control
            for control in (
                self._traffic_light_to_control_message(
                    traffic_light=traffic_light,
                    ego_pose=ego_pose,
                    sim_time_s=float(snapshot.timestamp_s),
                )
                for traffic_light in snapshot.traffic_lights
            )
            if control is not None
        ]
        selected_control = self._select_relevant_traffic_control(
            traffic_controls=traffic_controls,
            ego_location=ego_pose,
            ego_heading_rad=float(ego_pose["heading_rad"]),
            current_lane_id=current_lane_id,
            current_road_id=int(getattr(ego_waypoint, "road_id", 0) or 0),
            sim_time_s=float(snapshot.timestamp_s),
        )
        signal_context, stop_target = self._traffic_context_from_cp_control(selected_control=selected_control, ego_location=ego_pose)
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
        lane_assignments = self._assign_obstacles_to_lanes(tracked_obstacles)
        lane_safety_scores = self.lane_safety_scorer.compute_lane_scores(
            ego_snapshot=ego_snapshot,
            obstacle_snapshots=tracked_obstacles,
            lane_assignments=lane_assignments,
            ego_lane_id=current_lane_id,
            available_lane_ids=lane_ids,
            timestamp_s=float(snapshot.timestamp_s),
        )
        self.lane_safety_scorer.cleanup_stale_obstacles(set(lane_assignments.keys()))
        front_dist_by_lane = self._nearest_front_distance_by_lane(
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
                    in_junction=bool(getattr(ego_waypoint, "is_intersection", False)),
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
                in_junction=bool(getattr(ego_waypoint, "is_intersection", False)),
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
                message_path="",
                traffic_controls=[dict(item) for item in traffic_controls],
                selected_traffic_control=selected_control,
                lane_closures=[dict(item) for item in snapshot.lane_events],
                obstacles=[dict(item) for item in cp_obstacles],
            ),
        )

        tracker_diagnostics = dict(getattr(self.tracker, "diagnostics", {}))
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
                "cp_message_timestamp_s": float(snapshot.timestamp_s),
                "cp_message_age_s": 0.0,
                "cp_message_valid": True,
                **tracker_diagnostics,
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

    def _native_opencda_messages(self, *, local_object_snapshots, v2x_object_snapshots, ego_location, sim_time_s):
        """Build the same CP obstacle list as OpenCDACPProvider._native_opencda_messages."""
        messages = []
        seen_actor_ids = set()

        for index, snapshot in enumerate(list(local_object_snapshots or [])):
            actor_id = self._object_actor_id(snapshot, fallback="perception:{}".format(index))
            if actor_id in seen_actor_ids:
                continue
            message = self._object_to_cp_message(obj=snapshot, ego_location=ego_location, sim_time_s=float(sim_time_s), source="opencda_perception", provider_source="native_opencda_perception", fallback_id=actor_id)
            if isinstance(message, Mapping):
                seen_actor_ids.add(actor_id)
                messages.append(dict(message))

        for index, snapshot in enumerate(list(v2x_object_snapshots or [])):
            actor_id = self._object_actor_id(snapshot, fallback="v2x:{}".format(index))
            if actor_id in seen_actor_ids:
                continue
            message = self._object_to_cp_message(obj=snapshot, ego_location=ego_location, sim_time_s=float(sim_time_s), source="opencda_v2x", provider_source="native_opencda_v2x", fallback_id=actor_id)
            if isinstance(message, Mapping):
                seen_actor_ids.add(actor_id)
                messages.append(dict(message))

        return messages

    def _object_to_cp_message(self, *, obj, ego_location, sim_time_s, source, provider_source, fallback_id):
        """Copy OpenCDACPProvider._object_to_cp_message using primitive ROS object data and the custom map."""
        try:
            location = {"x": float(obj.get("x", 0.0)), "y": float(obj.get("y", 0.0)), "z": float(obj.get("z", 0.0))}
            speed_mps = max(0.0, float(obj.get("v", 0.0)))
            heading_rad = float(obj.get("psi", 0.0))
            distance_m = math.hypot(float(location["x"]) - float(ego_location["x"]), float(location["y"]) - float(ego_location["y"]))
        except Exception:
            return None
        if self.communication_range_m > 0.0 and distance_m > self.communication_range_m:
            return None

        lane_id = 0
        road_id = -1
        try:
            waypoint = self._map_waypoint_from_location(map_planner=self.map_planner, location=location)
            lane_id = int(canonical_lane_id_for_waypoint(waypoint))
            road_id = int(getattr(waypoint, "road_id", -1) or -1)
        except Exception:
            pass

        trajectory = self._constant_velocity_trajectory(x_m=float(location["x"]), y_m=float(location["y"]), speed_mps=float(speed_mps), heading_rad=float(heading_rad))
        actor_id = self._object_actor_id(obj, fallback=fallback_id)
        return {
            "id": "{}:{}".format(provider_source, actor_id),
            "type": "vehicle",
            "source": str(source),
            "provider_source": str(provider_source),
            "timestamp_s": float(sim_time_s),
            "ttl_s": max(0.2, 2.0 * float(self.mpc.dt_s)),
            "confidence": 1.0,
            "distance_m": float(distance_m),
            "state": [float(location["x"]), float(location["y"]), float(speed_mps), float(heading_rad)],
            "z": float(location["z"]),
            "shape": {
                "length_m": float(obj.get("length_m", 4.5)),
                "width_m": float(obj.get("width_m", 2.0)),
                "height_m": float(obj.get("height_m", 1.8)),
            },
            "road_id": int(road_id),
            "lane_id": int(lane_id),
            "trajectory": trajectory,
        }

    @staticmethod
    def _object_actor_id(obj, fallback):
        """Return the same stable actor identifier used by OpenCDACPProvider."""
        if isinstance(obj, Mapping):
            for key in ("id", "vehicle_id", "vid"):
                value = obj.get(key)
                if value is not None and str(value).strip():
                    return str(value)
        return str(fallback)

    @staticmethod
    def _map_waypoint_from_location(*, map_planner, location):
        """Use the custom map at the same boundary where OpenCDA queries its map planner."""
        if map_planner is None or location is None:
            return None
        get_waypoint = getattr(map_planner, "get_waypoint", None)
        if not callable(get_waypoint):
            return None
        try:
            return get_waypoint(location)
        except Exception:
            return None

    def _constant_velocity_trajectory(self, *, x_m, y_m, speed_mps, heading_rad):
        """Copy OpenCDACPProvider's constant-velocity CP trajectory calculation."""
        prediction_horizon_s = float(self.mpc.horizon_s)
        prediction_dt_s = float(self.mpc.dt_s)
        steps = max(1, int(round(prediction_horizon_s / prediction_dt_s)))
        cos_h = math.cos(float(heading_rad))
        sin_h = math.sin(float(heading_rad))
        trajectory = []
        for index in range(steps + 1):
            time_s = float(index) * prediction_dt_s
            trajectory.append([float(x_m + speed_mps * cos_h * time_s), float(y_m + speed_mps * sin_h * time_s), float(speed_mps), float(heading_rad)])
        return trajectory

    def _fused_planning_object_snapshots(
        self,
        *,
        local_object_snapshots: Sequence[Mapping[str, object]],
        cp_obstacles: Sequence[Mapping[str, object]],
        ego_location: Mapping[str, object],
        sim_time_s: float,
    ) -> list[dict[str, object]]:
        """Fuse local perception and V2X objects using the same priority rules as OpenCDA."""
        fused_by_key: dict[str, dict[str, object]] = {}
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
        cp_snapshot: Mapping[str, object],
        fused_snapshots: Sequence[Mapping[str, object]],
        max_position_delta_m: float = 1.0,
    ) -> bool:
        """Match OpenCDA's position check for a perception object repeated in cooperative data."""
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

    @staticmethod
    def _normalize_local_object_snapshot(snapshot: Mapping[str, object]) -> dict[str, object] | None:
        """Normalize a ROS perception object to the same planner format used by OpenCDA."""
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
    def _normalize_cp_obstacle_snapshot(obstacle: Mapping[str, object]) -> dict[str, object] | None:
        """Normalize a ROS V2X object to the cooperative-object format used by OpenCDA."""
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
            speed_mps = obstacle.get(
                "v",
                obstacle.get("speed_mps", state_values[2] if len(state_values) >= 3 else 0.0),
            )
            heading_rad = obstacle.get(
                "psi",
                obstacle.get("heading_rad", state_values[3] if len(state_values) >= 4 else 0.0),
            )
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
            }
        except Exception:
            return None

    @staticmethod
    def _obstacle_source_priority(snapshot: Mapping[str, object]) -> int:
        """Use OpenCDA's source order: perception, V2X, other sources, then fallbacks."""
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
    def _fused_obstacle_key(snapshot: Mapping[str, object]) -> str:
        """Create the same normalized duplicate-detection key used by OpenCDA."""
        obstacle_id = str(snapshot.get("vehicle_id", snapshot.get("id", ""))).strip()
        return obstacle_id.rsplit(":", 1)[-1] if ":" in obstacle_id else obstacle_id

    @classmethod
    def _upsert_fused_obstacle(
        cls,
        *,
        fused_by_key: dict[str, dict[str, object]],
        priorities_by_key: dict[str, int],
        snapshot: Mapping[str, object],
        priority: int,
    ) -> None:
        """Keep the higher-priority report, or the higher-confidence report when priorities match."""
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

    @staticmethod
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

    def _assign_obstacles_to_lanes(self, object_snapshots: Sequence[Mapping[str, object]]) -> dict[str, int]:
        """Use the custom map to assign every object to a canonical lane, matching OpenCDA's method."""
        assignments: dict[str, int] = {}
        for snapshot in list(object_snapshots or []):
            obstacle_id = self._object_track_id(snapshot)
            if not obstacle_id:
                continue

            waypoint = self.reference_map.get_waypoint(
                {
                    "x": float(snapshot.get("x", 0.0)),
                    "y": float(snapshot.get("y", 0.0)),
                    "z": float(snapshot.get("z", 0.0)),
                }
            )
            lane_id = int(canonical_lane_id_for_waypoint(waypoint))
            if int(lane_id) != 0:
                assignments[obstacle_id] = int(lane_id)

        return assignments

    @staticmethod
    def _object_track_id(snapshot: Mapping[str, object]) -> str:
        """Return the same stable object key that OpenCDA uses for tracking and lane assignment."""
        for key in ("track_id", "object_id", "vehicle_id", "actor_id", "id"):
            value = snapshot.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        try:
            return "xy:{:.1f}:{:.1f}".format(float(snapshot.get("x", snapshot.get("x_m", 0.0))), float(snapshot.get("y", snapshot.get("y_m", 0.0))))
        except Exception:
            return ""

    @staticmethod
    def _nearest_front_distance_by_lane(
        *,
        ego_snapshot: Mapping[str, object],
        obstacle_snapshots: Sequence[Mapping[str, object]],
        lane_assignments: Mapping[str, int],
        available_lane_ids: Sequence[int],
    ) -> dict[int, float]:
        """Measure the nearest forward object distance in each lane using OpenCDA's calculation."""
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
        map_heading_rad = world_heading_rad(waypoint) if waypoint is not None else None
        heading_rad = ego_heading_rad if map_heading_rad is None else float(map_heading_rad)
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

    def _select_relevant_traffic_control(self, *, traffic_controls, ego_location, ego_heading_rad, current_lane_id, current_road_id, sim_time_s):
        """Select the relevant unpassed traffic light with the same scoring order used by OpenCDA."""
        best_control = None
        best_score = None
        cos_h = math.cos(float(ego_heading_rad))
        sin_h = math.sin(float(ego_heading_rad))
        for control in list(traffic_controls or []):
            if not isinstance(control, Mapping) or not self._cp_message_is_fresh(control, sim_time_s=float(sim_time_s)):
                continue
            stop_line = control.get("stop_line_position", control.get("stop_line"))
            if not isinstance(stop_line, Mapping):
                continue
            x_value = stop_line.get("x", stop_line.get("x_m"))
            y_value = stop_line.get("y", stop_line.get("y_m"))
            if x_value is None or y_value is None:
                continue
            dx_m = float(x_value) - float(ego_location["x"])
            dy_m = float(y_value) - float(ego_location["y"])
            forward_m = cos_h * dx_m + sin_h * dy_m
            lateral_m = -sin_h * dx_m + cos_h * dy_m
            if bool(control.get("ego_passed_stop_line", False)) or forward_m < -1.0:
                continue
            lane_id = int(float(control.get("lane_id", stop_line.get("lane_id", 0)) or 0))
            road_id = int(float(control.get("road_id", stop_line.get("road_id", 0)) or 0))
            road_mismatch = 1.0 if road_id and current_road_id and road_id != current_road_id else 0.0
            lane_mismatch = 1.0 if lane_id and current_lane_id and lane_id != current_lane_id else 0.0
            score = (road_mismatch, lane_mismatch, abs(lateral_m) + 0.01 * forward_m)
            if best_score is None or score < best_score:
                best_control = control
                best_score = score
        return best_control

    @staticmethod
    def _traffic_context_from_cp_control(*, selected_control, ego_location):
        """Convert the selected internal control into the same signal context and stop target used by OpenCDA."""
        if not isinstance(selected_control, Mapping):
            return {"signal_state": "unknown", "from_cp": False}, None
        state = str(selected_control.get("signal_state", selected_control.get("state", "unknown")) or "unknown").strip().lower()
        stop_line = selected_control.get("stop_line_position", selected_control.get("stop_line"))
        stop_target = None
        if isinstance(stop_line, Mapping):
            x_value = stop_line.get("x", stop_line.get("x_m"))
            y_value = stop_line.get("y", stop_line.get("y_m"))
            if x_value is not None and y_value is not None:
                stop_target = {
                    "x_m": float(x_value),
                    "y_m": float(y_value),
                    "lane_id": int(float(selected_control.get("lane_id", stop_line.get("lane_id", 0)) or 0)),
                    "road_id": int(float(selected_control.get("road_id", stop_line.get("road_id", 0)) or 0)),
                    "distance_m": math.hypot(float(x_value) - float(ego_location["x"]), float(y_value) - float(ego_location["y"])),
                    "source": "opencda_cp_control",
                }
        context = {
            "signal_state": state,
            "signal_source": str(selected_control.get("source", "opencda_cp")),
            "source": str(selected_control.get("source", "opencda_cp")),
            "cp_control_id": str(selected_control.get("control_id", selected_control.get("id", ""))),
            "control_id": str(selected_control.get("control_id", selected_control.get("id", ""))),
            "cp_provider_source": str(selected_control.get("provider_source", "")),
            "provider_source": str(selected_control.get("provider_source", "")),
            "from_cp": True,
            "traffic_control_from_cp": True,
            "confidence": float(selected_control.get("confidence", 1.0) or 0.0),
            "ego_passed_stop_line": bool(selected_control.get("ego_passed_stop_line", False)),
        }
        return context, stop_target

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
