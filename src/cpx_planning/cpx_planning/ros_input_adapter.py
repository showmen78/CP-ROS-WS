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
        prediction_horizon_s=3.0,
        prediction_dt_s=0.2,
        min_front_gap_m=8.0,
        min_rear_gap_m=8.0,
        min_ttc_s=2.0,
    ):
        """Store the planner components, prediction settings, and latest-message placeholders."""
        self.map_planner = map_planner
        self.route_manager = route_manager
        self.tracker = tracker
        self.lane_safety_scorer = lane_safety_scorer
        self.prediction_horizon_s = float(prediction_horizon_s)
        self.prediction_dt_s = float(prediction_dt_s)
        self.min_front_gap_m = float(min_front_gap_m)
        self.min_rear_gap_m = float(min_rear_gap_m)
        self.min_ttc_s = float(min_ttc_s)

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
        """Return True after every regularly required ROS input has arrived at least once."""
        return (
            self._localization is not None
            and self._perception is not None
            and self._v2x is not None
            and self._traffic_lights is not None
            and self._final_destination is not None
        )

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
        self.route_manager.sync_route_progress(
            ego_x_m=float(ego_pose["x"]),
            ego_y_m=float(ego_pose["y"]),
            ego_heading_rad=float(ego_pose["heading_rad"]),
        )
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

        signal_context, stop_target, selected_signal = self._select_traffic_light(
            traffic_lights=snapshot.traffic_lights,
            ego_pose=ego_pose,
            current_lane_id=current_lane_id,
            current_road_id=int(getattr(ego_waypoint, "road_id", 0) or 0),
        )
        traffic_control_context = TrafficControlContext.from_signal_context(
            signal_context=signal_context,
            stop_target=stop_target,
        )

        object_snapshots = self._fused_planning_object_snapshots(
            local_object_snapshots=snapshot.perception_objects,
            cp_obstacles=snapshot.v2x_objects,
            sim_time_s=float(snapshot.timestamp_s),
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
            horizon_s=self.prediction_horizon_s,
            dt_s=self.prediction_dt_s,
            min_front_gap_m=self.min_front_gap_m,
            min_rear_gap_m=self.min_rear_gap_m,
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
        final_goal = [
            float(snapshot.final_goal["x"]),
            float(snapshot.final_goal["y"]),
            float(snapshot.final_goal.get("z", 0.0)),
        ]
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
                targets=TargetContext(stop_target=traffic_control_context.stop_target, final_goal=final_goal),
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
                source="ros_perception_and_v2x",
            ),
            prediction=PredictionContext(
                lane_assignments=dict(lane_assignments),
                lane_prediction_risks=dict(prediction_frame.lane_prediction_risks),
                obstacle_future_trajectories=dict(prediction_frame.obstacle_future_trajectories),
                model="constant_acceleration",
                horizon_s=self.prediction_horizon_s,
                dt_s=self.prediction_dt_s,
            ),
            cp_messages=CPMessageContext(
                message_path="",
                traffic_controls=[],
                selected_traffic_control=None,
                lane_closures=[dict(item) for item in snapshot.lane_events],
                obstacles=[dict(item) for item in snapshot.v2x_objects],
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
            selected_traffic_control=selected_signal,
            signal_context=dict(signal_context),
            stop_target=stop_target,
            source_quality={
                "planner_input_frame_timestamp_s": float(snapshot.timestamp_s),
                "input_source": "ros",
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
            perception_objects=self._tracked_objects(self._perception, source="ros_perception"),
            v2x_objects=self._tracked_objects(self._v2x, source="ros_v2x"),
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

    def _tracked_objects(self, message, *, source):
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
                    "provider_source": source,
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

    def _fused_planning_object_snapshots(
        self,
        *,
        local_object_snapshots: Sequence[Mapping[str, object]],
        cp_obstacles: Sequence[Mapping[str, object]],
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
            state_values = (
                list(state)
                if isinstance(state, Sequence) and not isinstance(state, (str, bytes, bytearray))
                else []
            )
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
                "source": str(obstacle.get("source", "opencda_cp")),
                "provider_source": str(obstacle.get("provider_source", "opencda_cp")),
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

            waypoint = self.map_planner.get_waypoint(
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
        for key in ("track_id", "vehicle_id", "id", "actor_id"):
            value = snapshot.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    def _nearest_front_distance_by_lane(
        self,
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

    def _select_traffic_light(
        self,
        *,
        traffic_lights,
        ego_pose,
        current_lane_id,
        current_road_id,
    ):
        """Select the most relevant forward traffic light and create its signal context and stop target."""
        best = None
        ego_heading = float(ego_pose["heading_rad"])
        cos_heading = math.cos(ego_heading)
        sin_heading = math.sin(ego_heading)

        for signal in traffic_lights:
            position = dict(signal.get("position", {}))
            dx = float(position.get("x", 0.0)) - float(ego_pose["x"])
            dy = float(position.get("y", 0.0)) - float(ego_pose["y"])
            forward_m = cos_heading * dx + sin_heading * dy
            lateral_m = -sin_heading * dx + cos_heading * dy

            # A light already behind the ego should not control the next motion.
            if forward_m < -1.0:
                continue

            waypoint = self.map_planner.get_waypoint(position)
            if waypoint is None:
                continue

            signal_lane_id = int(canonical_lane_id_for_waypoint(waypoint))
            signal_road_id = int(getattr(waypoint, "road_id", 0) or 0)
            road_mismatch = int(
                bool(current_road_id) and bool(signal_road_id) and signal_road_id != current_road_id
            )
            lane_mismatch = int(
                bool(current_lane_id) and bool(signal_lane_id) and signal_lane_id != current_lane_id
            )
            score = (road_mismatch, lane_mismatch, abs(float(lateral_m)), max(0.0, float(forward_m)))
            if best is None or score < best[0]:
                best = (score, dict(signal), waypoint, forward_m, lateral_m)

        if best is None:
            return (
                {
                    "signal_state": "unknown",
                    "signal_source": "ros_perception",
                    "from_cp": False,
                    "traffic_control_from_cp": False,
                    "confidence": 0.0,
                },
                None,
                None,
            )

        _, signal, waypoint, forward_m, lateral_m = best
        position = dict(signal["position"])
        distance_m = math.hypot(
            float(position["x"]) - float(ego_pose["x"]),
            float(position["y"]) - float(ego_pose["y"]),
        )
        lane_id = int(canonical_lane_id_for_waypoint(waypoint))
        road_id = int(getattr(waypoint, "road_id", 0) or 0)
        section_id = int(getattr(waypoint, "section_id", 0) or 0)
        heading_rad = world_heading_rad(waypoint)
        stop_target = {
            "x_m": float(position["x"]),
            "y_m": float(position["y"]),
            "heading_rad": ego_pose["heading_rad"] if heading_rad is None else float(heading_rad),
            "lane_id": lane_id,
            "road_id": road_id,
            "section_id": section_id,
            "distance_m": float(distance_m),
            "source": "ros_perception_traffic_light",
        }
        signal_context = {
            "signal_state": str(signal.get("state", "unknown")),
            "signal_source": "ros_perception",
            "source": "ros_perception",
            "control_id": str(signal.get("id", "")),
            "provider_source": "opencda_perception_ros",
            "from_cp": False,
            "traffic_control_from_cp": False,
            "confidence": float(signal.get("confidence", 0.0)),
            "ego_passed_stop_line": False,
            "signal_forward_m": float(forward_m),
            "signal_lateral_m": float(lateral_m),
        }
        return signal_context, stop_target, signal

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
