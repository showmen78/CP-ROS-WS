"""Route lifecycle manager for the CP-X OpenCDA pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass
class RouteManagerStatus:
    route_found: bool = False
    route_point_count: int = 0
    remaining_distance_m: float = 0.0
    reached_destination: bool = False
    debug_reason: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "route_found": bool(self.route_found),
            "route_point_count": int(self.route_point_count),
            "remaining_distance_m": float(self.remaining_distance_m),
            "reached_destination": bool(self.reached_destination),
            "debug_reason": str(self.debug_reason),
        }


@dataclass(frozen=True)
class RouteReplanResult:
    success: bool
    reason: str
    route_point_count: int = 0


class CPXRouteManager:
    """Own destination, active global route, progress, and query diagnostics."""

    def __init__(
        self,
        *,
        global_planner: Any,
        carla_map: Any = None,
        carla_api: Any = None,
        carla_route_sampling_resolution_m: float = 1.0,
        carla_reference_smoothing_passes: int = 3,
        carla_turn_connector_smoothing_passes: int = 16,
        carla_reference_boundary_aware: bool = True,
        carla_reference_vehicle_half_width_m: float = 1.0,
        carla_reference_boundary_margin_m: float = 0.15,
        carla_reference_tracking_reserve_m: float = 0.20,
        carla_rejoin_min_lateral_m: float = 0.35,
        carla_rejoin_max_lateral_m: float = 3.0,
        carla_rejoin_distance_m: float = 8.0,
        geometry_turn_min_heading_change_rad: float = math.radians(30.0),
        reached_distance_m: float = 3.0,
        stale_route_lateral_m: float = 12.0,
    ) -> None:
        self.global_planner = global_planner
        self.carla_map = carla_map
        self.carla_api = carla_api
        self.carla_route_sampling_resolution_m = max(
            0.25, float(carla_route_sampling_resolution_m)
        )
        self.carla_reference_smoothing_passes = max(
            0, int(carla_reference_smoothing_passes)
        )
        self.carla_turn_connector_smoothing_passes = max(
            0, int(carla_turn_connector_smoothing_passes)
        )
        self.carla_reference_boundary_aware = bool(
            carla_reference_boundary_aware
        )
        self.carla_reference_vehicle_half_width_m = max(
            0.0, float(carla_reference_vehicle_half_width_m)
        )
        self.carla_reference_boundary_margin_m = max(
            0.0, float(carla_reference_boundary_margin_m)
        )
        self.carla_reference_tracking_reserve_m = max(
            0.0, float(carla_reference_tracking_reserve_m)
        )
        self.carla_rejoin_min_lateral_m = max(
            0.0, float(carla_rejoin_min_lateral_m)
        )
        self.carla_rejoin_max_lateral_m = max(
            self.carla_rejoin_min_lateral_m,
            float(carla_rejoin_max_lateral_m),
        )
        self.carla_rejoin_distance_m = max(
            1.0, float(carla_rejoin_distance_m)
        )
        self.geometry_turn_min_heading_change_rad = min(
            math.pi,
            max(0.0, float(geometry_turn_min_heading_change_rad)),
        )
        self.reached_distance_m = max(0.1, float(reached_distance_m))
        self.stale_route_lateral_m = max(1.0, float(stale_route_lateral_m))
        self._active_route_summary = None
        self._start_point: Optional[Dict[str, float]] = None
        self._goal_point: Optional[Dict[str, float]] = None
        self._fallback_route_points: List[List[float]] = []
        self._carla_route_planner = None
        self._carla_route_entries: List[Any] = []
        self._carla_route_nodes_cache_key: Optional[Tuple[int, int, int, int]] = None
        self._carla_route_nodes_cache: Tuple[Tuple[float, float, float, Any, str], ...] = ()
        self._geometry_route_points_cache_key: Optional[Tuple[int, int, int, int]] = None
        self._geometry_route_points_cache: Tuple[Tuple[float, float, float, float], ...] = ()
        self._carla_route_progress_index = 0
        self._carla_route_progress_initialized = False
        self._carla_route_projection: Optional[Tuple[int, float, float, float, float]] = None
        self._carla_route_sync_reason = "carla_route_progress_not_initialized"
        self._carla_route_debug_reason = "carla_route_not_initialized"
        self._external_carla_route_active = False
        self._last_status = RouteManagerStatus(debug_reason="route_not_initialized")

    def set_destination(
        self,
        *,
        start_point: Mapping[str, object],
        goal_point: Mapping[str, object],
    ) -> Any:
        self._external_carla_route_active = False
        self._start_point = _point_dict(start_point)
        self._goal_point = _point_dict(goal_point)
        self._active_route_summary = self.global_planner.plan_route_from_locations(
            start_location=self._start_point,
            goal_location=self._goal_point,
            replace_stored_route=True,
        )
        self._build_carla_route(
            start_point=self._start_point,
            goal_point=self._goal_point,
        )
        imported_summary = self._register_carla_mission_route_with_global_planner()
        if imported_summary is not None:
            self._active_route_summary = imported_summary
        self._fallback_route_points = []
        if not bool(getattr(self._active_route_summary, "route_found", False)) or len(
            list(getattr(self._active_route_summary, "route_waypoints", []) or [])
        ) < 2:
            self._fallback_route_points = self._direct_fallback_route_points(
                start_point=self._start_point,
                goal_point=self._goal_point,
            )
        self._last_status = self._status_from_summary(self._active_route_summary)
        return self._active_route_summary

    def replan_from(
        self,
        *,
        start_point: Mapping[str, object],
        trigger_reason: str,
    ) -> RouteReplanResult:
        """Atomically rebuild the route from ego to the existing destination."""

        if self._goal_point is None:
            return RouteReplanResult(False, "route_replan_goal_unavailable")
        snapshot = {
            "_active_route_summary": self._active_route_summary,
            "_start_point": self._start_point,
            "_fallback_route_points": self._fallback_route_points,
            "_carla_route_entries": self._carla_route_entries,
            "_carla_route_progress_index": self._carla_route_progress_index,
            "_carla_route_progress_initialized": self._carla_route_progress_initialized,
            "_carla_route_projection": self._carla_route_projection,
            "_carla_route_sync_reason": self._carla_route_sync_reason,
            "_carla_route_debug_reason": self._carla_route_debug_reason,
            "_external_carla_route_active": self._external_carla_route_active,
            "_last_status": self._last_status,
        }
        normalized_start = _point_dict(start_point)
        try:
            summary = self.global_planner.plan_route_from_locations(
                start_location=normalized_start,
                goal_location=self._goal_point,
                replace_stored_route=True,
            )
            if str(trigger_reason).strip().lower().startswith("static_obstacle"):
                if (
                    not bool(getattr(summary, "route_found", False))
                    or len(list(getattr(summary, "route_waypoints", []) or [])) < 2
                ):
                    raise RuntimeError("blocked_summary_route_not_found")
                self._build_carla_route_from_summary(summary)
            else:
                self._build_carla_route(
                    start_point=normalized_start,
                    goal_point=self._goal_point,
                )
                imported_summary = (
                    self._register_carla_mission_route_with_global_planner()
                )
                if imported_summary is not None:
                    summary = imported_summary
            route_point_count = len(self._carla_route_nodes())
            if route_point_count < 2:
                raise RuntimeError(str(self._carla_route_debug_reason))
            self._active_route_summary = summary
            self._start_point = normalized_start
            self._external_carla_route_active = False
            self._fallback_route_points = []
            self._last_status = self._status_from_summary(summary)
            self._carla_route_debug_reason = (
                f"blocked_summary_route_replanned:{str(trigger_reason)}"
                if str(trigger_reason).strip().lower().startswith("static_obstacle")
                else f"carla_grp_route_replanned:{str(trigger_reason)}"
            )
            return RouteReplanResult(
                True,
                str(self._carla_route_debug_reason),
                route_point_count=route_point_count,
            )
        except Exception as exc:
            for name, value in snapshot.items():
                setattr(self, name, value)
            return RouteReplanResult(
                False,
                f"route_replan_failed:{str(trigger_reason)}:{exc}",
            )

    def set_external_carla_route(self, route_entries: Sequence[Any]) -> None:
        """Use a runtime-provided CARLA/Leaderboard route as source of truth."""

        normalized_entries: List[Any] = []
        for entry in list(route_entries or []):
            raw_node, option = _carla_route_entry(entry)
            # carla.Waypoint exposes ``transform`` as a property, while
            # carla.Transform exposes ``transform(location)`` as a method.
            # Testing only for the attribute therefore mistakes a Transform
            # for a Waypoint and silently drops every Leaderboard route point.
            transform_attr = getattr(raw_node, "transform", None)
            transform = (
                transform_attr
                if hasattr(transform_attr, "location")
                else None
            )
            if transform is None and hasattr(raw_node, "location"):
                transform = raw_node
            location = getattr(transform, "location", None)
            if location is None:
                continue
            is_waypoint = (
                hasattr(transform_attr, "location")
                and hasattr(raw_node, "road_id")
                and hasattr(raw_node, "lane_id")
            )
            waypoint = raw_node if is_waypoint else None
            if waypoint is None and self.carla_map is not None:
                try:
                    waypoint = self.carla_map.get_waypoint(location)
                except Exception:
                    waypoint = None
            if waypoint is None or not hasattr(waypoint, "transform"):
                continue
            normalized_entries.append((waypoint, option))

        if len(normalized_entries) < 2:
            raise ValueError(
                "External CARLA route must contain at least two map-projectable entries"
            )

        self._carla_route_entries = normalized_entries
        self._external_carla_route_active = True
        self._active_route_summary = None
        self._carla_route_progress_index = 0
        self._carla_route_progress_initialized = False
        self._carla_route_projection = None
        self._carla_route_sync_reason = "external_route_progress_not_initialized"
        self._carla_route_debug_reason = "external_leaderboard_route_ready"

        nodes = self._carla_route_nodes()
        self._fallback_route_points = []
        for index, node in enumerate(nodes):
            if index + 1 < len(nodes):
                next_node = nodes[index + 1]
                heading_rad = math.atan2(
                    float(next_node[1]) - float(node[1]),
                    float(next_node[0]) - float(node[0]),
                )
            elif self._fallback_route_points:
                heading_rad = float(self._fallback_route_points[-1][3])
            else:
                heading_rad = 0.0
            self._fallback_route_points.append(
                [float(node[0]), float(node[1]), float(node[2]), float(heading_rad)]
            )
        self._start_point = _point_dict(
            {
                "x": nodes[0][0],
                "y": nodes[0][1],
                "z": nodes[0][2],
            }
        )
        self._goal_point = _point_dict(
            {
                "x": nodes[-1][0],
                "y": nodes[-1][1],
                "z": nodes[-1][2],
            }
        )
        imported_summary = self._register_carla_mission_route_with_global_planner()
        if imported_summary is not None:
            self._active_route_summary = imported_summary
        self._last_status = RouteManagerStatus(
            route_found=True,
            route_point_count=len(nodes),
            remaining_distance_m=self._fallback_route_total_distance_m(),
            reached_destination=False,
            debug_reason="external_leaderboard_route_ready",
        )

    def _register_carla_mission_route_with_global_planner(self) -> Any:
        """Map-match the supplied mission route into the semantic backend.

        Start/goal-only Dijkstra is allowed to choose a different valid road
        sequence from CARLA/OpenCDA's scenario route.  Importing the ordered
        GRP geometry keeps one mission route while still letting AD-map own
        stable lane identity, topology, maneuver labels, and progress.
        """
        register = getattr(self.global_planner, "register_imported_route", None)
        nodes = self._carla_route_nodes()
        if not callable(register) or len(nodes) < 2:
            return None
        route_points = [
            [float(node[0]), float(node[1]), float(node[2])]
            for node in nodes
        ]
        try:
            return register(route_points)
        except Exception:
            # Geometry remains usable and the original planner route remains
            # intact if an optional semantic backend cannot import a point.
            return None

    def get_route_info(
        self,
        *,
        x_m: float,
        y_m: float,
        query_key: str,
        fallback_lane_id: int,
        ego_waypoint: Any = None,
    ) -> Dict[str, object]:
        if bool(self._external_carla_route_active):
            return self._external_route_info(
                x_m=float(x_m),
                y_m=float(y_m),
                fallback_lane_id=int(fallback_lane_id),
                ego_waypoint=ego_waypoint,
            )
        if len(self._carla_route_nodes()) >= 2:
            return self._carla_route_info(
                x_m=float(x_m),
                y_m=float(y_m),
                fallback_lane_id=int(fallback_lane_id),
                ego_waypoint=ego_waypoint,
            )
        try:
            summary = self.global_planner.get_current_route_info(
                x_m=float(x_m),
                y_m=float(y_m),
                query_key=str(query_key),
            )
        except Exception as exc:
            self._last_status = RouteManagerStatus(
                route_found=False,
                route_point_count=0,
                remaining_distance_m=0.0,
                reached_destination=False,
                debug_reason=f"route_query_failed:{exc}",
            )
            return self._fallback_summary(
                fallback_lane_id=int(fallback_lane_id),
                debug_reason=str(self._last_status.debug_reason),
            )

        if summary is None:
            summary = self._active_route_summary
        if summary is None:
            self._last_status = RouteManagerStatus(debug_reason="route_missing")
            return self._fallback_summary(
                fallback_lane_id=int(fallback_lane_id),
                debug_reason="route_missing",
            )

        self._active_route_summary = summary
        self._last_status = self._status_from_summary(summary)
        if (
            not bool(getattr(summary, "route_found", False))
            and len(self._fallback_route_points) >= 2
        ):
            remaining = self._remaining_distance_on_fallback_route(
                x_m=float(x_m),
                y_m=float(y_m),
            )
            self._last_status = RouteManagerStatus(
                route_found=True,
                route_point_count=len(self._fallback_route_points),
                remaining_distance_m=float(remaining),
                reached_destination=bool(remaining <= self.reached_distance_m),
                debug_reason=self._fallback_debug_reason(summary),
            )
            return {
                "route_found": True,
                "optimal_lane_id": int(fallback_lane_id),
                "current_road_option": "FALLBACK_DIRECT",
                "next_macro_maneuver": "Continue Straight",
                "debug_reason": str(self._last_status.debug_reason),
                "remaining_distance_m": float(remaining),
                "reached_destination": bool(self._last_status.reached_destination),
            }
        lane_id = _to_int(getattr(summary, "optimal_lane_id", fallback_lane_id), fallback_lane_id)
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
            "remaining_distance_m": float(
                getattr(summary, "distance_to_destination_m", 0.0) or 0.0
            ),
            "reached_destination": bool(self._last_status.reached_destination),
        }

    def _carla_route_info(
        self,
        *,
        x_m: float,
        y_m: float,
        fallback_lane_id: int,
        ego_waypoint: Any = None,
    ) -> Dict[str, object]:
        nodes = self._carla_route_nodes()
        if len(nodes) < 2:
            return self._fallback_summary(
                fallback_lane_id=int(fallback_lane_id),
                debug_reason="carla_route_empty",
            )
        if self._carla_route_projection is None:
            initial_heading_rad = math.atan2(
                float(nodes[1][1]) - float(nodes[0][1]),
                float(nodes[1][0]) - float(nodes[0][0]),
            )
            self.sync_carla_route_progress(
                ego_x_m=float(x_m),
                ego_y_m=float(y_m),
                ego_heading_rad=float(initial_heading_rad),
            )
        index = min(max(0, int(self._carla_route_progress_index)), len(nodes) - 2)
        options = [str(node[4]) for node in nodes[index + 1 : index + 81]]
        current_option = str(nodes[min(index + 1, len(nodes) - 1)][4])
        next_macro = _next_macro_from_carla_options(options)
        next_macro_distance_m = _distance_to_next_carla_macro(
            nodes=nodes,
            start_index=int(index),
            projection=self._carla_route_projection,
        )
        optimal_lane_id = _route_required_carla_lane_id(
            nodes=nodes,
            start_index=int(index),
            fallback_lane_id=int(fallback_lane_id),
            ego_waypoint=ego_waypoint,
        )
        remaining = self._remaining_distance_on_carla_route(start_index=int(index))
        reached = bool(remaining <= self.reached_distance_m)
        self._last_status = RouteManagerStatus(
            route_found=True,
            route_point_count=len(nodes),
            remaining_distance_m=float(remaining),
            reached_destination=bool(reached),
            debug_reason="carla_grp_route_active",
        )
        return {
            "route_found": True,
            "optimal_lane_id": int(optimal_lane_id),
            "current_road_option": str(current_option),
            "next_macro_maneuver": str(next_macro),
            "next_macro_distance_m": next_macro_distance_m,
            "debug_reason": "carla_grp_route_active",
            "remaining_distance_m": float(remaining),
            "reached_destination": bool(reached),
        }

    def _external_route_info(
        self,
        *,
        x_m: float,
        y_m: float,
        fallback_lane_id: int,
        ego_waypoint: Any = None,
    ) -> Dict[str, object]:
        del ego_waypoint  # Reserved: this path reports ego's own route-index
        # lane, not a maneuver point ahead, so it doesn't have the
        # cross-point mismatch _route_required_carla_lane_id fixes.
        nodes = self._carla_route_nodes()
        if len(nodes) < 2:
            return self._fallback_summary(
                fallback_lane_id=int(fallback_lane_id),
                debug_reason="external_route_empty",
            )
        if self._carla_route_projection is None:
            initial_heading_rad = math.atan2(
                float(nodes[1][1]) - float(nodes[0][1]),
                float(nodes[1][0]) - float(nodes[0][0]),
            )
            self.sync_carla_route_progress(
                ego_x_m=float(x_m),
                ego_y_m=float(y_m),
                ego_heading_rad=float(initial_heading_rad),
            )
        index = min(
            max(0, int(self._carla_route_progress_index)),
            len(nodes) - 2,
        )
        waypoint = nodes[index][3]
        lane_id = int(_canonical_carla_lane_id(waypoint, fallback_lane_id))
        options = [str(node[4]) for node in nodes[index + 1 : index + 81]]
        current_option = str(nodes[min(index + 1, len(nodes) - 1)][4])
        next_macro = _next_macro_from_carla_options(options)
        next_macro_distance_m = _distance_to_next_carla_macro(
            nodes=nodes,
            start_index=int(index),
            projection=self._carla_route_projection,
        )
        remaining = self._remaining_distance_on_fallback_route(
            x_m=float(x_m),
            y_m=float(y_m),
        )
        reached = bool(remaining <= self.reached_distance_m)
        self._last_status = RouteManagerStatus(
            route_found=True,
            route_point_count=len(nodes),
            remaining_distance_m=float(remaining),
            reached_destination=bool(reached),
            debug_reason="external_leaderboard_route_active",
        )
        return {
            "route_found": True,
            "optimal_lane_id": int(lane_id),
            "current_road_option": str(current_option),
            "next_macro_maneuver": str(next_macro),
            "next_macro_distance_m": next_macro_distance_m,
            "debug_reason": "external_leaderboard_route_active",
            "remaining_distance_m": float(remaining),
            "reached_destination": bool(reached),
        }

    def _remaining_distance_on_carla_route(self, *, start_index: int) -> float:
        nodes = self._carla_route_nodes()
        if len(nodes) < 2:
            return 0.0
        index = min(max(0, int(start_index)), len(nodes) - 2)
        if self._carla_route_projection is not None:
            previous_xy = (
                float(self._carla_route_projection[1]),
                float(self._carla_route_projection[2]),
            )
        else:
            previous_xy = (float(nodes[index][0]), float(nodes[index][1]))
        remaining_m = 0.0
        for node in nodes[index + 1 :]:
            current_xy = (float(node[0]), float(node[1]))
            remaining_m += math.hypot(
                current_xy[0] - previous_xy[0],
                current_xy[1] - previous_xy[1],
            )
            previous_xy = current_xy
        return float(remaining_m)

    def route_points(self, *, x_m: Optional[float] = None, y_m: Optional[float] = None, query_key: str = "") -> List[List[float]]:
        summary = None
        if x_m is not None and y_m is not None:
            try:
                summary = self.global_planner.get_current_route_info(
                    x_m=float(x_m),
                    y_m=float(y_m),
                    query_key=str(query_key or "route_points"),
                )
            except Exception:
                summary = None
        if summary is None:
            summary = self._active_route_summary
        route_waypoints = list(getattr(summary, "route_waypoints", []) or [])
        route_points = _route_points_from_waypoints(route_waypoints)
        if len(route_points) >= 2:
            return route_points
        return [list(point) for point in list(self._fallback_route_points or [])]

    def geometry_route_points(
        self,
        *,
        x_m: Optional[float] = None,
        y_m: Optional[float] = None,
        query_key: str = "",
    ) -> List[List[float]]:
        """Return the route geometry shared by reference generation and debug.

        The custom global planner remains responsible for route topology and
        maneuver semantics.  When available, CARLA GRP waypoints are the
        geometric source of truth because they follow the simulator lane
        center and include the selected junction connector.
        """

        nodes = self._carla_route_nodes()
        if len(nodes) >= 2:
            cache_key = self._carla_route_entries_cache_key()
            if self._geometry_route_points_cache_key == cache_key:
                return [list(point) for point in self._geometry_route_points_cache]
            points: List[List[float]] = []
            for index, node in enumerate(nodes):
                if index + 1 < len(nodes):
                    next_node = nodes[index + 1]
                    heading_rad = math.atan2(
                        float(next_node[1]) - float(node[1]),
                        float(next_node[0]) - float(node[0]),
                    )
                elif points:
                    heading_rad = float(points[-1][3])
                else:
                    heading_rad = 0.0
                points.append([
                    float(node[0]),
                    float(node[1]),
                    float(node[2]),
                    float(heading_rad),
                ])
            self._geometry_route_points_cache_key = cache_key
            self._geometry_route_points_cache = tuple(tuple(float(value) for value in point) for point in points)
            return points
        return self.route_points(x_m=x_m, y_m=y_m, query_key=query_key)

    def upcoming_turn(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        lookahead_m: float,
    ) -> Tuple[str, float, str]:
        """Return the first CARLA GRP turn option within the lookahead."""

        sync_reason = self.sync_carla_route_progress(
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            ego_heading_rad=float(ego_heading_rad),
        )
        nodes = self._carla_route_nodes()
        if len(nodes) < 2 or self._carla_route_projection is None:
            return "", float("inf"), str(sync_reason)

        segment_index, projection_x_m, projection_y_m, _, lateral_m = (
            self._carla_route_projection
        )
        if float(lateral_m) > float(self.stale_route_lateral_m):
            return "", float("inf"), str(sync_reason)

        distance_m = 0.0
        previous_xy = (float(projection_x_m), float(projection_y_m))
        limit_m = max(0.0, float(lookahead_m))
        baseline_heading_rad = float(ego_heading_rad)
        geometry_turn: Optional[Tuple[str, float]] = None
        lane_change_precedes_turn = False
        for node in nodes[int(segment_index) + 1 :]:
            current_xy = (float(node[0]), float(node[1]))
            segment_length_m = math.hypot(
                current_xy[0] - previous_xy[0],
                current_xy[1] - previous_xy[1],
            )
            if segment_length_m > 1.0e-3:
                segment_heading_rad = math.atan2(
                    current_xy[1] - previous_xy[1],
                    current_xy[0] - previous_xy[0],
                )
                if float(distance_m) <= 1.0e-6:
                    baseline_heading_rad = float(segment_heading_rad)
                heading_change_rad = _wrap_angle(
                    float(segment_heading_rad) - float(baseline_heading_rad)
                )
                if (
                    geometry_turn is None
                    and abs(float(heading_change_rad))
                    >= float(self.geometry_turn_min_heading_change_rad)
                ):
                    geometry_turn = (
                        "left" if float(heading_change_rad) > 0.0 else "right",
                        float(distance_m),
                    )
            distance_m += float(segment_length_m)
            option = str(node[4] or "").strip().upper()
            if option in {"CHANGELANELEFT", "CHANGELANERIGHT"}:
                # A GRP lane-change connector can easily exceed the heading
                # threshold used by the geometry fallback. It is a lateral
                # maneuver, not an intersection turn.
                lane_change_precedes_turn = True
                geometry_turn = None
            if option in {"LEFT", "RIGHT"}:
                if float(distance_m) <= float(limit_m):
                    return option.lower(), float(distance_m), "carla_route_turn_ahead"
                break
            if float(distance_m) > float(limit_m):
                break
            previous_xy = current_xy
        if bool(lane_change_precedes_turn):
            return "", float("inf"), "carla_route_lane_change_precedes_turn"
        if geometry_turn is not None and float(geometry_turn[1]) <= float(limit_m):
            return (
                str(geometry_turn[0]),
                float(geometry_turn[1]),
                "carla_route_geometry_turn_ahead",
            )
        return "", float("inf"), "carla_route_no_turn_in_lookahead"

    def carla_route_alignment(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        heading_lookahead_m: float = 5.0,
    ) -> Tuple[float, float, str]:
        """Return heading error and lateral distance to the active route."""

        sync_reason = self.sync_carla_route_progress(
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            ego_heading_rad=float(ego_heading_rad),
        )
        nodes = self._carla_route_nodes()
        if len(nodes) < 2 or self._carla_route_projection is None:
            return float("inf"), float("inf"), str(sync_reason)
        segment_index, _, _, _, lateral_m = self._carla_route_projection
        heading_index = min(int(segment_index), len(nodes) - 2)
        remaining_lookahead_m = max(0.0, float(heading_lookahead_m))
        previous_xy = (
            float(self._carla_route_projection[1]),
            float(self._carla_route_projection[2]),
        )
        for index in range(int(segment_index) + 1, len(nodes)):
            current_xy = (float(nodes[index][0]), float(nodes[index][1]))
            segment_m = math.hypot(
                current_xy[0] - previous_xy[0],
                current_xy[1] - previous_xy[1],
            )
            heading_index = min(max(0, index - 1), len(nodes) - 2)
            if float(segment_m) >= float(remaining_lookahead_m):
                break
            remaining_lookahead_m -= float(segment_m)
            previous_xy = current_xy
        first = nodes[int(heading_index)]
        second = nodes[min(int(heading_index) + 1, len(nodes) - 1)]
        route_heading_rad = math.atan2(
            float(second[1]) - float(first[1]),
            float(second[0]) - float(first[0]),
        )
        heading_error_rad = _wrap_angle(
            float(route_heading_rad) - float(ego_heading_rad)
        )
        return (
            float(heading_error_rad),
            float(lateral_m),
            "carla_route_alignment:" + str(sync_reason),
        )

    def carla_waypoint_reference(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        horizon_steps: int,
        step_distance_m: float,
        target_speed_mps: float,
        fallback_lane_id: int,
        anchor_to_ego_heading: bool = False,
        ego_anchor_distance_m: Optional[float] = None,
        allow_route_rejoin: bool = True,
        max_extrapolation_m: float = float("inf"),
        extrapolated_lane_width_m: Optional[float] = None,
    ) -> Tuple[List[Dict[str, object]], str]:
        """Sample a local reference from the CARLA GRP waypoint chain.

        The chain already contains the route-selected junction connector. Route
        progress is monotonic, so a nearby crossing or adjacent connector cannot
        make the local reference jump backward to another branch.

        Once the requested lookahead exceeds the real remaining polyline
        length, samples fall back to extrapolating straight ahead in the last
        real segment's heading -- fine when there's plenty of real route left
        (the common case), but on a short segment (e.g. an intersection turn
        connector) a long horizon can extrapolate far past the real curve.
        max_extrapolation_m bounds how far that fallback is allowed to drift
        past the real geometry; the default preserves today's unbounded
        behavior. extrapolated_lane_width_m, when given, overrides the lane
        width reported for those same extrapolated samples (which otherwise
        inherit the last real waypoint's own width) -- lets a caller loosen
        the road-boundary corridor for a tail it has no real geometry for,
        instead of forcing a normal-width corridor onto a parked/uncertain
        position. Real samples are never affected by either parameter.
        """

        sync_reason = self.sync_carla_route_progress(
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            ego_heading_rad=float(ego_heading_rad),
        )
        nodes = self._carla_route_nodes()
        if len(nodes) < 2:
            return [], str(sync_reason or "carla_route_waypoints_empty")
        if self._carla_route_projection is None:
            return [], str(sync_reason or "carla_route_projection_failed")
        (
            segment_index,
            projection_x_m,
            projection_y_m,
            projection_ratio,
            lateral_distance_m,
        ) = self._carla_route_projection
        if float(lateral_distance_m) > float(self.stale_route_lateral_m):
            return [], str(sync_reason)
        first = nodes[segment_index]
        second = nodes[min(segment_index + 1, len(nodes) - 1)]

        polyline: List[Tuple[float, float, float, Any, str]] = [
            (
                float(projection_x_m),
                float(projection_y_m),
                float(first[2]) + float(projection_ratio) * (float(second[2]) - float(first[2])),
                second[3],
                second[4],
            )
        ]
        polyline.extend(nodes[segment_index + 1 :])
        polyline = _deduplicate_carla_nodes(polyline)
        if len(polyline) < 2:
            return [], "carla_route_remaining_polyline_too_short"

        cumulative = [0.0]
        for previous, current in zip(polyline[:-1], polyline[1:]):
            cumulative.append(
                cumulative[-1]
                + math.hypot(current[0] - previous[0], current[1] - previous[1])
            )
        step_m = max(0.25, float(step_distance_m))
        samples: List[Dict[str, object]] = []
        sample_segment = 0
        for sample_index in range(max(1, int(horizon_steps))):
            target_s_m = float(sample_index + 1) * float(step_m)
            while sample_segment + 1 < len(cumulative) and cumulative[sample_segment + 1] < target_s_m:
                sample_segment += 1
            if sample_segment + 1 < len(polyline):
                node_a = polyline[sample_segment]
                node_b = polyline[sample_segment + 1]
                segment_length_m = max(
                    1.0e-6, cumulative[sample_segment + 1] - cumulative[sample_segment]
                )
                ratio = min(
                    1.0,
                    max(0.0, (target_s_m - cumulative[sample_segment]) / segment_length_m),
                )
                x_m = node_a[0] + ratio * (node_b[0] - node_a[0])
                y_m = node_a[1] + ratio * (node_b[1] - node_a[1])
                heading_rad = math.atan2(node_b[1] - node_a[1], node_b[0] - node_a[0])
                waypoint = node_b[3]
                option = node_b[4]
                is_extrapolated = False
            else:
                node_a = polyline[-2]
                node_b = polyline[-1]
                heading_rad = math.atan2(node_b[1] - node_a[1], node_b[0] - node_a[0])
                extra_m = max(0.0, target_s_m - cumulative[-1])
                extra_m = min(float(extra_m), float(max_extrapolation_m))
                x_m = node_b[0] + extra_m * math.cos(heading_rad)
                y_m = node_b[1] + extra_m * math.sin(heading_rad)
                waypoint = node_b[3]
                option = node_b[4]
                is_extrapolated = True
            normalized_option = str(option or "").strip().upper().replace("_", "")
            lane_transition_kind = (
                "lateral_lane_change"
                if normalized_option in {"CHANGELANELEFT", "CHANGELANERIGHT"}
                else "longitudinal_successor"
            )
            if is_extrapolated and extrapolated_lane_width_m is not None:
                lane_width_m = float(extrapolated_lane_width_m)
            else:
                lane_width_m = float(getattr(waypoint, "lane_width_m", getattr(waypoint, "lane_width", 3.5)) or 3.5)
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(heading_rad),
                "lane_id": int(_canonical_carla_lane_id(waypoint, fallback_lane_id)),
                "lane_width_m": float(lane_width_m),
                "road_id": int(getattr(waypoint, "road_id", 0) or 0),
                "road_option": str(option),
                "lane_transition_kind": str(lane_transition_kind),
                "speed_ref_mps": max(0.0, float(target_speed_mps)),
                "v_ref_mps": max(0.0, float(target_speed_mps)),
                "speed_mps": max(0.0, float(target_speed_mps)),
                "corridor_center_x_m": float(x_m),
                "corridor_center_y_m": float(y_m),
                "corridor_heading_rad": float(heading_rad),
            })
        samples = _smooth_carla_reference_samples(
            samples,
            passes=int(self.carla_reference_smoothing_passes),
            boundary_aware=bool(self.carla_reference_boundary_aware),
            vehicle_half_width_m=float(
                self.carla_reference_vehicle_half_width_m
            ),
            boundary_margin_m=float(
                self.carla_reference_boundary_margin_m
            ),
            tracking_reserve_m=float(
                self.carla_reference_tracking_reserve_m
            ),
        )
        reason = "carla_grp_waypoint_chain_smoothed"
        if bool(anchor_to_ego_heading):
            samples = _apply_ego_heading_connector(
                samples,
                ego_x_m=float(ego_x_m),
                ego_y_m=float(ego_y_m),
                ego_heading_rad=float(ego_heading_rad),
                step_distance_m=float(step_m),
                rejoin_distance_m=float(
                    ego_anchor_distance_m
                    if ego_anchor_distance_m is not None
                    else self.carla_rejoin_distance_m
                ),
                smoothing_passes=int(self.carla_turn_connector_smoothing_passes),
                # The ego-heading connector may begin outside the contracted
                # route corridor. Projecting its early samples onto unrelated
                # route stations breaks tangent continuity; its final geometry
                # is still checked by the swept-footprint gate.
                boundary_aware=False,
                vehicle_half_width_m=float(
                    self.carla_reference_vehicle_half_width_m
                ),
                boundary_margin_m=float(
                    self.carla_reference_boundary_margin_m
                ),
                tracking_reserve_m=float(
                    self.carla_reference_tracking_reserve_m
                ),
            )
            reason += ":ego_heading_connector"
        elif (
            bool(allow_route_rejoin)
            and
            float(lateral_distance_m) >= float(self.carla_rejoin_min_lateral_m)
            and float(lateral_distance_m) <= float(self.carla_rejoin_max_lateral_m)
        ):
            samples = _apply_route_rejoin_offset(
                samples,
                ego_x_m=float(ego_x_m),
                ego_y_m=float(ego_y_m),
                projection_x_m=float(projection_x_m),
                projection_y_m=float(projection_y_m),
                step_distance_m=float(step_m),
                rejoin_distance_m=float(self.carla_rejoin_distance_m),
            )
            reason += ":route_rejoin"
        return samples, reason

    def sync_carla_route_progress(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
    ) -> str:
        """Synchronize ego progress against the CARLA waypoint route.

        The first call searches the complete route. Later calls use a bounded
        forward window and preserve monotonic progress.
        """

        nodes = self._carla_route_nodes()
        if len(nodes) < 2:
            self._carla_route_projection = None
            self._carla_route_sync_reason = str(
                self._carla_route_debug_reason or "carla_route_unavailable"
            )
            return str(self._carla_route_sync_reason)

        if not bool(self._carla_route_progress_initialized):
            lower = 0
            upper = len(nodes) - 1
            search_mode = "global_init"
        else:
            lower = max(0, int(self._carla_route_progress_index) - 5)
            upper = min(
                len(nodes) - 1,
                max(lower + 1, int(self._carla_route_progress_index) + 80),
            )
            search_mode = "local_update"

        best = _best_carla_route_projection(
            nodes=nodes,
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            ego_heading_rad=float(ego_heading_rad),
            lower_index=int(lower),
            upper_index=int(upper),
        )
        if best is None:
            self._carla_route_sync_reason = f"carla_route_progress_{search_mode}_failed"
            return str(self._carla_route_sync_reason)

        _, _, best_index, _, _, _ = best
        segment_index = (
            int(best_index)
            if not bool(self._carla_route_progress_initialized)
            else max(int(self._carla_route_progress_index), int(best_index))
        )
        first = nodes[segment_index]
        second = nodes[min(segment_index + 1, len(nodes) - 1)]
        projection_x_m, projection_y_m, projection_ratio, lateral_distance_m = (
            _project_to_segment(
                x_m=float(ego_x_m),
                y_m=float(ego_y_m),
                first_xy=(first[0], first[1]),
                second_xy=(second[0], second[1]),
            )
        )
        self._carla_route_progress_index = int(segment_index)
        self._carla_route_progress_initialized = True
        self._carla_route_projection = (
            int(segment_index),
            float(projection_x_m),
            float(projection_y_m),
            float(projection_ratio),
            float(lateral_distance_m),
        )
        if float(lateral_distance_m) > float(self.stale_route_lateral_m):
            self._carla_route_sync_reason = (
                f"carla_route_stale:lateral={float(lateral_distance_m):.2f}"
            )
        else:
            self._carla_route_sync_reason = (
                f"carla_route_progress_{search_mode}:index={int(segment_index)}"
            )
        return str(self._carla_route_sync_reason)

    def _carla_route_nodes(self) -> List[Tuple[float, float, float, Any, str]]:
        cache_key = self._carla_route_entries_cache_key()
        if self._carla_route_nodes_cache_key == cache_key:
            return list(self._carla_route_nodes_cache)
        nodes: List[Tuple[float, float, float, Any, str]] = []
        for entry in list(self._carla_route_entries or []):
            waypoint, option = _carla_route_entry(entry)
            position = getattr(waypoint, "position", None)
            if not isinstance(position, Mapping):
                continue
            x_m = float(position["x"])
            y_m = float(position["y"])
            z_m = float(position.get("z", 0.0))
            if nodes and math.hypot(x_m - nodes[-1][0], y_m - nodes[-1][1]) < 1.0e-3:
                continue
            nodes.append((x_m, y_m, z_m, waypoint, _road_option_name(option)))
        self._carla_route_nodes_cache_key = cache_key
        self._carla_route_nodes_cache = tuple(nodes)
        return nodes

    def _carla_route_entries_cache_key(self) -> Tuple[int, int, int, int]:
        """Identify the immutable route-entry snapshot used by geometry caches."""
        entries = self._carla_route_entries
        return (id(entries), len(entries), id(entries[0]) if entries else 0, id(entries[-1]) if entries else 0)

    @property
    def carla_route_debug_reason(self) -> str:
        return str(self._carla_route_debug_reason)

    @property
    def carla_route_sync_reason(self) -> str:
        return str(self._carla_route_sync_reason)

    @property
    def carla_route_progress_index(self) -> int:
        return int(self._carla_route_progress_index)

    @property
    def last_status(self) -> RouteManagerStatus:
        return self._last_status

    @property
    def active_route_summary(self) -> Any:
        return self._active_route_summary

    def accept_authoritative_route_summary(
        self,
        summary: Any,
        *,
        debug_reason: str = "authoritative_route_summary",
    ) -> RouteManagerStatus:
        """Publish one route backend's per-tick progress as manager state.

        The custom AD-map path queries its route directly because AD lane
        identities and maneuver semantics must not be inferred from the CARLA
        geometry route.  Publishing that same query here keeps diagnostics,
        destination completion, and behavior input on one progress source.
        CARLA route progress remains available only for geometry sampling.
        """

        def value(name: str, default: Any = None) -> Any:
            if isinstance(summary, Mapping):
                return summary.get(name, default)
            return getattr(summary, name, default)

        route_found = bool(value("route_found", False))
        remaining = max(
            0.0,
            float(
                value(
                    "distance_to_destination_m",
                    value("remaining_distance_m", 0.0),
                )
                or 0.0
            ),
        )
        route_waypoints = list(value("route_waypoints", []) or [])
        route_point_count = max(
            len(route_waypoints),
            len(self._fallback_route_points),
        )
        reached = bool(route_found and remaining <= self.reached_distance_m)
        self._last_status = RouteManagerStatus(
            route_found=route_found,
            route_point_count=int(route_point_count),
            remaining_distance_m=float(remaining),
            reached_destination=reached,
            debug_reason=str(debug_reason),
        )
        return self._last_status

    def _build_carla_route(
        self,
        *,
        start_point: Mapping[str, object],
        goal_point: Mapping[str, object],
    ) -> None:
        """Populate the existing route interface from the custom AD-map route."""

        del start_point, goal_point
        self._build_carla_route_from_summary(self._active_route_summary)

    def _build_carla_route_from_summary(self, summary: Any) -> None:
        """Populate the route interface from one custom AD-map route summary."""

        self._carla_route_entries = []
        self._carla_route_progress_index = 0
        self._carla_route_progress_initialized = False
        self._carla_route_projection = None
        self._carla_route_sync_reason = "custom_route_progress_not_initialized"
        route_points = list(getattr(summary, "route_waypoints", []) or [])
        road_options = list(getattr(summary, "road_options", []) or [])
        for index, point in enumerate(route_points):
            if len(point) < 2:
                continue
            waypoint = self.global_planner.get_waypoint({"x": float(point[0]), "y": float(point[1]), "z": float(point[2]) if len(point) > 2 else 0.0})
            if waypoint is None:
                continue
            option = road_options[index] if index < len(road_options) else "LANEFOLLOW"
            self._carla_route_entries.append((waypoint, option))
        self._carla_route_debug_reason = "custom_admap_route_ready" if len(self._carla_route_entries) >= 2 else "custom_admap_route_empty"

    def _status_from_summary(self, summary: Any) -> RouteManagerStatus:
        route_points = list(getattr(summary, "route_waypoints", []) or [])
        remaining = float(getattr(summary, "distance_to_destination_m", 0.0) or 0.0)
        route_found = bool(getattr(summary, "route_found", False))
        return RouteManagerStatus(
            route_found=bool(route_found) or len(self._fallback_route_points) >= 2,
            route_point_count=max(len(route_points), len(self._fallback_route_points)),
            remaining_distance_m=float(
                remaining
                if bool(route_found)
                else self._fallback_route_total_distance_m()
            ),
            reached_destination=bool(
                (bool(route_found) and remaining <= self.reached_distance_m)
                or (
                    not bool(route_found)
                    and self._fallback_route_total_distance_m() <= self.reached_distance_m
                )
            ),
            debug_reason=self._fallback_debug_reason(summary)
            if not bool(route_found)
            else str(getattr(summary, "debug_reason", "route_active")),
        )

    @staticmethod
    def _fallback_summary(*, fallback_lane_id: int, debug_reason: str) -> Dict[str, object]:
        return {
            "route_found": False,
            "optimal_lane_id": int(fallback_lane_id),
            "current_road_option": "",
            "next_macro_maneuver": "Continue Straight",
            "debug_reason": str(debug_reason),
            "remaining_distance_m": 0.0,
            "reached_destination": False,
        }

    @staticmethod
    def _direct_fallback_route_points(
        *,
        start_point: Mapping[str, object],
        goal_point: Mapping[str, object],
    ) -> List[List[float]]:
        start = _point_dict(start_point)
        goal = _point_dict(goal_point)
        heading = math.atan2(float(goal["y"]) - float(start["y"]), float(goal["x"]) - float(start["x"]))
        return [
            [float(start["x"]), float(start["y"]), float(start.get("z", 0.0)), float(heading)],
            [float(goal["x"]), float(goal["y"]), float(goal.get("z", 0.0)), float(heading)],
        ]

    def _fallback_route_total_distance_m(self) -> float:
        if len(self._fallback_route_points) < 2:
            return 0.0
        first = self._fallback_route_points[0]
        last = self._fallback_route_points[-1]
        return math.hypot(float(last[0]) - float(first[0]), float(last[1]) - float(first[1]))

    def _remaining_distance_on_fallback_route(self, *, x_m: float, y_m: float) -> float:
        if len(self._fallback_route_points) < 2:
            return 0.0
        goal = self._fallback_route_points[-1]
        return math.hypot(float(goal[0]) - float(x_m), float(goal[1]) - float(y_m))

    @staticmethod
    def _fallback_debug_reason(summary: Any) -> str:
        raw_reason = str(getattr(summary, "debug_reason", "") or "").strip()
        if raw_reason:
            return f"global_route_failed_direct_fallback:{raw_reason}"
        return "global_route_failed_direct_fallback"


def _point_dict(point: Mapping[str, object]) -> Dict[str, float]:
    return {
        "x": float(point.get("x", point.get("x_m", 0.0))),
        "y": float(point.get("y", point.get("y_m", 0.0))),
        "z": float(point.get("z", point.get("z_m", 0.0))),
    }


def _route_points_from_waypoints(route_waypoints: List[Any]) -> List[List[float]]:
    points: List[List[float]] = []
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
        if points and math.hypot(points[-1][0] - x_m, points[-1][1] - y_m) < 1.0e-3:
            continue
        points.append([float(x_m), float(y_m), float(z_m), float(heading_rad)])
    return points


def _carla_route_entry(entry: Any) -> Tuple[Any, Any]:
    if isinstance(entry, (list, tuple)) and entry:
        return entry[0], entry[1] if len(entry) >= 2 else None
    return entry, None


def _road_option_name(option: Any) -> str:
    if option is None:
        return ""
    name = getattr(option, "name", None)
    if name is not None:
        return str(name).strip().upper()
    text = str(option).strip()
    return text.rsplit(".", 1)[-1].upper() if "." in text else text.upper()


def _canonical_carla_lane_id(waypoint: Any, fallback_lane_id: int) -> int:
    try:
        from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint

        lane_id = int(canonical_lane_id_for_waypoint(waypoint))
        if lane_id != 0:
            return lane_id
    except Exception:
        pass
    return int(fallback_lane_id)


def _project_to_segment(
    *,
    x_m: float,
    y_m: float,
    first_xy: Sequence[float],
    second_xy: Sequence[float],
) -> Tuple[float, float, float, float]:
    dx_m = float(second_xy[0]) - float(first_xy[0])
    dy_m = float(second_xy[1]) - float(first_xy[1])
    length_sq = dx_m * dx_m + dy_m * dy_m
    if length_sq <= 1.0e-9:
        ratio = 0.0
    else:
        ratio = (
            (float(x_m) - float(first_xy[0])) * dx_m
            + (float(y_m) - float(first_xy[1])) * dy_m
        ) / length_sq
        ratio = min(1.0, max(0.0, float(ratio)))
    px_m = float(first_xy[0]) + ratio * dx_m
    py_m = float(first_xy[1]) + ratio * dy_m
    return px_m, py_m, ratio, math.hypot(float(x_m) - px_m, float(y_m) - py_m)


def select_route_aligned_waypoint_candidate(
    *,
    candidates: Sequence[Any],
    route_points: Sequence[Sequence[float]] | None,
    previous_heading_rad: float,
    turn_direction: str = "",
    max_route_distance_m: float = 5.0,
) -> Any:
    """Select one topology successor against the local route polyline.

    Distance is measured to route segments rather than sparse route points.
    Route tangent and heading continuity disambiguate nearby fork branches;
    an explicit turn direction is only a prior, never the primary route.
    """

    route_xy = [
        (float(point[0]), float(point[1]))
        for point in list(route_points or [])
        if len(point) >= 2
    ]
    if not candidates or len(route_xy) < 2:
        return None

    direction = str(turn_direction or "").strip().lower()
    scored = []
    for candidate in candidates:
        transform = getattr(candidate, "transform", None)
        location = getattr(transform, "location", None)
        rotation = getattr(transform, "rotation", None)
        if location is None:
            continue
        candidate_heading = math.radians(
            float(getattr(rotation, "yaw", math.degrees(previous_heading_rad)))
        )
        best_projection = None
        for segment_index, (first, second) in enumerate(
            zip(route_xy[:-1], route_xy[1:])
        ):
            px_m, py_m, ratio, distance_m = _project_to_segment(
                x_m=float(location.x),
                y_m=float(location.y),
                first_xy=first,
                second_xy=second,
            )
            tangent = math.atan2(
                float(second[1]) - float(first[1]),
                float(second[0]) - float(first[0]),
            )
            candidate_projection = (
                float(distance_m),
                int(segment_index),
                float(ratio),
                float(px_m),
                float(py_m),
                float(tangent),
            )
            if best_projection is None or candidate_projection < best_projection:
                best_projection = candidate_projection
        if best_projection is None:
            continue
        route_distance_m = float(best_projection[0])
        route_heading_error = abs(
            _wrap_angle(float(candidate_heading) - float(best_projection[5]))
        )
        heading_delta = _wrap_angle(
            float(candidate_heading) - float(previous_heading_rad)
        )
        direction_penalty = 0.0
        if direction == "left" and float(heading_delta) < 0.0:
            direction_penalty = 2.0 * abs(float(heading_delta))
        elif direction == "right" and float(heading_delta) > 0.0:
            direction_penalty = 2.0 * abs(float(heading_delta))
        score = (
            float(route_distance_m)
            + 0.75 * float(route_heading_error)
            + 0.35 * abs(float(heading_delta))
            + float(direction_penalty)
        )
        scored.append(
            (
                float(score),
                float(route_distance_m),
                float(route_heading_error),
                candidate,
            )
        )
    if not scored:
        return None
    _, route_distance_m, route_heading_error, selected = min(
        scored,
        key=lambda item: (item[0], item[1], item[2]),
    )
    if float(route_distance_m) > max(0.1, float(max_route_distance_m)):
        return None
    if float(route_heading_error) > math.radians(100.0):
        return None
    return selected


def _turn_direction_from_option(option: Any) -> str:
    normalized = _road_option_name(option).replace("_", "")
    if normalized == "LEFT":
        return "left"
    if normalized == "RIGHT":
        return "right"
    return ""


def _best_carla_route_projection(
    *,
    nodes: Sequence[Tuple[float, float, float, Any, str]],
    ego_x_m: float,
    ego_y_m: float,
    ego_heading_rad: float,
    lower_index: int,
    upper_index: int,
) -> Optional[Tuple[float, float, int, float, float, float]]:
    best = None
    lower = max(0, int(lower_index))
    upper = min(len(nodes) - 1, int(upper_index))
    for index in range(lower, upper):
        first = nodes[index]
        second = nodes[index + 1]
        px_m, py_m, ratio, distance_m = _project_to_segment(
            x_m=float(ego_x_m),
            y_m=float(ego_y_m),
            first_xy=(first[0], first[1]),
            second_xy=(second[0], second[1]),
        )
        segment_heading = math.atan2(second[1] - first[1], second[0] - first[0])
        heading_error = abs(_wrap_angle(segment_heading - float(ego_heading_rad)))
        opposite_penalty = 25.0 if heading_error > 0.75 * math.pi else 0.0
        score = float(distance_m) + float(opposite_penalty) + 0.2 * float(heading_error)
        candidate = (
            float(score),
            float(distance_m),
            int(index),
            float(px_m),
            float(py_m),
            float(ratio),
        )
        if best is None or candidate < best:
            best = candidate
    return best


def _deduplicate_carla_nodes(
    nodes: Sequence[Tuple[float, float, float, Any, str]],
) -> List[Tuple[float, float, float, Any, str]]:
    result: List[Tuple[float, float, float, Any, str]] = []
    for node in nodes:
        if result and math.hypot(node[0] - result[-1][0], node[1] - result[-1][1]) < 1.0e-3:
            continue
        result.append(node)
    return result


def _smooth_carla_reference_samples(
    samples: Sequence[Mapping[str, object]],
    *,
    passes: int,
    boundary_aware: bool = False,
    vehicle_half_width_m: float = 1.0,
    boundary_margin_m: float = 0.15,
    tracking_reserve_m: float = 0.20,
) -> List[Dict[str, object]]:
    """Smooth a CARLA connector and project it into its route-owned corridor."""

    result = [dict(sample) for sample in list(samples or [])]
    if len(result) < 3:
        return _recompute_reference_headings(result)

    if bool(boundary_aware):
        for sample in result:
            try:
                corridor_heading_rad = float(
                    sample.get(
                        "corridor_heading_rad",
                        sample.get("heading_rad", 0.0),
                    )
                )
                dx_m = float(sample["x_ref_m"]) - float(
                    sample.get(
                        "corridor_center_x_m",
                        sample["x_ref_m"],
                    )
                )
                dy_m = float(sample["y_ref_m"]) - float(
                    sample.get(
                        "corridor_center_y_m",
                        sample["y_ref_m"],
                    )
                )
                sample["_corridor_initial_offset_abs_m"] = abs(
                    -math.sin(float(corridor_heading_rad)) * float(dx_m)
                    + math.cos(float(corridor_heading_rad)) * float(dy_m)
                )
            except (KeyError, TypeError, ValueError):
                sample["_corridor_initial_offset_abs_m"] = 0.0

    for _ in range(max(0, int(passes))):
        previous = [dict(sample) for sample in result]
        for index in range(1, len(result) - 1):
            before = previous[index - 1]
            current = previous[index]
            after = previous[index + 1]
            result[index]["x_ref_m"] = (
                float(before["x_ref_m"])
                + 2.0 * float(current["x_ref_m"])
                + float(after["x_ref_m"])
            ) / 4.0
            result[index]["y_ref_m"] = (
                float(before["y_ref_m"])
                + 2.0 * float(current["y_ref_m"])
                + float(after["y_ref_m"])
            ) / 4.0
            if bool(boundary_aware):
                try:
                    center_x_m = float(
                        current.get(
                            "corridor_center_x_m",
                            current["x_ref_m"],
                        )
                    )
                    center_y_m = float(
                        current.get(
                            "corridor_center_y_m",
                            current["y_ref_m"],
                        )
                    )
                    corridor_heading_rad = float(
                        current.get(
                            "corridor_heading_rad",
                            current.get("heading_rad", 0.0),
                        )
                    )
                    lane_width_m = float(
                        current.get("lane_width_m", 0.0) or 0.0
                    )
                    allowed_offset_m = max(
                        0.0,
                        0.5 * float(lane_width_m)
                        - max(0.0, float(vehicle_half_width_m))
                        - max(0.0, float(boundary_margin_m))
                        - max(0.0, float(tracking_reserve_m)),
                        float(
                            current.get(
                                "_corridor_initial_offset_abs_m",
                                0.0,
                            )
                            or 0.0
                        ),
                    )
                    dx_m = (
                        float(result[index]["x_ref_m"])
                        - float(center_x_m)
                    )
                    dy_m = (
                        float(result[index]["y_ref_m"])
                        - float(center_y_m)
                    )
                    lateral_m = (
                        -math.sin(float(corridor_heading_rad)) * float(dx_m)
                        + math.cos(float(corridor_heading_rad)) * float(dy_m)
                    )
                    bounded_lateral_m = min(
                        float(allowed_offset_m),
                        max(-float(allowed_offset_m), float(lateral_m)),
                    )
                    correction_m = (
                        float(bounded_lateral_m) - float(lateral_m)
                    )
                    result[index]["x_ref_m"] = (
                        float(result[index]["x_ref_m"])
                        - math.sin(float(corridor_heading_rad))
                        * float(correction_m)
                    )
                    result[index]["y_ref_m"] = (
                        float(result[index]["y_ref_m"])
                        + math.cos(float(corridor_heading_rad))
                        * float(correction_m)
                    )
                    result[index]["corridor_allowed_offset_m"] = float(
                        allowed_offset_m
                    )
                    result[index]["corridor_lateral_offset_m"] = float(
                        bounded_lateral_m
                    )
                except (KeyError, TypeError, ValueError):
                    pass
            result[index]["x"] = float(result[index]["x_ref_m"])
            result[index]["y"] = float(result[index]["y_ref_m"])
    return _recompute_reference_headings(result)


def _apply_route_rejoin_offset(
    samples: Sequence[Mapping[str, object]],
    *,
    ego_x_m: float,
    ego_y_m: float,
    projection_x_m: float,
    projection_y_m: float,
    step_distance_m: float,
    rejoin_distance_m: float,
) -> List[Dict[str, object]]:
    """Decay the ego-to-route offset while retaining the selected connector."""

    result = [dict(sample) for sample in list(samples or [])]
    offset_x_m = float(ego_x_m) - float(projection_x_m)
    offset_y_m = float(ego_y_m) - float(projection_y_m)
    merge_distance_m = max(1.0, float(rejoin_distance_m))
    step_m = max(0.1, float(step_distance_m))
    for index, sample in enumerate(result):
        progress = min(
            1.0,
            max(0.0, float(index + 1) * float(step_m) / float(merge_distance_m)),
        )
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        residual = 1.0 - float(smooth_progress)
        sample["x_ref_m"] = float(sample["x_ref_m"]) + residual * float(offset_x_m)
        sample["y_ref_m"] = float(sample["y_ref_m"]) + residual * float(offset_y_m)
        sample["x"] = float(sample["x_ref_m"])
        sample["y"] = float(sample["y_ref_m"])
    return _recompute_reference_headings(result)


def _apply_ego_heading_connector(
    samples: Sequence[Mapping[str, object]],
    *,
    ego_x_m: float,
    ego_y_m: float,
    ego_heading_rad: float,
    step_distance_m: float,
    rejoin_distance_m: float,
    smoothing_passes: int,
    boundary_aware: bool = False,
    vehicle_half_width_m: float = 1.0,
    boundary_margin_m: float = 0.15,
    tracking_reserve_m: float = 0.20,
) -> List[Dict[str, object]]:
    """Join ego pose to the selected route with a heading-continuous curve.

    The global route still selects the junction branch.  This local Hermite
    connector only removes the position and tangent discontinuity between the
    live vehicle pose and that branch.
    """

    route = [dict(sample) for sample in list(samples or [])]
    if len(route) < 2:
        return _recompute_reference_headings(route)

    step_m = max(0.1, float(step_distance_m))
    target_count = len(route)
    route_progress = [0.0]
    for previous, current in zip(route[:-1], route[1:]):
        route_progress.append(
            route_progress[-1]
            + math.hypot(
                float(current["x_ref_m"]) - float(previous["x_ref_m"]),
                float(current["y_ref_m"]) - float(previous["y_ref_m"]),
            )
        )

    desired_rejoin_m = min(
        max(2.0, float(rejoin_distance_m)),
        max(2.0, route_progress[-1] * 0.75),
    )
    join_index = next(
        (
            index
            for index, progress_m in enumerate(route_progress)
            if float(progress_m) >= float(desired_rejoin_m)
        ),
        len(route) - 1,
    )
    join_index = max(1, int(join_index))
    join = route[join_index]
    join_x_m = float(join["x_ref_m"])
    join_y_m = float(join["y_ref_m"])
    join_heading_rad = float(
        join.get(
            "heading_rad",
            math.atan2(
                join_y_m - float(route[join_index - 1]["y_ref_m"]),
                join_x_m - float(route[join_index - 1]["x_ref_m"]),
            ),
        )
    )
    chord_m = math.hypot(join_x_m - float(ego_x_m), join_y_m - float(ego_y_m))
    if chord_m < 0.5:
        return _recompute_reference_headings(route)

    # Equal, bounded tangent magnitudes avoid both a sharp entry kink and the
    # large Hermite overshoot that a long global-route segment can introduce.
    tangent_m = min(
        max(2.0, 1.15 * chord_m),
        max(3.0, 1.15 * desired_rejoin_m),
    )
    start_tangent = (
        tangent_m * math.cos(float(ego_heading_rad)),
        tangent_m * math.sin(float(ego_heading_rad)),
    )
    end_tangent = (
        tangent_m * math.cos(float(join_heading_rad)),
        tangent_m * math.sin(float(join_heading_rad)),
    )

    dense_count = max(32, int(math.ceil(chord_m / step_m)) * 8)
    dense_points: List[Tuple[float, float]] = []
    for dense_index in range(dense_count + 1):
        t = float(dense_index) / float(dense_count)
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        dense_points.append(
            (
                h00 * float(ego_x_m)
                + h10 * start_tangent[0]
                + h01 * join_x_m
                + h11 * end_tangent[0],
                h00 * float(ego_y_m)
                + h10 * start_tangent[1]
                + h01 * join_y_m
                + h11 * end_tangent[1],
            )
        )

    dense_progress = [0.0]
    for previous, current in zip(dense_points[:-1], dense_points[1:]):
        dense_progress.append(
            dense_progress[-1]
            + math.hypot(current[0] - previous[0], current[1] - previous[1])
        )

    connector: List[Dict[str, object]] = []
    dense_segment = 0
    sample_s_m = float(step_m)
    while sample_s_m < dense_progress[-1] - 0.25 * step_m:
        while (
            dense_segment + 1 < len(dense_progress)
            and dense_progress[dense_segment + 1] < sample_s_m
        ):
            dense_segment += 1
        segment_length_m = max(
            1.0e-6,
            dense_progress[dense_segment + 1] - dense_progress[dense_segment],
        )
        ratio = min(
            1.0,
            max(
                0.0,
                (sample_s_m - dense_progress[dense_segment]) / segment_length_m,
            ),
        )
        first_xy = dense_points[dense_segment]
        second_xy = dense_points[dense_segment + 1]
        x_m = first_xy[0] + ratio * (second_xy[0] - first_xy[0])
        y_m = first_xy[1] + ratio * (second_xy[1] - first_xy[1])
        route_ratio = min(1.0, sample_s_m / max(dense_progress[-1], 1.0e-6))
        template_index = min(join_index, int(round(route_ratio * join_index)))
        sample = dict(route[template_index])
        sample["x_ref_m"] = float(x_m)
        sample["y_ref_m"] = float(y_m)
        sample["x"] = float(x_m)
        sample["y"] = float(y_m)
        connector.append(sample)
        sample_s_m += float(step_m)

    combined = connector + [dict(sample) for sample in route[join_index:]]
    combined = _deduplicate_reference_samples(combined)
    if not combined:
        return _recompute_reference_headings(route)
    while len(combined) < target_count:
        last = dict(combined[-1])
        heading_rad = float(last.get("heading_rad", join_heading_rad))
        if len(combined) >= 2:
            previous = combined[-2]
            heading_rad = math.atan2(
                float(last["y_ref_m"]) - float(previous["y_ref_m"]),
                float(last["x_ref_m"]) - float(previous["x_ref_m"]),
            )
        last["x_ref_m"] = float(last["x_ref_m"]) + step_m * math.cos(heading_rad)
        last["y_ref_m"] = float(last["y_ref_m"]) + step_m * math.sin(heading_rad)
        last["x"] = float(last["x_ref_m"])
        last["y"] = float(last["y_ref_m"])
        combined.append(last)
    # Smooth across the Hermite/route splice as one curve.  The endpoints stay
    # fixed, so this does not change the selected branch or the ego anchor.
    return _smooth_carla_reference_samples(
        combined[:target_count],
        passes=int(smoothing_passes),
        boundary_aware=bool(boundary_aware),
        vehicle_half_width_m=float(vehicle_half_width_m),
        boundary_margin_m=float(boundary_margin_m),
        tracking_reserve_m=float(tracking_reserve_m),
    )


def _deduplicate_reference_samples(
    samples: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for sample in list(samples or []):
        current = dict(sample)
        if result and math.hypot(
            float(current["x_ref_m"]) - float(result[-1]["x_ref_m"]),
            float(current["y_ref_m"]) - float(result[-1]["y_ref_m"]),
        ) < 1.0e-3:
            continue
        result.append(current)
    return result


def _recompute_reference_headings(
    samples: Sequence[Mapping[str, object]],
) -> List[Dict[str, object]]:
    result = [dict(sample) for sample in list(samples or [])]
    for index, sample in enumerate(result):
        if len(result) < 2:
            break
        if index + 1 < len(result):
            first = sample
            second = result[index + 1]
        else:
            first = result[index - 1]
            second = sample
        dx_m = float(second["x_ref_m"]) - float(first["x_ref_m"])
        dy_m = float(second["y_ref_m"]) - float(first["y_ref_m"])
        if math.hypot(dx_m, dy_m) > 1.0e-6:
            sample["heading_rad"] = math.atan2(dy_m, dx_m)
        sample["x"] = float(sample["x_ref_m"])
        sample["y"] = float(sample["y_ref_m"])
    return result


def _next_macro_from_carla_options(options: Sequence[str]) -> str:
    for option in list(options or []):
        normalized = str(option).strip().upper().replace("_", "")
        if normalized == "CHANGELANELEFT":
            return "Lane Change Left"
        if normalized == "CHANGELANERIGHT":
            return "Lane Change Right"
        if normalized == "LEFT":
            return "Left Turn"
        if normalized == "RIGHT":
            return "Right Turn"
        if normalized == "STRAIGHT":
            return "Continue Straight"
    return "Continue Straight"


def _distance_to_next_carla_macro(
    *,
    nodes: Sequence[Tuple[float, float, float, Any, str]],
    start_index: int,
    projection: Optional[Tuple[int, float, float, float, float]] = None,
) -> Optional[float]:
    """Along-route distance from ego projection to the next macro node."""

    if len(nodes) < 2:
        return None
    index = min(max(0, int(start_index)), len(nodes) - 2)
    if projection is not None:
        previous_xy = (float(projection[1]), float(projection[2]))
    else:
        previous_xy = (float(nodes[index][0]), float(nodes[index][1]))
    distance_m = 0.0
    macro_options = {
        "CHANGELANELEFT",
        "CHANGELANERIGHT",
        "LEFT",
        "RIGHT",
        "STRAIGHT",
    }
    for node in list(nodes[index + 1 : index + 81]):
        current_xy = (float(node[0]), float(node[1]))
        distance_m += math.hypot(
            current_xy[0] - previous_xy[0],
            current_xy[1] - previous_xy[1],
        )
        option = str(node[4] or "").strip().upper().replace("_", "")
        if option in macro_options:
            return float(distance_m)
        previous_xy = current_xy
    return None


def _route_required_carla_lane_id(
    *,
    nodes: Sequence[Tuple[float, float, float, Any, str]],
    start_index: int,
    fallback_lane_id: int,
    ego_waypoint: Any = None,
) -> int:
    """Find the canonical lane id the route requires ego to reach next.

    Whenever possible, the target id is derived as ``current_lane_id +
    (real lane-adjacency hops from ego_waypoint to the maneuver point)``
    rather than by independently recounting lanes at the maneuver point --
    see ``lane_hop_offset`` for why: recounting at a different point on the
    route than ego's own position is unsound whenever the total lane count
    differs between the two points (a lane merges away, a turn-only lane
    appears/disappears), which silently produces a target id that doesn't
    correspond to the same physical lane as ego's own id, even though nothing
    about ego's real world position changed.
    """

    current_lane_id = int(fallback_lane_id)
    start = min(max(0, int(start_index) + 1), max(0, len(nodes) - 1))
    for node in list(nodes[start : start + 80]):
        option = str(node[4] or "").strip().upper().replace("_", "")
        if option not in {"CHANGELANELEFT", "CHANGELANERIGHT"}:
            if option in {"LEFT", "RIGHT", "STRAIGHT"}:
                break
            continue
        # A CARLA CHANGELANE option is an adjacent-lane edge, so its target
        # is exactly one stable lane-id step from ego.  Do not let an
        # independent canonical recount at the remote maneuver waypoint
        # override that fact: on roads where lanes appear/disappear between
        # ego and the maneuver point, that local recount can turn a physical
        # 2 -> 1 right change into an impossible target such as 5.
        expected_hop = 1 if option == "CHANGELANELEFT" else -1
        hop = None
        if ego_waypoint is not None:
            from cpx_planning.utility.lane_graph import lane_hop_offset

            hop = lane_hop_offset(ego_waypoint, node[3])
        if hop is not None and int(hop) == int(expected_hop):
            candidate_lane_id = int(current_lane_id) + int(hop)
        else:
            local_candidate_lane_id = int(
                _canonical_carla_lane_id(node[3], current_lane_id)
            )
            raw_lane_id = int(getattr(node[3], "lane_id", 0) or 0)
            if (
                local_candidate_lane_id == current_lane_id
                and raw_lane_id > 0
                and abs(int(raw_lane_id) - int(current_lane_id)) == 1
            ):
                local_candidate_lane_id = int(raw_lane_id)
            if abs(int(local_candidate_lane_id) - int(current_lane_id)) == 1:
                # Compatibility for externally supplied/synthetic routes
                # whose lane-id orientation is not the runtime stable-id
                # convention, but whose target is still unambiguously
                # adjacent.
                candidate_lane_id = int(local_candidate_lane_id)
            else:
                candidate_lane_id = int(current_lane_id) + int(expected_hop)
        if candidate_lane_id != 0 and candidate_lane_id != current_lane_id:
            return int(candidate_lane_id)
    return int(current_lane_id)


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def _to_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)
