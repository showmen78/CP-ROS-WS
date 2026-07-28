"""Route lifecycle manager for the CP-X Planning pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint


@dataclass
class RouteManagerStatus:
    """Store the small route-status summary that the rest of the planner needs."""

    route_found: bool = False
    route_point_count: int = 0
    remaining_distance_m: float = 0.0
    reached_destination: bool = False
    debug_reason: str = ""

    def as_dict(self) -> Dict[str, object]:
        """Return this status as a normal dictionary."""
        return {
            "route_found": bool(self.route_found),
            "route_point_count": int(self.route_point_count),
            "remaining_distance_m": float(self.remaining_distance_m),
            "reached_destination": bool(self.reached_destination),
            "debug_reason": str(self.debug_reason),
        }


class CPXRouteManager:
    """Keep one global route and turn its remaining part into reference points for the MPC."""

    def __init__(self, *, global_planner: Any, reached_distance_m: float = 3.0, stale_route_lateral_m: float = 12.0) -> None:
        """Create a route manager around the custom global planner.

        `global_planner`: custom planner adapter used to find routes and map-match route points.
        `reached_distance_m`: how close the ego vehicle must be to count the destination as reached.
        `stale_route_lateral_m`:  how far the ego may move away from the route before the route is treated
        as stale. 
        """
        self.global_planner = global_planner
        self.reached_distance_m = max(0.1, float(reached_distance_m))
        self.stale_route_lateral_m = max(1.0, float(stale_route_lateral_m))
        self._active_route_summary = None
        self._start_point: Optional[Dict[str, float]] = None
        self._goal_point: Optional[Dict[str, float]] = None
        self._fallback_route_points: List[List[float]] = []

        # A route node keeps both coordinates and custom-map information. The coordinates guide the MPC,
        # while the waypoint and route option tell the behavior planner which lane and maneuver apply there.
        self._route_nodes_cache: List[Tuple[float, float, float, Any, str]] = []
        self._route_progress_index = 0
        self._route_progress_initialized = False

        # This value describes the closest place on the route: segment number, projected x and y,
        # position within that segment, and sideways distance from the route.
        self._route_projection: Optional[Tuple[int, float, float, float, float]] = None
        self._route_sync_reason = "route_progress_not_initialized"
        self._route_debug_reason = "route_not_initialized"
        self._last_status = RouteManagerStatus(debug_reason="route_not_initialized")

    def set_destination(self, *, start_point: Mapping[str, object], goal_point: Mapping[str, object]) -> Any:
        """Ask the custom global planner for a route from `start_point` to `goal_point`.

        Both start_point and goal_point are dictionaries with x and y coordinates and an optional z coordinate. The output is
        the route summary returned by the custom planner. The route and its map information are also saved
        here so later planning cycles can check progress and build MPC reference points.
        """
        self._start_point = _point_dict(start_point)
        self._goal_point = _point_dict(goal_point)

        # Replacing the stored route keeps the route manager and global planner on the same active route.
        summary = self.global_planner.plan_route_from_locations(
            start_location=self._start_point, goal_location=self._goal_point, replace_stored_route=True
        )

        self._active_route_summary = summary
        self._build_route_nodes(summary)
        self._fallback_route_points = []

        route_found = bool(getattr(summary, "route_found", False))
        route_waypoints = list(getattr(summary, "route_waypoints", []) or [])

        # A direct two-point path keeps route output available if the custom planner cannot return a usable
        # route. It does not replace or imitate map-aware routing.
        if not route_found or len(route_waypoints) < 2:
            self._fallback_route_points = self._direct_fallback_route_points(
                start_point=self._start_point, goal_point=self._goal_point
            )

        self._last_status = self._status_from_summary(summary)
        return summary

    def get_route_info(self, *, x_m: float, y_m: float, query_key: str, fallback_lane_id: int) -> Dict[str, object]:
        """Return the route information needed for the current planning cycle.

        `x_m` and `y_m` are the ego position, `query_key` identifies this request to the global planner, and
        `fallback_lane_id` is used only when no valid lane ID can be read from the route. The output dictionary
        contains route availability, the selected lane, the current road action, the next large maneuver,
        remaining distance, destination state, and a debugging reason.
        """
        try:
            summary = self.global_planner.get_current_route_info(x_m=float(x_m), y_m=float(y_m), query_key=str(query_key))
        except Exception as exc:
            # A route query failure should not crash the complete planning cycle. The returned values clearly
            # show that the map route is unavailable so the caller can make a safe decision.
            self._last_status = RouteManagerStatus(route_found=False, debug_reason=f"route_query_failed:{exc}")
            return self._fallback_summary(
                fallback_lane_id=int(fallback_lane_id), debug_reason=self._last_status.debug_reason
            )

        if summary is None:
            summary = self._active_route_summary
        if summary is None:
            self._last_status = RouteManagerStatus(debug_reason="route_missing")
            return self._fallback_summary(fallback_lane_id=int(fallback_lane_id), debug_reason="route_missing")

        self._active_route_summary = summary
        self._last_status = self._status_from_summary(summary)

        # When global routing failed, measure progress to the saved direct destination instead. This preserves
        # the existing fallback behavior while making it clear that the result is not a map-aware route.
        if not bool(getattr(summary, "route_found", False)) and len(self._fallback_route_points) >= 2:
            remaining_distance_m = self._remaining_distance_on_fallback_route(x_m=float(x_m), y_m=float(y_m))
            reached = float(remaining_distance_m) <= float(self.reached_distance_m)
            self._last_status = RouteManagerStatus(
                route_found=True,
                route_point_count=len(self._fallback_route_points),
                remaining_distance_m=float(remaining_distance_m),
                reached_destination=bool(reached),
                debug_reason=self._fallback_debug_reason(summary),
            )
            return {
                "route_found": True,
                "optimal_lane_id": int(fallback_lane_id),
                "current_road_option": "FALLBACK_DIRECT",
                "next_macro_maneuver": "Continue Straight",
                "debug_reason": self._last_status.debug_reason,
                "remaining_distance_m": float(remaining_distance_m),
                "reached_destination": bool(reached),
            }

        lane_id = _to_int(getattr(summary, "optimal_lane_id", fallback_lane_id), fallback_lane_id)
        if lane_id == 0:
            lane_id = int(fallback_lane_id)

        return {
            "route_found": bool(getattr(summary, "route_found", False)),
            "optimal_lane_id": int(lane_id),
            "current_road_option": str(getattr(summary, "current_road_option", "")),
            "next_macro_maneuver": str(getattr(summary, "next_macro_maneuver", "Continue Straight")),
            "debug_reason": str(getattr(summary, "debug_reason", "custom_global_route")),
            "remaining_distance_m": float(getattr(summary, "distance_to_destination_m", 0.0) or 0.0),
            "reached_destination": bool(self._last_status.reached_destination),
        }

    def route_points(
        self, *, x_m: Optional[float] = None, y_m: Optional[float] = None, query_key: str = ""
    ) -> List[List[float]]:
        """Return the active route as `[x, y, z, heading]` points.

        `x_m` and `y_m` are optional ego coordinates used to refresh route progress before reading the route.
        `query_key` labels that refresh request. The output is the custom route when one is available; otherwise,
        it is the saved direct fallback route.
        """

        summary = None
        if x_m is not None and y_m is not None:
            try:
                summary = self.global_planner.get_current_route_info(
                    x_m=float(x_m), y_m=float(y_m), query_key=str(query_key or "route_points")
                )
            except Exception:
                # The previously stored route is still useful if only this progress refresh failed.
                summary = None

        if summary is None:
            summary = self._active_route_summary

        route_waypoints = list(getattr(summary, "route_waypoints", []) or [])
        route_points = _route_points_from_waypoints(route_waypoints)

        if len(route_points) >= 2:
            return route_points

        return [list(point) for point in self._fallback_route_points]

    def route_reference(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
        horizon_steps: int,
        step_distance_m: float,
        target_speed_mps: float,
        fallback_lane_id: int,
    ) -> Tuple[List[Dict[str, object]], str]:
        """Build the short route reference that the MPC follows."""

        sync_reason = self.sync_route_progress(
            ego_x_m=float(ego_x_m), ego_y_m=float(ego_y_m), ego_heading_rad=float(ego_heading_rad)
        )
        nodes = self._route_nodes()
        if len(nodes) < 2:
            return [], str(sync_reason or "route_waypoints_empty")
        if self._route_projection is None:
            return [], str(sync_reason or "route_projection_failed")

        segment_index, projection_x_m, projection_y_m, projection_ratio, lateral_distance_m = self._route_projection
        if float(lateral_distance_m) > float(self.stale_route_lateral_m):
            # Reference points from a route far away from the ego could make the MPC steer toward the wrong road.
            return [], str(sync_reason)

        first = nodes[segment_index]
        second = nodes[min(segment_index + 1, len(nodes) - 1)]
        projection_z_m = float(first[2]) + float(projection_ratio) * (float(second[2]) - float(first[2]))

        # Start the remaining route at the ego's projected place, not at an old route point behind the vehicle.
        polyline: List[Tuple[float, float, float, Any, str]] = [
            (float(projection_x_m), float(projection_y_m), float(projection_z_m), second[3], second[4])
        ]
        polyline.extend(nodes[segment_index + 1:])
        polyline = _deduplicate_route_nodes(polyline)
        if len(polyline) < 2:
            return [], "route_remaining_polyline_too_short"

        # Cumulative distance lets us sample the route at equal distances even when the original route points
        # are unevenly spaced.
        cumulative_distance = [0.0]
        for previous, current in zip(polyline[:-1], polyline[1:]):
            distance_m = math.hypot(float(current[0]) - float(previous[0]), float(current[1]) - float(previous[1]))
            cumulative_distance.append(cumulative_distance[-1] + distance_m)

        step_m = max(0.25, float(step_distance_m))
        samples: List[Dict[str, object]] = []
        sample_segment = 0

        for sample_index in range(max(1, int(horizon_steps))):
            target_distance_m = float(sample_index + 1) * step_m

            # Move forward until the target distance lies inside the current route segment.
            while (
                sample_segment + 1 < len(cumulative_distance)
                and cumulative_distance[sample_segment + 1] < target_distance_m
            ):
                sample_segment += 1

            if sample_segment + 1 < len(polyline):
                node_a = polyline[sample_segment]
                node_b = polyline[sample_segment + 1]
                segment_length_m = max(1.0e-6, cumulative_distance[sample_segment + 1] - cumulative_distance[sample_segment])
                ratio = min(1.0, max(0.0, (target_distance_m - cumulative_distance[sample_segment]) / segment_length_m))
                x_m = float(node_a[0]) + ratio * (float(node_b[0]) - float(node_a[0]))
                y_m = float(node_a[1]) + ratio * (float(node_b[1]) - float(node_a[1]))
                heading_rad = math.atan2(float(node_b[1]) - float(node_a[1]), float(node_b[0]) - float(node_a[0]))
                waypoint = node_b[3]
                road_option = node_b[4]
            else:
                # Near the route end, continue in the final segment's direction so the MPC still receives the
                # requested number of reference points.
                node_a = polyline[-2]
                node_b = polyline[-1]
                heading_rad = math.atan2(float(node_b[1]) - float(node_a[1]), float(node_b[0]) - float(node_a[0]))
                extra_distance_m = max(0.0, target_distance_m - cumulative_distance[-1])
                x_m = float(node_b[0]) + extra_distance_m * math.cos(heading_rad)
                y_m = float(node_b[1]) + extra_distance_m * math.sin(heading_rad)
                waypoint = node_b[3]
                road_option = node_b[4]

            # The geometry comes from the global route, while lane and road details come from the custom map
            # waypoint saved for this part of the route.
            lane_id = canonical_lane_id_for_waypoint(waypoint)
            if lane_id == 0:
                lane_id = int(fallback_lane_id)

            lane_width_m = getattr(waypoint, "lane_width_m", None)
            if lane_width_m is None:
                lane_width_m = 3.5

            road_id = getattr(waypoint, "road_id", 0)
            speed_mps = max(0.0, float(target_speed_mps))
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(heading_rad),
                "lane_id": int(lane_id),
                "lane_width_m": float(lane_width_m),
                "road_id": int(road_id or 0),
                "road_option": str(road_option),
                "speed_ref_mps": speed_mps,
                "v_ref_mps": speed_mps,
                "speed_mps": speed_mps,
            })

        return samples, "custom_global_route"

    def sync_route_progress(
        self,
        *,
        ego_x_m: float,
        ego_y_m: float,
        ego_heading_rad: float,
    ) -> str:
        """Find where the ego vehicle is on the active custom route."""

        nodes = self._route_nodes()
        if len(nodes) < 2:
            self._route_projection = None
            self._route_sync_reason = str(self._route_debug_reason or "route_unavailable")
            return self._route_sync_reason

        if not self._route_progress_initialized:
            # On the first update, search the whole route because the ego's starting segment is not known yet.
            lower_index = 0
            upper_index = len(nodes) - 1
            search_mode = "global_init"
        else:
            # After initialization, search near the last segment. This is faster and avoids jumping to a
            # different piece of road when the route crosses or runs close to itself.
            lower_index = max(0, int(self._route_progress_index) - 5)
            upper_index = min(len(nodes) - 1, max(lower_index + 1, int(self._route_progress_index) + 80))
            search_mode = "local_update"

        best_projection = _best_route_projection(
            nodes=nodes,
            ego_x_m=float(ego_x_m),
            ego_y_m=float(ego_y_m),
            ego_heading_rad=float(ego_heading_rad),
            lower_index=int(lower_index),
            upper_index=int(upper_index),
        )

        if best_projection is None:
            self._route_sync_reason = f"route_progress_{search_mode}_failed"
            return self._route_sync_reason

        _, _, best_index, _, _, _ = best_projection

        # Route progress is not allowed to move backward. This prevents noisy ego positions from making the
        # planner return reference points that the vehicle has already passed.
        if not self._route_progress_initialized:
            segment_index = int(best_index)
        else:
            segment_index = max(int(self._route_progress_index), int(best_index))

        first = nodes[segment_index]
        second = nodes[min(segment_index + 1, len(nodes) - 1)]
        projection_x_m, projection_y_m, projection_ratio, lateral_distance_m = _project_to_segment(
            x_m=float(ego_x_m), y_m=float(ego_y_m), first_xy=(first[0], first[1]), second_xy=(second[0], second[1])
        )

        self._route_progress_index = int(segment_index)
        self._route_progress_initialized = True
        self._route_projection = (
            int(segment_index),
            float(projection_x_m),
            float(projection_y_m),
            float(projection_ratio),
            float(lateral_distance_m),
        )

        if float(lateral_distance_m) > float(self.stale_route_lateral_m):
            self._route_sync_reason = f"route_stale:lateral={float(lateral_distance_m):.2f}"
        else:
            self._route_sync_reason = f"route_progress_{search_mode}:index={segment_index}"

        return self._route_sync_reason

    @property
    def route_debug_reason(self) -> str:
        """Return the latest message about route creation and route-node preparation."""
        return str(self._route_debug_reason)

    @property
    def route_sync_reason(self) -> str:
        """Return the latest message about matching the ego vehicle to the route."""
        return str(self._route_sync_reason)

    @property
    def route_progress_index(self) -> int:
        """Return the route segment number that the ego vehicle has reached."""
        return int(self._route_progress_index)

    @property
    def last_status(self) -> RouteManagerStatus:
        """Return the most recently calculated route status object."""
        return self._last_status

    def _route_nodes(self) -> List[Tuple[float, float, float, Any, str]]:
        """Return a copy of the saved route nodes so callers cannot replace the internal list."""
        return list(self._route_nodes_cache)

    def _build_route_nodes(self, summary: Any) -> None:
        """Convert a custom route summary into the route nodes used for progress and MPC reference generation."""
        self._route_nodes_cache = []
        self._route_progress_index = 0
        self._route_progress_initialized = False
        self._route_projection = None
        self._route_sync_reason = "route_progress_not_initialized"

        if not bool(getattr(summary, "route_found", False)):
            self._route_debug_reason = "custom_global_route_not_found"
            return

        route_waypoints = list(getattr(summary, "route_waypoints", []) or [])
        road_options = list(getattr(summary, "road_options", []) or [])

        for index, raw_point in enumerate(route_waypoints):
            point = _route_point_dict(raw_point)

            try:
                # Map-match each route point so later modules can get lane width, lane ID, road ID, and nearby
                # lane information without asking CARLA for road geometry.
                waypoint = self.global_planner.get_waypoint(point)
            except Exception:
                waypoint = None

            if waypoint is not None:
                position = getattr(waypoint, "position", {})
                z_m = float(position.get("z", point.get("z", 0.0)))
            else:
                z_m = float(point.get("z", 0.0))

            if road_options:
                road_option = str(road_options[min(index, len(road_options) - 1)])
            else:
                road_option = "LANEFOLLOW"

            node = (float(point["x"]), float(point["y"]), float(z_m), waypoint, road_option)

            # Repeated points create zero-length segments, which would make route projection and interpolation
            # unreliable, so adjacent duplicates are ignored.
            if (
                self._route_nodes_cache
                and math.hypot(node[0] - self._route_nodes_cache[-1][0], node[1] - self._route_nodes_cache[-1][1])
                < 1.0e-3
            ):
                continue

            self._route_nodes_cache.append(node)

        if len(self._route_nodes_cache) >= 2:
            self._route_debug_reason = "custom_global_route_ready"
        else:
            self._route_debug_reason = "custom_global_route_too_short"

    def _status_from_summary(self, summary: Any) -> RouteManagerStatus:
        """Create the small route status used by the rest of the planning pipeline."""
        
        route_points = list(getattr(summary, "route_waypoints", []) or [])
        remaining_distance_m = float(getattr(summary, "distance_to_destination_m", 0.0) or 0.0)
        route_found = bool(getattr(summary, "route_found", False))
        fallback_distance_m = self._fallback_route_total_distance_m()

        return RouteManagerStatus(
            route_found=route_found or len(self._fallback_route_points) >= 2,
            route_point_count=max(len(route_points), len(self._fallback_route_points)),
            remaining_distance_m=remaining_distance_m if route_found else fallback_distance_m,
            reached_destination=bool(
                (route_found and remaining_distance_m <= self.reached_distance_m)
                or (not route_found and fallback_distance_m <= self.reached_distance_m)
            ),
            debug_reason=(
                str(getattr(summary, "debug_reason", "route_active"))
                if route_found
                else self._fallback_debug_reason(summary)
            ),
        )

    @staticmethod
    def _fallback_summary(*, fallback_lane_id: int, debug_reason: str) -> Dict[str, object]:
        """Return safe route-information fields when no route summary is available.  """
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
        *, start_point: Mapping[str, object], goal_point: Mapping[str, object]
    ) -> List[List[float]]:
        """Create a straight two-point fallback between the requested start and destination."""
        start = _point_dict(start_point)
        goal = _point_dict(goal_point)
        heading_rad = math.atan2(float(goal["y"]) - float(start["y"]), float(goal["x"]) - float(start["x"]))

        return [
            [float(start["x"]), float(start["y"]), float(start["z"]), float(heading_rad)],
            [float(goal["x"]), float(goal["y"]), float(goal["z"]), float(heading_rad)],
        ]

    def _fallback_route_total_distance_m(self) -> float:
        """Return the full straight-line fallback distance, or zero when no fallback exists."""
        if len(self._fallback_route_points) < 2:
            return 0.0

        start = self._fallback_route_points[0]
        goal = self._fallback_route_points[-1]
        return math.hypot(float(goal[0]) - float(start[0]), float(goal[1]) - float(start[1]))

    def _remaining_distance_on_fallback_route(self, *, x_m: float, y_m: float) -> float:
        """Return the straight-line distance from the given ego position to the fallback destination."""
        if len(self._fallback_route_points) < 2:
            return 0.0

        goal = self._fallback_route_points[-1]
        return math.hypot(float(goal[0]) - float(x_m), float(goal[1]) - float(y_m))

    @staticmethod
    def _fallback_debug_reason(summary: Any) -> str:
        """Return a clear message explaining why the direct fallback route is being used."""
        reason = str(getattr(summary, "debug_reason", "") or "").strip()
        if reason:
            return "global_route_failed_direct_fallback:" + reason

        return "global_route_failed_direct_fallback"


def _point_dict(point: Mapping[str, object]) -> Dict[str, float]:
    """Convert a position dictionary into the coordinate names used by the custom planner."""
    return {
        "x": float(point.get("x", point.get("x_m", 0.0))),
        "y": float(point.get("y", point.get("y_m", 0.0))),
        "z": float(point.get("z", point.get("z_m", 0.0))),
    }


def _route_point_dict(point: Mapping[str, object] | Sequence[object]) -> Dict[str, float]:
    """Convert one route point into an `x`, `y`, and `z` dictionary.

    """
    if isinstance(point, Mapping):
        return _point_dict(point)

    if len(point) < 2:
        raise ValueError("A route point requires at least x and y.")

    return {
        "x": float(point[0]),
        "y": float(point[1]),
        "z": float(point[2]) if len(point) >= 3 else 0.0,
    }


def _route_points_from_waypoints(route_waypoints: Sequence[Any]) -> List[List[float]]:
    """Convert route waypoints into `[x, y, z, heading]` points."""
    points: List[List[float]] = []

    for index, raw_point in enumerate(route_waypoints):
        try:
            point = _route_point_dict(raw_point)
        except Exception:
            continue

        x_m = float(point["x"])
        y_m = float(point["y"])
        z_m = float(point["z"])

        if index < len(route_waypoints) - 1:
            try:
                next_point = _route_point_dict(route_waypoints[index + 1])
                heading_rad = math.atan2(float(next_point["y"]) - y_m, float(next_point["x"]) - x_m)
            except Exception:
                heading_rad = points[-1][3] if points else 0.0
        else:
            heading_rad = points[-1][3] if points else 0.0

        if points and math.hypot(float(points[-1][0]) - x_m, float(points[-1][1]) - y_m) < 1.0e-3:
            continue

        points.append([x_m, y_m, z_m, float(heading_rad)])

    return points


def _project_to_segment(
    *,
    x_m: float,
    y_m: float,
    first_xy: Sequence[float],
    second_xy: Sequence[float],
) -> Tuple[float, float, float, float]:
    """Find the closest place on one route segment to the ego position.

    `x_m` and `y_m` are the ego coordinates. `first_xy` and `second_xy` are the two ends of a route segment.
    The output contains projected x, projected y, a ratio from 0 at the first end to 1 at the second end, and
    the sideways distance from the ego to that projected place.
    """
    dx_m = float(second_xy[0]) - float(first_xy[0])
    dy_m = float(second_xy[1]) - float(first_xy[1])

    length_squared = dx_m * dx_m + dy_m * dy_m

    if length_squared <= 1.0e-9:
        ratio = 0.0
    else:
        ratio = ((float(x_m) - float(first_xy[0])) * dx_m + (float(y_m) - float(first_xy[1])) * dy_m) / length_squared
        ratio = min(1.0, max(0.0, float(ratio)))

    projected_x_m = float(first_xy[0]) + ratio * dx_m
    projected_y_m = float(first_xy[1]) + ratio * dy_m
    lateral_distance_m = math.hypot(float(x_m) - projected_x_m, float(y_m) - projected_y_m)
    return float(projected_x_m), float(projected_y_m), float(ratio), float(lateral_distance_m)


def _best_route_projection(
    *,
    nodes: Sequence[Tuple[float, float, float, Any, str]],
    ego_x_m: float,
    ego_y_m: float,
    ego_heading_rad: float,
    lower_index: int,
    upper_index: int,
) -> Optional[Tuple[float, float, int, float, float, float]]:
    """Choose the route segment that best matches the ego position and driving direction. """
    
    best = None
    lower = max(0, int(lower_index))
    upper = min(len(nodes) - 1, int(upper_index))

    for index in range(lower, upper):
        first = nodes[index]
        second = nodes[index + 1]

        projected_x_m, projected_y_m, ratio, distance_m = _project_to_segment(
            x_m=float(ego_x_m), y_m=float(ego_y_m), first_xy=(first[0], first[1]), second_xy=(second[0], second[1])
        )

        segment_heading_rad = math.atan2(float(second[1]) - float(first[1]), float(second[0]) - float(first[0]))
        heading_error_rad = abs(_wrap_angle(segment_heading_rad - float(ego_heading_rad)))

        # A nearby road going in the opposite direction is usually the wrong match, so it receives a large
        # penalty. A smaller heading penalty helps choose between nearby segments traveling similar directions.
        opposite_direction_penalty = 25.0 if heading_error_rad > 0.75 * math.pi else 0.0
        score = float(distance_m) + float(opposite_direction_penalty) + 0.2 * float(heading_error_rad)
        candidate = (
            float(score),
            float(distance_m),
            int(index),
            float(projected_x_m),
            float(projected_y_m),
            float(ratio),
        )

        if best is None or candidate < best:
            best = candidate

    return best


def _deduplicate_route_nodes(
    nodes: Sequence[Tuple[float, float, float, Any, str]]
) -> List[Tuple[float, float, float, Any, str]]:
    """Remove adjacent route nodes that have the same position."""
    result: List[Tuple[float, float, float, Any, str]] = []

    for node in nodes:
        if result and math.hypot(float(node[0]) - float(result[-1][0]), float(node[1]) - float(result[-1][1])) < 1.0e-3:
            continue
        result.append(node)

    return result


def _wrap_angle(angle_rad: float) -> float:
    """Return an input angle in the standard range from minus pi to plus pi."""
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def _to_int(value: object, default: int) -> int:
    """Convert a value to an integer and return `default` when the value cannot be converted."""
    try:
        return int(float(value))
    except Exception:
        return int(default)
