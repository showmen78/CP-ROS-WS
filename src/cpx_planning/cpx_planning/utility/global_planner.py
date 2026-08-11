"""Planning-facing adapter for the CARLA-independent OpenDRIVE planner."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import threading
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from cpx_planning.Global_Planner.global_planner import (
    GlobalPlanner,
    Route,
    Waypoint,
)


INVALID_LANE_ID = 0


def _euclidean_distance(first_point: Sequence[float], second_point: Sequence[float]) -> float:
    """Return Euclidean distance on Python 3.7 and newer."""
    return math.sqrt(
        sum((float(first) - float(second)) ** 2 for first, second in zip(first_point, second_point))
    )


@dataclass(frozen=True)
class WaypointQueryResult:
    index: int
    distance_m: float
    x_m: float
    y_m: float
    road_id: str
    lane_id: int
    direction: str


@dataclass
class RoutePlanSummary:
    route_found: bool
    start_road_id: str
    start_lane_id: int
    goal_road_id: str
    goal_lane_id: int
    optimal_lane_id: int
    distance_to_destination_m: float
    next_macro_maneuver: str
    route_waypoints: List[List[float]]
    road_options: List[str] = field(default_factory=list)
    current_road_option: str = "LANEFOLLOW"
    debug_reason: str = ""
    start_graph_index: int = -1
    goal_graph_index: int = -1
    start_graph_xy: Tuple[float, float] | None = None
    goal_graph_xy: Tuple[float, float] | None = None
    start_query_distance_m: float = float("inf")
    goal_query_distance_m: float = float("inf")


def _same_corridor(base: Waypoint, candidate: Waypoint | None) -> bool:
    if candidate is None:
        return False
    base_lane = int(base.lane_id or 0)
    candidate_lane = int(candidate.lane_id or 0)
    return (
        base_lane != 0
        and candidate_lane != 0
        and base_lane * candidate_lane > 0
        and int(base.road_id or 0) == int(candidate.road_id or 0)
    )


def canonical_lane_waypoints(waypoint: Waypoint | None) -> List[Waypoint]:
    if waypoint is None:
        return []

    rightmost = waypoint
    visited_right = {int(rightmost.ad_lane_id)}
    while True:
        candidate = rightmost.right()
        if not _same_corridor(waypoint, candidate):
            break
        assert candidate is not None
        if int(candidate.ad_lane_id) in visited_right:
            break
        visited_right.add(int(candidate.ad_lane_id))
        rightmost = candidate

    lanes = [rightmost]
    visited = {int(rightmost.ad_lane_id)}
    current = rightmost
    while True:
        candidate = current.left()
        if not _same_corridor(waypoint, candidate):
            break
        assert candidate is not None
        if int(candidate.ad_lane_id) in visited:
            break
        visited.add(int(candidate.ad_lane_id))
        lanes.append(candidate)
        current = candidate
    return lanes


def canonical_lane_ids_for_waypoint(waypoint: Waypoint | None) -> List[int]:
    return list(range(1, len(canonical_lane_waypoints(waypoint)) + 1))


def canonical_lane_id_for_waypoint(waypoint: Waypoint | None) -> int:
    if waypoint is None:
        return INVALID_LANE_ID
    
    for index, candidate in enumerate(canonical_lane_waypoints(waypoint), start=1):
        if int(candidate.ad_lane_id) == int(waypoint.ad_lane_id):
            return int(index)
    return INVALID_LANE_ID


def canonical_lane_waypoint_for_lane_id(
    waypoint: Waypoint | None,
    target_lane_id: int,
) -> Waypoint | None:

    lanes = canonical_lane_waypoints(waypoint)
    index = int(target_lane_id) - 1
    if 0 <= index < len(lanes):
        return lanes[index]
    return waypoint


def raw_opendrive_lane_id_for_waypoint(waypoint: Waypoint | None) -> int:

    return 0 if waypoint is None else int(waypoint.lane_id or 0)


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def _distance_2d(first: Sequence[float], second: Sequence[float]) -> float:
    return float(math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1])))


def _distance_3d(first: Sequence[float], second: Sequence[float]) -> float:
    return float(
        math.sqrt(
            (float(first[0]) - float(second[0])) ** 2
            + (float(first[1]) - float(second[1])) ** 2
            + (float(first[2]) - float(second[2])) ** 2
        )
    )


def world_heading_rad(waypoint: Waypoint | None) -> float | None:
    if waypoint is None:
        return None
    
    heading = getattr(waypoint, "heading", None)
    
    if heading is None: return None
    

    return _wrap_angle(-float(heading))


def lane_step_xy_heading(x_m: float, y_m: float, distance_m: float, *, get_waypoint_fn):
    """Advance along a custom lane center for obstacle prediction."""

    waypoint = get_waypoint_fn({"x": float(x_m), "y": float(y_m), "z": 0.0})
    if waypoint is None:
        return None
    if abs(float(distance_m)) <= 1.0e-6:
        candidate = waypoint
    else:
        stepper = waypoint.next if float(distance_m) >= 0.0 else waypoint.previous
        candidates = list(stepper(abs(float(distance_m))) or [])
        if not candidates:
            return None
        candidate = candidates[0]
    position = getattr(candidate, "position", None)
    if not isinstance(position, Mapping):
        return None
    return float(position["x"]), float(position["y"]), float(world_heading_rad(candidate) or 0.0)


def _point_dict(point: Mapping[str, object] | Sequence[object]) -> Dict[str, float]:
    if isinstance(point, Mapping):
        return {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "z": float(point.get("z", 0.0)),
        }
    if hasattr(point, "x") and hasattr(point, "y"):
        return {
            "x": float(getattr(point, "x")),
            "y": float(getattr(point, "y")),
            "z": float(getattr(point, "z", 0.0)),
        }
    if isinstance(point, Sequence) and not isinstance(point, (str, bytes)):
        if len(point) < 2:
            raise ValueError("A route point requires at least x and y.")
        return {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]) if len(point) >= 3 else 0.0,
        }
    raise TypeError("Planner points must be mappings or numeric sequences.")


class CustomGlobalPlannerAdapter:
    """Expose custom routing through the planning module's existing summary API."""

    def __init__(
        self,
        *,
        xodr_path: str,
        cache_root: str,
        route_sample_distance_m: float = 3.0,
        ad_map_install_root: str | None = None,
    ) -> None:
        self.core = GlobalPlanner(
            xodr_path=xodr_path,
            cache_root=cache_root,
            centerline_spacing_m=float(route_sample_distance_m),
            ad_map_install_root=ad_map_install_root,
        )
        self.route_sample_distance_m = max(0.5, float(route_sample_distance_m))
        self._stored_route_summary: RoutePlanSummary | None = None
        self._stored_route_xy: np.ndarray | None = None
        self._stored_route_cum_dists: np.ndarray | None = None
        self._stored_route_options: List[str] = []
        self._stored_route_lane_ids: List[int] = []
        self._stored_route_waypoints: List[Waypoint | None] = []
        self._query_indices: Dict[str, int] = {}
        self._lane_context_lock = threading.Lock()
        self._lane_context_cache: Tuple[float, float, float, Dict[str, object]] | None = None

    @property
    def blocked_lanes(self) -> List[int]:
        return self.core.blocked_lanes

    def load(self, force_rebuild: bool = False) -> None:
        self.core.load(force_rebuild=bool(force_rebuild))
        self.core.blocked_lanes.clear()

    def close(self) -> None:
        self.core.close()

    def get_waypoint(
        self,
        position: Mapping[str, object] | Sequence[object],
        search_radius_m: float | None = None,
    ) -> Waypoint | None:
        return self.core.get_waypoint(
            _point_dict(position),
            search_radius_m=search_radius_m,
        )

    @staticmethod
    def world_heading_rad(waypoint: Waypoint | None) -> float | None:
        return world_heading_rad(waypoint)

    def block_ad_lane_id(self, ad_lane_id: int) -> bool:
        normalized = int(ad_lane_id)
        # Validate through a public core API; do not access _lane_cache.
        self.core.get_lane_centerline(normalized)
        if normalized in self.core.blocked_lanes:
            return False
        self.core.blocked_lanes.append(normalized)
        return True

    def block_lane_at_position(
        self,
        position: Mapping[str, object] | Sequence[object],
    ) -> int | None:
        waypoint = self.get_waypoint(position)
        if waypoint is None:
            return None
        self.block_ad_lane_id(int(waypoint.ad_lane_id))
        return int(waypoint.ad_lane_id)

    def trace_route(
        self,
        start: Mapping[str, object] | Sequence[object],
        goal: Mapping[str, object] | Sequence[object],
        *,
        replace_stored_route: bool = False,
    ) -> RoutePlanSummary:
        start_point = _point_dict(start)
        goal_point = _point_dict(goal)
        try:
            route = self.core.trace_route(
                start_point,
                goal_point,
                sampling_resolution_m=self.route_sample_distance_m,
            )
        except Exception as exc:
            return self._failure_summary(start_point, goal_point, str(exc))
        summary = self._summary_from_route(route)
        if replace_stored_route and summary.route_found:
            self._store_route(summary, list(route.sampled_waypoints))
        return summary

    def plan_route_from_locations(
        self,
        *,
        start_location: Mapping[str, object] | Sequence[object],
        goal_location: Mapping[str, object] | Sequence[object],
        replace_stored_route: bool = False,
        **_unused,
    ) -> RoutePlanSummary:
        return self.trace_route(
            start_location,
            goal_location,
            replace_stored_route=replace_stored_route,
        )

    def register_imported_route(
        self,
        route_points: Sequence[Sequence[object]],
    ) -> RoutePlanSummary | None:
        points: List[List[float]] = []
        waypoints: List[Waypoint | None] = []
        for raw_point in route_points:
            point = _point_dict(raw_point)
            waypoint = self.get_waypoint(point)
            points.append([float(point["x"]), float(point["y"])])
            waypoints.append(waypoint)
        if len(points) < 2:
            return None
        summary = self._summary_from_samples(points, waypoints)
        self._store_route(summary, waypoints)
        return summary

    def replace_stored_route(
        self,
        summary: RoutePlanSummary,
        per_waypoint_options: Sequence[str] | None = None,
        per_waypoint_lane_ids: Sequence[int] | None = None,
    ) -> None:
        waypoints = [
            self.get_waypoint({"x": p[0], "y": p[1], "z": 0.0})
            for p in summary.route_waypoints
        ]
        self._store_route(
            summary,
            waypoints,
            options=per_waypoint_options,
            lane_ids=per_waypoint_lane_ids,
        )

    def nearest_waypoint_query(self, x_m: float, y_m: float) -> WaypointQueryResult | None:
        waypoint = self.get_waypoint({"x": x_m, "y": y_m, "z": 0.0})
        if waypoint is None:
            return None
        position = waypoint.position
        return WaypointQueryResult(
            index=-1,
            distance_m=math.hypot(float(position["x"]) - x_m, float(position["y"]) - y_m),
            x_m=float(position["x"]),
            y_m=float(position["y"]),
            road_id=f"{int(waypoint.road_id or 0)}:{int(waypoint.section_id or 0)}",
            lane_id=canonical_lane_id_for_waypoint(waypoint),
            direction="positive" if int(waypoint.lane_id or 0) > 0 else "negative",
        )

    def get_local_lane_context(
        self,
        x_m: float,
        y_m: float,
        heading_rad: float | None = None,
        z_m: float | None = None,
    ) -> Dict[str, object]:
        del heading_rad
        query_z = 0.0 if z_m is None else float(z_m)
        with self._lane_context_lock:
            cached = self._lane_context_cache
            if cached is not None and _distance_3d((x_m, y_m, query_z), cached[:3]) < 1.0:
                return dict(cached[3])

        waypoint = self.get_waypoint({"x": x_m, "y": y_m, "z": query_z})
        if waypoint is None:
            result = {
                "road_id": "unknown_road",
                "road_numeric_id": -1,
                "section_id": -1,
                "direction": "unknown",
                "lane_id": INVALID_LANE_ID,
                "lane_ids": [],
                "lane_count": 0,
                "min_lane_id": INVALID_LANE_ID,
                "max_lane_id": INVALID_LANE_ID,
                "can_change_left": False,
                "can_change_right": False,
                "heading_rad": None,
                "is_intersection": False,
                "lane_width_m": 3.5,
                "ad_lane_id": None,
                "opendrive_lane_id": 0,
            }
        else:
            lanes = canonical_lane_waypoints(waypoint)
            lane_ids = list(range(1, len(lanes) + 1))
            lane_id = canonical_lane_id_for_waypoint(waypoint)
            lane_index = lane_id - 1
            result = {
                "road_id": f"{int(waypoint.road_id or 0)}:{int(waypoint.section_id or 0)}",
                "road_numeric_id": int(waypoint.road_id or 0),
                "section_id": int(waypoint.section_id or 0),
                "direction": "positive" if int(waypoint.lane_id or 0) > 0 else "negative",
                "lane_id": int(lane_id),
                "lane_ids": lane_ids,
                "lane_count": len(lane_ids),
                "min_lane_id": min(lane_ids, default=INVALID_LANE_ID),
                "max_lane_id": max(lane_ids, default=INVALID_LANE_ID),
                "can_change_left": 0 <= lane_index < len(lanes) - 1,
                "can_change_right": lane_index > 0,
                "heading_rad": world_heading_rad(waypoint),
                "is_intersection": bool(waypoint.is_intersection),
                "lane_width_m": float(waypoint.lane_width_m or 3.5),
                "ad_lane_id": int(waypoint.ad_lane_id),
                "opendrive_lane_id": int(waypoint.lane_id or 0),
            }
        with self._lane_context_lock:
            self._lane_context_cache = (float(x_m), float(y_m), query_z, dict(result))
        return result

    def get_current_route_info(
        self,
        x_m: float,
        y_m: float,
        query_key: str = "default",
    ) -> RoutePlanSummary:
        if self._stored_route_summary is None or self._stored_route_xy is None:
            return self._failure_summary(
                {"x": x_m, "y": y_m, "z": 0.0},
                {"x": x_m, "y": y_m, "z": 0.0},
                "No route has been stored.",
            )
        index = self._nearest_stored_route_index(x_m, y_m, query_key)
        remaining = float(self._stored_route_cum_dists[-1] - self._stored_route_cum_dists[index])
        current_option = self._stored_route_options[index] if self._stored_route_options else "LANEFOLLOW"
        next_maneuver = self._next_macro_maneuver(self._stored_route_options, index)
        optimal_lane = self._optimal_lane_from_index(index)
        return replace(
            self._stored_route_summary,
            optimal_lane_id=int(optimal_lane),
            distance_to_destination_m=max(0.0, remaining),
            next_macro_maneuver=next_maneuver,
            current_road_option=current_option,
        )

    def _summary_from_route(self, route: Route) -> RoutePlanSummary:
        points = [
            [float(wp.position["x"]), float(wp.position["y"])]
            for wp in route.sampled_waypoints
        ]
        return self._summary_from_samples(points, list(route.sampled_waypoints), route.length_m)

    def _summary_from_samples(
        self,
        points: List[List[float]],
        waypoints: List[Waypoint | None],
        length_m: float | None = None,
    ) -> RoutePlanSummary:
        if len(points) < 2:
            return self._failure_summary(
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 0.0, "y": 0.0, "z": 0.0},
                "The route contains fewer than two points.",
            )
        start_wp = waypoints[0]
        goal_wp = waypoints[-1]
        lane_ids = [canonical_lane_id_for_waypoint(wp) for wp in waypoints]
        options = self._geometric_road_options(points)
        summary = RoutePlanSummary(
            route_found=True,
            start_road_id=self._road_key(start_wp),
            start_lane_id=canonical_lane_id_for_waypoint(start_wp),
            goal_road_id=self._road_key(goal_wp),
            goal_lane_id=canonical_lane_id_for_waypoint(goal_wp),
            optimal_lane_id=self._first_valid_lane_id(lane_ids),
            distance_to_destination_m=(
                float(length_m) if length_m is not None else self._polyline_length(points)
            ),
            next_macro_maneuver=self._next_macro_maneuver(options, 0),
            route_waypoints=points,
            road_options=options,
            current_road_option=options[0] if options else "LANEFOLLOW",
            start_graph_xy=(points[0][0], points[0][1]),
            goal_graph_xy=(points[-1][0], points[-1][1]),
        )
        return summary

    def _store_route(
        self,
        summary: RoutePlanSummary,
        waypoints: Sequence[Waypoint | None],
        *,
        options: Sequence[str] | None = None,
        lane_ids: Sequence[int] | None = None,
    ) -> None:
        self._stored_route_summary = summary
        self._stored_route_xy = np.asarray(summary.route_waypoints, dtype=float)
        self._stored_route_cum_dists = self._route_cumulative_distances(self._stored_route_xy)
        self._stored_route_waypoints = list(waypoints)
        self._stored_route_options = list(options or summary.road_options)
        self._stored_route_lane_ids = list(
            lane_ids
            or [canonical_lane_id_for_waypoint(wp) for wp in self._stored_route_waypoints]
        )
        self._query_indices.clear()

    def _nearest_stored_route_index(self, x_m: float, y_m: float, query_key: str) -> int:
        assert self._stored_route_xy is not None
        start_index = max(0, int(self._query_indices.get(str(query_key), 0)) - 5)
        candidate_xy = self._stored_route_xy[start_index:]
        distances_sq = (
            (candidate_xy[:, 0] - float(x_m)) ** 2
            + (candidate_xy[:, 1] - float(y_m)) ** 2
        )
        index = start_index + int(np.argmin(distances_sq))
        self._query_indices[str(query_key)] = index
        return index

    def _optimal_lane_from_index(self, index: int) -> int:
        for lane_id in self._stored_route_lane_ids[index:]:
            if int(lane_id) != INVALID_LANE_ID:
                return int(lane_id)
        return INVALID_LANE_ID

    @staticmethod
    def _road_key(waypoint: Waypoint | None) -> str:
        if waypoint is None:
            return "unknown_road"
        return f"{int(waypoint.road_id or 0)}:{int(waypoint.section_id or 0)}"

    @staticmethod
    def _first_valid_lane_id(lane_ids: Sequence[int]) -> int:
        return next((int(value) for value in lane_ids if int(value) != 0), 0)

    @staticmethod
    def _polyline_length(points: Sequence[Sequence[float]]) -> float:
        return sum(_distance_2d(a, b) for a, b in zip(points, points[1:]))

    @staticmethod
    def _route_cumulative_distances(route_xy: np.ndarray) -> np.ndarray:
        if len(route_xy) == 0:
            return np.asarray([], dtype=float)
        if len(route_xy) == 1:
            return np.asarray([0.0], dtype=float)
        segment_lengths = np.linalg.norm(np.diff(route_xy[:, :2], axis=0), axis=1)
        return np.concatenate(([0.0], np.cumsum(segment_lengths)))

    @classmethod
    def _geometric_road_options(cls, points: Sequence[Sequence[float]]) -> List[str]:
        options = ["LANEFOLLOW"] * len(points)
        for index in range(3, max(3, len(points) - 5)):
            before = math.atan2(
                float(points[index][1]) - float(points[index - 3][1]),
                float(points[index][0]) - float(points[index - 3][0]),
            )
            after = math.atan2(
                float(points[index + 5][1]) - float(points[index][1]),
                float(points[index + 5][0]) - float(points[index][0]),
            )
            delta = _wrap_angle(after - before)
            if abs(delta) >= math.radians(28.0):
                option = "LEFT" if delta > 0.0 else "RIGHT"
            elif abs(delta) >= math.radians(10.0):
                option = "STRAIGHT"
            else:
                continue
            for nearby in range(max(0, index - 2), min(len(options), index + 3)):
                options[nearby] = option
        return options

    @staticmethod
    def _next_macro_maneuver(options: Sequence[str], start_index: int) -> str:
        labels = {"LEFT": "Turn Left", "RIGHT": "Turn Right", "STRAIGHT": "Continue Straight"}
        for option in options[max(0, int(start_index)):]:
            if str(option).upper() in labels:
                return labels[str(option).upper()]
        return "Continue Straight"

    def _failure_summary(
        self,
        start: Mapping[str, float],
        goal: Mapping[str, float],
        reason: str,
    ) -> RoutePlanSummary:
        return RoutePlanSummary(
            route_found=False,
            start_road_id="unknown_road",
            start_lane_id=INVALID_LANE_ID,
            goal_road_id="unknown_road",
            goal_lane_id=INVALID_LANE_ID,
            optimal_lane_id=INVALID_LANE_ID,
            distance_to_destination_m=math.hypot(
                float(goal["x"]) - float(start["x"]),
                float(goal["y"]) - float(start["y"]),
            ),
            next_macro_maneuver="Continue Straight",
            route_waypoints=[],
            debug_reason=str(reason),
        )
