"""High-level global planner API built on top of AD-map."""

from __future__ import annotations

import heapq
import math
from pathlib import Path

from . import admap_backend as backend
from .cache import CACHE_VERSION, compute_xodr_signature, get_cache_paths, load_metadata, load_pickle, metadata_matches, save_metadata, save_pickle
from .geometry import as_carla_dict, euclidean_distance, normalize_position, points_are_close, to_carla_dict, to_enu_tuple
from .route import Route
from .waypoint import Waypoint


class GlobalPlanner:
    """Provide a CARLA-format planner API without importing CARLA in the core."""

    def __init__(
        self,
        xodr_path: str | Path,
        cache_root: str | Path = "cache",
        centerline_spacing_m: float = 3.0,
        route_engine: str = "custom",
        default_search_radius_m: float = 8.0,
        default_lane_change_penalty_m: float = 2.0,
        lane_change_distance_m: float = 8.0,
        overlap_margin: float = 0.05,
        ad_map_install_root: str | Path | None = None,
    ) -> None:
        """Store planner configuration without loading the map yet.

        input: planner settings (`str | Path | float`)
        output: none (`None`)
        """
        self.xodr_path = Path(xodr_path).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.centerline_spacing_m = float(centerline_spacing_m)
        self.route_engine = str(route_engine)
        self.default_search_radius_m = float(default_search_radius_m)
        self.default_lane_change_penalty_m = float(default_lane_change_penalty_m)
        self.lane_change_distance_m = float(lane_change_distance_m)
        self.overlap_margin = float(overlap_margin)
        self.ad_map_install_root = None if ad_map_install_root is None else Path(ad_map_install_root).resolve()

        self._loaded = False
        self._signature: dict | None = None
        self._cache_paths: dict[str, Path] | None = None
        self._lane_cache: dict[int, dict] = {}
        self.active_route: Route | None = None
        self.blocked_lanes: list[int] = []

        # Prepare the compiled AD-map runtime here so callers do not need to source shell scripts first.
        backend.ensure_runtime_ready(self.ad_map_install_root)

    def __enter__(self) -> "GlobalPlanner":
        """Load the planner when entering a context manager.

        input: none (`None`)
        output: loaded planner (`GlobalPlanner`)
        """
        self.load()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Unload AD-map singleton state when leaving a context manager.

        input: context-manager exception data (`object`)
        output: none (`None`)
        """
        self.close()

    def load(self, force_rebuild: bool = False) -> None:
        """Load the map and planner cache, rebuilding them when required.

        input: `force_rebuild` (`bool`)
        output: none (`None`)
        """
        if self.route_engine != "custom":
            raise NotImplementedError(f"Unsupported route engine: {self.route_engine}")

        backend.ensure_runtime_ready(self.ad_map_install_root)
        self._signature = compute_xodr_signature(self.xodr_path)
        self._cache_paths = get_cache_paths(self.xodr_path, self.cache_root, self._signature)
        metadata = load_metadata(self._cache_paths["metadata_file"])
        planner_cache_is_valid = (
            not force_rebuild
            and self._cache_paths["planner_cache_file"].exists()
            and metadata_matches(metadata, self._signature, self.centerline_spacing_m)
        )
        use_saved_adm_cache = (
            planner_cache_is_valid
            and backend.supports_adm_cache()
            and self._cache_paths["adm_config_file"].exists()
        )

        backend.close_map()
        if use_saved_adm_cache:
            backend.load_adm_map(self._cache_paths["adm_config_file"])
            self._lane_cache = load_pickle(self._cache_paths["planner_cache_file"])
        else:
            backend.load_open_drive_map(self.xodr_path, overlap_margin=self.overlap_margin)
            if planner_cache_is_valid:
                self._lane_cache = load_pickle(self._cache_paths["planner_cache_file"])
            else:
                self._lane_cache = self._build_lane_cache()
                save_pickle(self._cache_paths["planner_cache_file"], self._lane_cache)
                save_metadata(
                    self._cache_paths["metadata_file"],
                    {
                        "cache_version": CACHE_VERSION,
                        "centerline_spacing_m": self.centerline_spacing_m,
                        "xodr_signature": self._signature,
                    },
                )
            backend.save_adm_map(self._cache_paths["adm_file"])
        self._loaded = True

    def close(self) -> None:
        """Unload the active AD-map singleton state.

        input: none (`None`)
        output: none (`None`)
        """
        backend.close_map()
        self._loaded = False
        self.active_route = None

    def to_enu(self, position) -> tuple[float, float, float]:
        """Convert one CARLA-world point into the planner's ENU convention.

        input: `position` (`dict[str, float] | tuple[float, float, float]`)
        output: ENU point (`tuple[float, float, float]`)
        """
        return to_enu_tuple(position)

    def to_carla(self, position) -> dict[str, float]:
        """Convert one ENU point into CARLA-world coordinates.

        input: `position` (`dict[str, float] | tuple[float, float, float]`)
        output: CARLA point (`dict[str, float]`)
        """
        return to_carla_dict(position)

    def get_waypoint(self, position, search_radius_m: float | None = None) -> Waypoint | None:
        """Return the closest lane-center waypoint to the given CARLA-world point.

        input: `position` (`dict[str, float] | tuple[float, float, float]`), `search_radius_m` (`float | None`)
        output: snapped waypoint or no result (`Waypoint | None`)
        """
        self._ensure_loaded()
        enu_point = backend.create_enu_point(self.to_enu(position))
        candidates = self._build_route_candidates(
            enu_point,
            search_radius_m if search_radius_m is not None else self.default_search_radius_m,
        )
        if not candidates:
            return None
        best_candidate = candidates[0]
        return self._make_waypoint_from_lane_offset(
            best_candidate["lane_id"],
            best_candidate["parametric_offset"],
            enu_position=best_candidate["center_point"],
        )

    def get_lane_centerline(self, lane_id: int) -> list[Waypoint]:
        """Return cached lane-center waypoints for one AD lane id.

        input: `lane_id` (`int`)
        output: lane-center waypoints (`list[Waypoint]`)
        """
        self._ensure_loaded()
        lane_data = self._lane_cache.get(int(lane_id))
        if lane_data is None:
            raise KeyError(f"Unknown lane id: {lane_id}")
        return [
            self._make_waypoint_from_lane_offset(
                int(lane_id),
                sample["offset"],
                enu_position=sample["enu_position"],
            )
            for sample in lane_data["centerline_samples"]
        ]

    def trace_route(
        self,
        start,
        goal,
        sampling_resolution_m: float = 3.0,
        search_radius_m: float | None = None,
        lane_change_penalty_m: float | None = None,
        goal_append_distance_threshold_m: float = 1.0,
    ) -> Route:
        """Plan the shortest custom route between two CARLA-world points.

        input: route query points and planner settings (`dict[str, float] | tuple[float, float, float] | float`)
        output: planned route (`Route`)
        """
        self._ensure_loaded()
        start_carla = as_carla_dict(start)
        goal_carla = as_carla_dict(goal)
        start_enu = backend.create_enu_point(self.to_enu(start_carla))
        goal_enu = backend.create_enu_point(self.to_enu(goal_carla))
        route_data, route_length_m, resolved_start_enu, resolved_goal_enu = self._find_shortest_route(
            start_enu,
            goal_enu,
            search_radius=search_radius_m if search_radius_m is not None else self.default_search_radius_m,
            lane_change_penalty_m=(
                lane_change_penalty_m if lane_change_penalty_m is not None else self.default_lane_change_penalty_m
            ),
            lane_change_distance_m=self.lane_change_distance_m,
        )
        sampled_waypoints = self._sample_route_centerline(
            route_data,
            spacing=float(sampling_resolution_m),
            start_enu_point=resolved_start_enu,
            goal_enu_point=resolved_goal_enu,
            goal_append_distance_threshold_m=goal_append_distance_threshold_m,
        )
        route = Route(
            raw_start=start_carla,
            raw_goal=goal_carla,
            resolved_start=self._make_waypoint_from_enu_position(resolved_start_enu),
            resolved_goal=self._make_waypoint_from_enu_position(resolved_goal_enu),
            lane_path=list(route_data["lane_path"]),
            sampled_waypoints=sampled_waypoints,
            length_m=float(route_length_m),
            sampling_resolution_m=float(sampling_resolution_m),
            transition_types=list(route_data["transition_types"]),
        )
        self.active_route = route
        return route

    def find_opt_lane(self, position, search_radius_m: float | None = None) -> int | None:
        """Return the route lane id used by the active global route near one position.

        input: `position` (`dict[str, float] | tuple[float, float, float]`), `search_radius_m` (`float | None`)
        output: route lane id near the query or no result (`int | None`)
        """
        self._ensure_loaded()
        if self.active_route is None or len(self.active_route.sampled_waypoints) == 0:
            return None

        query_waypoint = self.get_waypoint(
            position,
            search_radius_m=search_radius_m if search_radius_m is not None else self.default_search_radius_m,
        )
        if query_waypoint is None:
            return None

        snapped_query_position = normalize_position(query_waypoint.position)
        route_waypoints = [
            waypoint
            for waypoint in self.active_route.sampled_waypoints
            if waypoint.road_id == query_waypoint.road_id
            and waypoint.section_id == query_waypoint.section_id
        ]
        if len(route_waypoints) == 0:
            route_waypoints = [
                waypoint
                for waypoint in self.active_route.sampled_waypoints
                if waypoint.road_id == query_waypoint.road_id
            ]
        if len(route_waypoints) == 0:
            return None

        best_route_waypoint = min(
            route_waypoints,
            key=lambda waypoint: euclidean_distance(
                normalize_position(waypoint.position),
                snapped_query_position,
            ),
        )
        return None if best_route_waypoint.lane_id is None else int(best_route_waypoint.lane_id)

    def _ensure_loaded(self) -> None:
        """Require the planner to be loaded before serving queries.

        input: none (`None`)
        output: none (`None`)
        """
        if not self._loaded:
            raise RuntimeError("The planner is not loaded. Call load() first.")

    def _build_lane_cache(self) -> dict[int, dict]:
        """Build the Python-side cache for lane metadata and centerline samples.

        input: none (`None`)
        output: lane cache (`dict[int, dict[str, object]]`)
        """
        lane_cache: dict[int, dict] = {}
        for lane_id in backend.get_routable_lane_ids():
            lane_cache[lane_id] = {
                **backend.get_opendrive_lane_info(lane_id),
                "length_m": backend.get_lane_length_m(lane_id),
                "direction_positive": backend.lane_is_positive(lane_id),
                "is_intersection": backend.lane_is_intersection(lane_id),
                "left_lane_id": backend.get_same_direction_adjacent_lane_id(
                    lane_id, backend.ad.map.lane.ContactLocation.LEFT
                ),
                "right_lane_id": backend.get_same_direction_adjacent_lane_id(
                    lane_id, backend.ad.map.lane.ContactLocation.RIGHT
                ),
                "forward_lane_ids": backend.get_contact_lane_ids(
                    lane_id,
                    backend.get_forward_contact_location(lane_id),
                    require_same_direction=True,
                ),
                "backward_lane_ids": backend.get_contact_lane_ids(
                    lane_id,
                    backend.get_backward_contact_location(lane_id),
                    require_same_direction=True,
                ),
                "centerline_samples": self._sample_lane_centerline(lane_id, self.centerline_spacing_m),
            }
        return lane_cache

    def _sample_lane_centerline(self, lane_id: int, spacing: float) -> list[dict]:
        """Sample the full centerline of one lane for cache storage.

        input: `lane_id` (`int`), `spacing` (`float`)
        output: centerline samples (`list[dict[str, object]]`)
        """
        samples = []
        start_offset = backend.get_travel_start_offset(lane_id)
        end_offset = backend.get_travel_end_offset(lane_id)
        for offset in self._sample_lane_offsets(lane_id, start_offset, end_offset, spacing):
            samples.append(
                {
                    "offset": offset,
                    "enu_position": backend.sample_lane_center_at_offset(lane_id, offset),
                }
            )
        return samples

    def _build_route_candidates(self, enu_point, search_radius: float) -> list[dict]:
        """Turn nearby lane matches into sortable snapped routing candidates.

        input: `enu_point` (`ad.map.point.ENUPoint`), `search_radius` (`float`)
        output: route candidates (`list[dict[str, object]]`)
        """
        matches = backend.get_map_matches(enu_point, search_radius)
        raw_point = backend.enu_point_to_tuple(enu_point)
        candidates = []
        for match in matches:
            lane_id = backend.get_match_lane_id(match)
            parametric_offset = backend.get_match_parametric_offset(match)
            center_point = backend.sample_lane_center_at_offset(lane_id, parametric_offset)
            candidates.append(
                {
                    "match": match,
                    "lane_id": lane_id,
                    "parametric_offset": parametric_offset,
                    "center_point": center_point,
                    "snap_distance": euclidean_distance(raw_point, center_point),
                    "is_in_lane": backend.match_is_in_lane(match),
                    "probability": backend.get_match_probability(match),
                }
            )
        candidates.sort(
            key=lambda candidate: (
                candidate["snap_distance"],
                not candidate["is_in_lane"],
                -candidate["probability"],
                candidate["lane_id"],
            )
        )
        return candidates

    def _blocked_lane_id_set(self) -> set[int]:
        """Return the currently blocked AD lane ids as a normalized set.

        input: none (`None`)
        output: blocked lane ids (`set[int]`)
        """
        return {int(lane_id) for lane_id in self.blocked_lanes}

    def _lane_is_blocked(self, lane_id: int) -> bool:
        """Check whether one AD lane id is currently blocked.

        input: `lane_id` (`int`)
        output: whether the lane is blocked (`bool`)
        """
        return int(lane_id) in self._blocked_lane_id_set()

    def _filter_route_candidates(self, candidates: list[dict]) -> list[dict]:
        """Drop blocked lanes from route-start and route-goal candidates.

        input: snapped route candidates (`list[dict[str, object]]`)
        output: usable route candidates (`list[dict[str, object]]`)
        """
        blocked_lane_ids = self._blocked_lane_id_set()
        if len(blocked_lane_ids) == 0:
            return list(candidates)
        return [
            candidate
            for candidate in candidates
            if int(candidate["lane_id"]) not in blocked_lane_ids
        ]

    def _get_adjacent_waypoint(self, waypoint: Waypoint, side: str) -> Waypoint | None:
        """Return one adjacent-lane waypoint on the requested side.

        input: `waypoint` (`Waypoint`), `side` (`str`)
        output: adjacent waypoint or no result (`Waypoint | None`)
        """
        lane_data = self._lane_cache.get(waypoint.ad_lane_id)
        if lane_data is None:
            return None
        key = "left_lane_id" if side == "left" else "right_lane_id"
        adjacent_lane_id = lane_data.get(key)
        if adjacent_lane_id is None:
            return None
        return self._make_waypoint_from_lane_offset(adjacent_lane_id, waypoint.parametric_offset)

    def _step_waypoint(self, waypoint: Waypoint, distance_m: float, forward: bool) -> list[Waypoint]:
        """Move forward or backward along the lane network from one waypoint.

        input: `waypoint` (`Waypoint`), `distance_m` (`float`), `forward` (`bool`)
        output: reachable waypoints (`list[Waypoint]`)
        """
        if distance_m < 0.0:
            raise ValueError("Waypoint step distance must be zero or greater.")
        lane_targets = self._advance_along_lane(
            waypoint.ad_lane_id,
            waypoint.parametric_offset,
            distance_m,
            forward=forward,
        )
        return [self._make_waypoint_from_lane_offset(lane_id, offset) for lane_id, offset in lane_targets]

    def _advance_along_lane(
        self,
        lane_id: int,
        start_offset: float,
        remaining_distance_m: float,
        forward: bool,
    ) -> list[tuple[int, float]]:
        """Advance along the lane network and return reachable lane-offset targets.

        input: lane position and travel settings (`int | float | bool`)
        output: reachable lane-offset pairs (`list[tuple[int, float]]`)
        """
        lane_length_m = backend.get_lane_length_m(lane_id)
        if lane_length_m <= 0.0:
            return []

        boundary_offset = self._get_boundary_offset(lane_id, forward)
        distance_to_boundary = abs(boundary_offset - start_offset) * lane_length_m
        if remaining_distance_m <= distance_to_boundary + 1e-6:
            return [(lane_id, self._advance_offset(lane_id, start_offset, remaining_distance_m, forward))]

        next_lane_ids = self._get_longitudinal_neighbors(lane_id, forward)
        if not next_lane_ids:
            return []

        remaining_distance_m -= distance_to_boundary
        next_offset = self._get_entry_offset(forward)
        targets = []
        for next_lane_id in next_lane_ids:
            entry_offset = next_offset(next_lane_id)
            targets.extend(self._advance_along_lane(next_lane_id, entry_offset, remaining_distance_m, forward))
        return targets

    def _get_boundary_offset(self, lane_id: int, forward: bool) -> float:
        """Return the exit offset used for forward or backward stepping.

        input: `lane_id` (`int`), `forward` (`bool`)
        output: boundary offset (`float`)
        """
        if forward:
            return backend.get_travel_end_offset(lane_id)
        return backend.get_travel_start_offset(lane_id)

    def _get_entry_offset(self, forward: bool):
        """Return the helper used to choose the next-lane entry offset.

        input: `forward` (`bool`)
        output: offset helper (`callable`)
        """
        if forward:
            return backend.get_travel_start_offset
        return backend.get_travel_end_offset

    def _get_longitudinal_neighbors(self, lane_id: int, forward: bool) -> list[int]:
        """Return same-direction longitudinally connected lanes for stepping.

        input: `lane_id` (`int`), `forward` (`bool`)
        output: connected lane ids (`list[int]`)
        """
        if forward:
            return list(self._lane_cache[lane_id]["forward_lane_ids"])
        return list(self._lane_cache[lane_id]["backward_lane_ids"])

    def _advance_offset(self, lane_id: int, start_offset: float, distance_m: float, forward: bool) -> float:
        """Advance one lane offset by a physical distance.

        input: `lane_id` (`int`), `start_offset` (`float`), `distance_m` (`float`), `forward` (`bool`)
        output: shifted offset (`float`)
        """
        lane_length_m = max(backend.get_lane_length_m(lane_id), 1e-6)
        offset_delta = max(distance_m, 0.0) / lane_length_m
        if forward:
            if backend.lane_is_positive(lane_id):
                return min(1.0, start_offset + offset_delta)
            return max(0.0, start_offset - offset_delta)
        if backend.lane_is_positive(lane_id):
            return max(0.0, start_offset - offset_delta)
        return min(1.0, start_offset + offset_delta)

    def _make_waypoint_from_enu_position(self, enu_position) -> Waypoint:
        """Create one waypoint by snapping the given ENU point to the nearest lane center.

        input: `enu_position` (`ad.map.point.ENUPoint | tuple[float, float, float]`)
        output: waypoint (`Waypoint`)
        """
        enu_tuple = backend.enu_point_to_tuple(enu_position) if hasattr(enu_position, "x") else normalize_position(enu_position)
        waypoint = self.get_waypoint(to_carla_dict(enu_tuple), search_radius_m=self.default_search_radius_m)
        if waypoint is None:
            raise RuntimeError("No drivable lane found near the requested point.")
        return waypoint

    def _make_waypoint_from_lane_offset(
        self,
        lane_id: int,
        parametric_offset: float,
        enu_position: tuple[float, float, float] | None = None,
    ) -> Waypoint:
        """Create one waypoint from a known lane id and parametric offset.

        input: `lane_id` (`int`), `parametric_offset` (`float`), `enu_position` (`tuple[float, float, float] | None`)
        output: waypoint (`Waypoint`)
        """
        lane_id = int(lane_id)
        lane_data = self._lane_cache[lane_id]
        point_enu = enu_position or backend.sample_lane_center_at_offset(lane_id, parametric_offset)
        return Waypoint(
            position=to_carla_dict(point_enu),
            enu_position=point_enu,
            ad_lane_id=lane_id,
            road_id=lane_data["road_id"],
            section_id=lane_data["section_id"],
            lane_id=lane_data["lane_id"],
            parametric_offset=float(parametric_offset),
            heading=backend.get_lane_heading(lane_id, parametric_offset),
            lane_length_m=lane_data["length_m"],
            lane_width_m=backend.get_lane_width_m(lane_id, parametric_offset),
            is_intersection=lane_data["is_intersection"],
            _planner=self,
        )

    def _find_shortest_route(
        self,
        start_enu_point,
        goal_enu_point,
        search_radius: float,
        lane_change_penalty_m: float,
        lane_change_distance_m: float,
    ) -> tuple[dict, float, tuple[float, float, float], tuple[float, float, float]]:
        """Route between legal start and goal lane candidates while preserving main snaps.

        input: snapped ENU query points and route settings
        output: route data, route length, snapped start, snapped goal (`tuple`)
        """
        start_candidates = self._build_route_candidates(start_enu_point, search_radius)
        goal_candidates = self._build_route_candidates(goal_enu_point, search_radius)
        if not start_candidates:
            raise RuntimeError("No drivable lane found near the start point.")
        if not goal_candidates:
            raise RuntimeError("No drivable lane found near the goal point.")

        start_candidates = self._filter_route_candidates(start_candidates)
        goal_candidates = self._filter_route_candidates(goal_candidates)
        if not start_candidates:
            raise RuntimeError("No unblocked drivable start lane found near the start point.")
        if not goal_candidates:
            raise RuntimeError("No unblocked drivable goal lane found near the goal point.")

        resolved_start_enu = start_candidates[0]["center_point"]
        resolved_goal_enu = goal_candidates[0]["center_point"]
        best_route = None
        best_score = None

        for start_candidate in start_candidates:
            for goal_candidate in goal_candidates:
                try:
                    route_data = self._find_route_from_snapped_matches(
                        start_candidate["match"],
                        goal_candidate["match"],
                        lane_change_penalty_m=lane_change_penalty_m,
                        lane_change_distance_m=lane_change_distance_m,
                    )
                except RuntimeError:
                    continue

                score = (
                    start_candidate["snap_distance"] + goal_candidate["snap_distance"],
                    route_data["route_length_m"],
                    not start_candidate["is_in_lane"],
                    not goal_candidate["is_in_lane"],
                    -(start_candidate["probability"] + goal_candidate["probability"]),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_route = route_data

        if best_route is None:
            raise RuntimeError("No legal route exists between any nearby start and goal lane candidates.")

        return (best_route, best_route["route_length_m"], resolved_start_enu, resolved_goal_enu)

    def _find_route_from_snapped_matches(
        self,
        start_match,
        goal_match,
        lane_change_penalty_m: float,
        lane_change_distance_m: float,
    ) -> dict:
        """Build one legal route between two already-snapped lane matches.

        input: snapped matches and route settings
        output: route data (`dict[str, object]`)
        """
        start_lane_id = backend.get_match_lane_id(start_match)
        goal_lane_id = backend.get_match_lane_id(goal_match)
        start_offset = backend.get_match_parametric_offset(start_match)
        goal_offset = backend.get_match_parametric_offset(goal_match)
        lane_path, transition_types = self._find_lane_path(
            start_lane_id,
            goal_lane_id,
            start_offset,
            goal_offset,
            lane_change_penalty_m=lane_change_penalty_m,
        )
        if lane_path is None or transition_types is None:
            raise RuntimeError("No legal route exists between the snapped start and goal lanes.")
        return self._build_custom_route(
            start_match,
            goal_match,
            lane_path,
            transition_types,
            lane_change_distance_m=lane_change_distance_m,
        )

    def _find_lane_path(
        self,
        start_lane_id: int,
        goal_lane_id: int,
        start_offset: float,
        goal_offset: float,
        lane_change_penalty_m: float,
    ) -> tuple[list[int] | None, list[str] | None]:
        """Run the custom lane-level shortest-path search.

        input: start and goal lane state plus route settings
        output: lane path and transition types (`tuple[list[int] | None, list[str] | None]`)
        """
        if self._lane_is_blocked(start_lane_id) or self._lane_is_blocked(goal_lane_id):
            return (None, None)
        direct_finish_allowed = (
            start_lane_id == goal_lane_id
            and backend.lane_offsets_move_forward(start_lane_id, start_offset, goal_offset)
        )
        frontier = [(0.0, start_lane_id, [start_lane_id], [], False)]
        best_costs = {(start_lane_id, False): 0.0}

        while frontier:
            current_cost, lane_id, lane_path, transition_types, has_left_start = heapq.heappop(frontier)
            state_key = (lane_id, has_left_start)
            if current_cost != best_costs.get(state_key):
                continue
            if lane_id == goal_lane_id and (has_left_start or direct_finish_allowed):
                return (lane_path, transition_types)

            for transition in self._get_allowed_lane_transitions(lane_id, lane_change_penalty_m):
                next_lane_id = transition["to_lane_id"]
                next_cost = current_cost + transition["cost_m"]
                next_state_key = (next_lane_id, True)
                if next_cost >= best_costs.get(next_state_key, float("inf")):
                    continue
                best_costs[next_state_key] = next_cost
                heapq.heappush(
                    frontier,
                    (
                        next_cost,
                        next_lane_id,
                        lane_path + [next_lane_id],
                        transition_types + [transition["type"]],
                        True,
                    ),
                )
        return (None, None)

    def _get_allowed_lane_transitions(self, lane_id: int, lane_change_penalty_m: float) -> list[dict]:
        """List legal route transitions from one lane.

        input: `lane_id` (`int`), `lane_change_penalty_m` (`float`)
        output: legal transitions (`list[dict[str, object]]`)
        """
        transitions = []
        forward_location = backend.get_forward_contact_location(lane_id)
        lane = backend.get_lane(lane_id)
        for contact_lane in backend.get_contact_lanes(lane):
            next_lane_id = backend.get_contact_target_lane_id(contact_lane)
            if next_lane_id <= 0 or not backend.is_routeable_lane(next_lane_id):
                continue
            if self._lane_is_blocked(next_lane_id):
                continue
            if contact_lane.location == forward_location:
                transitions.append(
                    {
                        "to_lane_id": next_lane_id,
                        "type": "forward",
                        "cost_m": backend.get_lane_length_m(next_lane_id),
                    }
                )
                continue
            if (
                not backend.lane_is_intersection(lane_id)
                and not backend.lane_is_intersection(next_lane_id)
                and contact_lane.location
                in (
                    backend.ad.map.lane.ContactLocation.LEFT,
                    backend.ad.map.lane.ContactLocation.RIGHT,
                )
                and backend.lanes_have_same_direction(lane_id, next_lane_id)
            ):
                transitions.append(
                    {
                        "to_lane_id": next_lane_id,
                        "type": "lane_change",
                        "cost_m": lane_change_penalty_m,
                    }
                )
        return transitions

    def _build_custom_route(
        self,
        start_match,
        goal_match,
        lane_path: list[int],
        transition_types: list[str],
        lane_change_distance_m: float,
    ) -> dict:
        """Turn one lane-id path into route data that can be sampled later.

        input: snapped matches, lane path, transition types, and lane-change settings
        output: custom route data (`dict[str, object]`)
        """
        start_lane_id = backend.get_match_lane_id(start_match)
        goal_lane_id = backend.get_match_lane_id(goal_match)
        start_offset = backend.get_match_parametric_offset(start_match)
        goal_offset = backend.get_match_parametric_offset(goal_match)
        route_steps = []
        current_offset = start_offset
        lane_change_indices = [
            index for index, transition_type in enumerate(transition_types) if transition_type == "lane_change"
        ]
        final_lateral_lane_changes = set(lane_change_indices[-2:])

        for index, lane_id in enumerate(lane_path):
            is_last_lane = index == len(lane_path) - 1
            if is_last_lane:
                if not backend.lane_offsets_move_forward(lane_id, current_offset, goal_offset):
                    raise RuntimeError("The goal point is behind the selected lane path direction.")
                route_steps.append(
                    {
                        "type": "lane_segment",
                        "lane_id": lane_id,
                        "start_offset": current_offset,
                        "end_offset": goal_offset,
                    }
                )
                break

            next_lane_id = lane_path[index + 1]
            transition_type = transition_types[index]
            if transition_type == "forward":
                end_offset = backend.get_travel_end_offset(lane_id)
                if not backend.lane_offsets_move_forward(lane_id, current_offset, end_offset):
                    raise RuntimeError("The selected route would move backward on the current lane.")
                route_steps.append(
                    {
                        "type": "lane_segment",
                        "lane_id": lane_id,
                        "start_offset": current_offset,
                        "end_offset": end_offset,
                    }
                )
                current_offset = backend.get_travel_start_offset(next_lane_id)
                continue

            if transition_type != "lane_change":
                raise RuntimeError(f"Unsupported lane transition type: {transition_type}")

            is_goal_lane = index + 1 == len(lane_path) - 1 and next_lane_id == goal_lane_id
            lane_change_from_offset, lane_change_to_offset = self._choose_lane_change_offsets(
                lane_id,
                next_lane_id,
                current_offset,
                goal_offset if is_goal_lane else None,
                lane_change_distance_m=lane_change_distance_m,
                use_lateral=index in final_lateral_lane_changes,
            )
            if not backend.lane_offsets_move_forward(lane_id, current_offset, lane_change_from_offset):
                raise RuntimeError("The selected lane change would move backward on the lane.")
            route_steps.append(
                {
                    "type": "lane_segment",
                    "lane_id": lane_id,
                    "start_offset": current_offset,
                    "end_offset": lane_change_from_offset,
                }
            )
            route_steps.append(
                {
                    "type": "lane_change",
                    "from_lane_id": lane_id,
                    "to_lane_id": next_lane_id,
                    "from_offset": lane_change_from_offset,
                    "to_offset": lane_change_to_offset,
                }
            )
            current_offset = lane_change_to_offset

        route_data = {
            "lane_path": lane_path,
            "transition_types": transition_types,
            "steps": route_steps,
            "start_lane_id": start_lane_id,
            "goal_lane_id": goal_lane_id,
            "start_offset": start_offset,
            "goal_offset": goal_offset,
        }
        route_data["route_length_m"] = self._get_custom_route_length(route_data)
        return route_data

    def _get_custom_route_length(self, route_data: dict) -> float:
        """Measure the analytical length of one custom route.

        input: `route_data` (`dict[str, object]`)
        output: route length in meters (`float`)
        """
        total_length_m = 0.0
        for step in route_data["steps"]:
            if step["type"] == "lane_segment":
                total_length_m += self._get_lane_segment_length_m(
                    step["lane_id"],
                    step["start_offset"],
                    step["end_offset"],
                )
                continue
            start_point = backend.sample_lane_center_at_offset(step["from_lane_id"], step["from_offset"])
            end_point = backend.sample_lane_center_at_offset(step["to_lane_id"], step["to_offset"])
            total_length_m += euclidean_distance(start_point, end_point)
        return total_length_m

    def _get_lane_segment_length_m(self, lane_id: int, start_offset: float, end_offset: float) -> float:
        """Measure one traveled lane segment.

        input: `lane_id` (`int`), `start_offset` (`float`), `end_offset` (`float`)
        output: traveled lane distance in meters (`float`)
        """
        return abs(end_offset - start_offset) * backend.get_lane_length_m(lane_id)

    def _advance_offset_forward(self, lane_id: int, start_offset: float, distance_m: float) -> float:
        """Move one lane offset forward along legal driving direction.

        input: `lane_id` (`int`), `start_offset` (`float`), `distance_m` (`float`)
        output: shifted offset (`float`)
        """
        return self._advance_offset(lane_id, start_offset, distance_m, forward=True)

    def _choose_lane_change_offset(
        self,
        lane_id: int,
        start_offset: float,
        goal_offset: float | None,
        lane_change_distance_m: float,
    ) -> float:
        """Choose the along-lane point where a lane change starts.

        input: current lane state and lane-change settings
        output: lane-change start offset (`float`)
        """
        offset_delta = lane_change_distance_m / max(backend.get_lane_length_m(lane_id), lane_change_distance_m)
        if backend.lane_is_positive(lane_id):
            lane_change_offset = min(0.95, start_offset + max(offset_delta, (1.0 - start_offset) * 0.35))
            if goal_offset is not None:
                lane_change_offset = min(lane_change_offset, max(start_offset + 0.02, goal_offset - 0.02))
            return lane_change_offset

        lane_change_offset = max(0.05, start_offset - max(offset_delta, start_offset * 0.35))
        if goal_offset is not None:
            lane_change_offset = max(lane_change_offset, min(start_offset - 0.02, goal_offset + 0.02))
        return lane_change_offset

    def _choose_lane_change_offsets(
        self,
        lane_id: int,
        next_lane_id: int,
        start_offset: float,
        goal_offset: float | None,
        lane_change_distance_m: float,
        use_lateral: bool,
    ) -> tuple[float, float]:
        """Choose the start and landing offsets for one lane change.

        input: lane-change state and tuning values
        output: current-lane and target-lane offsets (`tuple[float, float]`)
        """
        from_offset = self._choose_lane_change_offset(
            lane_id,
            start_offset,
            goal_offset,
            lane_change_distance_m,
        )
        if use_lateral:
            return (from_offset, from_offset)

        landing_distance_m = lane_change_distance_m * 0.45
        to_offset = self._advance_offset_forward(next_lane_id, from_offset, landing_distance_m)
        if goal_offset is not None:
            if backend.lane_is_positive(next_lane_id):
                to_offset = min(to_offset, goal_offset)
            else:
                to_offset = max(to_offset, goal_offset)
        return (from_offset, to_offset)

    def _sample_lane_offsets(self, lane_id: int, start_offset: float, end_offset: float, spacing: float) -> list[float]:
        """Create evenly spaced longitudinal offsets on one lane.

        input: `lane_id` (`int`), `start_offset` (`float`), `end_offset` (`float`), `spacing` (`float`)
        output: sampled offsets (`list[float]`)
        """
        lane_length_m = max(backend.get_lane_length_m(lane_id), spacing)
        offset_step = spacing / lane_length_m
        sampled_offsets = [start_offset]

        if backend.lane_is_positive(lane_id):
            current_offset = start_offset + offset_step
            while current_offset < end_offset - 1e-6:
                sampled_offsets.append(current_offset)
                current_offset += offset_step
        else:
            current_offset = start_offset - offset_step
            while current_offset > end_offset + 1e-6:
                sampled_offsets.append(current_offset)
                current_offset -= offset_step

        if abs(sampled_offsets[-1] - end_offset) > 1e-6:
            sampled_offsets.append(end_offset)
        return sampled_offsets

    def _interpolate_points(
        self,
        first_point: tuple[float, float, float],
        second_point: tuple[float, float, float],
        spacing: float,
    ) -> list[tuple[float, float, float]]:
        """Create short straight connector points between two route samples.

        input: connector endpoints (`tuple[float, float, float]`), `spacing` (`float`)
        output: interpolated connector points (`list[tuple[float, float, float]]`)
        """
        connector_length_m = euclidean_distance(first_point, second_point)
        connector_count = max(2, int(connector_length_m / spacing) + 1)
        return [
            tuple(
                first_value + (second_value - first_value) * (step / connector_count)
                for first_value, second_value in zip(first_point, second_point)
            )
            for step in range(1, connector_count + 1)
        ]

    def _append_sampled_waypoint(
        self,
        sampled_waypoints: list[Waypoint],
        point: tuple[float, float, float],
        lane_id: int,
        parametric_offset: float,
    ) -> None:
        """Append one sampled route point while skipping duplicates.

        input: route sample list plus point data
        output: none (`None`)
        """
        if sampled_waypoints and points_are_close(sampled_waypoints[-1].enu_position, point):
            return
        sampled_waypoints.append(
            self._make_waypoint_from_lane_offset(
                lane_id,
                parametric_offset,
                enu_position=point,
            )
        )

    def _sample_route_centerline(
        self,
        route_data: dict,
        spacing: float,
        start_enu_point: tuple[float, float, float],
        goal_enu_point: tuple[float, float, float],
        goal_append_distance_threshold_m: float,
    ) -> list[Waypoint]:
        """Sample one custom route into CARLA-format route waypoints.

        input: route data, sampling settings, and resolved endpoint snaps
        output: sampled route waypoints (`list[Waypoint]`)
        """
        if spacing <= 0.0:
            raise ValueError("Route-point spacing must be greater than zero.")
        if goal_append_distance_threshold_m < 0.0:
            raise ValueError("Goal append distance threshold must be zero or greater.")

        sampled_waypoints: list[Waypoint] = []
        for step in route_data["steps"]:
            if step["type"] == "lane_segment":
                for offset in self._sample_lane_offsets(
                    step["lane_id"],
                    step["start_offset"],
                    step["end_offset"],
                    spacing,
                ):
                    self._append_sampled_waypoint(
                        sampled_waypoints,
                        backend.sample_lane_center_at_offset(step["lane_id"], offset),
                        step["lane_id"],
                        offset,
                    )
                continue

            connector_start = backend.sample_lane_center_at_offset(step["from_lane_id"], step["from_offset"])
            connector_end = backend.sample_lane_center_at_offset(step["to_lane_id"], step["to_offset"])
            for connector_point in self._interpolate_points(connector_start, connector_end, spacing):
                self._append_sampled_waypoint(
                    sampled_waypoints,
                    connector_point,
                    step["to_lane_id"],
                    step["to_offset"],
                )

        resolved_start_tuple = (
            backend.enu_point_to_tuple(start_enu_point) if hasattr(start_enu_point, "x") else normalize_position(start_enu_point)
        )
        if not sampled_waypoints or not points_are_close(sampled_waypoints[0].enu_position, resolved_start_tuple):
            sampled_waypoints.insert(
                0,
                self._make_waypoint_from_enu_position(resolved_start_tuple),
            )

        resolved_goal_tuple = (
            backend.enu_point_to_tuple(goal_enu_point) if hasattr(goal_enu_point, "x") else normalize_position(goal_enu_point)
        )
        if (
            not sampled_waypoints
            or euclidean_distance(sampled_waypoints[-1].enu_position, resolved_goal_tuple) > goal_append_distance_threshold_m
        ):
            sampled_waypoints.append(self._make_waypoint_from_enu_position(resolved_goal_tuple))
        return sampled_waypoints
