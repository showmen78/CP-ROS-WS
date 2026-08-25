"""
Traffic-light stop helpers for the rule-based behavior planner.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Sequence

from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint, world_heading_rad
from cpx_planning.utility.map_api import coerce_map_planner, pose_from_transform

_STOP_TARGET_QUERY_STATE: Dict[str, tuple[tuple[object, ...], float, float, float]] = {}


def _waypoint_xy(waypoint) -> tuple[float, float]:
    transform = getattr(waypoint, "transform", None)
    location = getattr(transform, "location", None)
    if location is not None:
        return float(location.x), float(location.y)
    position = getattr(waypoint, "position", None)
    if isinstance(position, Mapping):
        return float(position["x"]), float(position["y"])
    raise AttributeError("Waypoint has neither CARLA transform nor custom position.")


def _waypoint_ad_lane_id(waypoint) -> int:
    ad_lane_id = getattr(waypoint, "ad_lane_id", None)
    if ad_lane_id is not None:
        return int(ad_lane_id)
    return int(getattr(waypoint, "lane_id", 0) or 0)


def _canonical_lane_id(waypoint) -> int:
    try:
        return int(canonical_lane_id_for_waypoint(waypoint))
    except Exception:
        return abs(int(getattr(waypoint, "lane_id", 0) or 0))


def _pose_xyz(pose: Mapping[str, object]) -> tuple[float, float, float]:
    return (
        float(pose["x"]),
        float(pose["y"]),
        float(pose.get("z", 0.0)),
    )


def normalize_signal_state(signal_state: object) -> str:
    # Do NOT use `signal_state or ""` — carla.TrafficLightState.Red has integer
    # value 0, which is falsy in Python, so `Red or ""` evaluates to "" and the
    # state is misidentified as "unknown".  Use an explicit None check instead.
    raw_name = (str(signal_state) if signal_state is not None else "").strip().upper()
    if "." in raw_name:
        raw_name = raw_name.rsplit(".", 1)[-1]
    if raw_name == "2":
        return "green"
    if raw_name == "1":
        return "yellow"
    if raw_name == "0":
        return "red"
    if raw_name in {"GREEN", "GO"}:
        return "green"
    if raw_name in {"YELLOW", "AMBER"}:
        return "yellow"
    if raw_name in {"RED", "STOP"}:
        return "red"
    return "unknown"


def _actor_transform(actor: Any):
    direct_transform = getattr(actor, "transform", None)
    if direct_transform is not None:
        return direct_transform
    get_transform_fn = getattr(actor, "get_transform", None)
    if callable(get_transform_fn):
        try:
            return get_transform_fn()
        except Exception:
            return None
    return None


def _traffic_light_actor_name(actor: Any) -> str:
    if actor is None:
        return ""
    raw_attributes = getattr(actor, "attributes", None)
    if isinstance(raw_attributes, Mapping):
        for key in ("name", "role_name", "object_name"):
            normalized_value = str(raw_attributes.get(key, "")).strip()
            if normalized_value:
                return normalized_value
    return str(getattr(actor, "type_id", "")).strip()


def _iter_world_traffic_lights(world) -> Iterable[Any]:
    """Yield only real traffic-light actors from the CARLA world.

    The old filter ``callable(getattr(actor, "get_state", None))`` matched any
    actor that exposes ``get_state()``.  In some CARLA versions vehicles,
    walkers, or traffic signs also expose this method and their states do not
    normalize to red/green/yellow → the wrong actor could win the candidate
    selection and report state="unknown".

    Strategy (in order):
    1. Use ``ActorList.filter("traffic.traffic_light*")`` — the cleanest CARLA
       API call.
    2. Fall back to iterating all actors and checking ``type_id``.
    Both paths additionally verify that ``get_state()`` is callable so the rest
    of the caller logic is unchanged.
    """
    get_actors_fn = getattr(world, "get_actors", None)
    if not callable(get_actors_fn):
        return
    try:
        all_actors = get_actors_fn()
    except Exception:
        return

    # Strategy 1: use CARLA's built-in pattern filter (most reliable)
    filter_fn = getattr(all_actors, "filter", None)
    if callable(filter_fn):
        try:
            for actor in filter_fn("traffic.traffic_light*"):
                if callable(getattr(actor, "get_state", None)):
                    yield actor
            return
        except Exception:
            pass

    # Strategy 2: fallback — iterate all actors and match by type_id string
    for actor in list(all_actors):
        type_id = str(getattr(actor, "type_id", "")).lower()
        if "traffic_light" in type_id and callable(getattr(actor, "get_state", None)):
            yield actor


def _signal_position_from_actor(actor: Any) -> Dict[str, float] | None:
    """Convert a CARLA actor/trigger transform to a primitive world point."""

    actor_transform = _actor_transform(actor)
    actor_location = getattr(actor_transform, "location", None)
    if actor_location is None:
        return None

    x_m = float(actor_location.x)
    y_m = float(actor_location.y)
    z_m = float(getattr(actor_location, "z", 0.0))
    trigger_location = getattr(getattr(actor, "trigger_volume", None), "location", None)
    if trigger_location is not None:
        rotation = getattr(actor_transform, "rotation", None)
        yaw_rad = math.radians(float(getattr(rotation, "yaw", 0.0)))
        local_x_m = float(getattr(trigger_location, "x", 0.0))
        local_y_m = float(getattr(trigger_location, "y", 0.0))
        x_m += math.cos(yaw_rad) * local_x_m - math.sin(yaw_rad) * local_y_m
        y_m += math.sin(yaw_rad) * local_x_m + math.cos(yaw_rad) * local_y_m
        z_m += float(getattr(trigger_location, "z", 0.0))
    return {"x": x_m, "y": y_m, "z": z_m}


def _signal_distance_from_position(
    *,
    ego_pose: Mapping[str, object],
    signal_position: Mapping[str, object] | None,
) -> float | None:
    if signal_position is None:
        return None
    ego_x_m, ego_y_m, _ = _pose_xyz(ego_pose)
    return float(
        math.hypot(
            ego_x_m - float(signal_position["x"]),
            ego_y_m - float(signal_position["y"]),
        )
    )


def _signal_forward_from_position(
    *,
    ego_pose: Mapping[str, object],
    signal_position: Mapping[str, object] | None,
) -> float | None:
    if signal_position is None:
        return None
    ego_x_m, ego_y_m, _ = _pose_xyz(ego_pose)
    ego_yaw_rad = float(ego_pose.get("heading_rad", 0.0))
    dx_m = float(signal_position["x"]) - ego_x_m
    dy_m = float(signal_position["y"]) - ego_y_m
    return float(math.cos(ego_yaw_rad) * dx_m + math.sin(ego_yaw_rad) * dy_m)


def _signal_lateral_from_position(
    *,
    ego_pose: Mapping[str, object],
    signal_position: Mapping[str, object] | None,
) -> float | None:
    if signal_position is None:
        return None
    ego_x_m, ego_y_m, _ = _pose_xyz(ego_pose)
    ego_yaw_rad = float(ego_pose.get("heading_rad", 0.0))
    dx_m = float(signal_position["x"]) - ego_x_m
    dy_m = float(signal_position["y"]) - ego_y_m
    return float(-math.sin(ego_yaw_rad) * dx_m + math.cos(ego_yaw_rad) * dy_m)


def _route_sample_hint_m(cum_dists: Sequence[float]) -> float:
    if len(cum_dists) < 2:
        return 2.0
    segment_lengths = [
        max(0.0, float(cum_dists[index + 1]) - float(cum_dists[index]))
        for index in range(len(cum_dists) - 1)
    ]
    positive_lengths = [
        float(length_m)
        for length_m in segment_lengths
        if float(length_m) > 1.0e-3
    ]
    if len(positive_lengths) == 0:
        return 2.0
    positive_lengths.sort()
    middle_index = len(positive_lengths) // 2
    if len(positive_lengths) % 2 == 1:
        median_length_m = float(positive_lengths[middle_index])
    else:
        median_length_m = 0.5 * float(
            positive_lengths[middle_index - 1] + positive_lengths[middle_index]
        )
    return max(1.0, min(10.0, float(median_length_m)))


def _route_signature_key(
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
) -> tuple[object, ...]:
    if len(route_points) == 0:
        return (0,)
    head_points = tuple(
        (
            round(float(point[0]), 3),
            round(float(point[1]), 3),
        )
        for point in list(route_points[:3])
    )
    tail_points = tuple(
        (
            round(float(point[0]), 3),
            round(float(point[1]), 3),
        )
        for point in list(route_points[-3:])
    )
    total_length_m = float(cum_dists[-1]) if len(cum_dists) > 0 else 0.0
    return (
        int(len(route_points)),
        head_points,
        tail_points,
        round(float(total_length_m), 3),
    )


def _clear_stop_target_query_state(query_key: str | None) -> None:
    cache_key = str(query_key or "").strip()
    if cache_key:
        _STOP_TARGET_QUERY_STATE.pop(cache_key, None)


def _stop_target_match_details(
    *,
    map_planner: Any,
    signal_position: Mapping[str, object],
    stop_target_xy: tuple[float, float],
    stop_target_lane_id: int | None,
    stop_target_road_id: int | None,
    stop_target_section_id: int | None,
) -> tuple[int, float] | None:
    custom_waypoint = map_planner.get_waypoint(
        {
            "x": float(signal_position["x"]),
            "y": float(signal_position["y"]),
            "z": float(signal_position.get("z", 0.0)),
        }
    )
    if custom_waypoint is None:
        return None

    match_distance_m = float(
        math.hypot(
            float(signal_position["x"]) - float(stop_target_xy[0]),
            float(signal_position["y"]) - float(stop_target_xy[1]),
        )
    )
    stop_waypoint_road_id = int(custom_waypoint.road_id or 0)
    stop_waypoint_section_id = int(custom_waypoint.section_id or 0)
    stop_waypoint_lane_id = int(canonical_lane_id_for_waypoint(custom_waypoint))

    road_matches = (
        stop_target_road_id is not None
        and int(stop_waypoint_road_id) == int(stop_target_road_id)
    )
    section_matches = (
        stop_target_section_id is not None
        and int(stop_waypoint_section_id) == int(stop_target_section_id)
    )
    lane_matches = (
        stop_target_lane_id is not None
        and int(stop_target_lane_id) != 0
        and int(stop_waypoint_lane_id) == int(stop_target_lane_id)
    )

    if road_matches and section_matches and lane_matches:
        return 0, float(match_distance_m)
    if road_matches and lane_matches:
        return 1, float(match_distance_m)
    if road_matches and section_matches:
        return 2, float(match_distance_m)
    return 3, float(match_distance_m)


def _find_relevant_signal_context_impl(
    *,
    world,
    map_planner,
    ego_vehicle,
    ego_pose: Mapping[str, object],
    stop_target: Mapping[str, object] | None,
    max_stop_waypoint_match_distance_m: float = 12.0,
    max_actor_position_match_distance_m: float = 40.0,
    max_actor_position_lateral_m: float = 4.5,
) -> Dict[str, object]:
    default_context: Dict[str, object] = {
        "signal_found": False,
        "signal_state": "unknown",
        "signal_distance_m": None,
        "signal_actor_id": None,
        "signal_actor_name": "",
        "signal_source": "none",
        "signal_match_distance_m": None,
        "signal_actor_raw_state": "",
    }

    ego_associated_actor = None
    ego_associated_state: str | None = None  # state read via dedicated vehicle API
    get_traffic_light_fn = getattr(ego_vehicle, "get_traffic_light", None)
    if callable(get_traffic_light_fn):
        try:
            ego_associated_actor = get_traffic_light_fn()
        except Exception:
            ego_associated_actor = None

    # When ego IS inside a trigger zone, use get_traffic_light_state() as the
    # authoritative state source.  This avoids the general get_state() path on
    # the actor object which can fail or be ambiguous.
    if ego_associated_actor is not None:
        get_tl_state_fn = getattr(ego_vehicle, "get_traffic_light_state", None)
        if callable(get_tl_state_fn):
            try:
                ego_associated_state = normalize_signal_state(get_tl_state_fn())
            except Exception:
                ego_associated_state = None

    stop_target_xy = None
    stop_target_distance_m = None
    stop_target_lane_id = None
    stop_target_road_id = None
    stop_target_section_id = None
    if isinstance(stop_target, Mapping):
        try:
            stop_target_xy = (
                float(stop_target.get("x_m", 0.0)),
                float(stop_target.get("y_m", 0.0)),
            )
            stop_target_distance_m = max(
                0.0,
                float(stop_target.get("distance_m", 0.0)),
            )
            raw_stop_target_lane_id = stop_target.get("lane_id", None)
            raw_stop_target_road_id = stop_target.get("road_id", None)
            raw_stop_target_section_id = stop_target.get("section_id", None)
            stop_target_lane_id = (
                None
                if raw_stop_target_lane_id is None
                else int(raw_stop_target_lane_id)
            )
            stop_target_road_id = (
                None
                if raw_stop_target_road_id is None
                else int(raw_stop_target_road_id)
            )
            stop_target_section_id = (
                None
                if raw_stop_target_section_id is None
                else int(raw_stop_target_section_id)
            )
        except Exception:
            stop_target_xy = None
            stop_target_distance_m = None
            stop_target_lane_id = None
            stop_target_road_id = None
            stop_target_section_id = None

    candidate_actors: list[Any] = []
    # Deduplicate by CARLA actor ID (integer), not Python id() (memory address).
    # Each call to world.get_actors() may return new Python wrapper objects for
    # the same underlying CARLA actor, so id() differs even for the same light.
    seen_candidate_actor_ids: set[object] = set()
    for candidate_actor in [ego_associated_actor, *_iter_world_traffic_lights(world)]:
        if candidate_actor is None:
            continue
        # Prefer CARLA's own integer actor ID; fall back to Python object id.
        carla_actor_id = getattr(candidate_actor, "id", None)
        dedup_key = carla_actor_id if carla_actor_id is not None else id(candidate_actor)
        if dedup_key in seen_candidate_actor_ids:
            continue
        seen_candidate_actor_ids.add(dedup_key)
        candidate_actors.append(candidate_actor)

    best_candidate: Dict[str, object] | None = None
    for candidate_actor in candidate_actors:
        # When this is the ego-associated actor, prefer the state read directly
        # via get_traffic_light_state() (more reliable) over get_state().
        candidate_raw_state = ""
        if candidate_actor is ego_associated_actor and ego_associated_state is not None:
            candidate_raw_state = str(ego_associated_state)
            candidate_state = str(ego_associated_state)
        else:
            candidate_state = "unknown"
            get_state_fn = getattr(candidate_actor, "get_state", None)
            if callable(get_state_fn):
                try:
                    raw_state = get_state_fn()
                    candidate_raw_state = str(raw_state)
                    candidate_state = normalize_signal_state(raw_state)
                except Exception:
                    candidate_state = "unknown"

        actor_name = _traffic_light_actor_name(candidate_actor)
        signal_position = _signal_position_from_actor(candidate_actor)
        signal_distance_m = _signal_distance_from_position(
            ego_pose=ego_pose,
            signal_position=signal_position,
        )
        signal_forward_m = _signal_forward_from_position(
            ego_pose=ego_pose,
            signal_position=signal_position,
        )
        signal_lateral_m = _signal_lateral_from_position(
            ego_pose=ego_pose,
            signal_position=signal_position,
        )
        match_distance_m = float("inf")
        signal_source = "actor_position_match"
        custom_position_match_rank = None

        get_stop_waypoints = getattr(candidate_actor, "get_stop_waypoints", None)
        actor_stop_waypoints = (
            list(get_stop_waypoints() or []) if callable(get_stop_waypoints) else []
        )
        if stop_target_xy is not None and actor_stop_waypoints:
            stop_matches = []
            for stop_waypoint in actor_stop_waypoints:
                stop_x_m, stop_y_m = _waypoint_xy(stop_waypoint)
                distance_m = math.hypot(
                    float(stop_x_m) - float(stop_target_xy[0]),
                    float(stop_y_m) - float(stop_target_xy[1]),
                )
                road_matches = (
                    stop_target_road_id is not None
                    and int(getattr(stop_waypoint, "road_id", 0) or 0)
                    == int(stop_target_road_id)
                )
                section_matches = (
                    stop_target_section_id is not None
                    and int(getattr(stop_waypoint, "section_id", 0) or 0)
                    == int(stop_target_section_id)
                )
                lane_matches = (
                    stop_target_lane_id is not None
                    and int(stop_target_lane_id) != 0
                    and _canonical_lane_id(stop_waypoint)
                    == int(stop_target_lane_id)
                )
                rank = (
                    0
                    if road_matches and section_matches and lane_matches
                    else 1
                    if road_matches and lane_matches
                    else 2
                    if road_matches and section_matches
                    else 3
                )
                stop_matches.append((int(rank), float(distance_m)))
            custom_position_match_rank, match_distance_m = min(stop_matches)
            signal_source = "stop_waypoint_match"
        elif stop_target_xy is not None and signal_position is not None:
            match_details = _stop_target_match_details(
                map_planner=map_planner,
                signal_position=signal_position,
                stop_target_xy=stop_target_xy,
                stop_target_lane_id=stop_target_lane_id,
                stop_target_road_id=stop_target_road_id,
                stop_target_section_id=stop_target_section_id,
            )
            if match_details is not None:
                custom_position_match_rank, match_distance_m = match_details
                if int(custom_position_match_rank) <= 2:
                    signal_source = "custom_lane_match"
            if not math.isfinite(match_distance_m):
                match_distance_m = float(
                    math.hypot(
                        float(signal_position["x"]) - float(stop_target_xy[0]),
                        float(signal_position["y"]) - float(stop_target_xy[1]),
                    )
                )
        elif signal_distance_m is not None:
            match_distance_m = float(signal_distance_m)

        ego_association_plausible = True
        if (
            candidate_actor is ego_associated_actor
            and stop_target_xy is not None
            and custom_position_match_rank is None
            and math.isfinite(match_distance_m)
        ):
            association_match_limit_m = float(max_actor_position_match_distance_m)
            if stop_target_distance_m is not None:
                association_match_limit_m = max(
                    association_match_limit_m,
                    float(stop_target_distance_m) + 20.0,
                )
            ego_association_plausible = (
                float(match_distance_m) <= float(association_match_limit_m)
            )

        if candidate_actor is ego_associated_actor and bool(ego_association_plausible):
            signal_source = "ego_vehicle_association"
            if stop_target_distance_m is not None and (
                not math.isfinite(match_distance_m) or float(match_distance_m) > float(stop_target_distance_m) + 20.0
            ):
                match_distance_m = float(stop_target_distance_m)

        if signal_source in {"custom_lane_match", "stop_waypoint_match"}:
            lane_aware_max_match_distance_m = max(
                float(max_stop_waypoint_match_distance_m),
                35.0,
            )
            section_aware_max_match_distance_m = max(
                float(max_stop_waypoint_match_distance_m),
                20.0,
            )
            if custom_position_match_rank in {0, 1}:
                if float(match_distance_m) > float(lane_aware_max_match_distance_m):
                    continue
            elif custom_position_match_rank == 2:
                if float(match_distance_m) > float(section_aware_max_match_distance_m):
                    continue
            elif float(match_distance_m) > float(max_stop_waypoint_match_distance_m):
                continue
        elif signal_source == "actor_position_match":
            if float(match_distance_m) > float(max_actor_position_match_distance_m):
                continue
            if signal_forward_m is not None and float(signal_forward_m) < -2.0:
                continue
            if (
                signal_lateral_m is not None
                and abs(float(signal_lateral_m)) > max(0.0, float(max_actor_position_lateral_m))
            ):
                continue

        candidate_context = {
            "signal_found": True,
            "signal_state": str(candidate_state),
            "signal_distance_m": (
                None
                if stop_target_distance_m is None
                else float(stop_target_distance_m)
            ) if stop_target_distance_m is not None else signal_distance_m,
            "signal_actor_id": getattr(candidate_actor, "id", None),
            "signal_actor_name": str(actor_name),
            "signal_source": str(signal_source),
            "signal_actor_raw_state": str(candidate_raw_state),
            "signal_match_distance_m": (
                None if not math.isfinite(match_distance_m) else float(match_distance_m)
            ),
            "signal_forward_m": signal_forward_m,
            "signal_lateral_m": signal_lateral_m,
            "signal_match_rank": custom_position_match_rank,
            "signal_actor": candidate_actor,
        }
        if best_candidate is None:
            best_candidate = candidate_context
            continue

        # Prefer a custom-map lane match when the ego-associated actor has no
        # stop-line/lane evidence. CARLA trigger volumes can be large around
        # dense junctions; blindly prioritising ego association lets a sibling
        # light keep a stale red after the route stop line has moved on.
        def source_priority(candidate: Mapping[str, object]) -> int:
            source = str(candidate.get("signal_source", ""))
            if source == "ego_vehicle_association":
                if stop_target_xy is not None and candidate.get("signal_match_rank", None) is None:
                    return 2
                return 0
            if source == "custom_lane_match":
                return 1
            if source == "actor_position_match":
                return 3
            return 9

        current_priority = (
            int(source_priority(candidate_context)),
            9
            if candidate_context.get("signal_match_rank", None) is None
            else int(candidate_context.get("signal_match_rank", 9)),
            float(candidate_context.get("signal_match_distance_m") or float("inf")),
            float(candidate_context.get("signal_distance_m") or float("inf")),
        )
        best_priority = (
            int(source_priority(best_candidate)),
            9
            if best_candidate.get("signal_match_rank", None) is None
            else int(best_candidate.get("signal_match_rank", 9)),
            float(best_candidate.get("signal_match_distance_m") or float("inf")),
            float(best_candidate.get("signal_distance_m") or float("inf")),
        )
        if current_priority < best_priority:
            best_candidate = candidate_context

    if best_candidate is None:
        return dict(default_context)
    best_candidate.pop("signal_actor", None)
    return best_candidate


def _route_cum_dists(route_points: Sequence[Sequence[float]]) -> list[float]:
    if len(route_points) == 0:
        return []
    dists = [0.0]
    for idx in range(1, len(route_points)):
        dists.append(
            float(dists[-1])
            + float(
                math.hypot(
                    float(route_points[idx][0]) - float(route_points[idx - 1][0]),
                    float(route_points[idx][1]) - float(route_points[idx - 1][1]),
                )
            )
        )
    return dists


def _project_to_route_arc(
    x_m: float,
    y_m: float,
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
    query_key: str | None = None,
) -> float:
    if len(route_points) < 2:
        _clear_stop_target_query_state(query_key)
        return 0.0

    best_distance_m = float("inf")
    candidates: list[tuple[float, float]] = []
    for idx in range(len(route_points) - 1):
        ax_m = float(route_points[idx][0])
        ay_m = float(route_points[idx][1])
        bx_m = float(route_points[idx + 1][0])
        by_m = float(route_points[idx + 1][1])
        dx_m = float(bx_m) - float(ax_m)
        dy_m = float(by_m) - float(ay_m)
        segment_len_sq = dx_m * dx_m + dy_m * dy_m
        if segment_len_sq <= 1.0e-12:
            alpha = 0.0
        else:
            alpha = (
                (float(x_m) - float(ax_m)) * float(dx_m)
                + (float(y_m) - float(ay_m)) * float(dy_m)
            ) / float(segment_len_sq)
            alpha = max(0.0, min(1.0, float(alpha)))
        proj_x_m = float(ax_m) + float(alpha) * float(dx_m)
        proj_y_m = float(ay_m) + float(alpha) * float(dy_m)
        distance_m = float(
            math.hypot(
                float(x_m) - float(proj_x_m),
                float(y_m) - float(proj_y_m),
            )
        )
        progress_m = float(cum_dists[idx]) + float(alpha) * float(
            math.sqrt(max(segment_len_sq, 0.0))
        )
        best_distance_m = min(float(best_distance_m), float(distance_m))
        candidates.append((float(distance_m), float(progress_m)))

    if len(candidates) == 0:
        _clear_stop_target_query_state(query_key)
        return 0.0

    sample_hint_m = _route_sample_hint_m(cum_dists)
    ambiguity_margin_m = max(1.0, 0.75 * float(sample_hint_m))
    plausible_candidates = [
        candidate
        for candidate in candidates
        if float(candidate[0]) <= float(best_distance_m) + float(ambiguity_margin_m)
    ]
    selected_candidates = plausible_candidates if len(plausible_candidates) > 0 else candidates
    selected_candidate = min(
        selected_candidates,
        key=lambda candidate: (float(candidate[1]), float(candidate[0])),
    )

    cache_key = str(query_key or "").strip()
    if cache_key:
        route_key = _route_signature_key(route_points, cum_dists)
        previous_state = _STOP_TARGET_QUERY_STATE.get(cache_key, None)
        if previous_state is not None:
            previous_route_key, previous_progress_m, previous_x_m, previous_y_m = previous_state
            if previous_route_key == route_key:
                movement_m = float(
                    math.hypot(
                        float(x_m) - float(previous_x_m),
                        float(y_m) - float(previous_y_m),
                    )
                )
                progress_backtrack_m = max(2.0, 2.0 * float(sample_hint_m))
                progress_advance_m = max(
                    8.0,
                    float(movement_m) + 6.0 * float(sample_hint_m),
                )
                preferred_progress_m = float(previous_progress_m) + min(
                    max(float(movement_m), float(sample_hint_m)),
                    float(progress_advance_m),
                )
                continuity_candidates = [
                    candidate
                    for candidate in selected_candidates
                    if (
                        float(previous_progress_m) - float(progress_backtrack_m)
                        <= float(candidate[1])
                        <= float(previous_progress_m) + float(progress_advance_m)
                    )
                ]
                if len(continuity_candidates) > 0:
                    selected_candidate = min(
                        continuity_candidates,
                        key=lambda candidate: (
                            abs(float(candidate[1]) - float(preferred_progress_m)),
                            float(candidate[0]),
                            float(candidate[1]),
                        ),
                    )
        _STOP_TARGET_QUERY_STATE[cache_key] = (
            route_key,
            float(selected_candidate[1]),
            float(x_m),
            float(y_m),
        )

    return float(selected_candidate[1])


def _find_stop_target_from_ego_impl(
    *,
    map_planner,
    ego_pose: Mapping[str, object],
    global_route_points: Sequence[Sequence[float]],
    search_distance_m: float = 100.0,
    query_key: str | None = None,
) -> Dict[str, float] | None:
    route_points = [
        [float(point[0]), float(point[1])]
        for point in list(global_route_points or [])
        if isinstance(point, Sequence) and len(point) >= 2
    ]
    if len(route_points) < 2:
        _clear_stop_target_query_state(query_key)
        return None

    ego_x_m, ego_y_m, ego_z_m = _pose_xyz(ego_pose)

    route_cum_dists = _route_cum_dists(route_points)
    if len(route_cum_dists) != len(route_points):
        return None
    ego_arc_m = _project_to_route_arc(
        x_m=float(ego_x_m),
        y_m=float(ego_y_m),
        route_points=route_points,
        cum_dists=route_cum_dists,
        query_key=query_key,
    )
    max_search_distance_m = max(0.0, float(search_distance_m))

    previous_waypoint = None
    previous_arc_m: float | None = None
    for idx, route_point in enumerate(route_points):
        point_arc_m = float(route_cum_dists[idx])
        if float(point_arc_m) < float(ego_arc_m) - 1.0e-3:
            continue
        forward_distance_m = float(point_arc_m) - float(ego_arc_m)
        if float(forward_distance_m) > float(max_search_distance_m):
            break

        route_waypoint = map_planner.get_waypoint(
            {
                "x": float(route_point[0]),
                "y": float(route_point[1]),
                "z": float(ego_z_m),
            }
        )
        if route_waypoint is None:
            continue

        if bool(
            getattr(
                route_waypoint,
                "is_intersection",
                getattr(route_waypoint, "is_junction", False),
            )
        ):
            if previous_waypoint is None or previous_arc_m is None:
                return None
            stop_distance_m = float(previous_arc_m) - float(ego_arc_m)
            if stop_distance_m <= 1.0e-3:
                return None
            stop_x_m, stop_y_m = _waypoint_xy(previous_waypoint)
            return {
                "x_m": float(stop_x_m),
                "y_m": float(stop_y_m),
                "heading_rad": float(world_heading_rad(previous_waypoint) or 0.0),
                "lane_id": _canonical_lane_id(previous_waypoint),
                "ad_lane_id": int(_waypoint_ad_lane_id(previous_waypoint)),
                "opendrive_lane_id": int(getattr(previous_waypoint, "lane_id", 0) or 0),
                "road_id": int(getattr(previous_waypoint, "road_id", 0)),
                "section_id": int(getattr(previous_waypoint, "section_id", 0)),
                "distance_m": float(stop_distance_m),
            }

        previous_waypoint = route_waypoint
        previous_arc_m = float(point_arc_m)

    return None


def should_stop_for_signal(
    *,
    signal_state: object,
    stop_target: Mapping[str, object] | None,
    ego_velocity_mps: float,
    ego_max_deceleration_mps2: float,
    ego_in_junction: bool,
    stop_buffer_m: float = 2.0,
) -> bool:
    if bool(ego_in_junction):
        return False

    normalized_signal_state = normalize_signal_state(signal_state)
    if normalized_signal_state == "green":
        return False
    if normalized_signal_state not in {"yellow", "red"}:
        return False
    if not isinstance(stop_target, Mapping):
        return False

    try:
        stop_distance_m = max(0.0, float(stop_target.get("distance_m", 0.0)))
    except Exception:
        return False
    if stop_distance_m <= 1.0e-3:
        return False

    ego_speed_mps = max(0.0, float(ego_velocity_mps))
    max_deceleration_mps2 = max(1.0e-6, float(ego_max_deceleration_mps2))
    required_stop_distance_m = (
        float(ego_speed_mps) * float(ego_speed_mps)
    ) / (2.0 * float(max_deceleration_mps2))
    required_stop_distance_m += max(0.0, float(stop_buffer_m))

    return float(stop_distance_m) >= float(required_stop_distance_m)


class _NoMapPlanner:
    def get_waypoint(self, point):
        del point
        return None


def find_relevant_signal_context(*args, **kwargs):
    """Normalize legacy CARLA arguments at the traffic-control boundary."""

    normalized = dict(kwargs)
    world_map = normalized.pop("world_map", None)
    carla_module = normalized.pop("carla", None)
    if normalized.get("map_planner") is None:
        normalized["map_planner"] = (
            coerce_map_planner(
                world_map=world_map,
                carla_module=carla_module,
            )
            if world_map is not None
            else _NoMapPlanner()
        )
    ego_transform = normalized.pop("ego_transform", None)
    if normalized.get("ego_pose") is None and ego_transform is not None:
        normalized["ego_pose"] = pose_from_transform(ego_transform)
    return _find_relevant_signal_context_impl(*args, **normalized)


def find_stop_target_from_ego(*args, **kwargs):
    """Normalize legacy CARLA arguments at the stop-target boundary."""

    normalized = dict(kwargs)
    world_map = normalized.pop("world_map", None)
    carla_module = normalized.pop("carla", None)
    if normalized.get("map_planner") is None and world_map is not None:
        normalized["map_planner"] = coerce_map_planner(
            world_map=world_map,
            carla_module=carla_module,
        )
    ego_transform = normalized.pop("ego_transform", None)
    if normalized.get("ego_pose") is None and ego_transform is not None:
        normalized["ego_pose"] = pose_from_transform(ego_transform)
    return _find_stop_target_from_ego_impl(*args, **normalized)
