"""
Temporary destination ("blue dot") for MPC.

The global route only provides longitudinal progress and junction branch
selection. The behavior planner owns lane selection:

  **LANE_FOLLOW**         Hold the current selected lane
  **LANE_CHANGE_LEFT**    Shift the blue dot to the selected left lane
  **LANE_CHANGE_RIGHT**   Shift the blue dot to the selected right lane
  **REROUTE**             Let the new global route choose the lane

Mode is checked from the **previous blue-dot position** only for
keeping an already-active intersection latch alive. Automatic
intersection discovery from CARLA waypoint topology is disabled here;
``INTERSECTION`` is entered only through an explicit override.

Key functions
-------------
* ``compute_temp_destination``  — single destination point for MPC
* ``build_reference_samples``   — horizon-length reference trajectory
* ``compute_ego_lane_offset``   — lateral offset from lane centre
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .planner import (
    is_emergency_brake_decision,
    is_fixed_stop_decision,
    normalize_behavior_decision,
    normalize_macro_maneuver,
)
from cpx_planning.utility.global_planner import (
    canonical_lane_id_for_waypoint,
    canonical_lane_waypoint_for_lane_id,
    canonical_lane_waypoints,
    world_heading_rad,
)

# -------------------------------------------------------------------- #
# Constants                                                              #
# -------------------------------------------------------------------- #
DEFAULT_STEP_M: float = 2.0
INTERSECTION_THRESHOLD_M: float = 30.0
JUNCTION_ROUTE_SNAP_DISTANCE_M: float = 1.5
JUNCTION_ROUTE_SNAP_HEADING_DEG: float = 35.0


# -------------------------------------------------------------------- #
# Route geometry helpers                                                 #
# -------------------------------------------------------------------- #
_cached_route_id: int | None = None
_cached_cum_dists: List[float] = []



def _waypoint_xy(waypoint) -> tuple[float, float]:
    position = getattr(waypoint, "position", None)
    if not isinstance(position, Mapping):
        raise AttributeError("Custom waypoint does not contain a position.")
    return float(position["x"]), float(position["y"])


def _waypoint_xyz(waypoint) -> Dict[str, float]:
    position = getattr(waypoint, "position", None)
    if not isinstance(position, Mapping):
        raise AttributeError("Custom waypoint does not contain a position.")
    return {
        "x": float(position["x"]),
        "y": float(position["y"]),
        "z": float(position.get("z", 0.0)),
    }


def _waypoint_z(waypoint, default_z: float = 0.0) -> float:
    position = getattr(waypoint, "position", None)
    if not isinstance(position, Mapping):
        return float(default_z)
    return float(position.get("z", default_z))


def _waypoint_heading_rad(waypoint) -> float:
    return float(world_heading_rad(waypoint) or 0.0)


def _project_waypoint(map_planner, point) -> object | None:
    return map_planner.get_waypoint(point)


def _pose_xyz(pose: Mapping[str, object]) -> Dict[str, float]:
    return {
        "x": float(pose["x"]),
        "y": float(pose["y"]),
        "z": float(pose.get("z", 0.0)),
    }

def _route_cum_dists(
    route_points: Sequence[Sequence[float]],
) -> List[float]:
    """Cumulative arc-length along the route polyline (cached by id)."""
    global _cached_route_id, _cached_cum_dists
    rid = id(route_points)
    if rid == _cached_route_id and len(_cached_cum_dists) == len(route_points):
        return _cached_cum_dists
    dists: List[float] = [0.0]
    for i in range(1, len(route_points)):
        d = math.hypot(
            float(route_points[i][0]) - float(route_points[i - 1][0]),
            float(route_points[i][1]) - float(route_points[i - 1][1]),
        )
        dists.append(dists[-1] + d)
    _cached_route_id = rid
    _cached_cum_dists = dists
    return dists


def project_ego_to_route(
    ego_x: float,
    ego_y: float,
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
) -> float:
    """
    Project ``(ego_x, ego_y)`` onto the route polyline.

    Returns the arc-length distance along the route at the projected
    point.
    """
    if len(route_points) < 2:
        return 0.0

    best_dist_sq = float("inf")
    best_arc = 0.0

    for i in range(len(route_points) - 1):
        ax = float(route_points[i][0])
        ay = float(route_points[i][1])
        bx = float(route_points[i + 1][0])
        by = float(route_points[i + 1][1])

        abx, aby = bx - ax, by - ay
        ab_sq = abx * abx + aby * aby

        if ab_sq < 1e-12:
            t = 0.0
        else:
            t = max(
                0.0,
                min(
                    1.0,
                    ((ego_x - ax) * abx + (ego_y - ay) * aby) / ab_sq,
                ),
            )

        px = ax + t * abx
        py = ay + t * aby
        d_sq = (ego_x - px) ** 2 + (ego_y - py) ** 2

        if d_sq < best_dist_sq:
            best_dist_sq = d_sq
            seg_len = math.sqrt(ab_sq)
            best_arc = float(cum_dists[i]) + t * seg_len

    return best_arc


def get_lookahead_route_point(
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
    ego_arc: float,
    lookahead_m: float,
) -> Tuple[float, float]:
    """
    Interpolate the route at ``ego_arc + lookahead_m``.

    Returns ``(x, y)``.
    """
    target = ego_arc + max(0.0, float(lookahead_m))
    total = float(cum_dists[-1]) if cum_dists else 0.0

    if target >= total:
        return float(route_points[-1][0]), float(route_points[-1][1])

    for i in range(len(cum_dists) - 1):
        if float(cum_dists[i + 1]) >= target:
            seg = max(1e-9, float(cum_dists[i + 1]) - float(cum_dists[i]))
            t = (target - float(cum_dists[i])) / seg
            x = float(route_points[i][0]) + t * (
                float(route_points[i + 1][0]) - float(route_points[i][0])
            )
            y = float(route_points[i][1]) + t * (
                float(route_points[i + 1][1]) - float(route_points[i][1])
            )
            return x, y

    return float(route_points[-1][0]), float(route_points[-1][1])


def _route_point_at_arc(
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
    target_arc: float,
) -> Tuple[float, float]:
    """Interpolate the route at the absolute arc-length ``target_arc``."""
    if len(route_points) == 0:
        return 0.0, 0.0
    if len(route_points) == 1 or len(cum_dists) < 2:
        return float(route_points[0][0]), float(route_points[0][1])
    total_arc = float(cum_dists[-1])
    clamped_arc = min(max(0.0, float(target_arc)), total_arc)
    return get_lookahead_route_point(
        route_points=route_points,
        cum_dists=cum_dists,
        ego_arc=0.0,
        lookahead_m=float(clamped_arc),
    )


def _route_heading_at_arc(
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
    target_arc: float,
) -> float:
    """Estimate route heading at ``target_arc`` using nearby route points."""
    if len(route_points) < 2 or len(cum_dists) < 2:
        return 0.0

    total_arc = float(cum_dists[-1])
    clamped_arc = min(max(0.0, float(target_arc)), total_arc)
    forward_arc = min(float(total_arc), float(clamped_arc) + 0.5)
    backward_arc = max(0.0, float(clamped_arc) - 0.5)

    x0_m, y0_m = _route_point_at_arc(route_points, cum_dists, backward_arc)
    x1_m, y1_m = _route_point_at_arc(route_points, cum_dists, forward_arc)
    if math.hypot(float(x1_m) - float(x0_m), float(y1_m) - float(y0_m)) <= 1.0e-9:
        if len(route_points) >= 2:
            return float(
                math.atan2(
                    float(route_points[-1][1]) - float(route_points[-2][1]),
                    float(route_points[-1][0]) - float(route_points[-2][0]),
                )
            )
        return 0.0
    return float(math.atan2(float(y1_m) - float(y0_m), float(x1_m) - float(x0_m)))


def _closest_route_arc_ahead(
    x_m: float,
    y_m: float,
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
    anchor_arc_m: float,
) -> float:
    """Closest remaining-route arc to ``(x_m, y_m)``, clamped ahead of anchor."""
    projected_arc_m = project_ego_to_route(
        ego_x=float(x_m),
        ego_y=float(y_m),
        route_points=route_points,
        cum_dists=cum_dists,
    )
    total_arc_m = float(cum_dists[-1]) if len(cum_dists) > 0 else 0.0
    return min(float(total_arc_m), max(float(anchor_arc_m), float(projected_arc_m)))


def _route_match_metrics(
    waypoint: Any,
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
    anchor_arc_m: float,
) -> Tuple[float, float, float]:
    """Return ``(route_arc_m, distance_to_route_m, heading_error_rad)``."""
    if waypoint is None:
        return float(anchor_arc_m), float("inf"), float("inf")

    waypoint_x_m, waypoint_y_m = _waypoint_xy(waypoint)

    route_arc_m = _closest_route_arc_ahead(
        waypoint_x_m,
        waypoint_y_m,
        route_points,
        cum_dists,
        anchor_arc_m,
    )
    route_x_m, route_y_m = _route_point_at_arc(route_points, cum_dists, route_arc_m)
    route_heading_rad = _route_heading_at_arc(route_points, cum_dists, route_arc_m)
    waypoint_heading_rad = _waypoint_heading_rad(waypoint)
    heading_error_rad = abs(
        math.atan2(
            math.sin(float(waypoint_heading_rad) - float(route_heading_rad)),
            math.cos(float(waypoint_heading_rad) - float(route_heading_rad)),
        )
    )
    return (
        float(route_arc_m),
        float(
            math.hypot(
                waypoint_x_m - float(route_x_m),
                waypoint_y_m - float(route_y_m),
            )
        ),
        float(heading_error_rad),
    )


def _snap_junction_waypoint_to_route(
    *,
    map_planner: Any,
    waypoint: Any,
    anchor_wp: Any,
    route_points: Sequence[Sequence[float]],
    cum_dists: Sequence[float],
    anchor_arc_m: float,
) -> Any:
    """Snap an off-route junction waypoint back onto the remaining route."""
    if waypoint is None or not bool(getattr(waypoint, "is_intersection", False)):
        return waypoint

    route_arc_m, route_distance_m, route_heading_error_rad = _route_match_metrics(
        waypoint,
        route_points,
        cum_dists,
        anchor_arc_m,
    )
    if (
        float(route_distance_m) <= float(JUNCTION_ROUTE_SNAP_DISTANCE_M)
        and float(route_heading_error_rad) <= math.radians(float(JUNCTION_ROUTE_SNAP_HEADING_DEG))
    ):
        return waypoint

    route_x_m, route_y_m = _route_point_at_arc(route_points, cum_dists, route_arc_m)
    snap_z_m = float(_waypoint_z(anchor_wp, _waypoint_z(waypoint, 0.0)))
    snapped_wp = map_planner.get_waypoint(
        {"x": float(route_x_m), "y": float(route_y_m), "z": snap_z_m}
    )
    if snapped_wp is not None:
        _, snapped_distance_m, snapped_heading_error_rad = _route_match_metrics(
            snapped_wp,
            route_points,
            cum_dists,
            anchor_arc_m,
        )
        if (
            float(snapped_distance_m) + 1.0e-6 < float(route_distance_m)
            or float(snapped_heading_error_rad) + 1.0e-6 < float(route_heading_error_rad)
        ):
            return snapped_wp

    walk_distance_m = max(0.0, float(route_arc_m) - float(anchor_arc_m))
    walked_wp = _walk_forward(
        anchor_wp,
        float(walk_distance_m),
        DEFAULT_STEP_M,
        route_points=route_points,
        cum_dists=cum_dists,
    )
    return walked_wp if walked_wp is not None else waypoint


# -------------------------------------------------------------------- #
# Custom-waypoint helpers                                                #
# -------------------------------------------------------------------- #
def _normalize_angle_deg(angle_deg: float) -> float:
    a = float(angle_deg) % 360.0
    if a > 180.0:
        a -= 360.0
    return a

def _walk_forward(
    wp: Any,
    distance_m: float,
    step_m: float,
    route_points: Sequence[Sequence[float]] | None = None,
    cum_dists: Sequence[float] | None = None,
    maneuver: str | None = None,
    target_position: Mapping[str, object] | None = None,
) -> Any:
    """Walk *wp* forward by *distance_m* using ``wp.next()``.

    When *route_points* / *cum_dists* are provided, junction branches
    are resolved by picking the candidate closest to the route's
    lookahead point first (route-guided). When route guidance is not
    available, *target_position* is used next, then maneuver bias, then the
    straightest successor.
    """
    target_xy = (
        None
        if target_position is None
        else (float(target_position["x"]), float(target_position["y"]))
    )
    target_arc = None
    if (
        target_xy is not None
        and route_points is not None
        and cum_dists is not None
        and len(route_points) >= 2
    ):
        target_arc = project_ego_to_route(
            float(target_xy[0]),
            float(target_xy[1]),
            route_points,
            cum_dists,
        )
    cumulative = 0.0
    while cumulative < distance_m:
        next_fn = getattr(wp, "next", None)
        if next_fn is None:
            break
        candidates = next_fn(step_m)
        if not candidates:
            break
        if len(candidates) == 1:
            wp = candidates[0]
        elif route_points is not None and cum_dists is not None:
            wp_x, wp_y = _waypoint_xy(wp)
            arc = project_ego_to_route(wp_x, wp_y, route_points, cum_dists)
            total_arc_m = float(cum_dists[-1]) if len(cum_dists) > 0 else float(arc)
            expected_arc_m = min(float(total_arc_m), float(arc) + float(step_m))
            preview_arc_m = min(float(total_arc_m), float(arc) + 2.0 * float(step_m))
            look_x, look_y = _route_point_at_arc(
                route_points,
                cum_dists,
                expected_arc_m,
            )

            def _route_guided_score(candidate: Any) -> tuple[float, float, float, float, float, float]:
                candidate_arc_m, candidate_route_distance_m, candidate_heading_error_rad = _route_match_metrics(
                    candidate,
                    route_points,
                    cum_dists,
                    float(arc),
                )
                candidate_x_m, candidate_y_m = _waypoint_xy(candidate)
                preview_wp = candidate
                preview_wp_candidates = preview_wp.next(step_m)
                if len(preview_wp_candidates) > 0:
                    preview_wp = min(
                        preview_wp_candidates,
                        key=lambda nxt: _route_match_metrics(
                            nxt,
                            route_points,
                            cum_dists,
                            float(candidate_arc_m),
                        )[1:],
                    )
                preview_arc_eval_m, preview_route_distance_m, _ = _route_match_metrics(
                    preview_wp,
                    route_points,
                    cum_dists,
                    float(candidate_arc_m),
                )
                backward_penalty = (
                    1.0 if float(candidate_arc_m) + 1.0e-6 < float(arc) else 0.0
                )
                expected_point_distance_m = math.hypot(
                    candidate_x_m - float(look_x),
                    candidate_y_m - float(look_y),
                )
                return (
                    float(backward_penalty),
                    float(candidate_route_distance_m),
                    abs(float(candidate_arc_m) - float(expected_arc_m)),
                    float(preview_route_distance_m),
                    abs(float(preview_arc_eval_m) - float(preview_arc_m)),
                    float(candidate_heading_error_rad) + 0.05 * float(expected_point_distance_m),
                )

            wp = min(
                candidates,
                key=_route_guided_score,
            )
        elif target_xy is not None:
            wp_x, wp_y = _waypoint_xy(wp)
            current_arc = (
                project_ego_to_route(wp_x, wp_y, route_points, cum_dists)
                if route_points is not None and cum_dists is not None
                else None
            )

            def _target_guided_score(candidate: Any) -> tuple[float, float, float, float]:
                candidate_x_m, candidate_y_m = _waypoint_xy(candidate)
                distance_to_target_m = math.hypot(
                    candidate_x_m - float(target_xy[0]),
                    candidate_y_m - float(target_xy[1]),
                )
                if (
                    route_points is not None
                    and cum_dists is not None
                    and target_arc is not None
                    and current_arc is not None
                ):
                    candidate_arc = project_ego_to_route(
                        candidate_x_m,
                        candidate_y_m,
                        route_points,
                        cum_dists,
                    )
                    behind_penalty = (
                        1.0
                        if float(candidate_arc) + 1.0e-6 < float(current_arc)
                        else 0.0
                    )
                    arc_error_m = abs(float(target_arc) - float(candidate_arc))
                else:
                    behind_penalty = 0.0
                    arc_error_m = 0.0
                heading_to_target_rad = math.atan2(
                    float(target_xy[1]) - candidate_y_m,
                    float(target_xy[0]) - candidate_x_m,
                )
                candidate_yaw_rad = _waypoint_heading_rad(candidate)
                heading_error_rad = abs(
                    math.atan2(
                        math.sin(float(candidate_yaw_rad) - float(heading_to_target_rad)),
                        math.cos(float(candidate_yaw_rad) - float(heading_to_target_rad)),
                    )
                )
                return (
                    float(behind_penalty),
                    float(distance_to_target_m),
                    float(arc_error_m),
                    float(heading_error_rad),
                )
            wp = min(candidates, key=_target_guided_score)
        elif maneuver is not None:
            current_yaw = math.degrees(_waypoint_heading_rad(wp))
            maneuver_name = str(maneuver).strip().upper()
            if "LEFT" in maneuver_name:
                wp = max(
                    candidates,
                    key=lambda c: _normalize_angle_deg(
                        math.degrees(_waypoint_heading_rad(c)) - current_yaw
                    ),
                )
            elif "RIGHT" in maneuver_name:
                wp = min(
                    candidates,
                    key=lambda c: _normalize_angle_deg(
                        math.degrees(_waypoint_heading_rad(c)) - current_yaw
                    ),
                )
            else:
                wp = min(
                    candidates,
                    key=lambda c: abs(
                        _normalize_angle_deg(math.degrees(_waypoint_heading_rad(c)) - current_yaw)
                    ),
                )
        else:
            # Straightest successor
            current_yaw = math.degrees(_waypoint_heading_rad(wp))
            wp = min(
                candidates,
                key=lambda c: abs(
                    _normalize_angle_deg(math.degrees(_waypoint_heading_rad(c)) - current_yaw)
                ),
            )
        cumulative += step_m
    return wp


def move_to_lane(
    wp: Any,
    target_lane_id: int,
    *,
    allow_junction_lane_snap: bool = False,
) -> Any:
    """
    Move *wp* laterally to the lane whose project lane id is
    *target_lane_id*.

    Returns *wp* unchanged when the target lane cannot be reached.

    Junction lane snapping stays disabled by default because route-based
    reference construction should remain conservative there. The temporary
    blue-dot path can opt in when an explicit lane-change decision has
    already been made by the behavior planner.
    """
    if wp is None:
        return wp
    if bool(getattr(wp, "is_intersection", False)) and not bool(allow_junction_lane_snap):
        return wp
    return canonical_lane_waypoint_for_lane_id(wp, int(target_lane_id))


def _internal_lane_id(wp: Any) -> int:
    """Return the project lane id for *wp*."""
    return int(canonical_lane_id_for_waypoint(wp))


def _lane_width_m(waypoint: Any, default_m: float = 0.0) -> float:
    try:
        lane_width_m = float(getattr(waypoint, "lane_width_m", default_m))
    except Exception:
        lane_width_m = float(default_m)
    if not math.isfinite(lane_width_m) or lane_width_m <= 0.0:
        return max(0.0, float(default_m))
    return float(lane_width_m)


def _road_boundary_fields_for_waypoint(
    waypoint: Any,
    default_lane_width_m: float = 0.0,
) -> Dict[str, float]:
    lane_width_m = _lane_width_m(waypoint, float(default_lane_width_m))
    fallback_half_width_m = 0.5 * max(0.0, float(lane_width_m))
    fallback = {
        "road_center_offset_m": 0.0,
        "road_left_width_m": float(fallback_half_width_m),
        "road_right_width_m": float(fallback_half_width_m),
    }
    if waypoint is None:
        return fallback

    heading_rad = _waypoint_heading_rad(waypoint)
    normal_x = -math.sin(float(heading_rad))
    normal_y = math.cos(float(heading_rad))
    ref_x, ref_y = _waypoint_xy(waypoint)

    lane_waypoints = canonical_lane_waypoints(waypoint)
    if len(lane_waypoints) == 0:
        lane_waypoints = [waypoint]

    left_edge_m = -math.inf
    right_edge_m = math.inf
    for lane_wp in lane_waypoints:
        width_m = _lane_width_m(lane_wp, float(lane_width_m))
        if not math.isfinite(width_m) or width_m <= 0.0:
            continue
        lane_x_m, lane_y_m = _waypoint_xy(lane_wp)
        dx_m = lane_x_m - float(ref_x)
        dy_m = lane_y_m - float(ref_y)
        lateral_offset_m = float(normal_x) * float(dx_m) + float(normal_y) * float(dy_m)
        left_edge_m = max(float(left_edge_m), float(lateral_offset_m) + 0.5 * float(width_m))
        right_edge_m = min(float(right_edge_m), float(lateral_offset_m) - 0.5 * float(width_m))

    if not math.isfinite(left_edge_m) or not math.isfinite(right_edge_m) or left_edge_m <= right_edge_m:
        return fallback

    road_center_offset_m = 0.5 * (float(left_edge_m) + float(right_edge_m))
    road_left_width_m = float(left_edge_m) - float(road_center_offset_m)
    road_right_width_m = float(road_center_offset_m) - float(right_edge_m)
    if road_left_width_m <= 0.0 or road_right_width_m <= 0.0:
        return fallback

    return {
        "road_center_offset_m": float(road_center_offset_m),
        "road_left_width_m": float(road_left_width_m),
        "road_right_width_m": float(road_right_width_m),
    }


# -------------------------------------------------------------------- #
# Mode constants (returned as 6th element of destination list)           #
# -------------------------------------------------------------------- #
MODE_NORMAL: float = 0.0
MODE_INTERSECTION: float = 1.0


def _apply_mode_override(
    is_intersection: bool,
    mode_override: str | float | None,
) -> bool:
    if mode_override is None:
        return bool(is_intersection)
    if isinstance(mode_override, (int, float)):
        return float(mode_override) > 0.5
    override_name = str(mode_override).strip().upper()
    if override_name == "INTERSECTION":
        return True
    if override_name == "NORMAL":
        return False
    return bool(is_intersection)


# -------------------------------------------------------------------- #
# Mode determination with explicit intersection latch                     #
# -------------------------------------------------------------------- #
def _determine_mode(
    ref_wp: Any,
    ego_wp: Any,
    step_m: float,
    intersection_threshold_m: float,
    prev_mode: float | None,
    prev_road_id: int | None,
    next_macro_maneuver: str | None = None,
    prev_entered_intersection: bool = False,
) -> Tuple[bool, int, bool]:
    """Determine NORMAL vs INTERSECTION from explicit latch state.

    Automatic intersection discovery is intentionally disabled here.
    Once INTERSECTION mode starts through an explicit override, it stays
    latched until the ego has entered and then exited the junction.

    Returns ``(is_intersection, road_id, entered_intersection)``.
    """
    road_id = int(ref_wp.road_id)
    del prev_road_id
    del step_m
    del intersection_threshold_m
    del next_macro_maneuver

    ego_in_junction = bool(getattr(ego_wp, "is_intersection", False))
    was_intersection = prev_mode is not None and float(prev_mode) > 0.5
    entered_intersection = bool(prev_entered_intersection) or bool(ego_in_junction)

    if was_intersection:
        if entered_intersection:
            if ego_in_junction:
                return True, road_id, True
            return False, road_id, False
        return True, road_id, False

    return False, road_id, entered_intersection


# -------------------------------------------------------------------- #
# Shared start-wp helper                                                 #
# -------------------------------------------------------------------- #
def _start_wp_for_decision(
    ego_wp: Any,
    decision: str,
    target_lane_id: int,
    *,
    allow_junction_lane_snap: bool = False,
) -> Any:
    """Return the waypoint to walk forward from.

    The behavior planner owns lane choice, but ``lane_follow`` is a strict
    current-lane command.  Only an explicit lane-change decision may laterally
    shift the rolling target to another lane.
    """
    normalized_decision = normalize_behavior_decision(decision)
    if normalized_decision in {"lane_change_left", "lane_change_right"}:
        return move_to_lane(
            ego_wp,
            int(target_lane_id),
            allow_junction_lane_snap=bool(allow_junction_lane_snap),
        )
    return ego_wp


def _follow_route_lane_for_decision(
    decision: str,
    explicit_follow_global_route_lane: bool | None = None,
) -> bool:
    normalized_decision = str(normalize_behavior_decision(decision))
    if normalized_decision in {"lane_change_left", "lane_change_right"}:
        return False
    if explicit_follow_global_route_lane is not None:
        return bool(explicit_follow_global_route_lane)
    return normalized_decision == "reroute"


def _normalized_stop_target_state(
    stop_target_state: Sequence[float] | Mapping[str, object] | None,
) -> List[float] | None:
    if stop_target_state is None:
        return None
    if isinstance(stop_target_state, Mapping):
        try:
            return [
                float(stop_target_state.get("x_m", 0.0)),
                float(stop_target_state.get("y_m", 0.0)),
                0.0,
                float(stop_target_state.get("heading_rad", 0.0)),
                float(stop_target_state.get("lane_id", 0)),
                MODE_INTERSECTION,
                float(stop_target_state.get("road_id", -1)),
                0.0,
            ]
        except Exception:
            return None
    if not isinstance(stop_target_state, Sequence) or len(stop_target_state) < 5:
        return None
    normalized_state = [float(value) for value in list(stop_target_state[:8])]
    while len(normalized_state) < 8:
        if len(normalized_state) == 5:
            normalized_state.append(MODE_INTERSECTION)
        elif len(normalized_state) == 6:
            normalized_state.append(-1.0)
        else:
            normalized_state.append(0.0)
    normalized_state[2] = 0.0
    normalized_state[5] = MODE_INTERSECTION
    normalized_state[7] = 0.0
    return normalized_state


def _normalized_follow_target_state(
    follow_target_state: Sequence[float] | Mapping[str, object] | None,
) -> List[float] | None:
    if follow_target_state is None:
        return None
    if isinstance(follow_target_state, Mapping):
        try:
            return [
                float(follow_target_state.get("x_m", 0.0)),
                float(follow_target_state.get("y_m", 0.0)),
                max(0.0, float(follow_target_state.get("target_v_mps", 0.0))),
                float(follow_target_state.get("heading_rad", 0.0)),
                float(follow_target_state.get("lane_id", 0)),
                MODE_NORMAL,
                float(follow_target_state.get("road_id", -1)),
                0.0,
            ]
        except Exception:
            return None
    if not isinstance(follow_target_state, Sequence) or len(follow_target_state) < 5:
        return None
    normalized_state = [float(value) for value in list(follow_target_state[:8])]
    while len(normalized_state) < 8:
        normalized_state.append(0.0)
    normalized_state[2] = max(0.0, float(normalized_state[2]))
    return normalized_state


def _build_stop_reference_samples(
    map_planner: Any,
    ego_pose: Mapping[str, object],
    stop_target_state: Sequence[float] | Mapping[str, object],
    global_route_points: Sequence[Sequence[float]],
    horizon_steps: int,
    step_distance_m: float,
) -> List[Dict[str, float]]:
    normalized_stop_target_state = _normalized_stop_target_state(stop_target_state)
    if normalized_stop_target_state is None:
        return []

    stop_x_m = float(normalized_stop_target_state[0])
    stop_y_m = float(normalized_stop_target_state[1])
    stop_heading_rad = float(normalized_stop_target_state[3])
    stop_lane_id = int(round(float(normalized_stop_target_state[4])))
    ego_position = _pose_xyz(ego_pose)
    ego_z_m = float(ego_position["z"])
    stop_wp = map_planner.get_waypoint(
        {"x": stop_x_m, "y": stop_y_m, "z": ego_z_m}
    )
    stop_wp = move_to_lane(
        stop_wp,
        int(stop_lane_id),
        allow_junction_lane_snap=True,
    )
    stop_lane_width_m = _lane_width_m(stop_wp, 0.0)

    route_points_valid = (
        global_route_points
        if global_route_points is not None and len(global_route_points) >= 2
        else None
    )
    if route_points_valid is None:
        return [
            {
                "x_ref_m": float(stop_x_m),
                "y_ref_m": float(stop_y_m),
                "heading_rad": float(stop_heading_rad),
                "lane_id": int(stop_lane_id),
                "lane_width_m": float(stop_lane_width_m),
                **_road_boundary_fields_for_waypoint(stop_wp, float(stop_lane_width_m)),
            }
            for _ in range(max(1, int(horizon_steps)))
        ]

    ego_x_m = float(ego_position["x"])
    ego_y_m = float(ego_position["y"])
    route_cum_dists = _route_cum_dists(route_points_valid)
    ego_arc_m = project_ego_to_route(
        ego_x=float(ego_x_m),
        ego_y=float(ego_y_m),
        route_points=route_points_valid,
        cum_dists=route_cum_dists,
    )
    stop_arc_m = project_ego_to_route(
        ego_x=float(stop_x_m),
        ego_y=float(stop_y_m),
        route_points=route_points_valid,
        cum_dists=route_cum_dists,
    )
    if float(stop_arc_m) <= float(ego_arc_m) + 1.0e-3:
        return [
            {
                "x_ref_m": float(stop_x_m),
                "y_ref_m": float(stop_y_m),
                "heading_rad": float(stop_heading_rad),
                "lane_id": int(stop_lane_id),
                "lane_width_m": float(stop_lane_width_m),
                **_road_boundary_fields_for_waypoint(stop_wp, float(stop_lane_width_m)),
            }
            for _ in range(max(1, int(horizon_steps)))
        ]

    z_m = float(ego_z_m)
    n = max(1, int(horizon_steps))
    sd = max(0.25, float(step_distance_m))
    samples: List[Dict[str, float]] = []

    for stage_idx in range(n):
        target_arc_m = min(
            float(stop_arc_m),
            float(ego_arc_m) + float(stage_idx) * float(sd),
        )
        if float(target_arc_m) >= float(stop_arc_m) - 1.0e-6:
            samples.append(
                {
                    "x_ref_m": float(stop_x_m),
                    "y_ref_m": float(stop_y_m),
                    "heading_rad": float(stop_heading_rad),
                    "lane_id": int(stop_lane_id),
                    "lane_width_m": float(stop_lane_width_m),
                    **_road_boundary_fields_for_waypoint(stop_wp, float(stop_lane_width_m)),
                }
            )
            continue

        sample_x_m, sample_y_m = get_lookahead_route_point(
            route_points=route_points_valid,
            cum_dists=route_cum_dists,
            ego_arc=float(ego_arc_m),
            lookahead_m=float(target_arc_m) - float(ego_arc_m),
        )
        sample_wp = map_planner.get_waypoint(
            {"x": float(sample_x_m), "y": float(sample_y_m), "z": z_m}
        )
        sample_wp = move_to_lane(
            sample_wp,
            int(stop_lane_id),
            allow_junction_lane_snap=False,
        )
        if sample_wp is None:
            next_x_m, next_y_m = get_lookahead_route_point(
                route_points=route_points_valid,
                cum_dists=route_cum_dists,
                ego_arc=float(ego_arc_m),
                lookahead_m=min(
                    float(stop_arc_m) - float(ego_arc_m),
                    float(target_arc_m) - float(ego_arc_m) + float(sd),
                ),
            )
            heading_rad = math.atan2(
                float(next_y_m) - float(sample_y_m),
                float(next_x_m) - float(sample_x_m),
            )
            samples.append(
                {
                    "x_ref_m": float(sample_x_m),
                    "y_ref_m": float(sample_y_m),
                "heading_rad": float(heading_rad),
                "lane_id": int(stop_lane_id),
                "lane_width_m": float(stop_lane_width_m),
                **_road_boundary_fields_for_waypoint(stop_wp, float(stop_lane_width_m)),
            }
            )
            continue

        samples.append(
            {
                "x_ref_m": _waypoint_xy(sample_wp)[0],
                "y_ref_m": _waypoint_xy(sample_wp)[1],
                "heading_rad": _waypoint_heading_rad(sample_wp),
                "lane_id": int(stop_lane_id),
                "lane_width_m": _lane_width_m(sample_wp, float(stop_lane_width_m)),
                **_road_boundary_fields_for_waypoint(sample_wp, float(stop_lane_width_m)),
            }
        )

    return samples


def _should_follow_turn_branch_from_route(
    is_intersection: bool,
    next_macro_maneuver: str | None,
    decision: str,
) -> bool:
    maneuver_name = normalize_macro_maneuver(next_macro_maneuver)
    return (
        bool(is_intersection)
        and maneuver_name in {"left", "right"}
        and str(normalize_behavior_decision(decision)) == "lane_follow"
    )


def _route_waypoint_from_anchor(
    map_planner: Any,
    anchor_wp: Any,
    route_points: Sequence[Sequence[float]],
    lookahead_m: float,
    fallback_wp: Any,
    target_lane_id: int | None = None,
    follow_route_lane: bool = True,
    allow_junction_lane_snap: bool = False,
) -> Any:
    if anchor_wp is None or route_points is None or len(route_points) < 2:
        return fallback_wp

    anchor_position = _waypoint_xyz(anchor_wp)

    cum_dists = _route_cum_dists(route_points)
    anchor_arc = project_ego_to_route(
        ego_x=float(anchor_position["x"]),
        ego_y=float(anchor_position["y"]),
        route_points=route_points,
        cum_dists=cum_dists,
    )
    target_x_m, target_y_m = get_lookahead_route_point(
        route_points=route_points,
        cum_dists=cum_dists,
        ego_arc=float(anchor_arc),
        lookahead_m=float(lookahead_m),
    )
    route_wp = map_planner.get_waypoint(
        {
            "x": float(target_x_m),
            "y": float(target_y_m),
            "z": float(anchor_position["z"]),
        }
    )
    resolved_wp = route_wp if route_wp is not None else fallback_wp
    if (
        not bool(follow_route_lane)
        and route_wp is not None
        and bool(getattr(route_wp, "is_intersection", False))
        and fallback_wp is not None
    ):
        resolved_wp = fallback_wp
    resolved_wp = _snap_junction_waypoint_to_route(
        map_planner=map_planner,
        waypoint=resolved_wp,
        anchor_wp=anchor_wp,
        route_points=route_points,
        cum_dists=cum_dists,
        anchor_arc_m=float(anchor_arc),
    )
    if bool(follow_route_lane) or resolved_wp is None or target_lane_id is None:
        return resolved_wp
    return move_to_lane(
        resolved_wp,
        int(target_lane_id),
        allow_junction_lane_snap=bool(allow_junction_lane_snap),
    )


def _build_route_reference_samples_from_anchor(
    map_planner: Any,
    anchor_wp: Any,
    route_points: Sequence[Sequence[float]],
    horizon_steps: int,
    step_distance_m: float,
    fallback_lane_id: int,
    target_lane_id: int | None = None,
    follow_route_lane: bool = True,
) -> List[Dict[str, float]]:
    if anchor_wp is None or route_points is None or len(route_points) < 2:
        return []

    anchor_position = _waypoint_xyz(anchor_wp)

    cum_dists = _route_cum_dists(route_points)
    anchor_arc = project_ego_to_route(
        ego_x=float(anchor_position["x"]),
        ego_y=float(anchor_position["y"]),
        route_points=route_points,
        cum_dists=cum_dists,
    )
    z_m = float(anchor_position["z"])
    samples: List[Dict[str, float]] = []
    n = max(1, int(horizon_steps))
    sd = max(0.25, float(step_distance_m))
    continuity_wp = anchor_wp

    for k in range(n):
        x_ref_m, y_ref_m = get_lookahead_route_point(
            route_points=route_points,
            cum_dists=cum_dists,
            ego_arc=float(anchor_arc),
            lookahead_m=float(k) * float(sd),
        )
        route_wp = map_planner.get_waypoint(
            {"x": float(x_ref_m), "y": float(y_ref_m), "z": z_m}
        )
        walked_wp = continuity_wp
        if k > 0 and continuity_wp is not None:
            walked_wp = _walk_forward(
                continuity_wp,
                float(sd),
                min(float(sd), float(DEFAULT_STEP_M)),
                route_points=route_points,
                cum_dists=cum_dists,
            )
        if route_wp is not None:
            sample_wp = route_wp
            if bool(follow_route_lane) and walked_wp is not None:
                _, projected_route_distance_m, projected_heading_error_rad = _route_match_metrics(
                    route_wp,
                    route_points,
                    cum_dists,
                    float(anchor_arc),
                )
                _, walked_route_distance_m, walked_heading_error_rad = _route_match_metrics(
                    walked_wp,
                    route_points,
                    cum_dists,
                    float(anchor_arc),
                )
                if (
                    float(walked_route_distance_m) + 0.25 < float(projected_route_distance_m)
                    or (
                        float(walked_route_distance_m) <= float(projected_route_distance_m) + 0.25
                        and float(walked_heading_error_rad) + 0.05 < float(projected_heading_error_rad)
                    )
                ):
                    sample_wp = walked_wp
            if not bool(follow_route_lane) and target_lane_id is not None:
                sample_wp = move_to_lane(route_wp, int(target_lane_id))
            continuity_wp = sample_wp
            next_x_ref_m, next_y_ref_m = get_lookahead_route_point(
                route_points=route_points,
                cum_dists=cum_dists,
                ego_arc=float(anchor_arc),
                lookahead_m=float(min(k + 1, n - 1)) * float(sd),
            )
            route_heading_rad = math.atan2(
                float(next_y_ref_m) - float(y_ref_m),
                float(next_x_ref_m) - float(x_ref_m),
            )
            use_exact_route_xy = bool(follow_route_lane)
            # Blend CARLA waypoint tangent (accurate road direction) with the
            # route chord (stable, noise-free).  Pure chord biases heading
            # toward the outside of curves causing the vehicle to run wide;
            # pure waypoint rotation.yaw can have CARLA discretization jumps
            # that amplify into steering oscillation at high q_psi.
            # 65 % waypoint tangent + 35 % chord is a good compromise.
            _wp_heading_raw = _waypoint_heading_rad(sample_wp)
            _d_heading = math.atan2(
                math.sin(_wp_heading_raw - route_heading_rad),
                math.cos(_wp_heading_raw - route_heading_rad),
            )
            blended_heading_rad = route_heading_rad + 0.65 * _d_heading
            samples.append({
                "x_ref_m": float(x_ref_m) if use_exact_route_xy else _waypoint_xy(sample_wp)[0],
                "y_ref_m": float(y_ref_m) if use_exact_route_xy else _waypoint_xy(sample_wp)[1],
                "heading_rad": float(blended_heading_rad),
                "lane_id": (
                    int(target_lane_id)
                    if not bool(follow_route_lane) and target_lane_id is not None
                    else _internal_lane_id(sample_wp)
                ),
                "lane_width_m": _lane_width_m(sample_wp, _lane_width_m(anchor_wp, 0.0)),
                **_road_boundary_fields_for_waypoint(
                    sample_wp,
                    _lane_width_m(sample_wp, _lane_width_m(anchor_wp, 0.0)),
                ),
            })
            continue

        next_x_ref_m, next_y_ref_m = get_lookahead_route_point(
            route_points=route_points,
            cum_dists=cum_dists,
            ego_arc=float(anchor_arc),
            lookahead_m=float(k + 1) * float(sd),
        )
        heading_rad = math.atan2(
            float(next_y_ref_m) - float(y_ref_m),
            float(next_x_ref_m) - float(x_ref_m),
        )
        if bool(follow_route_lane) and walked_wp is not None:
            continuity_wp = walked_wp
        samples.append({
            "x_ref_m": float(x_ref_m),
            "y_ref_m": float(y_ref_m),
            "heading_rad": float(heading_rad),
            "lane_id": int(
                target_lane_id
                if target_lane_id is not None
                else (
                    _internal_lane_id(walked_wp)
                    if walked_wp is not None
                    else fallback_lane_id
                )
            ),
            "lane_width_m": _lane_width_m(
                walked_wp if walked_wp is not None else anchor_wp,
                0.0,
            ),
            **_road_boundary_fields_for_waypoint(
                walked_wp if walked_wp is not None else anchor_wp,
                _lane_width_m(walked_wp if walked_wp is not None else anchor_wp, 0.0),
            ),
        })
    return samples


def _build_forward_reference_samples(
    anchor_wp: Any,
    *,
    horizon_steps: int,
    step_distance_m: float,
    route_points: Sequence[Sequence[float]] | None = None,
    fallback_lane_id: int,
    maneuver: str | None = None,
) -> List[Dict[str, float]]:
    if anchor_wp is None:
        return []

    route_points_valid = route_points if route_points is not None and len(route_points) >= 2 else None
    cum_dists = _route_cum_dists(route_points_valid) if route_points_valid is not None else None
    samples: List[Dict[str, float]] = []
    wp = anchor_wp
    n = max(1, int(horizon_steps))
    sd = max(0.25, float(step_distance_m))

    for k in range(n):
        if k > 0:
            candidates = wp.next(sd)
            if candidates:
                if len(candidates) == 1:
                    wp = candidates[0]
                elif maneuver is not None:
                    current_yaw = math.degrees(_waypoint_heading_rad(wp))
                    maneuver_name = str(maneuver).strip().upper()
                    if "LEFT" in maneuver_name:
                        wp = max(
                            candidates,
                            key=lambda c: _normalize_angle_deg(
                                math.degrees(_waypoint_heading_rad(c)) - current_yaw
                            ),
                        )
                    elif "RIGHT" in maneuver_name:
                        wp = min(
                            candidates,
                            key=lambda c: _normalize_angle_deg(
                                math.degrees(_waypoint_heading_rad(c)) - current_yaw
                            ),
                        )
                    else:
                        wp = min(
                            candidates,
                            key=lambda c: abs(
                                _normalize_angle_deg(
                                    math.degrees(_waypoint_heading_rad(c)) - current_yaw
                                )
                            ),
                        )
                elif route_points_valid is not None and cum_dists is not None:
                    wp_x, wp_y = _waypoint_xy(wp)
                    arc = project_ego_to_route(wp_x, wp_y, route_points_valid, cum_dists)
                    lx, ly = get_lookahead_route_point(route_points_valid, cum_dists, arc, sd * 3)
                    wp = min(
                        candidates,
                        key=lambda c: (
                            (_waypoint_xy(c)[0] - lx) ** 2
                            + (_waypoint_xy(c)[1] - ly) ** 2
                        ),
                    )
                else:
                    current_yaw = math.degrees(_waypoint_heading_rad(wp))
                    wp = min(
                        candidates,
                        key=lambda c: abs(
                            _normalize_angle_deg(
                                math.degrees(_waypoint_heading_rad(c)) - current_yaw
                            )
                        ),
                    )
        samples.append({
            "x_ref_m": _waypoint_xy(wp)[0],
            "y_ref_m": _waypoint_xy(wp)[1],
            "heading_rad": _waypoint_heading_rad(wp),
            "lane_id": _internal_lane_id(wp) if wp is not None else int(fallback_lane_id),
            "lane_width_m": _lane_width_m(wp, 0.0),
            **_road_boundary_fields_for_waypoint(wp, _lane_width_m(wp, 0.0)),
        })
    return samples


def _build_lane_reference_samples_to_target(
    *,
    start_wp: Any,
    target_wp: Any,
    horizon_steps: int,
    step_distance_m: float,
    walk_step_m: float,
    route_points: Sequence[Sequence[float]] | None = None,
    maneuver: str | None = None,
    target_x_m: float | None = None,
    target_y_m: float | None = None,
    target_heading_rad: float | None = None,
    target_lane_id: int | None = None,
) -> List[Dict[str, float]]:
    """Build lane-center samples from *start_wp* forward to *target_wp*.

    Samples follow the lane containing *start_wp*. Once the progression
    reaches the target distance, the target waypoint is repeated for the
    remaining horizon stages.
    """
    if start_wp is None or target_wp is None:
        return []

    start_position = _waypoint_xyz(start_wp)
    target_position = _waypoint_xyz(target_wp)

    exact_target_x_m = float(target_x_m) if target_x_m is not None else float(target_position["x"])
    exact_target_y_m = float(target_y_m) if target_y_m is not None else float(target_position["y"])
    exact_target_heading_rad = (
        float(target_heading_rad)
        if target_heading_rad is not None
        else _waypoint_heading_rad(target_wp)
    )
    exact_target_lane_id = (
        int(target_lane_id)
        if target_lane_id is not None
        else int(_internal_lane_id(target_wp))
    )
    guidance_target_position = {
        "x": exact_target_x_m,
        "y": exact_target_y_m,
        "z": float(target_position["z"]),
    }

    route_points_valid = (
        route_points if route_points is not None and len(route_points) >= 2 else None
    )
    route_cum_dists = (
        _route_cum_dists(route_points_valid)
        if route_points_valid is not None
        else None
    )
    distance_to_target_m = math.hypot(
        float(exact_target_x_m) - float(start_position["x"]),
        float(exact_target_y_m) - float(start_position["y"]),
    )
    if route_points_valid is not None:
        start_arc_m = project_ego_to_route(
            ego_x=float(start_position["x"]),
            ego_y=float(start_position["y"]),
            route_points=route_points_valid,
            cum_dists=route_cum_dists,
        )
        target_arc_m = project_ego_to_route(
            ego_x=float(exact_target_x_m),
            ego_y=float(exact_target_y_m),
            route_points=route_points_valid,
            cum_dists=route_cum_dists,
        )
        route_distance_m = max(0.0, float(target_arc_m) - float(start_arc_m))
        if float(route_distance_m) > 1.0e-6:
            distance_to_target_m = float(route_distance_m)

    samples: List[Dict[str, float]] = []
    n = max(1, int(horizon_steps))
    sd = max(0.25, float(step_distance_m))
    guided_path: List[Any] = [start_wp]
    guided_cum_dists: List[float] = [0.0]
    current_wp = start_wp
    best_distance_to_target_m = float(distance_to_target_m)
    stagnation_count = 0
    max_guided_steps = max(8, int(math.ceil(max(1.0, float(distance_to_target_m)) / max(0.25, float(walk_step_m)))) + 4)

    for _ in range(max_guided_steps):
        next_wp = _walk_forward(
            current_wp,
            float(walk_step_m),
            float(walk_step_m),
            route_points=route_points_valid,
            cum_dists=route_cum_dists,
            maneuver=maneuver,
            target_position=guidance_target_position,
        )
        next_x_m, next_y_m = _waypoint_xy(next_wp)
        current_x_m, current_y_m = _waypoint_xy(current_wp)
        segment_length_m = math.hypot(
            next_x_m - current_x_m,
            next_y_m - current_y_m,
        )
        if float(segment_length_m) <= 1.0e-6:
            break
        guided_path.append(next_wp)
        guided_cum_dists.append(float(guided_cum_dists[-1]) + float(segment_length_m))
        current_wp = next_wp
        current_distance_to_target_m = math.hypot(
            next_x_m - float(exact_target_x_m),
            next_y_m - float(exact_target_y_m),
        )
        if float(current_distance_to_target_m) <= max(0.5, 0.5 * float(walk_step_m)):
            break
        if float(current_distance_to_target_m) + 1.0e-3 < float(best_distance_to_target_m):
            best_distance_to_target_m = float(current_distance_to_target_m)
            stagnation_count = 0
        else:
            stagnation_count += 1
            if stagnation_count >= 3:
                break

    path_length_m = float(guided_cum_dists[-1]) if len(guided_cum_dists) > 0 else 0.0
    if float(path_length_m) > float(distance_to_target_m):
        distance_to_target_m = float(path_length_m)

    for stage_idx in range(n):
        target_distance_m = min(
            float(distance_to_target_m),
            float(stage_idx) * float(sd),
        )
        if float(target_distance_m) >= float(distance_to_target_m) - 1.0e-6:
            lane_width_m = _lane_width_m(target_wp, 0.0)
            samples.append(
                {
                    "x_ref_m": float(exact_target_x_m),
                    "y_ref_m": float(exact_target_y_m),
                    "heading_rad": float(exact_target_heading_rad),
                    "lane_id": int(exact_target_lane_id),
                    "lane_width_m": float(lane_width_m),
                    **_road_boundary_fields_for_waypoint(target_wp, float(lane_width_m)),
                }
            )
            continue
        sample_index = 0
        while (
            sample_index + 1 < len(guided_cum_dists)
            and float(guided_cum_dists[sample_index + 1]) <= float(target_distance_m) + 1.0e-6
        ):
            sample_index += 1
        sample_wp = guided_path[sample_index]

        lane_width_m = _lane_width_m(sample_wp, _lane_width_m(target_wp, 0.0))
        samples.append(
            {
                "x_ref_m": _waypoint_xy(sample_wp)[0],
                "y_ref_m": _waypoint_xy(sample_wp)[1],
                "heading_rad": _waypoint_heading_rad(sample_wp),
                "lane_id": _internal_lane_id(sample_wp),
                "lane_width_m": float(lane_width_m),
                **_road_boundary_fields_for_waypoint(sample_wp, float(lane_width_m)),
            }
        )

    return samples


def _interpolate_heading_rad(start_heading_rad: float, end_heading_rad: float, alpha: float) -> float:
    alpha = min(1.0, max(0.0, float(alpha)))
    start_x = math.cos(float(start_heading_rad))
    start_y = math.sin(float(start_heading_rad))
    end_x = math.cos(float(end_heading_rad))
    end_y = math.sin(float(end_heading_rad))
    blend_x = (1.0 - alpha) * start_x + alpha * end_x
    blend_y = (1.0 - alpha) * start_y + alpha * end_y
    if math.hypot(blend_x, blend_y) <= 1e-9:
        return float(start_heading_rad)
    return float(math.atan2(blend_y, blend_x))


def _smooth_lane_change_alpha(raw_alpha: float) -> float:
    """Return a zero-slope endpoint blend factor for lane-change references."""
    alpha = min(1.0, max(0.0, float(raw_alpha)))
    return float(alpha * alpha * alpha * (alpha * (alpha * 6.0 - 15.0) + 10.0))


def _blend_reference_samples(
    source_samples: Sequence[Mapping[str, object]],
    target_samples: Sequence[Mapping[str, object]],
    *,
    blend_steps: int,
) -> List[Dict[str, float]]:
    if len(target_samples) == 0:
        return [dict(sample) for sample in source_samples if isinstance(sample, Mapping)]
    if len(source_samples) == 0:
        return [dict(sample) for sample in target_samples if isinstance(sample, Mapping)]

    normalized_source = [dict(sample) for sample in source_samples if isinstance(sample, Mapping)]
    normalized_target = [dict(sample) for sample in target_samples if isinstance(sample, Mapping)]
    if len(normalized_source) == 0:
        return normalized_target
    if len(normalized_target) == 0:
        return normalized_source

    while len(normalized_source) < len(normalized_target):
        normalized_source.append(dict(normalized_source[-1]))
    while len(normalized_target) < len(normalized_source):
        normalized_target.append(dict(normalized_target[-1]))

    transition_steps = max(1, min(int(blend_steps), len(normalized_target) - 1))
    blended_samples: List[Dict[str, float]] = []

    for stage_idx in range(len(normalized_target)):
        source_sample = normalized_source[stage_idx]
        target_sample = normalized_target[stage_idx]
        if stage_idx >= transition_steps:
            completed_sample = dict(target_sample)
            completed_sample["blend_alpha"] = 1.0
            completed_sample["blend_raw_alpha"] = 1.0
            completed_sample["blend_transition_steps"] = int(transition_steps)
            completed_sample["blend_type"] = "lane_change_smootherstep"
            blended_samples.append(completed_sample)
            continue

        raw_alpha = float(stage_idx) / float(transition_steps)
        alpha = _smooth_lane_change_alpha(raw_alpha)
        source_heading_rad = float(source_sample.get("heading_rad", 0.0))
        target_heading_rad = float(target_sample.get("heading_rad", source_heading_rad))
        blended_samples.append(
            {
                "x_ref_m": (1.0 - alpha) * float(source_sample.get("x_ref_m", 0.0))
                + alpha * float(target_sample.get("x_ref_m", 0.0)),
                "y_ref_m": (1.0 - alpha) * float(source_sample.get("y_ref_m", 0.0))
                + alpha * float(target_sample.get("y_ref_m", 0.0)),
                "heading_rad": _interpolate_heading_rad(
                    source_heading_rad,
                    target_heading_rad,
                    alpha,
                ),
                "lane_id": int(
                    target_sample.get("lane_id", source_sample.get("lane_id", 0))
                    if alpha >= 0.5
                    else source_sample.get("lane_id", target_sample.get("lane_id", 0))
                ),
                "lane_width_m": (
                    (1.0 - alpha) * float(source_sample.get("lane_width_m", 0.0))
                    + alpha * float(
                        target_sample.get(
                            "lane_width_m",
                            source_sample.get("lane_width_m", 0.0),
                        )
                    )
                ),
                "road_center_offset_m": (
                    (1.0 - alpha) * float(source_sample.get("road_center_offset_m", 0.0))
                    + alpha * float(target_sample.get("road_center_offset_m", 0.0))
                ),
                "road_left_width_m": (
                    (1.0 - alpha) * float(source_sample.get("road_left_width_m", 0.0))
                    + alpha * float(target_sample.get("road_left_width_m", 0.0))
                ),
                "road_right_width_m": (
                    (1.0 - alpha) * float(source_sample.get("road_right_width_m", 0.0))
                    + alpha * float(target_sample.get("road_right_width_m", 0.0))
                ),
                "blend_alpha": float(alpha),
                "blend_raw_alpha": float(raw_alpha),
                "blend_transition_steps": int(transition_steps),
                "blend_type": "lane_change_smootherstep",
            }
        )
    return blended_samples


# -------------------------------------------------------------------- #
# Public API                                                             #
# -------------------------------------------------------------------- #
def compute_temp_destination_mode(
    map_planner: Any,
    ego_pose: Mapping[str, object],
    mode_reference_xy: Tuple[float, float] | None = None,
    prev_mode: float | None = None,
    prev_road_id: int | None = None,
    prev_entered_intersection: bool = False,
    next_macro_maneuver: str | None = None,
    mode_override: str | float | None = None,
    intersection_threshold_m: float = INTERSECTION_THRESHOLD_M,
    step_m: float = DEFAULT_STEP_M,
) -> Tuple[float, int, bool]:
    """Return the current blue-dot mode using the same explicit latch logic."""
    ego_position = _pose_xyz(ego_pose)

    ego_wp = map_planner.get_waypoint(ego_position)
    if ego_wp is None:
        return MODE_NORMAL, -1, False

    if mode_reference_xy is not None:
        ref_wp = map_planner.get_waypoint(
            {
                "x": float(mode_reference_xy[0]),
                "y": float(mode_reference_xy[1]),
                "z": float(ego_position["z"]),
            }
        )
        if ref_wp is None:
            ref_wp = ego_wp
    else:
        ref_wp = ego_wp

    is_intersection, road_id, entered_intersection = _determine_mode(
        ref_wp,
        ego_wp,
        float(step_m),
        float(intersection_threshold_m),
        prev_mode,
        prev_road_id,
        next_macro_maneuver,
        prev_entered_intersection=bool(prev_entered_intersection),
    )
    is_intersection = _apply_mode_override(
        is_intersection=is_intersection,
        mode_override=mode_override,
    )
    return (
        MODE_INTERSECTION if bool(is_intersection) else MODE_NORMAL,
        int(road_id),
        bool(entered_intersection) and bool(is_intersection),
    )


def compute_temp_destination(
    map_planner: Any,
    ego_pose: Mapping[str, object],
    target_lane_id: int,
    decision: str,
    lookahead_m: float,
    target_v_mps: float,
    global_route_points: Sequence[Sequence[float]],
    mode_reference_xy: Tuple[float, float] | None = None,
    prev_mode: float | None = None,
    prev_road_id: int | None = None,
    prev_entered_intersection: bool = False,
    next_macro_maneuver: str | None = None,
    mode_override: str | float | None = None,
    intersection_threshold_m: float = INTERSECTION_THRESHOLD_M,
    step_m: float = DEFAULT_STEP_M,
    stop_target_state: Sequence[float] | Mapping[str, object] | None = None,
    follow_target_state: Sequence[float] | Mapping[str, object] | None = None,
    follow_global_route_lane: bool | None = None,
) -> List[float]:
    """
    Compute the temporary destination state for MPC.

    Parameters
    ----------
    decision : str
        ``"LANE_KEEP"``, ``"LANE_CHANGE_LEFT"``, or
        ``"LANE_CHANGE_RIGHT"``. The selected lane always comes from
        ``target_lane_id``; the global route only provides longitudinal
        progress except during ``reroute``.
    mode_reference_xy : tuple | None
        ``(x, y)`` of the **previous** blue-dot position.  Mode is
        determined from this point.  Falls back to ego on first frame.
    prev_mode : float | None
        Previous frame's mode (``MODE_NORMAL`` / ``MODE_INTERSECTION``).
    prev_road_id : int | None
        Previous frame's ``road_id`` (7th element of result).  Preserved
        for compatibility with the rolling state layout.

    Returns
    -------
    [x, y, v_target, heading_rad, lane_id, mode, road_id, entered_intersection]

    *mode* is ``0.0`` for NORMAL and ``1.0`` for INTERSECTION.
    *road_id* is the OpenDRIVE ``road_id`` of the reference waypoint.
    *entered_intersection* is ``1.0`` while the ego has entered the
    current latched intersection and has not yet exited it.
    """
    ego_position = _pose_xyz(ego_pose)
    ego_x = float(ego_position["x"])
    ego_y = float(ego_position["y"])
    ego_z = float(ego_position["z"])
    ego_psi = float(ego_pose.get("heading_rad", 0.0))

    ego_wp = map_planner.get_waypoint(ego_position)
    if ego_wp is None:
        return [ego_x, ego_y, 0.0, ego_psi, 0,
                MODE_NORMAL, -1, 0.0]

    # ---- Reference wp for mode check (previous blue-dot pos) ---- #
    if mode_reference_xy is not None:
        ref_wp = map_planner.get_waypoint(
            {
                "x": float(mode_reference_xy[0]),
                "y": float(mode_reference_xy[1]),
                "z": ego_z,
            }
        )
        if ref_wp is None:
            ref_wp = ego_wp
    else:
        ref_wp = ego_wp

    # ---- Mode with road latch ---- #
    is_intersection, road_id, entered_intersection = _determine_mode(
        ref_wp, ego_wp, step_m, float(intersection_threshold_m),
        prev_mode, prev_road_id, next_macro_maneuver,
        prev_entered_intersection=bool(prev_entered_intersection),
    )
    is_intersection = _apply_mode_override(
        is_intersection=is_intersection,
        mode_override=mode_override,
    )
    if not bool(is_intersection):
        entered_intersection = False

    normalized_stop_target_state = _normalized_stop_target_state(stop_target_state)
    if (
        bool(is_fixed_stop_decision(decision))
        and normalized_stop_target_state is not None
    ):
        normalized_stop_target_state[6] = float(int(round(float(normalized_stop_target_state[6]))))
        normalized_stop_target_state[5] = MODE_INTERSECTION
        normalized_stop_target_state[7] = 1.0 if bool(entered_intersection) else 0.0
        return list(normalized_stop_target_state)

    normalized_follow_target_state = _normalized_follow_target_state(follow_target_state)
    if (
        bool(is_emergency_brake_decision(decision))
        and normalized_follow_target_state is not None
    ):
        normalized_follow_target_state[5] = (
            MODE_INTERSECTION if bool(is_intersection) else MODE_NORMAL
        )
        normalized_follow_target_state[6] = float(int(round(float(normalized_follow_target_state[6]))))
        normalized_follow_target_state[7] = (
            1.0 if bool(entered_intersection) and bool(is_intersection) else 0.0
        )
        return list(normalized_follow_target_state)

    normalized_decision = normalize_behavior_decision(decision)
    current_lane_id = _internal_lane_id(ego_wp)
    route_alignment_lane_id = (
        int(target_lane_id)
        if int(target_lane_id) != 0
        else int(current_lane_id)
    )
    route_points_valid = (
        global_route_points
        if global_route_points is not None and len(global_route_points) >= 2
        else None
    )
    follow_route_lane = _follow_route_lane_for_decision(
        str(normalized_decision),
        explicit_follow_global_route_lane=follow_global_route_lane,
    )
    blue_dot_target_lane_id = int(route_alignment_lane_id)
    blue_dot_follow_route_lane = bool(follow_route_lane)
    if (
        str(normalized_decision) == "lane_follow"
        and int(route_alignment_lane_id) != int(current_lane_id)
    ):
        # The route may already be on an adjacent lane/connector while the ego
        # is still in the current lane. Keep plain lane-follow on the ego lane.
        blue_dot_target_lane_id = int(current_lane_id)
        blue_dot_follow_route_lane = False

    # ---- Start wp (lane shift only on lane-change decision) ---- #
    # Do not snap laterally inside junctions. Junction lane topology is often
    # connector-like, so a lateral snap can move the blue dot off the intended
    # route branch and make MPC cut across the intersection.
    start_wp = _start_wp_for_decision(
        ego_wp, str(decision), int(blue_dot_target_lane_id),
        allow_junction_lane_snap=(
            str(normalized_decision) in {"lane_change_left", "lane_change_right"}
            and not bool(is_intersection)
        ),
    )
    if route_points_valid is not None:
        route_cum_dists = _route_cum_dists(route_points_valid)
        fallback_wp = _walk_forward(
            start_wp,
            float(lookahead_m),
            step_m,
            route_points=route_points_valid,
            cum_dists=route_cum_dists,
            maneuver=str(next_macro_maneuver or "") if bool(is_intersection) else None,
        )
        # Blue-dot longitudinal progress must always be measured from the
        # ego pose projected onto the global route, not from the previous
        # blue-dot position. ``mode_reference_xy`` is only for mode latching.
        anchor_wp = ego_wp
        anchor_x_m, anchor_y_m = _waypoint_xy(anchor_wp)
        anchor_arc = project_ego_to_route(
            ego_x=anchor_x_m,
            ego_y=anchor_y_m,
            route_points=route_points_valid,
            cum_dists=route_cum_dists,
        )
        route_x_m, route_y_m = get_lookahead_route_point(
            route_points=route_points_valid,
            cum_dists=route_cum_dists,
            ego_arc=float(anchor_arc),
            lookahead_m=float(lookahead_m),
        )
        next_route_x_m, next_route_y_m = get_lookahead_route_point(
            route_points=route_points_valid,
            cum_dists=route_cum_dists,
            ego_arc=float(anchor_arc),
            lookahead_m=float(lookahead_m) + max(0.5, float(step_m)),
        )
        route_heading_rad = math.atan2(
            float(next_route_y_m) - float(route_y_m),
            float(next_route_x_m) - float(route_x_m),
        )
        if not math.isfinite(float(route_heading_rad)) or (
            abs(float(next_route_x_m) - float(route_x_m)) < 1.0e-6
            and abs(float(next_route_y_m) - float(route_y_m)) < 1.0e-6
        ):
            route_heading_rad = float(ego_psi)

        route_candidate_wp = _route_waypoint_from_anchor(
            map_planner=map_planner,
            anchor_wp=anchor_wp,
            route_points=route_points_valid,
            lookahead_m=float(lookahead_m),
            fallback_wp=fallback_wp,
            target_lane_id=int(blue_dot_target_lane_id),
            follow_route_lane=bool(blue_dot_follow_route_lane),
        )
        if bool(blue_dot_follow_route_lane):
            route_wp = route_candidate_wp
        else:
            # When the behavior layer says to hold the current/selected lane,
            # the temporary destination must come from walking CARLA waypoints
            # on that lane. Projecting the global route first can snap the blue
            # dot to an adjacent road or turn connector before a lane-change
            # decision exists. If the forward walk cannot move, allow a route
            # projection only when it remains on the requested lane.
            fallback_x_m, fallback_y_m = _waypoint_xy(fallback_wp)
            fallback_progress_m = math.hypot(
                fallback_x_m - float(ego_x),
                fallback_y_m - float(ego_y),
            )
            candidate_lane_id = _internal_lane_id(route_candidate_wp)
            if (
                float(fallback_progress_m) < 0.25
                and int(candidate_lane_id) == int(blue_dot_target_lane_id)
            ):
                route_wp = route_candidate_wp
            else:
                route_wp = fallback_wp
        mode = MODE_INTERSECTION if bool(is_intersection) else MODE_NORMAL
        if (
            str(normalized_decision) == "lane_follow"
            and not bool(blue_dot_follow_route_lane)
            and not bool(is_intersection)
        ):
            ego_road_id = int(getattr(ego_wp, "road_id", road_id))
            route_road_id = int(getattr(route_wp, "road_id", ego_road_id))
            if int(route_road_id) != int(ego_road_id):
                lane_hold_wp = _walk_forward(
                    start_wp,
                    float(lookahead_m),
                    step_m,
                )
                lane_hold_road_id = int(getattr(lane_hold_wp, "road_id", ego_road_id))
                if int(lane_hold_road_id) == int(ego_road_id):
                    route_wp = lane_hold_wp

        # Do not use raw global-route geometry merely because the upcoming
        # point is in INTERSECTION mode.  On approaches to a junction the route
        # can already lie on a different lane/connector, which makes the blue
        # dot jump sideways and creates a false lane-change/turn intention.
        # Only follow route x/y when the caller explicitly says the route lane
        # should be followed. Otherwise keep the rolling target on the selected
        # CARLA lane waypoint.
        use_route_geometry = bool(blue_dot_follow_route_lane)
        return [
            float(route_x_m) if bool(use_route_geometry) else _waypoint_xy(route_wp)[0],
            float(route_y_m) if bool(use_route_geometry) else _waypoint_xy(route_wp)[1],
            float(target_v_mps),
            float(route_heading_rad) if bool(use_route_geometry) else _waypoint_heading_rad(route_wp),
            int(blue_dot_target_lane_id) if not bool(blue_dot_follow_route_lane) else _internal_lane_id(route_wp),
            mode,
            int(getattr(route_wp, "road_id", road_id)),
            1.0 if bool(entered_intersection) and float(mode) > 0.5 else 0.0,
        ]

    if bool(is_intersection):
        wp = _walk_forward(
            start_wp,
            float(lookahead_m),
            step_m,
            maneuver=str(next_macro_maneuver or ""),
        )
        mode = MODE_INTERSECTION
    else:
        wp = _walk_forward(
            start_wp,
            float(lookahead_m),
            step_m,
        )
        mode = MODE_NORMAL

    return [
        _waypoint_xy(wp)[0],
        _waypoint_xy(wp)[1],
        float(target_v_mps),
        _waypoint_heading_rad(wp),
        _internal_lane_id(wp),
        mode,
        road_id,
        1.0 if bool(entered_intersection) and float(mode) > 0.5 else 0.0,
    ]


def build_reference_samples(
    map_planner: Any,
    ego_pose: Mapping[str, object],
    target_lane_id: int,
    decision: str,
    horizon_steps: int,
    step_distance_m: float,
    global_route_points: Sequence[Sequence[float]],
    mode_reference_xy: Tuple[float, float] | None = None,
    prev_mode: float | None = None,
    prev_road_id: int | None = None,
    prev_entered_intersection: bool = False,
    next_macro_maneuver: str | None = None,
    mode_override: str | float | None = None,
    intersection_threshold_m: float = INTERSECTION_THRESHOLD_M,
    walk_step_m: float = DEFAULT_STEP_M,
    stop_target_state: Sequence[float] | Mapping[str, object] | None = None,
    follow_target_state: Sequence[float] | Mapping[str, object] | None = None,
    follow_global_route_lane: bool | None = None,
    force_stop_reference: bool = False,
) -> List[Dict[str, float]]:
    """
    Build a reference trajectory for MPC's lane-centre cost.

    Returns one sample per horizon step with
    ``{"x_ref_m", "y_ref_m", "heading_rad", "lane_id"}``.
    """
    ego_position = _pose_xyz(ego_pose)
    ego_x = float(ego_position["x"])
    ego_y = float(ego_position["y"])
    ego_z = float(ego_position["z"])
    ego_psi = float(ego_pose.get("heading_rad", 0.0))
    n = max(1, int(horizon_steps))
    sd = max(0.25, float(step_distance_m))

    ego_wp = map_planner.get_waypoint(ego_position)
    if ego_wp is None:
        return [
            {
                "x_ref_m": ego_x + float(k) * sd * math.cos(ego_psi),
                "y_ref_m": ego_y + float(k) * sd * math.sin(ego_psi),
                "heading_rad": ego_psi,
                "lane_id": int(target_lane_id),
                "lane_width_m": 0.0,
                **_road_boundary_fields_for_waypoint(None, 0.0),
            }
            for k in range(n)
        ]

    # ---- Reference wp for mode check ---- #
    if mode_reference_xy is not None:
        ref_wp = map_planner.get_waypoint(
            {
                "x": float(mode_reference_xy[0]),
                "y": float(mode_reference_xy[1]),
                "z": ego_z,
            }
        )
        if ref_wp is None:
            ref_wp = ego_wp
    else:
        ref_wp = ego_wp

    is_intersection, _, _ = _determine_mode(
        ref_wp, ego_wp, walk_step_m, float(intersection_threshold_m),
        prev_mode, prev_road_id, next_macro_maneuver,
        prev_entered_intersection=bool(prev_entered_intersection),
    )
    is_intersection = _apply_mode_override(
        is_intersection=is_intersection,
        mode_override=mode_override,
    )

    normalized_decision = normalize_behavior_decision(decision)
    if (
        (
            bool(is_fixed_stop_decision(normalized_decision))
            or bool(force_stop_reference)
        )
        and stop_target_state is not None
    ):
        stop_reference_samples = _build_stop_reference_samples(
            map_planner=map_planner,
            ego_pose=ego_pose,
            stop_target_state=stop_target_state,
            global_route_points=global_route_points,
            horizon_steps=n,
            step_distance_m=sd,
        )
        if len(stop_reference_samples) > 0:
            return stop_reference_samples
    del follow_target_state
    current_lane_id = _internal_lane_id(ego_wp)
    route_alignment_lane_id = (
        int(target_lane_id)
        if int(target_lane_id) != 0
        else int(current_lane_id)
    )
    follow_route_lane = _follow_route_lane_for_decision(
        str(normalized_decision),
        explicit_follow_global_route_lane=follow_global_route_lane,
    )

    route_points_valid = (
        global_route_points
        if global_route_points is not None and len(global_route_points) >= 2
        else None
    )
    use_route_reference = (
        bool(follow_route_lane)
        or str(normalized_decision) in {"lane_change_left", "lane_change_right", "reroute"}
    )
    if route_points_valid is not None and bool(use_route_reference):
        # Reference samples should originate from the ego's current route
        # progress; the previous blue-dot position is only used to decide mode.
        anchor_wp = ego_wp
        if (
            int(route_alignment_lane_id) != int(current_lane_id)
            and normalized_decision in ("lane_change_left", "lane_change_right", "reroute")
        ):
            source_route_samples = _build_route_reference_samples_from_anchor(
                map_planner=map_planner,
                anchor_wp=anchor_wp,
                route_points=route_points_valid,
                horizon_steps=n,
                step_distance_m=sd,
                fallback_lane_id=int(current_lane_id),
                target_lane_id=int(current_lane_id),
                follow_route_lane=False,
            )
            target_route_samples = _build_route_reference_samples_from_anchor(
                map_planner=map_planner,
                anchor_wp=anchor_wp,
                route_points=route_points_valid,
                horizon_steps=n,
                step_distance_m=sd,
                fallback_lane_id=int(route_alignment_lane_id),
                target_lane_id=int(route_alignment_lane_id),
                follow_route_lane=bool(follow_route_lane),
            )
            if len(source_route_samples) > 0 and len(target_route_samples) > 0:
                blend_steps = max(2, int(math.ceil(12.0 / max(0.25, sd))))
                return _blend_reference_samples(
                    source_samples=source_route_samples,
                    target_samples=target_route_samples,
                    blend_steps=blend_steps,
                )
        route_samples = _build_route_reference_samples_from_anchor(
            map_planner=map_planner,
            anchor_wp=anchor_wp,
            route_points=route_points_valid,
            horizon_steps=n,
            step_distance_m=sd,
            fallback_lane_id=int(route_alignment_lane_id),
            target_lane_id=int(route_alignment_lane_id),
            follow_route_lane=bool(follow_route_lane),
        )
        if len(route_samples) > 0:
            return route_samples

    start_wp = _start_wp_for_decision(
        ego_wp, str(normalized_decision), int(target_lane_id),
        allow_junction_lane_snap=str(normalized_decision) in {"lane_change_left", "lane_change_right"},
    )
    walk_maneuver = str(next_macro_maneuver or "") if bool(is_intersection) else None
    source_samples = _build_forward_reference_samples(
        ego_wp,
        horizon_steps=n,
        step_distance_m=sd,
        route_points=None,
        fallback_lane_id=int(current_lane_id),
        maneuver=walk_maneuver,
    )
    target_samples = _build_forward_reference_samples(
        start_wp,
        horizon_steps=n,
        step_distance_m=sd,
        route_points=None,
        fallback_lane_id=int(target_lane_id),
        maneuver=walk_maneuver,
    )

    if (
        int(target_lane_id) != int(current_lane_id)
        and normalized_decision in ("lane_change_left", "lane_change_right", "reroute")
    ):
        blend_steps = max(2, int(math.ceil(12.0 / max(0.25, sd))))
        return _blend_reference_samples(
            source_samples=source_samples,
            target_samples=target_samples,
            blend_steps=blend_steps,
        )

    return target_samples if len(target_samples) > 0 else source_samples


def compute_ego_lane_offset(
    map_planner: Any,
    ego_pose: Mapping[str, object],
) -> Dict[str, float]:
    """
    Lateral offset and heading error relative to lane centre.

    Returns ``{"lateral_offset_m", "heading_error_rad", "lane_id"}``.
    """
    ego_position = _pose_xyz(ego_pose)
    ego_wp = map_planner.get_waypoint(ego_position)
    if ego_wp is None:
        return {"lateral_offset_m": 0.0, "heading_error_rad": 0.0,
                "lane_id": 0}

    wp_x_m, wp_y_m = _waypoint_xy(ego_wp)
    dx = float(ego_position["x"]) - wp_x_m
    dy = float(ego_position["y"]) - wp_y_m
    lane_yaw = _waypoint_heading_rad(ego_wp)
    lateral = -math.sin(lane_yaw) * dx + math.cos(lane_yaw) * dy

    ego_yaw = float(ego_pose.get("heading_rad", 0.0))
    heading_error = math.atan2(
        math.sin(ego_yaw - lane_yaw),
        math.cos(ego_yaw - lane_yaw),
    )

    return {
        "lateral_offset_m": float(lateral),
        "heading_error_rad": float(heading_error),
        "lane_id": _internal_lane_id(ego_wp),
    }
