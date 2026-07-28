"""CARLA-independent local-goal and route-reference helpers."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

from cpx_planning.utility.global_planner import(
    INVALID_LANE_ID,
    canonical_lane_id_for_waypoint,
    canonical_lane_waypoint_for_lane_id,
    world_heading_rad,
)


def _euclidean_distance(first_point: Sequence[float], second_point: Sequence[float]) -> float:
    """Return Euclidean distance on Python 3.7 and newer."""
    return math.sqrt(
        sum((float(first) - float(second)) ** 2 for first, second in zip(first_point, second_point))
    )


def _point_xy(waypoint) -> tuple[float, float]:
    position = getattr(waypoint, "position", None)
    if not isinstance(position, Mapping):
        raise AttributeError("Custom waypoint does not contain a position.")
    return float(position["x"]), float(position["y"])


def _distance_xy(first: Sequence[float], second: Sequence[float]) -> float:
    return float(math.hypot(float(first[0]) - float(second[0]), float(first[1]) - float(second[1])))


def _wrap_angle_rad(angle_rad: float) -> float:
    return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))


def _select_forward_candidate(current_waypoint, candidates, target_lane_id: int):
    if not candidates:
        return None
    current_heading = float(world_heading_rad(current_waypoint) or 0.0)

    def score(candidate) -> tuple[int, float]:
        candidate_lane_id = int(canonical_lane_id_for_waypoint(candidate))
        candidate_heading = float(world_heading_rad(candidate) or current_heading)
        return (
            0 if candidate_lane_id == int(target_lane_id) else 1,
            abs(_wrap_angle_rad(candidate_heading - current_heading)),
        )

    return min(candidates, key=score)


def _sample_custom_lane_points(
    *,
    start_waypoint,
    target_lane_id: int,
    spacing_m: float,
    sample_count: int,
) -> List[tuple[float, float]]:
    current_waypoint = start_waypoint
    points: List[tuple[float, float]] = []
    for _ in range(max(0, int(sample_count))):
        candidates = list(current_waypoint.next(float(spacing_m)) or [])
        next_waypoint = _select_forward_candidate(
            current_waypoint,
            candidates,
            int(target_lane_id),
        )
        if next_waypoint is None:
            break
        next_point = _point_xy(next_waypoint)
        if points and _distance_xy(points[-1], next_point) <= 1.0e-6:
            break
        points.append(next_point)
        current_waypoint = next_waypoint
    return points


def _curvature_from_three_points(
    p1: Sequence[float],
    p2: Sequence[float],
    p3: Sequence[float],
) -> float:
    side_a = _distance_xy(p1, p2)
    side_b = _distance_xy(p2, p3)
    side_c = _distance_xy(p1, p3)
    if min(side_a, side_b, side_c) <= 1.0e-9:
        return 0.0
    twice_area = abs(
        (float(p2[0]) - float(p1[0])) * (float(p3[1]) - float(p1[1]))
        - (float(p3[0]) - float(p1[0])) * (float(p2[1]) - float(p1[1]))
    )
    return float(2.0 * twice_area / (side_a * side_b * side_c))


def compute_lane_lookahead_distance(
    ego_state: Sequence[float],
    map_planner,
    target_lane_id: int,
    local_goal_cfg: Mapping[str, object],
    ego_z_m: float = 0.0,
) -> float | None:
    """Compute speed/curvature lookahead using custom OpenDRIVE waypoints."""
    if not bool(local_goal_cfg.get("dynamic_lookahead_enabled", True)):
        return None

    min_distance_m = max(
        0.0,
        float(local_goal_cfg.get("dynamic_lookahead_min_distance_m", 20.0)),
    )
    max_distance_m = max(
        min_distance_m,
        float(local_goal_cfg.get("dynamic_lookahead_max_distance_m", min_distance_m)),
    )
    speed_gain = float(local_goal_cfg.get("dynamic_lookahead_speed_gain", 3.0))
    curvature_gain = float(local_goal_cfg.get("dynamic_lookahead_curvature_gain", 20.0))
    sample_spacing_m = max(
        0.1,
        float(local_goal_cfg.get("dynamic_lookahead_curvature_sample_spacing_m", 5.0)),
    )

    ego_waypoint = map_planner.get_waypoint(
        {"x": float(ego_state[0]), "y": float(ego_state[1]), "z": float(ego_z_m)}
    )
    if ego_waypoint is None:
        return float(min_distance_m)
    lane_waypoint = canonical_lane_waypoint_for_lane_id(
        ego_waypoint,
        int(target_lane_id),
    )
    if lane_waypoint is None:
        lane_waypoint = ego_waypoint

    samples = _sample_custom_lane_points(
        start_waypoint=lane_waypoint,
        target_lane_id=int(target_lane_id),
        spacing_m=float(sample_spacing_m),
        sample_count=3,
    )
    curvature = (
        _curvature_from_three_points(samples[0], samples[1], samples[2])
        if len(samples) >= 3
        else 0.0
    )
    raw_distance_m = (
        min_distance_m
        + speed_gain * max(0.0, float(ego_state[2]))
        - curvature_gain * abs(float(curvature))
    )
    return float(min(max_distance_m, max(min_distance_m, raw_distance_m)))


def _nearest_progress_along_route(
    route_points: Sequence[Sequence[float]],
    xy: Sequence[float],
) -> tuple[float, float]:
    if len(route_points) <= 1:
        return 0.0, 0.0
    total_progress_m = 0.0
    best_progress_m = 0.0
    best_distance_m = float("inf")
    px_m, py_m = float(xy[0]), float(xy[1])
    for start, end in zip(route_points, route_points[1:]):
        x0_m, y0_m = float(start[0]), float(start[1])
        x1_m, y1_m = float(end[0]), float(end[1])
        dx_m, dy_m = x1_m - x0_m, y1_m - y0_m
        segment_len_sq = dx_m * dx_m + dy_m * dy_m
        if segment_len_sq <= 1.0e-9:
            continue
        projection = min(
            1.0,
            max(0.0, ((px_m - x0_m) * dx_m + (py_m - y0_m) * dy_m) / segment_len_sq),
        )
        closest_x_m = x0_m + projection * dx_m
        closest_y_m = y0_m + projection * dy_m
        distance_m = math.hypot(px_m - closest_x_m, py_m - closest_y_m)
        segment_length_m = math.sqrt(segment_len_sq)
        if distance_m < best_distance_m:
            best_distance_m = distance_m
            best_progress_m = total_progress_m + projection * segment_length_m
        total_progress_m += segment_length_m
    return float(best_progress_m), float(total_progress_m)


def _sample_route_at_progress(
    route_points: Sequence[Sequence[float]],
    progress_m: float,
) -> List[float]:
    if not route_points:
        return [0.0, 0.0, 0.0]
    if len(route_points) == 1:
        return [float(route_points[0][0]), float(route_points[0][1]), 0.0]
    remaining_m = max(0.0, float(progress_m))
    for start, end in zip(route_points, route_points[1:]):
        x0_m, y0_m = float(start[0]), float(start[1])
        x1_m, y1_m = float(end[0]), float(end[1])
        segment_length_m = math.hypot(x1_m - x0_m, y1_m - y0_m)
        if segment_length_m <= 1.0e-9:
            continue
        if remaining_m <= segment_length_m:
            alpha = remaining_m / segment_length_m
            return [
                x0_m + alpha * (x1_m - x0_m),
                y0_m + alpha * (y1_m - y0_m),
                math.atan2(y1_m - y0_m, x1_m - x0_m),
            ]
        remaining_m -= segment_length_m
    previous, final = route_points[-2], route_points[-1]
    return [
        float(final[0]),
        float(final[1]),
        math.atan2(float(final[1]) - float(previous[1]), float(final[0]) - float(previous[0])),
    ]


def build_route_reference_samples(
    ego_snapshot: Mapping[str, object],
    route_points: Sequence[Sequence[float]],
    horizon_steps: int,
    step_distance_m: float,
    target_lane_id: int = int(INVALID_LANE_ID),
) -> List[Dict[str, float]]:
    """Sample an already-planned route polyline for MPC references."""
    if len(route_points) < 2 or int(horizon_steps) < 0:
        return []
    progress_m, route_length_m = _nearest_progress_along_route(
        route_points,
        [float(ego_snapshot.get("x", 0.0)), float(ego_snapshot.get("y", 0.0))],
    )
    fallback_heading = float(ego_snapshot.get("psi", 0.0))
    samples: List[Dict[str, float]] = []
    for index in range(int(horizon_steps) + 1):
        sample = _sample_route_at_progress(
            route_points,
            min(route_length_m, progress_m + index * float(step_distance_m)),
        )
        samples.append(
            {
                "x_ref_m": float(sample[0]),
                "y_ref_m": float(sample[1]),
                "heading_rad": float(sample[2]) if math.isfinite(float(sample[2])) else fallback_heading,
                "lane_id": int(target_lane_id),
            }
        )
    return samples


def compute_route_lookahead_distance(
    ego_state: Sequence[float],
    route_points: Sequence[Sequence[float]],
    local_goal_cfg: Mapping[str, object] | None = None,
) -> float | None:
    """Compute lookahead directly from route-polyline curvature."""
    if len(route_points) < 2:
        return None
    config = dict(local_goal_cfg or {})
    min_distance_m = float(config.get("dynamic_lookahead_min_distance_m", 20.0))
    if not bool(config.get("dynamic_lookahead_enabled", True)):
        return min_distance_m
    max_distance_m = max(
        min_distance_m,
        float(config.get("dynamic_lookahead_max_distance_m", 100.0)),
    )
    progress_m, _ = _nearest_progress_along_route(route_points, ego_state[:2])
    spacing_m = max(
        1.0,
        float(config.get("dynamic_lookahead_curvature_sample_spacing_m", 10.0)),
    )
    points = [
        _sample_route_at_progress(route_points, progress_m + spacing_m * offset)
        for offset in (1.0, 2.0, 3.0)
    ]
    curvature = _curvature_from_three_points(points[0], points[1], points[2])
    raw_distance_m = (
        min_distance_m
        + float(config.get("dynamic_lookahead_speed_gain", 3.0)) * max(0.0, float(ego_state[2]))
        - float(config.get("dynamic_lookahead_curvature_gain", 20.0)) * abs(curvature)
    )
    return float(min(max_distance_m, max(min_distance_m, raw_distance_m)))
