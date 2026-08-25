"""CARLA-to-CP traffic-light adapter.

This module converts CARLA traffic-light actors/state and primitive actor
positions into the cooperative-perception traffic-control message consumed by
the behavior planner. Road and stop geometry comes from the custom planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .traffic_light_stop import find_relevant_signal_context, find_stop_target_from_ego
from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint, world_heading_rad


_STOP_SIGNAL_STATES = {"red", "yellow", "amber"}


def _waypoint_xy(waypoint) -> Tuple[float, float]:
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


@dataclass
class CarlaTrafficLightCPResult:
    """Output of one CARLA traffic-light CP adaptation tick."""

    control_message: Optional[Dict[str, object]]
    signal_context: Optional[Dict[str, object]]
    stop_target: Optional[Dict[str, object]]
    ego_in_junction: bool


def normalize_cp_signal_state(signal_state: object) -> str:
    """Normalize CARLA/CP signal state to the planner-facing vocabulary."""

    state = str(signal_state or "unknown").strip().lower()
    if state == "amber":
        return "yellow"
    if state in {"red", "yellow", "green", "unknown", "off"}:
        return state
    return "unknown"


def signal_state_requires_stop(signal_state: object) -> bool:
    return normalize_cp_signal_state(signal_state) in _STOP_SIGNAL_STATES


def stop_target_forward_lateral_m(
    ego_pose: Mapping[str, object],
    stop_target: Mapping[str, object] | None,
) -> Tuple[float, float]:
    """Return stop target coordinates in ego-forward/lateral frame."""

    if not isinstance(stop_target, Mapping):
        return float("nan"), float("nan")
    try:
        dx_m = float(stop_target.get("x_m", stop_target.get("x", 0.0))) - float(
            ego_pose["x"]
        )
        dy_m = float(stop_target.get("y_m", stop_target.get("y", 0.0))) - float(
            ego_pose["y"]
        )
        yaw_rad = float(ego_pose.get("heading_rad", 0.0))
        forward_m = dx_m * math.cos(yaw_rad) + dy_m * math.sin(yaw_rad)
        lateral_m = -dx_m * math.sin(yaw_rad) + dy_m * math.cos(yaw_rad)
        return float(forward_m), float(lateral_m)
    except Exception:
        return float("nan"), float("nan")


def with_stop_target_distance_from_ego(
    *,
    ego_pose: Mapping[str, object],
    stop_target: Mapping[str, object],
) -> Dict[str, object]:
    updated = dict(stop_target)
    forward_m, lateral_m = stop_target_forward_lateral_m(
        ego_pose=ego_pose,
        stop_target=updated,
    )
    if math.isfinite(float(forward_m)):
        updated["distance_m"] = max(0.0, float(forward_m))
        updated["forward_m"] = float(forward_m)
    if math.isfinite(float(lateral_m)):
        updated["lateral_m"] = float(lateral_m)
    return updated


def stop_line_from_stop_target(
    stop_target: Mapping[str, object] | None,
) -> Optional[Dict[str, object]]:
    if not isinstance(stop_target, Mapping):
        return None
    try:
        return {
            "x_m": float(stop_target.get("x_m", stop_target.get("x", 0.0))),
            "y_m": float(stop_target.get("y_m", stop_target.get("y", 0.0))),
            "heading_rad": float(stop_target.get("heading_rad", 0.0)),
            "lane_id": int(stop_target.get("lane_id", 0)),
            "road_id": int(stop_target.get("road_id", 0)),
            "section_id": int(stop_target.get("section_id", 0)),
        }
    except Exception:
        return None


def _fallback_signal_stop_target_from_ego(
    *,
    map_planner: Any,
    ego_pose: Mapping[str, object],
    signal_context: Mapping[str, object] | None,
    search_distance_m: float,
    stop_buffer_m: float,
) -> Optional[Dict[str, object]]:
    """Synthesize a conservative stop line when CARLA exposes state but no stop line."""

    if not isinstance(signal_context, Mapping):
        return None
    if not signal_state_requires_stop(signal_context.get("signal_state", "")):
        return None

    signal_source = str(signal_context.get("signal_source", "")).strip().lower()
    try:
        signal_forward_m = float(signal_context.get("signal_forward_m", float("inf")))
    except Exception:
        signal_forward_m = float("inf")
    if math.isfinite(signal_forward_m) and signal_forward_m < -2.0:
        return None
    try:
        signal_lateral_m = float(signal_context.get("signal_lateral_m", 0.0))
    except Exception:
        signal_lateral_m = 0.0
    if math.isfinite(signal_lateral_m) and abs(signal_lateral_m) > 4.5:
        return None

    if signal_source == "actor_position_match":
        max_actor_fallback_forward_m = 35.0
        if math.isfinite(signal_forward_m) and signal_forward_m > max_actor_fallback_forward_m:
            return None

    try:
        signal_distance_m = float(signal_context.get("signal_distance_m", search_distance_m))
    except Exception:
        signal_distance_m = float(search_distance_m)
    if not math.isfinite(signal_distance_m):
        signal_distance_m = float(search_distance_m)
    if signal_distance_m <= 0.0:
        return None
    if signal_source == "actor_position_match" and signal_distance_m > 35.0:
        return None

    raw_stop_distance_m = signal_distance_m - max(0.0, float(stop_buffer_m))
    if raw_stop_distance_m <= 1.0:
        return None
    forward_distance_m = max(3.0, min(float(search_distance_m), float(raw_stop_distance_m)))
    if forward_distance_m <= 0.0:
        return None

    try:
        ego_waypoint = map_planner.get_waypoint(
            {
                "x": float(ego_pose["x"]),
                "y": float(ego_pose["y"]),
                "z": float(ego_pose.get("z", 0.0)),
            }
        )
        if ego_waypoint is None:
            return None
        next_waypoints = ego_waypoint.next(float(forward_distance_m))
        stop_waypoint = next_waypoints[0] if next_waypoints else ego_waypoint
        stop_x_m, stop_y_m = _waypoint_xy(stop_waypoint)
        return {
            "x_m": float(stop_x_m),
            "y_m": float(stop_y_m),
            "heading_rad": float(world_heading_rad(stop_waypoint) or 0.0),
            "lane_id": int(canonical_lane_id_for_waypoint(stop_waypoint)),
            "ad_lane_id": int(_waypoint_ad_lane_id(stop_waypoint)),
            "opendrive_lane_id": int(getattr(stop_waypoint, "lane_id", 0) or 0),
            "road_id": int(getattr(stop_waypoint, "road_id", 0) or 0),
            "section_id": int(getattr(stop_waypoint, "section_id", 0) or 0),
            "distance_m": float(forward_distance_m),
            "source": "fallback_signal_stop_target",
            "signal_distance_m": float(signal_distance_m),
        }
    except Exception:
        return None


def _signal_control_id(signal_context: Mapping[str, object]) -> str:
    actor_id = str(signal_context.get("signal_actor_id", "") or "").strip()
    actor_name = str(signal_context.get("signal_actor_name", "") or "").strip()
    signal_source = str(signal_context.get("signal_source", "carla") or "carla").strip()
    return actor_id or actor_name or signal_source or "unknown_signal"


def cp_traffic_control_from_signal_context(
    *,
    signal_context: Mapping[str, object] | None,
    stop_target: Mapping[str, object] | None,
    ego_pose: Mapping[str, object],
    sim_time_s: float,
    valid_for_s: float = 0.5,
    search_distance_m: float = 100.0,
) -> Optional[Dict[str, object]]:
    """Build a planner-facing CP traffic-light control message."""

    if not isinstance(signal_context, Mapping):
        return None
    signal_state = normalize_cp_signal_state(signal_context.get("signal_state", "unknown"))
    control_id = _signal_control_id(signal_context)
    valid_for_s = max(0.0, float(valid_for_s))
    stop_target_with_distance = (
        with_stop_target_distance_from_ego(
            ego_pose=ego_pose,
            stop_target=stop_target,
        )
        if isinstance(stop_target, Mapping)
        else None
    )
    stop_line = stop_line_from_stop_target(stop_target_with_distance)
    forward_m, lateral_m = stop_target_forward_lateral_m(
        ego_pose=ego_pose,
        stop_target=stop_target_with_distance,
    )
    ego_passed_stop_line = bool(math.isfinite(forward_m) and forward_m < -1.0)
    confidence = 1.0 if signal_state in {"red", "yellow", "green"} else 0.3

    try:
        signal_forward_m = float(signal_context.get("signal_forward_m", float("nan")))
    except Exception:
        signal_forward_m = float("nan")
    try:
        signal_lateral_m = float(signal_context.get("signal_lateral_m", float("nan")))
    except Exception:
        signal_lateral_m = float("nan")

    valid_range: Dict[str, object] = {
        "search_distance_m": float(search_distance_m),
        "forward_m": float(forward_m) if math.isfinite(forward_m) else signal_context.get("signal_forward_m", ""),
        "lateral_m": float(lateral_m) if math.isfinite(lateral_m) else signal_context.get("signal_lateral_m", ""),
        "signal_forward_m": float(signal_forward_m) if math.isfinite(signal_forward_m) else "",
        "signal_lateral_m": float(signal_lateral_m) if math.isfinite(signal_lateral_m) else "",
    }
    if isinstance(stop_line, Mapping):
        valid_range.update(
            {
                "road_id": stop_line.get("road_id", ""),
                "section_id": stop_line.get("section_id", ""),
                "lane_id": stop_line.get("lane_id", ""),
            }
        )

    message: Dict[str, object] = {
        "type": "traffic_light",
        "id": f"cp_tl:{control_id}",
        # Backward-compatible fields consumed by the current planner.
        "state": str(signal_state),
        "valid_until_s": float(sim_time_s) + valid_for_s,
        # Explicit CP traffic-control contract.
        "signal_state": str(signal_state),
        "control_id": str(control_id),
        "timestamp_s": float(sim_time_s),
        "ttl_s": float(valid_for_s),
        "confidence": float(confidence),
        "valid_range": valid_range,
        "ego_passed_stop_line": bool(ego_passed_stop_line),
        "source": "cp_adapter",
        "provider_source": str(signal_context.get("signal_source", "carla") or "carla"),
        "cp_adapter_generated": True,
        "signal_actor_id": str(signal_context.get("signal_actor_id", "")),
        "signal_actor_name": str(signal_context.get("signal_actor_name", "")),
        "signal_actor_raw_state": str(signal_context.get("signal_actor_raw_state", "")),
        "signal_distance_m": signal_context.get("signal_distance_m", ""),
        "signal_forward_m": signal_context.get("signal_forward_m", ""),
        "signal_lateral_m": signal_context.get("signal_lateral_m", ""),
        "signal_match_distance_m": signal_context.get("signal_match_distance_m", ""),
        "signal_match_rank": signal_context.get("signal_match_rank", ""),
    }
    if stop_line is not None:
        message["stop_line"] = dict(stop_line)
        message["stop_line_position"] = dict(stop_line)
        if isinstance(stop_target_with_distance, Mapping):
            message["distance_m"] = float(stop_target_with_distance.get("distance_m", 0.0))
            message["stop_target_forward_m"] = stop_target_with_distance.get("forward_m", "")
            message["stop_target_lateral_m"] = stop_target_with_distance.get("lateral_m", "")
    return message


def build_carla_traffic_light_cp_message(
    *,
    world: Any,
    map_planner: Any,
    ego_vehicle: Any,
    ego_pose: Mapping[str, object],
    global_route_points: Sequence[object],
    sim_time_s: float,
    search_distance_m: float,
    stop_buffer_m: float,
    valid_for_s: float,
    query_key: str = "ego",
    max_stop_waypoint_match_distance_m: float = 12.0,
    max_actor_position_match_distance_m: float | None = None,
    max_actor_position_lateral_m: float = 4.5,
) -> CarlaTrafficLightCPResult:
    """Read CARLA traffic-light data and expose it as a CP control message."""

    ego_waypoint = map_planner.get_waypoint(
        {
            "x": float(ego_pose["x"]),
            "y": float(ego_pose["y"]),
            "z": float(ego_pose.get("z", 0.0)),
        }
    )
    ego_in_junction = bool(getattr(ego_waypoint, "is_intersection", False))
    stop_target = None
    if not ego_in_junction:
        stop_target = find_stop_target_from_ego(
            map_planner=map_planner,
            ego_pose=ego_pose,
            global_route_points=global_route_points,
            search_distance_m=float(search_distance_m),
            query_key=str(query_key),
        )

    signal_context = find_relevant_signal_context(
        world=world,
        map_planner=map_planner,
        ego_vehicle=ego_vehicle,
        ego_pose=ego_pose,
        stop_target=stop_target,
        max_stop_waypoint_match_distance_m=float(max_stop_waypoint_match_distance_m),
        max_actor_position_match_distance_m=float(
            search_distance_m
            if max_actor_position_match_distance_m is None
            else max_actor_position_match_distance_m
        ),
        max_actor_position_lateral_m=float(max_actor_position_lateral_m),
    )

    if (
        not ego_in_junction
        and stop_target is None
        and signal_state_requires_stop(dict(signal_context or {}).get("signal_state", ""))
    ):
        stop_target = _fallback_signal_stop_target_from_ego(
            map_planner=map_planner,
            ego_pose=ego_pose,
            signal_context=signal_context,
            search_distance_m=float(search_distance_m),
            stop_buffer_m=float(stop_buffer_m),
        )
        if isinstance(stop_target, Mapping):
            signal_context = dict(signal_context or {})
            signal_context["fallback_stop_target"] = True
            signal_context["stop_target_source"] = "fallback_signal_stop_target"

    control_message = cp_traffic_control_from_signal_context(
        signal_context=signal_context,
        stop_target=stop_target,
        ego_pose=ego_pose,
        sim_time_s=float(sim_time_s),
        valid_for_s=float(valid_for_s),
        search_distance_m=float(search_distance_m),
    )

    return CarlaTrafficLightCPResult(
        control_message=dict(control_message) if isinstance(control_message, Mapping) else None,
        signal_context=dict(signal_context) if isinstance(signal_context, Mapping) else None,
        stop_target=dict(stop_target) if isinstance(stop_target, Mapping) else None,
        ego_in_junction=bool(ego_in_junction),
    )
