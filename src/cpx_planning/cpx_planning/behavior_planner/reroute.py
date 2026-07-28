"""Lane-closure rerouting through the custom global planner."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

from cpx_planning.utility.cp_messages import(
    CP_MESSAGE_PATH,
    control_messages,
    ensure_cp_message_file_exists,
    lane_closure_messages,
    load_control_messages,
    load_cp_message_payload,
    load_cp_messages,
    load_lane_closure_messages,
    pop_lane_closure_messages,
    remove_cp_messages_by_id,
    reset_cp_message_payload,
    write_cp_message_payload,
    write_cp_messages,
)
def _message_position(message: Mapping[str, object]) -> Dict[str, float] | None:
    raw = message.get("position", None)
    try:
        if isinstance(raw, Mapping) and "x" in raw and "y" in raw:
            return {
                "x": float(raw["x"]),
                "y": float(raw["y"]),
                "z": float(raw.get("z", 0.0)),
            }
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
            return {
                "x": float(raw[0]),
                "y": float(raw[1]),
                "z": float(raw[2]) if len(raw) >= 3 else 0.0,
            }
    except (TypeError, ValueError):
        return None
    return None


def _resolve_blocked_ad_lane_id(
    message: Mapping[str, object],
    global_planner,
) -> int | None:
    position = _message_position(message)
    if position is not None:
        if hasattr(global_planner, "block_lane_at_position"):
            try:
                return global_planner.block_lane_at_position(position)
            except Exception:
                pass
        waypoint = global_planner.get_waypoint(position)
        if waypoint is None:
            return None
        ad_lane_id = getattr(waypoint, "ad_lane_id", None)
        if ad_lane_id is not None:
            return int(ad_lane_id)
        return int(getattr(waypoint, "lane_id", 0) or 0)

    raw_ad_lane_id = message.get("ad_lane_id", None)
    if raw_ad_lane_id is None:
        return None
    try:
        ad_lane_id = int(raw_ad_lane_id)
        core = getattr(global_planner, "core", None)
        if core is not None:
            core.get_lane_centerline(ad_lane_id)
        return ad_lane_id
    except Exception:
        return None


def reroute_from_lane_closure_messages(
    *,
    messages: Sequence[Mapping[str, object]],
    global_planner,
    ego_position: Mapping[str, object] | Sequence[object],
    goal_position: Mapping[str, object] | Sequence[object],
    current_route_points: Sequence[Sequence[float]],
) -> Dict[str, object]:
    del current_route_points
    handled_ids: List[str] = []
    blocked_ad_lane_ids: List[int] = []

    for raw_message in list(messages or []):
        if not isinstance(raw_message, Mapping):
            continue
        message = dict(raw_message)
        message_id = str(message.get("id", "") or "").strip()
        if not message_id or str(message.get("type", "")).strip().lower() != "lane_closure":
            continue
        ad_lane_id = _resolve_blocked_ad_lane_id(message, global_planner)
        if ad_lane_id is None:
            continue
        block_ad_lane = getattr(global_planner, "block_ad_lane_id", None)
        if callable(block_ad_lane):
            try:
                block_ad_lane(ad_lane_id)
            except Exception:
                continue
        if ad_lane_id not in blocked_ad_lane_ids:
            blocked_ad_lane_ids.append(ad_lane_id)
        if message_id not in handled_ids:
            handled_ids.append(message_id)

    if not blocked_ad_lane_ids:
        return {
            "route_summary": None,
            "route_points": [],
            "handled_message_ids": [],
            "blocked_ad_lane_ids": [],
            "debug_reason": "No lane-closure message resolved to an AD lane.",
        }

    route_summary = global_planner.trace_route(
        ego_position,
        goal_position,
        replace_stored_route=True,
    )
    if not route_summary.route_found:
        return {
            "route_summary": None,
            "route_points": [],
            "handled_message_ids": [],
            "blocked_ad_lane_ids": blocked_ad_lane_ids,
            "debug_reason": route_summary.debug_reason,
        }

    return {
        "route_summary": route_summary,
        "route_points": [list(point) for point in route_summary.route_waypoints],
        "handled_message_ids": handled_ids,
        "blocked_ad_lane_ids": blocked_ad_lane_ids,
        "debug_reason": "",
    }
