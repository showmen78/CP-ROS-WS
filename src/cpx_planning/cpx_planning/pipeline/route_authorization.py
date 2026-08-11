"""Route-level authorization for lane-change decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Mapping, Optional, Sequence


class RouteManeuver(Enum):
    LANE_FOLLOW = "lane_follow"
    GO_STRAIGHT = "go_straight"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LaneChangeAuthorization:
    allowed: bool
    direction: Optional[str]
    reason: str
    required_by_route: bool
    distance_to_maneuver_m: Optional[float]
    target_lane_id: int
    maneuver: str

    def as_debug_fields(self) -> Mapping[str, object]:
        return {
            "lane_change_authorized": bool(self.allowed),
            "lane_change_authorization_direction": "" if self.direction is None else str(self.direction),
            "lane_change_authorization_reason": str(self.reason),
            "lane_change_required_by_route": bool(self.required_by_route),
            "lane_change_distance_to_maneuver_m": (
                "" if self.distance_to_maneuver_m is None else float(self.distance_to_maneuver_m)
            ),
            "lane_change_authorized_target_lane_id": int(self.target_lane_id),
            "route_maneuver_normalized": str(self.maneuver),
        }


def normalize_route_maneuver(value: object) -> RouteManeuver:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "none", "unknown"}:
        return RouteManeuver.UNKNOWN
    if text in {"lanefollow", "lane_follow", "follow", "keep_lane"}:
        return RouteManeuver.LANE_FOLLOW
    if text in {"continue_straight", "straight", "go_straight", "through"}:
        return RouteManeuver.GO_STRAIGHT
    if text in {"left", "turn_left", "left_turn", "leftturn"}:
        return RouteManeuver.TURN_LEFT
    if text in {"right", "turn_right", "right_turn", "rightturn"}:
        return RouteManeuver.TURN_RIGHT
    if text in {"lane_change_left", "change_left", "left_lane_change"}:
        return RouteManeuver.LANE_CHANGE_LEFT
    if text in {"lane_change_right", "change_right", "right_lane_change"}:
        return RouteManeuver.LANE_CHANGE_RIGHT
    return RouteManeuver.UNKNOWN


def authorize_route_lane_change(
    *,
    route_lane_change_allowed: bool,
    current_lane_id: int,
    route_required_lane_id: int,
    next_macro_maneuver: object,
    current_road_option: object,
    remaining_distance_m: Optional[float],
    available_lane_ids: Sequence[int],
    lane_safety_scores: Mapping[int, float],
    lane_prediction_risks: Mapping[int, Mapping[str, object]],
    preparation_start_distance_m: float,
    latest_start_distance_m: float,
    target_safety_threshold: float,
    require_adjacent: bool = True,
) -> LaneChangeAuthorization:
    if not bool(route_lane_change_allowed):
        return _denied("route_lane_change_not_allowed", next_macro_maneuver, remaining_distance_m, current_lane_id)

    maneuver = normalize_route_maneuver(next_macro_maneuver)
    current_option = normalize_route_maneuver(current_road_option)
    target_lane_id = int(route_required_lane_id or 0)
    current_lane_id = int(current_lane_id or 0)

    if maneuver in {RouteManeuver.GO_STRAIGHT, RouteManeuver.LANE_FOLLOW, RouteManeuver.UNKNOWN}:
        return _denied(
            "route_maneuver_does_not_require_lane_change",
            maneuver,
            remaining_distance_m,
            current_lane_id,
        )
    if current_option in {RouteManeuver.TURN_LEFT, RouteManeuver.TURN_RIGHT}:
        return _denied("already_in_turn_connector", maneuver, remaining_distance_m, current_lane_id)
    if target_lane_id == 0:
        return _denied("missing_required_lane_id", maneuver, remaining_distance_m, current_lane_id)
    if target_lane_id == current_lane_id:
        return LaneChangeAuthorization(
            allowed=False,
            direction=None,
            reason="already_in_required_lane",
            required_by_route=False,
            distance_to_maneuver_m=_finite_or_none(remaining_distance_m),
            target_lane_id=int(target_lane_id),
            maneuver=str(maneuver.value),
        )

    available = set()
    for lane_id in list(available_lane_ids or []):
        try:
            normalized_lane_id = int(lane_id)
        except Exception:
            continue
        if normalized_lane_id != 0:
            available.add(normalized_lane_id)
    if target_lane_id not in available:
        return _denied("required_lane_not_available", maneuver, remaining_distance_m, target_lane_id)
    lane_delta = int(target_lane_id) - int(current_lane_id)
    if bool(require_adjacent) and abs(int(lane_delta)) != 1:
        return _denied("required_lane_not_adjacent", maneuver, remaining_distance_m, target_lane_id)

    expected_direction = _direction_for_delta(lane_delta)
    if maneuver == RouteManeuver.TURN_LEFT and expected_direction != "left":
        return _denied("required_lane_direction_mismatch_left_turn", maneuver, remaining_distance_m, target_lane_id)
    if maneuver == RouteManeuver.TURN_RIGHT and expected_direction != "right":
        return _denied("required_lane_direction_mismatch_right_turn", maneuver, remaining_distance_m, target_lane_id)
    if maneuver == RouteManeuver.LANE_CHANGE_LEFT and expected_direction != "left":
        return _denied("required_lane_direction_mismatch_left_change", maneuver, remaining_distance_m, target_lane_id)
    if maneuver == RouteManeuver.LANE_CHANGE_RIGHT and expected_direction != "right":
        return _denied("required_lane_direction_mismatch_right_change", maneuver, remaining_distance_m, target_lane_id)

    distance = _finite_or_none(remaining_distance_m)
    explicit_lane_change = maneuver in {
        RouteManeuver.LANE_CHANGE_LEFT,
        RouteManeuver.LANE_CHANGE_RIGHT,
    }
    if distance is not None and not explicit_lane_change:
        if float(distance) > float(preparation_start_distance_m):
            return _denied("maneuver_too_far_for_lane_change", maneuver, distance, target_lane_id)
        if float(distance) < float(latest_start_distance_m):
            return _denied("maneuver_too_close_for_lane_change", maneuver, distance, target_lane_id)

    safety = float(lane_safety_scores.get(int(target_lane_id), 0.0))
    if safety < float(target_safety_threshold):
        return _denied("target_lane_safety_below_threshold", maneuver, distance, target_lane_id)
    risk = dict(lane_prediction_risks.get(int(target_lane_id), {}) or {})
    if bool(risk.get("risk", False)):
        return _denied("target_lane_prediction_risk", maneuver, distance, target_lane_id)

    return LaneChangeAuthorization(
        allowed=True,
        direction=str(expected_direction),
        reason="route_lane_change_authorized",
        required_by_route=True,
        distance_to_maneuver_m=distance,
        target_lane_id=int(target_lane_id),
        maneuver=str(maneuver.value),
    )


def _direction_for_delta(lane_delta: int) -> str:
    # The project treats increasing canonical lane id as a leftward move in the
    # OpenCDA debug traces used for the integration scenarios.
    return "left" if int(lane_delta) > 0 else "right"


def _denied(
    reason: str,
    maneuver: object,
    distance: Optional[float],
    target_lane_id: int,
) -> LaneChangeAuthorization:
    normalized = normalize_route_maneuver(maneuver)
    return LaneChangeAuthorization(
        allowed=False,
        direction=None,
        reason=str(reason),
        required_by_route=False,
        distance_to_maneuver_m=_finite_or_none(distance),
        target_lane_id=int(target_lane_id or 0),
        maneuver=str(normalized.value),
    )


def _finite_or_none(value: object) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return float(number)
