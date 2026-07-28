"""
Rule-based behavior planner.

Runs synchronously in the main loop with no network calls.

Decision pipeline
-----------------
NORMAL mode
1. **Lane follow means hold current lane** — the blue dot stays on the
   current selected lane until the planner explicitly decides to change
2. **Route-first objective** — try to be on the global route's optimal
   lane whenever it is safe
3. **Safety-score detour** — if the optimal lane becomes unsafe
   (score below threshold), move one lane at a time to the adjacent
   safest lane
4. **Return-to-route** — once the optimal lane is safe again, move back
   toward it one lane at a time

INTERSECTION mode
1. **Lane follow only** — do not start safety-driven lane changes
2. **Hold current lane** — the blue dot stays on the current lane while
   following the route geometry through the intersection

Shared logic
1. **Lane-change state machine** — IDLE / CHANGING_LEFT / CHANGING_RIGHT
2. **Completion check** — lane change finishes when ego is in the target
   lane, laterally centred, and heading-aligned
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from .contract import BehaviorCommand
from .reroute import (
    CP_MESSAGE_PATH,
    load_control_messages,
    load_lane_closure_messages,
    remove_cp_messages_by_id,
)
from .traffic_light_stop import normalize_signal_state, should_stop_for_signal

# --------------------------------------------------------------------- #
# Defaults                                                                #
# --------------------------------------------------------------------- #
DEFAULT_HYSTERESIS_DELTA = 0.15
DEFAULT_LATERAL_COMPLETE_M = 0.8
DEFAULT_HEADING_COMPLETE_RAD = math.radians(8.0)
DEFAULT_LANE_CHANGE_TARGET_SAFETY_THRESHOLD = 0.10
DEFAULT_LANE_CHANGE_ABORT_SAFETY_THRESHOLD = 0.50
DEFAULT_OPTIMAL_LANE_UNSAFE_THRESHOLD = 0.50
DEFAULT_INTERSECTION_REROUTE_STOP_DISTANCE_THRESHOLD_M = 30.0
DEFAULT_MOVING_OBSTACLE_SPEED_THRESHOLD_MPS = 0.5
DEFAULT_COOPERATIVE_MESSAGE_CHECK_FREQUENCY_HZ = 1.0
DEFAULT_STOP_SIGN_WAIT_DURATION_S = 3.0
DEFAULT_PREPARE_LANE_CHANGE_MIN_HOLD_S = 0.5
DEFAULT_EXECUTE_LANE_CHANGE_MIN_HOLD_S = 1.0
DEFAULT_LANE_KEEP_MIN_HOLD_S = 0.3
DEFAULT_LANE_CHANGE_SAFETY_ABORT_COMMIT_S = 2.0
DEFAULT_STOP_SIGN_COUNTDOWN_SPEED_THRESHOLD_MPS = 1.0
DEFAULT_STOP_SIGN_COUNTDOWN_DISTANCE_THRESHOLD_M = 2.0
DEFAULT_STOP_SIGN_COUNTDOWN_START_DISTANCE_THRESHOLD_M = 5.0
DEFAULT_EMERGENCY_BRAKE_FOLLOW_BUFFER_M = 1.0
DEFAULT_REROUTE_ROUTE_CLEARANCE_M = 3.0
DEFAULT_REROUTE_ROUTE_BEHIND_TOLERANCE_M = 5.0

# Behavior finite-state machine states.
_LANE_KEEP = "LANE_KEEP"
_PREPARE_LANE_CHANGE_LEFT = "PREPARE_LANE_CHANGE_LEFT"
_PREPARE_LANE_CHANGE_RIGHT = "PREPARE_LANE_CHANGE_RIGHT"
_EXECUTE_LANE_CHANGE_LEFT = "EXECUTE_LANE_CHANGE_LEFT"
_EXECUTE_LANE_CHANGE_RIGHT = "EXECUTE_LANE_CHANGE_RIGHT"
_ABORT_LANE_CHANGE = "ABORT_LANE_CHANGE"
_CANCEL_LANE_CHANGE = "CANCEL_LANE_CHANGE"
_REROUTE_STATE = "REROUTE"
_STOP_STATE = "STOP"
_YIELD_STATE = "YIELD"

# Compatibility aliases used internally by older helper names.
_IDLE = _LANE_KEEP
_CHANGING_LEFT = _EXECUTE_LANE_CHANGE_LEFT
_CHANGING_RIGHT = _EXECUTE_LANE_CHANGE_RIGHT

_DECISION_FOLLOW = "lane_follow"
_DECISION_CHANGE_LEFT = "lane_change_left"
_DECISION_CHANGE_RIGHT = "lane_change_right"
_DECISION_REROUTE = "reroute"
_DECISION_STOP_AT_INTERSECTION = "stop_at_intersection"
_DECISION_STOP_SIGN = "stop_sign"
_DECISION_EMERGENCY_BRAKE = "emergency_brake"

_MANEUVER_LEFT = "left"
_MANEUVER_RIGHT = "right"
_MANEUVER_STRAIGHT = "straight"


def _normalize_control_message_type(message_type: object) -> str:
    normalized_type = str(message_type or "").strip().lower()
    if normalized_type == "intersection":
        return "traffic_light"
    return normalized_type


def _optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    normalized_value = str(value or "").strip().lower()
    if normalized_value in {"true", "1", "yes", "y", "stop"}:
        return True
    if normalized_value in {"false", "0", "no", "n", "go"}:
        return False
    return None


def normalize_behavior_decision(decision: str | None) -> str:
    normalized_name = (
        str(decision or "")
        .strip()
        .upper()
        .replace(" ", "_")
    )
    if normalized_name in {"LANE_CHANGE_LEFT", "CHANGE_LEFT"}:
        return _DECISION_CHANGE_LEFT
    if normalized_name in {"LANE_CHANGE_RIGHT", "CHANGE_RIGHT"}:
        return _DECISION_CHANGE_RIGHT
    if normalized_name in {"REROUTE", "RE_ROUTE"}:
        return _DECISION_REROUTE
    if normalized_name in {
        "STOP",
        "STOP_AT_INTERSECTION",
        "STOP_INTERSECTION",
        "TRAFFIC_LIGHT_STOP",
    }:
        return _DECISION_STOP_AT_INTERSECTION
    if normalized_name in {"STOP_SIGN", "STOP_AT_STOP_SIGN"}:
        return _DECISION_STOP_SIGN
    if normalized_name in {
        "EMERGENCE_STOP",
        "EMERGENCY_STOP",
        "EMERGENCY_BRAKE",
        "EMERGENCY",
    }:
        return _DECISION_EMERGENCY_BRAKE
    if normalized_name in {"LANE_KEEP", "KEEP_LANE", "KEEP"}:
        return _DECISION_FOLLOW
    return _DECISION_FOLLOW


def is_fixed_stop_decision(decision: str | None) -> bool:
    return str(normalize_behavior_decision(decision)) in {
        _DECISION_STOP_AT_INTERSECTION,
        _DECISION_STOP_SIGN,
    }


def is_stop_decision(decision: str | None) -> bool:
    return bool(is_fixed_stop_decision(decision))


def is_emergency_brake_decision(decision: str | None) -> bool:
    return str(normalize_behavior_decision(decision)) == _DECISION_EMERGENCY_BRAKE


def is_emergence_stop_decision(decision: str | None) -> bool:
    return bool(is_emergency_brake_decision(decision))


def _behavior_command_invariant_violations(
    command: Mapping[str, object],
    *,
    ego_lane_id: int,
) -> list[dict[str, object]]:
    """Check the "Required invariant" rule from the architecture proposal's
    Behavior Layer section: stop and emergency-brake decisions must keep the
    ego lane as target.

    Note: a lane-change decision whose ``target_lane_id`` equals the ego lane
    is *not* flagged here. That state is legitimately reached when an
    in-progress lane change is reversed before the vehicle has physically
    left its source lane (see
    ``test_ongoing_lane_change_reverses_when_target_lane_becomes_worse_than_previous_lane``)
    -- the decision string still names the direction of travel, not "target
    differs from current lane".

    Returns structured records (``kind`` + human-readable ``message``) rather
    than plain strings so the caller can apply the matching correction, not
    just log the problem.
    """
    violations: list[dict[str, object]] = []
    decision = str(command.get("decision", ""))
    raw_target_lane_id = command.get("target_lane_id", None)
    try:
        target_lane_id = None if raw_target_lane_id is None else int(raw_target_lane_id)
    except Exception:
        target_lane_id = None
    if target_lane_id is None:
        return violations

    if (
        (is_fixed_stop_decision(decision) or is_emergency_brake_decision(decision))
        and int(target_lane_id) != int(ego_lane_id)
    ):
        violations.append({
            "kind": "stop_target_lane_mismatch",
            "message": (
                f"decision={decision!r} must target ego lane {int(ego_lane_id)}, "
                f"got target_lane_id={int(target_lane_id)}"
            ),
        })
    return violations


def normalize_macro_maneuver(next_macro_maneuver: str | None) -> str:
    normalized_name = (
        str(next_macro_maneuver or "")
        .strip()
        .upper()
        .replace(" ", "_")
    )
    if normalized_name in {"LEFT", "LEFT_TURN", "LEFTTURN"}:
        return _MANEUVER_LEFT
    if normalized_name in {"RIGHT", "RIGHT_TURN", "RIGHTTURN"}:
        return _MANEUVER_RIGHT
    return _MANEUVER_STRAIGHT


def intersection_route_follow_maneuver(
    mode: str,
    next_macro_maneuver: str | None,
    decision: str,
    target_lane_id: int,
    available_lane_ids: Sequence[int],
    current_road_option: str | None = None,
) -> str:
    mode_name = str(mode or "NORMAL").strip().upper()
    del decision
    del target_lane_id
    del available_lane_ids

    maneuver_name = normalize_macro_maneuver(next_macro_maneuver)
    normalized_current_option = normalize_macro_maneuver(current_road_option)

    if mode_name != "INTERSECTION":
        return str(maneuver_name)
    if normalized_current_option in {_MANEUVER_LEFT, _MANEUVER_RIGHT}:
        return _DECISION_FOLLOW
    return str(maneuver_name)


def evaluate_intersection_obstacle_response(
    mode: str,
    front_obstacle_speed_mps: float | None,
    original_max_velocity_mps: float,
    moving_obstacle_speed_threshold_mps: float = DEFAULT_MOVING_OBSTACLE_SPEED_THRESHOLD_MPS,
    route_lane_safety_score: float | None = None,
    static_obstacle_replan_lane_safety_threshold: float = 0.5,
) -> Dict[str, Any]:
    response = {
        "speed_cap_mps": float(max(0.0, float(original_max_velocity_mps))),
        "follow_moving_obstacle": False,
        "request_static_obstacle_replan": False,
    }

    if str(mode or "NORMAL").strip().upper() != "INTERSECTION":
        return response
    if front_obstacle_speed_mps is None:
        return response

    obstacle_speed_mps = max(0.0, float(front_obstacle_speed_mps))
    if obstacle_speed_mps > max(0.0, float(moving_obstacle_speed_threshold_mps)):
        response["speed_cap_mps"] = min(
            float(response["speed_cap_mps"]),
            float(obstacle_speed_mps),
        )
        response["follow_moving_obstacle"] = True
        return response

    if (
        route_lane_safety_score is not None
        and float(route_lane_safety_score) < float(static_obstacle_replan_lane_safety_threshold)
    ):
        response["request_static_obstacle_replan"] = True
    return response


def _coerce_xy(point_xy: object) -> tuple[float, float] | None:
    if not isinstance(point_xy, Sequence) or isinstance(point_xy, (str, bytes, bytearray)) or len(point_xy) < 2:
        return None
    try:
        return float(point_xy[0]), float(point_xy[1])
    except Exception:
        return None


def _normalized_route_points_xy(
    route_points: Sequence[Sequence[object]] | None,
) -> list[tuple[float, float]]:
    normalized_points: list[tuple[float, float]] = []
    for route_point in list(route_points or []):
        if not isinstance(route_point, Sequence) or len(route_point) < 2:
            continue
        try:
            normalized_points.append((float(route_point[0]), float(route_point[1])))
        except Exception:
            continue
    return normalized_points


def _route_cumulative_distances(
    route_points: Sequence[tuple[float, float]],
) -> list[float]:
    if len(route_points) == 0:
        return []
    cumulative_distances = [0.0]
    for index in range(1, len(route_points)):
        cumulative_distances.append(
            float(cumulative_distances[-1])
            + math.hypot(
                float(route_points[index][0]) - float(route_points[index - 1][0]),
                float(route_points[index][1]) - float(route_points[index - 1][1]),
            )
        )
    return cumulative_distances


def _project_point_to_route_arc_m(
    *,
    point_xy: Sequence[float],
    route_points: Sequence[tuple[float, float]],
    cumulative_distances: Sequence[float],
) -> float | None:
    if len(route_points) == 0:
        return None
    if len(route_points) == 1:
        return 0.0

    px_m = float(point_xy[0])
    py_m = float(point_xy[1])
    best_distance_sq = float("inf")
    best_arc_m = 0.0

    for index in range(len(route_points) - 1):
        ax_m = float(route_points[index][0])
        ay_m = float(route_points[index][1])
        bx_m = float(route_points[index + 1][0])
        by_m = float(route_points[index + 1][1])
        dx_m = float(bx_m) - float(ax_m)
        dy_m = float(by_m) - float(ay_m)
        segment_length_sq = float(dx_m) * float(dx_m) + float(dy_m) * float(dy_m)
        if float(segment_length_sq) <= 1.0e-9:
            projection = 0.0
        else:
            projection = (
                (float(px_m) - float(ax_m)) * float(dx_m)
                + (float(py_m) - float(ay_m)) * float(dy_m)
            ) / float(segment_length_sq)
            projection = max(0.0, min(1.0, float(projection)))
        closest_x_m = float(ax_m) + float(projection) * float(dx_m)
        closest_y_m = float(ay_m) + float(projection) * float(dy_m)
        distance_sq = (
            (float(px_m) - float(closest_x_m)) * (float(px_m) - float(closest_x_m))
            + (float(py_m) - float(closest_y_m)) * (float(py_m) - float(closest_y_m))
        )
        if float(distance_sq) >= float(best_distance_sq):
            continue
        segment_length_m = math.sqrt(float(segment_length_sq))
        best_distance_sq = float(distance_sq)
        best_arc_m = float(cumulative_distances[index]) + float(projection) * float(segment_length_m)

    return float(best_arc_m)


def _distance_point_to_segment_m(
    point_xy: Sequence[float],
    segment_start_xy: Sequence[float],
    segment_end_xy: Sequence[float],
) -> float:
    px_m = float(point_xy[0])
    py_m = float(point_xy[1])
    ax_m = float(segment_start_xy[0])
    ay_m = float(segment_start_xy[1])
    bx_m = float(segment_end_xy[0])
    by_m = float(segment_end_xy[1])
    dx_m = float(bx_m) - float(ax_m)
    dy_m = float(by_m) - float(ay_m)
    segment_len_sq = dx_m * dx_m + dy_m * dy_m
    if float(segment_len_sq) <= 1.0e-9:
        return float(math.hypot(float(px_m) - float(ax_m), float(py_m) - float(ay_m)))
    projection = (
        (float(px_m) - float(ax_m)) * float(dx_m)
        + (float(py_m) - float(ay_m)) * float(dy_m)
    ) / float(segment_len_sq)
    projection = max(0.0, min(1.0, float(projection)))
    closest_x_m = float(ax_m) + float(projection) * float(dx_m)
    closest_y_m = float(ay_m) + float(projection) * float(dy_m)
    return float(math.hypot(float(px_m) - float(closest_x_m), float(py_m) - float(closest_y_m)))


def _route_min_distance_to_point_m(
    route_points: Sequence[tuple[float, float]],
    point_xy: Sequence[float],
) -> float:
    if len(route_points) == 0:
        return float("inf")
    if len(route_points) == 1:
        return float(
            math.hypot(
                float(point_xy[0]) - float(route_points[0][0]),
                float(point_xy[1]) - float(route_points[0][1]),
            )
        )
    return min(
        _distance_point_to_segment_m(
            point_xy=point_xy,
            segment_start_xy=segment_start_xy,
            segment_end_xy=segment_end_xy,
        )
        for segment_start_xy, segment_end_xy in zip(route_points[:-1], route_points[1:])
    )


class RuleBasedBehaviorPlanner:
    """Stateful rule-based lane-selection planner."""

    def __init__(
        self,
        hysteresis_delta: float = DEFAULT_HYSTERESIS_DELTA,
        lateral_complete_m: float = DEFAULT_LATERAL_COMPLETE_M,
        heading_complete_rad: float = DEFAULT_HEADING_COMPLETE_RAD,
        lane_change_target_safety_threshold: float | None = None,
        lane_change_abort_safety_threshold: float = DEFAULT_LANE_CHANGE_ABORT_SAFETY_THRESHOLD,
        optimal_lane_unsafe_threshold: float = DEFAULT_OPTIMAL_LANE_UNSAFE_THRESHOLD,
        intersection_lane_change_safety_threshold: float | None = None,
        intersection_reroute_stop_distance_threshold_m: float = DEFAULT_INTERSECTION_REROUTE_STOP_DISTANCE_THRESHOLD_M,
        cp_message_path: str | None = None,
        cooperative_message_check_frequency_hz: float = DEFAULT_COOPERATIVE_MESSAGE_CHECK_FREQUENCY_HZ,
        stop_sign_wait_duration_s: float = DEFAULT_STOP_SIGN_WAIT_DURATION_S,
        emergency_brake_follow_buffer_m: float = DEFAULT_EMERGENCY_BRAKE_FOLLOW_BUFFER_M,
        prepare_lane_change_min_hold_s: float = DEFAULT_PREPARE_LANE_CHANGE_MIN_HOLD_S,
        execute_lane_change_min_hold_s: float = DEFAULT_EXECUTE_LANE_CHANGE_MIN_HOLD_S,
        lane_keep_min_hold_s: float = DEFAULT_LANE_KEEP_MIN_HOLD_S,
        lane_change_safety_abort_commit_s: float = DEFAULT_LANE_CHANGE_SAFETY_ABORT_COMMIT_S,
        candidate_route_deviation_weight: float = 0.15,
        candidate_lane_change_weight: float = 0.20,
    ) -> None:
        self._hysteresis = float(hysteresis_delta)
        self._lateral_complete = float(lateral_complete_m)
        self._heading_complete = float(heading_complete_rad)
        resolved_target_lane_safety_threshold = (
            lane_change_target_safety_threshold
            if lane_change_target_safety_threshold is not None
            else intersection_lane_change_safety_threshold
        )
        if resolved_target_lane_safety_threshold is None:
            resolved_target_lane_safety_threshold = (
                DEFAULT_LANE_CHANGE_TARGET_SAFETY_THRESHOLD
            )
        self._target_lane_safety_threshold = float(
            resolved_target_lane_safety_threshold
        )
        self._lane_change_abort_safety_threshold = float(
            max(0.0, min(1.0, float(lane_change_abort_safety_threshold)))
        )
        self._optimal_lane_unsafe_threshold = float(
            max(0.0, min(1.0, float(optimal_lane_unsafe_threshold)))
        )
        self._intersection_reroute_stop_distance_threshold_m = max(
            0.0,
            float(intersection_reroute_stop_distance_threshold_m),
        )
        self._cp_message_path = None if cp_message_path is None else str(cp_message_path)
        self._cp_message_check_period_s = (
            0.0
            if float(cooperative_message_check_frequency_hz) <= 0.0
            else 1.0 / float(cooperative_message_check_frequency_hz)
        )
        self._stop_sign_wait_duration_s = max(0.0, float(stop_sign_wait_duration_s))
        self._emergency_brake_follow_buffer_m = max(
            0.0,
            float(emergency_brake_follow_buffer_m),
        )
        self._prepare_lane_change_min_hold_s = max(0.0, float(prepare_lane_change_min_hold_s))
        self._execute_lane_change_min_hold_s = max(0.0, float(execute_lane_change_min_hold_s))
        self._lane_keep_min_hold_s = max(0.0, float(lane_keep_min_hold_s))
        self._lane_change_safety_abort_commit_s = max(
            0.0,
            float(lane_change_safety_abort_commit_s),
        )
        self._candidate_route_deviation_weight = max(0.0, float(candidate_route_deviation_weight))
        self._candidate_lane_change_weight = max(0.0, float(candidate_lane_change_weight))
        self._last_cp_message_check_time_s: float | None = None
        self._cached_lane_closure_messages: list[dict] = []
        self._cached_control_messages: list[dict] = []
        self._processed_cp_message_ids: set[str] = set()
        self._acknowledged_reroute_ids: set[str] = set()
        self._pending_reroute_messages: list[dict] = []
        self._completed_stop_sign_message_ids: set[str] = set()

        # Persistent state across calls
        self._lc_state: str = _IDLE
        self._target_lane_id: int | None = None
        self._source_lane_id: int | None = None
        self._selected_lane_id: int | None = None
        self._last_mode: str = "NORMAL"
        self._stop: bool = False
        self._stopping_point: Dict[str, Any] | None = None
        self._active_control_message_id: str | None = None
        self._active_control_message_type: str | None = None
        self._stop_sign_wait_started_wall_time_s: float | None = None
        self._state_enter_time_s: float | None = None
        self._last_logged_lc_state: str = str(self._lc_state)
        self._current_update_time_s: float | None = None
        self._current_ego_lane_id: int = 0
        self._pending_transition_reason: str = "init"
        self._transition_events: list[dict] = []
        self._current_candidate_evaluation: Dict[str, Any] | None = None

    # ----------------------------------------------------------------- #
    # Public                                                              #
    # ----------------------------------------------------------------- #
    def update(
        self,
        lane_safety_scores: Dict[int, float],
        ego_lane_id: int,
        selected_lane_id: int | None = None,
        ego_lateral_offset_m: float = 0.0,
        ego_heading_error_rad: float = 0.0,
        mode: str = "NORMAL",
        route_optimal_lane_id: int | None = None,
        next_macro_maneuver: str | None = None,
        front_obstacle_distance_by_lane: Mapping[int, float] | None = None,
        current_time_s: float | None = None,
        wall_time_s: float | None = None,
        traffic_signal_state: str | None = None,
        traffic_stop_target: Mapping[str, object] | None = None,
        traffic_signal_context: Mapping[str, object] | None = None,
        ego_speed_mps: float = 0.0,
        ego_max_deceleration_mps2: float = 0.0,
        ego_in_junction: bool = False,
        ego_position_xy: Sequence[float] | None = None,
        global_route_points: Sequence[Sequence[float]] | None = None,
        nearest_front_obstacles_by_lane: Mapping[int, Mapping[str, object]] | None = None,
        lane_prediction_risks: Mapping[int, Mapping[str, object]] | None = None,
        preferred_target_lane_id: int | None = None,
    ) -> BehaviorCommand:
        """
        Run one planning cycle.

        Parameters
        ----------
        lane_safety_scores    : {lane_id: score}  score in [0, 1]
        ego_lane_id           : current project lane id
        ego_lateral_offset_m  : signed lateral offset from lane centre
        ego_heading_error_rad : heading error w.r.t. lane direction

        mode                  : "NORMAL" | "INTERSECTION"
        route_optimal_lane_id : lane that the global route wants at the
                                 current planning position.
        next_macro_maneuver   : upcoming route maneuver used by the
                                 intersection-mode latch.
        front_obstacle_distance_by_lane :
                                 optional nearest front-obstacle distance
                                 for each lane. Exposed for debug and
                                 other planners; the rule-based normal
                                 mode now keys detours directly off the
                                 lane safety score.
        current_time_s       : optional simulation time. Used as a fallback
                                 only when wall-clock time is unavailable.
        wall_time_s         : optional wall-clock time used for real-time
                                 stop-sign waiting and cooperative-message
                                 polling.
        traffic_signal_state : relevant traffic-light state ahead of ego
        traffic_stop_target  : optional stop target before the next junction
        ego_speed_mps        : current ego speed used for stop feasibility
        ego_max_deceleration_mps2 :
                                 strongest comfortable/allowed braking used
                                 for stop feasibility
        ego_in_junction      : whether ego is physically inside a junction
        preferred_target_lane_id : optional lane-level candidate selected by
                                 the strategy layer. The FSM still validates
                                 safety and prediction risk before preparing
                                 or executing a lane change.

        Returns
        -------
        {"decision": "lane_follow" | "lane_change_left" | "lane_change_right" | "reroute" |
                     "stop_at_intersection" | "stop_sign" | "emergency_brake",
         "target_lane_id": int,
         "lc_state": str}
        """
        self._current_candidate_evaluation = None
        self._current_update_time_s = (
            float(current_time_s)
            if current_time_s is not None
            else (float(wall_time_s) if wall_time_s is not None else None)
        )
        if self._state_enter_time_s is None and self._current_update_time_s is not None:
            self._state_enter_time_s = float(self._current_update_time_s)

        if len(lane_safety_scores) == 0:
            available = [int(ego_lane_id)] if int(ego_lane_id) != 0 else []
        else:
            available = sorted(
                {
                    int(lane_id)
                    for lane_id in lane_safety_scores.keys()
                    if int(lane_id) != 0
                }
            )
            if int(ego_lane_id) not in available and len(available) > 0:
                ego_lane_id = min(available, key=lambda lane_id: abs(int(lane_id) - int(ego_lane_id)))
        self._current_ego_lane_id = int(ego_lane_id)

        if selected_lane_id is not None and int(selected_lane_id) != 0:
            self._selected_lane_id = int(selected_lane_id)
        elif self._selected_lane_id is None:
            self._selected_lane_id = int(ego_lane_id)

        if len(available) > 0 and int(self._selected_lane_id or 0) not in available:
            if int(ego_lane_id) in available:
                self._selected_lane_id = int(ego_lane_id)
            else:
                self._selected_lane_id = int(available[0])
        if str(self._lc_state) in {
            _ABORT_LANE_CHANGE,
            _CANCEL_LANE_CHANGE,
            _REROUTE_STATE,
            _STOP_STATE,
            _YIELD_STATE,
        }:
            self._reset_lane_change_state(reason="reset")

        planner_mode = str(mode or "NORMAL").strip().upper()
        if str(planner_mode) != str(self._last_mode):
            self._reset_lane_change_state(reason="reset")
        self._last_mode = str(planner_mode)

        (
            cp_lane_closure_messages,
            cp_control_messages,
            cp_messages_refreshed,
        ) = self._cooperative_messages(
            current_time_s=current_time_s,
            wall_time_s=wall_time_s,
        )
        reroute_messages = self._poll_cooperative_messages(
            current_messages=cp_lane_closure_messages,
            refreshed=bool(cp_messages_refreshed),
        )
        active_route_reroute_messages = self._active_route_reroute_messages(
            reroute_messages=reroute_messages,
            ego_position_xy=ego_position_xy,
            global_route_points=global_route_points,
        )
        if len(reroute_messages) > 0 and len(active_route_reroute_messages) == 0:
            self._pending_reroute_messages = []
        if len(active_route_reroute_messages) > 0:
            self._set_transition_reason("cooperative_reroute")
            self._lc_state = _REROUTE_STATE
            self._target_lane_id = None
            self._source_lane_id = None
            self._selected_lane_id = int(ego_lane_id)
            self._clear_stop_state()
            reroute_ids = [
                str(message.get("id", "")).strip()
                for message in list(active_route_reroute_messages)
                if str(message.get("id", "")).strip()
            ]
            print(
                "[BEHAVIOR] decision=reroute triggered by cooperative message ids="
                f"{reroute_ids}"
            )
            if len(reroute_ids) > 0:
                self._processed_cp_message_ids.update(reroute_ids)
            self._simple_candidate_evaluation(
                selected_candidate="cooperative_reroute",
                decision=_DECISION_REROUTE,
                target_lane_id=int(ego_lane_id),
                reason="cooperative_route_blockage",
                rejected_reasons={
                    "lane_keep": "cooperative_reroute_required",
                    "lane_change_left": "cooperative_reroute_required",
                    "lane_change_right": "cooperative_reroute_required",
                },
            )
            return self._make_result(
                _DECISION_REROUTE,
                int(ego_lane_id),
                reroute_messages=active_route_reroute_messages,
            )

        stop_result, traffic_light_debug = self._cp_control_stop_result(
            ego_lane_id=int(ego_lane_id),
            control_messages=cp_control_messages,
            ego_position_xy=ego_position_xy,
            global_route_points=global_route_points,
            traffic_signal_state=traffic_signal_state,
            traffic_stop_target=traffic_stop_target,
            traffic_signal_context=traffic_signal_context,
            ego_speed_mps=float(ego_speed_mps),
            ego_max_deceleration_mps2=float(ego_max_deceleration_mps2),
            ego_in_junction=bool(ego_in_junction),
            current_time_s=current_time_s,
            wall_time_s=wall_time_s,
        )
        if stop_result is not None:
            self._set_transition_reason("traffic_or_cp_stop")
            self._lc_state = _STOP_STATE
            self._target_lane_id = None
            self._source_lane_id = None
            self._simple_candidate_evaluation(
                selected_candidate="lane_keep_stop",
                decision=str(stop_result.get("decision", _DECISION_STOP_AT_INTERSECTION)),
                target_lane_id=int(stop_result.get("target_lane_id", ego_lane_id)),
                reason="traffic_control_active",
                rejected_reasons={
                    "lane_change_left": "traffic_control_active",
                    "lane_change_right": "traffic_control_active",
                    "reroute": "traffic_control_active",
                },
            )
            return self._attach_current_candidate_evaluation(stop_result)

        emergency_brake_result = self._emergency_brake_result(
            planner_mode=str(planner_mode),
            ego_lane_id=int(ego_lane_id),
            route_optimal_lane_id=route_optimal_lane_id,
            available_lane_ids=available,
            lane_safety_scores=lane_safety_scores,
            nearest_front_obstacles_by_lane=nearest_front_obstacles_by_lane,
            traffic_light_debug=traffic_light_debug,
        )
        if emergency_brake_result is not None:
            self._set_transition_reason("emergency_brake")
            self._lc_state = _YIELD_STATE
            self._target_lane_id = None
            self._source_lane_id = None
            self._selected_lane_id = int(ego_lane_id)
            self._simple_candidate_evaluation(
                selected_candidate="emergency_brake",
                decision=_DECISION_EMERGENCY_BRAKE,
                target_lane_id=int(ego_lane_id),
                reason="immediate_collision_risk",
                rejected_reasons={
                    "lane_keep": "emergency_brake_required",
                    "lane_change_left": "emergency_brake_required",
                    "lane_change_right": "emergency_brake_required",
                },
            )
            return self._attach_current_candidate_evaluation(emergency_brake_result)

        if len(lane_safety_scores) == 0:
            self._simple_candidate_evaluation(
                selected_candidate="lane_keep",
                decision=_DECISION_FOLLOW,
                target_lane_id=int(self._selected_lane_id or ego_lane_id),
                reason="no_lane_safety_scores",
            )
            return self._make_result(
                _DECISION_FOLLOW,
                int(self._selected_lane_id or ego_lane_id),
                traffic_light_debug=traffic_light_debug,
            )

        if (
            planner_mode == "INTERSECTION"
            and not bool(ego_in_junction)
            and self._is_prepare_lane_change_state()
        ):
            lane_keep_id = int(self._source_lane_id or self._selected_lane_id or ego_lane_id)
            self._reset_lane_change_state(reason="intersection_approach_lane_lock")
            self._selected_lane_id = int(lane_keep_id)
            self._simple_candidate_evaluation(
                selected_candidate="intersection_approach_lane_keep",
                decision=_DECISION_FOLLOW,
                target_lane_id=int(lane_keep_id),
                reason="ego_not_in_junction",
                rejected_reasons={
                    "intersection_route_lane_change": "ego_not_in_junction",
                },
            )
            return self._make_result(
                _DECISION_FOLLOW,
                int(lane_keep_id),
                traffic_light_debug=dict(
                    dict(traffic_light_debug or {}),
                    intersection_approach_lane_lock=True,
                ),
            )

        # -------------------------------------------------------------- #
        # Lane-change completion (checked first)                            #
        # -------------------------------------------------------------- #
        if self._is_execute_lane_change_state() and self._target_lane_id is not None:
            if self._lane_change_complete(
                ego_lane_id=ego_lane_id,
                target_lane_id=self._target_lane_id,
                lateral_offset_m=ego_lateral_offset_m,
                heading_error_rad=ego_heading_error_rad,
            ):
                completed_target_lane_id = int(self._target_lane_id)
                self._reset_lane_change_state(reason="reset")
                self._selected_lane_id = int(completed_target_lane_id)

        if self._is_prepare_lane_change_state() and self._target_lane_id is not None:
            prepared_lane_change_result = self._maybe_advance_prepared_lane_change(
                ego_lane_id=int(ego_lane_id),
                lane_safety_scores=lane_safety_scores,
                lane_prediction_risks=lane_prediction_risks,
                traffic_light_debug=traffic_light_debug,
            )
            if prepared_lane_change_result is not None:
                self._simple_candidate_evaluation(
                    selected_candidate="lane_change_prepare_or_execute",
                    decision=str(prepared_lane_change_result.get("decision", _DECISION_FOLLOW)),
                    target_lane_id=int(prepared_lane_change_result.get("target_lane_id", self._target_lane_id or ego_lane_id)),
                    reason="continue_lane_change_intent",
                    rejected_reasons={
                        "opposite_lane_change": "intent_locked",
                    },
                )
                return self._attach_current_candidate_evaluation(prepared_lane_change_result)

        if self._is_execute_lane_change_state() and self._target_lane_id is not None and planner_mode == "INTERSECTION":
            # Compare against the *physical* ego lane, not
            # self._selected_lane_id: entering EXECUTE sets selected_lane_id
            # to the target immediately, so comparing intent-to-intent here
            # would fire this reset one tick after every EXECUTE start (the
            # vehicle hasn't moved yet), cancelling the maneuver before it
            # can make any lateral progress.
            desired_lane_id = self._intersection_target_lane_id(
                ego_lane_id=int(ego_lane_id),
                route_optimal_lane_id=route_optimal_lane_id,
                next_macro_maneuver=next_macro_maneuver,
                available_lane_ids=available,
            )
            if desired_lane_id is not None and int(ego_lane_id) == int(desired_lane_id):
                self._reset_lane_change_state(reason="reset")

        # -------------------------------------------------------------- #
        # Reverse an ongoing lane change if the target lane becomes          #
        # meaningfully worse than the previous lane.                         #
        # -------------------------------------------------------------- #
        if self._is_execute_lane_change_state() and self._target_lane_id is not None:
            lane_change_reversal = self._maybe_reverse_ongoing_lane_change(
                ego_lane_id=int(ego_lane_id),
                lane_safety_scores=lane_safety_scores,
                available_lane_ids=available,
                lane_prediction_risks=lane_prediction_risks,
                traffic_light_debug=traffic_light_debug,
            )
            if lane_change_reversal is not None:
                self._simple_candidate_evaluation(
                    selected_candidate="lane_change_abort",
                    decision=str(lane_change_reversal.get("decision", _DECISION_FOLLOW)),
                    target_lane_id=int(lane_change_reversal.get("target_lane_id", ego_lane_id)),
                    reason="abort_lane_change_safety_override",
                )
                return self._attach_current_candidate_evaluation(lane_change_reversal)

        # -------------------------------------------------------------- #
        # Do NOT interrupt ongoing lane change                              #
        # -------------------------------------------------------------- #
        if self._is_execute_lane_change_state() and self._target_lane_id is not None:
            selected_decision = (
                _DECISION_CHANGE_LEFT
                if self._lc_state == _CHANGING_LEFT
                else _DECISION_CHANGE_RIGHT
            )
            self._simple_candidate_evaluation(
                selected_candidate="lane_change_execute",
                decision=str(selected_decision),
                target_lane_id=int(self._target_lane_id),
                reason="continue_locked_lane_change",
                rejected_reasons={
                    "lane_keep": "lane_change_intent_locked",
                    "opposite_lane_change": "lane_change_intent_locked",
                },
            )
            return self._make_result(
                str(selected_decision),
                self._target_lane_id,
                traffic_light_debug=traffic_light_debug,
            )

        # -------------------------------------------------------------- #
        # INTERSECTION mode: move one lane at a time toward the green-dot  #
        # / route target lane, but only after checking target-lane safety. #
        # -------------------------------------------------------------- #
        if planner_mode == "INTERSECTION":
            current_selected_lane_id = int(self._selected_lane_id or ego_lane_id)
            if bool(ego_in_junction):
                self._simple_candidate_evaluation(
                    selected_candidate="intersection_inside_lane_keep",
                    decision=_DECISION_FOLLOW,
                    target_lane_id=int(current_selected_lane_id),
                    reason="ego_in_junction_lane_change_locked",
                    rejected_reasons={
                        "intersection_route_lane_change": "ego_in_junction",
                    },
                )
                return self._make_result(
                    _DECISION_FOLLOW,
                    int(current_selected_lane_id),
                    traffic_light_debug=dict(
                        dict(traffic_light_debug or {}),
                        intersection_lane_change_locked=True,
                        intersection_lane_change_lock_reason="ego_in_junction",
                    ),
                )
            if not bool(ego_in_junction):
                self._simple_candidate_evaluation(
                    selected_candidate="intersection_approach_lane_keep",
                    decision=_DECISION_FOLLOW,
                    target_lane_id=int(current_selected_lane_id),
                    reason="ego_not_in_junction",
                    rejected_reasons={
                        "intersection_route_lane_change": "ego_not_in_junction",
                    },
                )
                return self._make_result(
                    _DECISION_FOLLOW,
                    int(current_selected_lane_id),
                    traffic_light_debug=dict(
                        dict(traffic_light_debug or {}),
                        intersection_approach_lane_lock=True,
                    ),
                )
            desired_lane_id = self._intersection_target_lane_id(
                ego_lane_id=int(ego_lane_id),
                route_optimal_lane_id=route_optimal_lane_id,
                next_macro_maneuver=next_macro_maneuver,
                available_lane_ids=available,
            )
            if desired_lane_id is None:
                self._simple_candidate_evaluation(
                    selected_candidate="intersection_lane_keep",
                    decision=_DECISION_FOLLOW,
                    target_lane_id=int(current_selected_lane_id),
                    reason="no_intersection_route_lane",
                )
                return self._make_result(
                    _DECISION_FOLLOW,
                    int(current_selected_lane_id),
                    traffic_light_debug=traffic_light_debug,
                )
            blocked_optimal_lane_result = self._intersection_blocked_optimal_lane_result(
                ego_lane_id=int(ego_lane_id),
                current_selected_lane_id=int(current_selected_lane_id),
                desired_lane_id=int(desired_lane_id),
                lane_safety_scores=lane_safety_scores,
                traffic_stop_target=traffic_stop_target,
                traffic_light_debug=traffic_light_debug,
            )
            if blocked_optimal_lane_result is not None:
                return blocked_optimal_lane_result
            if int(desired_lane_id) == int(current_selected_lane_id):
                self._simple_candidate_evaluation(
                    selected_candidate="intersection_lane_keep",
                    decision=_DECISION_FOLLOW,
                    target_lane_id=int(current_selected_lane_id),
                    reason="already_on_intersection_route_lane",
                )
                return self._make_result(
                    _DECISION_FOLLOW,
                    int(current_selected_lane_id),
                    traffic_light_debug=traffic_light_debug,
                )
            self._set_candidate_evaluation(
                selected_candidate="intersection_route_lane_change",
                candidates=[
                    self._candidate_record(
                        name="intersection_lane_keep",
                        decision=_DECISION_FOLLOW,
                        target_lane_id=int(current_selected_lane_id),
                        cost=0.5,
                        reason="not_on_required_intersection_lane",
                    ),
                    self._candidate_record(
                        name="intersection_route_lane_change",
                        decision=(
                            _DECISION_CHANGE_LEFT
                            if int(desired_lane_id) > int(ego_lane_id)
                            else _DECISION_CHANGE_RIGHT
                        ),
                        target_lane_id=int(desired_lane_id),
                        cost=max(0.0, 1.0 - float(lane_safety_scores.get(int(desired_lane_id), 0.0))),
                        reason="move_toward_intersection_route_lane",
                    ),
                ],
                preferred_target_lane_id=int(desired_lane_id),
                reason="intersection_route_lane_required",
            )
            return self._start_one_step_lane_change(
                ego_lane_id=int(ego_lane_id),
                desired_lane_id=int(desired_lane_id),
                available_lane_ids=available,
                lane_safety_scores=lane_safety_scores,
                min_target_lane_safety=self._target_lane_safety_threshold,
                lane_prediction_risks=lane_prediction_risks,
                traffic_light_debug=traffic_light_debug,
            )

        # -------------------------------------------------------------- #
        # NORMAL mode: generate lane-level candidates, evaluate them,     #
        # then let the existing FSM execute the selected maneuver.        #
        # -------------------------------------------------------------- #
        route_lane_id = self._preferred_route_lane_id(
            route_optimal_lane_id=route_optimal_lane_id,
            available_lane_ids=available,
            fallback_lane_id=int(ego_lane_id),
        )
        current_selected_lane_id = int(self._selected_lane_id or ego_lane_id)
        candidate_evaluation = self._evaluate_normal_behavior_candidates(
            ego_lane_id=int(ego_lane_id),
            current_selected_lane_id=int(current_selected_lane_id),
            route_lane_id=int(route_lane_id),
            available_lane_ids=available,
            lane_safety_scores=lane_safety_scores,
            lane_prediction_risks=lane_prediction_risks,
            preferred_target_lane_id=preferred_target_lane_id,
        )
        candidate_target_lane_id = candidate_evaluation.get("preferred_target_lane_id", "")
        try:
            candidate_target_lane_id = int(candidate_target_lane_id)
        except Exception:
            candidate_target_lane_id = int(current_selected_lane_id)

        # Require the lane-keep state to be held for at least
        # lane_keep_min_hold_s before starting a new lane change.  Without
        # this gate, a cancelled/completed lane change drops back into
        # LANE_KEEP for a single tick and the candidate evaluator (driven by
        # noisy TTC-based safety scores near intersections with crossing
        # traffic) can immediately re-trigger the same lane change, which
        # looks like the temp_des point flickering between prepare-lane-
        # change and lane-follow every tick.
        if (
            int(candidate_target_lane_id) in available
            and int(candidate_target_lane_id) != int(current_selected_lane_id)
            and not self._state_min_hold_active()
        ):
            candidate_debug = dict(traffic_light_debug or {})
            candidate_debug["candidate_target_lane_id"] = int(candidate_target_lane_id)
            candidate_debug["candidate_selector_active"] = True
            candidate_debug["selected_candidate"] = str(candidate_evaluation.get("selected_candidate", ""))
            candidate_debug["candidate_selection_reason"] = str(candidate_evaluation.get("selection_reason", ""))
            return self._start_one_step_lane_change(
                ego_lane_id=int(ego_lane_id),
                desired_lane_id=int(candidate_target_lane_id),
                available_lane_ids=available,
                lane_safety_scores=lane_safety_scores,
                min_target_lane_safety=self._target_lane_safety_threshold,
                lane_prediction_risks=lane_prediction_risks,
                traffic_light_debug=candidate_debug,
            )

        prediction_rejected_candidates = [
            dict(candidate)
            for candidate in list(candidate_evaluation.get("rejected_candidates", []) or [])
            if str(dict(candidate).get("reason", "")) == "prediction_risk"
        ]
        follow_debug = dict(
            dict(traffic_light_debug or {}),
            candidate_selector_active=True,
            selected_candidate=str(candidate_evaluation.get("selected_candidate", "lane_keep")),
            candidate_selection_reason=str(candidate_evaluation.get("selection_reason", "")),
            lane_keep_cooldown_active=bool(self._state_min_hold_active()),
        )
        if len(prediction_rejected_candidates) > 0:
            follow_debug["lane_change_blocked_by_prediction"] = True
            follow_debug["lane_change_prediction_risk"] = dict(
                prediction_rejected_candidates[0].get("components", {})
            )
        return self._make_result(
            _DECISION_FOLLOW,
            int(current_selected_lane_id),
            traffic_light_debug=follow_debug,
        )

    # ----------------------------------------------------------------- #
    # Candidate evaluation layer                                          #
    # ----------------------------------------------------------------- #
    @staticmethod
    def _candidate_record(
        *,
        name: str,
        decision: str,
        target_lane_id: int,
        cost: float,
        status: str = "valid",
        reason: str = "",
        components: Mapping[str, object] | None = None,
    ) -> Dict[str, Any]:
        return {
            "name": str(name),
            "decision": str(normalize_behavior_decision(decision)),
            "target_lane_id": int(target_lane_id),
            "cost": float(cost),
            "status": str(status),
            "reason": str(reason),
            "components": dict(components or {}),
        }

    def _set_candidate_evaluation(
        self,
        *,
        selected_candidate: str,
        candidates: Sequence[Mapping[str, object]] | None = None,
        rejected_candidates: Sequence[Mapping[str, object]] | None = None,
        preferred_target_lane_id: int | None = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        candidate_rows = [dict(candidate) for candidate in list(candidates or [])]
        rejected_rows = [dict(candidate) for candidate in list(rejected_candidates or [])]
        for candidate in candidate_rows:
            if str(candidate.get("name", "")) == str(selected_candidate):
                candidate["status"] = "selected"
        candidate_rows.sort(key=lambda row: float(row.get("cost", float("inf"))))
        self._current_candidate_evaluation = {
            "selected_candidate": str(selected_candidate),
            "candidate_scores": candidate_rows,
            "rejected_candidates": rejected_rows,
            "preferred_target_lane_id": (
                "" if preferred_target_lane_id is None else int(preferred_target_lane_id)
            ),
            "selection_reason": str(reason),
        }
        return dict(self._current_candidate_evaluation)

    def _simple_candidate_evaluation(
        self,
        *,
        selected_candidate: str,
        decision: str,
        target_lane_id: int,
        reason: str,
        rejected_reasons: Mapping[str, str] | None = None,
    ) -> Dict[str, Any]:
        selected = self._candidate_record(
            name=str(selected_candidate),
            decision=str(decision),
            target_lane_id=int(target_lane_id),
            cost=0.0,
            status="selected",
            reason=str(reason),
        )
        rejected = [
            self._candidate_record(
                name=str(name),
                decision=_DECISION_FOLLOW,
                target_lane_id=int(target_lane_id),
                cost=1.0e6,
                status="rejected",
                reason=str(reject_reason),
            )
            for name, reject_reason in dict(rejected_reasons or {}).items()
        ]
        return self._set_candidate_evaluation(
            selected_candidate=str(selected_candidate),
            candidates=[selected],
            rejected_candidates=rejected,
            preferred_target_lane_id=int(target_lane_id),
            reason=str(reason),
        )

    def _attach_current_candidate_evaluation(
        self,
        result: Mapping[str, object],
    ) -> Dict[str, Any]:
        updated_result = dict(result)
        if isinstance(self._current_candidate_evaluation, Mapping):
            candidate_evaluation = dict(self._current_candidate_evaluation)
            updated_result["selected_candidate"] = str(candidate_evaluation.get("selected_candidate", ""))
            updated_result["candidate_scores"] = [
                dict(candidate)
                for candidate in list(candidate_evaluation.get("candidate_scores", []) or [])
            ]
            updated_result["rejected_candidates"] = [
                dict(candidate)
                for candidate in list(candidate_evaluation.get("rejected_candidates", []) or [])
            ]
            updated_result["candidate_evaluation"] = candidate_evaluation
        return updated_result

    def _lane_change_candidate_cost(
        self,
        *,
        lane_id: int,
        ego_lane_id: int,
        route_lane_id: int,
        lane_safety_scores: Mapping[int, float],
        lane_prediction_risks: Mapping[int, Mapping[str, object]] | None,
    ) -> tuple[float, Dict[str, float], str]:
        lane_safety = max(0.0, min(1.0, float(lane_safety_scores.get(int(lane_id), 0.0))))
        safety_cost = 1.0 - float(lane_safety)
        route_cost = float(self._candidate_route_deviation_weight) * abs(int(lane_id) - int(route_lane_id))
        lane_change_cost = float(self._candidate_lane_change_weight) * abs(int(lane_id) - int(ego_lane_id))
        prediction_cost = 0.0
        prediction_risk = (
            dict(lane_prediction_risks.get(int(lane_id), {}))
            if lane_prediction_risks is not None
            else {}
        )
        if bool(prediction_risk.get("risk", False)):
            prediction_cost = 1000.0
        total_cost = float(safety_cost + route_cost + lane_change_cost + prediction_cost)
        components = {
            "safety_cost": float(safety_cost),
            "route_cost": float(route_cost),
            "lane_change_cost": float(lane_change_cost),
            "prediction_cost": float(prediction_cost),
            "lane_safety": float(lane_safety),
        }
        reject_reason = "prediction_risk" if prediction_cost >= 1000.0 else ""
        return float(total_cost), components, str(reject_reason)

    def _evaluate_normal_behavior_candidates(
        self,
        *,
        ego_lane_id: int,
        current_selected_lane_id: int,
        route_lane_id: int,
        available_lane_ids: Sequence[int],
        lane_safety_scores: Mapping[int, float],
        lane_prediction_risks: Mapping[int, Mapping[str, object]] | None,
        preferred_target_lane_id: int | None = None,
    ) -> Dict[str, Any]:
        candidates: list[dict] = []
        rejected: list[dict] = []
        current_lane_score = float(lane_safety_scores.get(int(current_selected_lane_id), 0.0))
        route_lane_score = float(lane_safety_scores.get(int(route_lane_id), 0.0))
        keep_cost = max(0.0, 1.0 - max(0.0, min(1.0, current_lane_score)))
        if int(current_selected_lane_id) != int(route_lane_id):
            keep_cost += 0.20
        candidates.append(self._candidate_record(
            name="lane_keep",
            decision=_DECISION_FOLLOW,
            target_lane_id=int(current_selected_lane_id),
            cost=float(keep_cost),
            components={
                "safety_cost": float(max(0.0, 1.0 - current_lane_score)),
                "route_cost": 0.20 if int(current_selected_lane_id) != int(route_lane_id) else 0.0,
                "lane_safety": float(current_lane_score),
            },
        ))

        ordered_lane_ids = [int(lane_id) for lane_id in list(available_lane_ids or []) if int(lane_id) != 0]
        for lane_id in ordered_lane_ids:
            if int(lane_id) == int(current_selected_lane_id):
                continue
            direction = "left" if int(lane_id) > int(ego_lane_id) else "right"
            adjacent_lane_id = self._adjacent_lane_id(
                reference_lane_id=int(ego_lane_id),
                available_lane_ids=ordered_lane_ids,
                direction=str(direction),
            )
            if adjacent_lane_id is None or int(adjacent_lane_id) != int(lane_id):
                rejected.append(self._candidate_record(
                    name=f"lane_change_{direction}_to_{int(lane_id)}",
                    decision=_DECISION_CHANGE_LEFT if direction == "left" else _DECISION_CHANGE_RIGHT,
                    target_lane_id=int(lane_id),
                    cost=1.0e6,
                    status="rejected",
                    reason="not_adjacent",
                ))
                continue
            lane_score = float(lane_safety_scores.get(int(lane_id), 0.0))
            cost, components, reject_reason = self._lane_change_candidate_cost(
                lane_id=int(lane_id),
                ego_lane_id=int(ego_lane_id),
                route_lane_id=int(route_lane_id),
                lane_safety_scores=lane_safety_scores,
                lane_prediction_risks=lane_prediction_risks,
            )
            candidate_name = f"lane_change_{direction}_to_{int(lane_id)}"
            if float(lane_score) <= float(self._target_lane_safety_threshold):
                rejected.append(self._candidate_record(
                    name=str(candidate_name),
                    decision=_DECISION_CHANGE_LEFT if direction == "left" else _DECISION_CHANGE_RIGHT,
                    target_lane_id=int(lane_id),
                    cost=1.0e6,
                    status="rejected",
                    reason="target_lane_safety",
                    components=components,
                ))
                continue
            if reject_reason:
                rejected.append(self._candidate_record(
                    name=str(candidate_name),
                    decision=_DECISION_CHANGE_LEFT if direction == "left" else _DECISION_CHANGE_RIGHT,
                    target_lane_id=int(lane_id),
                    cost=1.0e6,
                    status="rejected",
                    reason=str(reject_reason),
                    components=components,
                ))
                continue
            candidates.append(self._candidate_record(
                name=str(candidate_name),
                decision=_DECISION_CHANGE_LEFT if direction == "left" else _DECISION_CHANGE_RIGHT,
                target_lane_id=int(lane_id),
                cost=float(cost),
                components=components,
            ))

        route_lane_is_unsafe = float(route_lane_score) < float(self._optimal_lane_unsafe_threshold)
        current_lane_is_safe = float(current_lane_score) >= float(self._optimal_lane_unsafe_threshold)
        preferred_lane = None
        selection_reason = "keep_current_lane"
        if preferred_target_lane_id is not None and int(preferred_target_lane_id) in ordered_lane_ids:
            preferred_lane = int(preferred_target_lane_id)
            selection_reason = "external_preferred_candidate"
        elif bool(route_lane_is_unsafe) and int(current_selected_lane_id) == int(route_lane_id):
            preferred_lane = self._best_adjacent_safe_lane(
                reference_lane_id=int(current_selected_lane_id),
                available_lane_ids=ordered_lane_ids,
                lane_safety_scores=lane_safety_scores,
            )
            selection_reason = "route_lane_unsafe_detour"
        elif not bool(route_lane_is_unsafe) and not bool(current_lane_is_safe):
            preferred_lane = int(route_lane_id)
            selection_reason = "return_to_safe_route_lane"
        elif bool(route_lane_is_unsafe) and not bool(current_lane_is_safe):
            preferred_lane = self._best_adjacent_safe_lane(
                reference_lane_id=int(current_selected_lane_id),
                available_lane_ids=ordered_lane_ids,
                lane_safety_scores=lane_safety_scores,
                excluded_lane_ids=[int(route_lane_id)],
            )
            selection_reason = "current_lane_unsafe_detour"

        if preferred_lane is None or int(preferred_lane) == int(current_selected_lane_id):
            selected_name = "lane_keep"
            preferred_lane = int(current_selected_lane_id)
        else:
            selected_match = [
                row for row in candidates
                if int(row.get("target_lane_id", 0)) == int(preferred_lane)
            ]
            if len(selected_match) == 0:
                selected_name = "lane_keep"
                preferred_lane = int(current_selected_lane_id)
                selection_reason = "preferred_candidate_rejected"
            else:
                selected_name = str(selected_match[0].get("name", "lane_change"))

        return self._set_candidate_evaluation(
            selected_candidate=str(selected_name),
            candidates=candidates,
            rejected_candidates=rejected,
            preferred_target_lane_id=int(preferred_lane),
            reason=str(selection_reason),
        )

    # ----------------------------------------------------------------- #
    # Helpers                                                             #
    # ----------------------------------------------------------------- #
    def _set_transition_reason(self, reason: str) -> None:
        self._pending_transition_reason = str(reason or "unspecified")

    def _state_elapsed_s(self) -> float:
        if self._current_update_time_s is None or self._state_enter_time_s is None:
            return float("inf")
        return max(0.0, float(self._current_update_time_s) - float(self._state_enter_time_s))

    def _state_min_hold_s(self, state: str | None = None) -> float:
        state_name = str(self._lc_state if state is None else state)
        if state_name in {_PREPARE_LANE_CHANGE_LEFT, _PREPARE_LANE_CHANGE_RIGHT}:
            return float(self._prepare_lane_change_min_hold_s)
        if state_name in {_EXECUTE_LANE_CHANGE_LEFT, _EXECUTE_LANE_CHANGE_RIGHT}:
            return float(self._execute_lane_change_min_hold_s)
        if state_name == _LANE_KEEP:
            return float(self._lane_keep_min_hold_s)
        return 0.0

    def _state_min_hold_active(self, state: str | None = None) -> bool:
        return float(self._state_elapsed_s()) < float(self._state_min_hold_s(state))

    def _reset_lane_change_state(self, reason: str = "reset") -> None:
        self._set_transition_reason(reason)
        self._lc_state = _LANE_KEEP
        self._target_lane_id = None
        self._source_lane_id = None

    def _is_prepare_lane_change_state(self) -> bool:
        return str(self._lc_state) in {
            _PREPARE_LANE_CHANGE_LEFT,
            _PREPARE_LANE_CHANGE_RIGHT,
        }

    def _is_execute_lane_change_state(self) -> bool:
        return str(self._lc_state) in {
            _EXECUTE_LANE_CHANGE_LEFT,
            _EXECUTE_LANE_CHANGE_RIGHT,
        }

    def _decision_for_lane_change_state(self) -> str:
        if str(self._lc_state) in {
            _PREPARE_LANE_CHANGE_LEFT,
            _EXECUTE_LANE_CHANGE_LEFT,
        }:
            return _DECISION_CHANGE_LEFT
        if str(self._lc_state) in {
            _PREPARE_LANE_CHANGE_RIGHT,
            _EXECUTE_LANE_CHANGE_RIGHT,
        }:
            return _DECISION_CHANGE_RIGHT
        return _DECISION_FOLLOW

    def _prediction_risk_blocks_lane_change(
        self,
        target_lane_id: int,
        lane_prediction_risks: Mapping[int, Mapping[str, object]] | None,
    ) -> tuple[bool, dict]:
        target_lane_prediction_risk = (
            dict(lane_prediction_risks.get(int(target_lane_id), {}))
            if lane_prediction_risks is not None
            else {}
        )
        return (
            bool(target_lane_prediction_risk.get("risk", False)),
            dict(target_lane_prediction_risk),
        )

    def _cancel_prepared_lane_change(
        self,
        *,
        ego_lane_id: int,
        traffic_light_debug: Mapping[str, object] | None,
        risk_debug: Mapping[str, object] | None = None,
    ) -> Dict[str, Any]:
        selected_lane_id = int(self._source_lane_id or self._selected_lane_id or ego_lane_id)
        self._selected_lane_id = int(selected_lane_id)
        self._target_lane_id = None
        self._source_lane_id = None
        self._set_transition_reason("cancel_prepared_lane_change")
        self._lc_state = _CANCEL_LANE_CHANGE
        debug = dict(traffic_light_debug or {})
        if risk_debug is not None:
            debug.update(dict(risk_debug))
        debug["lane_change_source_lane_id"] = int(selected_lane_id)
        debug["lane_change_recovery_lane_id"] = int(selected_lane_id)
        return self._make_result(
            _DECISION_FOLLOW,
            int(selected_lane_id),
            traffic_light_debug=debug,
        )

    def _maybe_advance_prepared_lane_change(
        self,
        *,
        ego_lane_id: int,
        lane_safety_scores: Mapping[int, float],
        lane_prediction_risks: Mapping[int, Mapping[str, object]] | None,
        traffic_light_debug: Mapping[str, object] | None = None,
    ) -> Dict[str, Any] | None:
        if not self._is_prepare_lane_change_state() or self._target_lane_id is None:
            return None

        target_lane_id = int(self._target_lane_id)
        target_lane_safety = float(lane_safety_scores.get(int(target_lane_id), 0.0))
        if self._state_min_hold_active():
            return self._make_result(
                _DECISION_FOLLOW,
                int(self._source_lane_id or self._selected_lane_id or ego_lane_id),
                traffic_light_debug=dict(
                    dict(traffic_light_debug or {}),
                    lane_change_min_hold_active=True,
                    lane_change_state_elapsed_s=float(self._state_elapsed_s()),
                    lane_change_state_min_hold_s=float(self._state_min_hold_s()),
                ),
            )
        if float(target_lane_safety) <= float(self._lane_change_abort_safety_threshold):
            return self._cancel_prepared_lane_change(
                ego_lane_id=int(ego_lane_id),
                traffic_light_debug=traffic_light_debug,
                risk_debug={
                    "lane_change_cancelled": True,
                    "lane_change_cancel_reason": "target_lane_safety",
                    "target_lane_safety": float(target_lane_safety),
                },
            )

        prediction_blocked, prediction_risk = self._prediction_risk_blocks_lane_change(
            target_lane_id=int(target_lane_id),
            lane_prediction_risks=lane_prediction_risks,
        )
        if bool(prediction_blocked):
            return self._cancel_prepared_lane_change(
                ego_lane_id=int(ego_lane_id),
                traffic_light_debug=traffic_light_debug,
                risk_debug={
                    "lane_change_cancelled": True,
                    "lane_change_cancel_reason": "prediction_risk",
                    "lane_change_prediction_risk": dict(prediction_risk),
                },
            )

        return self._enter_execute_lane_change_state(
            decision=self._decision_for_lane_change_state(),
            target_lane_id=int(target_lane_id),
            source_lane_id=int(self._source_lane_id or ego_lane_id),
            traffic_light_debug=traffic_light_debug,
        )

    def _poll_cooperative_messages(
        self,
        *,
        current_messages: Sequence[Mapping[str, object]] | None,
        refreshed: bool,
    ) -> Sequence[Mapping[str, object]]:
        if bool(refreshed):
            normalized_messages = [
                dict(message)
                for message in list(current_messages or [])
                if str(message.get("id", "")).strip()
            ]
            self._pending_reroute_messages = [dict(message) for message in normalized_messages]
            return [dict(message) for message in normalized_messages]
        if len(self._pending_reroute_messages) == 0:
            return []
        return [
            dict(message)
            for message in list(self._pending_reroute_messages)
            if str(message.get("id", "")).strip()
        ]

    def _cooperative_messages(
        self,
        *,
        current_time_s: float | None,
        wall_time_s: float | None,
    ) -> tuple[list[dict], list[dict], bool]:
        if not self._cp_message_path:
            return [], [], False

        refreshed = self._should_check_cooperative_messages(
            current_time_s=current_time_s,
            wall_time_s=wall_time_s,
        )
        if bool(refreshed):
            self._cached_lane_closure_messages = [
                dict(message)
                for message in load_lane_closure_messages(message_path=self._cp_message_path)
            ]
            self._cached_control_messages = [
                dict(message)
                for message in load_control_messages(message_path=self._cp_message_path)
            ]
            check_time_s = self._cooperative_message_check_reference_time_s(
                current_time_s=current_time_s,
                wall_time_s=wall_time_s,
            )
            if check_time_s is not None:
                self._last_cp_message_check_time_s = float(check_time_s)

        return (
            [dict(message) for message in list(self._cached_lane_closure_messages)],
            [dict(message) for message in list(self._cached_control_messages)],
            bool(refreshed),
        )

    def _active_route_reroute_messages(
        self,
        *,
        reroute_messages: Sequence[Mapping[str, object]] | None,
        ego_position_xy: Sequence[float] | None,
        global_route_points: Sequence[Sequence[float]] | None,
    ) -> list[dict]:
        normalized_messages = [
            dict(message)
            for message in list(reroute_messages or [])
            if isinstance(message, Mapping)
        ]
        if len(normalized_messages) == 0:
            return []

        route_points = _normalized_route_points_xy(global_route_points)
        ego_xy = _coerce_xy(ego_position_xy)
        if len(route_points) < 2 or ego_xy is None:
            return normalized_messages

        route_cumulative_distances = _route_cumulative_distances(route_points)
        ego_arc_m = _project_point_to_route_arc_m(
            point_xy=ego_xy,
            route_points=route_points,
            cumulative_distances=route_cumulative_distances,
        )
        if ego_arc_m is None:
            return normalized_messages

        active_messages: list[dict] = []
        for message in normalized_messages:
            message_xy = _coerce_xy(message.get("position", None))
            if message_xy is None:
                active_messages.append(dict(message))
                continue
            if (
                _route_min_distance_to_point_m(route_points, message_xy)
                > float(DEFAULT_REROUTE_ROUTE_CLEARANCE_M)
            ):
                continue
            message_arc_m = _project_point_to_route_arc_m(
                point_xy=message_xy,
                route_points=route_points,
                cumulative_distances=route_cumulative_distances,
            )
            if (
                message_arc_m is not None
                and float(message_arc_m)
                < float(ego_arc_m) - float(DEFAULT_REROUTE_ROUTE_BEHIND_TOLERANCE_M)
            ):
                continue
            active_messages.append(dict(message))
        return active_messages

    def _clear_stop_state(self) -> None:
        self._stop = False
        self._stopping_point = None
        self._active_control_message_id = None
        self._active_control_message_type = None
        self._stop_sign_wait_started_wall_time_s = None

    def _cp_control_stop_result(
        self,
        *,
        ego_lane_id: int,
        control_messages: Sequence[Mapping[str, object]],
        ego_position_xy: Sequence[float] | None,
        global_route_points: Sequence[Sequence[float]] | None,
        traffic_signal_state: str | None,
        traffic_stop_target: Mapping[str, object] | None,
        traffic_signal_context: Mapping[str, object] | None,
        ego_speed_mps: float,
        ego_max_deceleration_mps2: float,
        ego_in_junction: bool,
        current_time_s: float | None,
        wall_time_s: float | None,
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        control_result, control_debug = self._control_message_result(
            ego_lane_id=int(ego_lane_id),
            control_messages=control_messages,
            ego_position_xy=ego_position_xy,
            global_route_points=global_route_points,
            ego_speed_mps=float(ego_speed_mps),
            ego_max_deceleration_mps2=float(ego_max_deceleration_mps2),
            ego_in_junction=bool(ego_in_junction),
            current_time_s=current_time_s,
            wall_time_s=wall_time_s,
        )
        control_signal_state = normalize_signal_state(
            control_debug.get("signal_state", "unknown")
        )
        if control_result is not None or (
            bool(control_debug.get("control_found", False))
            and str(control_signal_state) in {"green", "red", "yellow"}
        ):
            return control_result, control_debug
        return self._traffic_light_stop_result(
            ego_lane_id=int(ego_lane_id),
            traffic_signal_state=traffic_signal_state,
            traffic_stop_target=traffic_stop_target,
            traffic_signal_context=traffic_signal_context,
            ego_speed_mps=float(ego_speed_mps),
            ego_max_deceleration_mps2=float(ego_max_deceleration_mps2),
            ego_in_junction=bool(ego_in_junction),
        )

    def _control_message_result(
        self,
        *,
        ego_lane_id: int,
        control_messages: Sequence[Mapping[str, object]],
        ego_position_xy: Sequence[float] | None,
        global_route_points: Sequence[Sequence[float]] | None,
        ego_speed_mps: float,
        ego_max_deceleration_mps2: float,
        ego_in_junction: bool,
        current_time_s: float | None,
        wall_time_s: float | None,
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        debug = {
            "signal_state": "unknown",
            "signal_distance_m": None,
            "signal_found": False,
            "signal_actor_id": None,
            "signal_actor_name": "",
            "signal_source": "cp_message",
            "stop_target_distance_m": None,
            "should_stop_now": False,
            "stop_latched": bool(self._stop),
            "stop_decision_active": bool(self._stop),
            "latched_signal_actor_id": self._active_control_message_id,
            "control_found": False,
            "control_type": "",
            "control_message_id": "",
            "stop_wait_elapsed_s": None,
            "stop_wait_remaining_s": None,
            "stop_wait_duration_s": float(self._stop_sign_wait_duration_s),
            "stop_wait_started": bool(self._stop_sign_wait_started_wall_time_s is not None),
        }
        countdown_time_s = (
            float(wall_time_s)
            if wall_time_s is not None
            else (
                float(current_time_s)
                if current_time_s is not None
                else 0.0
            )
        )
        selected_control = self._nearest_control_message_ahead(
            control_messages=control_messages,
            ego_position_xy=ego_position_xy,
            global_route_points=global_route_points,
        )
        if selected_control is None:
            if (
                self._stop
                and str(self._active_control_message_type) == "stop"
                and self._stop_sign_wait_started_wall_time_s is not None
                and isinstance(self._stopping_point, Mapping)
            ):
                stop_wait_elapsed_s = max(
                    0.0,
                    float(countdown_time_s) - float(self._stop_sign_wait_started_wall_time_s),
                )
                target_lane_id = int(self._stopping_point.get("lane_id", int(ego_lane_id)))
                self._selected_lane_id = int(target_lane_id)
                debug.update(
                    {
                        "control_found": True,
                        "control_type": "stop",
                        "control_message_id": str(self._active_control_message_id or ""),
                        "signal_actor_id": str(self._active_control_message_id or "") or None,
                        "signal_actor_name": str(self._active_control_message_id or ""),
                        "stop_target_distance_m": self._stopping_point.get("distance_m", None),
                        "should_stop_now": True,
                        "stop_latched": True,
                        "stop_decision_active": True,
                        "latched_signal_actor_id": self._active_control_message_id,
                        "stop_wait_elapsed_s": float(stop_wait_elapsed_s),
                        "stop_wait_remaining_s": max(
                            0.0,
                            float(self._stop_sign_wait_duration_s) - float(stop_wait_elapsed_s),
                        ),
                        "stop_wait_duration_s": float(self._stop_sign_wait_duration_s),
                        "stop_wait_started": True,
                    }
                )
                if float(stop_wait_elapsed_s) >= float(self._stop_sign_wait_duration_s):
                    if str(self._active_control_message_id or "").strip():
                        self._completed_stop_sign_message_ids.add(
                            str(self._active_control_message_id).strip()
                        )
                    self._clear_stop_state()
                    debug.update(
                        {
                            "should_stop_now": False,
                            "stop_latched": False,
                            "stop_decision_active": False,
                            "latched_signal_actor_id": None,
                        }
                    )
                    return None, debug
                return (
                    self._make_result(
                        _DECISION_STOP_SIGN,
                        int(target_lane_id),
                        stop_target=self._stopping_point,
                        traffic_light_debug=debug,
                        blue_dot_rolling=False,
                        mode_override="INTERSECTION",
                        stop=True,
                    ),
                    debug,
                )
            if self._active_control_message_type in {"traffic_light", "stop"}:
                self._clear_stop_state()
            return None, debug

        message, stop_target = selected_control
        message_type = _normalize_control_message_type(message.get("type", ""))
        message_id = str(message.get("id", "")).strip()
        signal_state = normalize_signal_state(message.get("state", "unknown"))
        explicit_stop = _optional_bool(message.get("stop", None))
        signal_actor_name = str(
            message.get("signal_actor_name", message.get("intersection_marker_name", message_id))
        ).strip()
        stop_distance_m = None
        try:
            stop_distance_m = float(stop_target.get("distance_m", 0.0))
        except Exception:
            stop_distance_m = None

        debug.update(
            {
                "signal_state": str(signal_state),
                "signal_distance_m": stop_distance_m,
                "signal_found": bool(message_type == "traffic_light"),
                "signal_actor_id": str(message_id) if message_id else None,
                "signal_actor_name": str(signal_actor_name),
                "stop_target_distance_m": stop_distance_m,
                "stop_latched": bool(self._stop),
                "stop_decision_active": bool(self._stop),
                "latched_signal_actor_id": self._active_control_message_id,
                "control_found": True,
                "control_type": str(message_type),
                "control_message_id": str(message_id),
            }
        )

        if str(message_type) == "traffic_light":
            should_stop_now = False
            if str(signal_state) == "green":
                self._clear_stop_state()
            elif explicit_stop is not None:
                should_stop_now = bool(explicit_stop)
                if bool(should_stop_now) and isinstance(stop_target, Mapping):
                    try:
                        should_stop_now = float(stop_target.get("distance_m", 0.0)) >= -1.0
                    except Exception:
                        should_stop_now = True
                elif not bool(should_stop_now):
                    self._clear_stop_state()
            elif (
                self._stop
                and str(self._active_control_message_type) == "traffic_light"
                and str(self._active_control_message_id or "") == str(message_id)
                and str(signal_state) in {"red", "yellow"}
            ):
                should_stop_now = True
            elif str(signal_state) == "red":
                should_stop_now = isinstance(stop_target, Mapping)
                if isinstance(stop_target, Mapping):
                    try:
                        should_stop_now = float(stop_target.get("distance_m", 0.0)) >= 0.0
                    except Exception:
                        should_stop_now = True
            elif str(signal_state) == "yellow":
                should_stop_now = should_stop_for_signal(
                    signal_state=signal_state,
                    stop_target=stop_target,
                    ego_velocity_mps=float(ego_speed_mps),
                    ego_max_deceleration_mps2=float(ego_max_deceleration_mps2),
                    ego_in_junction=bool(ego_in_junction),
                )
            else:
                should_stop_now = False
                if (
                    self._stop
                    and str(self._active_control_message_type) == "traffic_light"
                    and str(self._active_control_message_id or "") == str(message_id)
                ):
                    self._clear_stop_state()
            debug["should_stop_now"] = bool(should_stop_now)
            debug["stop_latched"] = bool(self._stop)
            debug["stop_decision_active"] = bool(should_stop_now or self._stop)
            if not bool(should_stop_now):
                return None, debug
            self._stop = True
            self._stopping_point = dict(stop_target)
            self._active_control_message_id = str(message_id)
            self._active_control_message_type = "traffic_light"
            self._stop_sign_wait_started_wall_time_s = None
            target_lane_id = int(ego_lane_id)
            self._stopping_point["lane_id"] = int(target_lane_id)
            self._selected_lane_id = int(target_lane_id)
            debug["stop_latched"] = True
            debug["stop_decision_active"] = True
            debug["latched_signal_actor_id"] = self._active_control_message_id
            return (
                self._make_result(
                    _DECISION_STOP_AT_INTERSECTION,
                    int(target_lane_id),
                    stop_target=self._stopping_point,
                    traffic_light_debug=debug,
                    blue_dot_rolling=False,
                    mode_override="INTERSECTION",
                    stop=True,
                ),
                debug,
            )

        if str(message_type) != "stop":
            return None, debug
        if message_id in self._completed_stop_sign_message_ids:
            return None, debug

        self._stop = True
        self._stopping_point = dict(stop_target)
        self._active_control_message_id = str(message_id)
        self._active_control_message_type = "stop"
        if self._stop_sign_wait_started_wall_time_s is None:
            reached_stop_target = self._has_reached_stop_target(
                stop_target=stop_target,
                ego_speed_mps=float(ego_speed_mps),
            )
            if bool(reached_stop_target):
                self._stop_sign_wait_started_wall_time_s = float(countdown_time_s)
                if self._cp_message_path and str(message_id).strip():
                    remove_cp_messages_by_id(
                        [str(message_id).strip()],
                        message_path=self._cp_message_path,
                    )
        if self._stop_sign_wait_started_wall_time_s is not None:
            stop_wait_elapsed_s = max(
                0.0,
                float(countdown_time_s) - float(self._stop_sign_wait_started_wall_time_s),
            )
            debug["stop_wait_elapsed_s"] = float(stop_wait_elapsed_s)
            debug["stop_wait_remaining_s"] = max(
                0.0,
                float(self._stop_sign_wait_duration_s) - float(stop_wait_elapsed_s),
            )
            debug["stop_wait_duration_s"] = float(self._stop_sign_wait_duration_s)
            debug["stop_wait_started"] = True
            if float(stop_wait_elapsed_s) >= float(self._stop_sign_wait_duration_s):
                self._completed_stop_sign_message_ids.add(str(message_id))
                self._clear_stop_state()
                debug.update(
                    {
                        "should_stop_now": False,
                        "stop_latched": False,
                        "stop_decision_active": False,
                        "latched_signal_actor_id": None,
                    }
                )
                return None, debug

        target_lane_id = int(stop_target.get("lane_id", int(ego_lane_id)))
        self._selected_lane_id = int(target_lane_id)
        debug["should_stop_now"] = True
        debug["stop_latched"] = True
        debug["stop_decision_active"] = True
        return (
            self._make_result(
                _DECISION_STOP_SIGN,
                int(target_lane_id),
                stop_target=self._stopping_point,
                traffic_light_debug=debug,
                blue_dot_rolling=False,
                mode_override="INTERSECTION",
                stop=True,
            ),
            debug,
        )

    def _nearest_control_message_ahead(
        self,
        *,
        control_messages: Sequence[Mapping[str, object]],
        ego_position_xy: Sequence[float] | None,
        global_route_points: Sequence[Sequence[float]] | None,
    ) -> tuple[dict, dict] | None:
        ego_xy = _coerce_xy(ego_position_xy)
        if ego_xy is None:
            return None

        route_points = _normalized_route_points_xy(global_route_points)
        route_cumulative_distances = _route_cumulative_distances(route_points)
        ego_arc_m = (
            _project_point_to_route_arc_m(
                point_xy=ego_xy,
                route_points=route_points,
                cumulative_distances=route_cumulative_distances,
            )
            if len(route_points) >= 2
            else None
        )
        candidates: list[tuple[float, float, dict, dict]] = []
        active_message_candidate: tuple[float, float, dict, dict] | None = None

        for raw_message in list(control_messages or []):
            if not isinstance(raw_message, Mapping):
                continue
            message_type = _normalize_control_message_type(raw_message.get("type", ""))
            if message_type not in {"traffic_light", "stop"}:
                continue
            message_id = str(raw_message.get("id", "")).strip()
            if not message_id:
                continue
            stop_xy = _coerce_xy(raw_message.get("stopping_point", None))
            if stop_xy is None:
                continue
            euclidean_distance_m = math.hypot(
                float(stop_xy[0]) - float(ego_xy[0]),
                float(stop_xy[1]) - float(ego_xy[1]),
            )
            forward_distance_m = float(euclidean_distance_m)
            if ego_arc_m is not None and len(route_points) >= 2:
                stop_arc_m = _project_point_to_route_arc_m(
                    point_xy=stop_xy,
                    route_points=route_points,
                    cumulative_distances=route_cumulative_distances,
                )
                if stop_arc_m is None:
                    continue
                forward_distance_m = float(stop_arc_m) - float(ego_arc_m)
            stop_target = {
                "id": str(message_id),
                "type": str(message_type),
                "state": str(raw_message.get("state", "")),
                "x_m": float(stop_xy[0]),
                "y_m": float(stop_xy[1]),
                "lane_id": int(raw_message.get("lane_id", 0) or 0),
                "road_id": int(raw_message.get("road_id", -1) or -1),
                "distance_m": float(forward_distance_m),
            }
            if (
                self._stop
                and str(self._active_control_message_id or "") == str(message_id)
                and str(self._active_control_message_type or "") == str(message_type)
            ):
                active_message_candidate = (
                    float(forward_distance_m),
                    float(euclidean_distance_m),
                    dict(raw_message),
                    dict(stop_target),
                )
            if float(forward_distance_m) < -1.0:
                continue
            candidates.append(
                (
                    max(0.0, float(forward_distance_m)),
                    float(euclidean_distance_m),
                    dict(raw_message),
                    dict(stop_target),
                )
            )

        if active_message_candidate is not None:
            active_forward_distance_m, active_euclidean_distance_m, active_message, active_stop_target = active_message_candidate
            active_stop_target["distance_m"] = float(active_forward_distance_m)
            active_stop_target["euclidean_distance_m"] = float(active_euclidean_distance_m)
            return dict(active_message), dict(active_stop_target)
        if len(candidates) == 0:
            return None
        candidates.sort(key=lambda item: (float(item[0]), float(item[1])))
        best_forward_distance_m, best_euclidean_distance_m, best_message, best_stop_target = candidates[0]
        best_stop_target["distance_m"] = float(best_forward_distance_m)
        best_stop_target["euclidean_distance_m"] = float(best_euclidean_distance_m)
        return dict(best_message), dict(best_stop_target)

    def _has_reached_stop_target(
        self,
        *,
        stop_target: Mapping[str, object],
        ego_speed_mps: float,
    ) -> bool:
        if float(ego_speed_mps) >= float(DEFAULT_STOP_SIGN_COUNTDOWN_SPEED_THRESHOLD_MPS):
            return False

        distance_candidates_m: list[float] = []
        for distance_key in ("euclidean_distance_m", "distance_m"):
            try:
                distance_value_m = float(stop_target.get(distance_key, float("inf")))
            except Exception:
                continue
            if math.isfinite(distance_value_m):
                distance_candidates_m.append(abs(float(distance_value_m)))
        if len(distance_candidates_m) == 0:
            return True
        return min(distance_candidates_m) <= float(
            DEFAULT_STOP_SIGN_COUNTDOWN_START_DISTANCE_THRESHOLD_M
        )

    def _emergency_brake_result(
        self,
        *,
        planner_mode: str,
        ego_lane_id: int,
        route_optimal_lane_id: int | None,
        available_lane_ids: Sequence[int],
        lane_safety_scores: Mapping[int, float],
        nearest_front_obstacles_by_lane: Mapping[int, Mapping[str, object]] | None,
        traffic_light_debug: Mapping[str, object] | None = None,
    ) -> Dict[str, Any] | None:
        if not self._should_trigger_emergency_brake(
            planner_mode=str(planner_mode),
            ego_lane_id=int(ego_lane_id),
            route_optimal_lane_id=route_optimal_lane_id,
            available_lane_ids=available_lane_ids,
            lane_safety_scores=lane_safety_scores,
        ):
            return None
        if not isinstance(nearest_front_obstacles_by_lane, Mapping):
            return None
        lead_obstacle = nearest_front_obstacles_by_lane.get(int(ego_lane_id), None)
        if not isinstance(lead_obstacle, Mapping):
            return None

        target_lane_id = int(ego_lane_id)
        obstacle_heading_rad = float(lead_obstacle.get("psi", 0.0))
        follow_target = {
            "x_m": float(lead_obstacle.get("x", 0.0)),
            "y_m": float(lead_obstacle.get("y", 0.0)),
            "target_v_mps": max(0.0, float(lead_obstacle.get("v", 0.0))),
            "heading_rad": float(obstacle_heading_rad),
            "lane_id": int(target_lane_id),
            "road_id": int(lead_obstacle.get("road_id", -1) or -1),
            "distance_m": max(0.0, float(lead_obstacle.get("front_distance_m", 0.0))),
            "source_obstacle_id": str(lead_obstacle.get("vehicle_id", "")),
        }
        self._selected_lane_id = int(target_lane_id)
        return self._make_result(
            _DECISION_EMERGENCY_BRAKE,
            int(target_lane_id),
            traffic_light_debug=traffic_light_debug,
            follow_target=follow_target,
        )

    def _should_trigger_emergency_brake(
        self,
        *,
        planner_mode: str,
        ego_lane_id: int,
        route_optimal_lane_id: int | None,
        available_lane_ids: Sequence[int],
        lane_safety_scores: Mapping[int, float],
    ) -> bool:
        if len(lane_safety_scores) == 0:
            return False
        current_lane_score = float(lane_safety_scores.get(int(ego_lane_id), 0.0))
        if current_lane_score > 0.0:
            return False
        normalized_mode = str(planner_mode or "NORMAL").strip().upper()
        if normalized_mode == "INTERSECTION":
            optimal_lane_id = self._preferred_route_lane_id(
                route_optimal_lane_id=route_optimal_lane_id,
                available_lane_ids=available_lane_ids,
                fallback_lane_id=int(ego_lane_id),
            )
            optimal_lane_score = float(lane_safety_scores.get(int(optimal_lane_id), 0.0))
            if int(optimal_lane_id) == int(ego_lane_id):
                return float(optimal_lane_score) <= 0.0
            return float(current_lane_score) <= 0.0 and float(optimal_lane_score) <= 0.0

        reachable_adjacent_lane_ids = []
        for direction in ("left", "right"):
            adjacent_lane_id = self._adjacent_lane_id(
                reference_lane_id=int(ego_lane_id),
                available_lane_ids=available_lane_ids,
                direction=str(direction),
            )
            if adjacent_lane_id is None:
                continue
            reachable_adjacent_lane_ids.append(int(adjacent_lane_id))
        if len(reachable_adjacent_lane_ids) == 0:
            return True
        return all(
            float(lane_safety_scores.get(int(lane_id), 0.0)) <= 0.0
            for lane_id in reachable_adjacent_lane_ids
        )

    def _cooperative_message_check_reference_time_s(
        self,
        *,
        current_time_s: float | None,
        wall_time_s: float | None,
    ) -> float | None:
        if wall_time_s is not None:
            return float(wall_time_s)
        if current_time_s is not None:
            return float(current_time_s)
        return None

    def _should_check_cooperative_messages(
        self,
        *,
        current_time_s: float | None,
        wall_time_s: float | None,
    ) -> bool:
        reference_time_s = self._cooperative_message_check_reference_time_s(
            current_time_s=current_time_s,
            wall_time_s=wall_time_s,
        )
        if reference_time_s is None:
            return True
        if self._last_cp_message_check_time_s is None:
            return True
        if float(reference_time_s) < float(self._last_cp_message_check_time_s):
            return True
        return (
            float(reference_time_s) - float(self._last_cp_message_check_time_s)
        ) >= float(self._cp_message_check_period_s)

    def acknowledge_reroute_success(
        self,
        message_ids: Sequence[object] | None,
    ) -> None:
        normalized_ids = {
            str(message_id).strip()
            for message_id in list(message_ids or [])
            if str(message_id).strip()
        }
        if len(normalized_ids) == 0:
            return
        self._processed_cp_message_ids.update(normalized_ids)
        self._acknowledged_reroute_ids.update(normalized_ids)
        self._pending_reroute_messages = [
            dict(message)
            for message in list(self._pending_reroute_messages)
            if str(message.get("id", "")).strip() not in normalized_ids
        ]

    def _intersection_target_lane_id(
        self,
        ego_lane_id: int,
        route_optimal_lane_id: int | None,
        next_macro_maneuver: str | None,
        available_lane_ids: Sequence[int],
    ) -> int | None:
        del next_macro_maneuver

        if len(available_lane_ids) == 0:
            return int(ego_lane_id)
        return int(
            self._preferred_route_lane_id(
                route_optimal_lane_id=route_optimal_lane_id,
                available_lane_ids=available_lane_ids,
                fallback_lane_id=int(ego_lane_id),
            )
        )

    @staticmethod
    def _stop_target_distance_m(
        traffic_stop_target: Mapping[str, object] | None,
    ) -> float | None:
        if not isinstance(traffic_stop_target, Mapping):
            return None
        try:
            return max(0.0, float(traffic_stop_target.get("distance_m", 0.0)))
        except Exception:
            return None

    @staticmethod
    def _intersection_blocked_optimal_lane_message(
        *,
        traffic_stop_target: Mapping[str, object] | None,
        desired_lane_id: int,
    ) -> Dict[str, Any] | None:
        if not isinstance(traffic_stop_target, Mapping):
            return None
        try:
            road_id = int(traffic_stop_target.get("road_id", 0))
        except Exception:
            return None
        if int(road_id) == 0:
            return None

        section_id = None
        raw_section_id = traffic_stop_target.get("section_id", None)
        if raw_section_id is not None:
            try:
                section_id = int(raw_section_id)
            except Exception:
                section_id = None

        message_id = f"bp_intersection_lane_block:{int(road_id)}:{section_id if section_id is not None else 'na'}:{int(desired_lane_id)}"
        message: Dict[str, Any] = {
            "id": str(message_id),
            "type": "lane_closure",
            "road_id": int(road_id),
            "lane_ids": [int(desired_lane_id)],
            "source": "behavior_planner_intersection",
        }
        if section_id is not None:
            message["section_id"] = int(section_id)
        return message

    def _intersection_blocked_optimal_lane_result(
        self,
        *,
        ego_lane_id: int,
        current_selected_lane_id: int,
        desired_lane_id: int,
        lane_safety_scores: Mapping[int, float],
        traffic_stop_target: Mapping[str, object] | None,
        traffic_light_debug: Mapping[str, object] | None = None,
    ) -> Dict[str, Any] | None:
        if int(desired_lane_id) == int(ego_lane_id):
            return None

        optimal_lane_safety = float(lane_safety_scores.get(int(desired_lane_id), 0.0))
        if float(optimal_lane_safety) > float(self._target_lane_safety_threshold):
            return None

        stop_target_distance_m = self._stop_target_distance_m(traffic_stop_target)
        if stop_target_distance_m is None:
            return None
        should_reroute = (
            float(stop_target_distance_m) < float(self._intersection_reroute_stop_distance_threshold_m)
        )
        if not bool(should_reroute):
            self._set_candidate_evaluation(
                selected_candidate="intersection_lane_keep",
                candidates=[
                    self._candidate_record(
                        name="intersection_lane_keep",
                        decision=_DECISION_FOLLOW,
                        target_lane_id=int(current_selected_lane_id),
                        cost=0.0,
                        reason="blocked_target_lane_stop_is_far",
                    ),
                ],
                rejected_candidates=[
                    self._candidate_record(
                        name="intersection_route_lane_change",
                        decision=(
                            _DECISION_CHANGE_LEFT
                            if int(desired_lane_id) > int(ego_lane_id)
                            else _DECISION_CHANGE_RIGHT
                        ),
                        target_lane_id=int(desired_lane_id),
                        cost=1.0e6,
                        status="rejected",
                        reason="target_lane_safety",
                    ),
                    self._candidate_record(
                        name="intersection_reroute",
                        decision=_DECISION_REROUTE,
                        target_lane_id=int(ego_lane_id),
                        cost=1.0e6,
                        status="rejected",
                        reason="stop_target_too_far",
                    ),
                ],
                preferred_target_lane_id=int(current_selected_lane_id),
                reason="blocked_target_lane_hold",
            )
            return self._make_result(
                _DECISION_FOLLOW,
                int(current_selected_lane_id),
                traffic_light_debug=traffic_light_debug,
            )

        reroute_message = self._intersection_blocked_optimal_lane_message(
            traffic_stop_target=traffic_stop_target,
            desired_lane_id=int(desired_lane_id),
        )
        if reroute_message is None:
            self._simple_candidate_evaluation(
                selected_candidate="intersection_lane_keep",
                decision=_DECISION_FOLLOW,
                target_lane_id=int(current_selected_lane_id),
                reason="blocked_target_lane_no_reroute_message",
                rejected_reasons={
                    "intersection_route_lane_change": "target_lane_safety",
                    "intersection_reroute": "missing_reroute_metadata",
                },
            )
            return self._make_result(
                _DECISION_FOLLOW,
                int(current_selected_lane_id),
                traffic_light_debug=traffic_light_debug,
            )

        reroute_message_id = str(reroute_message.get("id", "")).strip()
        if reroute_message_id and reroute_message_id in self._acknowledged_reroute_ids:
            self._simple_candidate_evaluation(
                selected_candidate="intersection_lane_keep",
                decision=_DECISION_FOLLOW,
                target_lane_id=int(current_selected_lane_id),
                reason="reroute_already_acknowledged",
                rejected_reasons={
                    "intersection_route_lane_change": "target_lane_safety",
                    "intersection_reroute": "already_acknowledged",
                },
            )
            return self._make_result(
                _DECISION_FOLLOW,
                int(current_selected_lane_id),
                traffic_light_debug=traffic_light_debug,
            )

        self._lc_state = _IDLE
        self._target_lane_id = None
        self._selected_lane_id = int(ego_lane_id)
        self._source_lane_id = None
        print(
            "[BEHAVIOR] decision=reroute triggered by blocked intersection optimal lane "
            f"road={reroute_message.get('road_id', None)} lane={int(desired_lane_id)} "
            f"stop_dist={float(stop_target_distance_m):.2f}m"
        )
        self._set_candidate_evaluation(
            selected_candidate="intersection_reroute",
            candidates=[
                self._candidate_record(
                    name="intersection_lane_keep",
                    decision=_DECISION_FOLLOW,
                    target_lane_id=int(current_selected_lane_id),
                    cost=0.5,
                    reason="target_lane_blocked_near_stop",
                ),
                self._candidate_record(
                    name="intersection_reroute",
                    decision=_DECISION_REROUTE,
                    target_lane_id=int(ego_lane_id),
                    cost=0.0,
                    reason="blocked_target_lane_near_stop",
                ),
            ],
            rejected_candidates=[
                self._candidate_record(
                    name="intersection_route_lane_change",
                    decision=(
                        _DECISION_CHANGE_LEFT
                        if int(desired_lane_id) > int(ego_lane_id)
                        else _DECISION_CHANGE_RIGHT
                    ),
                    target_lane_id=int(desired_lane_id),
                    cost=1.0e6,
                    status="rejected",
                    reason="target_lane_safety",
                ),
            ],
            preferred_target_lane_id=int(ego_lane_id),
            reason="blocked_intersection_target_lane",
        )
        return self._make_result(
            _DECISION_REROUTE,
            int(ego_lane_id),
            reroute_messages=[reroute_message],
            traffic_light_debug=traffic_light_debug,
        )

    @staticmethod
    def _preferred_route_lane_id(
        route_optimal_lane_id: int | None,
        available_lane_ids: Sequence[int],
        fallback_lane_id: int,
    ) -> int:
        if len(available_lane_ids) == 0:
            return int(fallback_lane_id)
        if route_optimal_lane_id is None:
            return int(fallback_lane_id)
        normalized_route_lane_id = int(route_optimal_lane_id)
        if normalized_route_lane_id in available_lane_ids:
            return int(normalized_route_lane_id)
        if normalized_route_lane_id == 0:
            return int(fallback_lane_id)
        return int(
            min(
                available_lane_ids,
                key=lambda lane_id: abs(int(lane_id) - int(normalized_route_lane_id)),
            )
        )

    @staticmethod
    def _adjacent_lane_id(
        reference_lane_id: int,
        available_lane_ids: Sequence[int],
        direction: str,
    ) -> int | None:
        ordered_lane_ids = [
            int(lane_id)
            for lane_id in list(available_lane_ids or [])
            if int(lane_id) != 0
        ]
        if int(reference_lane_id) not in ordered_lane_ids:
            return None
        lane_index = ordered_lane_ids.index(int(reference_lane_id))
        if str(direction) == "left":
            if int(lane_index) + 1 >= len(ordered_lane_ids):
                return None
            return int(ordered_lane_ids[int(lane_index) + 1])
        if int(lane_index) == 0:
            return None
        return int(ordered_lane_ids[int(lane_index) - 1])

    @staticmethod
    def _lane_has_front_obstacle(
        lane_id: int,
        front_obstacle_distance_by_lane: Mapping[int, float] | None,
    ) -> bool:
        if not isinstance(front_obstacle_distance_by_lane, Mapping):
            return False
        if int(lane_id) not in front_obstacle_distance_by_lane:
            return False
        try:
            return math.isfinite(
                float(front_obstacle_distance_by_lane.get(int(lane_id), float("inf")))
            )
        except Exception:
            return False

    def _best_adjacent_safe_lane(
        self,
        reference_lane_id: int,
        available_lane_ids: Sequence[int],
        lane_safety_scores: Mapping[int, float],
        excluded_lane_ids: Sequence[int] | None = None,
    ) -> int | None:
        excluded = {int(lane_id) for lane_id in list(excluded_lane_ids or [])}
        candidates = []
        for direction in ("left", "right"):
            candidate_lane_id = self._adjacent_lane_id(
                reference_lane_id=int(reference_lane_id),
                available_lane_ids=available_lane_ids,
                direction=str(direction),
            )
            if candidate_lane_id is None or int(candidate_lane_id) in excluded:
                continue
            if int(candidate_lane_id) not in candidates:
                candidates.append(int(candidate_lane_id))
        if len(candidates) == 0:
            return None

        safe_candidates = [
            lane_id
            for lane_id in candidates
            if float(lane_safety_scores.get(int(lane_id), 0.0))
            > float(self._target_lane_safety_threshold)
        ]
        if len(safe_candidates) == 0:
            return None

        return int(
            max(
                safe_candidates,
                key=lambda lane_id: float(lane_safety_scores.get(int(lane_id), 0.0)),
            )
        )

    def _start_one_step_lane_change(
        self,
        ego_lane_id: int,
        desired_lane_id: int,
        available_lane_ids: Sequence[int],
        lane_safety_scores: Mapping[int, float] | None = None,
        min_target_lane_safety: float | None = None,
        lane_prediction_risks: Mapping[int, Mapping[str, object]] | None = None,
        traffic_light_debug: Mapping[str, object] | None = None,
    ) -> Dict[str, Any]:
        ordered_lane_ids = [
            int(lane_id)
            for lane_id in list(available_lane_ids or [])
            if int(lane_id) != 0
        ]
        if (
            int(ego_lane_id) not in ordered_lane_ids
            or int(desired_lane_id) not in ordered_lane_ids
        ):
            return self._make_result(
                _DECISION_FOLLOW,
                int(self._selected_lane_id or ego_lane_id),
                traffic_light_debug=traffic_light_debug,
            )

        ego_lane_index = ordered_lane_ids.index(int(ego_lane_id))
        desired_lane_index = ordered_lane_ids.index(int(desired_lane_id))
        if int(desired_lane_index) > int(ego_lane_index):
            target_lane_id = self._adjacent_lane_id(
                reference_lane_id=int(ego_lane_id),
                available_lane_ids=ordered_lane_ids,
                direction="left",
            )
            decision = _DECISION_CHANGE_LEFT
        elif int(desired_lane_index) < int(ego_lane_index):
            target_lane_id = self._adjacent_lane_id(
                reference_lane_id=int(ego_lane_id),
                available_lane_ids=ordered_lane_ids,
                direction="right",
            )
            decision = _DECISION_CHANGE_RIGHT
        else:
            return self._make_result(
                _DECISION_FOLLOW,
                int(self._selected_lane_id or ego_lane_id),
                traffic_light_debug=traffic_light_debug,
            )

        if target_lane_id is None:
            return self._make_result(
                _DECISION_FOLLOW,
                int(self._selected_lane_id or ego_lane_id),
                traffic_light_debug=traffic_light_debug,
            )

        if int(target_lane_id) not in available_lane_ids:
            return self._make_result(
                _DECISION_FOLLOW,
                int(self._selected_lane_id or ego_lane_id),
                traffic_light_debug=traffic_light_debug,
            )

        if min_target_lane_safety is not None:
            target_lane_safety = float(
                (lane_safety_scores or {}).get(int(target_lane_id), 0.0)
            )
            if not (float(target_lane_safety) > float(min_target_lane_safety)):
                return self._make_result(
                    _DECISION_FOLLOW,
                    int(self._selected_lane_id or ego_lane_id),
                    traffic_light_debug=traffic_light_debug,
                )

        target_lane_prediction_risk = (
            dict(lane_prediction_risks.get(int(target_lane_id), {}))
            if lane_prediction_risks is not None
            else {}
        )
        if bool(target_lane_prediction_risk.get("risk", False)):
            risk_debug = dict(traffic_light_debug or {})
            risk_debug["lane_change_blocked_by_prediction"] = True
            risk_debug["lane_change_prediction_risk"] = dict(target_lane_prediction_risk)
            return self._make_result(
                _DECISION_FOLLOW,
                int(self._selected_lane_id or ego_lane_id),
                traffic_light_debug=risk_debug,
            )

        if self._current_update_time_s is None and lane_prediction_risks is None:
            return self._enter_execute_lane_change_state(
                decision=str(decision),
                target_lane_id=int(target_lane_id),
                source_lane_id=int(ego_lane_id),
                traffic_light_debug=traffic_light_debug,
            )
        return self._enter_prepare_lane_change_state(
            decision=str(decision),
            target_lane_id=int(target_lane_id),
            source_lane_id=int(ego_lane_id),
            traffic_light_debug=traffic_light_debug,
        )

    def _maybe_reverse_ongoing_lane_change(
        self,
        *,
        ego_lane_id: int,
        lane_safety_scores: Mapping[int, float],
        available_lane_ids: Sequence[int],
        lane_prediction_risks: Mapping[int, Mapping[str, object]] | None = None,
        traffic_light_debug: Mapping[str, object] | None = None,
    ) -> Dict[str, Any] | None:
        if self._lc_state == _IDLE or self._target_lane_id is None:
            return None
        if self._source_lane_id is None:
            return None

        source_lane_id = int(self._source_lane_id)
        target_lane_id = int(self._target_lane_id)
        ordered_lane_ids = [
            int(lane_id)
            for lane_id in list(available_lane_ids or [])
            if int(lane_id) != 0
        ]
        if (
            int(source_lane_id) not in ordered_lane_ids
            or int(target_lane_id) not in ordered_lane_ids
        ):
            return None

        source_lane_safety = float(
            lane_safety_scores.get(int(source_lane_id), 0.0)
        )
        target_lane_safety = float(
            lane_safety_scores.get(int(target_lane_id), 0.0)
        )
        prediction_blocked, prediction_risk = self._prediction_risk_blocks_lane_change(
            target_lane_id=int(target_lane_id),
            lane_prediction_risks=lane_prediction_risks,
        )
        if bool(prediction_blocked):
            self._set_transition_reason("abort_ongoing_lane_change_prediction_risk")
            self._lc_state = _ABORT_LANE_CHANGE
            self._target_lane_id = None
            self._source_lane_id = None
            self._selected_lane_id = int(source_lane_id)
            abort_debug = dict(traffic_light_debug or {})
            abort_debug["lane_change_aborted"] = True
            abort_debug["lane_change_abort_reason"] = "prediction_risk"
            abort_debug["lane_change_prediction_risk"] = dict(prediction_risk)
            abort_debug["lane_change_source_lane_id"] = int(source_lane_id)
            abort_debug["lane_change_target_lane_id"] = int(target_lane_id)
            abort_debug["lane_change_recovery_lane_id"] = int(source_lane_id)
            return self._make_result(
                _DECISION_FOLLOW,
                int(source_lane_id),
                traffic_light_debug=abort_debug,
            )
        if self._state_min_hold_active():
            return None
        # Once execute has run past the commit window, a soft safety-score dip
        # alone no longer reverses the maneuver -- by then the vehicle has
        # likely made real lateral progress toward the target lane, and
        # reversing costs more (an extra unwind maneuver) than finishing.
        # Imminent-collision risk (prediction_blocked, checked above) still
        # aborts regardless of commit time. An infinite elapsed time means no
        # sim/wall clock was supplied this cycle -- treat that as "not yet
        # committed" rather than silently disabling the abort path.
        state_elapsed_s = float(self._state_elapsed_s())
        if math.isfinite(state_elapsed_s) and state_elapsed_s >= float(
            self._lane_change_safety_abort_commit_s
        ):
            return None
        if float(target_lane_safety) >= float(self._lane_change_abort_safety_threshold):
            return None
        if float(source_lane_safety) <= float(target_lane_safety) + float(self._hysteresis):
            return None

        if int(source_lane_id) == int(ego_lane_id):
            source_lane_index = ordered_lane_ids.index(int(source_lane_id))
            target_lane_index = ordered_lane_ids.index(int(target_lane_id))
            reverse_decision = (
                _DECISION_CHANGE_RIGHT
                if int(source_lane_index) < int(target_lane_index)
                else _DECISION_CHANGE_LEFT
            )
            self._set_transition_reason("abort_ongoing_lane_change")
            self._lc_state = _ABORT_LANE_CHANGE
            self._target_lane_id = None
            self._source_lane_id = None
            self._selected_lane_id = int(source_lane_id)
            abort_debug = dict(traffic_light_debug or {})
            abort_debug["lane_change_aborted"] = True
            abort_debug["lane_change_abort_reason"] = "target_lane_safety"
            abort_debug["lane_change_source_lane_id"] = int(source_lane_id)
            abort_debug["lane_change_target_lane_id"] = int(target_lane_id)
            abort_debug["lane_change_recovery_lane_id"] = int(source_lane_id)
            return self._make_result(
                str(reverse_decision),
                int(source_lane_id),
                traffic_light_debug=abort_debug,
            )
        return None

    def _enter_prepare_lane_change_state(
        self,
        decision: str,
        target_lane_id: int,
        source_lane_id: int | None = None,
        traffic_light_debug: Mapping[str, object] | None = None,
    ) -> Dict[str, Any]:
        normalized_decision = normalize_behavior_decision(decision)
        self._set_transition_reason("enter_prepare_lane_change")
        self._lc_state = (
            _PREPARE_LANE_CHANGE_LEFT
            if str(normalized_decision) == _DECISION_CHANGE_LEFT
            else _PREPARE_LANE_CHANGE_RIGHT
        )
        self._target_lane_id = int(target_lane_id)
        if source_lane_id is not None:
            self._source_lane_id = int(source_lane_id)
        elif self._selected_lane_id is not None:
            self._source_lane_id = int(self._selected_lane_id)
        else:
            self._source_lane_id = int(target_lane_id)
        self._selected_lane_id = int(self._source_lane_id)
        prepare_debug = dict(traffic_light_debug or {})
        prepare_debug["lane_change_preparing"] = True
        prepare_debug["lane_change_prepare_target_lane_id"] = int(target_lane_id)
        prepare_debug["lane_change_source_lane_id"] = int(self._source_lane_id)
        prepare_debug["lane_change_target_lane_id"] = int(target_lane_id)
        return self._make_result(
            _DECISION_FOLLOW,
            int(self._selected_lane_id),
            traffic_light_debug=prepare_debug,
        )

    def _enter_execute_lane_change_state(
        self,
        decision: str,
        target_lane_id: int,
        source_lane_id: int | None = None,
        traffic_light_debug: Mapping[str, object] | None = None,
    ) -> Dict[str, Any]:
        normalized_decision = normalize_behavior_decision(decision)
        self._set_transition_reason("enter_execute_lane_change")
        self._lc_state = (
            _EXECUTE_LANE_CHANGE_LEFT
            if str(normalized_decision) == _DECISION_CHANGE_LEFT
            else _EXECUTE_LANE_CHANGE_RIGHT
        )
        self._target_lane_id = int(target_lane_id)
        if source_lane_id is not None:
            self._source_lane_id = int(source_lane_id)
        elif self._source_lane_id is None:
            self._source_lane_id = int(self._selected_lane_id or target_lane_id)
        self._selected_lane_id = int(target_lane_id)
        execute_debug = dict(traffic_light_debug or {})
        execute_debug["lane_change_executing"] = True
        execute_debug["lane_change_source_lane_id"] = int(self._source_lane_id)
        execute_debug["lane_change_target_lane_id"] = int(target_lane_id)
        return self._make_result(
            str(normalized_decision),
            int(target_lane_id),
            traffic_light_debug=execute_debug,
        )

    def _lane_change_complete(
        self,
        ego_lane_id: int,
        target_lane_id: int,
        lateral_offset_m: float,
        heading_error_rad: float,
    ) -> bool:
        """Check if the ongoing lane change has finished."""
        if self._state_min_hold_active():
            return False
        if ego_lane_id != target_lane_id:
            return False
        if abs(lateral_offset_m) > self._lateral_complete:
            return False
        if abs(heading_error_rad) > self._heading_complete:
            return False
        return True

    def _make_result(
        self,
        decision: str,
        target_lane_id: int,
        reroute_messages: Sequence[Mapping[str, object]] | None = None,
        stop_target: Mapping[str, object] | None = None,
        follow_target: Mapping[str, object] | None = None,
        traffic_light_debug: Mapping[str, object] | None = None,
        blue_dot_rolling: bool = True,
        mode_override: str | None = None,
        stop: bool | None = None,
    ) -> BehaviorCommand:
        normalized_mode_override = (
            str(mode_override).strip().upper() if mode_override is not None else ""
        )
        active_stop = bool(self._stop if stop is None else stop)
        active_stopping_point = (
            dict(stop_target)
            if stop_target is not None
            else (
                dict(self._stopping_point)
                if isinstance(self._stopping_point, Mapping)
                else None
            )
        )
        debug = dict(traffic_light_debug or {})
        exported_lc_state = "IDLE" if str(self._lc_state) == _LANE_KEEP else str(self._lc_state)
        lane_change_phase = "idle"
        if str(exported_lc_state).startswith("PREPARE_LANE_CHANGE"):
            lane_change_phase = "prepare"
        elif str(exported_lc_state).startswith("EXECUTE_LANE_CHANGE"):
            lane_change_phase = "execute"
        elif str(exported_lc_state) == _ABORT_LANE_CHANGE:
            lane_change_phase = "abort"
        elif str(exported_lc_state) == _CANCEL_LANE_CHANGE:
            lane_change_phase = "cancel"
        elif str(exported_lc_state) == _STOP_STATE:
            lane_change_phase = "stop"
        elif str(exported_lc_state) == _YIELD_STATE:
            lane_change_phase = "yield"
        lane_change_source_lane_id = debug.get(
            "lane_change_source_lane_id",
            "" if self._source_lane_id is None else int(self._source_lane_id),
        )
        lane_change_target_lane_id = debug.get(
            "lane_change_target_lane_id",
            "" if self._target_lane_id is None else int(self._target_lane_id),
        )
        lane_change_recovery_lane_id = debug.get("lane_change_recovery_lane_id", "")
        result = {
            "decision": str(normalize_behavior_decision(decision)),
            "target_lane_id": int(target_lane_id),
            "selected_lane_id": int(
                self._selected_lane_id if self._selected_lane_id is not None else target_lane_id
            ),
            "lc_state": str(exported_lc_state),
            "fsm_state": str(exported_lc_state),
            "lane_change_phase": str(lane_change_phase),
            "lane_change_abort_reason": str(debug.get("lane_change_abort_reason", "")),
            "lane_change_cancel_reason": str(debug.get("lane_change_cancel_reason", "")),
            "lane_change_source_lane_id": lane_change_source_lane_id,
            "lane_change_target_lane_id": lane_change_target_lane_id,
            "lane_change_recovery_lane_id": lane_change_recovery_lane_id,
            "lane_change_state_elapsed_s": float(self._state_elapsed_s()),
            "lane_change_state_min_hold_s": float(self._state_min_hold_s()),
            "blue_dot_rolling": bool(blue_dot_rolling),
            "mode_override": str(normalized_mode_override) if normalized_mode_override else None,
            "stop": bool(active_stop),
            "stopping_point": active_stopping_point,
        }
        if reroute_messages is not None:
            result["reroute_messages"] = [dict(message) for message in list(reroute_messages or [])]
        if stop_target is not None:
            result["stop_target"] = dict(stop_target)
        if follow_target is not None:
            result["follow_target"] = dict(follow_target)
        if traffic_light_debug is not None:
            result["traffic_light_debug"] = dict(debug)
        if isinstance(self._current_candidate_evaluation, Mapping):
            candidate_evaluation = dict(self._current_candidate_evaluation)
            result["selected_candidate"] = str(candidate_evaluation.get("selected_candidate", ""))
            result["candidate_scores"] = [
                dict(candidate)
                for candidate in list(candidate_evaluation.get("candidate_scores", []) or [])
            ]
            result["rejected_candidates"] = [
                dict(candidate)
                for candidate in list(candidate_evaluation.get("rejected_candidates", []) or [])
            ]
            result["candidate_evaluation"] = candidate_evaluation
        result["blocking_obstacle_id"] = (
            str(follow_target.get("source_obstacle_id", ""))
            if isinstance(follow_target, Mapping)
            else ""
        )
        self._enforce_behavior_command_invariants(result)
        result["decision_reason"] = str(self._pending_transition_reason or "")
        self._record_transition_if_needed(result=result, traffic_light_debug=traffic_light_debug)
        return result

    def _enforce_behavior_command_invariants(self, result: Dict[str, Any]) -> None:
        """Apply the Behavior Layer's "Required invariant" rules in place.

        This used to only log a warning when a command violated an
        invariant (e.g. a stop decision targeting a neighboring lane instead
        of the ego lane) and let it flow downstream unchanged. Path/MPC have
        no way to safely recover from a self-contradictory command, so a
        violation here must be corrected before the command leaves the
        behavior layer, not just reported.
        """
        ego_lane_id = int(self._current_ego_lane_id)
        violations = _behavior_command_invariant_violations(result, ego_lane_id=ego_lane_id)
        if not violations:
            return
        for violation in violations:
            if str(violation.get("kind", "")) == "stop_target_lane_mismatch":
                # A stop/emergency-brake command must always resolve to the
                # ego lane: there is nowhere else "stop" can mean.
                result["target_lane_id"] = int(ego_lane_id)
                result["selected_lane_id"] = int(ego_lane_id)
                self._selected_lane_id = int(ego_lane_id)
        result["invariant_violations"] = [str(v.get("message", "")) for v in violations]
        for violation in violations:
            print(f"[BEHAVIOR][INVARIANT_CORRECTED] {violation.get('message', '')}")

    def _record_transition_if_needed(
        self,
        *,
        result: Mapping[str, object],
        traffic_light_debug: Mapping[str, object] | None = None,
    ) -> None:
        new_state = str(self._lc_state)
        old_state = str(self._last_logged_lc_state)
        if new_state == old_state:
            return
        event_time_s = self._current_update_time_s
        self._transition_events.append({
            "sim_time_s": "" if event_time_s is None else float(event_time_s),
            "old_state": old_state,
            "new_state": new_state,
            "reason": str(self._pending_transition_reason or "unspecified"),
            "decision": str(result.get("decision", "")),
            "target_lane_id": result.get("target_lane_id", ""),
            "selected_lane_id": result.get("selected_lane_id", ""),
            "source_lane_id": dict(traffic_light_debug or {}).get(
                "lane_change_source_lane_id",
                "" if self._source_lane_id is None else int(self._source_lane_id),
            ),
            "lane_change_phase": result.get("lane_change_phase", ""),
            "state_elapsed_s": float(self._state_elapsed_s()),
            "state_min_hold_s": float(self._state_min_hold_s(old_state)),
            "target_lane_safety": (
                dict(traffic_light_debug or {}).get("target_lane_safety", "")
            ),
            "lane_change_cancel_reason": (
                dict(traffic_light_debug or {}).get("lane_change_cancel_reason", "")
            ),
            "lane_change_abort_reason": (
                dict(traffic_light_debug or {}).get("lane_change_abort_reason", "")
            ),
            "lane_change_target_lane_id": dict(traffic_light_debug or {}).get(
                "lane_change_target_lane_id",
                result.get("target_lane_id", ""),
            ),
            "lane_change_recovery_lane_id": dict(traffic_light_debug or {}).get(
                "lane_change_recovery_lane_id",
                "",
            ),
            "candidate_target_lane_id": (
                dict(traffic_light_debug or {}).get("candidate_target_lane_id", "")
            ),
        })
        self._last_logged_lc_state = str(new_state)
        self._state_enter_time_s = event_time_s
        self._pending_transition_reason = ""

    def _traffic_light_stop_result(
        self,
        *,
        ego_lane_id: int,
        traffic_signal_state: str | None,
        traffic_stop_target: Mapping[str, object] | None,
        traffic_signal_context: Mapping[str, object] | None,
        ego_speed_mps: float,
        ego_max_deceleration_mps2: float,
        ego_in_junction: bool,
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any]]:
        normalized_signal_state = normalize_signal_state(traffic_signal_state)
        signal_distance_m = None
        signal_found = False
        signal_actor_id = None
        signal_actor_name = ""
        signal_source = "none"
        signal_forward_m = None
        signal_lateral_m = None
        signal_match_distance_m = None
        signal_match_rank = None
        signal_actor_raw_state = ""
        if isinstance(traffic_signal_context, Mapping):
            signal_found = bool(traffic_signal_context.get("signal_found", False))
            signal_actor_id = traffic_signal_context.get("signal_actor_id", None)
            signal_actor_name = str(traffic_signal_context.get("signal_actor_name", ""))
            signal_actor_raw_state = str(traffic_signal_context.get("signal_actor_raw_state", ""))
            signal_source = str(traffic_signal_context.get("signal_source", "none"))
            raw_signal_distance_m = traffic_signal_context.get("signal_distance_m", None)
            try:
                signal_distance_m = None if raw_signal_distance_m is None else float(raw_signal_distance_m)
            except Exception:
                signal_distance_m = None
            try:
                raw_signal_forward_m = traffic_signal_context.get("signal_forward_m", None)
                signal_forward_m = None if raw_signal_forward_m is None else float(raw_signal_forward_m)
            except Exception:
                signal_forward_m = None
            try:
                raw_signal_lateral_m = traffic_signal_context.get("signal_lateral_m", None)
                signal_lateral_m = None if raw_signal_lateral_m is None else float(raw_signal_lateral_m)
            except Exception:
                signal_lateral_m = None
            try:
                raw_signal_match_distance_m = traffic_signal_context.get("signal_match_distance_m", None)
                signal_match_distance_m = None if raw_signal_match_distance_m is None else float(raw_signal_match_distance_m)
            except Exception:
                signal_match_distance_m = None
            try:
                raw_signal_match_rank = traffic_signal_context.get("signal_match_rank", None)
                signal_match_rank = None if raw_signal_match_rank is None else int(raw_signal_match_rank)
            except Exception:
                signal_match_rank = None

        # A green signal must always release the stop latch. Signal matching
        # can be unstable across ticks near/inside junctions, and keeping a
        # red latch alive after a confirmed green causes the planner to hold
        # `stop` incorrectly.
        release_for_green = str(normalized_signal_state) == "green"
        should_release_latch = bool(ego_in_junction) or bool(release_for_green)
        if should_release_latch:
            self._clear_stop_state()

        if str(normalized_signal_state) == "red":
            should_stop_now = (
                not bool(ego_in_junction) and isinstance(traffic_stop_target, Mapping)
            )
        elif str(normalized_signal_state) == "yellow":
            should_stop_now = should_stop_for_signal(
                signal_state=traffic_signal_state,
                stop_target=traffic_stop_target,
                ego_velocity_mps=float(ego_speed_mps),
                ego_max_deceleration_mps2=float(ego_max_deceleration_mps2),
                ego_in_junction=bool(ego_in_junction),
            )
        else:
            should_stop_now = False

        stop_target_distance_m = None
        if isinstance(traffic_stop_target, Mapping):
            try:
                stop_target_distance_m = float(traffic_stop_target.get("distance_m", 0.0))
            except Exception:
                stop_target_distance_m = None

        if should_stop_now and isinstance(traffic_stop_target, Mapping):
            self._stop = True
            self._stopping_point = dict(traffic_stop_target)
            self._active_control_message_type = "legacy_traffic_light"
            if bool(signal_found):
                self._active_control_message_id = str(signal_actor_id)
        elif (
            self._stop
            and self._stopping_point is not None
            and str(self._active_control_message_type) == "legacy_traffic_light"
            and not bool(ego_in_junction)
            and str(normalized_signal_state) != "green"
        ):
            if isinstance(traffic_stop_target, Mapping):
                self._stopping_point = dict(traffic_stop_target)
            if bool(signal_found):
                self._active_control_message_id = str(signal_actor_id)

        traffic_light_debug = {
            "signal_state": str(normalized_signal_state),
            "signal_distance_m": signal_distance_m,
            "signal_found": bool(signal_found),
            "signal_actor_id": signal_actor_id,
            "signal_actor_name": str(signal_actor_name),
            "signal_actor_raw_state": str(signal_actor_raw_state),
            "signal_source": str(signal_source),
            "signal_forward_m": signal_forward_m,
            "signal_lateral_m": signal_lateral_m,
            "signal_match_distance_m": signal_match_distance_m,
            "signal_match_rank": signal_match_rank,
            "stop_target_distance_m": stop_target_distance_m,
            "should_stop_now": bool(should_stop_now),
            "stop_latched": bool(self._stop),
            "stop_decision_active": bool(self._stop),
            "latched_signal_actor_id": self._active_control_message_id,
            "control_found": bool(signal_found),
            "control_type": "traffic_light" if bool(signal_found) else "",
            "control_message_id": str(signal_actor_id) if signal_actor_id is not None else "",
        }

        if not self._stop or self._stopping_point is None:
            return None, traffic_light_debug

        # Red/yellow light handling is a longitudinal stop commitment, not a
        # lane-change request.  Keep the stop decision on the ego lane so a
        # route or traffic-light stop marker on a neighboring lane does not
        # pull the temporary destination sideways at the start of the run.
        target_lane_id = int(ego_lane_id)
        self._stopping_point = dict(self._stopping_point)
        self._stopping_point["lane_id"] = int(target_lane_id)
        self._selected_lane_id = int(target_lane_id)
        return (
            self._make_result(
                _DECISION_STOP_AT_INTERSECTION,
                int(target_lane_id),
                stop_target=self._stopping_point,
                traffic_light_debug=traffic_light_debug,
                blue_dot_rolling=False,
                mode_override="INTERSECTION",
                stop=True,
            ),
            traffic_light_debug,
        )

    @property
    def lc_state(self) -> str:
        return str(self._lc_state)

    @property
    def target_lane_id(self) -> int | None:
        return self._target_lane_id

    @property
    def selected_lane_id(self) -> int | None:
        return self._selected_lane_id

    @property
    def stop(self) -> bool:
        return bool(self._stop)

    @property
    def stopping_point(self) -> Dict[str, Any] | None:
        return None if self._stopping_point is None else dict(self._stopping_point)

    @property
    def transition_events(self) -> list[dict]:
        return [dict(event) for event in self._transition_events]

    def reset(self) -> None:
        """Reset lane-change state machine (e.g. on scenario restart)."""
        self._lc_state = _IDLE
        self._target_lane_id = None
        self._source_lane_id = None
        self._selected_lane_id = None
        self._last_mode = "NORMAL"
        self._state_enter_time_s = None
        self._last_logged_lc_state = str(self._lc_state)
        self._current_update_time_s = None
        self._pending_transition_reason = "reset"
        self._transition_events = []
        self._current_candidate_evaluation = None
        self._last_cp_message_check_time_s = None
        self._cached_lane_closure_messages = []
        self._cached_control_messages = []
        self._processed_cp_message_ids.clear()
        self._acknowledged_reroute_ids.clear()
        self._pending_reroute_messages = []
        self._completed_stop_sign_message_ids.clear()
        self._clear_stop_state()
