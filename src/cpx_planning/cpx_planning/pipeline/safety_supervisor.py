"""Last-mile safety filtering for CP-X planner outputs."""

from __future__ import annotations

from typing import Any, Mapping, Tuple


class SafetySupervisor:
    """Filter planner controls before OpenCDA applies them."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_steer_delta: float = 0.25,
        max_throttle_delta: float = 0.45,
        max_brake_delta: float = 0.60,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_steer_delta = max(0.0, float(max_steer_delta))
        self.max_throttle_delta = max(0.0, float(max_throttle_delta))
        self.max_brake_delta = max(0.0, float(max_brake_delta))
        self._last_control = None

    def filter_control(
        self,
        *,
        control: Any,
        carla_module: Any,
        safety_manager: Any = None,
        input_frame: Any = None,
        behavior_decision: str = "",
        traffic_signal_state: str = "",
        stop_goal_active: bool = False,
        planner_accel_mps2: float = 0.0,
    ) -> Tuple[Any, str]:
        del input_frame
        if not bool(self.enabled):
            self._last_control = control
            return control, ""
        hazard_reason = self._hazard_reason(safety_manager)
        if hazard_reason:
            filtered_hazard = self._filtered_hazard_reason(
                hazard_reason=str(hazard_reason),
                behavior_decision=str(behavior_decision),
                traffic_signal_state=str(traffic_signal_state),
                stop_goal_active=bool(stop_goal_active),
                planner_accel_mps2=float(planner_accel_mps2),
            )
            if not filtered_hazard:
                self._last_control = control
                return control, "safety_supervisor_release:" + str(hazard_reason)
            safe = carla_module.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
            self._last_control = safe
            return safe, "safety_supervisor_emergency_stop:" + filtered_hazard
        if self._last_control is None:
            self._last_control = control
            return control, ""
        filtered = carla_module.VehicleControl(
            throttle=self._limit_delta(
                float(getattr(control, "throttle", 0.0)),
                float(getattr(self._last_control, "throttle", 0.0)),
                self.max_throttle_delta,
            ),
            brake=self._limit_delta(
                float(getattr(control, "brake", 0.0)),
                float(getattr(self._last_control, "brake", 0.0)),
                self.max_brake_delta,
            ),
            steer=self._limit_delta(
                float(getattr(control, "steer", 0.0)),
                float(getattr(self._last_control, "steer", 0.0)),
                self.max_steer_delta,
            ),
        )
        self._last_control = filtered
        if (
            abs(float(getattr(filtered, "throttle", 0.0)) - float(getattr(control, "throttle", 0.0))) > 1.0e-6
            or abs(float(getattr(filtered, "brake", 0.0)) - float(getattr(control, "brake", 0.0))) > 1.0e-6
            or abs(float(getattr(filtered, "steer", 0.0)) - float(getattr(control, "steer", 0.0))) > 1.0e-6
        ):
            return filtered, "safety_supervisor_rate_limit"
        return filtered, ""

    @staticmethod
    def _limit_delta(value: float, previous: float, max_delta: float) -> float:
        delta = float(value) - float(previous)
        if delta > float(max_delta):
            return float(previous) + float(max_delta)
        if delta < -float(max_delta):
            return float(previous) - float(max_delta)
        return float(value)

    @staticmethod
    def _hazard_reason(safety_manager: Any) -> str:
        queue = getattr(safety_manager, "status_queue", None)
        if not queue:
            return ""
        try:
            _, status = queue[-1]
        except Exception:
            return ""
        if not isinstance(status, Mapping):
            return ""
        active = [
            str(key)
            for key, value in dict(status).items()
            if bool(value)
        ]
        return ",".join(active)

    @staticmethod
    def _filtered_hazard_reason(
        *,
        hazard_reason: str,
        behavior_decision: str,
        traffic_signal_state: str,
        stop_goal_active: bool,
        planner_accel_mps2: float,
    ) -> str:
        hazards = {
            str(item).strip().lower()
            for item in str(hazard_reason or "").split(",")
            if str(item).strip()
        }
        if not hazards:
            return ""
        if "collision" in hazards:
            return "collision"
        normalized_behavior = str(behavior_decision or "").strip().lower()
        normalized_signal = str(traffic_signal_state or "").strip().lower()
        stop_like = bool(stop_goal_active) or normalized_behavior in {
            "stop_at_intersection",
            "stop_sign",
            "emergency_brake",
        }
        turn_like = normalized_behavior in {
            "intersection_turn_left",
            "intersection_turn_right",
        }
        if "stuck" in hazards:
            if (
                normalized_signal in {"green", "unknown"}
                and (normalized_behavior == "lane_follow" or bool(turn_like))
                and not bool(stop_like)
                and float(planner_accel_mps2) > 0.05
            ):
                hazards.discard("stuck")
            elif bool(stop_like) or normalized_signal in {"red", "yellow"}:
                return "stuck"
        return ",".join(sorted(hazards))
