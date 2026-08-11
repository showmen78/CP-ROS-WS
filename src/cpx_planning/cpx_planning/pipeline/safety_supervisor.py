"""Last-mile safety filtering for CP-X planner outputs."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence, Tuple


class SafetySupervisor:
    """Filter planner controls before OpenCDA applies them."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_steer_delta: float = 0.25,
        max_throttle_delta: float = 0.45,
        max_brake_delta: float = 0.60,
        stuck_release_min_accel_mps2: float = 0.01,
    ) -> None:
        self.enabled = bool(enabled)
        self.max_steer_delta = max(0.0, float(max_steer_delta))
        self.max_throttle_delta = max(0.0, float(max_throttle_delta))
        self.max_brake_delta = max(0.0, float(max_brake_delta))
        self.stuck_release_min_accel_mps2 = max(
            0.0, float(stuck_release_min_accel_mps2)
        )
        self._last_control = None
        self._turn_boundary_hard_stop_frames = 0
        self._turn_boundary_recovery_active = False
        self._turn_boundary_recovery_phase = ""

    @property
    def turn_boundary_recovery_active(self) -> bool:
        return bool(self._turn_boundary_recovery_active)

    @property
    def turn_boundary_recovery_phase(self) -> str:
        return str(self._turn_boundary_recovery_phase)

    def enforce_signal_stop(
        self,
        *,
        control: Any,
        carla_module: Any,
        accel_mps2: float,
        steer_rad: float,
        ego_transform: Any,
        ego_speed_mps: float,
        destination_state: Sequence[float],
        stop_goal_active: bool,
        traffic_signal_state: str,
        min_acceleration_mps2: float,
        control_factory: Callable[[float, float], Any],
        config: Mapping[str, object],
        stop_target_forward_m: object = None,
    ) -> Tuple[Any, float, float, str]:
        """Enforce the final red/yellow braking envelope.

        Routine speed tracking remains owned by SpeedPlanner and MPC.  This
        method acts only when a stop goal is active for a red or yellow signal.
        """

        signal_state = str(traffic_signal_state or "").strip().lower()
        if not bool(stop_goal_active) or signal_state not in {"red", "yellow"}:
            return control, float(accel_mps2), float(steer_rad), ""

        max_brake_accel = max(1.0e-6, abs(float(min_acceleration_mps2)))
        hold_speed_mps = max(
            0.0,
            float(config.get("full_stop_hold_speed_mps", 0.10)),
        )
        if float(ego_speed_mps) <= float(hold_speed_mps):
            hold_brake = min(
                1.0,
                max(0.0, float(config.get("full_stop_hold_brake", 0.60))),
            )
            held = carla_module.VehicleControl(
                throttle=0.0,
                brake=float(hold_brake),
                steer=0.0,
            )
            return (
                held,
                -float(hold_brake) * float(max_brake_accel),
                0.0,
                "red_yellow_stop_stationary_hold",
            )

        signed_stop_distance_m = None
        try:
            candidate_stop_distance_m = float(stop_target_forward_m)
            if math.isfinite(candidate_stop_distance_m):
                signed_stop_distance_m = float(candidate_stop_distance_m)
        except (TypeError, ValueError):
            pass
        if signed_stop_distance_m is None and destination_state is not None and len(destination_state) >= 2:
            ego_yaw_rad = math.radians(
                float(getattr(ego_transform.rotation, "yaw", 0.0))
            )
            dx_m = float(destination_state[0]) - float(ego_transform.location.x)
            dy_m = float(destination_state[1]) - float(ego_transform.location.y)
            signed_stop_distance_m = float(
                math.cos(ego_yaw_rad) * dx_m + math.sin(ego_yaw_rad) * dy_m,
            )
        if signed_stop_distance_m is None:
            signed_stop_distance_m = 1.0

        if float(signed_stop_distance_m) <= 0.0:
            hold_brake = min(
                1.0,
                max(0.0, float(config.get("full_stop_hold_brake", 0.60))),
            )
            held = carla_module.VehicleControl(
                throttle=0.0,
                brake=float(hold_brake),
                steer=0.0,
            )
            return (
                held,
                -float(hold_brake) * float(max_brake_accel),
                0.0,
                "red_yellow_stop_overshoot_hold",
            )

        stop_buffer_m = max(
            0.0,
            float(config.get("red_yellow_stop_guard_buffer_m", 1.5)),
        )
        remaining_m = max(
            0.0,
            float(signed_stop_distance_m) - float(stop_buffer_m),
        )
        comfortable_decel_mps2 = max(
            0.1,
            float(config.get("red_yellow_stop_guard_comfort_decel_mps2", 1.6)),
        )
        approach_speed_cap_mps = max(
            0.1,
            float(config.get("red_yellow_stop_guard_approach_speed_mps", 1.5)),
        )
        target_speed_mps = min(
            float(approach_speed_cap_mps),
            math.sqrt(
                max(
                    0.0,
                    2.0 * float(comfortable_decel_mps2) * float(remaining_m),
                )
            ),
        )
        if float(remaining_m) <= 0.05:
            target_speed_mps = 0.0
        speed_error_mps = float(target_speed_mps) - float(ego_speed_mps)
        desired_accel_mps2 = float(
            config.get("red_yellow_stop_guard_speed_kp", 0.8)
        ) * float(speed_error_mps)
        max_approach_accel_mps2 = float(
            config.get("red_yellow_stop_guard_approach_max_accel_mps2", 0.45)
        )
        min_decel_mps2 = -min(
            float(max_brake_accel),
            float(config.get("red_yellow_stop_guard_max_decel_mps2", 2.2)),
        )
        guarded_accel_mps2 = min(
            0.0,
            float(max_approach_accel_mps2),
            max(float(min_decel_mps2), float(desired_accel_mps2)),
        )
        guarded = control_factory(float(guarded_accel_mps2), float(steer_rad))
        reason = (
            "red_yellow_stop_final_brake_envelope"
            if float(target_speed_mps) <= 0.05
            else "red_yellow_stop_braking_envelope"
        )
        return (
            guarded,
            float(guarded_accel_mps2),
            float(steer_rad),
            str(reason),
        )

    def enforce_turn_boundary(
        self,
        *,
        control: Any,
        carla_module: Any,
        accel_mps2: float,
        steer_rad: float,
        ego_speed_mps: float,
        behavior_decision: str,
        boundary_clearance_m: object,
        min_acceleration_mps2: float,
        config: Mapping[str, object],
        control_factory: Callable[[float, float], Any] | None = None,
        boundary_recovery_planned: bool = False,
    ) -> Tuple[Any, float, float, str]:
        """Apply a continuous turn-boundary envelope.

        A boundary error can only be corrected while the vehicle retains
        steering authority and slow forward motion. Collision handling remains
        the hard-stop owner's responsibility in ``filter_control``.
        """

        if not bool(config.get("turn_road_boundary_speed_guard_enabled", True)):
            self._reset_turn_boundary_recovery()
            return control, float(accel_mps2), float(steer_rad), ""
        if str(behavior_decision or "").strip().lower() not in {
            "intersection_turn_left",
            "intersection_turn_right",
        }:
            self._reset_turn_boundary_recovery()
            return control, float(accel_mps2), float(steer_rad), ""
        try:
            clearance_m = float(boundary_clearance_m)
        except (TypeError, ValueError):
            return control, float(accel_mps2), float(steer_rad), ""
        if not math.isfinite(clearance_m):
            return control, float(accel_mps2), float(steer_rad), ""
        if bool(boundary_recovery_planned):
            self._turn_boundary_hard_stop_frames = 0
            self._turn_boundary_recovery_active = True
            self._turn_boundary_recovery_phase = "planned_recovery"
            return (
                control,
                float(accel_mps2),
                float(steer_rad),
                "turn_road_boundary_planned_recovery_monitor:"
                f"clearance={float(clearance_m):.2f}",
            )

        warning_clearance_m = float(
            config.get("turn_road_boundary_warning_clearance_m", 0.15)
        )
        release_clearance_m = max(
            float(warning_clearance_m),
            float(config.get("turn_road_boundary_release_clearance_m", 0.20)),
        )
        guard_was_active = bool(self._turn_boundary_recovery_active)
        if (
            float(clearance_m) >= float(warning_clearance_m)
            and (
                not bool(guard_was_active)
                or float(clearance_m) >= float(release_clearance_m)
            )
        ):
            self._reset_turn_boundary_recovery()
            return control, float(accel_mps2), float(steer_rad), ""

        critical_clearance_m = float(
            config.get("turn_road_boundary_critical_clearance_m", -0.30)
        )
        if float(critical_clearance_m) >= float(warning_clearance_m):
            critical_clearance_m = float(warning_clearance_m) - 0.10

        self._turn_boundary_hard_stop_frames = 0
        self._turn_boundary_recovery_active = True
        critical_recovery = bool(
            float(clearance_m) <= float(critical_clearance_m)
        )
        self._turn_boundary_recovery_phase = (
            "critical_recovery"
            if bool(critical_recovery)
            else "continuous_recovery"
        )
        warning_speed_cap_mps = max(
            0.1,
            float(config.get("turn_road_boundary_warning_speed_mps", 1.0)),
        )
        minimum_recovery_speed_mps = min(
            float(warning_speed_cap_mps),
            max(
                0.1,
                float(
                    config.get(
                        "turn_road_boundary_min_recovery_speed_mps",
                        0.65,
                    )
                ),
            ),
        )
        clearance_span_m = max(
            1.0e-3,
            float(warning_clearance_m) - float(critical_clearance_m),
        )
        severity = min(
            1.0,
            max(
                0.0,
                (float(warning_clearance_m) - float(clearance_m))
                / float(clearance_span_m),
            ),
        )
        if bool(critical_recovery):
            speed_cap_mps = min(
                float(warning_speed_cap_mps),
                max(
                    0.1,
                    float(
                        config.get(
                            "turn_road_boundary_critical_recovery_speed_mps",
                            0.35,
                        )
                    ),
                ),
            )
        else:
            speed_cap_mps = (
                float(warning_speed_cap_mps)
                - float(severity)
                * (
                    float(warning_speed_cap_mps)
                    - float(minimum_recovery_speed_mps)
                )
            )
        recovery_kp = max(
            0.0,
            float(config.get("turn_road_boundary_recovery_speed_kp", 1.2)),
        )
        recovery_max_accel_mps2 = max(
            0.1,
            float(
                config.get(
                    "turn_road_boundary_recovery_max_accel_mps2",
                    0.80,
                )
            ),
        )
        envelope_accel_mps2 = min(
            float(recovery_max_accel_mps2),
            float(recovery_kp)
            * (float(speed_cap_mps) - float(ego_speed_mps)),
        )
        guarded_accel_mps2 = min(
            float(accel_mps2),
            float(envelope_accel_mps2),
        )
        if (
            bool(critical_recovery)
            and float(ego_speed_mps) < 0.5 * float(speed_cap_mps)
            and float(envelope_accel_mps2) > 0.0
        ):
            startup_accel_mps2 = min(
                float(recovery_max_accel_mps2),
                max(
                    0.05,
                    float(
                        config.get(
                            "turn_road_boundary_recovery_startup_accel_mps2",
                            0.30,
                        )
                    ),
                ),
            )
            guarded_accel_mps2 = min(
                float(envelope_accel_mps2),
                max(float(guarded_accel_mps2), float(startup_accel_mps2)),
            )
        if control_factory is None:
            return control, float(accel_mps2), float(steer_rad), ""
        guarded = control_factory(
            float(guarded_accel_mps2),
            float(steer_rad),
        )
        return (
            guarded,
            float(guarded_accel_mps2),
            float(steer_rad),
            (
                "turn_road_boundary_critical_recovery:"
                if bool(critical_recovery)
                else "turn_road_boundary_continuous_recovery:"
            )
            +
            f"clearance={float(clearance_m):.2f};"
            f"speed_cap={float(speed_cap_mps):.2f};"
            f"severity={float(severity):.2f}",
        )

    def _reset_turn_boundary_recovery(self) -> None:
        self._turn_boundary_hard_stop_frames = 0
        self._turn_boundary_recovery_active = False
        self._turn_boundary_recovery_phase = ""

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
                stuck_release_min_accel_mps2=float(
                    self.stuck_release_min_accel_mps2
                ),
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
        rate_limited = (
            abs(
                float(getattr(filtered, "throttle", 0.0))
                - float(getattr(control, "throttle", 0.0))
            )
            > 1.0e-6
            or abs(
                float(getattr(filtered, "brake", 0.0))
                - float(getattr(control, "brake", 0.0))
            )
            > 1.0e-6
            or abs(
                float(getattr(filtered, "steer", 0.0))
                - float(getattr(control, "steer", 0.0))
            )
            > 1.0e-6
        )
        requested_throttle = float(getattr(control, "throttle", 0.0))
        requested_brake = float(getattr(control, "brake", 0.0))
        if float(getattr(filtered, "throttle", 0.0)) > 1.0e-6 and float(
            getattr(filtered, "brake", 0.0)
        ) > 1.0e-6:
            if requested_brake > 1.0e-6 and requested_throttle <= 1.0e-6:
                filtered.throttle = 0.0
            else:
                filtered.brake = 0.0
        self._last_control = filtered
        if bool(rate_limited):
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

    def _filtered_hazard_reason(
        self,
        *,
        hazard_reason: str,
        behavior_decision: str,
        traffic_signal_state: str,
        stop_goal_active: bool,
        planner_accel_mps2: float,
        stuck_release_min_accel_mps2: float,
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
        driving_recovery_like = normalized_behavior in {
            "lane_change_left",
            "lane_change_right",
            "route_recovery",
        }
        if "ran_light" in hazards:
            if (
                normalized_signal in {"green", "unknown"}
                and (
                    normalized_behavior == "lane_follow"
                    or bool(turn_like)
                    or bool(driving_recovery_like)
                )
                and not bool(stop_like)
            ):
                hazards.discard("ran_light")
            elif bool(stop_like) or normalized_signal in {"red", "yellow"}:
                return "ran_light"
        if "stuck" in hazards:
            if (
                normalized_signal in {"green", "unknown"}
                and (
                    normalized_behavior == "lane_follow"
                    or bool(turn_like)
                    or bool(driving_recovery_like)
                )
                and not bool(stop_like)
                and (
                    float(planner_accel_mps2)
                    >= float(stuck_release_min_accel_mps2)
                    or (
                        bool(turn_like)
                        and bool(self._turn_boundary_recovery_active)
                    )
                )
            ):
                hazards.discard("stuck")
            elif bool(stop_like):
                return "stuck"
            elif (
                normalized_signal in {"red", "yellow"}
                and float(planner_accel_mps2)
                >= float(stuck_release_min_accel_mps2)
            ):
                # Signal colour alone does not authorize a stop. ScenarioManager
                # owns approach/commit/hold and exposes the committed state via
                # stop_goal_active. This permits progress toward a distant light.
                hazards.discard("stuck")
        if (
            "offroad" in hazards
            and bool(turn_like)
            and bool(self._turn_boundary_recovery_active)
            and normalized_signal in {"green", "unknown"}
            and not bool(stop_like)
        ):
            hazards.discard("offroad")
        return ",".join(sorted(hazards))
