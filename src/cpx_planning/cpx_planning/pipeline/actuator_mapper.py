"""CARLA actuator mapping and measured longitudinal-state feedback."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActuatorCommand:
    throttle: float
    brake: float


class CarlaActuatorMapper:
    """Map physical MPC acceleration to CARLA's normalized pedal commands.

    CARLA throttle is not a physical acceleration input.  The feedforward term
    compensates rolling/engine resistance while the speed-error term corrects
    residual inverse-model error.  Traffic-stop intent always disables both.
    """

    def __init__(self, config=None):
        cfg = dict(config or {})
        self.enabled = bool(cfg.get("actuator_compensation_enabled", True))
        self.cruise_feedforward = max(
            0.0, float(cfg.get("actuator_cruise_throttle_feedforward", 0.28))
        )
        self.cruise_max_throttle = min(
            1.0,
            max(0.0, float(cfg.get("actuator_cruise_max_throttle", 0.55))),
        )
        self.speed_error_gain = max(
            0.0, float(cfg.get("actuator_speed_error_throttle_gain", 0.0))
        )
        self.max_speed_error_boost = max(
            0.0, float(cfg.get("actuator_max_speed_error_throttle_boost", 0.20))
        )
        self.feedforward_min_target_mps = max(
            0.0, float(cfg.get("actuator_feedforward_min_target_mps", 0.3))
        )
        self.speed_deadband_mps = max(
            0.0, float(cfg.get("actuator_speed_deadband_mps", 0.15))
        )
        # CARLA's normalized brake command is substantially stronger than the
        # MPC acceleration bound. Keep an explicit, empirically calibrated
        # inverse model instead of treating brake=1 as only -3 m/s^2.
        self.tracking_brake_decel_per_unit_mps2 = max(
            1.0,
            float(cfg.get("actuator_tracking_brake_decel_per_unit_mps2", 14.0)),
        )
        self.stop_brake_decel_per_unit_mps2 = max(
            1.0,
            float(cfg.get("actuator_stop_brake_decel_per_unit_mps2", 10.0)),
        )
        self.tracking_max_brake = min(
            1.0,
            max(0.0, float(cfg.get("actuator_tracking_max_brake", 0.22))),
        )
        self.stop_max_brake = min(
            1.0,
            max(
                self.tracking_max_brake,
                float(cfg.get("actuator_stop_max_brake", 0.35)),
            ),
        )
        self.measurement_alpha = min(
            1.0,
            max(0.0, float(cfg.get("actuator_measured_accel_alpha", 0.25))),
        )
        self.max_measured_accel_abs_mps2 = max(
            0.1, float(cfg.get("actuator_max_measured_accel_abs_mps2", 3.0))
        )
        self._previous_speed_mps = None
        self._previous_time_s = None
        self.measured_accel_mps2 = 0.0

    def update_measurement(self, *, speed_mps: float, timestamp_s: float) -> float:
        speed = max(0.0, float(speed_mps))
        timestamp = float(timestamp_s)
        if self._previous_speed_mps is not None and self._previous_time_s is not None:
            dt_s = timestamp - float(self._previous_time_s)
            if 1.0e-3 < dt_s <= 0.5:
                raw = (speed - float(self._previous_speed_mps)) / dt_s
                limit = float(self.max_measured_accel_abs_mps2)
                raw = min(limit, max(-limit, raw))
                alpha = float(self.measurement_alpha)
                self.measured_accel_mps2 = (
                    (1.0 - alpha) * float(self.measured_accel_mps2)
                    + alpha * float(raw)
                )
        self._previous_speed_mps = speed
        self._previous_time_s = timestamp
        return float(self.measured_accel_mps2)

    def map_acceleration(
        self,
        *,
        acceleration_mps2: float,
        max_acceleration_mps2: float,
        min_acceleration_mps2: float,
        ego_speed_mps: float,
        target_speed_mps: float,
        stop_goal_active: bool,
    ) -> ActuatorCommand:
        accel = float(acceleration_mps2)
        max_accel = max(1.0e-6, float(max_acceleration_mps2))
        max_brake = max(1.0e-6, abs(float(min_acceleration_mps2)))
        speed_error = float(target_speed_mps) - float(ego_speed_mps)
        if accel < 0.0:
            brake_scale = (
                self.stop_brake_decel_per_unit_mps2
                if bool(stop_goal_active)
                else self.tracking_brake_decel_per_unit_mps2
            )
            brake_limit = (
                self.stop_max_brake
                if bool(stop_goal_active)
                else self.tracking_max_brake
            )
            return ActuatorCommand(
                throttle=0.0,
                brake=min(
                    float(brake_limit),
                    max(0.0, -float(accel) / float(brake_scale)),
                ),
            )
        if bool(stop_goal_active) or float(target_speed_mps) <= 0.0:
            return ActuatorCommand(throttle=0.0, brake=0.0)
        if float(speed_error) < -float(self.speed_deadband_mps):
            return ActuatorCommand(throttle=0.0, brake=0.0)

        throttle = min(1.0, max(0.0, accel / max_accel))
        if self.enabled and float(target_speed_mps) >= self.feedforward_min_target_mps:
            speed_error = max(0.0, float(speed_error))
            speed_boost = min(
                float(self.max_speed_error_boost),
                float(self.speed_error_gain) * speed_error,
            )
            throttle += float(self.cruise_feedforward) + float(speed_boost)
        return ActuatorCommand(
            throttle=min(
                float(self.cruise_max_throttle),
                max(0.0, throttle),
            ),
            brake=0.0,
        )

    def acceleration_from_command(
        self,
        *,
        throttle: float,
        brake: float,
        max_acceleration_mps2: float,
        min_acceleration_mps2: float,
        ego_speed_mps: float,
        target_speed_mps: float,
        stop_goal_active: bool,
    ) -> float:
        max_accel = max(1.0e-6, float(max_acceleration_mps2))
        max_brake = max(1.0e-6, abs(float(min_acceleration_mps2)))
        brake_value = min(1.0, max(0.0, float(brake)))
        if brake_value > 0.0:
            brake_scale = (
                self.stop_brake_decel_per_unit_mps2
                if bool(stop_goal_active)
                else self.tracking_brake_decel_per_unit_mps2
            )
            return -float(brake_value) * float(brake_scale)
        effective_throttle = min(1.0, max(0.0, float(throttle)))
        if (
            self.enabled
            and not bool(stop_goal_active)
            and float(target_speed_mps) >= self.feedforward_min_target_mps
        ):
            speed_error = max(0.0, float(target_speed_mps) - float(ego_speed_mps))
            speed_boost = min(
                float(self.max_speed_error_boost),
                float(self.speed_error_gain) * speed_error,
            )
            effective_throttle -= float(self.cruise_feedforward) + float(speed_boost)
        return max(0.0, effective_throttle) * max_accel
