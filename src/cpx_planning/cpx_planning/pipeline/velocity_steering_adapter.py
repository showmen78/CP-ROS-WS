"""CARLA adapter for the planner's velocity and steering command boundary."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

# Reference tick length the max_*_delta config values below are expressed
# against (CARLA's current fixed_delta_seconds). Steering was already rate
# based (rad/s * dt_s, see run_step); throttle/brake used a flat per-call
# delta instead, so their configured cap silently meant a different
# physical rate of change whenever this adapter is driven at something
# other than this reference tick rate (e.g. a real-vehicle actuator loop
# running slower than CARLA's 20Hz). Dividing by this constant turns the
# existing config values into a rate (per second), then multiplying by the
# actual measured dt_s below reproduces today's exact behavior at the
# reference rate and scales correctly at any other rate.
_REFERENCE_DT_S = 0.05


@dataclass(frozen=True)
class VelocitySteeringCommand:
    target_speed_mps: float
    target_steering_rad: float
    emergency_stop: bool = False
    stop_goal_active: bool = False


class CarlaVelocitySteeringAdapter:
    """Convert planner v/steering to pedals while preserving CARLA dynamics."""

    def __init__(self, config=None):
        cfg = dict(config or {})
        self.kp = max(0.0, float(cfg.get("velocity_adapter_kp", 0.18)))
        self.ki = max(0.0, float(cfg.get("velocity_adapter_ki", 0.025)))
        self.integral_limit = max(
            0.0, float(cfg.get("velocity_adapter_integral_limit", 2.0))
        )
        self.cruise_feedforward = max(
            0.0, float(cfg.get("velocity_adapter_cruise_feedforward", 0.28))
        )
        self.speed_deadband_mps = max(
            0.0, float(cfg.get("velocity_adapter_speed_deadband_mps", 0.12))
        )
        self.brake_activation_error_mps = max(
            self.speed_deadband_mps,
            float(cfg.get("velocity_adapter_brake_activation_error_mps", 0.35)),
        )
        self.max_throttle = min(
            1.0, max(0.0, float(cfg.get("velocity_adapter_max_throttle", 0.48)))
        )
        self.max_tracking_brake = min(
            1.0,
            max(0.0, float(cfg.get("velocity_adapter_max_tracking_brake", 0.10))),
        )
        self.max_stop_brake = min(
            1.0,
            max(
                self.max_tracking_brake,
                float(cfg.get("velocity_adapter_max_stop_brake", 0.35)),
            ),
        )
        self.stop_hold_speed_mps = max(
            0.0, float(cfg.get("velocity_adapter_stop_hold_speed_mps", 0.08))
        )
        self.max_throttle_delta_rate_per_s = max(
            0.0, float(cfg.get("velocity_adapter_max_throttle_delta", 0.08))
        ) / _REFERENCE_DT_S
        self.max_brake_delta_rate_per_s = max(
            0.0, float(cfg.get("velocity_adapter_max_brake_delta", 0.05))
        ) / _REFERENCE_DT_S
        self.max_steering_rate_rad_s = math.radians(max(
            0.0,
            float(cfg.get("velocity_adapter_max_steering_rate_deg_s", 25.0)),
        ))
        self.stop_steering_decay_rate_rad_s = math.radians(max(
            0.0,
            float(
                cfg.get(
                    "velocity_adapter_stop_steering_decay_rate_deg_s",
                    12.0,
                )
            ),
        ))
        self.emergency_steering_decay_rate_rad_s = math.radians(max(
            0.0,
            float(
                cfg.get(
                    "velocity_adapter_emergency_steering_decay_rate_deg_s",
                    25.0,
                )
            ),
        ))
        self._integral_error = 0.0
        self._previous_time_s: Optional[float] = None
        self._last_control = None

    def run_step(
        self,
        *,
        command: VelocitySteeringCommand,
        actual_speed_mps: float,
        sim_time_s: float,
        max_steering_rad: float,
        carla_module: Any,
    ):
        dt_s = (
            0.05
            if self._previous_time_s is None
            else max(1.0e-3, min(0.2, float(sim_time_s) - float(self._previous_time_s)))
        )
        target_speed = max(0.0, float(command.target_speed_mps))
        actual_speed = max(0.0, float(actual_speed_mps))
        speed_error = float(target_speed) - float(actual_speed)
        stop_goal = bool(command.stop_goal_active) or target_speed <= 0.05

        if bool(command.emergency_stop):
            self._integral_error = 0.0
            throttle = 0.0
            brake = 1.0
            reason = "velocity_adapter_emergency_stop"
        elif bool(stop_goal):
            self._integral_error = 0.0
            throttle = 0.0
            if actual_speed <= float(self.stop_hold_speed_mps):
                brake = float(self.max_stop_brake)
            else:
                brake = min(
                    float(self.max_stop_brake),
                    max(0.05, float(self.kp) * actual_speed),
                )
            reason = "velocity_adapter_stop"
        elif speed_error > float(self.speed_deadband_mps):
            self._integral_error = min(
                float(self.integral_limit),
                max(
                    0.0,
                    float(self._integral_error) + float(speed_error) * dt_s,
                ),
            )
            throttle = min(
                float(self.max_throttle),
                float(self.cruise_feedforward)
                + float(self.kp) * float(speed_error)
                + float(self.ki) * float(self._integral_error),
            )
            brake = 0.0
            reason = "velocity_adapter_accelerate"
        elif speed_error < -float(self.brake_activation_error_mps):
            self._integral_error *= 0.5
            throttle = 0.0
            brake = min(
                float(self.max_tracking_brake),
                float(self.kp)
                * (abs(float(speed_error)) - float(self.brake_activation_error_mps)),
            )
            reason = "velocity_adapter_tracking_brake"
        else:
            self._integral_error *= 0.95
            throttle = (
                float(self.cruise_feedforward)
                if speed_error >= -float(self.speed_deadband_mps)
                else 0.0
            )
            brake = 0.0
            reason = "velocity_adapter_cruise_or_coast"

        steering_limit_rad = max(1.0e-6, abs(float(max_steering_rad)))
        desired_steering_rad = (
            0.0
            if bool(command.emergency_stop) or bool(stop_goal)
            else min(
                float(steering_limit_rad),
                max(
                    -float(steering_limit_rad),
                    float(command.target_steering_rad),
                ),
            )
        )
        if self._last_control is not None:
            if bool(command.emergency_stop):
                throttle = 0.0
                brake = 1.0
            else:
                throttle = self._limit_delta(
                    throttle,
                    float(getattr(self._last_control, "throttle", 0.0)),
                    float(self.max_throttle_delta_rate_per_s) * float(dt_s),
                )
                brake = self._limit_delta(
                    brake,
                    float(getattr(self._last_control, "brake", 0.0)),
                    float(self.max_brake_delta_rate_per_s) * float(dt_s),
                )
            previous_steering_rad = (
                float(getattr(self._last_control, "steer", 0.0))
                * float(steering_limit_rad)
            )
            steering_rate_rad_s = (
                float(self.emergency_steering_decay_rate_rad_s)
                if bool(command.emergency_stop)
                else float(self.stop_steering_decay_rate_rad_s)
                if bool(stop_goal)
                else float(self.max_steering_rate_rad_s)
            )
            desired_steering_rad = self._limit_delta(
                desired_steering_rad,
                previous_steering_rad,
                float(steering_rate_rad_s) * float(dt_s),
            )
        steer = min(
            1.0,
            max(
                -1.0,
                float(desired_steering_rad) / float(steering_limit_rad),
            ),
        )
        if brake > 1.0e-6:
            throttle = 0.0
        control = carla_module.VehicleControl(
            throttle=float(throttle),
            brake=float(brake),
            steer=float(steer),
        )
        self._last_control = control
        self._previous_time_s = float(sim_time_s)
        return control, str(reason)

    @staticmethod
    def _limit_delta(value: float, previous: float, maximum_delta: float) -> float:
        delta = max(
            -float(maximum_delta),
            min(float(maximum_delta), float(value) - float(previous)),
        )
        return float(previous) + float(delta)
