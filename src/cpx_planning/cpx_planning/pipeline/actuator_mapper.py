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
        # The proportional boost above always leaves a residual
        # steady-state error at whatever throttle happens to cancel drag --
        # confirmed via telemetry: applied_throttle plateaus dead flat while
        # accel_cmd stays positive and measured accel settles near zero,
        # regardless of how high speed_error_gain is raised. An integral
        # term keeps accumulating extra throttle for as long as the error
        # persists, so it can close that residual instead of asymptoting
        # to it. Disabled by default (gain 0.0) so existing tuned scenarios
        # are unaffected.
        self.speed_error_integral_gain = max(
            0.0, float(cfg.get("actuator_speed_error_integral_gain", 0.0))
        )
        self.max_speed_error_integral_boost = max(
            0.0,
            float(cfg.get("actuator_max_speed_error_integral_boost", 0.15)),
        )
        self._speed_error_integral = 0.0
        self._integral_previous_time_s = None
        self.feedforward_min_target_mps = max(
            0.0, float(cfg.get("actuator_feedforward_min_target_mps", 0.3))
        )
        self.speed_deadband_mps = max(
            0.0, float(cfg.get("actuator_speed_deadband_mps", 0.15))
        )
        # Crossing the deadband used to cut throttle straight to zero in
        # one tick -- a fine, unnoticeable step at low cruise speeds, but
        # a real disturbance at higher ones: confirmed via telemetry at
        # 20 m/s cruise, IDM's catch-up acceleration briefly overshot the
        # target by ~0.2 m/s, throttle dropped 0.75->0.0 in a single
        # step, and aerodynamic drag (much larger at 20 m/s than at
        # 11 m/s) coasted the car down by over 2 m/s before it recovered.
        # Taper smoothly over this many m/s of overspeed instead of
        # stepping, so the same crossing costs a gentle roll-off rather
        # than a hard cut.
        self.overspeed_taper_mps = max(
            0.0, float(cfg.get("actuator_overspeed_taper_mps", 1.0))
        )
        # accel<0 used to route straight to the brake branch with zero
        # tolerance, so MPC's commanded acceleration merely flickering
        # across zero (reference-tracking noise while holding near
        # target, observed magnitudes ~+-0.01 m/s^2) chattered between
        # full brake-path throttle=0 and normal throttle every other
        # tick -- confirmed via telemetry as the real cause behind an
        # apparent 20 m/s "overshoot" that kept recurring regardless of
        # the overspeed taper above (that taper never even ran on the
        # ticks routed to the brake branch). Treat anything smaller in
        # magnitude than this as noise, not a real brake request.
        self.brake_deadzone_mps2 = max(
            0.0, float(cfg.get("actuator_brake_deadzone_mps2", 0.1))
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
        timestamp_s: float | None = None,
    ) -> ActuatorCommand:
        accel = float(acceleration_mps2)
        max_accel = max(1.0e-6, float(max_acceleration_mps2))
        max_brake = max(1.0e-6, abs(float(min_acceleration_mps2)))
        speed_error = float(target_speed_mps) - float(ego_speed_mps)
        # accel<0 used to route straight to the brake branch with no
        # deadzone, so a commanded acceleration that is really just
        # reference-tracking noise around zero (observed: flickering
        # between roughly -0.006 and +0.013 m/s^2 tick to tick while MPC
        # holds speed near target) got treated as "brake now, zero
        # throttle" on the negative ticks and "normal throttle" on the
        # positive ones -- a real chattering discontinuity, confirmed via
        # telemetry as the actual cause of the 20 m/s overshoot/coast-down
        # cycle (the overspeed taper above never even ran on those ticks,
        # since this check fires first). Clamp anything inside the
        # deadzone to 0 instead of branching to the brake path on it.
        if accel < 0.0 and accel < -float(self.brake_deadzone_mps2):
            self._reset_speed_error_integral(timestamp_s=timestamp_s)
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
        if accel < 0.0:
            # Inside the deadzone: near-zero noise, not a real brake
            # request. Treat it as "hold," not "accelerate" or "brake".
            accel = 0.0
        if bool(stop_goal_active) or float(target_speed_mps) <= 0.0:
            self._reset_speed_error_integral(timestamp_s=timestamp_s)
            return ActuatorCommand(throttle=0.0, brake=0.0)
        if float(speed_error) < -float(self.speed_deadband_mps):
            self._reset_speed_error_integral(timestamp_s=timestamp_s)
            overspeed_amount_mps = -float(speed_error) - float(self.speed_deadband_mps)
            taper_ratio = max(
                0.0,
                1.0
                - float(overspeed_amount_mps) / max(1.0e-3, float(self.overspeed_taper_mps)),
            )
            if taper_ratio <= 0.0:
                return ActuatorCommand(throttle=0.0, brake=0.0)
            # Roll the base (non-feedforward) throttle off smoothly across
            # the taper band instead of stepping it to zero; no
            # feedforward/speed-error boost here since ego is already
            # past target, not building toward it.
            tapered_throttle = min(1.0, max(0.0, accel / max_accel)) * float(taper_ratio)
            return ActuatorCommand(
                throttle=min(float(self.cruise_max_throttle), max(0.0, tapered_throttle)),
                brake=0.0,
            )

        throttle = min(1.0, max(0.0, accel / max_accel))
        if self.enabled and float(target_speed_mps) >= self.feedforward_min_target_mps:
            speed_error = max(0.0, float(speed_error))
            speed_boost = min(
                float(self.max_speed_error_boost),
                float(self.speed_error_gain) * speed_error,
            )
            integral_boost = self._integrate_speed_error(
                speed_error=float(speed_error), timestamp_s=timestamp_s
            )
            throttle += (
                float(self.cruise_feedforward)
                + float(speed_boost)
                + float(integral_boost)
            )
        return ActuatorCommand(
            throttle=min(
                float(self.cruise_max_throttle),
                max(0.0, throttle),
            ),
            brake=0.0,
        )

    def _reset_speed_error_integral(self, *, timestamp_s: float | None) -> None:
        self._speed_error_integral = 0.0
        self._integral_previous_time_s = timestamp_s

    def _integrate_speed_error(
        self, *, speed_error: float, timestamp_s: float | None
    ) -> float:
        if timestamp_s is None or float(self.speed_error_integral_gain) <= 0.0:
            self._integral_previous_time_s = timestamp_s
            return 0.0
        if self._integral_previous_time_s is not None:
            dt_s = float(timestamp_s) - float(self._integral_previous_time_s)
            if 1.0e-3 < dt_s <= 0.5:
                # Anti-windup: cap the accumulator itself at the point
                # where gain*accumulator already saturates the boost, so it
                # can't keep growing unboundedly while stuck at a plateau
                # and then overshoot once conditions change.
                integral_ceiling = float(self.max_speed_error_integral_boost) / float(
                    self.speed_error_integral_gain
                )
                self._speed_error_integral = max(
                    0.0,
                    min(
                        float(integral_ceiling),
                        float(self._speed_error_integral)
                        + float(speed_error) * dt_s,
                    ),
                )
        self._integral_previous_time_s = timestamp_s
        return min(
            float(self.max_speed_error_integral_boost),
            float(self.speed_error_integral_gain) * float(self._speed_error_integral),
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
