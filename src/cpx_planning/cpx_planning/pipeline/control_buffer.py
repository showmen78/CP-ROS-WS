"""MPC control sequence buffer for CARLA tick-rate execution."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple


class MPCControlBuffer:
    """Reuse the latest MPC control sequence between slower replans."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        replan_period_s: float = 0.25,
        max_reuse_s: float = 0.35,
        max_reference_anchor_jump_m: float = 0.75,
    ) -> None:
        self.enabled = bool(enabled)
        self.replan_period_s = max(0.0, float(replan_period_s))
        self.max_reuse_s = max(0.0, float(max_reuse_s))
        self.max_reference_anchor_jump_m = max(
            0.0, float(max_reference_anchor_jump_m)
        )
        self._plan_time_s: Optional[float] = None
        self._dt_s = 0.05
        self._sequence: List[Tuple[float, float]] = []
        self._context_key = ""
        self._reference_anchor_xy: Optional[Tuple[float, float]] = None
        self._last_reason = "control_buffer_empty"
        self._previous_speed_error_mps: Optional[float] = None

    def should_replan(
        self,
        *,
        sim_time_s: float,
        force_replan: bool = False,
        context_key: str = "",
        reference_anchor_xy: Optional[Tuple[float, float]] = None,
        ego_speed_mps: Optional[float] = None,
        target_speed_mps: Optional[float] = None,
        speed_error_crossing_deadband_mps: float = 0.15,
    ) -> bool:
        longitudinal_reason = self._longitudinal_replan_reason(
            ego_speed_mps=ego_speed_mps,
            target_speed_mps=target_speed_mps,
            deadband_mps=float(speed_error_crossing_deadband_mps),
        )
        if not self.enabled:
            self._last_reason = "control_buffer_disabled"
            return True
        if bool(force_replan):
            self._last_reason = "control_buffer_force_replan"
            return True
        if longitudinal_reason:
            self._last_reason = str(longitudinal_reason)
            return True
        if self._plan_time_s is None or not self._sequence:
            self._last_reason = "control_buffer_empty"
            return True
        if str(context_key or "") != str(self._context_key or ""):
            self._last_reason = "control_buffer_context_changed"
            return True
        if self._reference_anchor_jump_exceeded(reference_anchor_xy):
            self._last_reason = "control_buffer_reference_anchor_jump"
            return True
        age_s = max(0.0, float(sim_time_s) - float(self._plan_time_s))
        if age_s >= float(self.replan_period_s):
            self._last_reason = "control_buffer_replan_period_elapsed"
            return True
        if age_s > float(self.max_reuse_s):
            self._last_reason = "control_buffer_max_reuse_elapsed"
            return True
        self._last_reason = "control_buffer_reuse"
        return False

    def update_from_solution(
        self,
        *,
        u_solution: Any,
        plan_time_s: float,
        dt_s: float,
        context_key: str = "",
        reference_anchor_xy: Optional[Tuple[float, float]] = None,
    ) -> None:
        self._plan_time_s = float(plan_time_s)
        self._dt_s = max(1.0e-3, float(dt_s))
        sequence: List[Tuple[float, float]] = []
        try:
            step_count = int(len(u_solution))
        except Exception:
            step_count = 0
        for index in range(step_count):
            try:
                accel = float(u_solution[index, 0])
                steer = float(u_solution[index, 1])
            except Exception:
                try:
                    accel = float(u_solution[index][0])
                    steer = float(u_solution[index][1])
                except Exception:
                    continue
            sequence.append((float(accel), float(steer)))
        self._sequence = sequence
        self._context_key = str(context_key or "")
        self._reference_anchor_xy = self._finite_anchor(reference_anchor_xy)
        self._last_reason = "control_buffer_updated"

    def sample(
        self,
        *,
        sim_time_s: float,
        context_key: str = "",
        reference_anchor_xy: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float, str]]:
        if self._plan_time_s is None or not self._sequence:
            self._last_reason = "control_buffer_empty"
            return None
        age_s = max(0.0, float(sim_time_s) - float(self._plan_time_s))
        if age_s > float(self.max_reuse_s):
            self._last_reason = "control_buffer_max_reuse_elapsed"
            return None
        if str(context_key or "") != str(self._context_key or ""):
            self._last_reason = "control_buffer_context_changed"
            return None
        if self._reference_anchor_jump_exceeded(reference_anchor_xy):
            self._last_reason = "control_buffer_reference_anchor_jump"
            return None
        index = int(round(age_s / max(1.0e-3, float(self._dt_s))))
        index = max(0, min(index, len(self._sequence) - 1))
        accel, steer = self._sequence[index]
        self._last_reason = f"control_buffer_reuse_step:{int(index)}"
        return float(accel), float(steer), str(self._last_reason)

    def reset(self, *, reason: str = "control_buffer_reset") -> None:
        self._plan_time_s = None
        self._sequence = []
        self._context_key = ""
        self._reference_anchor_xy = None
        self._previous_speed_error_mps = None
        self._last_reason = str(reason)

    def _longitudinal_replan_reason(
        self,
        *,
        ego_speed_mps: Optional[float],
        target_speed_mps: Optional[float],
        deadband_mps: float,
    ) -> str:
        try:
            speed_error_mps = float(target_speed_mps) - float(ego_speed_mps)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(speed_error_mps):
            return ""
        previous = self._previous_speed_error_mps
        self._previous_speed_error_mps = float(speed_error_mps)
        if previous is None or self._plan_time_s is None or not self._sequence:
            return ""
        deadband = max(0.0, float(deadband_mps))
        crossed_target = bool(
            (float(previous) > deadband and float(speed_error_mps) <= 0.0)
            or (float(previous) < -deadband and float(speed_error_mps) >= 0.0)
        )
        entered_target_band = bool(
            abs(float(speed_error_mps)) <= deadband
            and abs(float(previous)) > deadband
        )
        if crossed_target:
            return "control_buffer_speed_target_crossed"
        if entered_target_band:
            return "control_buffer_speed_target_band_entered"
        return ""

    def _reference_anchor_jump_exceeded(
        self,
        reference_anchor_xy: Optional[Tuple[float, float]],
    ) -> bool:
        current = self._finite_anchor(reference_anchor_xy)
        previous = self._reference_anchor_xy
        if current is None or previous is None:
            return bool(current is not None or previous is not None)
        return bool(
            math.hypot(
                float(current[0]) - float(previous[0]),
                float(current[1]) - float(previous[1]),
            )
            > float(self.max_reference_anchor_jump_m)
        )

    @staticmethod
    def _finite_anchor(
        value: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        if value is None:
            return None
        try:
            x_m = float(value[0])
            y_m = float(value[1])
        except (IndexError, TypeError, ValueError):
            return None
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            return None
        return float(x_m), float(y_m)

    @property
    def last_reason(self) -> str:
        return str(self._last_reason)

    @property
    def buffered_step_count(self) -> int:
        return len(self._sequence)
