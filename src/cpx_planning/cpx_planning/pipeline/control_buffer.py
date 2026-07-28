"""MPC control sequence buffer for CARLA tick-rate execution."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


class MPCControlBuffer:
    """Reuse the latest MPC control sequence between slower replans."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        replan_period_s: float = 0.25,
        max_reuse_s: float = 0.35,
    ) -> None:
        self.enabled = bool(enabled)
        self.replan_period_s = max(0.0, float(replan_period_s))
        self.max_reuse_s = max(0.0, float(max_reuse_s))
        self._plan_time_s: Optional[float] = None
        self._dt_s = 0.05
        self._sequence: List[Tuple[float, float]] = []
        self._last_reason = "control_buffer_empty"

    def should_replan(self, *, sim_time_s: float, force_replan: bool = False) -> bool:
        if not self.enabled:
            self._last_reason = "control_buffer_disabled"
            return True
        if bool(force_replan):
            self._last_reason = "control_buffer_force_replan"
            return True
        if self._plan_time_s is None or not self._sequence:
            self._last_reason = "control_buffer_empty"
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
        self._last_reason = "control_buffer_updated"

    def sample(self, *, sim_time_s: float) -> Optional[Tuple[float, float, str]]:
        if self._plan_time_s is None or not self._sequence:
            self._last_reason = "control_buffer_empty"
            return None
        age_s = max(0.0, float(sim_time_s) - float(self._plan_time_s))
        if age_s > float(self.max_reuse_s):
            self._last_reason = "control_buffer_max_reuse_elapsed"
            return None
        index = int(round(age_s / max(1.0e-3, float(self._dt_s))))
        index = max(0, min(index, len(self._sequence) - 1))
        accel, steer = self._sequence[index]
        self._last_reason = f"control_buffer_reuse_step:{int(index)}"
        return float(accel), float(steer), str(self._last_reason)

    def reset(self, *, reason: str = "control_buffer_reset") -> None:
        self._plan_time_s = None
        self._sequence = []
        self._last_reason = str(reason)

    @property
    def last_reason(self) -> str:
        return str(self._last_reason)

    @property
    def buffered_step_count(self) -> int:
        return len(self._sequence)
