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
        max_predicted_speed_error_mps: float = 0.75,
        max_target_speed_jump_mps: float = 1.0,
    ) -> None:
        self.enabled = bool(enabled)
        self.replan_period_s = max(0.0, float(replan_period_s))
        self.max_reuse_s = max(0.0, float(max_reuse_s))
        self.max_reference_anchor_jump_m = max(
            0.0, float(max_reference_anchor_jump_m)
        )
        self.max_predicted_speed_error_mps = max(
            0.0, float(max_predicted_speed_error_mps)
        )
        self.max_target_speed_jump_mps = max(
            0.0, float(max_target_speed_jump_mps)
        )
        self._plan_time_s: Optional[float] = None
        self._dt_s = 0.05
        self._sequence: List[Tuple[float, float]] = []
        self._predicted_speed_sequence_mps: List[float] = []
        self._plan_target_speed_mps: Optional[float] = None
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
        if self._target_speed_jump_exceeded(target_speed_mps):
            self._last_reason = "control_buffer_target_speed_jumped"
            return True
        age_s = max(0.0, float(sim_time_s) - float(self._plan_time_s))
        if age_s >= float(self.replan_period_s):
            self._last_reason = "control_buffer_replan_period_elapsed"
            return True
        if age_s > float(self.max_reuse_s):
            self._last_reason = "control_buffer_max_reuse_elapsed"
            return True
        if self._predicted_speed_error_exceeded(
            age_s=float(age_s),
            ego_speed_mps=ego_speed_mps,
        ):
            self._last_reason = "control_buffer_predicted_speed_diverged"
            return True
        self._last_reason = "control_buffer_reuse"
        return False

    def _target_speed_jump_exceeded(
        self,
        target_speed_mps: Optional[float],
    ) -> bool:
        """Detect the requested cruise speed itself jumping between ticks.

        This is distinct from ``_longitudinal_replan_reason``'s crossing/
        deadband check, which only fires when ego's speed error relative to
        target changes sign or enters a narrow band -- it stays silent
        whenever ego was already on the same side of a moving target both
        before and after a large jump (e.g. target steps from ~2 m/s to
        ~5 m/s while ego, already below both, never crosses anything). A
        buffered plan solved against the old target has no reason to still
        be valid once the target itself has moved this much.
        """

        if target_speed_mps is None or self._plan_target_speed_mps is None:
            return False
        try:
            current_target_mps = float(target_speed_mps)
            plan_target_mps = float(self._plan_target_speed_mps)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(current_target_mps) or not math.isfinite(plan_target_mps):
            return False
        return bool(
            abs(current_target_mps - plan_target_mps)
            > float(self.max_target_speed_jump_mps)
        )

    def _predicted_speed_error_exceeded(
        self,
        *,
        age_s: float,
        ego_speed_mps: Optional[float],
    ) -> bool:
        """Detect open-loop drift between the buffered plan and reality.

        ``_sequence``/``_predicted_speed_sequence_mps`` are the *open-loop*
        acceleration and state trajectory from a single MPC solve, replayed
        for up to ``max_reuse_s`` with no feedback in between. If the
        vehicle's actual speed has already drifted away from what that
        solve predicted for "now" -- e.g. real deceleration outrunning the
        plan because of actuator lag or a solve that itself dips low before
        recovering later in its own horizon -- continuing to play back the
        rest of that stale plan compounds the error instead of correcting
        it. Forcing an early replan here is the feedback a pure open-loop
        buffer is missing.
        """

        if ego_speed_mps is None or not self._predicted_speed_sequence_mps:
            return False
        try:
            actual_speed_mps = float(ego_speed_mps)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(actual_speed_mps):
            return False
        predicted_speed_mps = self._predicted_speed_at_age(float(age_s))
        if predicted_speed_mps is None:
            return False
        return bool(
            abs(actual_speed_mps - float(predicted_speed_mps))
            > float(self.max_predicted_speed_error_mps)
        )

    def _predicted_speed_at_age(self, age_s: float) -> Optional[float]:
        if not self._predicted_speed_sequence_mps:
            return None
        index = self._index_for_age(float(age_s))
        if index >= len(self._predicted_speed_sequence_mps):
            return None
        return float(self._predicted_speed_sequence_mps[index])

    def _index_for_age(self, age_s: float) -> int:
        raw_index = int(round(float(age_s) / max(1.0e-3, float(self._dt_s))))
        return max(0, raw_index)

    def update_from_solution(
        self,
        *,
        u_solution: Any,
        plan_time_s: float,
        dt_s: float,
        context_key: str = "",
        reference_anchor_xy: Optional[Tuple[float, float]] = None,
        predicted_speed_sequence_mps: Optional[Any] = None,
        target_speed_mps: Optional[float] = None,
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
        # predicted_speed_sequence_mps is x_solution[:, 2] -- the state
        # trajectory's own velocity prediction, aligned index-for-index
        # with the *state* at each step (one longer than u_solution, whose
        # entry k is the control applied *between* states k and k+1).
        # Reusing the same age-based index as sample()/u_solution here is
        # deliberate: it lets should_replan() ask "is reality still close
        # to what step `sample() is about to use` assumed," not just "close
        # to what step 0 assumed."
        predicted_speeds: List[float] = []
        try:
            for index in range(int(len(predicted_speed_sequence_mps or []))):
                try:
                    predicted_speeds.append(
                        float(predicted_speed_sequence_mps[index])
                    )
                except (TypeError, ValueError, IndexError):
                    break
        except Exception:
            predicted_speeds = []
        self._predicted_speed_sequence_mps = predicted_speeds
        try:
            self._plan_target_speed_mps = (
                None if target_speed_mps is None else float(target_speed_mps)
            )
        except (TypeError, ValueError):
            self._plan_target_speed_mps = None
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
        index = min(self._index_for_age(float(age_s)), len(self._sequence) - 1)
        accel, steer = self._sequence[index]
        self._last_reason = f"control_buffer_reuse_step:{int(index)}"
        return float(accel), float(steer), str(self._last_reason)

    def reset(self, *, reason: str = "control_buffer_reset") -> None:
        self._plan_time_s = None
        self._sequence = []
        self._predicted_speed_sequence_mps = []
        self._plan_target_speed_mps = None
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
