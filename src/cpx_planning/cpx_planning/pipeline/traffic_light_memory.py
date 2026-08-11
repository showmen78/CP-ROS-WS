"""Temporal traffic-light state resolution for planning inputs."""

from __future__ import annotations

from typing import Mapping


class TrafficLightMemory:
    """Debounce raw signal readings before scenario selection.

    This component is the single owner of red/yellow/green temporal memory.
    The scenario FSM receives its resolved state and must not add another
    signal debounce window.
    """

    def __init__(
        self,
        *,
        hold_unknown_s: float = 0.8,
        hold_green_unknown_s: float = 0.2,
        green_confirm_s: float = 0.0,
        hold_stop_unknown_until_green: bool = False,
    ) -> None:
        self.hold_unknown_s = max(0.0, float(hold_unknown_s))
        self.hold_green_unknown_s = max(0.0, float(hold_green_unknown_s))
        self.green_confirm_s = max(0.0, float(green_confirm_s))
        self.hold_stop_unknown_until_green = bool(
            hold_stop_unknown_until_green
        )
        self._last_stop_state = "unknown"
        self._last_stop_target: dict[str, object] | None = None
        self._hold_until_s = -float("inf")
        self._green_since_s: float | None = None

    def update(
        self,
        *,
        state: str,
        stop_target: Mapping[str, object] | None,
        sim_time_s: float,
    ) -> tuple[str, dict[str, object] | None, str]:
        normalized_state = str(state or "unknown").strip().lower()
        reason = ""
        if normalized_state in {"red", "yellow"}:
            self._green_since_s = None
            self._last_stop_state = str(normalized_state)
            self._last_stop_target = (
                dict(stop_target)
                if isinstance(stop_target, Mapping)
                else None
            )
            self._hold_until_s = float(sim_time_s) + float(self.hold_unknown_s)
            return str(normalized_state), self._last_stop_target, "raw_stop"

        if normalized_state == "green":
            if self._green_since_s is None:
                self._green_since_s = float(sim_time_s)
            if (
                self._last_stop_state in {"red", "yellow"}
                and float(sim_time_s) - float(self._green_since_s)
                < float(self.green_confirm_s)
            ):
                self._hold_until_s = max(
                    float(self._hold_until_s),
                    float(sim_time_s) + float(self.hold_unknown_s),
                )
                return (
                    str(self._last_stop_state),
                    self._last_stop_target,
                    "traffic_memory_wait_green_confirm",
                )
            self._last_stop_state = "green"
            self._last_stop_target = None
            self._hold_until_s = (
                float(sim_time_s) + float(self.hold_green_unknown_s)
            )
            reason = (
                "traffic_memory_green_release"
                if float(self.green_confirm_s) > 0.0
                else ""
            )
            return "green", None, reason

        if (
            normalized_state == "unknown"
            and float(sim_time_s) <= float(self._hold_until_s)
            and self._last_stop_state in {"red", "yellow"}
        ):
            reason = f"traffic_memory_hold_{self._last_stop_state}"
            return str(self._last_stop_state), self._last_stop_target, reason

        if (
            normalized_state == "unknown"
            and bool(self.hold_stop_unknown_until_green)
            and self._last_stop_state in {"red", "yellow"}
            and self._last_stop_target is not None
        ):
            return (
                str(self._last_stop_state),
                self._last_stop_target,
                f"traffic_memory_fail_safe_hold_{self._last_stop_state}_until_green",
            )

        if (
            normalized_state == "unknown"
            and float(sim_time_s) <= float(self._hold_until_s)
            and self._last_stop_state == "green"
        ):
            return "green", None, "traffic_memory_hold_green"

        if normalized_state == "unknown":
            self._green_since_s = None
        return str(normalized_state), None, reason
