"""Scenario-level finite state machine for the CP-X OpenCDA bridge.

The manager owns high-level driving context that should not be scattered
between behavior, reference generation, and control guards. It follows the
same broad separation used by production planners: scenario selection first,
then behavior path and velocity generation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Optional


LANE_FOLLOW = "LANE_FOLLOW"
TRAFFIC_LIGHT_APPROACH = "TRAFFIC_LIGHT_APPROACH"
TRAFFIC_LIGHT_STOP = "TRAFFIC_LIGHT_STOP"
INTERSECTION_TURN = "INTERSECTION_TURN"
CREEP = "CREEP"
RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class CPXScenarioDecision:
    state: str
    behavior_signal_state: str
    behavior_stop_target: Optional[Mapping[str, object]]
    behavior_override_decision: str = ""
    behavior_override_lc_state: str = ""
    speed_cap_mps: Optional[float] = None
    stop_goal_active: bool = False
    traffic_stop_forward_m: float = 0.0
    traffic_stop_commit_distance_m: float = 0.0
    reason: str = ""
    turn_direction: str = ""
    turn_latched: bool = False

    def as_debug_fields(self) -> dict[str, object]:
        return {
            "scenario_fsm_state": str(self.state),
            "scenario_fsm_reason": str(self.reason),
            "scenario_behavior_signal_state": str(self.behavior_signal_state),
            "scenario_behavior_override_decision": str(self.behavior_override_decision),
            "scenario_speed_cap_mps": (
                "" if self.speed_cap_mps is None else float(self.speed_cap_mps)
            ),
            "scenario_stop_goal_active": bool(self.stop_goal_active),
            "scenario_turn_direction": str(self.turn_direction),
            "scenario_turn_latched": bool(self.turn_latched),
        }


class CPXScenarioManager:
    """Manage traffic-light and intersection-turn scenario states."""

    def __init__(self, config: Optional[Mapping[str, object]] = None) -> None:
        cfg = dict(config or {})
        self.target_speed_mps = float(cfg.get("target_speed_mps", 8.0))
        self.commit_decel_mps2 = max(
            0.1, float(cfg.get("traffic_stop_commit_decel_mps2", 2.0))
        )
        self.commit_buffer_m = max(
            0.0, float(cfg.get("traffic_stop_commit_buffer_m", 4.0))
        )
        self.min_commit_distance_m = max(
            0.0, float(cfg.get("traffic_stop_min_commit_distance_m", 10.0))
        )
        self.approach_slow_distance_m = max(
            self.min_commit_distance_m,
            float(cfg.get("traffic_stop_approach_slow_distance_m", 22.0)),
        )
        self.approach_far_speed_cap_mps = max(
            0.1,
            float(cfg.get("traffic_stop_approach_far_speed_cap_mps", self.target_speed_mps)),
        )
        self.approach_near_speed_cap_mps = max(
            0.1, float(cfg.get("traffic_stop_approach_near_speed_cap_mps", 2.5))
        )
        self.turn_speed_cap_mps = max(
            0.1, float(cfg.get("full_intersection_turn_speed_cap_mps", 2.2))
        )
        self.creep_speed_cap_mps = max(
            0.1, float(cfg.get("scenario_creep_speed_cap_mps", 0.9))
        )
        self.green_release_s = max(
            0.0, float(cfg.get("scenario_green_release_s", 0.8))
        )
        self.turn_exit_hold_s = max(
            0.0, float(cfg.get("scenario_turn_exit_hold_s", 0.6))
        )
        self._state = LANE_FOLLOW
        self._turn_direction = ""
        self._release_until_s = -float("inf")
        self._turn_latch_until_s = -float("inf")

    @property
    def state(self) -> str:
        return str(self._state)

    def reset(self) -> None:
        self._state = LANE_FOLLOW
        self._turn_direction = ""
        self._release_until_s = -float("inf")
        self._turn_latch_until_s = -float("inf")

    def update(
        self,
        *,
        traffic_state: str,
        stop_target: Optional[Mapping[str, object]],
        stop_forward_m: float,
        stop_target_reliable: bool,
        ego_speed_mps: float,
        ego_in_junction: bool,
        current_road_option: str,
        next_macro_maneuver: str,
        sim_time_s: float,
        reference_geometry_bad: bool = False,
        collision_hazard: bool = False,
    ) -> CPXScenarioDecision:
        if bool(collision_hazard):
            self._state = RECOVERY
            return CPXScenarioDecision(
                state=RECOVERY,
                behavior_signal_state="unknown",
                behavior_stop_target=None,
                behavior_override_decision="emergency_brake",
                behavior_override_lc_state="LANE_KEEP",
                speed_cap_mps=0.0,
                stop_goal_active=True,
                traffic_stop_forward_m=float(stop_forward_m),
                traffic_stop_commit_distance_m=self._commit_distance(float(ego_speed_mps)),
                reason="collision_hazard_recovery",
            )

        signal_decision = self._traffic_light_decision(
            traffic_state=str(traffic_state),
            stop_target=stop_target,
            stop_forward_m=float(stop_forward_m),
            stop_target_reliable=bool(stop_target_reliable),
            ego_speed_mps=float(ego_speed_mps),
            ego_in_junction=bool(ego_in_junction),
            sim_time_s=float(sim_time_s),
        )
        if signal_decision is not None:
            return signal_decision

        turn_decision = self._intersection_turn_decision(
            ego_in_junction=bool(ego_in_junction),
            current_road_option=str(current_road_option),
            next_macro_maneuver=str(next_macro_maneuver),
            sim_time_s=float(sim_time_s),
            reference_geometry_bad=bool(reference_geometry_bad),
        )
        if turn_decision is not None:
            return turn_decision

        self._state = LANE_FOLLOW
        self._turn_direction = ""
        return CPXScenarioDecision(
            state=LANE_FOLLOW,
            behavior_signal_state="unknown",
            behavior_stop_target=None,
            speed_cap_mps=self.target_speed_mps,
            reason="lane_follow_default",
        )

    def _traffic_light_decision(
        self,
        *,
        traffic_state: str,
        stop_target: Optional[Mapping[str, object]],
        stop_forward_m: float,
        stop_target_reliable: bool,
        ego_speed_mps: float,
        ego_in_junction: bool,
        sim_time_s: float,
    ) -> Optional[CPXScenarioDecision]:
        state = str(traffic_state or "unknown").strip().lower()
        commit_distance_m = self._commit_distance(float(ego_speed_mps))
        if state in {"green"}:
            if self._state == TRAFFIC_LIGHT_STOP:
                self._release_until_s = float(sim_time_s) + float(self.green_release_s)
            if float(sim_time_s) <= float(self._release_until_s):
                self._state = LANE_FOLLOW
                return CPXScenarioDecision(
                    state=LANE_FOLLOW,
                    behavior_signal_state="green",
                    behavior_stop_target=None,
                    speed_cap_mps=min(self.target_speed_mps, self.approach_near_speed_cap_mps),
                    traffic_stop_forward_m=float(stop_forward_m),
                    traffic_stop_commit_distance_m=float(commit_distance_m),
                    reason="traffic_light_green_release",
                )
            return None
        if state not in {"red", "yellow"}:
            return None
        if (
            bool(stop_target_reliable)
            and not bool(ego_in_junction)
            and float(stop_forward_m) > float(commit_distance_m)
        ):
            self._state = TRAFFIC_LIGHT_APPROACH
            speed_cap = self.approach_far_speed_cap_mps
            if float(stop_forward_m) <= float(self.approach_slow_distance_m):
                speed_cap = min(speed_cap, self.approach_near_speed_cap_mps)
            return CPXScenarioDecision(
                state=TRAFFIC_LIGHT_APPROACH,
                behavior_signal_state="unknown",
                behavior_stop_target=None,
                speed_cap_mps=min(self.target_speed_mps, float(speed_cap)),
                stop_goal_active=False,
                traffic_stop_forward_m=float(stop_forward_m),
                traffic_stop_commit_distance_m=float(commit_distance_m),
                reason=(
                    "traffic_light_approach:"
                    f"signal={state}:stop_f={float(stop_forward_m):.2f}:"
                    f"commit={float(commit_distance_m):.2f}"
                ),
            )
        self._state = TRAFFIC_LIGHT_STOP
        return CPXScenarioDecision(
            state=TRAFFIC_LIGHT_STOP,
            behavior_signal_state=str(state),
            behavior_stop_target=dict(stop_target or {}) if isinstance(stop_target, Mapping) else None,
            behavior_override_decision="stop_at_intersection",
            behavior_override_lc_state="LANE_KEEP",
            speed_cap_mps=0.0,
            stop_goal_active=True,
            traffic_stop_forward_m=float(stop_forward_m),
            traffic_stop_commit_distance_m=float(commit_distance_m),
            reason=(
                "traffic_light_stop_commit:"
                f"signal={state}:stop_f={float(stop_forward_m):.2f}:"
                f"commit={float(commit_distance_m):.2f}"
            ),
        )

    def _intersection_turn_decision(
        self,
        *,
        ego_in_junction: bool,
        current_road_option: str,
        next_macro_maneuver: str,
        sim_time_s: float,
        reference_geometry_bad: bool,
    ) -> Optional[CPXScenarioDecision]:
        direction = self._turn_direction_from_route_option(
            current_road_option=str(current_road_option)
        )
        if not direction and bool(ego_in_junction):
            direction = self._turn_direction_from_macro(
                next_macro_maneuver=str(next_macro_maneuver)
            )
        if direction:
            self._turn_direction = str(direction)
            self._turn_latch_until_s = float(sim_time_s) + float(self.turn_exit_hold_s)
        elif (
            self._state == INTERSECTION_TURN
            and (bool(ego_in_junction) or float(sim_time_s) <= float(self._turn_latch_until_s))
            and self._turn_direction
        ):
            direction = str(self._turn_direction)
        else:
            return None

        if bool(reference_geometry_bad):
            self._state = CREEP
            return CPXScenarioDecision(
                state=CREEP,
                behavior_signal_state="unknown",
                behavior_stop_target=None,
                behavior_override_decision=f"intersection_turn_{direction}",
                behavior_override_lc_state=f"INTERSECTION_TURN_{direction.upper()}",
                speed_cap_mps=float(self.creep_speed_cap_mps),
                reason="turn_geometry_bad_creep",
                turn_direction=str(direction),
                turn_latched=True,
            )

        self._state = INTERSECTION_TURN
        return CPXScenarioDecision(
            state=INTERSECTION_TURN,
            behavior_signal_state="unknown",
            behavior_stop_target=None,
            behavior_override_decision=f"intersection_turn_{direction}",
            behavior_override_lc_state=f"INTERSECTION_TURN_{direction.upper()}",
            speed_cap_mps=float(self.turn_speed_cap_mps),
            reason=(
                "intersection_turn_latched"
                if not self._turn_direction_from_route_option(current_road_option)
                else f"route_option_turn:{current_road_option}"
            ),
            turn_direction=str(direction),
            turn_latched=True,
        )

    def _commit_distance(self, ego_speed_mps: float) -> float:
        braking_distance_m = (float(ego_speed_mps) ** 2) / (
            2.0 * float(self.commit_decel_mps2)
        )
        return max(
            float(self.min_commit_distance_m),
            float(braking_distance_m) + float(self.commit_buffer_m),
        )

    @staticmethod
    def _turn_direction_from_route_option(current_road_option: str) -> str:
        option = str(current_road_option or "").strip().upper()
        if option in {"LEFT", "RIGHT"}:
            return option.lower()
        return ""

    @staticmethod
    def _turn_direction_from_macro(next_macro_maneuver: str) -> str:
        maneuver = str(next_macro_maneuver or "").strip().lower()
        if maneuver in {"left", "right"}:
            return str(maneuver)
        return ""
