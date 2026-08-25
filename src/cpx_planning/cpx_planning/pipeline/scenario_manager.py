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

from .speed_planner import turn_approach_lookahead_m


LANE_FOLLOW = "LANE_FOLLOW"
TRAFFIC_LIGHT_APPROACH = "TRAFFIC_LIGHT_APPROACH"
TRAFFIC_LIGHT_STOP = "TRAFFIC_LIGHT_STOP"
PREPARE_TURN = "PREPARE_TURN"
INTERSECTION_TURN = "INTERSECTION_TURN"
TURN_EXIT_STABILIZATION = "TURN_EXIT_STABILIZATION"
CREEP = "CREEP"
BOUNDARY_RECOVERY = "BOUNDARY_RECOVERY"
RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class BoundaryRecoveryRequest:
    valid: bool = False
    active: bool = False
    clearance_m: float = float("inf")
    lateral_offset_m: float = 0.0
    heading_error_rad: float = 0.0
    turn_direction: str = ""
    timestamp_s: float = 0.0
    reason: str = ""


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
    boundary_recovery_active: bool = False
    boundary_clearance_m: float = float("inf")
    boundary_lateral_offset_m: float = 0.0
    boundary_heading_error_rad: float = 0.0

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
            "scenario_boundary_recovery_active": bool(
                self.boundary_recovery_active
            ),
            "scenario_boundary_clearance_m": (
                ""
                if not math.isfinite(float(self.boundary_clearance_m))
                else float(self.boundary_clearance_m)
            ),
            "scenario_boundary_lateral_offset_m": float(
                self.boundary_lateral_offset_m
            ),
            "scenario_boundary_heading_error_rad": float(
                self.boundary_heading_error_rad
            ),
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
        self.turn_prepare_speed_cap_mps = max(
            self.turn_speed_cap_mps,
            float(cfg.get("scenario_turn_prepare_speed_cap_mps", 2.8)),
        )
        # Route may advertise the next macro maneuver for the whole remaining
        # route.  PREPARE_TURN is a local scenario state, so it must use the
        # same dynamic preview contract as SpeedPlanner/RouteManager.
        self.turn_prepare_lookahead_m = float(
            turn_approach_lookahead_m(
                cruise_speed_mps=float(self.target_speed_mps),
                config=cfg,
            )
        )
        self.creep_speed_cap_mps = max(
            0.1, float(cfg.get("scenario_creep_speed_cap_mps", 0.9))
        )
        self.boundary_recovery_enabled = bool(
            cfg.get("boundary_recovery_enabled", False)
        )
        self.boundary_recovery_speed_mps = max(
            0.1, float(cfg.get("boundary_recovery_speed_mps", 0.55))
        )
        self.boundary_recovery_release_clearance_m = float(
            cfg.get("boundary_recovery_release_clearance_m", 0.10)
        )
        self.boundary_recovery_release_lateral_m = max(
            0.0, float(cfg.get("boundary_recovery_release_lateral_m", 0.20))
        )
        self.boundary_recovery_release_heading_rad = max(
            0.0,
            float(cfg.get("boundary_recovery_release_heading_rad", 0.0873)),
        )
        self.boundary_recovery_release_frames = max(
            1, int(cfg.get("boundary_recovery_release_frames", 5))
        )
        self.turn_exit_hold_s = max(
            0.0, float(cfg.get("scenario_turn_exit_hold_s", 0.6))
        )
        self.turn_exit_stable_frames = max(
            1, int(cfg.get("scenario_turn_exit_stable_frames", 8))
        )
        self._state = LANE_FOLLOW
        self._turn_direction = ""
        self._turn_latch_until_s = -float("inf")
        self._turn_connector_seen = False
        self._boundary_recovery_stable_frames = 0
        self._turn_exit_stable_frames = 0

    @property
    def state(self) -> str:
        return str(self._state)

    def reset(self) -> None:
        self._state = LANE_FOLLOW
        self._turn_direction = ""
        self._turn_latch_until_s = -float("inf")
        self._turn_connector_seen = False
        self._boundary_recovery_stable_frames = 0
        self._turn_exit_stable_frames = 0

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
        upcoming_turn_direction: str = "",
        upcoming_turn_distance_m: float = float("inf"),
        reference_geometry_bad: bool = False,
        collision_hazard: bool = False,
        turn_exit_alignment_valid: bool = False,
        turn_exit_aligned: bool = True,
        turn_exit_heading_error_rad: float = 0.0,
        turn_exit_lateral_m: float = 0.0,
        boundary_recovery_request: Optional[BoundaryRecoveryRequest] = None,
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

        # Once the connector turn owns lateral motion, keep that ownership
        # through physical exit stabilization.  A downstream traffic light
        # must not replace the turn reference with lane-follow while the ego
        # is still rotating/converging onto the outgoing lane.
        turn_execution_active = self._state in {
            INTERSECTION_TURN,
            TURN_EXIT_STABILIZATION,
            CREEP,
        }
        if bool(turn_execution_active):
            turn_decision = self._intersection_turn_decision(
                ego_in_junction=bool(ego_in_junction),
                current_road_option=str(current_road_option),
                next_macro_maneuver=str(next_macro_maneuver),
                sim_time_s=float(sim_time_s),
                upcoming_turn_direction=str(upcoming_turn_direction),
                upcoming_turn_distance_m=float(upcoming_turn_distance_m),
                reference_geometry_bad=bool(reference_geometry_bad),
                turn_exit_alignment_valid=bool(turn_exit_alignment_valid),
                turn_exit_aligned=bool(turn_exit_aligned),
                turn_exit_heading_error_rad=float(turn_exit_heading_error_rad),
                turn_exit_lateral_m=float(turn_exit_lateral_m),
            )
            if turn_decision is not None:
                return turn_decision
            # Physical exit convergence completed in this update.  Publish a
            # neutral transition frame and let traffic-light/lane-follow
            # ownership resume on the next tick; otherwise a connector-side
            # route hint can immediately re-enter PREPARE_TURN.
            return CPXScenarioDecision(
                state=LANE_FOLLOW,
                behavior_signal_state="unknown",
                behavior_stop_target=None,
                speed_cap_mps=float(self.target_speed_mps),
                reason="turn_exit_stabilized",
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

        if bool(self.boundary_recovery_enabled):
            boundary_decision = self._boundary_recovery_decision(
                request=boundary_recovery_request,
                current_road_option=str(current_road_option),
                next_macro_maneuver=str(next_macro_maneuver),
                ego_in_junction=bool(ego_in_junction),
                sim_time_s=float(sim_time_s),
            )
            if boundary_decision is not None:
                return boundary_decision

        turn_decision = self._intersection_turn_decision(
            ego_in_junction=bool(ego_in_junction),
            current_road_option=str(current_road_option),
            next_macro_maneuver=str(next_macro_maneuver),
            sim_time_s=float(sim_time_s),
            upcoming_turn_direction=str(upcoming_turn_direction),
            upcoming_turn_distance_m=float(upcoming_turn_distance_m),
            reference_geometry_bad=bool(reference_geometry_bad),
            turn_exit_alignment_valid=bool(turn_exit_alignment_valid),
            turn_exit_aligned=bool(turn_exit_aligned),
            turn_exit_heading_error_rad=float(turn_exit_heading_error_rad),
            turn_exit_lateral_m=float(turn_exit_lateral_m),
        )
        if turn_decision is not None:
            return turn_decision

        self._state = LANE_FOLLOW
        self._turn_direction = ""
        self._turn_connector_seen = False
        return CPXScenarioDecision(
            state=LANE_FOLLOW,
            behavior_signal_state="unknown",
            behavior_stop_target=None,
            speed_cap_mps=self.target_speed_mps,
            reason="lane_follow_default",
        )

    def _boundary_recovery_decision(
        self,
        *,
        request: Optional[BoundaryRecoveryRequest],
        current_road_option: str,
        next_macro_maneuver: str,
        ego_in_junction: bool,
        sim_time_s: float,
    ) -> Optional[CPXScenarioDecision]:
        if request is None or not bool(request.valid):
            if self._state == BOUNDARY_RECOVERY:
                self._boundary_recovery_stable_frames = 0
            return None

        direction = str(request.turn_direction or "").strip().lower()
        if direction not in {"left", "right"}:
            direction = self._turn_direction_from_route_option(
                current_road_option
            )
        if direction not in {"left", "right"}:
            direction = self._turn_direction_from_macro(next_macro_maneuver)
        if direction not in {"left", "right"}:
            direction = str(self._turn_direction or "")

        converged = bool(
            not bool(request.active)
            and float(request.clearance_m)
            >= float(self.boundary_recovery_release_clearance_m)
            and abs(float(request.lateral_offset_m))
            <= float(self.boundary_recovery_release_lateral_m)
            and abs(float(request.heading_error_rad))
            <= float(self.boundary_recovery_release_heading_rad)
        )
        if self._state == BOUNDARY_RECOVERY and bool(converged):
            self._boundary_recovery_stable_frames += 1
            if (
                int(self._boundary_recovery_stable_frames)
                >= int(self.boundary_recovery_release_frames)
            ):
                self._boundary_recovery_stable_frames = 0
                return None
        elif bool(request.active) or self._state == BOUNDARY_RECOVERY:
            self._boundary_recovery_stable_frames = 0
        else:
            return None

        if not direction:
            return None
        self._state = BOUNDARY_RECOVERY
        self._turn_direction = str(direction)
        self._turn_latch_until_s = max(
            float(self._turn_latch_until_s),
            float(sim_time_s) + float(self.turn_exit_hold_s),
        )
        return CPXScenarioDecision(
            state=BOUNDARY_RECOVERY,
            behavior_signal_state="unknown",
            behavior_stop_target=None,
            behavior_override_decision=f"intersection_turn_{direction}",
            behavior_override_lc_state=(
                f"BOUNDARY_RECOVERY_{direction.upper()}"
            ),
            speed_cap_mps=float(self.boundary_recovery_speed_mps),
            reason=(
                "boundary_recovery:"
                f"clearance={float(request.clearance_m):.3f}:"
                f"lateral={float(request.lateral_offset_m):.3f}:"
                f"heading={float(request.heading_error_rad):.3f}:"
                f"{str(request.reason)}"
            ),
            turn_direction=str(direction),
            turn_latched=True,
            boundary_recovery_active=True,
            boundary_clearance_m=float(request.clearance_m),
            boundary_lateral_offset_m=float(request.lateral_offset_m),
            boundary_heading_error_rad=float(request.heading_error_rad),
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
            # ``traffic_state`` has already passed through
            # ``TrafficLightMemory`` owns all
            # temporal debounce/hysteresis for the raw signal reading --
            # including holding "green" through brief unknown dropouts via
            # ``hold_green_unknown_s``. This FSM layer used to keep a second,
            # independent grace window (``green_release_s``) on top of that,
            # but since it only re-derived a decision from a state that was
            # already resolved to "green", it never added information; it was
            # removed to avoid two mechanisms doing the same debounce with
            # different, easy-to-desync timing constants.
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
        upcoming_turn_direction: str,
        upcoming_turn_distance_m: float,
        reference_geometry_bad: bool,
        turn_exit_alignment_valid: bool,
        turn_exit_aligned: bool,
        turn_exit_heading_error_rad: float,
        turn_exit_lateral_m: float,
    ) -> Optional[CPXScenarioDecision]:
        # Route progress can advance to the next lane-change edge before the
        # vehicle and controller have converged to the outgoing lane. Keep
        # the turn geometry authoritative until the physical exit alignment
        # has remained valid for several consecutive frames; releasing on the
        # first macro transition caused a turn-reference -> lane-follow jump
        # and large alternating steering commands.
        turn_execution_active = self._state in {
            INTERSECTION_TURN,
            TURN_EXIT_STABILIZATION,
            CREEP,
        }
        if (
            bool(turn_execution_active)
            and bool(self._turn_connector_seen)
            and not self._turn_direction_from_route_option(current_road_option)
        ):
            exit_aligned = bool(
                not bool(ego_in_junction)
                and bool(turn_exit_alignment_valid)
                and bool(turn_exit_aligned)
            )
            self._turn_exit_stable_frames = (
                int(self._turn_exit_stable_frames) + 1
                if bool(exit_aligned)
                else 0
            )
            if int(self._turn_exit_stable_frames) >= int(
                self.turn_exit_stable_frames
            ):
                self._state = LANE_FOLLOW
                self._turn_direction = ""
                self._turn_latch_until_s = -float("inf")
                self._turn_connector_seen = False
                self._turn_exit_stable_frames = 0
                return None
            direction = str(self._turn_direction or "").strip().lower()
            if direction in {"left", "right"}:
                self._state = TURN_EXIT_STABILIZATION
                return CPXScenarioDecision(
                    state=TURN_EXIT_STABILIZATION,
                    behavior_signal_state="unknown",
                    behavior_stop_target=None,
                    behavior_override_decision=f"intersection_turn_{direction}",
                    behavior_override_lc_state=(
                        f"INTERSECTION_TURN_{direction.upper()}"
                    ),
                    speed_cap_mps=float(self.turn_speed_cap_mps),
                    reason=(
                        "turn_exit_stabilization:"
                        f"stable_frames={int(self._turn_exit_stable_frames)}/"
                        f"{int(self.turn_exit_stable_frames)}:"
                        f"heading_error={float(turn_exit_heading_error_rad):.3f}:"
                        f"lateral={float(turn_exit_lateral_m):.3f}"
                    ),
                    turn_direction=str(direction),
                    turn_latched=True,
                )

        direction = self._turn_direction_from_route_option(
            current_road_option=str(current_road_option)
        )
        prepare_direction = str(upcoming_turn_direction or "").strip().lower()
        if prepare_direction not in {"left", "right"}:
            prepare_direction = ""
        prepare_distance_valid = bool(
            math.isfinite(float(upcoming_turn_distance_m))
            and 0.0 <= float(upcoming_turn_distance_m)
            <= float(self.turn_prepare_lookahead_m)
        )
        if (
            not direction
            and prepare_direction
            and prepare_distance_valid
            and not bool(ego_in_junction)
        ):
            self._state = PREPARE_TURN
            self._turn_direction = str(prepare_direction)
            self._turn_latch_until_s = (
                float(sim_time_s) + float(self.turn_exit_hold_s)
            )
            return CPXScenarioDecision(
                state=PREPARE_TURN,
                behavior_signal_state="unknown",
                behavior_stop_target=None,
                # Prepare only reserves speed and direction. The junction
                # connector must not become the active control reference until
                # the current route option reaches LEFT/RIGHT or ego enters the
                # junction.
                behavior_override_decision="lane_follow",
                behavior_override_lc_state="LANE_KEEP",
                # SpeedPlanner owns the distance-based deceleration profile.
                # PREPARE_TURN only reserves the maneuver direction.
                speed_cap_mps=float(self.target_speed_mps),
                reason=(
                    f"prepare_turn_{prepare_direction}:"
                    f"distance={float(upcoming_turn_distance_m):.2f}"
                ),
                turn_direction=str(prepare_direction),
                turn_latched=False,
            )
        connector_direction = str(direction)
        if not direction and bool(ego_in_junction):
            # CARLA's junction polygon begins before the actual connector.
            # It is nevertheless the correct point to activate a turn
            # reference with a straight lead-in. Do not mark the connector as
            # seen until the route option itself becomes LEFT/RIGHT; that
            # distinction prevents the premature TURN_EXIT observed earlier.
            direction = self._turn_direction_from_macro(
                next_macro_maneuver=str(next_macro_maneuver)
            )
        if direction:
            self._turn_direction = str(direction)
            if connector_direction:
                self._turn_connector_seen = True
            self._turn_latch_until_s = float(sim_time_s) + float(self.turn_exit_hold_s)
            self._turn_exit_stable_frames = 0
        elif (
            self._state in {PREPARE_TURN, INTERSECTION_TURN, CREEP}
            and (
                bool(ego_in_junction)
                or float(sim_time_s) <= float(self._turn_latch_until_s)
                or (
                    bool(turn_exit_alignment_valid)
                    and not bool(turn_exit_aligned)
                )
            )
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
                (
                    "intersection_turn_exit_alignment_hold:"
                    f"heading_error={float(turn_exit_heading_error_rad):.3f}:"
                    f"lateral={float(turn_exit_lateral_m):.3f}"
                )
                if (
                    bool(turn_exit_alignment_valid)
                    and not bool(turn_exit_aligned)
                    and not bool(ego_in_junction)
                    and not str(upcoming_turn_direction)
                )
                else "intersection_turn_latched"
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
