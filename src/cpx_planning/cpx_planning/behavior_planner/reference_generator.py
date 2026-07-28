"""Reference intent selection between behavior planning and MPC.

This module does not build CARLA waypoints directly.  It defines the contract
for what kind of local reference the MPC should track after the behavior
planner has selected a maneuver and target lane.
"""

from __future__ import annotations

from dataclasses import dataclass

from .planner import is_emergency_brake_decision, is_fixed_stop_decision, normalize_behavior_decision


ReferenceMode = str


@dataclass(frozen=True)
class ReferenceIntent:
    """Semantic contract consumed by the local reference generator and MPC."""

    mode: ReferenceMode
    target_lane_id: int
    follow_global_route_lane: bool
    reason: str
    lateral_reference_source: str = "lane_center"
    longitudinal_target_kind: str = "speed_profile"
    stop_target_role: str = "none"
    route_role: str = "mission_hint"


def select_reference_intent(
    *,
    behavior_decision: str,
    planner_fsm_state: str,
    ego_in_junction: bool,
    reference_target_lane_id: int,
    current_lane_id: int,
    route_optimal_lane_id: int,
    global_route_reference_allowed: bool,
    traffic_control_lane_lock_active: bool,
) -> ReferenceIntent:
    """Choose the reference semantics for the current planning tick.

    Layering policy:
    - Stop/follow-lead decisions control longitudinal behavior, but lateral
      reference remains lane based unless a final stop target is explicitly
      passed to the lower reference builder.
    - Lane-change decisions track a committed transition trajectory.
    - Junction lane-follow may track the global route branch because there may
      be no stable lane-center continuation through the connector.
    - Ordinary road lane-follow tracks the selected/current lane centerline.
    """

    raw_decision = str(behavior_decision or "").strip().lower()
    normalized_decision = str(normalize_behavior_decision(behavior_decision))
    normalized_fsm = str(planner_fsm_state or "").strip().upper()
    target_lane_id = int(reference_target_lane_id or current_lane_id or route_optimal_lane_id or 0)

    if raw_decision in {"intersection_turn_left", "intersection_turn_right"}:
        return ReferenceIntent(
            mode="intersection_turn",
            target_lane_id=int(target_lane_id),
            follow_global_route_lane=bool(global_route_reference_allowed),
            reason=f"route_option_{raw_decision}",
            lateral_reference_source="global_route_branch",
            longitudinal_target_kind="speed_profile",
            stop_target_role="none",
            route_role="mpc_branch_constraint",
        )

    if bool(is_fixed_stop_decision(normalized_decision)):
        return ReferenceIntent(
            mode="stop",
            target_lane_id=int(target_lane_id),
            follow_global_route_lane=False,
            reason=f"behavior_{normalized_decision}",
            lateral_reference_source="lane_center",
            longitudinal_target_kind="stop_target",
            stop_target_role="longitudinal_speed_target",
            route_role="mission_hint",
        )

    if bool(is_emergency_brake_decision(normalized_decision)):
        return ReferenceIntent(
            mode="follow_lead",
            target_lane_id=int(target_lane_id),
            follow_global_route_lane=False,
            reason="emergency_brake",
            lateral_reference_source="lane_center",
            longitudinal_target_kind="follow_or_brake",
            stop_target_role="none",
            route_role="mission_hint",
        )

    if normalized_decision in {"lane_change_left", "lane_change_right"} or normalized_fsm in {
        "PREPARE_LANE_CHANGE_LEFT",
        "PREPARE_LANE_CHANGE_RIGHT",
        "EXECUTE_LANE_CHANGE_LEFT",
        "EXECUTE_LANE_CHANGE_RIGHT",
    }:
        return ReferenceIntent(
            mode="lane_change",
            target_lane_id=int(target_lane_id),
            follow_global_route_lane=False,
            reason=f"decision_or_fsm_{normalized_decision}:{normalized_fsm}",
            lateral_reference_source="lane_change_blend",
            longitudinal_target_kind="speed_profile",
            stop_target_role="none",
            route_role="mission_hint",
        )

    if bool(global_route_reference_allowed) and int(route_optimal_lane_id) != 0:
        if bool(ego_in_junction):
            return ReferenceIntent(
                mode="route_branch_follow",
                target_lane_id=int(target_lane_id),
                follow_global_route_lane=True,
                reason="junction_follow_global_route_branch",
                lateral_reference_source="global_route_branch",
                longitudinal_target_kind="speed_profile",
                stop_target_role="none",
                route_role="mpc_branch_constraint",
            )
        return ReferenceIntent(
            mode="lane_follow",
            target_lane_id=int(target_lane_id),
            follow_global_route_lane=False,
            reason="route_hint_local_lane_reference",
            lateral_reference_source="lane_center",
            longitudinal_target_kind="speed_profile",
            stop_target_role="none",
            route_role="lane_choice_hint_only",
        )

    return ReferenceIntent(
        mode="lane_follow",
        target_lane_id=int(target_lane_id),
        follow_global_route_lane=False,
        reason="lane_center_follow",
        lateral_reference_source="lane_center",
        longitudinal_target_kind="speed_profile",
        stop_target_role="none",
        route_role="mission_hint",
    )
