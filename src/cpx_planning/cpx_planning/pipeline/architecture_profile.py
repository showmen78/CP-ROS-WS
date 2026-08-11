"""Planner architecture ownership and configuration normalization.

The original integration accumulated several stateful mechanisms that acted
on the same signal.  A full-pipeline run now has one owner per concern:

* reference continuity: reference contract/stabilizer
* control time alignment: MPC control buffer
* final hazard/rate enforcement: SafetySupervisor

Removed legacy modes are rejected. Conflicting historical full-pipeline
switches are disabled before components are constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


@dataclass(frozen=True)
class ArchitectureProfile:
    name: str
    behavior_owner: str
    speed_owner: str
    reference_owner: str
    control_memory_owner: str
    safety_owner: str
    normalized_overrides: Tuple[str, ...] = ()

    def as_debug_fields(self) -> Dict[str, object]:
        return {
            "architecture_profile": str(self.name),
            "architecture_behavior_owner": str(self.behavior_owner),
            "architecture_speed_owner": str(self.speed_owner),
            "architecture_reference_owner": str(self.reference_owner),
            "architecture_control_memory_owner": str(self.control_memory_owner),
            "architecture_safety_owner": str(self.safety_owner),
            "architecture_normalized_overrides": "|".join(self.normalized_overrides),
        }


_FULL_PIPELINE_SINGLE_OWNER_FLAGS = {
    "opencda_style_reference_conditioning_enabled": False,
    "low_speed_lateral_recovery_enabled": False,
    "lane_follow_speed_recovery_enabled": False,
    "lane_follow_negative_accel_release_enabled": False,
    "full_dense_traffic_lane_change_lock_enabled": False,
    "overspeed_guard_enabled": False,
    "full_low_speed_launch_enabled": False,
    "full_low_speed_launch_ramp_enabled": False,
    "full_low_speed_straight_steer_guard_enabled": False,
    "strict_reference_validator_veto_enabled": False,
    # Road-boundary geometry is a ReferencePipeline concern. In full mode it
    # remains observable for metrics, but cannot create another scenario,
    # reference, speed plan, or post-MPC control owner.
    "boundary_recovery_enabled": False,
    "turn_road_boundary_speed_guard_enabled": False,
}


def normalize_architecture_config(
    config: Mapping[str, object] | None,
) -> tuple[Dict[str, object], ArchitectureProfile]:
    normalized = dict(config or {})
    mode = str(normalized.get("mode", "full_cpx_mpc")).strip().lower()
    if mode != "full_cpx_mpc":
        raise ValueError(
            f"Planner mode '{mode}' is unsupported; use 'full_cpx_mpc'."
        )

    overrides = []
    for key, required_value in _FULL_PIPELINE_SINGLE_OWNER_FLAGS.items():
        previous_value = bool(normalized.get(key, required_value))
        if previous_value != bool(required_value):
            overrides.append(f"{key}:{previous_value}->{required_value}")
        normalized[key] = bool(required_value)

    normalized["full_mpc_reference_stabilizer_enabled"] = True
    normalized["control_buffer_enabled"] = True
    normalized["safety_supervisor_enabled"] = True
    velocity_steering_interface = bool(
        normalized.get("velocity_steering_interface_enabled", False)
    )
    profile_name = (
        "unified_velocity_steering_v1"
        if bool(velocity_steering_interface)
        else "unified_full_v2"
    )
    normalized["architecture_profile"] = str(profile_name)
    return normalized, ArchitectureProfile(
        name=str(profile_name),
        behavior_owner="BehaviorPlanner+CandidateEvaluator",
        speed_owner="SpeedPlanner",
        reference_owner="ReferenceGenerator+ReferencePipeline",
        control_memory_owner=(
            "CarlaVelocitySteeringAdapter"
            if bool(velocity_steering_interface)
            else "MPCControlBuffer"
        ),
        safety_owner="MinimalSafetySupervisor",
        normalized_overrides=tuple(overrides),
    )
