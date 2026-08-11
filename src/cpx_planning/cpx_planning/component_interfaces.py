"""Small data structures used between the ROS boundary and the planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass(frozen=True)
class PlannerLocation:
    """Numeric position used by the planner without a simulator object."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class PlannerRotation:
    """Numeric world rotation used by the planner without a simulator object."""

    yaw: float = 0.0


@dataclass(frozen=True)
class PlannerTransform:
    """Numeric pose compatible with the existing planner method inputs."""

    location: PlannerLocation = field(default_factory=PlannerLocation)
    rotation: PlannerRotation = field(default_factory=PlannerRotation)


@dataclass(frozen=True)
class PlannerControl:
    """Simulator-independent throttle, brake, and normalized steering values."""

    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0


class PlannerRuntime:
    """Factory names expected by copied actuator logic, backed only by plain values."""

    Location = PlannerLocation
    Transform = PlannerTransform
    VehicleControl = PlannerControl
    Rotation = PlannerRotation


class PlannerSafetyManager:
    """Primitive view of the latest OpenCDA safety-manager status."""

    def __init__(self, timestamp_s: float, status: Dict[str, bool]):
        self.status_queue = [(float(timestamp_s), dict(status))]



@dataclass(frozen=True)
class ROSInputSnapshot:
    """Plain values collected from the latest ROS input messages.

    It only holds the values while ROSInputAdapter creates the existing PlannerInputFrame.
    """

    timestamp_s: float
    ego_pose: Dict[str, float]
    ego_speed_mps: float
    perception_objects: List[Dict[str, object]] = field(default_factory=list)
    v2x_objects: List[Dict[str, object]] = field(default_factory=list)
    traffic_lights: List[Dict[str, object]] = field(default_factory=list)
    lane_events: List[Dict[str, object]] = field( default_factory=list)
    final_goal: Dict[str, float] = field(default_factory=dict)
