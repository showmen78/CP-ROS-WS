"""Small data structures used between the ROS boundary and the planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


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