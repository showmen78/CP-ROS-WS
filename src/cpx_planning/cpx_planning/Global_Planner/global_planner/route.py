"""Route wrapper objects exposed by the planner package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .waypoint import Waypoint


@dataclass
class Route:
    """Store one planned route and the data most callers need to inspect."""

    raw_start: dict[str, float]
    raw_goal: dict[str, float]
    resolved_start: "Waypoint"
    resolved_goal: "Waypoint"
    lane_path: list[int]
    sampled_waypoints: list["Waypoint"]
    length_m: float
    sampling_resolution_m: float
    transition_types: list[str] = field(default_factory=list)

    def to_point_dicts(self) -> list[dict[str, float]]:
        """Return the sampled route as CARLA-style points.

        input: none (`None`)
        output: sampled route points (`list[dict[str, float]]`)
        """
        return [waypoint.position for waypoint in self.sampled_waypoints]
