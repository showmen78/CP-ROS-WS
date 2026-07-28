"""Waypoint wrapper objects exposed by the planner package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .planner import GlobalPlanner


@dataclass
class Waypoint:
    """Represent one planner waypoint with CARLA-format and ENU coordinates."""

    position: dict[str, float]
    enu_position: tuple[float, float, float]
    ad_lane_id: int
    road_id: int | None
    section_id: int | None
    lane_id: int | None
    parametric_offset: float
    heading: float | None
    lane_length_m: float | None
    lane_width_m: float | None
    is_intersection: bool
    _planner: "GlobalPlanner" = field(repr=False, compare=False)

    @property
    def lane_width(self) -> float | None:
        """Return lane width using the CARLA-style attribute name.

        input: none (`None`)
        output: lane width in meters (`float | None`)
        """
        return self.lane_width_m

    def left(self) -> "Waypoint" | None:
        """Return the adjacent same-direction left-lane waypoint if it exists.

        input: none (`None`)
        output: adjacent left waypoint or no result (`Waypoint | None`)
        """
        return self._planner._get_adjacent_waypoint(self, side="left")

    def right(self) -> "Waypoint" | None:
        """Return the adjacent same-direction right-lane waypoint if it exists.

        input: none (`None`)
        output: adjacent right waypoint or no result (`Waypoint | None`)
        """
        return self._planner._get_adjacent_waypoint(self, side="right")

    def next(self, distance_m: float) -> list["Waypoint"]:
        """Advance forward along the lane network by the requested distance.

        input: `distance_m` (`float`)
        output: reachable forward waypoints (`list[Waypoint]`)
        """
        return self._planner._step_waypoint(self, distance_m=distance_m, forward=True)

    def previous(self, distance_m: float) -> list["Waypoint"]:
        """Move backward along the lane network by the requested distance.

        input: `distance_m` (`float`)
        output: reachable backward waypoints (`list[Waypoint]`)
        """
        return self._planner._step_waypoint(self, distance_m=distance_m, forward=False)

    def to_dict(self) -> dict[str, Any]:
        """Return one JSON-friendly waypoint summary.

        input: none (`None`)
        output: serialized waypoint data (`dict[str, object]`)
        """
        return {
            "position": dict(self.position),
            "enu_position": tuple(self.enu_position),
            "ad_lane_id": self.ad_lane_id,
            "road_id": self.road_id,
            "section_id": self.section_id,
            "lane_id": self.lane_id,
            "parametric_offset": self.parametric_offset,
            "heading": self.heading,
            "lane_length_m": self.lane_length_m,
            "lane_width_m": self.lane_width_m,
            "lane_width": self.lane_width,
            "is_intersection": self.is_intersection,
        }
