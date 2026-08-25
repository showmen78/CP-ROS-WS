"""Compatibility helpers for the planning module's map-facing APIs.

The planner internally uses mapping-based positions and poses.  Legacy CARLA
callers still provide ``world_map``, ``carla`` and ``ego_transform`` objects.
This module keeps that conversion at the boundary so planning algorithms do
not need parallel CARLA-specific implementations.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


class CarlaMapPlannerAdapter:
    """Expose a CARLA map through the planner's ``get_waypoint(dict)`` API."""

    def __init__(self, world_map: Any, carla_module: Any = None) -> None:
        self._world_map = world_map
        self._carla = carla_module

    def get_waypoint(self, point: Any):
        if isinstance(point, Mapping):
            location_type = getattr(self._carla, "Location", None)
            if location_type is not None:
                point = location_type(
                    x=float(point.get("x", 0.0)),
                    y=float(point.get("y", 0.0)),
                    z=float(point.get("z", 0.0)),
                )
        get_waypoint = getattr(self._world_map, "get_waypoint", None)
        if not callable(get_waypoint):
            return None
        try:
            return get_waypoint(point)
        except TypeError:
            return get_waypoint(point, project_to_road=True)


def coerce_map_planner(
    *,
    map_planner: Any = None,
    world_map: Any = None,
    carla_module: Any = None,
) -> Any:
    """Return the native planner or adapt a legacy CARLA world map."""

    if map_planner is not None:
        return map_planner
    if world_map is None:
        raise TypeError("map_planner (or legacy world_map) is required")
    return CarlaMapPlannerAdapter(world_map, carla_module)


def pose_from_transform(transform: Any) -> dict[str, float]:
    """Convert a CARLA transform to the planner's mapping-based ego pose."""

    location = getattr(transform, "location", None)
    rotation = getattr(transform, "rotation", None)
    if location is None:
        raise TypeError("ego_transform.location is required")
    return {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(getattr(location, "z", 0.0)),
        "heading_rad": math.radians(float(getattr(rotation, "yaw", 0.0))),
    }
