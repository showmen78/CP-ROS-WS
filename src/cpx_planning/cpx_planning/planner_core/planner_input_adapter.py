"""The unchanged planner-facing output contract produced by the ROS input adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from cpx_planning.utility.planning_context import PlannerInputFrame


@dataclass(frozen=True)
class PlannerInputAdapterOutput:
    """Planner-facing input and reusable per-tick adapter products."""

    frame: PlannerInputFrame
    ego_pose: Dict[str, float]
    current_state: List[float]
    current_lane_id: int
    lane_ids: List[int]
    ego_waypoint: Any
    ego_snapshot: Dict[str, float]
    lane_assignments: Dict[str, int]
    lane_safety_scores: Dict[int, float]
    front_distance_by_lane: Dict[int, float]
    route_points: List[List[float]]
    route_summary: Dict[str, object]
    route_optimal_lane_id: int
    route_reference_allowed: bool
    route_reference_gate_reason: str
    selected_traffic_control: Optional[Mapping[str, object]]
    signal_context: Dict[str, object]
    stop_target: Optional[Mapping[str, object]]
    source_quality: Dict[str, object]

