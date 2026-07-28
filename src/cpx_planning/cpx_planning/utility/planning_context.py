"""Typed planning context shared between runner, behavior, reference, and MPC.

The runner still owns the CARLA/SUMO event loop, but the values passed across
planning layers should have stable semantics.  These dataclasses are a small
boundary object for that purpose: traffic-control stop targets, lane-reference
targets, and final goals are intentionally separate concepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


@dataclass(frozen=True)
class EgoPlanningState:
    """Ego state used by planning layers."""

    x_m: float
    y_m: float
    speed_mps: float
    heading_rad: float
    lane_id: int = 0
    road_id: int = 0
    section_id: int = 0
    in_junction: bool = False

    @classmethod
    def from_sequences(
        cls,
        *,
        ego_state: Sequence[float],
        lane_context: Mapping[str, object] | None = None,
        lane_id: int | None = None,
        in_junction: bool | None = None,
    ) -> "EgoPlanningState":
        context = dict(lane_context or {})
        return cls(
            x_m=_to_float(ego_state[0]) if len(ego_state) >= 1 else 0.0,
            y_m=_to_float(ego_state[1]) if len(ego_state) >= 2 else 0.0,
            speed_mps=_to_float(ego_state[2]) if len(ego_state) >= 3 else 0.0,
            heading_rad=_to_float(ego_state[3]) if len(ego_state) >= 4 else 0.0,
            lane_id=_to_int(lane_id if lane_id is not None else context.get("lane_id", 0)),
            road_id=_to_int(context.get("road_id", 0)),
            section_id=_to_int(context.get("section_id", 0)),
            in_junction=bool(
                in_junction
                if in_junction is not None
                else context.get("is_intersection", context.get("in_junction", False))
            ),
        )


@dataclass(frozen=True)
class StopTargetContext:
    """Longitudinal stopping target.  It is not a lane-reference target."""

    active: bool = False
    x_m: float | None = None
    y_m: float | None = None
    lane_id: int = 0
    road_id: int = 0
    distance_m: float | None = None
    source: str = ""

    @classmethod
    def from_mapping(cls, stop_target: Mapping[str, object] | None) -> "StopTargetContext":
        if not isinstance(stop_target, Mapping):
            return cls()
        x_value = stop_target.get("x_m", stop_target.get("x", None))
        y_value = stop_target.get("y_m", stop_target.get("y", None))
        distance_value = stop_target.get("distance_m", None)
        return cls(
            active=x_value is not None and y_value is not None,
            x_m=None if x_value is None else _to_float(x_value),
            y_m=None if y_value is None else _to_float(y_value),
            lane_id=_to_int(stop_target.get("lane_id", 0)),
            road_id=_to_int(stop_target.get("road_id", 0)),
            distance_m=None if distance_value is None else _to_float(distance_value),
            source=str(stop_target.get("source", stop_target.get("stop_target_source", ""))),
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "active": bool(self.active),
            "x_m": "" if self.x_m is None else float(self.x_m),
            "y_m": "" if self.y_m is None else float(self.y_m),
            "lane_id": int(self.lane_id),
            "road_id": int(self.road_id),
            "distance_m": "" if self.distance_m is None else float(self.distance_m),
            "source": str(self.source),
        }


@dataclass(frozen=True)
class TrafficControlContext:
    """Traffic-control state after CP/CARLA message selection."""

    signal_state: str = "unknown"
    source: str = ""
    control_id: str = ""
    provider_source: str = ""
    from_cp: bool = False
    confidence: float = 0.0
    ego_passed_stop_line: bool = False
    stop_target: StopTargetContext = field(default_factory=StopTargetContext)

    @classmethod
    def from_signal_context(
        cls,
        *,
        signal_context: Mapping[str, object] | None,
        stop_target: Mapping[str, object] | None,
    ) -> "TrafficControlContext":
        context = dict(signal_context or {})
        return cls(
            signal_state=str(context.get("signal_state", "unknown") or "unknown").strip().lower(),
            source=str(context.get("signal_source", context.get("source", ""))),
            control_id=str(context.get("cp_control_id", context.get("control_id", ""))),
            provider_source=str(context.get("cp_provider_source", context.get("provider_source", ""))),
            from_cp=bool(context.get("traffic_control_from_cp", context.get("from_cp", False))),
            confidence=_to_float(context.get("confidence", 0.0)),
            ego_passed_stop_line=bool(context.get("ego_passed_stop_line", False)),
            stop_target=StopTargetContext.from_mapping(stop_target),
        )


@dataclass(frozen=True)
class RouteContext:
    """Mission-level route hint.  This is not an MPC tracking reference."""

    optimal_lane_id: int = 0
    next_macro_maneuver: str = "straight"
    current_road_option: str = ""
    remaining_distance_m: float = 0.0
    remaining_points_count: int = 0
    route_found: bool = False

    @classmethod
    def from_summary(
        cls,
        *,
        route_summary: object,
        route_points: Sequence[Sequence[float]] | None = None,
    ) -> "RouteContext":
        return cls(
            optimal_lane_id=_to_int(getattr(route_summary, "optimal_lane_id", 0)),
            next_macro_maneuver=str(getattr(route_summary, "next_macro_maneuver", "straight")),
            current_road_option=str(getattr(route_summary, "current_road_option", "")),
            remaining_distance_m=_to_float(getattr(route_summary, "remaining_distance_m", 0.0)),
            remaining_points_count=len(list(route_points or [])),
            route_found=bool(getattr(route_summary, "route_found", False)),
        )


@dataclass(frozen=True)
class TargetContext:
    """Separated target semantics for planner diagnostics and downstream logic."""

    local_goal: Sequence[float] | None = None
    stop_target: StopTargetContext = field(default_factory=StopTargetContext)
    final_goal: Sequence[float] | None = None


@dataclass(frozen=True)
class MapLaneContext:
    """Lane-level map context available to the planner on this tick."""

    lane_id: int = 0
    road_id: int = 0
    section_id: int = 0
    lane_count: int = 0
    allowed_lane_ids: Sequence[int] = field(default_factory=list)
    in_junction: bool = False
    route_lane_id: int = 0
    route_maneuver: str = "straight"

    @classmethod
    def from_local_context(
        cls,
        *,
        local_context: Mapping[str, object] | None,
        allowed_lane_ids: Sequence[int] | None,
        in_junction: bool,
        route_context: RouteContext,
    ) -> "MapLaneContext":
        context = dict(local_context or {})
        normalized_allowed = [
            _to_int(lane_id)
            for lane_id in list(allowed_lane_ids or [])
        ]
        return cls(
            lane_id=_to_int(context.get("lane_id", 0)),
            road_id=_to_int(context.get("road_id", 0)),
            section_id=_to_int(context.get("section_id", 0)),
            lane_count=len(normalized_allowed),
            allowed_lane_ids=normalized_allowed,
            in_junction=bool(in_junction),
            route_lane_id=int(route_context.optimal_lane_id),
            route_maneuver=str(route_context.next_macro_maneuver),
        )


@dataclass(frozen=True)
class PerceptionContext:
    """Object-level perception snapshots exposed to planning."""

    dynamic_objects: Sequence[Mapping[str, object]] = field(default_factory=list)
    static_objects: Sequence[Mapping[str, object]] = field(default_factory=list)
    planning_objects: Sequence[Mapping[str, object]] = field(default_factory=list)
    source: str = "carla"

    @property
    def dynamic_count(self) -> int:
        return len(list(self.dynamic_objects or []))

    @property
    def static_count(self) -> int:
        return len(list(self.static_objects or []))

    @property
    def planning_count(self) -> int:
        return len(list(self.planning_objects or []))


@dataclass(frozen=True)
class PredictionContext:
    """Future-object information and lane-level risk used by behavior planning."""

    lane_assignments: Mapping[str, int] = field(default_factory=dict)
    lane_prediction_risks: Mapping[int, Mapping[str, object]] = field(default_factory=dict)
    obstacle_future_trajectories: Mapping[str, Sequence[Sequence[float]]] = field(default_factory=dict)
    model: str = "constant_acceleration"
    horizon_s: float = 0.0
    dt_s: float = 0.0

    @property
    def assigned_object_count(self) -> int:
        return len(dict(self.lane_assignments or {}))

    @property
    def risky_lane_count(self) -> int:
        return sum(
            1
            for risk in dict(self.lane_prediction_risks or {}).values()
            if bool(dict(risk or {}).get("risk", False))
        )

    @property
    def predicted_object_count(self) -> int:
        return len(dict(self.obstacle_future_trajectories or {}))


@dataclass(frozen=True)
class CPMessageContext:
    """Cooperative perception/control messages visible to the planner."""

    message_path: str = ""
    traffic_controls: Sequence[Mapping[str, object]] = field(default_factory=list)
    selected_traffic_control: Mapping[str, object] | None = None
    lane_closures: Sequence[Mapping[str, object]] = field(default_factory=list)
    obstacles: Sequence[Mapping[str, object]] = field(default_factory=list)
    generated_traffic_light_control: Mapping[str, object] | None = None

    @property
    def traffic_control_count(self) -> int:
        return len(list(self.traffic_controls or []))

    @property
    def lane_closure_count(self) -> int:
        return len(list(self.lane_closures or []))

    @property
    def obstacle_count(self) -> int:
        return len(list(self.obstacles or []))

    @property
    def selected_control_id(self) -> str:
        if not isinstance(self.selected_traffic_control, Mapping):
            return ""
        return str(
            self.selected_traffic_control.get(
                "control_id",
                self.selected_traffic_control.get("id", ""),
            )
        )


@dataclass(frozen=True)
class PlanningContext:
    """Stable per-cycle context object for planning-layer boundaries."""

    sim_time_s: float
    ego: EgoPlanningState
    route: RouteContext
    traffic_control: TrafficControlContext
    targets: TargetContext
    behavior_decision: str = "lane_follow"
    behavior_fsm_state: str = "LANE_KEEP"
    reference_priority: str = ""
    global_route_reference_allowed: bool = False
    global_route_reference_gate_reason: str = ""

    def trace_fields(self) -> Dict[str, object]:
        """Flatten key context values for CSV/debug logs."""
        return {
            "planning_context_signal_state": str(self.traffic_control.signal_state),
            "planning_context_signal_source": str(self.traffic_control.source),
            "planning_context_control_id": str(self.traffic_control.control_id),
            "planning_context_from_cp": int(bool(self.traffic_control.from_cp)),
            "planning_context_stop_target_active": int(bool(self.traffic_control.stop_target.active)),
            "planning_context_stop_target_distance_m": (
                ""
                if self.traffic_control.stop_target.distance_m is None
                else float(self.traffic_control.stop_target.distance_m)
            ),
            "planning_context_route_lane_id": int(self.route.optimal_lane_id),
            "planning_context_route_maneuver": str(self.route.next_macro_maneuver),
            "planning_context_reference_priority": str(self.reference_priority),
            "planning_context_global_route_reference_allowed": int(
                bool(self.global_route_reference_allowed)
            ),
            "planning_context_global_route_reference_gate_reason": str(
                self.global_route_reference_gate_reason
            ),
        }


@dataclass(frozen=True)
class PlannerInputFrame:
    """Complete planner-facing input bundle for one planning tick.

    This is the intended boundary between upstream data providers
    (CARLA/SUMO/CP/perception/prediction/map/route) and the planning stack.
    Existing code can still consume the smaller `PlanningContext`, while new
    modules can depend on this richer frame.
    """

    planning: PlanningContext
    map_lane: MapLaneContext = field(default_factory=MapLaneContext)
    perception: PerceptionContext = field(default_factory=PerceptionContext)
    prediction: PredictionContext = field(default_factory=PredictionContext)
    cp_messages: CPMessageContext = field(default_factory=CPMessageContext)

    def trace_fields(self) -> Dict[str, object]:
        fields = dict(self.planning.trace_fields())
        fields.update({
            "planner_input_lane_id": int(self.map_lane.lane_id),
            "planner_input_road_id": int(self.map_lane.road_id),
            "planner_input_section_id": int(self.map_lane.section_id),
            "planner_input_allowed_lane_count": int(self.map_lane.lane_count),
            "planner_input_in_junction": int(bool(self.map_lane.in_junction)),
            "planner_input_perception_dynamic_count": int(self.perception.dynamic_count),
            "planner_input_perception_static_count": int(self.perception.static_count),
            "planner_input_perception_planning_count": int(self.perception.planning_count),
            "planner_input_perception_source": str(self.perception.source),
            "planner_input_prediction_model": str(self.prediction.model),
            "planner_input_prediction_assigned_object_count": int(
                self.prediction.assigned_object_count
            ),
            "planner_input_prediction_predicted_object_count": int(
                self.prediction.predicted_object_count
            ),
            "planner_input_prediction_risky_lane_count": int(
                self.prediction.risky_lane_count
            ),
            "planner_input_cp_message_path": str(self.cp_messages.message_path),
            "planner_input_cp_traffic_control_count": int(
                self.cp_messages.traffic_control_count
            ),
            "planner_input_cp_lane_closure_count": int(
                self.cp_messages.lane_closure_count
            ),
            "planner_input_cp_obstacle_count": int(self.cp_messages.obstacle_count),
            "planner_input_cp_selected_control_id": str(
                self.cp_messages.selected_control_id
            ),
            "planner_input_cp_generated_tl": int(
                isinstance(self.cp_messages.generated_traffic_light_control, Mapping)
            ),
        })
        return fields
