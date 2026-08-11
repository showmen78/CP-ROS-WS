"""Unified straight, lane-change, turn, and stop reference generation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

from cpx_planning.component_interfaces import PlannerLocation


@dataclass(frozen=True)
class GeneratedReference:
    samples: list[dict[str, object]]
    destination_state: list[float]
    source: str
    reason: str = ""


@dataclass(frozen=True)
class SweptFootprintValidation:
    valid: bool
    checked_pose_count: int
    violation_count: int
    min_clearance_m: float
    reason: str = ""


@dataclass(frozen=True)
class LaneCorridorOccupancy:
    valid: bool
    lateral_offset_m: float
    heading_error_rad: float
    footprint_clearance_m: float
    lane_width_m: float
    reason: str = ""


@dataclass(frozen=True)
class ReferenceCorridorProjection:
    valid: bool
    occupancy: LaneCorridorOccupancy
    segment_index: int
    segment_ratio: float
    projected_x_m: float
    projected_y_m: float
    raw_heading_rad: float
    conditioned_heading_rad: float
    continuity_limited: bool
    reason: str = ""


@dataclass(frozen=True)
class BoundaryRecoveryValidation:
    valid: bool
    checked_pose_count: int
    initial_clearance_m: float
    min_clearance_m: float
    terminal_clearance_m: float
    improvement_m: float
    reason: str = ""


@dataclass(frozen=True)
class DrivableFootprintOccupancy:
    valid: bool
    inside: bool
    checked_point_count: int
    min_clearance_m: float
    reason: str = ""


class ReferenceGenerator:
    """Generate MPC reference geometry from route and CARLA waypoint context."""

    def __init__(
        self,
        *,
        config: Mapping[str, object],
        mpc: Any,
        map_planner: Any,
        map_waypoint_from_location: Callable[[PlannerLocation], Any],
        lane_id_at_location: Callable[[PlannerLocation], int],
        body_frame_xy: Callable[..., tuple[float, float]],
        target_speed_mps: float,
        lookahead_m: float,
        drivable_waypoint_from_location: (
            Callable[[PlannerLocation], Any] | None
        ) = None,
    ) -> None:
        self.config = dict(config)
        self.mpc = mpc
        self.map_planner = map_planner
        self._map_waypoint_callback = map_waypoint_from_location
        self._lane_id_callback = lane_id_at_location
        self._body_frame_callback = body_frame_xy
        self.target_speed_mps = float(target_speed_mps)
        self.lookahead_m = float(lookahead_m)
        self._drivable_waypoint_callback = (
            drivable_waypoint_from_location
        )
        self._last_corridor_projection_geometry: (
            tuple[float, float, float, float] | None
        ) = None

    def _map_waypoint_from_location(self, location: PlannerLocation):
        return self._map_waypoint_callback(location)

    def _lane_id_at_location(self, location: PlannerLocation) -> int:
        return int(self._lane_id_callback(location))

    def _body_frame_xy(self, **kwargs: float) -> tuple[float, float]:
        return self._body_frame_callback(**kwargs)

    @staticmethod
    def _wrap_angle_static(angle_rad: float) -> float:
        return math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad)))

    # Public geometry API. Callers select a semantic reference mode and do not
    # depend on the waypoint/polyline helpers below.
    def build_route_reference(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        speed_ref_mps: float,
    ):
        destination, samples = self._build_route_reference(
            ego_location=ego_location,
            ego_yaw_rad=ego_yaw_rad,
            speed_ref_mps=speed_ref_mps,
        )
        return GeneratedReference(
            samples=[dict(sample) for sample in samples],
            destination_state=list(destination),
            source="global_route",
        )

    def build_lane_fallback(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        current_state: Sequence[float],
        speed_ref_mps: float,
    ):
        destination, samples = self._build_current_lane_fallback_reference(
            ego_location=ego_location,
            ego_yaw_rad=ego_yaw_rad,
            current_state=current_state,
            speed_ref_mps=speed_ref_mps,
        )
        return GeneratedReference(
            samples=[dict(sample) for sample in samples],
            destination_state=list(destination),
            source="current_lane_fallback",
        )

    def map_waypoint(self, location: PlannerLocation):
        return self._map_waypoint_from_location(location)

    def lane_center_samples(self, **kwargs: Any) -> list[dict[str, float]]:
        return self._current_lane_center_reference_samples(**kwargs)

    def lane_recovery_samples(self, **kwargs: Any) -> list[dict[str, float]]:
        return self._ego_anchored_lane_recovery_reference_samples(**kwargs)

    def target_lane_stabilization_samples(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        target_lane_id: int,
        horizon_steps: int,
        step_distance_m: float,
        route_points: Sequence[Sequence[float]] | None = None,
    ) -> list[dict[str, object]]:
        """Build a short ego-anchored handoff to the target lane center."""

        reference = self._ego_anchored_lane_recovery_reference_samples(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            start_waypoint=self._map_waypoint_from_location(ego_location),
            current_lane_id=int(target_lane_id),
            horizon_steps=int(horizon_steps),
            step_distance_m=float(step_distance_m),
            route_points=route_points,
        )
        for sample in reference:
            sample["lane_id"] = int(target_lane_id)
            sample["lane_change_progress"] = 1.0
            sample["lane_transition_kind"] = "target_lane_stabilization"
        return [dict(sample) for sample in reference]

    def turn_samples(self, **kwargs: Any) -> list[dict[str, float]]:
        return self._ego_anchored_turn_reference_samples(**kwargs)

    def build_boundary_recovery(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        current_lane_id: int,
        base_reference_samples: Sequence[Mapping[str, object]],
        target_speed_mps: float,
        horizon_steps: int,
        dt_s: float,
    ) -> GeneratedReference:
        """Build a C1 ego-anchored connector back to the turn corridor."""

        base = []
        for sample in list(base_reference_samples or []):
            try:
                x_m = float(sample.get("x_ref_m", sample.get("x", "")))
                y_m = float(sample.get("y_ref_m", sample.get("y", "")))
            except (TypeError, ValueError):
                continue
            if math.isfinite(x_m) and math.isfinite(y_m):
                row = dict(sample)
                row.update({"x_ref_m": x_m, "y_ref_m": y_m})
                base.append(row)
        if len(base) < 2:
            return GeneratedReference(
                samples=[],
                destination_state=[],
                source="boundary_recovery",
                reason="boundary_recovery_missing_base_reference",
            )

        speed_mps = max(0.1, float(target_speed_mps))
        step_distance_m = max(
            0.05,
            float(speed_mps) * max(1.0e-3, float(dt_s)),
        )
        first_arc_m = max(
            0.10,
            float(
                self.config.get(
                    "boundary_recovery_first_arc_m",
                    0.20,
                )
            ),
        )
        required_arc_m = (
            float(first_arc_m)
            + max(1, int(horizon_steps) - 1) * float(step_distance_m)
        )
        target_arc_m = max(
            required_arc_m + 0.25,
            float(
                self.config.get(
                    "boundary_recovery_connector_lookahead_m",
                    1.6,
                )
            ),
        )
        accumulated_m = 0.0
        previous_xy = (float(ego_location.x), float(ego_location.y))
        target_index = len(base) - 1
        for index, sample in enumerate(base):
            current_xy = (
                float(sample["x_ref_m"]),
                float(sample["y_ref_m"]),
            )
            accumulated_m += math.hypot(
                float(current_xy[0]) - float(previous_xy[0]),
                float(current_xy[1]) - float(previous_xy[1]),
            )
            if float(accumulated_m) >= float(target_arc_m):
                target_index = int(index)
                break
            previous_xy = current_xy

        target = base[int(target_index)]
        target_x_m = float(target["x_ref_m"])
        target_y_m = float(target["y_ref_m"])
        try:
            target_heading_rad = float(target.get("heading_rad", ""))
        except (TypeError, ValueError):
            target_heading_rad = float("nan")
        if not math.isfinite(target_heading_rad):
            previous = base[max(0, int(target_index) - 1)]
            target_heading_rad = math.atan2(
                float(target_y_m) - float(previous["y_ref_m"]),
                float(target_x_m) - float(previous["x_ref_m"]),
            )

        ego_x_m = float(ego_location.x)
        ego_y_m = float(ego_location.y)
        chord_m = max(
            0.1,
            math.hypot(
                float(target_x_m) - float(ego_x_m),
                float(target_y_m) - float(ego_y_m),
            ),
        )
        handle_m = min(
            0.55 * float(chord_m),
            max(
                0.30,
                float(
                    self.config.get(
                        "boundary_recovery_tangent_handle_m",
                        0.65,
                    )
                ),
            ),
        )
        p0 = (float(ego_x_m), float(ego_y_m))
        p1 = (
            float(ego_x_m) + float(handle_m) * math.cos(float(ego_yaw_rad)),
            float(ego_y_m) + float(handle_m) * math.sin(float(ego_yaw_rad)),
        )
        p3 = (float(target_x_m), float(target_y_m))
        p2 = (
            float(target_x_m)
            - float(handle_m) * math.cos(float(target_heading_rad)),
            float(target_y_m)
            - float(handle_m) * math.sin(float(target_heading_rad)),
        )

        dense = []
        dense_count = max(80, 6 * int(horizon_steps))
        for index in range(int(dense_count) + 1):
            t = float(index) / float(dense_count)
            one_minus_t = 1.0 - float(t)
            x_m = (
                one_minus_t ** 3 * float(p0[0])
                + 3.0 * one_minus_t ** 2 * t * float(p1[0])
                + 3.0 * one_minus_t * t ** 2 * float(p2[0])
                + t ** 3 * float(p3[0])
            )
            y_m = (
                one_minus_t ** 3 * float(p0[1])
                + 3.0 * one_minus_t ** 2 * t * float(p1[1])
                + 3.0 * one_minus_t * t ** 2 * float(p2[1])
                + t ** 3 * float(p3[1])
            )
            derivative_x = (
                3.0 * one_minus_t ** 2 * (float(p1[0]) - float(p0[0]))
                + 6.0 * one_minus_t * t * (float(p2[0]) - float(p1[0]))
                + 3.0 * t ** 2 * (float(p3[0]) - float(p2[0]))
            )
            derivative_y = (
                3.0 * one_minus_t ** 2 * (float(p1[1]) - float(p0[1]))
                + 6.0 * one_minus_t * t * (float(p2[1]) - float(p1[1]))
                + 3.0 * t ** 2 * (float(p3[1]) - float(p2[1]))
            )
            heading = math.atan2(float(derivative_y), float(derivative_x))
            dense.append((float(x_m), float(y_m), float(heading)))

        dense_arcs = [0.0]
        for first, second in zip(dense[:-1], dense[1:]):
            dense_arcs.append(
                float(dense_arcs[-1])
                + math.hypot(
                    float(second[0]) - float(first[0]),
                    float(second[1]) - float(first[1]),
                )
            )
        if float(dense_arcs[-1]) < float(required_arc_m):
            return GeneratedReference(
                samples=[],
                destination_state=[],
                source="boundary_recovery",
                reason="boundary_recovery_connector_too_short",
            )

        samples: list[dict[str, object]] = []
        for index in range(max(1, int(horizon_steps))):
            desired_arc_m = (
                float(first_arc_m) + float(index) * float(step_distance_m)
            )
            x_m, y_m, heading = self._interpolate_dense_pose_at_arc(
                dense=dense,
                cumulative_arcs=dense_arcs,
                target_arc_m=float(desired_arc_m),
            )
            nearest = min(
                base,
                key=lambda sample: (
                    float(sample["x_ref_m"]) - float(x_m)
                ) ** 2
                + (
                    float(sample["y_ref_m"]) - float(y_m)
                ) ** 2,
            )
            row = dict(nearest)
            row.update({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(heading),
                "lane_id": int(
                    nearest.get("lane_id", current_lane_id)
                    or current_lane_id
                ),
                "speed_ref_mps": float(speed_mps),
                "v_ref_mps": float(speed_mps),
                "speed_mps": float(speed_mps),
                "reference_mode": "boundary_recovery",
                "boundary_recovery_progress": (
                    float(index)
                    / float(max(1, int(horizon_steps) - 1))
                ),
            })
            samples.append(row)

        terminal = samples[-1]
        destination = [
            float(terminal["x_ref_m"]),
            float(terminal["y_ref_m"]),
            float(speed_mps),
            float(terminal["heading_rad"]),
            int(terminal.get("lane_id", current_lane_id) or current_lane_id),
        ]
        return GeneratedReference(
            samples=samples,
            destination_state=destination,
            source="boundary_recovery",
            reason=(
                "ego_anchored_boundary_recovery:"
                f"speed={float(speed_mps):.2f}:"
                f"ds={float(step_distance_m):.3f}:"
                f"target_index={int(target_index)}"
            ),
        )

    @staticmethod
    def _interpolate_dense_pose_at_arc(
        *,
        dense: Sequence[tuple[float, float, float]],
        cumulative_arcs: Sequence[float],
        target_arc_m: float,
    ) -> tuple[float, float, float]:
        target = min(
            float(cumulative_arcs[-1]),
            max(0.0, float(target_arc_m)),
        )
        upper = 1
        while (
            int(upper) < len(cumulative_arcs)
            and float(cumulative_arcs[upper]) < float(target)
        ):
            upper += 1
        upper = min(len(dense) - 1, int(upper))
        lower = max(0, int(upper) - 1)
        span = max(
            1.0e-9,
            float(cumulative_arcs[upper])
            - float(cumulative_arcs[lower]),
        )
        ratio = (
            float(target) - float(cumulative_arcs[lower])
        ) / float(span)
        first = dense[lower]
        second = dense[upper]
        heading_delta = ReferenceGenerator._wrap_angle_static(
            float(second[2]) - float(first[2])
        )
        return (
            float(first[0]) + float(ratio) * (
                float(second[0]) - float(first[0])
            ),
            float(first[1]) + float(ratio) * (
                float(second[1]) - float(first[1])
            ),
            ReferenceGenerator._wrap_angle_static(
                float(first[2]) + float(ratio) * float(heading_delta)
            ),
        )

    def lane_corridor_occupancy(
        self,
        *,
        x_m: float,
        y_m: float,
        heading_rad: float,
        ego_half_width_m: float,
        ego_half_length_m: float,
        safety_margin_m: float = 0.0,
        corridor_sample: Mapping[str, object] | None = None,
        prefer_tracking_point: bool = False,
    ) -> LaneCorridorOccupancy:
        """Measure an oriented vehicle footprint in the local lane corridor."""

        geometry = self._sample_corridor_geometry(
            sample=corridor_sample,
            x_m=float(x_m),
            y_m=float(y_m),
            prefer_tracking_point=bool(prefer_tracking_point),
        )
        if geometry is None:
            return LaneCorridorOccupancy(
                valid=False,
                lateral_offset_m=float("inf"),
                heading_error_rad=float("inf"),
                footprint_clearance_m=float("-inf"),
                lane_width_m=0.0,
                reason="lane_corridor_occupancy:no_geometry",
            )
        return self._occupancy_from_corridor_geometry(
            x_m=float(x_m),
            y_m=float(y_m),
            heading_rad=float(heading_rad),
            ego_half_width_m=float(ego_half_width_m),
            ego_half_length_m=float(ego_half_length_m),
            safety_margin_m=float(safety_margin_m),
            geometry=geometry,
        )

    def drivable_footprint_occupancy(
        self,
        *,
        x_m: float,
        y_m: float,
        z_m: float = 0.0,
        heading_rad: float,
        ego_half_width_m: float,
        ego_half_length_m: float,
        safety_margin_m: float = 0.0,
    ) -> DrivableFootprintOccupancy:
        """Check the vehicle envelope against CARLA's driving-lane union.

        A single tangent strip is overly conservative in a junction because
        the front and rear of a turning vehicle occupy different local lane
        sections. Querying each footprint point independently preserves the
        actual union of incoming, connector, and outgoing driving lanes.
        """

        callback = self._drivable_waypoint_callback
        if not callable(callback):
            return DrivableFootprintOccupancy(
                valid=False,
                inside=False,
                checked_point_count=0,
                min_clearance_m=float("-inf"),
                reason="drivable_footprint:no_map_callback",
            )

        half_width_m = max(0.0, float(ego_half_width_m))
        half_length_m = max(0.0, float(ego_half_length_m))
        longitudinal_offsets = (
            -half_length_m,
            0.0,
            half_length_m,
        )
        lateral_offsets = (
            -half_width_m,
            0.0,
            half_width_m,
        )
        points = {
            (float(longitudinal_m), float(lateral_m))
            for longitudinal_m in longitudinal_offsets
            for lateral_m in lateral_offsets
            if not (
                abs(float(longitudinal_m)) < 1.0e-9
                and abs(float(lateral_m)) < 1.0e-9
            )
        }
        cos_h = math.cos(float(heading_rad))
        sin_h = math.sin(float(heading_rad))
        clearances: list[float] = []
        for longitudinal_m, lateral_m in sorted(points):
            point_x_m = (
                float(x_m)
                + float(longitudinal_m) * float(cos_h)
                - float(lateral_m) * float(sin_h)
            )
            point_y_m = (
                float(y_m)
                + float(longitudinal_m) * float(sin_h)
                + float(lateral_m) * float(cos_h)
            )
            try:
                waypoint = callback(
                    PlannerLocation(
                        x=float(point_x_m),
                        y=float(point_y_m),
                        z=float(z_m),
                    )
                )
            except Exception:
                waypoint = None
            if waypoint is None:
                return DrivableFootprintOccupancy(
                    valid=True,
                    inside=False,
                    checked_point_count=len(clearances) + 1,
                    min_clearance_m=-max(
                        0.05,
                        float(safety_margin_m),
                    ),
                    reason="drivable_footprint:point_outside_driving_lane",
                )
            try:
                position = waypoint.position
                center_x_m = float(position["x"])
                center_y_m = float(position["y"])
                from cpx_planning.utility.global_planner import world_heading_rad
                lane_heading_rad = float(world_heading_rad(waypoint) or 0.0)
                lane_width_m = self._waypoint_lane_width(waypoint)
            except Exception:
                return DrivableFootprintOccupancy(
                    valid=False,
                    inside=False,
                    checked_point_count=len(clearances),
                    min_clearance_m=float("-inf"),
                    reason="drivable_footprint:invalid_waypoint_geometry",
                )
            if float(lane_width_m) <= 0.0:
                return DrivableFootprintOccupancy(
                    valid=False,
                    inside=False,
                    checked_point_count=len(clearances),
                    min_clearance_m=float("-inf"),
                    reason="drivable_footprint:invalid_lane_width",
                )
            dx_m = float(point_x_m) - float(center_x_m)
            dy_m = float(point_y_m) - float(center_y_m)
            point_lateral_m = (
                -math.sin(float(lane_heading_rad)) * float(dx_m)
                + math.cos(float(lane_heading_rad)) * float(dy_m)
            )
            clearances.append(
                0.5 * float(lane_width_m)
                - abs(float(point_lateral_m))
                - max(0.0, float(safety_margin_m))
            )

        if not clearances:
            return DrivableFootprintOccupancy(
                valid=False,
                inside=False,
                checked_point_count=0,
                min_clearance_m=float("-inf"),
                reason="drivable_footprint:no_points",
            )
        min_clearance_m = min(clearances)
        return DrivableFootprintOccupancy(
            valid=True,
            # Every query returned a non-projected driving waypoint. A
            # negative clearance only means that the configurable soft
            # margin was consumed, not that the footprint left the road.
            inside=True,
            checked_point_count=len(clearances),
            min_clearance_m=float(min_clearance_m),
            reason="drivable_footprint:carla_driving_lane_union",
        )

    def _occupancy_from_corridor_geometry(
        self,
        *,
        x_m: float,
        y_m: float,
        heading_rad: float,
        ego_half_width_m: float,
        ego_half_length_m: float,
        safety_margin_m: float,
        geometry: tuple[float, float, float, float],
    ) -> LaneCorridorOccupancy:
        center_x_m, center_y_m, lane_heading_rad, lane_width_m = geometry
        dx_m = float(x_m) - float(center_x_m)
        dy_m = float(y_m) - float(center_y_m)
        lateral_offset_m = (
            -math.sin(float(lane_heading_rad)) * float(dx_m)
            + math.cos(float(lane_heading_rad)) * float(dy_m)
        )
        heading_error_rad = self._wrap_angle_static(
            float(heading_rad) - float(lane_heading_rad)
        )
        projected_half_width_m = (
            abs(math.cos(float(heading_error_rad))) * float(ego_half_width_m)
            + abs(math.sin(float(heading_error_rad))) * float(ego_half_length_m)
        )
        clearance_m = (
            0.5 * float(lane_width_m)
            - abs(float(lateral_offset_m))
            - float(projected_half_width_m)
            - max(0.0, float(safety_margin_m))
        )
        return LaneCorridorOccupancy(
            valid=True,
            lateral_offset_m=float(lateral_offset_m),
            heading_error_rad=float(heading_error_rad),
            footprint_clearance_m=float(clearance_m),
            lane_width_m=float(lane_width_m),
            reason="lane_corridor_occupancy:valid",
        )

    def project_reference_corridor(
        self,
        *,
        reference_samples: Sequence[Mapping[str, object]],
        x_m: float,
        y_m: float,
        heading_rad: float,
        ego_half_width_m: float,
        ego_half_length_m: float,
        safety_margin_m: float = 0.0,
        max_heading_step_rad: float = 0.04,
        continuity_reset_distance_m: float = 2.5,
        max_position_step_m: float = 0.5,
    ) -> ReferenceCorridorProjection:
        """Project ego continuously onto the route-associated lane corridor."""

        candidates: list[
            tuple[
                float,
                int,
                float,
                tuple[float, float, float, float],
            ]
        ] = []
        samples = [dict(sample) for sample in list(reference_samples or [])]
        geometries: list[tuple[float, float, float, float] | None] = []
        for sample in samples:
            try:
                sample_x_m = float(
                    sample.get("x_ref_m", sample.get("x", x_m))
                )
                sample_y_m = float(
                    sample.get("y_ref_m", sample.get("y", y_m))
                )
            except (TypeError, ValueError):
                geometries.append(None)
                continue
            geometries.append(
                self._sample_corridor_geometry(
                    sample=sample,
                    x_m=float(sample_x_m),
                    y_m=float(sample_y_m),
                    prefer_tracking_point=True,
                )
            )
        previous = self._last_corridor_projection_geometry
        for index, (first, second) in enumerate(
            zip(geometries[:-1], geometries[1:])
        ):
            if first is None or second is None:
                continue
            projected_x_m, projected_y_m, ratio, distance_m = (
                self._project_xy_to_segment(
                    x_m=float(x_m),
                    y_m=float(y_m),
                    first_xy=(float(first[0]), float(first[1])),
                    second_xy=(float(second[0]), float(second[1])),
                )
            )
            geometry = self._interpolate_corridor_geometry_at(
                first,
                second,
                ratio=float(ratio),
            )
            if geometry is None:
                continue
            heading_error = abs(
                self._wrap_angle_static(
                    float(heading_rad) - float(geometry[2])
                )
            )
            score = float(distance_m) + 0.10 * float(heading_error)
            if previous is not None:
                score += 0.15 * min(
                    max(0.1, float(continuity_reset_distance_m)),
                    math.hypot(
                        float(projected_x_m) - float(previous[0]),
                        float(projected_y_m) - float(previous[1]),
                    ),
                )
                score += 0.10 * abs(
                    self._wrap_angle_static(
                        float(geometry[2]) - float(previous[2])
                    )
                )
            candidates.append((
                float(score),
                int(index),
                float(ratio),
                (
                    float(projected_x_m),
                    float(projected_y_m),
                    float(geometry[2]),
                    float(geometry[3]),
                ),
            ))

        if not candidates:
            occupancy = self.lane_corridor_occupancy(
                x_m=float(x_m),
                y_m=float(y_m),
                heading_rad=float(heading_rad),
                ego_half_width_m=float(ego_half_width_m),
                ego_half_length_m=float(ego_half_length_m),
                safety_margin_m=float(safety_margin_m),
                corridor_sample=(samples[0] if samples else None),
                prefer_tracking_point=True,
            )
            result = ReferenceCorridorProjection(
                valid=bool(occupancy.valid),
                occupancy=occupancy,
                segment_index=0,
                segment_ratio=0.0,
                projected_x_m=float(x_m),
                projected_y_m=float(y_m),
                raw_heading_rad=(
                    float(heading_rad)
                    if not bool(occupancy.valid)
                    else self._wrap_angle_static(
                        float(heading_rad) - float(occupancy.heading_error_rad)
                    )
                ),
                conditioned_heading_rad=(
                    float(heading_rad)
                    if not bool(occupancy.valid)
                    else self._wrap_angle_static(
                        float(heading_rad) - float(occupancy.heading_error_rad)
                    )
                ),
                continuity_limited=False,
                reason="reference_corridor_projection:fallback_geometry",
            )
            return result

        _, segment_index, segment_ratio, raw_geometry = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        raw_heading_rad = float(raw_geometry[2])
        conditioned_geometry = raw_geometry
        continuity_limited = False
        if previous is not None:
            center_jump_m = math.hypot(
                float(raw_geometry[0]) - float(previous[0]),
                float(raw_geometry[1]) - float(previous[1]),
            )
            if float(center_jump_m) <= max(
                0.1, float(continuity_reset_distance_m)
            ):
                # Candidate selection above only *penalizes* (softly, and
                # capped at continuity_reset_distance_m) jumping to a
                # different segment/lane than last tick's projection -- it
                # does not prevent one outright. Two near-parallel segments
                # (e.g. source-lane vs target-lane during a lane change) can
                # have almost identical raw distance scores, so the argmin
                # can flip between them tick to tick. Previously only
                # heading was rate-limited here; position was accepted as-is
                # whenever the jump was under continuity_reset_distance_m,
                # which is far larger than a real one-tick position change
                # at driving speed -- that gap is what let a ~2m spurious
                # jump straight through. Rate-limit position the same way
                # heading already is.
                conditioned_x_m = float(raw_geometry[0])
                conditioned_y_m = float(raw_geometry[1])
                position_limit_m = max(0.0, float(max_position_step_m))
                if (
                    float(position_limit_m) > 0.0
                    and float(center_jump_m) > float(position_limit_m)
                ):
                    clamp_fraction = float(position_limit_m) / float(center_jump_m)
                    conditioned_x_m = float(previous[0]) + clamp_fraction * (
                        float(raw_geometry[0]) - float(previous[0])
                    )
                    conditioned_y_m = float(previous[1]) + clamp_fraction * (
                        float(raw_geometry[1]) - float(previous[1])
                    )
                    continuity_limited = True

                conditioned_heading_rad = float(raw_heading_rad)
                heading_delta = self._wrap_angle_static(
                    float(raw_heading_rad) - float(previous[2])
                )
                heading_limit = max(0.0, float(max_heading_step_rad))
                if (
                    float(heading_limit) > 0.0
                    and abs(float(heading_delta)) > float(heading_limit)
                ):
                    conditioned_heading_rad = self._wrap_angle_static(
                        float(previous[2])
                        + math.copysign(
                            float(heading_limit),
                            float(heading_delta),
                        )
                    )
                    continuity_limited = True

                conditioned_geometry = (
                    float(conditioned_x_m),
                    float(conditioned_y_m),
                    float(conditioned_heading_rad),
                    float(raw_geometry[3]),
                )
            else:
                self._last_corridor_projection_geometry = None

        self._last_corridor_projection_geometry = conditioned_geometry
        occupancy = self._occupancy_from_corridor_geometry(
            x_m=float(x_m),
            y_m=float(y_m),
            heading_rad=float(heading_rad),
            ego_half_width_m=float(ego_half_width_m),
            ego_half_length_m=float(ego_half_length_m),
            safety_margin_m=float(safety_margin_m),
            geometry=conditioned_geometry,
        )
        reason = (
            "reference_corridor_projection:continuity_limited"
            if bool(continuity_limited)
            else "reference_corridor_projection:continuous"
        )
        result = ReferenceCorridorProjection(
            valid=bool(occupancy.valid),
            occupancy=occupancy,
            segment_index=int(segment_index),
            segment_ratio=float(segment_ratio),
            projected_x_m=float(conditioned_geometry[0]),
            projected_y_m=float(conditioned_geometry[1]),
            raw_heading_rad=float(raw_heading_rad),
            conditioned_heading_rad=float(conditioned_geometry[2]),
            continuity_limited=bool(continuity_limited),
            reason=str(reason),
        )
        return result

    def reset_corridor_projection(self) -> None:
        self._last_corridor_projection_geometry = None

    def validate_turn_swept_footprint(
        self,
        *,
        reference_samples: Sequence[Mapping[str, object]],
        ego_half_width_m: float,
        ego_half_length_m: float,
        safety_margin_m: float,
        max_violations: int = 0,
    ) -> SweptFootprintValidation:
        """Validate the oriented vehicle footprint along the full turn path."""

        samples = [dict(sample) for sample in list(reference_samples or [])]
        poses: list[
            tuple[
                float,
                float,
                float,
                tuple[float, float, float, float] | None,
            ]
        ] = []
        for sample in samples:
            try:
                poses.append((
                    float(sample.get("x_ref_m", sample.get("x", ""))),
                    float(sample.get("y_ref_m", sample.get("y", ""))),
                    float(sample.get("heading_rad", "")),
                    self._sample_corridor_geometry(
                        sample=sample,
                        x_m=float(
                            sample.get(
                                "x_ref_m",
                                sample.get("x", ""),
                            )
                        ),
                        y_m=float(
                            sample.get(
                                "y_ref_m",
                                sample.get("y", ""),
                            )
                        ),
                    ),
                ))
            except (TypeError, ValueError):
                continue
        # Midpoint poses close the gap between discrete MPC samples.
        swept_poses = list(poses)
        for first, second in zip(poses[:-1], poses[1:]):
            heading_delta = self._wrap_angle_static(second[2] - first[2])
            midpoint_geometry = self._interpolate_corridor_geometry(
                first[3],
                second[3],
            )
            swept_poses.append((
                0.5 * (first[0] + second[0]),
                0.5 * (first[1] + second[1]),
                self._wrap_angle_static(first[2] + 0.5 * heading_delta),
                midpoint_geometry,
            ))

        checked = 0
        violations = 0
        min_clearance_m = float("inf")
        drivable_union_checks = 0
        for x_m, y_m, heading_rad, geometry in swept_poses:
            # Junction turns span the incoming lane, connector and outgoing
            # lane.  A nearest-lane tangent strip cannot represent that union
            # and falsely rejects valid vehicle envelopes near the apex.
            # Prefer CARLA's non-projected Driving-lane occupancy whenever it
            # is available; retain the tangent-strip check as a map-agnostic
            # fallback for tests and custom maps.
            drivable = self.drivable_footprint_occupancy(
                x_m=float(x_m),
                y_m=float(y_m),
                heading_rad=float(heading_rad),
                ego_half_width_m=float(ego_half_width_m),
                ego_half_length_m=float(ego_half_length_m),
                safety_margin_m=float(safety_margin_m),
            )
            if bool(drivable.valid):
                checked += 1
                drivable_union_checks += 1
                min_clearance_m = min(
                    float(min_clearance_m),
                    float(drivable.min_clearance_m),
                )
                if not bool(drivable.inside):
                    violations += 1
                continue
            if geometry is None:
                geometry = self._lane_corridor_geometry(
                    x_m=float(x_m),
                    y_m=float(y_m),
                )
            if geometry is None:
                continue
            checked += 1
            center_x_m, center_y_m, lane_heading_rad, lane_width_m = geometry
            dx_m = float(x_m) - float(center_x_m)
            dy_m = float(y_m) - float(center_y_m)
            center_lateral_m = (
                -math.sin(float(lane_heading_rad)) * float(dx_m)
                + math.cos(float(lane_heading_rad)) * float(dy_m)
            )
            heading_delta = self._wrap_angle_static(
                float(heading_rad) - float(lane_heading_rad)
            )
            projected_half_width_m = (
                abs(math.cos(float(heading_delta))) * float(ego_half_width_m)
                + abs(math.sin(float(heading_delta))) * float(ego_half_length_m)
            )
            clearance_m = (
                0.5 * float(lane_width_m)
                - abs(float(center_lateral_m))
                - float(projected_half_width_m)
                - max(0.0, float(safety_margin_m))
            )
            min_clearance_m = min(float(min_clearance_m), float(clearance_m))
            if float(clearance_m) < 0.0:
                violations += 1

        if checked == 0:
            return SweptFootprintValidation(
                valid=False,
                checked_pose_count=0,
                violation_count=0,
                min_clearance_m=float("-inf"),
                reason="turn_swept_footprint:no_corridor_geometry",
            )
        allowed_by_contract = bool(
            int(violations) <= max(0, int(max_violations))
        )
        union_seam_tolerated = False
        if (
            int(drivable_union_checks) == int(checked)
            and int(checked) > 0
            and int(violations) > max(0, int(max_violations))
        ):
            seam_max_poses = max(
                0,
                int(
                    self.config.get(
                        "turn_drivable_union_seam_max_pose_violations",
                        3,
                    )
                ),
            )
            seam_max_ratio = min(
                1.0,
                max(
                    0.0,
                    float(
                        self.config.get(
                            "turn_drivable_union_seam_max_violation_ratio",
                            0.10,
                        )
                    ),
                ),
            )
            union_seam_tolerated = bool(
                int(violations) <= int(seam_max_poses)
                and float(violations) / float(checked) <= float(seam_max_ratio)
            )
        valid = bool(allowed_by_contract or union_seam_tolerated)
        return SweptFootprintValidation(
            valid=bool(valid),
            checked_pose_count=int(checked),
            violation_count=int(violations),
            min_clearance_m=float(min_clearance_m),
            reason=(
                (
                    (
                        "turn_swept_footprint:drivable_union_seam_tolerated:"
                        f"violations={int(violations)}:"
                        f"checked={int(checked)}:"
                        f"min_clearance={float(min_clearance_m):.3f}"
                    )
                    if bool(union_seam_tolerated)
                    else "turn_swept_footprint:drivable_union_valid"
                    if int(drivable_union_checks) == int(checked)
                    else "turn_swept_footprint:valid"
                )
                if bool(valid)
                else (
                    "turn_swept_footprint:outside_drivable_union:"
                    if int(drivable_union_checks) == int(checked)
                    else "turn_swept_footprint:outside_corridor:"
                )
                + f"violations={int(violations)}:"
                + f"min_clearance={float(min_clearance_m):.3f}"
            ),
        )

    def ensure_turn_swept_footprint(
        self,
        *,
        reference_samples: Sequence[Mapping[str, object]],
        horizon_steps: int,
        step_distance_m: float,
        fallback_heading_rad: float,
        ego_half_width_m: float,
        ego_half_length_m: float,
        safety_margin_m: float,
        max_violations: int = 0,
    ) -> tuple[
        list[dict[str, object]],
        SweptFootprintValidation,
        str,
    ]:
        """Correct violating turn centers toward the local drivable corridor."""

        original = [dict(sample) for sample in list(reference_samples or [])]
        validation = self.validate_turn_swept_footprint(
            reference_samples=original,
            ego_half_width_m=float(ego_half_width_m),
            ego_half_length_m=float(ego_half_length_m),
            safety_margin_m=float(safety_margin_m),
            max_violations=int(max_violations),
        )
        if validation.valid:
            return original, validation, validation.reason
        if int(validation.checked_pose_count) == 0:
            # Without map corridor geometry there is no defensible correction
            # direction. Preserve the horizon exactly and let the final gate
            # reject the unverifiable reference.
            return original, validation, validation.reason
        if "outside_drivable_union" in str(validation.reason):
            # There is no single lateral correction direction in a junction
            # lane union. Per-point nearest-lane projection introduces kinks
            # and can turn a feasible connector into an impossible one.
            return original, validation, validation.reason

        correction_padding_m = max(
            0.0,
            float(
                self.config.get(
                    "turn_swept_footprint_correction_padding_m",
                    0.05,
                )
            ),
        )
        corrected: list[dict[str, object]] = []
        correction_count = 0
        for sample in original:
            row = dict(sample)
            try:
                x_m = float(row.get("x_ref_m", row.get("x", "")))
                y_m = float(row.get("y_ref_m", row.get("y", "")))
                heading_rad = float(
                    row.get("heading_rad", fallback_heading_rad)
                )
            except (TypeError, ValueError):
                corrected.append(row)
                continue
            geometry = self._sample_corridor_geometry(
                sample=row,
                x_m=x_m,
                y_m=y_m,
            )
            if geometry is None:
                corrected.append(row)
                continue
            center_x_m, center_y_m, lane_heading_rad, lane_width_m = geometry
            dx_m = float(x_m) - float(center_x_m)
            dy_m = float(y_m) - float(center_y_m)
            lateral_m = (
                -math.sin(float(lane_heading_rad)) * float(dx_m)
                + math.cos(float(lane_heading_rad)) * float(dy_m)
            )
            heading_delta = self._wrap_angle_static(
                float(heading_rad) - float(lane_heading_rad)
            )
            projected_half_width_m = (
                abs(math.cos(float(heading_delta))) * float(ego_half_width_m)
                + abs(math.sin(float(heading_delta))) * float(ego_half_length_m)
            )
            allowed_offset_m = max(
                0.0,
                0.5 * float(lane_width_m)
                - float(projected_half_width_m)
                - max(0.0, float(safety_margin_m))
                - float(correction_padding_m),
            )
            excess_m = max(0.0, abs(float(lateral_m)) - float(allowed_offset_m))
            if float(excess_m) > 0.0:
                normal_x = -math.sin(float(lane_heading_rad))
                normal_y = math.cos(float(lane_heading_rad))
                direction = 1.0 if float(lateral_m) > 0.0 else -1.0
                x_m -= float(direction) * float(excess_m) * float(normal_x)
                y_m -= float(direction) * float(excess_m) * float(normal_y)
                correction_count += 1
            row.update({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "turn_swept_footprint_corrected": bool(excess_m > 0.0),
            })
            corrected.append(row)

        corrected = self._smooth_reference_polyline_samples(
            raw_samples=corrected,
            horizon_steps=int(horizon_steps),
            step_distance_m=float(step_distance_m),
            fallback_heading_rad=float(fallback_heading_rad),
        )
        corrected_validation = self.validate_turn_swept_footprint(
            reference_samples=corrected,
            ego_half_width_m=float(ego_half_width_m),
            ego_half_length_m=float(ego_half_length_m),
            safety_margin_m=float(safety_margin_m),
            max_violations=int(max_violations),
        )
        reason = (
            "turn_swept_footprint_corrected:"
            f"points={int(correction_count)}:"
            f"{corrected_validation.reason}"
        )
        return [dict(sample) for sample in corrected], corrected_validation, reason

    def _lane_corridor_geometry(
        self,
        *,
        x_m: float,
        y_m: float,
    ) -> tuple[float, float, float, float] | None:
        try:
            waypoint = self._map_waypoint_from_location(
                PlannerLocation(x=float(x_m), y=float(y_m), z=0.0)
            )
            position = waypoint.position
            center_x_m = float(position["x"])
            center_y_m = float(position["y"])
            from cpx_planning.utility.global_planner import world_heading_rad
            lane_heading_rad = float(world_heading_rad(waypoint) or 0.0)
            lane_width_m = self._waypoint_lane_width(waypoint)
        except Exception:
            return None
        if lane_width_m <= 0.0:
            return None
        return (
            float(center_x_m),
            float(center_y_m),
            float(lane_heading_rad),
            float(lane_width_m),
        )

    def _sample_corridor_geometry(
        self,
        *,
        sample: Mapping[str, object] | None,
        x_m: float,
        y_m: float,
        prefer_tracking_point: bool = False,
    ) -> tuple[float, float, float, float] | None:
        if isinstance(sample, Mapping):
            try:
                lane_width_m = float(sample.get("lane_width_m", 0.0) or 0.0)
                if lane_width_m > 0.0 and all(
                    key in sample
                    for key in (
                        "corridor_center_x_m",
                        "corridor_center_y_m",
                        "corridor_heading_rad",
                    )
                ):
                    return (
                        float(sample["corridor_center_x_m"]),
                        float(sample["corridor_center_y_m"]),
                        float(sample["corridor_heading_rad"]),
                        float(lane_width_m),
                    )
                heading_rad = sample.get("heading_rad", sample.get("psi_ref"))
                if (
                    bool(prefer_tracking_point)
                    and lane_width_m > 0.0
                    and heading_rad is not None
                ):
                    # No explicit corridor tag: this sample is a maneuver or
                    # tracking-reference point (lane change, turn, recovery,
                    # ...), not a route-manager lane-corridor sample. Falling
                    # back to a live nearest-lane map lookup here is
                    # ambiguous mid-maneuver -- the queried (x, y) can snap
                    # to either the departure or the target lane, and
                    # "distance to nearest static lane" is not a meaningful
                    # concept while the vehicle is intentionally
                    # transitioning between lanes (this produced a spurious
                    # multi-meter "offset" that tracked the vehicle's own
                    # heading change during a lane change, not real drift).
                    # Use the tracking point itself as the corridor
                    # center/heading so the projected offset reflects
                    # genuine tracking error against the commanded path.
                    # Callers that need an *independent* ground-truth check
                    # against the true map lane (e.g. turn-swept-footprint
                    # validation/correction) must leave this at its default
                    # so they keep querying the live map below.
                    return (
                        float(x_m),
                        float(y_m),
                        float(heading_rad),
                        float(lane_width_m),
                    )
            except (TypeError, ValueError):
                pass
        return self._lane_corridor_geometry(
            x_m=float(x_m),
            y_m=float(y_m),
        )

    @staticmethod
    def _interpolate_corridor_geometry(
        first: tuple[float, float, float, float] | None,
        second: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        if first is None:
            return second
        if second is None:
            return first
        heading_delta = ReferenceGenerator._wrap_angle_static(
            float(second[2]) - float(first[2])
        )
        return (
            0.5 * (float(first[0]) + float(second[0])),
            0.5 * (float(first[1]) + float(second[1])),
            ReferenceGenerator._wrap_angle_static(
                float(first[2]) + 0.5 * float(heading_delta)
            ),
            0.5 * (float(first[3]) + float(second[3])),
        )

    @staticmethod
    def _interpolate_corridor_geometry_at(
        first: tuple[float, float, float, float] | None,
        second: tuple[float, float, float, float] | None,
        *,
        ratio: float,
    ) -> tuple[float, float, float, float] | None:
        if first is None:
            return second
        if second is None:
            return first
        alpha = min(1.0, max(0.0, float(ratio)))
        heading_delta = ReferenceGenerator._wrap_angle_static(
            float(second[2]) - float(first[2])
        )
        return (
            float(first[0]) + float(alpha) * (
                float(second[0]) - float(first[0])
            ),
            float(first[1]) + float(alpha) * (
                float(second[1]) - float(first[1])
            ),
            ReferenceGenerator._wrap_angle_static(
                float(first[2]) + float(alpha) * float(heading_delta)
            ),
            float(first[3]) + float(alpha) * (
                float(second[3]) - float(first[3])
            ),
        )

    @staticmethod
    def _project_xy_to_segment(
        *,
        x_m: float,
        y_m: float,
        first_xy: tuple[float, float],
        second_xy: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        dx_m = float(second_xy[0]) - float(first_xy[0])
        dy_m = float(second_xy[1]) - float(first_xy[1])
        length_sq = float(dx_m) * float(dx_m) + float(dy_m) * float(dy_m)
        if float(length_sq) <= 1.0e-9:
            ratio = 0.0
        else:
            ratio = (
                (float(x_m) - float(first_xy[0])) * float(dx_m)
                + (float(y_m) - float(first_xy[1])) * float(dy_m)
            ) / float(length_sq)
            ratio = min(1.0, max(0.0, float(ratio)))
        projected_x_m = float(first_xy[0]) + float(ratio) * float(dx_m)
        projected_y_m = float(first_xy[1]) + float(ratio) * float(dy_m)
        return (
            float(projected_x_m),
            float(projected_y_m),
            float(ratio),
            math.hypot(
                float(x_m) - float(projected_x_m),
                float(y_m) - float(projected_y_m),
            ),
        )

    def route_aligned_samples(self, **kwargs: Any) -> list[dict[str, float]]:
        return self._route_aligned_reference_samples(**kwargs)

    def discrete_curvature_1pm(
        self, reference_samples: Sequence[Mapping[str, object]]
    ) -> float:
        """Public accessor for the discrete curvature used by curvature_feasible_samples."""
        return self._max_discrete_curvature_1pm(list(reference_samples or []))

    def curvature_feasible_samples(
        self,
        *,
        reference_samples: Sequence[Mapping[str, object]],
        ego_location: PlannerLocation,
        ego_heading_rad: float,
        max_curvature_1pm: float,
        mode: str = "reference",
    ) -> tuple[list[dict[str, object]], str]:
        """Return moving geometry whose discrete curvature fits the vehicle."""

        raw = [dict(sample) for sample in list(reference_samples or [])]
        contract_limit = max(1.0e-3, float(max_curvature_1pm))
        # Shape inside the contract boundary. Saturating exactly at the hard
        # limit makes a valid reference fail on floating-point reconstruction
        # in the independent validator.
        limit = 0.98 * float(contract_limit)
        raw_curvature = self._max_discrete_curvature_1pm(raw)
        if len(raw) < 2 or float(raw_curvature) <= float(limit):
            return raw, ""

        result: list[dict[str, object]] = []
        current_x = float(ego_location.x)
        current_y = float(ego_location.y)
        current_heading = float(ego_heading_rad)
        previous_raw_x = float(ego_location.x)
        previous_raw_y = float(ego_location.y)
        for sample in raw:
            raw_x = float(sample.get("x_ref_m", sample.get("x", current_x)))
            raw_y = float(sample.get("y_ref_m", sample.get("y", current_y)))
            step_m = max(
                0.10,
                math.hypot(
                    float(raw_x) - float(previous_raw_x),
                    float(raw_y) - float(previous_raw_y),
                ),
            )
            desired_heading = math.atan2(
                float(raw_y) - float(current_y),
                float(raw_x) - float(current_x),
            )
            heading_error = self._wrap_angle_static(
                float(desired_heading) - float(current_heading)
            )
            max_heading_step = float(limit) * float(step_m)
            heading_step = min(
                float(max_heading_step),
                max(-float(max_heading_step), float(heading_error)),
            )
            current_heading = self._wrap_angle_static(
                float(current_heading) + float(heading_step)
            )
            current_x += float(step_m) * math.cos(float(current_heading))
            current_y += float(step_m) * math.sin(float(current_heading))
            shaped = dict(sample)
            shaped.update({
                "x_ref_m": float(current_x),
                "y_ref_m": float(current_y),
                "x": float(current_x),
                "y": float(current_y),
                "heading_rad": float(current_heading),
                "reference_curvature_limited": True,
            })
            result.append(shaped)
            previous_raw_x = float(raw_x)
            previous_raw_y = float(raw_y)

        shaped_curvature = self._max_discrete_curvature_1pm(result)
        return (
            result,
            f"curvature_feasible_{str(mode or 'reference')}:"
            f"raw={float(raw_curvature):.3f};"
            f"limit={float(contract_limit):.3f};"
            f"shaping_limit={float(limit):.3f};"
            f"shaped={float(shaped_curvature):.3f}",
        )

    def curvature_feasible_turn_samples(
        self,
        **kwargs: Any,
    ) -> tuple[list[dict[str, object]], str]:
        """Compatibility wrapper for the original turn-only public API."""

        return self.curvature_feasible_samples(mode="turn", **kwargs)

    def stop_reference(self, **kwargs: Any) -> GeneratedReference:
        samples, destination, reason = self._build_independent_stop_reference(
            **kwargs
        )
        return GeneratedReference(
            samples=[dict(sample) for sample in samples],
            destination_state=list(destination),
            source="independent_stop",
            reason=str(reason),
        )

    def validate_boundary_recovery_progress(
        self,
        *,
        reference_samples: Sequence[Mapping[str, object]],
        ego_half_width_m: float,
        ego_half_length_m: float,
        safety_margin_m: float,
        max_worsening_m: float = 0.08,
        min_terminal_improvement_m: float = 0.03,
    ) -> BoundaryRecoveryValidation:
        """Allow an invalid start only when every recovery step trends safer."""

        clearances = []
        for sample in list(reference_samples or []):
            try:
                occupancy = self.lane_corridor_occupancy(
                    x_m=float(
                        sample.get("x_ref_m", sample.get("x", ""))
                    ),
                    y_m=float(
                        sample.get("y_ref_m", sample.get("y", ""))
                    ),
                    heading_rad=float(sample.get("heading_rad", "")),
                    ego_half_width_m=float(ego_half_width_m),
                    ego_half_length_m=float(ego_half_length_m),
                    safety_margin_m=float(safety_margin_m),
                    corridor_sample=sample,
                )
            except (TypeError, ValueError):
                continue
            if bool(occupancy.valid):
                clearances.append(float(occupancy.footprint_clearance_m))
        if len(clearances) < 2:
            return BoundaryRecoveryValidation(
                valid=False,
                checked_pose_count=len(clearances),
                initial_clearance_m=float("-inf"),
                min_clearance_m=float("-inf"),
                terminal_clearance_m=float("-inf"),
                improvement_m=0.0,
                reason="boundary_recovery_progress:no_valid_geometry",
            )
        initial_clearance_m = float(clearances[0])
        terminal_clearance_m = float(clearances[-1])
        min_clearance_m = min(clearances)
        improvement_m = (
            float(terminal_clearance_m) - float(initial_clearance_m)
        )
        # A fixed worsening tolerance does not account for how far off the
        # corridor the recovery is starting from. A vehicle that is already
        # well past the boundary often has to swing out a little further
        # before it can curve back onto the lane -- the same fixed 8cm
        # tolerance that is right for a 5cm violation makes recovery from a
        # 50cm+ violation nearly impossible to ever pass, turning every such
        # case into a permanent stop (see reference_pipeline.py's boundary
        # recovery path, which does not retry after this check fails).
        # Scale the allowance with the size of the initial violation instead.
        already_off_m = max(0.0, -float(initial_clearance_m))
        effective_max_worsening_m = max(0.0, float(max_worsening_m)) + 0.6 * float(already_off_m)
        non_worsening = bool(
            float(min_clearance_m)
            >= float(initial_clearance_m) - float(effective_max_worsening_m)
        )
        improved = bool(
            float(improvement_m)
            >= max(0.0, float(min_terminal_improvement_m))
            or float(initial_clearance_m) >= 0.0
        )
        valid = bool(non_worsening and improved)
        reason = (
            "boundary_recovery_progress:valid"
            if valid
            else "boundary_recovery_progress:"
            f"initial={float(initial_clearance_m):.3f}:"
            f"min={float(min_clearance_m):.3f}:"
            f"terminal={float(terminal_clearance_m):.3f}:"
            f"improvement={float(improvement_m):.3f}"
        )
        return BoundaryRecoveryValidation(
            valid=bool(valid),
            checked_pose_count=len(clearances),
            initial_clearance_m=float(initial_clearance_m),
            min_clearance_m=float(min_clearance_m),
            terminal_clearance_m=float(terminal_clearance_m),
            improvement_m=float(improvement_m),
            reason=str(reason),
        )

    def emergency_stop_reference(self, **kwargs: Any) -> GeneratedReference:
        samples, destination = self._build_ego_heading_emergency_stop_reference(
            **kwargs
        )
        return GeneratedReference(
            samples=[dict(sample) for sample in samples],
            destination_state=list(destination),
            source="ego_heading_emergency_stop",
            reason="emergency_stop_reference",
        )

    def straight_samples(self, **kwargs: Any) -> list[dict[str, object]]:
        return self._straight_reference_samples(**kwargs)

    def stop_target_forward(self, **kwargs: Any) -> tuple[float, bool]:
        return self._stop_target_forward_m(**kwargs)

    def reference_opposes_heading(self, **kwargs: Any) -> bool:
        return self._reference_opposes_heading(**kwargs)

    def reference_lateral_offset_too_large(self, **kwargs: Any) -> bool:
        return self._reference_lateral_offset_too_large(**kwargs)

    def reference_has_heading_jump(self, **kwargs: Any) -> bool:
        return self._reference_has_heading_jump(**kwargs)

    def project_to_route_s(self, **kwargs: Any):
        return self._project_point_to_polyline_s(**kwargs)

    def waypoint_geometry(self, waypoint: Any):
        return self._waypoint_xy_heading(waypoint)

    def waypoint_lane_width(self, waypoint: Any) -> float:
        return self._waypoint_lane_width(waypoint)

    @staticmethod
    def _max_discrete_curvature_1pm(
        samples: Sequence[Mapping[str, object]],
    ) -> float:
        points = []
        for sample in list(samples or []):
            try:
                points.append((
                    float(sample.get("x_ref_m", sample.get("x", ""))),
                    float(sample.get("y_ref_m", sample.get("y", ""))),
                ))
            except Exception:
                continue
        headings = []
        distances = []
        for first, second in zip(points[:-1], points[1:]):
            dx_m = float(second[0]) - float(first[0])
            dy_m = float(second[1]) - float(first[1])
            distance_m = math.hypot(dx_m, dy_m)
            if distance_m <= 1.0e-6:
                continue
            headings.append(math.atan2(dy_m, dx_m))
            distances.append(float(distance_m))
        maximum = 0.0
        for index, (previous, current) in enumerate(
            zip(headings[:-1], headings[1:])
        ):
            ds_m = max(1.0e-6, float(distances[min(index + 1, len(distances) - 1)]))
            maximum = max(
                float(maximum),
                abs(ReferenceGenerator._wrap_angle_static(
                    float(current) - float(previous)
                )) / float(ds_m),
            )
        return float(maximum)

    def _build_route_reference(
        self,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        speed_ref_mps: float,
    ):
        samples = self._route_samples_from_custom_planner(
            ego_location=ego_location,
        )
        if not samples:
            dest_x = float(ego_location.x + self.lookahead_m * math.cos(ego_yaw_rad))
            dest_y = float(ego_location.y + self.lookahead_m * math.sin(ego_yaw_rad))
            return [dest_x, dest_y, float(speed_ref_mps), float(ego_yaw_rad)], []

        destination = samples[-1]
        return [
            float(destination["x_ref_m"]),
            float(destination["y_ref_m"]),
            float(speed_ref_mps),
            float(destination["heading_rad"]),
        ], samples

    def _build_current_lane_fallback_reference(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        current_state: Sequence[float],
        speed_ref_mps: float,
    ):
        current_lane_id = self._lane_id_at_location(ego_location)
        start_waypoint = self._map_waypoint_from_location(ego_location)
        step_distance_m = max(
            0.5,
            float(self.mpc.dt_s) * max(1.0, min(float(speed_ref_mps), float(self.target_speed_mps))),
        )
        samples = self._current_lane_center_reference_samples(
            start_waypoint=start_waypoint,
            current_lane_id=int(current_lane_id),
            horizon_steps=int(self.mpc.horizon_steps),
            step_distance_m=float(step_distance_m),
            route_points=[],
        )
        if not samples:
            samples = self._straight_reference_samples(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_lane_id=int(current_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                step_distance_m=float(step_distance_m),
            )
        destination = None
        if samples:
            from cpx_planning.behavior_planner.reference_pipeline import (
                lane_center_destination_from_reference,
            )

            destination = lane_center_destination_from_reference(
                destination_state=[
                    float(samples[-1].get("x_ref_m", samples[-1].get("x", ego_location.x))),
                    float(samples[-1].get("y_ref_m", samples[-1].get("y", ego_location.y))),
                    float(speed_ref_mps),
                    float(samples[-1].get("heading_rad", ego_yaw_rad)),
                    int(current_lane_id),
                ],
                lane_center_reference=samples,
                ego_state=current_state,
                target_forward_m=float(
                    self.config.get("full_lane_follow_guard_destination_forward_m", 8.0)
                ),
            )
        if destination is None:
            destination = [
                float(ego_location.x) + 6.0 * math.cos(float(ego_yaw_rad)),
                float(ego_location.y) + 6.0 * math.sin(float(ego_yaw_rad)),
                float(speed_ref_mps),
                float(ego_yaw_rad),
                int(current_lane_id),
            ]
        return list(destination), list(samples)

    def _route_samples_from_custom_planner(self, *, ego_location: PlannerLocation):
        if self.map_planner is None:
            return []
        waypoint = self._map_waypoint_from_location(ego_location)
        if waypoint is None:
            return []

        from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint, world_heading_rad

        samples: list[dict[str, float]] = []
        traveled_m = 0.0
        current = waypoint
        step_m = max(1.0, min(3.0, float(self.lookahead_m)))
        while current is not None and traveled_m <= float(self.lookahead_m):
            xy_heading = self._waypoint_xy_heading(current)
            if xy_heading is None:
                break
            lane_width_m = self._waypoint_lane_width(current)
            samples.append({
                "x_ref_m": float(xy_heading[0]),
                "y_ref_m": float(xy_heading[1]),
                "heading_rad": float(world_heading_rad(current) or xy_heading[2]),
                "lane_id": int(canonical_lane_id_for_waypoint(current)),
                "lane_width_m": lane_width_m,
                "road_center_offset_m": 0.0,
                "road_left_width_m": 0.5 * lane_width_m,
                "road_right_width_m": 0.5 * lane_width_m,
            })
            candidates = list(current.next(step_m) or [])
            if not candidates:
                break
            current = candidates[0]
            traveled_m += step_m
        return samples

    def _current_lane_center_reference_samples(
        self,
        *,
        start_waypoint: Any,
        current_lane_id: int,
        horizon_steps: int,
        step_distance_m: float,
        route_points: Sequence[Sequence[float]] | None = None,
        minimum_step_m: float = 0.5,
        first_point_distance_m: Optional[float] = None,
    ) -> list[dict[str, float]]:
        """Build a strict lane-follow reference from the current CARLA lane center."""

        if start_waypoint is None:
            return []

        from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint, world_heading_rad

        samples: list[dict[str, float]] = []
        current = start_waypoint
        step_m = max(float(minimum_step_m), float(step_distance_m))
        previous_heading = float(world_heading_rad(current) or 0.0)
        first_step_m = (
            max(float(minimum_step_m), float(first_point_distance_m))
            if first_point_distance_m is not None
            else max(
                step_m,
                float(
                    self.config.get(
                        "lane_follow_reference_first_point_m",
                        2.0,
                    )
                ),
            )
        )
        first_candidates = list(current.next(first_step_m) or [])
        if first_candidates:
            first_current = self._select_smooth_next_waypoint(
                current_waypoint=current,
                candidates=first_candidates,
                previous_heading_rad=float(previous_heading),
                route_points=route_points,
            )
            if first_current is not None:
                current = first_current
                previous_heading = float(world_heading_rad(current) or previous_heading)
        for _ in range(max(1, int(horizon_steps)) + 1):
            xy_heading = self._waypoint_xy_heading(current)
            if xy_heading is None:
                break
            lane_width_m = self._waypoint_lane_width(current)
            heading_rad = float(world_heading_rad(current) or xy_heading[2])
            lane_id = int(canonical_lane_id_for_waypoint(current) or current_lane_id)
            samples.append({
                "x_ref_m": float(xy_heading[0]),
                "y_ref_m": float(xy_heading[1]),
                "x": float(xy_heading[0]),
                "y": float(xy_heading[1]),
                "heading_rad": float(heading_rad),
                "lane_id": int(lane_id),
                "lane_transition_kind": "longitudinal_successor",
                "lane_width_m": float(lane_width_m),
                "road_center_offset_m": 0.0,
                "road_left_width_m": 0.5 * float(lane_width_m),
                "road_right_width_m": 0.5 * float(lane_width_m),
            })

            candidates = list(current.next(step_m) or [])
            if not candidates:
                break
            current = self._select_smooth_next_waypoint(
                current_waypoint=current,
                candidates=candidates,
                previous_heading_rad=float(previous_heading),
                route_points=route_points,
            )
            if current is None:
                break
            previous_heading = float(world_heading_rad(current) or previous_heading)
        return samples

    def _ego_anchored_lane_recovery_reference_samples(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        start_waypoint: Any,
        current_lane_id: int,
        horizon_steps: int,
        step_distance_m: float,
        route_points: Sequence[Sequence[float]] | None = None,
    ) -> list[dict[str, float]]:
        """Join the ego pose smoothly back to the current CARLA lane center."""

        lane_samples = self._current_lane_center_reference_samples(
            start_waypoint=start_waypoint,
            current_lane_id=int(current_lane_id),
            horizon_steps=max(2, int(horizon_steps) + 2),
            step_distance_m=float(step_distance_m),
            route_points=route_points,
            minimum_step_m=0.05,
            first_point_distance_m=max(0.5, float(step_distance_m)),
        )
        if not lane_samples:
            return []
        anchor_forward_m = max(
            0.6,
            float(self.config.get("lane_recovery_anchor_forward_m", 0.8)),
        )
        lane_width_m = float(lane_samples[0].get("lane_width_m", 3.5))
        anchor_x_m = float(ego_location.x) + float(anchor_forward_m) * math.cos(
            float(ego_yaw_rad)
        )
        anchor_y_m = float(ego_location.y) + float(anchor_forward_m) * math.sin(
            float(ego_yaw_rad)
        )
        target_arc_m = max(
            3.0,
            float(anchor_forward_m)
            + float(step_distance_m) * max(2, int(horizon_steps) - 1),
        )
        target_sample = min(
            lane_samples,
            key=lambda sample: abs(
                math.hypot(
                    float(sample.get("x_ref_m", sample.get("x", ego_location.x)))
                    - float(ego_location.x),
                    float(sample.get("y_ref_m", sample.get("y", ego_location.y)))
                    - float(ego_location.y),
                )
                - float(target_arc_m)
            ),
        )
        target_x_m = float(
            target_sample.get("x_ref_m", target_sample.get("x", anchor_x_m))
        )
        target_y_m = float(
            target_sample.get("y_ref_m", target_sample.get("y", anchor_y_m))
        )
        target_heading_rad = float(
            target_sample.get("heading_rad", ego_yaw_rad)
        )
        connector_length_m = max(
            1.0,
            math.hypot(
                float(target_x_m) - float(anchor_x_m),
                float(target_y_m) - float(anchor_y_m),
            ),
        )
        anchor = {
            "x_ref_m": float(anchor_x_m),
            "y_ref_m": float(anchor_y_m),
            "x": float(anchor_x_m),
            "y": float(anchor_y_m),
            "heading_rad": float(ego_yaw_rad),
            "lane_id": int(current_lane_id),
            "lane_width_m": float(lane_width_m),
            "lane_transition_kind": "ego_anchored_lane_recovery",
        }
        x_coefficients = self._quintic_pose_axis_coefficients(
            start_position=float(anchor_x_m),
            end_position=float(target_x_m),
            start_derivative=float(connector_length_m)
            * math.cos(float(ego_yaw_rad)),
            end_derivative=float(connector_length_m)
            * math.cos(float(target_heading_rad)),
        )
        y_coefficients = self._quintic_pose_axis_coefficients(
            start_position=float(anchor_y_m),
            end_position=float(target_y_m),
            start_derivative=float(connector_length_m)
            * math.sin(float(ego_yaw_rad)),
            end_derivative=float(connector_length_m)
            * math.sin(float(target_heading_rad)),
        )
        smoothed = [dict(anchor)]
        connector_intervals = max(1, int(horizon_steps) - 1)
        for index in range(1, max(1, int(horizon_steps))):
            progress = float(index) / float(connector_intervals)
            x_m = self._quintic_pose_axis_value(
                x_coefficients,
                progress,
            )
            y_m = self._quintic_pose_axis_value(
                y_coefficients,
                progress,
            )
            dx_du = self._quintic_pose_axis_derivative(
                x_coefficients,
                progress,
            )
            dy_du = self._quintic_pose_axis_derivative(
                y_coefficients,
                progress,
            )
            heading_rad = (
                math.atan2(float(dy_du), float(dx_du))
                if math.hypot(float(dx_du), float(dy_du)) > 1.0e-6
                else float(target_heading_rad)
            )
            smoothed.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(heading_rad),
                "lane_id": int(current_lane_id),
                "lane_width_m": float(lane_width_m),
                "lane_transition_kind": "ego_anchored_quintic_lane_recovery",
            })
        for sample in smoothed:
            sample["lane_id"] = int(current_lane_id)
            sample["lane_transition_kind"] = (
                "ego_anchored_quintic_lane_recovery"
            )
            sample["lane_width_m"] = float(
                sample.get("lane_width_m", lane_width_m)
            )
        return smoothed

    @staticmethod
    def _quintic_pose_axis_coefficients(
        *,
        start_position: float,
        end_position: float,
        start_derivative: float,
        end_derivative: float,
    ) -> tuple[float, float, float, float, float, float]:
        """Quintic Hermite coefficients with zero endpoint acceleration."""

        delta = float(end_position) - float(start_position)
        return (
            float(start_position),
            float(start_derivative),
            0.0,
            10.0 * float(delta)
            - 6.0 * float(start_derivative)
            - 4.0 * float(end_derivative),
            -15.0 * float(delta)
            + 8.0 * float(start_derivative)
            + 7.0 * float(end_derivative),
            6.0 * float(delta)
            - 3.0 * float(start_derivative)
            - 3.0 * float(end_derivative),
        )

    @staticmethod
    def _quintic_pose_axis_value(
        coefficients: Sequence[float],
        progress: float,
    ) -> float:
        u = min(1.0, max(0.0, float(progress)))
        return float(
            sum(float(value) * float(u) ** index for index, value in enumerate(coefficients))
        )

    @staticmethod
    def _quintic_pose_axis_derivative(
        coefficients: Sequence[float],
        progress: float,
    ) -> float:
        u = min(1.0, max(0.0, float(progress)))
        return float(
            sum(
                float(index) * float(coefficients[index]) * float(u) ** (index - 1)
                for index in range(1, len(coefficients))
            )
        )

    @staticmethod
    def _select_smooth_next_waypoint(
        *,
        current_waypoint: Any,
        candidates: Sequence[Any],
        previous_heading_rad: float,
        route_points: Sequence[Sequence[float]] | None = None,
    ):
        if not candidates:
            return None
        route_candidate = ReferenceGenerator._select_route_aligned_candidate(
            candidates=candidates,
            route_points=route_points,
            previous_heading_rad=float(previous_heading_rad),
        )
        if route_candidate is not None:
            return route_candidate
        current_road_id = int(getattr(current_waypoint, "road_id", 0) or 0)
        current_lane_id = int(getattr(current_waypoint, "lane_id", 0) or 0)

        def heading_of(waypoint: Any) -> float:
            from cpx_planning.utility.global_planner import world_heading_rad
            return float(world_heading_rad(waypoint) or previous_heading_rad)

        def heading_cost(waypoint: Any) -> float:
            delta = heading_of(waypoint) - float(previous_heading_rad)
            return abs(math.atan2(math.sin(delta), math.cos(delta)))

        same_lane = [
            candidate
            for candidate in candidates
            if int(getattr(candidate, "road_id", 0) or 0) == current_road_id
            and int(getattr(candidate, "lane_id", 0) or 0) == current_lane_id
        ]
        if same_lane:
            return min(same_lane, key=heading_cost)
        same_raw_lane = [
            candidate
            for candidate in candidates
            if int(getattr(candidate, "lane_id", 0) or 0) == current_lane_id
        ]
        if same_raw_lane:
            return min(same_raw_lane, key=heading_cost)
        return min(candidates, key=heading_cost)

    @staticmethod
    def _reference_opposes_heading(
        *,
        reference_samples: Sequence[Mapping[str, Any]],
        ego_heading_rad: float,
        max_heading_error_rad: float,
    ) -> bool:
        points = [
            (
                float(sample.get("x_ref_m", sample.get("x", 0.0))),
                float(sample.get("y_ref_m", sample.get("y", 0.0))),
            )
            for sample in list(reference_samples or [])
        ]
        for first, second in zip(points[:-1], points[1:]):
            dx = float(second[0]) - float(first[0])
            dy = float(second[1]) - float(first[1])
            if math.hypot(dx, dy) <= 0.25:
                continue
            reference_heading = math.atan2(dy, dx)
            error = ReferenceGenerator._wrap_angle_static(
                float(ego_heading_rad) - float(reference_heading)
            )
            return abs(float(error)) > float(max_heading_error_rad)
        return False

    @staticmethod
    def _reference_lateral_offset_too_large(
        *,
        reference_samples: Sequence[Mapping[str, Any]],
        ego_state: Sequence[float],
        max_lateral_offset_m: float,
    ) -> bool:
        if not reference_samples or len(ego_state) < 4:
            return False
        sample = dict(list(reference_samples)[0])
        dx_m = float(sample.get("x_ref_m", sample.get("x", 0.0))) - float(ego_state[0])
        dy_m = float(sample.get("y_ref_m", sample.get("y", 0.0))) - float(ego_state[1])
        heading_rad = float(ego_state[3])
        lateral_m = -math.sin(heading_rad) * dx_m + math.cos(heading_rad) * dy_m
        return abs(float(lateral_m)) > max(0.0, float(max_lateral_offset_m))

    def _ego_anchored_turn_reference_samples(
        self,
        *,
        ego_location: PlannerLocation,
        ego_heading_rad: float,
        current_lane_id: int,
        horizon_steps: int,
        step_distance_m: float,
        turn_direction: str,
        route_points: Sequence[Sequence[float]] | None,
    ) -> list[dict[str, float]]:
        start_waypoint = self._map_waypoint_from_location(ego_location)
        if start_waypoint is None:
            return []

        from cpx_planning.utility.global_planner import canonical_lane_id_for_waypoint, world_heading_rad

        step_m = max(0.5, float(step_distance_m))
        raw_points: list[tuple[float, float, int, float]] = []
        anchor_forward_m = max(
            0.6,
            float(self.config.get("turn_reference_anchor_forward_m", 1.0)),
        )
        raw_points.append((
            float(ego_location.x) + float(anchor_forward_m) * math.cos(float(ego_heading_rad)),
            float(ego_location.y) + float(anchor_forward_m) * math.sin(float(ego_heading_rad)),
            int(current_lane_id),
            3.5,
        ))

        current = start_waypoint
        previous_heading = float(world_heading_rad(current) or ego_heading_rad)
        first_step_m = max(
            step_m,
            float(self.config.get("turn_reference_first_waypoint_step_m", 1.5)),
        )
        for index in range(max(2, int(horizon_steps) + 3)):
            candidates = list(current.next(first_step_m if index == 0 else step_m) or [])
            if not candidates:
                break
            selected = self._select_turn_next_waypoint(
                current_waypoint=current,
                candidates=candidates,
                previous_heading_rad=float(previous_heading),
                turn_direction=str(turn_direction),
                route_points=route_points,
            )
            if selected is None:
                break
            current = selected
            xy_heading = self._waypoint_xy_heading(current)
            if xy_heading is None:
                break
            lane_width_m = self._waypoint_lane_width(current)
            raw_points.append((
                float(xy_heading[0]),
                float(xy_heading[1]),
                int(canonical_lane_id_for_waypoint(current) or current_lane_id),
                float(lane_width_m),
            ))
            previous_heading = float(world_heading_rad(current) or xy_heading[2])

        if len(raw_points) < 2:
            return []
        raw_samples = []
        for x_m, y_m, lane_id, lane_width_m in raw_points:
            raw_samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(ego_heading_rad),
                "lane_id": int(lane_id),
                "lane_width_m": float(lane_width_m),
            })
        return self._smooth_reference_polyline_samples(
            raw_samples=raw_samples,
            horizon_steps=int(horizon_steps),
            step_distance_m=float(step_m),
            fallback_heading_rad=float(ego_heading_rad),
        )

    def _build_ego_heading_reference_samples(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        current_lane_id: int,
        horizon_steps: int,
        step_distance_m: float,
        speed_mps: float,
    ) -> list[dict[str, float]]:
        samples: list[dict[str, float]] = []
        for index in range(max(2, int(horizon_steps))):
            forward_m = float(index + 1) * max(0.35, float(step_distance_m))
            x_m = float(ego_location.x) + float(forward_m) * math.cos(float(ego_yaw_rad))
            y_m = float(ego_location.y) + float(forward_m) * math.sin(float(ego_yaw_rad))
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(ego_yaw_rad),
                "lane_id": int(current_lane_id),
                "lane_width_m": 3.5,
                "speed_ref_mps": float(speed_mps),
                "v_ref_mps": float(speed_mps),
                "speed_mps": float(speed_mps),
            })
        return samples

    @staticmethod
    def _select_turn_next_waypoint(
        *,
        current_waypoint: Any,
        candidates: Sequence[Any],
        previous_heading_rad: float,
        turn_direction: str,
        route_points: Sequence[Sequence[float]] | None = None,
    ):
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        from .route_manager import select_route_aligned_waypoint_candidate

        selected = select_route_aligned_waypoint_candidate(
            candidates=candidates,
            route_points=route_points,
            previous_heading_rad=float(previous_heading_rad),
            turn_direction=str(turn_direction),
        )
        if selected is not None:
            return selected
        return min(
            candidates,
            key=lambda waypoint: abs(
                ReferenceGenerator._wrap_angle_static(
                    float(ReferenceGenerator._waypoint_xy_heading(waypoint)[2])
                    - float(previous_heading_rad)
                )
            ),
        )

    def _route_aligned_reference_samples(
        self,
        *,
        ego_location: PlannerLocation,
        ego_heading_rad: float,
        current_lane_id: int,
        horizon_steps: int,
        step_distance_m: float,
        route_points: Sequence[Sequence[float]] | None,
    ) -> list[dict[str, float]]:
        route_xy = [
            (float(point[0]), float(point[1]))
            for point in list(route_points or [])
            if len(point) >= 2
        ]
        if len(route_xy) < 2:
            return []
        ego_xy = (float(ego_location.x), float(ego_location.y))
        route_progress = [0.0]
        for first, second in zip(route_xy[:-1], route_xy[1:]):
            route_progress.append(
                float(route_progress[-1])
                + math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))
            )
        base_s = self._project_point_to_polyline_s(
            route_xy=route_xy,
            route_progress=route_progress,
            point_xy=ego_xy,
        )
        preview_m = max(
            0.5,
            float(self.config.get("route_aligned_reference_first_forward_m", 1.0)),
        )
        base_s = float(base_s) + float(preview_m)
        samples: list[dict[str, float]] = []
        lane_width_m = self._waypoint_lane_width(self._map_waypoint_from_location(ego_location))
        step_m = max(0.5, float(step_distance_m))
        oversample_step_m = max(
            0.25,
            min(float(step_m), float(self.config.get("route_aligned_reference_oversample_step_m", 0.5))),
        )
        raw_samples: list[dict[str, float]] = []
        raw_count = max(4, (max(1, int(horizon_steps)) + 1) * 2)
        for step_index in range(raw_count):
            target_s = float(base_s) + float(step_index) * float(oversample_step_m)
            x_m, y_m, heading_rad = self._sample_polyline_at_s(
                route_xy=route_xy,
                route_progress=route_progress,
                target_s=target_s,
                fallback_heading_rad=float(ego_heading_rad),
            )
            forward_m, _ = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_heading_rad),
                target_x_m=float(x_m),
                target_y_m=float(y_m),
            )
            if float(forward_m) < float(self.config.get("route_aligned_reference_min_forward_m", 0.25)):
                continue
            raw_samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(heading_rad),
                "lane_id": int(current_lane_id),
                "lane_width_m": float(lane_width_m),
                "road_center_offset_m": 0.0,
                "road_left_width_m": 0.5 * float(lane_width_m),
                "road_right_width_m": 0.5 * float(lane_width_m),
            })
        if not raw_samples:
            return []
        samples = self._smooth_reference_polyline_samples(
            raw_samples=raw_samples,
            horizon_steps=int(horizon_steps),
            step_distance_m=float(step_m),
            fallback_heading_rad=float(ego_heading_rad),
        )
        return samples

    def _smooth_reference_polyline_samples(
        self,
        *,
        raw_samples: Sequence[Mapping[str, object]],
        horizon_steps: int,
        step_distance_m: float,
        fallback_heading_rad: float,
    ) -> list[dict[str, float]]:
        points = [
            (
                float(sample.get("x_ref_m", sample.get("x", 0.0))),
                float(sample.get("y_ref_m", sample.get("y", 0.0))),
                int(sample.get("lane_id", 0) or 0),
                float(sample.get("lane_width_m", 3.5) or 3.5),
            )
            for sample in list(raw_samples or [])
        ]
        if len(points) < 2:
            return [dict(sample) for sample in list(raw_samples or [])]
        progress = [0.0]
        for first, second in zip(points[:-1], points[1:]):
            progress.append(
                progress[-1] + math.hypot(second[0] - first[0], second[1] - first[1])
            )
        samples: list[dict[str, float]] = []
        previous_heading = None
        for index in range(max(1, int(horizon_steps)) + 1):
            target_s = min(float(progress[-1]), float(index) * max(0.5, float(step_distance_m)))
            x_m, y_m, heading_rad = self._sample_polyline_at_s(
                route_xy=[(p[0], p[1]) for p in points],
                route_progress=progress,
                target_s=float(target_s),
                fallback_heading_rad=float(fallback_heading_rad),
            )
            if previous_heading is not None:
                while heading_rad - previous_heading > math.pi:
                    heading_rad -= 2.0 * math.pi
                while heading_rad - previous_heading < -math.pi:
                    heading_rad += 2.0 * math.pi
            previous_heading = float(heading_rad)
            nearest = min(
                points,
                key=lambda point: math.hypot(float(point[0]) - float(x_m), float(point[1]) - float(y_m)),
            )
            lane_id = int(nearest[2])
            lane_width_m = max(0.1, float(nearest[3]))
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(heading_rad),
                "lane_id": int(lane_id),
                "lane_width_m": float(lane_width_m),
                "road_center_offset_m": 0.0,
                "road_left_width_m": 0.5 * float(lane_width_m),
                "road_right_width_m": 0.5 * float(lane_width_m),
            })
        return samples

    @staticmethod
    def _project_point_to_polyline_s(
        *,
        route_xy: Sequence[tuple[float, float]],
        route_progress: Sequence[float],
        point_xy: tuple[float, float],
    ) -> float:
        best_distance_m = float("inf")
        best_s = float(route_progress[0]) if route_progress else 0.0
        px, py = float(point_xy[0]), float(point_xy[1])
        for idx, (first, second) in enumerate(zip(route_xy[:-1], route_xy[1:])):
            ax, ay = float(first[0]), float(first[1])
            bx, by = float(second[0]), float(second[1])
            dx = bx - ax
            dy = by - ay
            segment_len_sq = dx * dx + dy * dy
            if segment_len_sq <= 1.0e-9:
                continue
            ratio = ((px - ax) * dx + (py - ay) * dy) / segment_len_sq
            ratio = min(1.0, max(0.0, float(ratio)))
            proj_x = ax + ratio * dx
            proj_y = ay + ratio * dy
            distance_m = math.hypot(px - proj_x, py - proj_y)
            if distance_m < best_distance_m:
                best_distance_m = float(distance_m)
                segment_len_m = math.sqrt(segment_len_sq)
                best_s = float(route_progress[idx]) + ratio * segment_len_m
        return float(best_s)

    @staticmethod
    def _sample_polyline_at_s(
        *,
        route_xy: Sequence[tuple[float, float]],
        route_progress: Sequence[float],
        target_s: float,
        fallback_heading_rad: float,
    ) -> tuple[float, float, float]:
        if len(route_xy) == 0:
            return 0.0, 0.0, float(fallback_heading_rad)
        if len(route_xy) == 1:
            return float(route_xy[0][0]), float(route_xy[0][1]), float(fallback_heading_rad)
        if float(target_s) <= float(route_progress[0]):
            first, second = route_xy[0], route_xy[1]
            heading = math.atan2(float(second[1]) - float(first[1]), float(second[0]) - float(first[0]))
            return float(first[0]), float(first[1]), float(heading)
        for idx in range(len(route_xy) - 1):
            s0 = float(route_progress[idx])
            s1 = float(route_progress[idx + 1])
            if float(target_s) > s1:
                continue
            first = route_xy[idx]
            second = route_xy[idx + 1]
            denom = max(1.0e-6, float(s1) - float(s0))
            ratio = min(1.0, max(0.0, (float(target_s) - float(s0)) / denom))
            x_m = float(first[0]) + ratio * (float(second[0]) - float(first[0]))
            y_m = float(first[1]) + ratio * (float(second[1]) - float(first[1]))
            heading = math.atan2(float(second[1]) - float(first[1]), float(second[0]) - float(first[0]))
            return float(x_m), float(y_m), float(heading)
        before = route_xy[-2]
        last = route_xy[-1]
        heading = math.atan2(float(last[1]) - float(before[1]), float(last[0]) - float(before[0]))
        return float(last[0]), float(last[1]), float(heading)

    @staticmethod
    def _select_route_aligned_candidate(
        *,
        candidates: Sequence[Any],
        route_points: Sequence[Sequence[float]] | None,
        previous_heading_rad: float,
    ):
        from .route_manager import select_route_aligned_waypoint_candidate

        return select_route_aligned_waypoint_candidate(
            candidates=candidates,
            route_points=route_points,
            previous_heading_rad=float(previous_heading_rad),
        )

    @staticmethod
    def _waypoint_xy_heading(waypoint):
        transform = getattr(waypoint, "transform", None)
        location = getattr(transform, "location", None)
        rotation = getattr(transform, "rotation", None)
        if location is not None:
            heading_rad = math.radians(float(getattr(rotation, "yaw", 0.0)))
            return float(location.x), float(location.y), float(heading_rad)
        position = getattr(waypoint, "position", None)
        if isinstance(position, Mapping):
            from cpx_planning.utility.global_planner import world_heading_rad
            return (
                float(position["x"]),
                float(position["y"]),
                float(world_heading_rad(waypoint) or 0.0),
            )
        return None

    @staticmethod
    def _waypoint_lane_width(waypoint) -> float:
        for attr_name in ("lane_width_m", "lane_width"):
            value = getattr(waypoint, attr_name, None)
            if value is not None:
                try:
                    return max(0.1, float(value))
                except Exception:
                    pass
        return 3.5

    def _build_independent_stop_reference(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        current_state: Sequence[float],
        current_lane_id: int,
        stop_target: Mapping[str, object] | None,
        fallback_destination_state: Sequence[float],
        ego_speed_mps: float,
    ) -> tuple[list[dict[str, object]], list[float], str]:
        stop_buffer_m = max(0.0, float(self.config.get("full_stop_reference_buffer_m", 1.5)))
        comfortable_decel_mps2 = max(
            0.1,
            float(self.config.get("full_stop_reference_decel_mps2", 2.0)),
        )
        stop_speed_cap_mps = max(
            0.1,
            float(self.config.get("full_stop_reference_speed_cap_mps", 2.0)),
        )
        stop_forward_m, stop_target_reliable = self._stop_target_forward_m(
            ego_location=ego_location,
            ego_yaw_rad=float(ego_yaw_rad),
            stop_target=stop_target,
            fallback_destination_state=fallback_destination_state,
        )
        if not bool(stop_target_reliable):
            emergency_reference, emergency_destination = self._build_ego_heading_emergency_stop_reference(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_lane_id=int(current_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                step_distance_m=max(
                    0.5,
                    float(self.mpc.dt_s) * max(0.5, min(float(ego_speed_mps), 1.5)),
                ),
            )
            return (
                emergency_reference,
                emergency_destination,
                "independent_stop_reference;stop_missing_target_hard_lock",
            )
        stop_forward_m = max(0.5, float(stop_forward_m) - float(stop_buffer_m))
        step_distance_m = max(
            0.5,
            float(self.mpc.dt_s) * max(1.0, min(float(ego_speed_mps), stop_speed_cap_mps)),
        )
        start_waypoint = self._map_waypoint_from_location(ego_location)
        lane_reference = self._current_lane_center_reference_samples(
            start_waypoint=start_waypoint,
            current_lane_id=int(current_lane_id),
            horizon_steps=int(self.mpc.horizon_steps),
            step_distance_m=float(step_distance_m),
            route_points=[],
        )
        if not lane_reference:
            lane_reference = self._straight_reference_samples(
                ego_location=ego_location,
                ego_yaw_rad=float(ego_yaw_rad),
                current_lane_id=int(current_lane_id),
                horizon_steps=int(self.mpc.horizon_steps),
                step_distance_m=float(step_distance_m),
            )

        shaped_reference: list[dict[str, object]] = []
        chosen_destination = None
        best_distance_error_m = float("inf")
        previous_speed_mps = min(float(stop_speed_cap_mps), max(0.0, float(ego_speed_mps)))
        for sample in lane_reference:
            shaped = dict(sample)
            x_m = float(shaped.get("x_ref_m", shaped.get("x", ego_location.x)))
            y_m = float(shaped.get("y_ref_m", shaped.get("y", ego_location.y)))
            forward_m, _ = self._body_frame_xy(
                origin_x_m=float(ego_location.x),
                origin_y_m=float(ego_location.y),
                heading_rad=float(ego_yaw_rad),
                target_x_m=float(x_m),
                target_y_m=float(y_m),
            )
            remaining_m = max(0.0, float(stop_forward_m) - float(forward_m))
            speed_ref_mps = min(
                float(stop_speed_cap_mps),
                math.sqrt(max(0.0, 2.0 * float(comfortable_decel_mps2) * float(remaining_m))),
                float(previous_speed_mps),
            )
            if float(forward_m) >= float(stop_forward_m):
                speed_ref_mps = 0.0
            previous_speed_mps = float(speed_ref_mps)
            shaped["v_ref_mps"] = float(speed_ref_mps)
            shaped["speed_ref_mps"] = float(speed_ref_mps)
            shaped["speed_mps"] = float(speed_ref_mps)
            shaped["lane_id"] = int(current_lane_id)
            shaped_reference.append(shaped)
            distance_error_m = abs(float(forward_m) - float(stop_forward_m))
            if distance_error_m < best_distance_error_m:
                best_distance_error_m = float(distance_error_m)
                chosen_destination = dict(shaped)

        if shaped_reference:
            shaped_reference[-1]["v_ref_mps"] = 0.0
            shaped_reference[-1]["speed_ref_mps"] = 0.0
            shaped_reference[-1]["speed_mps"] = 0.0
        if chosen_destination is None and shaped_reference:
            chosen_destination = dict(shaped_reference[-1])

        if chosen_destination is not None:
            destination = [
                float(chosen_destination.get("x_ref_m", chosen_destination.get("x", ego_location.x))),
                float(chosen_destination.get("y_ref_m", chosen_destination.get("y", ego_location.y))),
                0.0,
                float(chosen_destination.get("heading_rad", ego_yaw_rad)),
                int(current_lane_id),
            ]
        else:
            destination = [
                float(ego_location.x) + float(stop_forward_m) * math.cos(float(ego_yaw_rad)),
                float(ego_location.y) + float(stop_forward_m) * math.sin(float(ego_yaw_rad)),
                0.0,
                float(ego_yaw_rad),
                int(current_lane_id),
            ]
        return shaped_reference, destination, "independent_stop_reference"

    def _stop_target_forward_m(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        stop_target: Mapping[str, object] | None,
        fallback_destination_state: Sequence[float],
    ) -> tuple[float, bool]:
        if isinstance(stop_target, Mapping):
            try:
                has_x = "x_m" in stop_target or "x" in stop_target
                has_y = "y_m" in stop_target or "y" in stop_target
                if not bool(has_x and has_y) and "distance_m" in stop_target:
                    return max(0.0, float(stop_target.get("distance_m", 0.0))), True
                target_x = float(stop_target.get("x_m", stop_target.get("x", ego_location.x)))
                target_y = float(stop_target.get("y_m", stop_target.get("y", ego_location.y)))
                forward_m, _ = self._body_frame_xy(
                    origin_x_m=float(ego_location.x),
                    origin_y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    target_x_m=float(target_x),
                    target_y_m=float(target_y),
                )
                return max(0.0, float(forward_m)), True
            except Exception:
                pass
        if fallback_destination_state is not None and len(fallback_destination_state) >= 2:
            try:
                forward_m, _ = self._body_frame_xy(
                    origin_x_m=float(ego_location.x),
                    origin_y_m=float(ego_location.y),
                    heading_rad=float(ego_yaw_rad),
                    target_x_m=float(fallback_destination_state[0]),
                    target_y_m=float(fallback_destination_state[1]),
                )
                return max(0.0, float(forward_m)), False
            except Exception:
                pass
        return max(2.0, float(self.config.get("full_stop_guard_destination_forward_m", 6.0))), False

    def _straight_reference_samples(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        current_lane_id: int,
        horizon_steps: int,
        step_distance_m: float,
    ) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        for index in range(max(1, int(horizon_steps)) + 1):
            distance_m = float(index + 1) * max(0.5, float(step_distance_m))
            x_m = float(ego_location.x) + distance_m * math.cos(float(ego_yaw_rad))
            y_m = float(ego_location.y) + distance_m * math.sin(float(ego_yaw_rad))
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(ego_yaw_rad),
                "lane_id": int(current_lane_id),
                "lane_width_m": 3.5,
                "road_center_offset_m": 0.0,
                "road_left_width_m": 1.75,
                "road_right_width_m": 1.75,
            })
        return samples

    def _build_ego_heading_emergency_stop_reference(
        self,
        *,
        ego_location: PlannerLocation,
        ego_yaw_rad: float,
        current_lane_id: int,
        horizon_steps: int,
        step_distance_m: float,
    ) -> tuple[list[dict[str, object]], list[float]]:
        samples = []
        for index in range(max(1, int(horizon_steps)) + 1):
            distance_m = float(index + 1) * max(0.5, float(step_distance_m))
            x_m = float(ego_location.x) + float(distance_m) * math.cos(float(ego_yaw_rad))
            y_m = float(ego_location.y) + float(distance_m) * math.sin(float(ego_yaw_rad))
            samples.append({
                "x_ref_m": float(x_m),
                "y_ref_m": float(y_m),
                "x": float(x_m),
                "y": float(y_m),
                "heading_rad": float(ego_yaw_rad),
                "lane_id": int(current_lane_id),
                "lane_width_m": 3.5,
                "road_center_offset_m": 0.0,
                "road_left_width_m": 1.75,
                "road_right_width_m": 1.75,
                "v_ref_mps": 0.0,
                "speed_ref_mps": 0.0,
                "speed_mps": 0.0,
            })
        first = samples[0] if samples else {
            "x_ref_m": float(ego_location.x),
            "y_ref_m": float(ego_location.y),
            "heading_rad": float(ego_yaw_rad),
        }
        destination = [
            float(first.get("x_ref_m", ego_location.x)),
            float(first.get("y_ref_m", ego_location.y)),
            0.0,
            float(first.get("heading_rad", ego_yaw_rad)),
            int(current_lane_id),
        ]
        return samples, destination

    @staticmethod
    def _reference_has_heading_jump(
        *,
        reference_samples: Sequence[Mapping[str, object]],
        max_heading_step_rad: float,
    ) -> bool:
        points = [
            (
                float(sample.get("x_ref_m", sample.get("x", 0.0))),
                float(sample.get("y_ref_m", sample.get("y", 0.0))),
            )
            for sample in list(reference_samples or [])
        ]
        if len(points) < 3:
            return False
        previous_heading = None
        for first, second in zip(points[:-1], points[1:]):
            dx_m = float(second[0]) - float(first[0])
            dy_m = float(second[1]) - float(first[1])
            if math.hypot(dx_m, dy_m) <= 0.25:
                continue
            heading = math.atan2(dy_m, dx_m)
            if previous_heading is not None:
                delta = ReferenceGenerator._wrap_angle_static(
                    float(heading) - float(previous_heading)
                )
                if abs(float(delta)) > float(max_heading_step_rad):
                    return True
            previous_heading = float(heading)
        return False
