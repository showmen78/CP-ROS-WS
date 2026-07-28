"""Reusable QP-friendly lane-keeping math helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class LaneKeepingStageReference:
    """Lane geometry attached to one MPC prediction stage."""

    x_center_m: float
    y_center_m: float
    heading_rad: float
    lane_width_m: float
    lane_id: int = 0
    road_center_offset_m: float = 0.0
    road_left_width_m: float = 0.0
    road_right_width_m: float = 0.0

    @property
    def half_lane_width_m(self) -> float:
        return 0.5 * max(1.0e-6, float(self.lane_width_m))

    @property
    def left_road_width_m(self) -> float:
        value = float(self.road_left_width_m)
        if not math.isfinite(value) or value <= 0.0:
            return self.half_lane_width_m
        return value

    @property
    def right_road_width_m(self) -> float:
        value = float(self.road_right_width_m)
        if not math.isfinite(value) or value <= 0.0:
            return self.half_lane_width_m
        return value


@dataclass(frozen=True)
class LaneKeepingAffineForm:
    """Affine signed lateral-offset model d_perp = a*x + b*y + c."""

    x_coef: float
    y_coef: float
    constant: float
    reference: LaneKeepingStageReference

    def evaluate(self, x_m: float, y_m: float) -> float:
        return (
            float(self.x_coef) * float(x_m)
            + float(self.y_coef) * float(y_m)
            + float(self.constant)
        )


@dataclass(frozen=True)
class LaneKeepingStageMetrics:
    """Evaluated lane-keeping quantities for one prediction stage."""

    stage_index: int
    d_perp_m: float
    centering_cost: float
    boundary_cost: float
    stage_cost: float
    boundary_excess_m: float
    d_safe_m: float
    d_max_m: float
    lane_width_m: float
    road_center_offset_m: float
    road_left_width_m: float
    road_right_width_m: float
    road_left_excess_m: float
    road_right_excess_m: float
    lane_id: int
    lane_heading_rad: float
    inside_safe_core: bool
    outside_lane: bool
    outside_road: bool


@dataclass(frozen=True)
class LaneKeepingProfile:
    """Lane-keeping diagnostics across the full horizon."""

    stage_metrics: Tuple[LaneKeepingStageMetrics, ...]
    total_cost: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "stage_index": [int(metric.stage_index) for metric in self.stage_metrics],
            "d_perp_m": [float(metric.d_perp_m) for metric in self.stage_metrics],
            "U_lane": [float(metric.stage_cost) for metric in self.stage_metrics],
            "J_lane": float(self.total_cost),
            "boundary_excess_m": [
                float(metric.boundary_excess_m) for metric in self.stage_metrics
            ],
            "d_safe_m": [float(metric.d_safe_m) for metric in self.stage_metrics],
            "d_max_m": [float(metric.d_max_m) for metric in self.stage_metrics],
            "lane_width_m": [float(metric.lane_width_m) for metric in self.stage_metrics],
            "road_center_offset_m": [
                float(metric.road_center_offset_m) for metric in self.stage_metrics
            ],
            "road_left_width_m": [
                float(metric.road_left_width_m) for metric in self.stage_metrics
            ],
            "road_right_width_m": [
                float(metric.road_right_width_m) for metric in self.stage_metrics
            ],
            "road_left_excess_m": [
                float(metric.road_left_excess_m) for metric in self.stage_metrics
            ],
            "road_right_excess_m": [
                float(metric.road_right_excess_m) for metric in self.stage_metrics
            ],
            "lane_id": [int(metric.lane_id) for metric in self.stage_metrics],
            "outside_lane": [bool(metric.outside_lane) for metric in self.stage_metrics],
            "outside_road": [bool(metric.outside_road) for metric in self.stage_metrics],
        }


def _clamp_safe_region_alpha(alpha: float) -> float:
    return min(0.999999, max(1.0e-6, float(alpha)))


def normalize_lane_reference_sample(
    sample: Mapping[str, object] | None,
    *,
    default_lane_width_m: float = 4.0,
) -> LaneKeepingStageReference | None:
    """Convert a reference-sample mapping into a typed lane reference."""

    if not isinstance(sample, Mapping):
        return None
    if not {"x_ref_m", "y_ref_m", "heading_rad"}.issubset(sample.keys()):
        return None

    fallback_lane_width_m = float(default_lane_width_m)
    if not math.isfinite(fallback_lane_width_m) or fallback_lane_width_m <= 0.0:
        fallback_lane_width_m = 4.0

    lane_width_m = float(sample.get("lane_width_m", fallback_lane_width_m))
    if not math.isfinite(lane_width_m) or lane_width_m <= 0.0:
        lane_width_m = float(fallback_lane_width_m)
    fallback_half_width_m = 0.5 * float(lane_width_m)

    road_center_offset_m = float(sample.get("road_center_offset_m", 0.0))
    if not math.isfinite(road_center_offset_m):
        road_center_offset_m = 0.0
    road_left_width_m = float(sample.get("road_left_width_m", fallback_half_width_m))
    if not math.isfinite(road_left_width_m) or road_left_width_m <= 0.0:
        road_left_width_m = float(fallback_half_width_m)
    road_right_width_m = float(sample.get("road_right_width_m", fallback_half_width_m))
    if not math.isfinite(road_right_width_m) or road_right_width_m <= 0.0:
        road_right_width_m = float(fallback_half_width_m)

    return LaneKeepingStageReference(
        x_center_m=float(sample.get("x_ref_m", 0.0)),
        y_center_m=float(sample.get("y_ref_m", 0.0)),
        heading_rad=float(sample.get("heading_rad", 0.0)),
        lane_width_m=float(lane_width_m),
        lane_id=int(sample.get("lane_id", 0)),
        road_center_offset_m=float(road_center_offset_m),
        road_left_width_m=float(road_left_width_m),
        road_right_width_m=float(road_right_width_m),
    )


def signed_lateral_offset_affine_form(
    reference: LaneKeepingStageReference,
) -> LaneKeepingAffineForm:
    """Return the affine form of the signed perpendicular lane offset."""

    lane_heading_rad = float(reference.heading_rad)
    sin_heading = math.sin(lane_heading_rad)
    cos_heading = math.cos(lane_heading_rad)
    return LaneKeepingAffineForm(
        x_coef=-sin_heading,
        y_coef=cos_heading,
        constant=(
            sin_heading * float(reference.x_center_m)
            - cos_heading * float(reference.y_center_m)
        ),
        reference=reference,
    )


def signed_lateral_offset(
    x_m: float,
    y_m: float,
    reference: LaneKeepingStageReference,
) -> float:
    """Signed perpendicular offset from the reference lane center."""

    return float(
        signed_lateral_offset_affine_form(reference).evaluate(
            x_m=float(x_m),
            y_m=float(y_m),
        )
    )


def safe_core_half_width_m(
    reference: LaneKeepingStageReference,
    safe_region_alpha: float,
) -> float:
    """Inner lane half-width where only the centering term is active."""

    return (
        _clamp_safe_region_alpha(float(safe_region_alpha))
        * float(reference.half_lane_width_m)
    )


def lane_boundary_excess_m(d_perp_m: float, d_safe_m: float) -> float:
    """Distance outside the inner safe core used by the piecewise penalty."""

    return max(
        0.0,
        abs(float(d_perp_m)) - max(0.0, float(d_safe_m)),
    )


def road_boundary_excesses_m(
    *,
    d_perp_m: float,
    reference: LaneKeepingStageReference,
    margin_m: float,
) -> Tuple[float, float]:
    """Left/right slack demand for the drivable-road boundary."""

    margin = max(0.0, float(margin_m))
    e_road_m = float(d_perp_m) - float(reference.road_center_offset_m)
    left_clearance_m = float(reference.left_road_width_m) - float(e_road_m)
    right_clearance_m = float(reference.right_road_width_m) + float(e_road_m)
    return (
        max(0.0, float(margin) - float(left_clearance_m)),
        max(0.0, float(margin) - float(right_clearance_m)),
    )


def evaluate_lane_keeping_stage(
    *,
    stage_index: int,
    x_m: float,
    y_m: float,
    reference: LaneKeepingStageReference,
    centering_weight: float,
    boundary_weight: float,
    safe_region_alpha: float,
    road_boundary_margin_m: float = 0.5,
) -> LaneKeepingStageMetrics:
    """Evaluate lane-centering and road-boundary cost at one stage."""

    d_perp_m = signed_lateral_offset(
        x_m=float(x_m),
        y_m=float(y_m),
        reference=reference,
    )
    d_max_m = max(float(reference.left_road_width_m), float(reference.right_road_width_m))
    d_safe_m = max(0.0, float(road_boundary_margin_m))
    left_excess_m, right_excess_m = road_boundary_excesses_m(
        d_perp_m=float(d_perp_m),
        reference=reference,
        margin_m=float(road_boundary_margin_m),
    )
    boundary_excess_m = max(float(left_excess_m), float(right_excess_m))

    centering_cost = max(0.0, float(centering_weight)) * float(d_perp_m) * float(d_perp_m)
    boundary_cost = max(0.0, float(boundary_weight)) * (
        float(left_excess_m) * float(left_excess_m)
        + float(right_excess_m) * float(right_excess_m)
    )
    stage_cost = float(centering_cost + boundary_cost)
    e_road_m = float(d_perp_m) - float(reference.road_center_offset_m)
    outside_road = (
        float(e_road_m) > float(reference.left_road_width_m) + 1.0e-9
        or float(e_road_m) < -float(reference.right_road_width_m) - 1.0e-9
    )

    return LaneKeepingStageMetrics(
        stage_index=int(stage_index),
        d_perp_m=float(d_perp_m),
        centering_cost=float(centering_cost),
        boundary_cost=float(boundary_cost),
        stage_cost=float(stage_cost),
        boundary_excess_m=float(boundary_excess_m),
        d_safe_m=float(d_safe_m),
        d_max_m=float(d_max_m),
        lane_width_m=float(reference.lane_width_m),
        road_center_offset_m=float(reference.road_center_offset_m),
        road_left_width_m=float(reference.left_road_width_m),
        road_right_width_m=float(reference.right_road_width_m),
        road_left_excess_m=float(left_excess_m),
        road_right_excess_m=float(right_excess_m),
        lane_id=int(reference.lane_id),
        lane_heading_rad=float(reference.heading_rad),
        inside_safe_core=(
            float(left_excess_m) <= 1.0e-9
            and float(right_excess_m) <= 1.0e-9
        ),
        outside_lane=bool(outside_road),
        outside_road=bool(outside_road),
    )


def evaluate_lane_keeping_profile(
    *,
    state_xy: Sequence[Sequence[float]],
    lane_references: Sequence[LaneKeepingStageReference | Mapping[str, object] | None],
    centering_weight: float,
    boundary_weight: float,
    safe_region_alpha: float,
    road_boundary_margin_m: float = 0.5,
    default_lane_width_m: float = 4.0,
) -> LaneKeepingProfile:
    """Evaluate d_perp, U_lane, and J_lane across a horizon."""

    stage_metrics = []
    stage_count = min(len(state_xy), len(lane_references))
    for stage_index in range(stage_count):
        state = state_xy[stage_index]
        if not isinstance(state, Sequence) or len(state) < 2:
            continue

        reference_item = lane_references[stage_index]
        if isinstance(reference_item, LaneKeepingStageReference):
            reference = reference_item
        else:
            reference = normalize_lane_reference_sample(
                reference_item,
                default_lane_width_m=float(default_lane_width_m),
            )
        if reference is None:
            continue

        stage_metrics.append(
            evaluate_lane_keeping_stage(
                stage_index=int(stage_index),
                x_m=float(state[0]),
                y_m=float(state[1]),
                reference=reference,
                centering_weight=float(centering_weight),
                boundary_weight=float(boundary_weight),
                safe_region_alpha=float(safe_region_alpha),
                road_boundary_margin_m=float(road_boundary_margin_m),
            )
        )

    total_cost = sum(float(metric.stage_cost) for metric in stage_metrics)
    return LaneKeepingProfile(
        stage_metrics=tuple(stage_metrics),
        total_cost=float(total_cost),
    )
