"""Reusable QP-friendly lane-keeping math helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Mapping, Sequence, Tuple


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


def signed_longitudinal_progress_affine_form(
    reference: LaneKeepingStageReference,
) -> LaneKeepingAffineForm:
    """Return the affine form of the signed along-track (longitudinal)
    offset from the reference point, in the lane frame -- the component
    perpendicular to signed_lateral_offset_affine_form's lateral offset.

    Used to decompose a raw world-(x,y) attraction term into lane-local
    longitudinal/lateral components instead of pulling x and y toward
    (x_center_m, y_center_m) independently, which couples world-frame x/y
    errors together and doesn't rotate with lane heading the way the
    lateral-offset term already does.
    """

    lane_heading_rad = float(reference.heading_rad)
    sin_heading = math.sin(lane_heading_rad)
    cos_heading = math.cos(lane_heading_rad)
    return LaneKeepingAffineForm(
        x_coef=cos_heading,
        y_coef=sin_heading,
        constant=-(
            cos_heading * float(reference.x_center_m)
            + sin_heading * float(reference.y_center_m)
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


@dataclass(frozen=True)
class RoadEnvelopeBlock:
    """A static hyperellipse "safe block" (Yu et al., arXiv:2509.18506, Eq 36-38).

    Used as a lane-change-local drivable-corridor primitive that stays fixed
    once built, unlike a single reference line that can jump when the
    tracked reference source switches lanes mid-maneuver.
    """

    x_center_m: float
    y_center_m: float
    heading_rad: float
    half_length_m: float
    half_width_m: float
    shape_exponent: float = 4.0

    def __post_init__(self) -> None:
        if float(self.shape_exponent) < 2.0 or int(round(float(self.shape_exponent))) % 2 != 0:
            raise ValueError("RoadEnvelopeBlock.shape_exponent must be an even integer >= 2.")


def road_envelope_block_signed_distance(
    block: RoadEnvelopeBlock,
    x_m: float,
    y_m: float,
) -> Tuple[float, float, float]:
    """Signed hyperellipse distance g_b = d_b - 1 and its exact gradient.

    g_b <= 0 means (x_m, y_m) is inside the block. shape_exponent is
    required even (see RoadEnvelopeBlock.__post_init__), so |t|^p == t^p
    everywhere -- no abs()-kink, fully smooth except at the block's own
    center (deep interior, never relevant to a boundary constraint).
    """

    p = float(block.shape_exponent)
    half_length_m = max(1.0e-6, float(block.half_length_m))
    half_width_m = max(1.0e-6, float(block.half_width_m))
    cos_h = math.cos(float(block.heading_rad))
    sin_h = math.sin(float(block.heading_rad))
    dx = float(x_m) - float(block.x_center_m)
    dy = float(y_m) - float(block.y_center_m)
    along = cos_h * dx + sin_h * dy
    across = cos_h * dy - sin_h * dx

    along_term = (along / half_length_m) ** p
    across_term = (across / half_width_m) ** p
    s_value = float(along_term + across_term)
    s_value_safe = max(s_value, 1.0e-12)

    d_along_term_dalong = p * (along / half_length_m) ** (p - 1.0) / half_length_m
    d_across_term_dacross = p * (across / half_width_m) ** (p - 1.0) / half_width_m
    ds_dx = d_along_term_dalong * cos_h - d_across_term_dacross * sin_h
    ds_dy = d_along_term_dalong * sin_h + d_across_term_dacross * cos_h

    d_value = s_value_safe ** (1.0 / p)
    dd_ds = (1.0 / p) * s_value_safe ** (1.0 / p - 1.0)
    g_b = float(d_value - 1.0)
    dg_dx = float(dd_ds * ds_dx)
    dg_dy = float(dd_ds * ds_dy)
    return g_b, dg_dx, dg_dy


def road_envelope_union_logsumexp(
    blocks: Sequence[RoadEnvelopeBlock],
    rho: float,
    x_m: float,
    y_m: float,
) -> Tuple[float, float, float, Tuple[float, ...]]:
    """Smooth OR (union) of block memberships via negative-rho LogSumExp.

    (Yu et al., Lemma 4, Eq 48.) Bounds: g_min + ln(n)/rho <= g_lse <= g_min
    for rho < 0. The gradient of a LogSumExp is exactly the softmax-weighted
    blend of the constituent gradients -- closed form, no finite differences.
    """

    rho_value = float(rho)
    if rho_value >= 0.0:
        raise ValueError("road_envelope_union_logsumexp requires rho < 0.")
    if not blocks:
        raise ValueError("road_envelope_union_logsumexp requires at least one block.")

    per_block = [
        road_envelope_block_signed_distance(block, float(x_m), float(y_m))
        for block in blocks
    ]
    g_values = [float(item[0]) for item in per_block]
    z_values = [rho_value * g for g in g_values]

    z_max = max(z_values)
    exp_shifted = [math.exp(z - z_max) for z in z_values]
    exp_sum = sum(exp_shifted)
    g_lse = float((z_max + math.log(exp_sum)) / rho_value)

    weights = tuple(float(value / exp_sum) for value in exp_shifted)
    dg_dx = sum(w * item[1] for w, item in zip(weights, per_block))
    dg_dy = sum(w * item[2] for w, item in zip(weights, per_block))
    return g_lse, float(dg_dx), float(dg_dy), weights


def road_envelope_block_boundary_probe_points_xy(
    block: RoadEnvelopeBlock,
    *,
    along_fractions: Sequence[float] = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0),
) -> List[Tuple[float, float]]:
    """World-frame points exactly on this block's own S=1 boundary surface.

    Used to compute the Theorem-1 conservativeness correction. For each
    along-axis fraction in [-1, 1], solves for the across-axis position
    that keeps the point exactly on S=1 (closed-form since shape_exponent
    is even), rather than using the (S>1, outside) bounding-rectangle
    corners.
    """

    p = float(block.shape_exponent)
    half_length_m = max(1.0e-6, float(block.half_length_m))
    half_width_m = max(1.0e-6, float(block.half_width_m))
    cos_h = math.cos(float(block.heading_rad))
    sin_h = math.sin(float(block.heading_rad))

    local_points: List[Tuple[float, float]] = []
    for frac in along_fractions:
        frac_clamped = min(1.0, max(-1.0, float(frac)))
        along = frac_clamped * half_length_m
        remainder = max(0.0, 1.0 - abs(frac_clamped) ** p)
        across_mag = half_width_m * (remainder ** (1.0 / p))
        if across_mag > 1.0e-9:
            local_points.append((along, across_mag))
            local_points.append((along, -across_mag))
        else:
            local_points.append((along, 0.0))

    world_points: List[Tuple[float, float]] = []
    for along, across in local_points:
        x_m = float(block.x_center_m) + cos_h * along - sin_h * across
        y_m = float(block.y_center_m) + sin_h * along + cos_h * across
        world_points.append((x_m, y_m))
    return world_points


def road_envelope_conservativeness_correction(
    blocks: Sequence[RoadEnvelopeBlock],
    rho: float,
) -> float:
    """Theorem-1 correction epsilon0: min g_LSE over each block's own boundary.

    Enforcing g_LSE(x,y) - epsilon0 <= 0 at runtime guarantees the resulting
    feasible set stays inside the true (exact) union of blocks despite the
    LogSumExp union's smooth optimism at the seam between blocks.
    """

    probe_points = [
        point
        for block in blocks
        for point in road_envelope_block_boundary_probe_points_xy(block)
    ]
    if not probe_points:
        return 0.0
    lse_values = [
        road_envelope_union_logsumexp(blocks, float(rho), x_m, y_m)[0]
        for x_m, y_m in probe_points
    ]
    return float(min(lse_values))
