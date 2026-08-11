"""Hard safety contract for references handed to MPC."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class ReferenceContract:
    mode: str
    expected_lane_id: int
    horizon_steps: int
    min_first_forward_m: float
    max_first_lateral_abs_m: float
    max_destination_lane_error_m: float
    max_destination_body_lateral_abs_m: Optional[float]
    max_point_jump_m: float
    max_heading_jump_rad: float
    max_curvature_1pm: float
    max_speed_mps: float
    require_monotonic_progress: bool = True
    require_zero_terminal_speed: bool = False
    allow_lane_transition: bool = False
    allow_route_branch: bool = False
    allow_padding: bool = True


@dataclass
class ReferenceValidationResult:
    valid: bool
    violations: list[str] = field(default_factory=list)
    first_forward_m: float = 0.0
    first_lateral_m: float = 0.0
    destination_body_lateral_m: float = 0.0
    destination_lane_error_m: float = 0.0
    max_point_jump_m: float = 0.0
    max_heading_jump_rad: float = 0.0
    max_curvature_1pm: float = 0.0
    contract_max_curvature_1pm: float = 0.0
    curvature_margin_1pm: float = 0.0

    def reason(self) -> str:
        return ";".join(dict.fromkeys(str(v) for v in self.violations if str(v)))


def contract_from_config(
    *,
    mode: str,
    expected_lane_id: int,
    horizon_steps: int,
    config: Mapping[str, object],
    default_speed_mps: float,
) -> ReferenceContract:
    normalized_mode = str(mode or "lane_follow").strip().lower()
    prefix = f"reference_contract_{normalized_mode}_"
    defaults = _mode_defaults(normalized_mode)
    max_body_default = defaults["max_destination_body_lateral_abs_m"]
    max_body_value = config.get(
        prefix + "max_destination_body_lateral_abs_m",
        max_body_default,
    )
    max_body = None if max_body_value is None else float(max_body_value)
    configured_max_curvature_1pm = float(
        config.get(
            prefix + "max_curvature_1pm",
            defaults["max_curvature_1pm"],
        )
    )
    vehicle_max_curvature = config.get("reference_vehicle_max_curvature_1pm")
    if vehicle_max_curvature is not None:
        configured_max_curvature_1pm = min(
            float(configured_max_curvature_1pm),
            max(1.0e-3, float(vehicle_max_curvature)),
        )
    return ReferenceContract(
        mode=normalized_mode,
        expected_lane_id=int(expected_lane_id),
        horizon_steps=int(horizon_steps),
        min_first_forward_m=float(config.get(prefix + "min_first_forward_m", defaults["min_first_forward_m"])),
        max_first_lateral_abs_m=float(config.get(prefix + "max_first_lateral_abs_m", defaults["max_first_lateral_abs_m"])),
        max_destination_lane_error_m=float(config.get(prefix + "max_destination_lane_error_m", defaults["max_destination_lane_error_m"])),
        max_destination_body_lateral_abs_m=max_body,
        max_point_jump_m=float(config.get(prefix + "max_point_jump_m", defaults["max_point_jump_m"])),
        max_heading_jump_rad=float(config.get(prefix + "max_heading_jump_rad", defaults["max_heading_jump_rad"])),
        max_curvature_1pm=float(configured_max_curvature_1pm),
        max_speed_mps=float(config.get(prefix + "max_speed_mps", default_speed_mps)),
        require_monotonic_progress=bool(
            config.get(
                prefix + "require_monotonic_progress",
                defaults.get("require_monotonic_progress", True),
            )
        ),
        require_zero_terminal_speed=bool(config.get(prefix + "require_zero_terminal_speed", normalized_mode == "stop")),
        allow_lane_transition=bool(config.get(prefix + "allow_lane_transition", normalized_mode in ("lane_change", "lane_change_direct"))),
        allow_route_branch=bool(config.get(prefix + "allow_route_branch", normalized_mode == "intersection_turn")),
        allow_padding=bool(config.get(prefix + "allow_padding", True)),
    )


def validate_reference_contract(
    *,
    reference_samples: Sequence[Mapping[str, object]],
    destination_state: Sequence[float] | None,
    ego_state: Sequence[float],
    contract: ReferenceContract,
    check_destination_body_lateral: bool,
) -> ReferenceValidationResult:
    samples = [dict(sample) for sample in list(reference_samples or [])]
    result = ReferenceValidationResult(
        valid=True,
        contract_max_curvature_1pm=float(contract.max_curvature_1pm),
        curvature_margin_1pm=float(contract.max_curvature_1pm),
    )
    violations: list[str] = []
    if len(ego_state) < 4:
        return ReferenceValidationResult(valid=False, violations=["missing_ego_state"])
    if not samples:
        return ReferenceValidationResult(valid=False, violations=["empty_reference"])
    if len(samples) < int(contract.horizon_steps) and not bool(contract.allow_padding):
        violations.append("horizon_too_short")

    points: list[tuple[float, float]] = []
    point_samples: list[Mapping[str, object]] = []
    speeds: list[float] = []
    lane_ids: list[int] = []
    progress_values: list[float] = []
    for index, sample in enumerate(samples):
        try:
            x_m = float(sample.get("x_ref_m", sample.get("x", "")))
            y_m = float(sample.get("y_ref_m", sample.get("y", "")))
        except Exception:
            violations.append(f"sample_{index}_not_finite")
            continue
        if not math.isfinite(x_m) or not math.isfinite(y_m):
            violations.append(f"sample_{index}_not_finite")
            continue
        points.append((x_m, y_m))
        point_samples.append(sample)
        lane_ids.append(_to_int(sample.get("lane_id", contract.expected_lane_id), contract.expected_lane_id))
        speed = _sample_speed_mps(sample)
        if speed is not None:
            speeds.append(float(speed))
            if float(speed) < -1.0e-3:
                violations.append("negative_speed")
            if float(speed) > float(contract.max_speed_mps) + 1.0e-3:
                violations.append("speed_above_contract")
        forward_m, lateral_m = _body_frame_xy(
            ego_state=ego_state,
            target_x_m=x_m,
            target_y_m=y_m,
        )
        progress_values.append(float(forward_m))
        if index == 0:
            result.first_forward_m = float(forward_m)
            result.first_lateral_m = float(lateral_m)

    if not points:
        return ReferenceValidationResult(valid=False, violations=["no_valid_points"])

    if result.first_forward_m < float(contract.min_first_forward_m):
        violations.append("first_forward_before_contract")
    if abs(result.first_lateral_m) > float(contract.max_first_lateral_abs_m):
        violations.append("first_lateral_out_of_contract")
    if (
        not bool(contract.allow_lane_transition)
        and not bool(contract.allow_route_branch)
        and int(contract.expected_lane_id) != 0
    ):
        bad_lanes = [
            lane_id
            for lane_id, sample in zip(lane_ids, point_samples)
            if (
                int(lane_id) != 0
                and int(lane_id) != int(contract.expected_lane_id)
                and not _is_longitudinal_lane_successor(sample)
            )
        ]
        if bad_lanes:
            violations.append("lane_id_transition_not_allowed")

    if bool(contract.require_monotonic_progress):
        for previous, current in zip(progress_values[:-1], progress_values[1:]):
            if float(current) + 1.0e-3 < float(previous):
                violations.append("non_monotonic_progress")
                break

    headings: list[float] = []
    for first, second in zip(points[:-1], points[1:]):
        distance_m = math.hypot(second[0] - first[0], second[1] - first[1])
        if distance_m <= 1.0e-6:
            violations.append("zero_point_spacing")
            continue
        result.max_point_jump_m = max(result.max_point_jump_m, float(distance_m))
        if distance_m > float(contract.max_point_jump_m):
            violations.append("point_jump_out_of_contract")
        headings.append(math.atan2(second[1] - first[1], second[0] - first[0]))

    for index, (prev_heading, cur_heading) in enumerate(zip(headings[:-1], headings[1:])):
        delta = abs(_wrap_angle(cur_heading - prev_heading))
        result.max_heading_jump_rad = max(result.max_heading_jump_rad, float(delta))
        if delta > float(contract.max_heading_jump_rad):
            violations.append("heading_jump_out_of_contract")
        ds = max(1.0e-6, math.hypot(points[index + 2][0] - points[index + 1][0], points[index + 2][1] - points[index + 1][1]))
        curvature = float(delta) / float(ds)
        result.max_curvature_1pm = max(result.max_curvature_1pm, float(curvature))
        if curvature > float(contract.max_curvature_1pm):
            violations.append("curvature_out_of_contract")
    result.curvature_margin_1pm = (
        float(contract.max_curvature_1pm) - float(result.max_curvature_1pm)
    )

    if destination_state is not None and len(destination_state) >= 2:
        try:
            dest_x = float(destination_state[0])
            dest_y = float(destination_state[1])
            _, dest_lateral = _body_frame_xy(
                ego_state=ego_state,
                target_x_m=dest_x,
                target_y_m=dest_y,
            )
            result.destination_body_lateral_m = float(dest_lateral)
            result.destination_lane_error_m = _nearest_lane_error_m(
                x_m=dest_x,
                y_m=dest_y,
                reference_points=points,
                reference_lane_ids=lane_ids,
                reference_samples=point_samples,
                expected_lane_id=(
                    0
                    if bool(contract.allow_route_branch)
                    else int(contract.expected_lane_id)
                ),
            )
            if result.destination_lane_error_m > float(contract.max_destination_lane_error_m):
                violations.append("destination_lane_error_out_of_contract")
            if (
                bool(check_destination_body_lateral)
                and contract.max_destination_body_lateral_abs_m is not None
                and abs(result.destination_body_lateral_m) > float(contract.max_destination_body_lateral_abs_m)
            ):
                violations.append("destination_body_lateral_out_of_contract")
        except Exception:
            violations.append("destination_not_finite")

    if bool(contract.require_zero_terminal_speed):
        terminal_speed = speeds[-1] if speeds else _sample_speed_mps(samples[-1])
        if terminal_speed is None or abs(float(terminal_speed)) > 1.0e-3:
            violations.append("terminal_speed_not_zero")
        for previous, current in zip(speeds[:-1], speeds[1:]):
            if float(current) > float(previous) + 1.0e-3:
                violations.append("stop_speed_not_monotonic")
                break

    result.violations = list(dict.fromkeys(violations))
    result.valid = not bool(result.violations)
    return result


def _is_longitudinal_lane_successor(sample: Mapping[str, object]) -> bool:
    """Return whether a lane-id change is a trusted CARLA topology successor."""

    transition_kind = str(sample.get("lane_transition_kind", "")).strip().lower()
    return transition_kind == "longitudinal_successor"


def _mode_defaults(mode: str) -> Mapping[str, object]:
    common = {
        "max_point_jump_m": 4.0,
        "max_heading_jump_rad": 0.75,
        "max_curvature_1pm": 0.35,
    }
    table = {
        "lane_follow": {
            "min_first_forward_m": 0.5,
            "max_first_lateral_abs_m": 0.75,
            "max_destination_lane_error_m": 0.75,
            "max_destination_body_lateral_abs_m": 1.5,
        },
        "stop": {
            "min_first_forward_m": 0.2,
            "max_first_lateral_abs_m": 0.75,
            "max_destination_lane_error_m": 0.75,
            "max_destination_body_lateral_abs_m": 1.5,
        },
        "lane_change": {
            "min_first_forward_m": 0.2,
            "max_first_lateral_abs_m": 1.25,
            "max_destination_lane_error_m": 0.75,
            "max_destination_body_lateral_abs_m": None,
        },
        "lane_change_direct": {
            # Used when MPC tracks the target lane's own (unblended)
            # centerline directly instead of a pre-shaped source-to-target
            # blend -- the first reference sample legitimately sits close to
            # a full lane width from ego at lock time, so the "lane_change"
            # mode's tighter 1.25m limit (sized for an already-ramping
            # blend) would veto every such reference. Sized to cover one
            # lane width plus margin.
            "min_first_forward_m": 0.2,
            "max_first_lateral_abs_m": 4.0,
            "max_destination_lane_error_m": 0.75,
            "max_destination_body_lateral_abs_m": None,
        },
        "intersection_turn": {
            "min_first_forward_m": 0.2,
            "max_first_lateral_abs_m": 1.5,
            "max_destination_lane_error_m": 1.0,
            "max_destination_body_lateral_abs_m": None,
            "max_heading_jump_rad": 0.95,
            "max_curvature_1pm": 0.55,
            # Route progress is enforced by the monotonic CARLA route index.
            # Ego-body forward distance is not monotonic around a real turn.
            "require_monotonic_progress": False,
        },
        "emergency_stop": {
            "min_first_forward_m": 0.1,
            "max_first_lateral_abs_m": 0.75,
            "max_destination_lane_error_m": 1.0,
            "max_destination_body_lateral_abs_m": None,
        },
    }
    defaults = dict(common)
    defaults.update(table.get(str(mode), table["lane_follow"]))
    return defaults


def _body_frame_xy(
    *,
    ego_state: Sequence[float],
    target_x_m: float,
    target_y_m: float,
) -> tuple[float, float]:
    ego_x_m = float(ego_state[0])
    ego_y_m = float(ego_state[1])
    heading_rad = float(ego_state[3])
    dx_m = float(target_x_m) - ego_x_m
    dy_m = float(target_y_m) - ego_y_m
    forward_m = math.cos(heading_rad) * dx_m + math.sin(heading_rad) * dy_m
    lateral_m = -math.sin(heading_rad) * dx_m + math.cos(heading_rad) * dy_m
    return float(forward_m), float(lateral_m)


def _nearest_lane_error_m(
    *,
    x_m: float,
    y_m: float,
    reference_points: Sequence[tuple[float, float]],
    reference_lane_ids: Sequence[int],
    reference_samples: Sequence[Mapping[str, object]],
    expected_lane_id: int,
) -> float:
    candidates = [
        point
        for point, lane_id, sample in zip(
            reference_points,
            reference_lane_ids,
            reference_samples,
        )
        if (
            int(expected_lane_id) == 0
            or int(lane_id) == 0
            or int(lane_id) == int(expected_lane_id)
            or _is_longitudinal_lane_successor(sample)
        )
    ]
    if not candidates:
        candidates = list(reference_points)
    if not candidates:
        return float("inf")
    return min(math.hypot(float(x_m) - px, float(y_m) - py) for px, py in candidates)


def _sample_speed_mps(sample: Mapping[str, object]) -> Optional[float]:
    for key in ("speed_ref_mps", "v_ref_mps", "speed_mps", "v"):
        if key not in sample:
            continue
        try:
            value = float(sample[key])
        except Exception:
            return None
        if math.isfinite(value):
            return float(value)
    return None


def _to_int(value: object, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _wrap_angle(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
