"""Trapezoidal stop speed profile for smooth deceleration to a stop line.

Mirrors the simplified Apollo speed DP used for stop-line approach planning.
Produces a time-indexed sequence of speed caps so that the vehicle arrives at
the stop line at zero speed with a comfortable deceleration profile.

Usage:
    profile = trapezoidal_stop_profile(current_v=8.0, distance_to_stop_m=30.0)
    v_cap_now = profile[0]   # allowable speed at this tick
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Sequence


def trapezoidal_stop_profile(
    current_v: float,
    distance_to_stop_m: float,
    *,
    a_decel: float = 2.5,
    stop_buffer_m: float = 1.5,
    n_steps: int = 20,
    dt_s: float = 0.1,
) -> list[float]:
    """Return per-step speed caps [m/s] for decelerating to a stop line.

    Element i is the maximum allowable speed at time (i+1)*dt_s from now.
    The profile uses the kinematic constraint v <= sqrt(2 * a_decel * d).
    This is a speed cap, not a target speed: it must not be clamped by the
    current ego speed, otherwise a vehicle that has already slowed too early
    can get stuck with a near-zero cap while the stop line is still far ahead.

    Args:
        current_v: current ego speed [m/s]
        distance_to_stop_m: distance to the stop line [m]
        a_decel: deceleration magnitude [m/s^2]
        stop_buffer_m: buffer before stop line where speed must reach 0 [m]
        n_steps: number of time steps to project
        dt_s: time step duration [s]

    Returns:
        List of n_steps speed caps in m/s.  First element is the immediate
        constraint; subsequent elements tighten as the projected distance
        shrinks.
    """
    # Keep the argument for API compatibility and documentation of the caller's
    # intent, but do not use it as a cap.  The cap is determined by remaining
    # distance and comfortable deceleration only.
    _ = max(0.0, float(current_v))
    d = max(0.0, float(distance_to_stop_m) - float(stop_buffer_m))
    a_decel = max(0.1, float(a_decel))
    dt = max(1e-3, float(dt_s))

    profile: list[float] = []
    for _ in range(int(n_steps)):
        v_brake_cap = math.sqrt(max(0.0, 2.0 * a_decel * d))
        profile.append(float(v_brake_cap))
        d = max(0.0, d - float(v_brake_cap) * dt)

    return profile


def idm_following_speed_cap_mps(
    *,
    idm_accel_mps2: float | None,
    idm_gap_m: float,
    is_fixed_stop: bool,
    stop_decision_active: bool,
    signal_state: str,
    idm_non_stop_buffer_m: float = 3.0,
    idm_speed_cap_braking_deceleration_mps2: float = 2.5,
    idm_non_stop_min_speed_cap_mps: float = 2.0,
    current_target_v_mps: float = 0.0,
    idm_lead_v_mps: float = 0.0,
) -> float | None:
    """Speed cap [m/s] when IDM wants to brake for a lead vehicle.

    Returns ``None`` when IDM is not currently braking (``idm_accel_mps2``
    is missing or not below the braking-intent threshold), meaning this cap
    source does not apply this cycle.
    """
    if idm_accel_mps2 is None or float(idm_accel_mps2) >= -0.3:
        return None

    gap_m = max(0.0, float(idm_gap_m))
    stop_context = bool(is_fixed_stop) or (
        bool(stop_decision_active)
        and str(signal_state).strip().lower() in {"red", "yellow"}
    )
    buffer_m = 6.0 if stop_context else float(idm_non_stop_buffer_m)
    brake_mps2 = max(0.5, float(idm_speed_cap_braking_deceleration_mps2))
    v_cap = math.sqrt(max(0.0, 2.0 * brake_mps2 * max(0.0, gap_m - buffer_m)))

    if not stop_context:
        min_release_speed_mps = float(idm_non_stop_min_speed_cap_mps)
        if gap_m >= buffer_m + 2.0:
            v_cap = max(min_release_speed_mps, v_cap)
        v_cap = max(v_cap, min(float(current_target_v_mps), float(idm_lead_v_mps) + 1.5))

    return float(v_cap)


def stop_profile_speed_cap_mps(
    *,
    is_fixed_stop: bool,
    stop_target_distance_m: float | None,
    current_speed_mps: float,
    min_acceleration_mps2: float,
    max_stop_target_distance_m: float = 60.0,
    stop_buffer_m: float = 1.5,
) -> float | None:
    """Speed cap [m/s] from the trapezoidal stop profile, or ``None`` when a
    fixed stop is not active or the stop line is too far to matter yet.
    """
    if not bool(is_fixed_stop) or stop_target_distance_m is None:
        return None
    if float(stop_target_distance_m) >= float(max_stop_target_distance_m):
        return None

    profile = trapezoidal_stop_profile(
        current_v=float(current_speed_mps),
        distance_to_stop_m=float(stop_target_distance_m),
        a_decel=max(1.0, abs(float(min_acceleration_mps2))),
        stop_buffer_m=float(stop_buffer_m),
    )
    if not profile:
        return None
    return float(profile[0])


def curvature_speed_cap_mps(
    *,
    curve_curvature_abs: float,
    curve_min_curvature: float,
    current_speed_mps: float,
    curve_lateral_accel_limit_mps2: float = 1.3,
    speed_enable_threshold_mps: float = 1.0,
) -> float | None:
    """Speed cap [m/s] that keeps lateral acceleration bounded on a curve,
    or ``None`` when moving too slowly to matter or the path is too straight.
    """
    if float(current_speed_mps) <= float(speed_enable_threshold_mps):
        return None
    if float(curve_curvature_abs) <= float(curve_min_curvature):
        return None

    accel_limit_mps2 = max(0.1, float(curve_lateral_accel_limit_mps2))
    return float(math.sqrt(accel_limit_mps2 / max(1.0e-6, float(curve_curvature_abs))))


def reference_jump_speed_cap_mps(
    *,
    is_fixed_stop: bool,
    last_reference_jump_m: float,
    reference_jump_speed_cap_threshold_m: float = 1.5,
    configured_cap_mps: float = 5.0,
) -> float | None:
    """Speed cap [m/s] applied for one cycle right after the lane reference
    path jumps, so MPC doesn't chase a discontinuous reference at speed.
    Returns ``None`` when the jump is within the tolerated threshold.
    """
    if bool(is_fixed_stop):
        return None
    if float(last_reference_jump_m) <= float(reference_jump_speed_cap_threshold_m):
        return None
    return max(1.5, float(configured_cap_mps))


def rate_limit_speed_cap_rise_mps(
    *,
    previous_cap_mps: float,
    new_cap_mps: float,
    dt_s: float,
    max_rise_mps2: float = 2.0,
) -> float:
    """Limit how fast the combined speed cap may *increase* tick to tick.

    Independent caps (IDM following, stop profile, curvature, ...) are each
    recomputed from scratch every cycle and combined by taking the tightest.
    When the binding cap is something noisy like a lead vehicle's
    fluctuating gap (IDM), the combined cap can swing down and back up by
    several m/s within a couple of ticks even while the vehicle is otherwise
    just sitting at a comfortable, smoothly-shrinking stop-profile distance.
    That sawtooth reads to the MPC as alternating hard-brake / floor-it
    commands, which in turn produces exaggerated lane-center corrective
    steering once the vehicle's position has drifted during the jerk.

    Only the *rise* is rate-limited -- a cap that needs to drop (a new
    obstacle, a tighter stop requirement, an emergency brake) always takes
    effect immediately. This cannot delay a safety-relevant tightening, it
    only smooths how quickly the vehicle is allowed to speed back up after
    one of these caps briefly bound.
    """
    if float(new_cap_mps) <= float(previous_cap_mps):
        return float(new_cap_mps)
    max_rise_mps = max(0.0, float(max_rise_mps2)) * max(0.0, float(dt_s))
    return float(min(float(new_cap_mps), float(previous_cap_mps) + float(max_rise_mps)))


def apply_sequential_speed_caps(
    base_max_velocity_mps: float,
    candidates: "list[tuple[str, float | None, float | None]]",
) -> "tuple[float, list[str]]":
    """Tighten ``base_max_velocity_mps`` by each applicable candidate cap, in
    order, mirroring the historical inline
    ``if cap < mpc.constraints.max_velocity_mps: mpc.constraints.max_velocity_mps = cap``
    stacking this replaces.

    Each candidate is ``(name, value_or_None, floor_or_None)``:
      - ``value`` of ``None`` means that cap source does not apply this cycle
        and is skipped.
      - a non-``None`` ``floor`` is applied *after* the tightening decision,
        not before -- e.g. the curvature cap floors the assigned value at
        1.5 m/s even when the raw curvature-implied cap is lower, but only
        once it has already been decided that the curvature cap is the
        tightest constraint so far. This matches pre-existing behavior and
        is intentionally preserved, not a new rule.

    Returns the final capped speed and the ordered list of cap names that
    actually tightened the running value.
    """
    current_mps = float(base_max_velocity_mps)
    applied: list[str] = []
    for name, value, floor in candidates:
        if value is None:
            continue
        if float(value) < current_mps:
            current_mps = float(value) if floor is None else max(float(floor), float(value))
            applied.append(str(name))
    return current_mps, applied


@dataclass(frozen=True)
class SpeedCapTrace:
    """Compact trace of the stacked speed-envelope decision for one tick."""

    base_max_velocity_mps: float
    final_max_velocity_mps: float
    final_after_rise_limit_mps: float
    active_caps: list[str]
    binding_cap: str
    idm_cap_mps: float | None = None
    ego_corridor_obstacle_cap_mps: float | None = None
    stop_profile_cap_mps: float | None = None
    curvature_cap_mps: float | None = None
    reference_jump_cap_mps: float | None = None
    rise_limited: bool = False
    rise_dt_s: float = 0.0
    path_speed_cap_reason: str = ""

    def as_trace_fields(self) -> Dict[str, object]:
        return {
            "speed_cap_base_max_velocity_mps": float(self.base_max_velocity_mps),
            "speed_cap_final_raw_mps": float(self.final_max_velocity_mps),
            "speed_cap_final_after_rise_limit_mps": float(self.final_after_rise_limit_mps),
            "speed_cap_active_caps": "|".join(str(name) for name in self.active_caps),
            "speed_cap_binding_cap": str(self.binding_cap),
            "speed_cap_idm_mps": "" if self.idm_cap_mps is None else float(self.idm_cap_mps),
            "speed_cap_ego_corridor_obstacle_mps": (
                "" if self.ego_corridor_obstacle_cap_mps is None else float(self.ego_corridor_obstacle_cap_mps)
            ),
            "speed_cap_stop_profile_mps": (
                "" if self.stop_profile_cap_mps is None else float(self.stop_profile_cap_mps)
            ),
            "speed_cap_curvature_mps": "" if self.curvature_cap_mps is None else float(self.curvature_cap_mps),
            "speed_cap_reference_jump_mps": (
                "" if self.reference_jump_cap_mps is None else float(self.reference_jump_cap_mps)
            ),
            "speed_cap_rise_limited": int(bool(self.rise_limited)),
            "speed_cap_rise_dt_s": float(self.rise_dt_s),
            "speed_cap_path_reason": str(self.path_speed_cap_reason),
        }


def build_speed_cap_trace(
    *,
    base_max_velocity_mps: float,
    final_raw_mps: float,
    final_after_rise_limit_mps: float,
    active_caps: Sequence[str],
    idm_cap_mps: float | None,
    ego_corridor_obstacle_cap_mps: float | None,
    stop_profile_cap_mps: float | None,
    curvature_cap_mps: float | None,
    reference_jump_cap_mps: float | None,
    rise_dt_s: float,
    path_speed_cap_reason: str = "",
) -> SpeedCapTrace:
    """Build a trace for the stacked speed caps after rate limiting."""

    active_cap_list = [str(name) for name in list(active_caps or [])]
    binding_cap = active_cap_list[-1] if active_cap_list else "base"
    rise_limited = float(final_after_rise_limit_mps) < float(final_raw_mps) - 1.0e-6
    return SpeedCapTrace(
        base_max_velocity_mps=float(base_max_velocity_mps),
        final_max_velocity_mps=float(final_raw_mps),
        final_after_rise_limit_mps=float(final_after_rise_limit_mps),
        active_caps=active_cap_list,
        binding_cap=str(binding_cap),
        idm_cap_mps=None if idm_cap_mps is None else float(idm_cap_mps),
        ego_corridor_obstacle_cap_mps=(
            None if ego_corridor_obstacle_cap_mps is None else float(ego_corridor_obstacle_cap_mps)
        ),
        stop_profile_cap_mps=None if stop_profile_cap_mps is None else float(stop_profile_cap_mps),
        curvature_cap_mps=None if curvature_cap_mps is None else float(curvature_cap_mps),
        reference_jump_cap_mps=None if reference_jump_cap_mps is None else float(reference_jump_cap_mps),
        rise_limited=bool(rise_limited),
        rise_dt_s=float(rise_dt_s),
        path_speed_cap_reason=str(path_speed_cap_reason),
    )


def _path_speed_cap_reason(active_caps: Sequence[str], ego_corridor_cap_reason: str = "") -> str:
    active = {str(name) for name in list(active_caps or [])}
    if "reference_jump" in active:
        return "reference_jump"
    if "curvature" in active:
        return "curve_curvature"
    if "ego_corridor_obstacle" in active:
        return str(ego_corridor_cap_reason)
    return ""


def compute_speed_envelope(
    *,
    base_max_velocity_mps: float,
    previous_max_velocity_mps: float,
    dt_s: float,
    max_rise_mps2: float,
    idm_cap_mps: float | None,
    ego_corridor_obstacle_cap_mps: float | None,
    ego_corridor_cap_reason: str = "",
    stop_profile_cap_mps: float | None,
    curvature_cap_mps: float | None,
    reference_jump_cap_mps: float | None,
    curvature_floor_mps: float = 1.5,
) -> SpeedCapTrace:
    """Compute the final speed envelope and return a traceable contract.

    This is the speed-layer boundary consumed by the runner.  It owns cap
    stacking order, curvature floor semantics, and rise-rate limiting, so the
    runner does not need to duplicate speed-policy mechanics inline.
    """

    final_raw_mps, active_caps = apply_sequential_speed_caps(
        float(base_max_velocity_mps),
        [
            ("idm", idm_cap_mps, None),
            ("ego_corridor_obstacle", ego_corridor_obstacle_cap_mps, None),
            ("stop_profile", stop_profile_cap_mps, None),
            ("curvature", curvature_cap_mps, float(curvature_floor_mps)),
            ("reference_jump", reference_jump_cap_mps, None),
        ],
    )
    final_after_rise_limit_mps = rate_limit_speed_cap_rise_mps(
        previous_cap_mps=float(previous_max_velocity_mps),
        new_cap_mps=float(final_raw_mps),
        dt_s=float(dt_s),
        max_rise_mps2=float(max_rise_mps2),
    )
    return build_speed_cap_trace(
        base_max_velocity_mps=float(base_max_velocity_mps),
        final_raw_mps=float(final_raw_mps),
        final_after_rise_limit_mps=float(final_after_rise_limit_mps),
        active_caps=active_caps,
        idm_cap_mps=idm_cap_mps,
        ego_corridor_obstacle_cap_mps=ego_corridor_obstacle_cap_mps,
        stop_profile_cap_mps=stop_profile_cap_mps,
        curvature_cap_mps=curvature_cap_mps,
        reference_jump_cap_mps=reference_jump_cap_mps,
        rise_dt_s=float(dt_s),
        path_speed_cap_reason=_path_speed_cap_reason(
            active_caps,
            ego_corridor_cap_reason=str(ego_corridor_cap_reason),
        ),
    )


def summarize_speed_cap_history(history: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    """Aggregate speed-cap trace rows into a compact run summary."""

    rows = [dict(row) for row in list(history or [])]
    binding_counter: Counter[str] = Counter()
    active_counter: Counter[str] = Counter()
    path_reason_counter: Counter[str] = Counter()
    rise_limited_count = 0
    min_final_cap_mps = None
    max_final_cap_mps = 0.0
    min_nonzero_final_cap_mps = None

    for row in rows:
        binding = str(row.get("speed_cap_binding_cap", ""))
        if binding:
            binding_counter[binding] += 1
        path_reason = str(row.get("speed_cap_path_reason", ""))
        if path_reason:
            path_reason_counter[path_reason] += 1
        for name in str(row.get("speed_cap_active_caps", "")).split("|"):
            if name:
                active_counter[name] += 1
        try:
            rise_limited_count += int(bool(int(row.get("speed_cap_rise_limited", 0) or 0)))
        except Exception:
            rise_limited_count += int(bool(row.get("speed_cap_rise_limited", False)))
        try:
            final_cap = float(row.get("speed_cap_final_after_rise_limit_mps", 0.0) or 0.0)
            min_final_cap_mps = final_cap if min_final_cap_mps is None else min(float(min_final_cap_mps), final_cap)
            max_final_cap_mps = max(float(max_final_cap_mps), final_cap)
            if final_cap > 1.0e-6:
                min_nonzero_final_cap_mps = (
                    final_cap
                    if min_nonzero_final_cap_mps is None
                    else min(float(min_nonzero_final_cap_mps), final_cap)
                )
        except Exception:
            pass

    total_rows = len(rows)
    return {
        "samples": int(total_rows),
        "rise_limited_samples": int(rise_limited_count),
        "rise_limited_rate": 0.0 if total_rows <= 0 else float(rise_limited_count) / float(total_rows),
        "min_final_cap_mps": "" if min_final_cap_mps is None else float(min_final_cap_mps),
        "max_final_cap_mps": float(max_final_cap_mps),
        "min_nonzero_final_cap_mps": (
            "" if min_nonzero_final_cap_mps is None else float(min_nonzero_final_cap_mps)
        ),
        "binding_caps": dict(binding_counter.most_common()),
        "active_caps": dict(active_counter.most_common()),
        "path_speed_cap_reasons": dict(path_reason_counter.most_common()),
    }
