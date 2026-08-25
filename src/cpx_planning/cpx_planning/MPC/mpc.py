"""
LTV-MPC (QP + OSQP) for MPC_custom.

Model and optimization summary:
1. State:  X_k = [x_k, y_k, v_k, psi_k]
2. Input:  U_k = [a_k, delta_k]
3. Nonlinear kinematic bicycle model (CG-reference with slip angle beta) is
   used for reference rollout.
4. Dynamics are linearized around the reference rollout (LTV form).
5. The resulting convex QP is solved with OSQP.
6. Output is the future state sequence only (no control sequence returned).

Cost function:
    J_total = sum_{k=1..N} (
        Cost_ref + Cost_LaneCenter + Cost_RoadBoundary + Cost_Repulsive + Cost_Control
    )

    Cost_ref:
      quadratic pull toward destination reference state.

    Cost_LaneCenter:
      soft lane-center tracking term when enabled.

    Cost_RoadBoundary:
      piecewise-quadratic road-edge proximity penalty using nonnegative slacks.

    Cost_Repulsive:
      obstacle repulsive potential field, approximated in QP form.

    Cost_Control:
      control smoothness cost using acceleration-rate and steering-rate penalties.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import time
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .lane_keep import (
    LaneKeepingProfile,
    RoadEnvelopeBlock,
    evaluate_lane_keeping_profile,
    normalize_lane_reference_sample,
    road_envelope_union_logsumexp,
    signed_lateral_offset_affine_form,
    signed_longitudinal_progress_affine_form,
)

try:
    import osqp

    _OSQP_AVAILABLE = True
except Exception:  # pragma: no cover - import error path depends on environment
    osqp = None  # type: ignore[assignment]
    _OSQP_AVAILABLE = False



@dataclass
class MPCConstraintSpec:
    """Hard bounds and enabled safety constraints used by the QP."""

    min_velocity_mps: float
    max_velocity_mps: float
    min_acceleration_mps2: float
    max_acceleration_mps2: float
    max_jerk_mps3: float
    min_steer_rad: float
    max_steer_rad: float
    min_steer_rate_rps: float
    max_steer_rate_rps: float
    enforce_terminal_velocity_constraint: bool
    terminal_velocity_mps: float


@dataclass
class MPCComfortCostSpec:
    """Comfort cost weights for J_ctrl (reference tracking is in J_safe by user request)."""

    w_comf: float
    qx: float
    qy: float
    qv: float
    qpsi: float
    qa: float
    qdelta: float


@dataclass
class MPCSafetyCostSpec:
    """Safety cost top-level weight for reference-state tracking."""

    w_safe: float


@dataclass
class MPCRepulsivePotentialSpec:
    """
    Super-ellipsoid collision-potential configuration.

    For each obstacle and stage:
        J_obs = w_c * exp(-k_c * (r_c - s_c))

    where:
        r_c : normalized distance to the tighter collision zone
        w_c : collision-zone weight
        k_c : collision-zone exponential gain
        s_c : collision-zone distance shift
    """

    enabled: bool
    w_safe_zone: float
    w_collision_zone: float
    safe_exponential_gain: float
    safe_distance_shift: float
    collision_exponential_gain: float
    collision_distance_shift: float
    max_braking_deceleration_mps2: float
    comfort_deceleration_mps2: float
    reaction_time_s: float
    static_longitudinal_buffer_m: float
    static_lateral_buffer_m: float
    shape_exponent: float
    min_lateral_approach_speed_mps: float
    max_longitudinal_zone_length_m: float
    limit_lateral_zone_to_lane_width: bool
    max_lateral_zone_lane_fraction: float
    project_hessian_psd: bool
    min_hessian_eig: float
    log_barrier_enabled: bool
    log_barrier_replace_exponential: bool
    w_log_barrier: float
    log_barrier_gain: float
    # Off by default. When on, the cost's cross-track (lateral, relative to
    # ego's own stage heading) gradient/Hessian contribution is scaled down
    # for a nearly-straight-ahead obstacle, leaving the along-track (braking)
    # contribution untouched -- see _cross_track_lateral_scale.
    cross_track_suppression_enabled: bool
    cross_track_full_suppression_m: float
    cross_track_full_response_m: float


@dataclass
class QPIndex:
    """
    Decision-variable indexing helper.

    Variable layout:
        z = [X(0..N), U(0..N-1), S_road_left/right(1..N), S_speed(1..N), S_envelope(1..N)]
    """

    nx: int
    nu: int
    horizon_steps: int
    road_boundary_slack_pair_count: int = 0
    speed_slack_count: int = 0
    road_envelope_slack_count: int = 0

    @property
    def state_offset(self) -> int:
        return 0

    @property
    def control_offset(self) -> int:
        return (self.horizon_steps + 1) * self.nx

    @property
    def road_boundary_slack_offset(self) -> int:
        return self.control_offset + self.horizon_steps * self.nu

    @property
    def road_boundary_slack_count(self) -> int:
        return 2 * int(self.road_boundary_slack_pair_count)

    @property
    def speed_slack_offset(self) -> int:
        return self.road_boundary_slack_offset + self.road_boundary_slack_count

    @property
    def road_envelope_slack_offset(self) -> int:
        return self.speed_slack_offset + int(self.speed_slack_count)

    @property
    def total_variables(self) -> int:
        return self.road_envelope_slack_offset + int(self.road_envelope_slack_count)

    def state_index(self, k: int, i: int) -> int:
        return self.state_offset + k * self.nx + i

    def control_index(self, k: int, i: int) -> int:
        return self.control_offset + k * self.nu + i

    def road_boundary_left_slack_index(self, k: int) -> int:
        return self.road_boundary_slack_offset + 2 * (k - 1)

    def road_boundary_right_slack_index(self, k: int) -> int:
        return self.road_boundary_slack_offset + 2 * (k - 1) + 1

    def speed_slack_index(self, k: int) -> int:
        return self.speed_slack_offset + (k - 1)

    def road_envelope_slack_index(self, k: int) -> int:
        return self.road_envelope_slack_offset + (k - 1)


class MPC:

    """
    Intent:
        Compute an optimal future trajectory of states [x,y,v,psi] over a finite
        horizon using an LTV-MPC QP solved by OSQP.

    Inputs to `plan_trajectory`:
        current_state:
            sequence[float], shape (4), [x, y, v, psi]
        destination_state:
            sequence[float], shape (2) or (4), [x, y, v, psi]
        object_snapshots:
            sequence of non-ego object snapshots. Each snapshot may include
            tracker predictions under `predicted_trajectory`.
        current_acceleration_mps2:
            float, previous applied acceleration command for jerk penalty/constraint.
        current_steering_rad:
            float, previous applied steering command for smoothness.

    Output:
        list[list[float]], shape (M, 4), future states for k=1..M where
        M<=N if the destination is reached within the horizon.
    """

    def __init__(
        self,
        mpc_cfg: Mapping[str, object],
        road_cfg: Mapping[str, object],
    ) -> None:

        self.horizon_s = float(mpc_cfg.get("horizon_s", 5.0))
        self.dt_s = float(mpc_cfg.get("plan_dt_s", 0.05))
        if self.horizon_s <= 0.0 or self.dt_s <= 0.0:
            raise ValueError("mpc.horizon_s and mpc.plan_dt_s must be > 0.")
        self.horizon_steps = max(1, int(round(self.horizon_s / self.dt_s)))
        self.horizon_s = float(self.horizon_steps * self.dt_s)

        # Off by default: horizon_s above governs everything unless a caller
        # (the bridge, per behavior mode + obstacle state) explicitly drives
        # blend_toward_horizon_s every tick. _build_qp rebuilds the QP fresh
        # from self.horizon_steps on every call (no persisted OSQP problem),
        # so changing it between ticks needs no other special handling.
        self.adaptive_horizon_enabled = bool(
            mpc_cfg.get("adaptive_horizon_enabled", False)
        )
        self.adaptive_horizon_min_s = max(
            self.dt_s, float(mpc_cfg.get("adaptive_horizon_min_s", 1.0))
        )
        self.adaptive_horizon_max_s = max(
            float(self.adaptive_horizon_min_s),
            float(mpc_cfg.get("adaptive_horizon_max_s", 5.0)),
        )
        # blend_toward_horizon_s only actually applies a change once the
        # blended value has drifted at least this many steps from the
        # current horizon_steps. _build_shifted_previous_solution_seed drops
        # the warm start outright on ANY horizon_steps change (a shape
        # mismatch), so continuously drifting it by 1 step almost every tick
        # (e.g. while a lead vehicle's distance shrinks smoothly) forces a
        # cold-start solve nearly every tick, producing small solve-to-solve
        # steering jitter even on an otherwise straight lane-follow. Holding
        # steady between coarser jumps lets the warm start survive across
        # most ticks.
        self.adaptive_horizon_min_step_change = max(
            1, int(mpc_cfg.get("adaptive_horizon_min_step_change", 3))
        )
        # Continuously-tracked blend target, independent of the committed
        # (possibly held-steady) self.horizon_s -- see blend_toward_horizon_s.
        self._adaptive_horizon_continuous_s = float(self.horizon_s)

        self.trajectory_generation_frequency_hz = max(
            1e-3,
            float(mpc_cfg.get("trajectory_generation_frequency_hz", 2.0)),
        )
        self.trajectory_generation_period_s = 1.0 / self.trajectory_generation_frequency_hz

        self.destination_reached_threshold_m = max(
            0.05,
            float(mpc_cfg.get("destination_reached_threshold_m", 0.5)),
        )

        self.wheelbase_m = float(mpc_cfg.get("wheelbase_m", 2.7))
        if self.wheelbase_m <= 0.0:
            raise ValueError("mpc.wheelbase_m must be > 0.")
        # CG-reference model parameters. In this project we assume the CG is
        # centered between axles unless a scenario-specific axle split is added.
        self.l_r_m = max(1e-9, 0.5 * float(self.wheelbase_m))
        self.ego_length_m = max(0.0, float(mpc_cfg.get("ego_length_m", 0.0)))
        self.ego_width_m = max(0.0, float(mpc_cfg.get("ego_width_m", 0.0)))

        constraints_cfg = dict(mpc_cfg.get("constraints", {}))
        lane_count = max(1, int(road_cfg.get("lane_count", 3)))
        self.lane_count = int(lane_count)
        lane_width_m = float(road_cfg.get("lane_width_m", 4.0))
        self.lane_width_m = float(lane_width_m)
        self.constraints = MPCConstraintSpec(
            min_velocity_mps=float(constraints_cfg.get("min_velocity_mps", 0.0)),
            max_velocity_mps=float(constraints_cfg.get("max_velocity_mps", 15.0)),
            min_acceleration_mps2=float(constraints_cfg.get("min_acceleration_mps2", -3.0)),
            max_acceleration_mps2=float(constraints_cfg.get("max_acceleration_mps2", 3.0)),
            max_jerk_mps3=abs(float(constraints_cfg.get("max_jerk_mps3", 10.0))),
            min_steer_rad=float(constraints_cfg.get("min_steer_rad", -0.3)),
            max_steer_rad=float(constraints_cfg.get("max_steer_rad", 0.3)),
            min_steer_rate_rps=min(
                float(constraints_cfg.get("min_steer_rate_rps", -0.02)),
                float(constraints_cfg.get("max_steer_rate_rps", 0.02)),
            ),
            max_steer_rate_rps=max(
                float(constraints_cfg.get("min_steer_rate_rps", -0.02)),
                float(constraints_cfg.get("max_steer_rate_rps", 0.02)),
            ),
            enforce_terminal_velocity_constraint=bool(constraints_cfg.get("enforce_terminal_velocity_constraint", True)),
            terminal_velocity_mps=float(constraints_cfg.get("terminal_velocity_mps", 0.0)),
        )
        final_stop_speed_cap_cfg = dict(mpc_cfg.get("final_stop_speed_cap", {}))
        self.final_stop_speed_cap_enabled = bool(final_stop_speed_cap_cfg.get("enabled", True))
        self.final_stop_speed_cap_activation_threshold_mps = max(
            0.0,
            float(final_stop_speed_cap_cfg.get("destination_speed_activation_threshold_mps", 0.05)),
        )
        self.final_stop_speed_cap_stop_buffer_m = max(
            0.0,
            float(final_stop_speed_cap_cfg.get("stop_buffer_m", 2.0)),
        )

        cost_cfg = dict(mpc_cfg.get("cost", {}))
        attractive_cfg = dict(cost_cfg.get("attractive", {}))
        control_cfg = dict(cost_cfg.get("control", {}))
        self.comfort_cost = MPCComfortCostSpec(
            # Control weight (legacy fallback: cost.w_comf).
            w_comf=max(0.0, float(control_cfg.get("w_control", cost_cfg.get("w_comf", 0.3)))),
            # Attractive term weights (legacy fallback: cost.q_*).
            qx=max(0.0, float(attractive_cfg.get("q_x", cost_cfg.get("q_x", 5.0)))),
            qy=max(0.0, float(attractive_cfg.get("q_y", cost_cfg.get("q_y", 8.0)))),
            qv=max(0.0, float(attractive_cfg.get("q_v", cost_cfg.get("q_v", 2.0)))),
            qpsi=max(0.0, float(attractive_cfg.get("q_psi", cost_cfg.get("q_psi", 4.0)))),
            # Control-rate weights (legacy fallback: cost.q_a, cost.q_delta).
            qa=max(0.0, float(control_cfg.get("q_a", cost_cfg.get("q_a", 2.0)))),
            qdelta=max(0.0, float(control_cfg.get("q_delta", cost_cfg.get("q_delta", 4.0)))),
        )
        self.safety_cost = MPCSafetyCostSpec(
            # Attractive weight (legacy fallback: cost.w_safe).
            w_safe=max(0.0, float(attractive_cfg.get("w_attractive", cost_cfg.get("w_safe", 0.7)))),
        )
        repulsive_cfg = dict(cost_cfg.get("repulsive_potential", {}))
        legacy_static_buffer_m = max(0.0, float(repulsive_cfg.get("static_buffer_m", 0.5)))
        self.repulsive_cost = MPCRepulsivePotentialSpec(
            enabled=bool(repulsive_cfg.get("enabled", True)),
            w_safe_zone=max(0.0, float(repulsive_cfg.get("w_safe_zone", 10.0))),
            w_collision_zone=max(0.0, float(repulsive_cfg.get("w_collision_zone", 100.0))),
            safe_exponential_gain=max(0.0, float(repulsive_cfg.get("safe_exponential_gain", 10.0))),
            safe_distance_shift=float(repulsive_cfg.get("safe_distance_shift", 1.5)),
            collision_exponential_gain=max(0.0, float(repulsive_cfg.get("collision_exponential_gain", 6.0))),
            collision_distance_shift=float(repulsive_cfg.get("collision_distance_shift", 1.5)),
            max_braking_deceleration_mps2=max(
                1e-6,
                float(
                    repulsive_cfg.get(
                        "max_braking_deceleration_mps2",
                        max(1e-6, abs(float(self.constraints.min_acceleration_mps2))),
                    )
                ),
            ),
            comfort_deceleration_mps2=max(
                1e-6,
                float(repulsive_cfg.get("comfort_deceleration_mps2", 2.0)),
            ),
            reaction_time_s=max(
                0.0,
                float(repulsive_cfg.get("reaction_time_s", 1.0)),
            ),
            static_longitudinal_buffer_m=max(
                0.0,
                float(repulsive_cfg.get("static_longitudinal_buffer_m", legacy_static_buffer_m)),
            ),
            static_lateral_buffer_m=max(
                0.0,
                float(repulsive_cfg.get("static_lateral_buffer_m", legacy_static_buffer_m)),
            ),
            shape_exponent=max(
                2.0,
                float(repulsive_cfg.get("shape_exponent", 4.0)),
            ),
            min_lateral_approach_speed_mps=max(
                1e-6,
                float(repulsive_cfg.get("min_lateral_approach_speed_mps", 0.1)),
            ),
            max_longitudinal_zone_length_m=max(
                1e-6,
                float(repulsive_cfg.get("max_longitudinal_zone_length_m", 10.0)),
            ),
            limit_lateral_zone_to_lane_width=bool(
                repulsive_cfg.get("limit_lateral_zone_to_lane_width", True)
            ),
            max_lateral_zone_lane_fraction=max(
                1e-3,
                float(repulsive_cfg.get("max_lateral_zone_lane_fraction", 1.0)),
            ),
            project_hessian_psd=bool(repulsive_cfg.get("project_hessian_psd", repulsive_cfg.get("taylor_project_hessian_psd", True))),
            min_hessian_eig=max(
                0.0,
                float(repulsive_cfg.get("min_hessian_eig", repulsive_cfg.get("taylor_min_hessian_eig", 1e-9))),
            ),
            log_barrier_enabled=bool(repulsive_cfg.get("log_barrier_enabled", False)),
            log_barrier_replace_exponential=bool(
                repulsive_cfg.get("log_barrier_replace_exponential", False)
            ),
            w_log_barrier=max(0.0, float(repulsive_cfg.get("w_log_barrier", 0.0))),
            log_barrier_gain=max(1e-6, float(repulsive_cfg.get("log_barrier_gain", 4.0))),
            cross_track_suppression_enabled=bool(
                repulsive_cfg.get("cross_track_suppression_enabled", False)
            ),
            cross_track_full_suppression_m=max(
                0.0,
                float(repulsive_cfg.get("cross_track_full_suppression_m", 1.0)),
            ),
            cross_track_full_response_m=max(
                float(repulsive_cfg.get("cross_track_full_suppression_m", 1.0)) + 1e-6,
                float(repulsive_cfg.get("cross_track_full_response_m", 2.0)),
            ),
        )

        lane_center_cfg = dict(cost_cfg.get("lane_center_follow", {}))
        self.lane_center_follow_enabled = bool(lane_center_cfg.get("enabled", False))
        self.lane_center_follow_weight = max(0.0, float(lane_center_cfg.get("w0", lane_center_cfg.get("w_lane_center", 0.0))))
        self.lane_center_follow_xy_weight = max(
            0.0,
            float(
                lane_center_cfg.get(
                    "xy_w0",
                    lane_center_cfg.get("w_xy", lane_center_cfg.get("centerline_xy_weight", 0.0)),
                )
            ),
        )
        self.lane_center_follow_qpsi = max(
            0.0,
            float(lane_center_cfg.get("q_psi", lane_center_cfg.get("heading_weight", 0.0))),
        )
        road_boundary_cfg = dict(cost_cfg.get("road_boundary", {}))
        self.road_boundary_enabled = bool(road_boundary_cfg.get("enabled", True))
        self.road_boundary_weight = max(
            0.0,
            float(
                road_boundary_cfg.get(
                    "w_boundary",
                    road_boundary_cfg.get(
                        "w_road",
                        lane_center_cfg.get("w_boundary", lane_center_cfg.get("boundary_weight", 1.0e4)),
                    ),
                )
            ),
        )
        configured_road_boundary_margin_m = max(
            0.0,
            float(road_boundary_cfg.get("margin_m", road_boundary_cfg.get("margin", 0.5))),
        )
        footprint_extra_margin_m = max(
            0.0,
            float(road_boundary_cfg.get("footprint_extra_margin_m", 0.2)),
        )
        footprint_margin_m = 0.5 * float(self.ego_width_m) + float(footprint_extra_margin_m)
        self.road_boundary_margin_m = max(
            float(configured_road_boundary_margin_m),
            float(footprint_margin_m),
        )
        self.road_boundary_max_slack_m = max(
            0.0,
            float(road_boundary_cfg.get("max_slack_m", 0.25)),
        )
        self.lane_keep_boundary_weight = float(self.road_boundary_weight)

        # Road-envelope block-union hard constraint (Yu et al.,
        # "Spatial Envelope MPC," arXiv:2509.18506, Sec. III-B1). Off by
        # default: this only ever activates when the bridge explicitly
        # supplies `road_envelope_blocks` to plan_trajectory (during a
        # locked route-tracking lane change), and even then only when this
        # flag is also on. Replaces the single-reference-line
        # `road_boundary` constraint for that call only -- the union of two
        # *static* blocks (source lane + target lane, fixed once at lock
        # time) stays satisfiable even as the tracked reference switches
        # lanes mid-maneuver, unlike a line tied to whichever reference
        # sample is active that tick.
        road_envelope_cfg = dict(cost_cfg.get("road_envelope", {}))
        self.road_envelope_enabled = bool(road_envelope_cfg.get("enabled", False))
        self.road_envelope_weight = max(
            0.0,
            float(road_envelope_cfg.get("w_envelope", 10000.0)),
        )
        self.road_envelope_max_slack_m = max(
            0.0,
            float(road_envelope_cfg.get("max_slack_m", 0.10)),
        )
        self.road_envelope_shape_exponent = max(
            2.0,
            float(road_envelope_cfg.get("shape_exponent", 4.0)),
        )
        self.road_envelope_rho = min(
            -1.0e-6,
            float(road_envelope_cfg.get("rho", -8.0)),
        )

        # Speed upper-bound soft constraint. Disabled by default: the hard
        # per-stage bound (`add_constraint(..., min_velocity_mps,
        # stage_speed_upper_bound_mps)` in _build_qp) is kept exactly as
        # before unless this is explicitly enabled. When enabled, a per-stage
        # slack absorbs overshoot above the posted cap (mirrors the
        # road-boundary slack pattern above), penalized quadratically and
        # itself hard-capped at speed_soft_max_slack_mps so the softened
        # constraint cannot be violated without bound.
        speed_soft_cfg = dict(cost_cfg.get("speed_soft_constraint", {}))
        self.speed_soft_constraint_enabled = bool(speed_soft_cfg.get("enabled", False))
        self.speed_soft_constraint_weight = max(
            0.0,
            float(speed_soft_cfg.get("weight", 200.0)),
        )
        self.speed_soft_max_slack_mps = max(
            0.0,
            float(speed_soft_cfg.get("max_slack_mps", 3.0)),
        )
        self.lane_keep_safe_region_alpha = min(
            0.999999,
            max(
                1.0e-6,
                float(
                    lane_center_cfg.get(
                        "safe_region_alpha",
                        lane_center_cfg.get("alpha", 0.7),
                    )
                ),
            ),
        )
        self.lane_center_reference_local_window = max(
            0,
            int(lane_center_cfg.get("local_stage_window", 2)),
        )
        # Arc-length ("Frenet-like") lane-reference lookup: off by default,
        # preserving today's array-index/local-window lookup exactly. When
        # enabled, each stage's lane-center sample is selected by how far
        # along the reference path that stage's own rollout position has
        # actually traveled, instead of by its position in the array --
        # the two only coincide on a straight reference built at exactly the
        # nominal step distance, so this specifically targets curved-road
        # reference inconsistency. See _get_lane_center_stage_sample_by_progress.
        self.lane_center_follow_use_progress_lookup = bool(
            lane_center_cfg.get("use_progress_lookup", False)
        )
        # When true, the raw world-(x,y) centerline_xy_weight pull is
        # decomposed into lane-local longitudinal/lateral components instead
        # of independently attracting raw x and y (which couples world-frame
        # x/y errors together and doesn't rotate with lane heading the way
        # the lateral-offset term already does). Off by default, preserving
        # today's per-mode-profile tuning until validated.
        self.lane_center_follow_xy_uses_frenet_decomposition = bool(
            lane_center_cfg.get("xy_term_uses_frenet_decomposition", False)
        )

        self.reference_cfg = dict(mpc_cfg.get("reference_rollout", {}))
        self.reference_heading_gain = float(self.reference_cfg.get("heading_gain", 1.6))
        self.reference_speed_gain = float(self.reference_cfg.get("speed_gain", 1.2))
        self.reference_speed_upper_bound_margin_mps = max(
            0.0,
            float(
                self.reference_cfg.get(
                    "speed_upper_bound_margin_mps",
                    0.5,
                )
            ),
        )
        self.reference_prefer_lane_center_path = bool(self.reference_cfg.get("prefer_lane_center_path", True))
        self.reference_path_los_heading_blend = min(
            1.0,
            max(0.0, float(self.reference_cfg.get("path_los_heading_blend", 0.35))),
        )
        self.reference_use_previous_solution_seed = bool(
            self.reference_cfg.get("use_previous_solution_seed", True)
        )
        self.reference_consecutive_solver_failure_reset_threshold = max(
            0,
            int(
                self.reference_cfg.get(
                    "consecutive_solver_failure_reset_threshold",
                    self.reference_cfg.get("failed_solution_reset_threshold", 4),
                )
            ),
        )
        self.fail_safe_gentle_brake_deceleration_mps2 = max(
            1e-6,
            float(
                self.reference_cfg.get(
                    "fail_safe_gentle_brake_deceleration_mps2", 2.0
                )
            ),
        )
        self.fail_safe_emergency_stop_failure_threshold = max(
            1,
            int(
                self.reference_cfg.get(
                    "fail_safe_emergency_stop_failure_threshold",
                    max(1, int(self.reference_consecutive_solver_failure_reset_threshold)) * 2,
                )
            ),
        )
        self.solver_failure_log_every_n = max(
            1,
            int(self.reference_cfg.get("solver_failure_log_every_n", 50)),
        )
        self.log_solution_memory_resets = bool(
            self.reference_cfg.get("log_solution_memory_resets", False)
        )
        self._solver_failure_log_event_count = 0
        self._solver_failure_emergency_logged = False
        self.reference_previous_solution_search_steps = max(
            0,
            int(self.reference_cfg.get("previous_solution_search_steps", 15)),
        )
        self.reference_previous_solution_max_position_error_m = max(
            0.0,
            float(self.reference_cfg.get("previous_solution_max_position_error_m", 3.0)),
        )
        self.reference_previous_solution_max_heading_error_rad = max(
            0.0,
            float(self.reference_cfg.get("previous_solution_max_heading_error_rad", 0.75)),
        )
        self.reference_previous_solution_max_speed_error_mps = max(
            0.0,
            float(self.reference_cfg.get("previous_solution_max_speed_error_mps", 4.0)),
        )
        self.reference_sequential_iterations = max(
            1,
            int(self.reference_cfg.get("sequential_linearization_iterations", 2)),
        )
        self.reference_obstacle_aware_speed_enabled = bool(
            self.reference_cfg.get("obstacle_aware_speed_enabled", True)
        )
        self.reference_obstacle_check_horizon_s = max(
            float(self.dt_s),
            float(self.reference_cfg.get("obstacle_check_horizon_s", 3.0)),
        )
        self.reference_lead_obstacle_trigger_distance_m = max(
            0.0,
            float(self.reference_cfg.get("lead_obstacle_trigger_distance_m", 18.0)),
        )
        self.reference_lead_obstacle_lateral_margin_m = max(
            0.0,
            float(self.reference_cfg.get("lead_obstacle_lateral_margin_m", 1.2)),
        )
        self.reference_lead_obstacle_stop_buffer_m = max(
            0.0,
            float(self.reference_cfg.get("lead_obstacle_stop_buffer_m", 6.0)),
        )
        self.reference_lead_obstacle_braking_decel_mps2 = max(
            1e-6,
            float(
                self.reference_cfg.get(
                    "lead_obstacle_braking_deceleration_mps2",
                    max(1e-6, abs(float(self.constraints.min_acceleration_mps2))),
                )
            ),
        )
        self.mode_cost_profiles = dict(
            mpc_cfg.get("mode_cost_profiles", mpc_cfg.get("mpc_profiles", {}))
        )
        self.mode_cost_profile_blend_alpha = min(
            1.0,
            max(0.0, float(mpc_cfg.get("mode_cost_profile_blend_alpha", 0.35))),
        )
        self._base_mode_cost_state = self._capture_mode_cost_state()
        self.active_cost_profile_name = "base"

        self.solver_cfg = dict(mpc_cfg.get("solver", {}))
        self.qp_max_iter = int(self.solver_cfg.get("max_iter", 4000))
        self.qp_eps_abs = float(self.solver_cfg.get("eps_abs", 1e-3))
        self.qp_eps_rel = float(self.solver_cfg.get("eps_rel", 1e-3))
        self.qp_polish = bool(self.solver_cfg.get("polish", True))
        if not _OSQP_AVAILABLE:
            raise ImportError("OSQP is required for the MPC QP solver. Install `osqp` in the environment.")

        self.nx = 4
        self.nu = 2
        self._last_status = "not_solved"
        self._last_solve_time_ms = 0.0
        self._last_active_max_velocity_mps = float(self.constraints.max_velocity_mps)
        self._last_cost_terms: Dict[str, float] = {
            "Cost_ref": 0.0,
            "Cost_LaneCenter": 0.0,
            "Cost_CenterlineXY": 0.0,
            "Cost_RoadBoundary": 0.0,
            "Cost_LaneBoundary": 0.0,
            "Cost_Lane": 0.0,
            "Cost_Repulsive_Safe": 0.0,
            "Cost_Repulsive_Collision": 0.0,
            "Cost_Repulsive_LogBarrier": 0.0,
            "Cost_Repulsive": 0.0,
            "Cost_Control": 0.0,
            "Cost_VelocitySlack": 0.0,
        }
        self._last_lane_keeping_profile = LaneKeepingProfile(stage_metrics=tuple(), total_cost=0.0)
        self._last_x_solution: np.ndarray | None = None
        self._last_u_solution: np.ndarray | None = None
        self._previous_x_solution: np.ndarray | None = None
        self._previous_u_solution: np.ndarray | None = None
        self._consecutive_solver_failure_count: int = 0
        self._last_failure_reset_triggered: bool = False
        # Track whether the previous plan_trajectory call was a stop goal.
        # Used to detect the stop→resume transition and prevent the v=0 stop
        # plan from being reused as a linearisation seed, which would cause a
        # degenerate QP and prevent re-acceleration after a stop.
        self._last_was_stop_goal: bool = False

        # --- Internal replan rate-limiting ---
        self._last_replan_sim_time_s: float = -1.0

    def _capture_mode_cost_state(self) -> Dict[str, float]:
        return {
            "w_attractive": float(self.safety_cost.w_safe),
            "q_x": float(self.comfort_cost.qx),
            "q_y": float(self.comfort_cost.qy),
            "q_v": float(self.comfort_cost.qv),
            "q_psi": float(self.comfort_cost.qpsi),
            "w_control": float(self.comfort_cost.w_comf),
            "q_a": float(self.comfort_cost.qa),
            "q_delta": float(self.comfort_cost.qdelta),
            "lane_center_w0": float(self.lane_center_follow_weight),
            "lane_center_xy_w0": float(self.lane_center_follow_xy_weight),
            "lane_center_q_psi": float(self.lane_center_follow_qpsi),
            "road_boundary_w": float(self.road_boundary_weight),
            "road_boundary_margin_m": float(self.road_boundary_margin_m),
            "road_boundary_max_slack_m": float(self.road_boundary_max_slack_m),
            "road_envelope_w": float(self.road_envelope_weight),
            "road_envelope_max_slack_m": float(self.road_envelope_max_slack_m),
            "speed_soft_constraint_w": float(self.speed_soft_constraint_weight),
            "speed_soft_max_slack_mps": float(self.speed_soft_max_slack_mps),
        }

    @staticmethod
    def _profile_value(profile: Mapping[str, object], *keys: str) -> float | None:
        for key in keys:
            if key in profile:
                try:
                    value = float(profile.get(key, 0.0))
                except Exception:
                    return None
                if math.isfinite(value):
                    return float(value)
        return None

    def _target_mode_cost_state(self, profile_name: str) -> Dict[str, float]:
        target = dict(self._base_mode_cost_state)
        raw_profile = self.mode_cost_profiles.get(str(profile_name), {})
        if not isinstance(raw_profile, Mapping):
            return target
        aliases = {
            "w_attractive": ("w_attractive", "w_safe"),
            "q_x": ("q_x", "qx"),
            "q_y": ("q_y", "qy"),
            "q_v": ("q_v", "qv"),
            "q_psi": ("q_psi", "qpsi"),
            "w_control": ("w_control", "w_comf"),
            "q_a": ("q_a", "qa"),
            "q_delta": ("q_delta", "qdelta"),
            "lane_center_w0": ("lane_center_w0", "lane_center_weight", "w_lane_center", "w0"),
            "lane_center_xy_w0": (
                "lane_center_xy_w0",
                "lane_center_xy_weight",
                "centerline_xy_weight",
                "xy_w0",
                "w_xy",
            ),
            "lane_center_q_psi": ("lane_center_q_psi", "lane_center_heading_weight"),
            "road_boundary_w": ("road_boundary_w", "road_boundary_weight", "w_boundary"),
            "road_boundary_margin_m": ("road_boundary_margin_m", "road_boundary_margin"),
            "road_boundary_max_slack_m": ("road_boundary_max_slack_m", "road_boundary_max_slack"),
            "road_envelope_w": ("road_envelope_w", "road_envelope_weight", "w_envelope"),
            "road_envelope_max_slack_m": ("road_envelope_max_slack_m", "road_envelope_max_slack"),
            "speed_soft_constraint_w": ("speed_soft_constraint_w", "speed_soft_constraint_weight"),
            "speed_soft_max_slack_mps": ("speed_soft_max_slack_mps",),
        }
        for canonical_key, key_aliases in aliases.items():
            value = self._profile_value(raw_profile, *key_aliases)
            if value is not None:
                target[canonical_key] = max(0.0, float(value))
        return target

    def apply_mode_cost_profile(self, profile_name: str, blend_alpha: float | None = None) -> str:
        """Apply behavior-mode-specific MPC cost weights.

        Only objective weights are changed here; hard constraints stay under
        the existing speed-cap/constraint layer so mode switching does not
        suddenly alter feasibility.
        """
        normalized_profile = str(profile_name or "base").strip()
        if not normalized_profile:
            normalized_profile = "base"
        if normalized_profile != "base" and normalized_profile not in self.mode_cost_profiles:
            normalized_profile = "lane_follow" if "lane_follow" in self.mode_cost_profiles else "base"

        target = self._target_mode_cost_state(normalized_profile)
        alpha = (
            float(self.mode_cost_profile_blend_alpha)
            if blend_alpha is None
            else float(blend_alpha)
        )
        alpha = min(1.0, max(0.0, float(alpha)))
        current = self._capture_mode_cost_state()
        blended = {
            key: float(current.get(key, 0.0)) * (1.0 - alpha) + float(target[key]) * alpha
            for key in target.keys()
        }
        self.safety_cost.w_safe = float(blended["w_attractive"])
        self.comfort_cost.qx = float(blended["q_x"])
        self.comfort_cost.qy = float(blended["q_y"])
        self.comfort_cost.qv = float(blended["q_v"])
        self.comfort_cost.qpsi = float(blended["q_psi"])
        self.comfort_cost.w_comf = float(blended["w_control"])
        self.comfort_cost.qa = float(blended["q_a"])
        self.comfort_cost.qdelta = float(blended["q_delta"])
        self.lane_center_follow_weight = float(blended["lane_center_w0"])
        self.lane_center_follow_xy_weight = float(blended["lane_center_xy_w0"])
        self.lane_center_follow_qpsi = float(blended["lane_center_q_psi"])
        self.road_boundary_weight = float(blended["road_boundary_w"])
        self.lane_keep_boundary_weight = float(self.road_boundary_weight)
        self.road_boundary_margin_m = float(blended["road_boundary_margin_m"])
        self.road_boundary_max_slack_m = float(blended["road_boundary_max_slack_m"])
        self.road_envelope_weight = float(blended["road_envelope_w"])
        self.road_envelope_max_slack_m = float(blended["road_envelope_max_slack_m"])
        self.speed_soft_constraint_weight = float(blended["speed_soft_constraint_w"])
        self.speed_soft_max_slack_mps = float(blended["speed_soft_max_slack_mps"])
        self.active_cost_profile_name = str(normalized_profile)
        return str(normalized_profile)

    def blend_toward_horizon_s(
        self, target_horizon_s: float, *, blend_alpha: float | None = None
    ) -> float:
        """Smoothly move the prediction horizon toward target_horizon_s.

        Clamped to [adaptive_horizon_min_s, adaptive_horizon_max_s]. No
        persisted OSQP problem depends on horizon_steps between calls (see
        _build_qp), so changing horizon_steps here needs no other special
        handling -- the next plan_trajectory call just builds a
        differently-sized QP.

        However, _build_shifted_previous_solution_seed drops MPC's warm
        start on ANY horizon_steps change (an exact-shape check), so
        committing a new horizon_steps every single call -- e.g. while a
        lead vehicle's distance shrinks smoothly and the target drifts by
        a fraction of a step each tick -- forces a cold-start solve nearly
        every tick, which shows up as small solve-to-solve steering noise
        even during otherwise-straight lane_follow. To avoid that, the
        continuous blend target is tracked every call (so it never lags
        behind target_horizon_s), but self.horizon_steps/self.horizon_s --
        the values actually used to build the QP -- are only updated once
        the continuous target has drifted at least
        adaptive_horizon_min_step_change steps away from the currently
        committed value. Most ticks hold steady and keep their warm start;
        only once the drift accumulates enough does the horizon jump.
        """
        alpha = (
            float(self.mode_cost_profile_blend_alpha)
            if blend_alpha is None
            else float(blend_alpha)
        )
        alpha = min(1.0, max(0.0, alpha))
        target = min(
            float(self.adaptive_horizon_max_s),
            max(float(self.adaptive_horizon_min_s), float(target_horizon_s)),
        )
        self._adaptive_horizon_continuous_s = (
            float(self._adaptive_horizon_continuous_s) * (1.0 - alpha)
            + target * alpha
        )
        candidate_steps = max(
            1, int(round(self._adaptive_horizon_continuous_s / self.dt_s))
        )
        if (
            abs(candidate_steps - self.horizon_steps)
            >= self.adaptive_horizon_min_step_change
        ):
            self.horizon_steps = candidate_steps
            self.horizon_s = float(self.horizon_steps * self.dt_s)
        return self.horizon_s

    def should_replan(self, sim_time_s: float) -> bool:
        """Return True when enough simulation time has elapsed for a new plan.

        This keeps the replan-rate logic inside the MPC module so every
        scenario runner gets the same behaviour without duplicating the
        timing check.
        """
        if self._last_replan_sim_time_s < 0.0:
            return True
        return (float(sim_time_s) - self._last_replan_sim_time_s) >= self.trajectory_generation_period_s - 1e-9

    def mark_replanned(self, sim_time_s: float) -> None:
        """Record that a replan just happened at *sim_time_s*."""
        self._last_replan_sim_time_s = float(sim_time_s)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi

    def _align_angle_near(self, angle_rad: float, around_rad: float) -> float:
        """
        Return the angle equivalent to `angle_rad` that is closest to `around_rad`.

        This keeps quadratic heading tracking terms consistent near the wrap
        boundary at +/-pi.
        """

        return float(around_rad) + float(self._wrap_angle(float(angle_rad) - float(around_rad)))

    def _normalized_lane_reference_sample_dict(
        self,
        sample: Mapping[str, object],
    ) -> Dict[str, float] | None:
        if not {"x_ref_m", "y_ref_m", "heading_rad"}.issubset(sample.keys()):
            return None
        default_lane_width_m = float(getattr(self, "lane_width_m", 4.0))
        lane_width_m = float(sample.get("lane_width_m", default_lane_width_m))
        if not math.isfinite(lane_width_m) or lane_width_m <= 0.0:
            lane_width_m = float(default_lane_width_m)
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

        progress_m = sample.get("progress_m", float("nan"))
        try:
            progress_m = float(progress_m)
        except (TypeError, ValueError):
            progress_m = float("nan")

        return {
            "x_ref_m": float(sample.get("x_ref_m", 0.0)),
            "y_ref_m": float(sample.get("y_ref_m", 0.0)),
            "heading_rad": float(sample.get("heading_rad", 0.0)),
            "lane_id": int(sample.get("lane_id", 0)),
            "lane_width_m": float(lane_width_m),
            "road_center_offset_m": float(road_center_offset_m),
            "road_left_width_m": float(road_left_width_m),
            "road_right_width_m": float(road_right_width_m),
            "progress_m": float(progress_m),
        }

    @staticmethod
    def _blend_heading_angles(path_heading_rad: float, los_heading_rad: float, los_weight: float) -> float:
        los_weight = min(1.0, max(0.0, float(los_weight)))
        path_weight = 1.0 - los_weight
        blended_x = path_weight * math.cos(float(path_heading_rad)) + los_weight * math.cos(float(los_heading_rad))
        blended_y = path_weight * math.sin(float(path_heading_rad)) + los_weight * math.sin(float(los_heading_rad))
        if math.hypot(blended_x, blended_y) <= 1e-12:
            return float(path_heading_rad)
        return float(math.atan2(blended_y, blended_x))

    def _get_lane_center_stage_sample(
        self,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
        stage_index: int,
        query_x_m: float | None = None,
        query_y_m: float | None = None,
    ) -> Dict[str, float] | None:
        """
        Fetch the normalized lane-reference sample for stage k.

        Behavior:
            - Default: returns the stage-indexed lane-center sample.
            - If `query_x_m` and `query_y_m` are provided, returns the closest
              available lane-center waypoint to that query point. This is used
              in scenario4 so each MPC stage aligns with the nearest lane-center
              waypoint to the stage reference trajectory point.
        """

        if lane_center_reference is None or len(lane_center_reference) == 0:
            return None

        valid_samples: List[Dict[str, float]] = []
        for sample in lane_center_reference:
            if not isinstance(sample, Mapping):
                continue
            normalized_sample = self._normalized_lane_reference_sample_dict(sample)
            if normalized_sample is not None:
                valid_samples.append(normalized_sample)

        if len(valid_samples) == 0:
            return None

        idx = max(0, min(int(stage_index), len(valid_samples) - 1))

        # Query-aware mode: keep the search local to the requested stage so the
        # reference cannot jump far ahead/back on tight curves.
        if query_x_m is not None and query_y_m is not None:
            qx = float(query_x_m)
            qy = float(query_y_m)
            window_radius = max(0, int(getattr(self, "lane_center_reference_local_window", 0)))
            start_idx = max(0, int(idx) - int(window_radius))
            stop_idx = min(len(valid_samples), int(idx) + int(window_radius) + 1)
            candidate_samples = valid_samples[start_idx:stop_idx] if stop_idx > start_idx else [valid_samples[idx]]
            best = min(
                candidate_samples,
                key=lambda sample: math.hypot(
                    float(sample.get("x_ref_m", 0.0)) - qx,
                    float(sample.get("y_ref_m", 0.0)) - qy,
                ),
            )
            return dict(best)

        sample = valid_samples[idx]
        return dict(sample)

    def _get_lane_center_stage_ref(
        self,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
        stage_index: int,
        query_x_m: float | None = None,
        query_y_m: float | None = None,
    ) -> Tuple[float, float, float] | None:
        """Backward-compatible tuple view of the lane-reference sample."""

        sample = self._get_lane_center_stage_sample(
            lane_center_reference=lane_center_reference,
            stage_index=int(stage_index),
            query_x_m=query_x_m,
            query_y_m=query_y_m,
        )
        if sample is None:
            return None
        return (
            float(sample.get("x_ref_m", 0.0)),
            float(sample.get("y_ref_m", 0.0)),
            float(sample.get("heading_rad", 0.0)),
        )

    def _get_lane_center_stage_sample_by_progress(
        self,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
        query_progress_m: float,
    ) -> Dict[str, float] | None:
        """Arc-length ("Frenet-like") lane-reference lookup.

        Interpolates between the two reference samples bracketing
        `query_progress_m`, instead of indexing by stage number -- this is
        what makes the lookup follow how far the vehicle has actually
        traveled along a curve rather than which array position it happens
        to occupy (the two only coincide on a straight reference built at
        exactly the nominal step distance).

        Returns None (falls back to index/local-window lookup at the call
        site) if the reference is empty or any sample lacks a finite
        `progress_m` tag -- older/partial reference data degrades gracefully
        instead of crashing.
        """

        if lane_center_reference is None or len(lane_center_reference) == 0:
            return None

        valid_samples: List[Dict[str, float]] = []
        for sample in lane_center_reference:
            if not isinstance(sample, Mapping):
                continue
            normalized_sample = self._normalized_lane_reference_sample_dict(sample)
            if normalized_sample is None:
                continue
            if not math.isfinite(float(normalized_sample.get("progress_m", float("nan")))):
                return None
            valid_samples.append(normalized_sample)

        if len(valid_samples) == 0:
            return None
        if len(valid_samples) == 1:
            return dict(valid_samples[0])

        query_progress_m = float(query_progress_m)
        first_progress_m = float(valid_samples[0]["progress_m"])
        last_progress_m = float(valid_samples[-1]["progress_m"])
        if query_progress_m <= first_progress_m:
            return dict(valid_samples[0])
        if query_progress_m >= last_progress_m:
            return dict(valid_samples[-1])

        for idx in range(len(valid_samples) - 1):
            lower = valid_samples[idx]
            upper = valid_samples[idx + 1]
            lower_progress_m = float(lower["progress_m"])
            upper_progress_m = float(upper["progress_m"])
            if query_progress_m > upper_progress_m:
                continue
            span_m = upper_progress_m - lower_progress_m
            if span_m <= 1e-9:
                return dict(lower)
            fraction = min(1.0, max(0.0, (query_progress_m - lower_progress_m) / span_m))
            heading_rad = self._blend_heading_angles(
                path_heading_rad=float(lower["heading_rad"]),
                los_heading_rad=float(upper["heading_rad"]),
                los_weight=float(fraction),
            )
            nearer = lower if fraction < 0.5 else upper
            return {
                "x_ref_m": float(lower["x_ref_m"]) + fraction * (float(upper["x_ref_m"]) - float(lower["x_ref_m"])),
                "y_ref_m": float(lower["y_ref_m"]) + fraction * (float(upper["y_ref_m"]) - float(lower["y_ref_m"])),
                "heading_rad": float(heading_rad),
                "lane_id": int(nearer["lane_id"]),
                "lane_width_m": float(nearer["lane_width_m"]),
                "road_center_offset_m": float(nearer["road_center_offset_m"]),
                "road_left_width_m": float(nearer["road_left_width_m"]),
                "road_right_width_m": float(nearer["road_right_width_m"]),
                "progress_m": float(query_progress_m),
            }

        return dict(valid_samples[-1])

    def _get_lane_center_stage_ref_by_progress(
        self,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
        query_progress_m: float,
    ) -> Tuple[float, float, float] | None:
        """Tuple view of _get_lane_center_stage_sample_by_progress, mirroring
        _get_lane_center_stage_ref's relationship to
        _get_lane_center_stage_sample."""

        sample = self._get_lane_center_stage_sample_by_progress(
            lane_center_reference=lane_center_reference,
            query_progress_m=float(query_progress_m),
        )
        if sample is None:
            return None
        return (
            float(sample.get("x_ref_m", 0.0)),
            float(sample.get("y_ref_m", 0.0)),
            float(sample.get("heading_rad", 0.0)),
        )
    @staticmethod
    def _lane_center_waypoint_position(waypoint: Mapping[str, object]) -> Tuple[float, float] | None:
        position_raw = waypoint.get("position")
        if not isinstance(position_raw, (list, tuple)) or len(position_raw) < 2:
            return None
        return float(position_raw[0]), float(position_raw[1])

    @staticmethod
    def _lane_center_waypoint_key(x_m: float, y_m: float) -> Tuple[float, float]:
        return (round(float(x_m), 3), round(float(y_m), 3))

    @staticmethod
    def _nearest_progress_along_route(
        route_points: Sequence[Sequence[float]],
        xy: Sequence[float],
    ) -> tuple[float, float]:
        if len(route_points) <= 1:
            return 0.0, 0.0

        total_progress_m = 0.0
        best_progress_m = 0.0
        best_distance_m = float("inf")
        px_m = float(xy[0])
        py_m = float(xy[1])

        for idx in range(len(route_points) - 1):
            x0_m, y0_m = float(route_points[idx][0]), float(route_points[idx][1])
            x1_m, y1_m = float(route_points[idx + 1][0]), float(route_points[idx + 1][1])
            dx_m = x1_m - x0_m
            dy_m = y1_m - y0_m
            seg_len_sq = dx_m * dx_m + dy_m * dy_m
            if seg_len_sq <= 1e-9:
                continue
            proj = ((px_m - x0_m) * dx_m + (py_m - y0_m) * dy_m) / seg_len_sq
            proj = min(1.0, max(0.0, proj))
            cx_m = x0_m + proj * dx_m
            cy_m = y0_m + proj * dy_m
            distance_m = math.hypot(px_m - cx_m, py_m - cy_m)
            if distance_m < best_distance_m:
                best_distance_m = distance_m
                best_progress_m = total_progress_m + proj * math.sqrt(seg_len_sq)
            total_progress_m += math.sqrt(seg_len_sq)
        return best_progress_m, total_progress_m

    @staticmethod
    def _sample_route_at_progress(
        route_points: Sequence[Sequence[float]],
        progress_m: float,
    ) -> Tuple[float, float, float]:
        if len(route_points) == 0:
            return 0.0, 0.0, 0.0
        if len(route_points) == 1:
            return float(route_points[0][0]), float(route_points[0][1]), 0.0

        remaining_m = max(0.0, float(progress_m))
        for idx in range(len(route_points) - 1):
            x0_m, y0_m = float(route_points[idx][0]), float(route_points[idx][1])
            x1_m, y1_m = float(route_points[idx + 1][0]), float(route_points[idx + 1][1])
            segment_length_m = math.hypot(x1_m - x0_m, y1_m - y0_m)
            if segment_length_m <= 1e-9:
                continue
            if remaining_m <= segment_length_m:
                alpha = remaining_m / segment_length_m
                heading_rad = math.atan2(y1_m - y0_m, x1_m - x0_m)
                return (
                    float(x0_m + alpha * (x1_m - x0_m)),
                    float(y0_m + alpha * (y1_m - y0_m)),
                    float(heading_rad),
                )
            remaining_m -= segment_length_m

        last_idx = len(route_points) - 1
        prev_idx = max(0, last_idx - 1)
        heading_rad = math.atan2(
            float(route_points[last_idx][1]) - float(route_points[prev_idx][1]),
            float(route_points[last_idx][0]) - float(route_points[prev_idx][0]),
        )
        return (
            float(route_points[last_idx][0]),
            float(route_points[last_idx][1]),
            float(heading_rad),
        )

    def _build_route_reference(
        self,
        current_state: np.ndarray,
        destination_state: np.ndarray,
        route_reference_points: Sequence[Sequence[float]] | None,
        destination_lane_id: int | None = None,
    ) -> List[Dict[str, object]]:
        if route_reference_points is None or len(route_reference_points) < 2:
            return []

        start_progress_m, route_length_m = self._nearest_progress_along_route(
            route_points=route_reference_points,
            xy=[float(current_state[0]), float(current_state[1])],
        )
        reference_speed_mps = max(
            1.0,
            float(current_state[2]),
            abs(float(destination_state[2])),
        )
        step_distance_m = max(0.5, float(reference_speed_mps) * float(self.dt_s))

        stage_reference: List[Dict[str, float]] = []
        for k in range(self.horizon_steps + 1):
            progress_m = min(float(route_length_m), float(start_progress_m) + float(k) * float(step_distance_m))
            x_ref_m, y_ref_m, heading_rad = self._sample_route_at_progress(
                route_points=route_reference_points,
                progress_m=float(progress_m),
            )
            stage_reference.append(
                {
                    "x_ref_m": float(x_ref_m),
                    "y_ref_m": float(y_ref_m),
                    "heading_rad": float(heading_rad),
                    "lane_id": int(destination_lane_id if destination_lane_id is not None else -1),
                    "lane_width_m": float(self.lane_width_m),
                    "road_center_offset_m": 0.0,
                    "road_left_width_m": 0.5 * float(self.lane_width_m),
                    "road_right_width_m": 0.5 * float(self.lane_width_m),
                    "progress_m": float(progress_m),
                }
            )
        return stage_reference

    def _build_lane_center_reference(
        self,
        current_state: np.ndarray,
        destination_state: np.ndarray,
        lane_center_waypoints: Sequence[Mapping[str, object]] | None,
        destination_lane_id: int | None = None,
    ) -> List[Dict[str, float]]:
        """
        Build per-stage lane-center reference inside MPC.

        The integration layer provides road/lane waypoints. MPC owns the
        lane-keeping cost and the reference chain used by that cost.
        """

        if lane_center_waypoints is None or len(lane_center_waypoints) == 0:
            return []

        x_ego_m = float(current_state[0])
        y_ego_m = float(current_state[1])
        x_target_m = float(destination_state[0])
        y_target_m = float(destination_state[1])

        valid_waypoints: List[Dict[str, object]] = []
        waypoint_by_xy: Dict[Tuple[float, float], Dict[str, object]] = {}
        for waypoint in lane_center_waypoints:
            if not isinstance(waypoint, Mapping):
                continue
            position = self._lane_center_waypoint_position(waypoint)
            if position is None:
                continue
            waypoint_copy = dict(waypoint)
            waypoint_copy["heading_rad"] = float(waypoint_copy.get("heading_rad", 0.0))
            waypoint_key = self._lane_center_waypoint_key(position[0], position[1])
            waypoint_by_xy[waypoint_key] = waypoint_copy
            valid_waypoints.append(waypoint_copy)

        if len(valid_waypoints) == 0:
            return []

        normalized_destination_lane_id = (
            None
            if destination_lane_id is None
            else int(destination_lane_id)
        )
        if normalized_destination_lane_id is not None:
            target_lane_waypoints = [
                waypoint
                for waypoint in valid_waypoints
                if int(waypoint.get("lane_id", 0)) == int(normalized_destination_lane_id)
                and self._lane_center_waypoint_position(waypoint) is not None
            ]
        else:
            target_lane_waypoints = []

        if len(target_lane_waypoints) == 0:
            target_lane_waypoint = min(
                valid_waypoints,
                key=lambda waypoint: math.hypot(
                    float(self._lane_center_waypoint_position(waypoint)[0]) - x_target_m,
                    float(self._lane_center_waypoint_position(waypoint)[1]) - y_target_m,
                ) if self._lane_center_waypoint_position(waypoint) is not None else 1.0e9,
            )
            target_lane_id = int(target_lane_waypoint.get("lane_id", 0))
            target_lane_waypoints = [
                waypoint
                for waypoint in valid_waypoints
                if int(waypoint.get("lane_id", 0)) == target_lane_id
                and self._lane_center_waypoint_position(waypoint) is not None
            ]
            if len(target_lane_waypoints) == 0:
                target_lane_waypoints = valid_waypoints
        else:
            target_lane_id = int(normalized_destination_lane_id)

        destination_anchor_waypoint = min(
            target_lane_waypoints,
            key=lambda waypoint: math.hypot(
                float(self._lane_center_waypoint_position(waypoint)[0]) - x_target_m,
                float(self._lane_center_waypoint_position(waypoint)[1]) - y_target_m,
            ) if self._lane_center_waypoint_position(waypoint) is not None else 1.0e9,
        )
        destination_anchor_position = self._lane_center_waypoint_position(destination_anchor_waypoint)
        destination_anchor_key = (
            None
            if destination_anchor_position is None
            else self._lane_center_waypoint_key(
                float(destination_anchor_position[0]),
                float(destination_anchor_position[1]),
            )
        )

        forward_target_lane_waypoints = []
        for waypoint in target_lane_waypoints:
            position = self._lane_center_waypoint_position(waypoint)
            if position is None:
                continue
            longitudinal_offset_m = (
                math.cos(float(waypoint.get("heading_rad", 0.0))) * (float(position[0]) - x_ego_m)
                + math.sin(float(waypoint.get("heading_rad", 0.0))) * (float(position[1]) - y_ego_m)
            )
            if longitudinal_offset_m >= -1e-6:
                forward_target_lane_waypoints.append(waypoint)
        seed_candidates = (
            forward_target_lane_waypoints
            if len(forward_target_lane_waypoints) > 0
            else target_lane_waypoints
        )
        current_waypoint = min(
            seed_candidates,
            key=lambda waypoint: math.hypot(
                float(self._lane_center_waypoint_position(waypoint)[0]) - x_ego_m,
                float(self._lane_center_waypoint_position(waypoint)[1]) - y_ego_m,
            ) if self._lane_center_waypoint_position(waypoint) is not None else 1.0e9,
        )

        stage_reference: List[Dict[str, float]] = []
        visited_keys: set[Tuple[float, float]] = set()
        cumulative_progress_m = 0.0
        for _k in range(self.horizon_steps + 1):
            current_position = self._lane_center_waypoint_position(current_waypoint)
            if current_position is None:
                break
            lane_width_m = float(current_waypoint.get("lane_width_m", self.lane_width_m))
            if not math.isfinite(lane_width_m) or lane_width_m <= 0.0:
                lane_width_m = float(self.lane_width_m)
            stage_reference.append(
                {
                    "x_ref_m": float(current_position[0]),
                    "y_ref_m": float(current_position[1]),
                    "heading_rad": float(current_waypoint.get("heading_rad", 0.0)),
                    "lane_id": int(target_lane_id),
                    "lane_width_m": float(lane_width_m),
                    "road_center_offset_m": float(current_waypoint.get("road_center_offset_m", 0.0)),
                    "road_left_width_m": float(current_waypoint.get("road_left_width_m", 0.5 * lane_width_m)),
                    "road_right_width_m": float(current_waypoint.get("road_right_width_m", 0.5 * lane_width_m)),
                    "progress_m": float(cumulative_progress_m),
                }
            )

            current_key = self._lane_center_waypoint_key(
                float(current_position[0]),
                float(current_position[1]),
            )
            if destination_anchor_key is not None and current_key == destination_anchor_key:
                break

            next_position_raw = current_waypoint.get("next", None)
            if not isinstance(next_position_raw, (list, tuple)) or len(next_position_raw) < 2:
                break
            next_key = self._lane_center_waypoint_key(float(next_position_raw[0]), float(next_position_raw[1]))
            if next_key in visited_keys:
                break
            visited_keys.add(next_key)
            next_waypoint = waypoint_by_xy.get(next_key)
            if next_waypoint is None:
                break
            cumulative_progress_m += math.hypot(
                float(next_position_raw[0]) - float(current_position[0]),
                float(next_position_raw[1]) - float(current_position[1]),
            )
            current_waypoint = next_waypoint

        if len(stage_reference) == 0:
            return []
        while len(stage_reference) < self.horizon_steps + 1:
            held_sample = dict(stage_reference[-1])
            # Keep tagging progress_m even for the held/duplicated tail
            # samples (destination reached before the horizon fills), so
            # progress-based lookups downstream still see a monotonically
            # valid (if flat) progress value instead of an implicit repeat.
            stage_reference.append(held_sample)
        return stage_reference

    def _normalize_lane_center_reference_samples(
        self,
        lane_center_reference_samples: Sequence[Mapping[str, object]] | None,
    ) -> List[Dict[str, float]]:
        if lane_center_reference_samples is None:
            return []

        normalized: List[Dict[str, float]] = []
        for sample in lane_center_reference_samples:
            if not isinstance(sample, Mapping):
                continue
            normalized_sample = self._normalized_lane_reference_sample_dict(sample)
            if normalized_sample is not None:
                normalized.append(normalized_sample)

        if len(normalized) == 0:
            return []
        while len(normalized) < self.horizon_steps + 1:
            normalized.append(dict(normalized[-1]))
        return normalized[: self.horizon_steps + 1]

    def _get_object_state_at_stage(
        self,
        object_snapshot: Mapping[str, object],
        stage_index: int,
        dt_s: float,
    ) -> List[float]:
        """
        Return obstacle state [x,y,v,psi] at prediction stage using tracker
        prediction if available, otherwise constant-velocity propagation.
        """

        predicted = object_snapshot.get("predicted_trajectory", object_snapshot.get("future_trajectory", []))
        if isinstance(predicted, Sequence) and 0 <= int(stage_index) < len(predicted):
            state = predicted[int(stage_index)]
            if isinstance(state, Sequence) and len(state) >= 4:
                return [float(state[0]), float(state[1]), float(state[2]), float(state[3])]

        x = float(object_snapshot.get("x", 0.0))
        y = float(object_snapshot.get("y", 0.0))
        v = float(object_snapshot.get("v", 0.0))
        psi = float(object_snapshot.get("psi", 0.0))
        t = float(max(0, int(stage_index) + 1)) * float(dt_s)
        return [
            float(x + v * math.cos(psi) * t),
            float(y + v * math.sin(psi) * t),
            float(v),
            float(psi),
        ]

    @staticmethod
    def _translate_object_snapshots(
        object_snapshots: Sequence[Mapping[str, object]],
        *,
        origin_x_m: float,
        origin_y_m: float,
    ) -> List[Dict[str, object]]:
        """Translate obstacle states into the ego-origin QP frame."""

        translated: List[Dict[str, object]] = []
        for raw_snapshot in list(object_snapshots or []):
            if not isinstance(raw_snapshot, Mapping):
                continue
            snapshot: Dict[str, object] = dict(raw_snapshot)
            try:
                snapshot["x"] = float(snapshot.get("x", 0.0)) - float(origin_x_m)
                snapshot["y"] = float(snapshot.get("y", 0.0)) - float(origin_y_m)
            except (TypeError, ValueError):
                continue
            for trajectory_key in ("predicted_trajectory", "future_trajectory"):
                raw_trajectory = snapshot.get(trajectory_key)
                if not isinstance(raw_trajectory, Sequence):
                    continue
                local_trajectory: List[object] = []
                for raw_state in raw_trajectory:
                    if isinstance(raw_state, Mapping):
                        state = dict(raw_state)
                        if "x" in state and "y" in state:
                            state["x"] = float(state["x"]) - float(origin_x_m)
                            state["y"] = float(state["y"]) - float(origin_y_m)
                        local_trajectory.append(state)
                    elif isinstance(raw_state, Sequence) and len(raw_state) >= 2:
                        state = list(raw_state)
                        state[0] = float(state[0]) - float(origin_x_m)
                        state[1] = float(state[1]) - float(origin_y_m)
                        local_trajectory.append(state)
                    else:
                        local_trajectory.append(raw_state)
                snapshot[trajectory_key] = local_trajectory
            translated.append(snapshot)
        return translated

    @staticmethod
    def _translate_lane_reference(
        lane_center_reference: Sequence[Mapping[str, object]],
        *,
        origin_x_m: float,
        origin_y_m: float,
    ) -> List[Dict[str, float]]:
        """Translate lane samples while preserving all geometric metadata."""

        translated: List[Dict[str, object]] = []
        for raw_sample in list(lane_center_reference or []):
            sample = dict(raw_sample)
            sample["x_ref_m"] = float(sample.get("x_ref_m", 0.0)) - float(
                origin_x_m
            )
            sample["y_ref_m"] = float(sample.get("y_ref_m", 0.0)) - float(
                origin_y_m
            )
            if "x" in sample:
                sample["x"] = float(sample["x"]) - float(origin_x_m)
            if "y" in sample:
                sample["y"] = float(sample["y"]) - float(origin_y_m)
            translated.append(sample)
        return translated

    def _build_shifted_previous_solution_seed(self, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | None:
        """
        Reuse the previous solved MPC trajectory as the next linearization seed.

        The current ego state is matched to a nearby stage of the previous
        solution, then the remainder of that solution is shifted forward.

        Only the state/control *dimensionality* (nx/nu) has to match --
        horizon *length* does not. blend_toward_horizon_s changes
        self.horizon_steps between calls (e.g. every lane change ramps the
        horizon from the lane_follow profile's ~3s to the lane-change
        profile's ~4.5s over several adaptive_horizon_min_step_change-sized
        jumps); a stale exact-length check here used to force a cold-start
        rebuild on every one of those jumps, discarding the previous
        solution's speed/steering profile right as the maneuver's reference
        geometry is at its most demanding. The index-clamping below
        (``min(best_idx + k, prev_x.shape[0] - 1)``) already tolerates a
        shorter/longer previous array by repeating its last stage, so once
        x_seed/u_seed below are sized to the *current* horizon_steps
        instead of copied from prev_x/prev_u's old shape, reuse across a
        horizon-length change falls out for free.
        """

        if not bool(self.reference_use_previous_solution_seed):
            return None
        if self._previous_x_solution is None or self._previous_u_solution is None:
            return None

        prev_x = self._previous_x_solution
        prev_u = self._previous_u_solution
        if prev_x.ndim != 2 or prev_x.shape[0] < 1 or prev_x.shape[1] != self.nx:
            return None
        if prev_u.ndim != 2 or prev_u.shape[1] != self.nu:
            return None

        search_limit = min(
            int(self.reference_previous_solution_search_steps),
            prev_x.shape[0] - 1,
        )
        best_idx: int | None = None
        best_score = float("inf")

        for idx in range(search_limit + 1):
            position_error_m = math.hypot(
                float(prev_x[idx, 0]) - float(x0[0]),
                float(prev_x[idx, 1]) - float(x0[1]),
            )
            heading_error_rad = abs(self._wrap_angle(float(prev_x[idx, 3]) - float(x0[3])))
            speed_error_mps = abs(float(prev_x[idx, 2]) - float(x0[2]))
            if position_error_m > float(self.reference_previous_solution_max_position_error_m):
                continue
            if heading_error_rad > float(self.reference_previous_solution_max_heading_error_rad):
                continue
            if speed_error_mps > float(self.reference_previous_solution_max_speed_error_mps):
                continue

            score = position_error_m + 0.5 * heading_error_rad + 0.25 * speed_error_mps
            if score < best_score:
                best_score = float(score)
                best_idx = int(idx)

        if best_idx is None:
            return None

        x_seed = np.zeros((self.horizon_steps + 1, self.nx), dtype=float)
        u_seed = np.zeros((self.horizon_steps, self.nu), dtype=float)
        x_seed[0] = np.asarray(x0, dtype=float)

        for k in range(1, self.horizon_steps + 1):
            src_idx = min(best_idx + k, prev_x.shape[0] - 1)
            x_seed[k] = np.asarray(prev_x[src_idx], dtype=float)
            x_seed[k, 3] = self._wrap_angle(float(x_seed[k, 3]))

        for k in range(self.horizon_steps):
            src_idx = min(best_idx + k, prev_u.shape[0] - 1)
            u_seed[k] = np.asarray(prev_u[src_idx], dtype=float)

        return x_seed, u_seed

    def _compute_reference_rollout_speed_limit(
        self,
        stage_x_m: float,
        stage_y_m: float,
        stage_heading_rad: float,
        stage_index: int,
        base_speed_mps: float,
        object_snapshots: Sequence[Mapping[str, object]],
    ) -> float:
        """
        Obstacle-aware speed heuristic for rollout generation.

        The rollout stays generic: if a lead obstacle is predicted ahead in the
        same corridor, cap the rollout speed to a simple stopping-speed bound.
        """

        base_speed_mps = max(0.0, float(base_speed_mps))
        if not bool(self.reference_obstacle_aware_speed_enabled):
            return float(base_speed_mps)
        if base_speed_mps <= 0.0:
            return 0.0
        if len(object_snapshots) == 0:
            return float(base_speed_mps)
        if float(stage_index) * float(self.dt_s) > float(self.reference_obstacle_check_horizon_s):
            return float(base_speed_mps)

        cos_heading = math.cos(float(stage_heading_rad))
        sin_heading = math.sin(float(stage_heading_rad))
        best_gap_m = float("inf")

        for object_snapshot in object_snapshots:
            obj_state = self._get_object_state_at_stage(
                object_snapshot=object_snapshot,
                stage_index=int(stage_index),
                dt_s=float(self.dt_s),
            )
            obj_x_m = float(obj_state[0])
            obj_y_m = float(obj_state[1])
            dx_m = obj_x_m - float(stage_x_m)
            dy_m = obj_y_m - float(stage_y_m)
            along_track_m = dx_m * cos_heading + dy_m * sin_heading
            cross_track_m = -dx_m * sin_heading + dy_m * cos_heading

            if along_track_m < 0.0:
                continue
            if along_track_m > float(self.reference_lead_obstacle_trigger_distance_m):
                continue

            obj_half_width_m = 0.5 * float(object_snapshot.get("width_m", 2.0))
            lateral_limit_m = obj_half_width_m + float(self.reference_lead_obstacle_lateral_margin_m)
            if abs(cross_track_m) > lateral_limit_m:
                continue

            best_gap_m = min(float(best_gap_m), float(along_track_m))

        if not math.isfinite(best_gap_m):
            return float(base_speed_mps)

        remaining_gap_m = max(0.0, float(best_gap_m) - float(self.reference_lead_obstacle_stop_buffer_m))
        stop_speed_limit_mps = math.sqrt(
            max(
                0.0,
                2.0 * float(self.reference_lead_obstacle_braking_decel_mps2) * remaining_gap_m,
            )
        )
        return float(min(base_speed_mps, stop_speed_limit_mps))

    @staticmethod
    def _cross_track_lateral_scale(
        *,
        cross_track_abs_m: float,
        full_suppression_m: float,
        full_response_m: float,
    ) -> float:
        """Smooth ramp: 0 at/below full_suppression_m (obstacle basically
        straight ahead -- suppress the lateral pull entirely), 1 at/above
        full_response_m (obstacle clearly off to the side -- leave the
        lateral pull untouched), linear in between."""

        offset_m = abs(float(cross_track_abs_m))
        low_m = max(0.0, float(full_suppression_m))
        high_m = max(low_m + 1.0e-6, float(full_response_m))
        if offset_m <= low_m:
            return 0.0
        if offset_m >= high_m:
            return 1.0
        return float((offset_m - low_m) / (high_m - low_m))

    def _superellipsoid_obstacle_cost_components(
        self,
        ego_state: Sequence[float],
        obstacle_state: Sequence[float],
        obstacle_length_m: float,
        obstacle_width_m: float,
    ) -> Tuple[float, float]:
        """
        Super-ellipsoid obstacle cost components.

        Cost:
            J_obs = w_c * exp(-k_c * (r_c - s_c))

        where:
            r_c = normalized distance to the collision zone
        """

        geometry = self._superellipsoid_zone_geometry(
            ego_state=ego_state,
            obstacle_state=obstacle_state,
            obstacle_length_m=obstacle_length_m,
            obstacle_width_m=obstacle_width_m,
        )
        rc = float(geometry["rc"])
        cost_collision = float(self.repulsive_cost.w_collision_zone) * math.exp(
            -float(self.repulsive_cost.collision_exponential_gain)
            * (float(rc) - float(self.repulsive_cost.collision_distance_shift))
        )
        return 0.0, float(cost_collision)

    def _superellipsoid_zone_geometry(
        self,
        ego_state: Sequence[float],
        obstacle_state: Sequence[float],
        obstacle_length_m: float,
        obstacle_width_m: float,
    ) -> Dict[str, float]:
        """
        Return the active collision-zone geometry used by the live cost.

        The returned values are aligned to the obstacle frame.
        """

        ego_x_m = float(ego_state[0]) if len(ego_state) >= 1 else 0.0
        ego_y_m = float(ego_state[1]) if len(ego_state) >= 2 else 0.0

        obs_x_m = float(obstacle_state[0]) if len(obstacle_state) >= 1 else 0.0
        obs_y_m = float(obstacle_state[1]) if len(obstacle_state) >= 2 else 0.0
        obs_psi_rad = float(obstacle_state[3]) if len(obstacle_state) >= 4 else 0.0

        cos_obs = math.cos(float(obs_psi_rad))
        sin_obs = math.sin(float(obs_psi_rad))
        dx_m = float(ego_x_m) - float(obs_x_m)
        dy_m = float(ego_y_m) - float(obs_y_m)
        x_local_m = dx_m * cos_obs + dy_m * sin_obs
        y_local_m = -dx_m * sin_obs + dy_m * cos_obs

        obstacle_length_m = max(1e-6, float(obstacle_length_m))
        obstacle_width_m = max(1e-6, float(obstacle_width_m))
        x0_m = 0.5 * (
            float(obstacle_length_m)
            + float(self.repulsive_cost.static_longitudinal_buffer_m)
        )
        y0_m = 0.5 * (
            float(obstacle_width_m)
            + float(self.repulsive_cost.static_lateral_buffer_m)
        )
        xc_m = max(1e-6, float(x0_m))
        yc_m = max(1e-6, float(y0_m))

        n = max(2.0, float(self.repulsive_cost.shape_exponent))
        rc = (abs(float(x_local_m) / max(1e-6, float(xc_m))) ** n + abs(float(y_local_m) / max(1e-6, float(yc_m))) ** n) ** (1.0 / n)

        return {
            "x_local_m": float(x_local_m),
            "y_local_m": float(y_local_m),
            "x0_m": float(x0_m),
            "y0_m": float(y0_m),
            "xc_m": float(xc_m),
            "yc_m": float(yc_m),
            "xs_m": float(xc_m),
            "ys_m": float(yc_m),
            "shape_exponent": float(n),
            "obstacle_x_m": float(obs_x_m),
            "obstacle_y_m": float(obs_y_m),
            "obstacle_psi_rad": float(obs_psi_rad),
            "rc": float(rc),
            "rs": float(rc),
        }

    def _log_barrier_obstacle_cost_component(self, rc: float) -> float:
        """Softplus envelope-containment cost (per Yu et al., "Spatial
        Envelope MPC: High Performance Driving without a Reference",
        arXiv:2509.18506), adapted to this module's existing rc/s_c
        collision-zone geometry:

            m = rc - s_c                          (margin beyond the zone surface)
            J = w_log * log(1 + exp(-theta * m))   (smooth hinge, ~0 for m >> 0)

        Unlike a `-log(margin)` interior-point barrier, this is defined and
        finite for every real m (no domain restriction, no log(0)/NaN risk),
        which matters here because _superellipsoid_cost_taylor_terms probes
        this function at finite perturbed states in both directions to form
        a central-difference gradient/Hessian.
        """

        margin = float(rc) - float(self.repulsive_cost.collision_distance_shift)
        theta = float(self.repulsive_cost.log_barrier_gain)
        # np.logaddexp(0, x) == log(1 + exp(x)), computed in a numerically
        # stable way (no overflow for large |x|).
        softplus = float(np.logaddexp(0.0, -theta * margin))
        return float(self.repulsive_cost.w_log_barrier) * softplus

    def _superellipsoid_obstacle_cost(
        self,
        ego_state: Sequence[float],
        obstacle_state: Sequence[float],
        obstacle_length_m: float,
        obstacle_width_m: float,
    ) -> float:
        geometry = self._superellipsoid_zone_geometry(
            ego_state=ego_state,
            obstacle_state=obstacle_state,
            obstacle_length_m=obstacle_length_m,
            obstacle_width_m=obstacle_width_m,
        )
        cost_safe, cost_collision = self._superellipsoid_obstacle_cost_components(
            ego_state=ego_state,
            obstacle_state=obstacle_state,
            obstacle_length_m=obstacle_length_m,
            obstacle_width_m=obstacle_width_m,
        )
        if bool(self.repulsive_cost.log_barrier_replace_exponential):
            cost_collision = 0.0
        cost_log_barrier = (
            self._log_barrier_obstacle_cost_component(float(geometry["rc"]))
            if bool(self.repulsive_cost.log_barrier_enabled)
            else 0.0
        )
        return float(cost_safe + cost_collision + cost_log_barrier)

    def _superellipsoid_cost_taylor_terms(
        self,
        ego_state_ref: Sequence[float],
        obstacle_state: Sequence[float],
        obstacle_length_m: float,
        obstacle_width_m: float,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Numerical Taylor ingredients of the super-ellipsoid obstacle cost with
        respect to ego state [x, y, v, psi] at one stage reference point.
        """

        state_ref = np.array(
            [
                float(ego_state_ref[0]) if len(ego_state_ref) >= 1 else 0.0,
                float(ego_state_ref[1]) if len(ego_state_ref) >= 2 else 0.0,
                float(ego_state_ref[2]) if len(ego_state_ref) >= 3 else 0.0,
                self._wrap_angle(float(ego_state_ref[3]) if len(ego_state_ref) >= 4 else 0.0),
            ],
            dtype=float,
        )
        step_sizes = np.array([0.05, 0.05, 0.05, 0.01], dtype=float)

        def evaluate(query_state: np.ndarray) -> float:
            state_eval = np.asarray(query_state, dtype=float).copy()
            state_eval[3] = self._wrap_angle(float(state_eval[3]))
            return self._superellipsoid_obstacle_cost(
                ego_state=state_eval,
                obstacle_state=obstacle_state,
                obstacle_length_m=float(obstacle_length_m),
                obstacle_width_m=float(obstacle_width_m),
            )

        p0 = float(evaluate(state_ref))
        gradient = np.zeros(4, dtype=float)
        hessian = np.zeros((4, 4), dtype=float)

        for idx in range(4):
            delta = np.zeros(4, dtype=float)
            delta[idx] = float(step_sizes[idx])
            f_plus = float(evaluate(state_ref + delta))
            f_minus = float(evaluate(state_ref - delta))
            gradient[idx] = (f_plus - f_minus) / (2.0 * float(step_sizes[idx]))
            hessian[idx, idx] = (f_plus - 2.0 * p0 + f_minus) / (float(step_sizes[idx]) ** 2)

        for row in range(4):
            for col in range(row + 1, 4):
                delta_row = np.zeros(4, dtype=float)
                delta_col = np.zeros(4, dtype=float)
                delta_row[row] = float(step_sizes[row])
                delta_col[col] = float(step_sizes[col])
                f_pp = float(evaluate(state_ref + delta_row + delta_col))
                f_pm = float(evaluate(state_ref + delta_row - delta_col))
                f_mp = float(evaluate(state_ref - delta_row + delta_col))
                f_mm = float(evaluate(state_ref - delta_row - delta_col))
                mixed = (f_pp - f_pm - f_mp + f_mm) / (4.0 * float(step_sizes[row]) * float(step_sizes[col]))
                hessian[row, col] = mixed
                hessian[col, row] = mixed

        hessian = 0.5 * (hessian + hessian.T)
        return float(p0), np.asarray(gradient, dtype=float), np.asarray(hessian, dtype=float)

    def _project_symmetric_hessian_to_psd(self, hessian: np.ndarray) -> np.ndarray:
        """
        Project a symmetric Hessian to PSD by clamping eigenvalues.

        This keeps the local quadratic obstacle approximation convex for OSQP.
        """

        H = np.asarray(hessian, dtype=float)
        H = 0.5 * (H + H.T)
        eigvals, eigvecs = np.linalg.eigh(H)
        eig_floor = float(self.repulsive_cost.min_hessian_eig)
        eigvals_clamped = np.maximum(eigvals, eig_floor)
        return np.asarray(eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T, dtype=float)

    def get_runtime_status(self) -> Dict[str, object]:
        return {
            "solver_status": str(self._last_status),
            "solve_time_ms": float(self._last_solve_time_ms),
            "horizon_steps": int(self.horizon_steps),
            "plan_dt_s": float(self.dt_s),
            "horizon_s": float(self.horizon_s),
            "trajectory_generation_frequency_hz": float(self.trajectory_generation_frequency_hz),
            "active_max_velocity_mps": float(self._last_active_max_velocity_mps),
            "compute_backend": "cpu_osqp_qp",
            "consecutive_solver_failure_count": int(self._consecutive_solver_failure_count),
            "consecutive_solver_failure_reset_threshold": int(
                self.reference_consecutive_solver_failure_reset_threshold
            ),
            "failure_reset_triggered": bool(self._last_failure_reset_triggered),
        }

    def get_last_cost_terms(self) -> Dict[str, float]:
        """Return the most recently evaluated MPC cost terms."""

        return dict(self._last_cost_terms)

    def clear_previous_solution_seed(self) -> None:
        """Drop any stored warm-start solution used for rollout seeding."""

        self._previous_x_solution = None
        self._previous_u_solution = None

    def _clear_all_solution_memory(self) -> None:
        self._last_x_solution = None
        self._last_u_solution = None
        self._previous_x_solution = None
        self._previous_u_solution = None

    def _record_solver_failure_state(self, solved: bool) -> bool:
        if bool(solved):
            self._consecutive_solver_failure_count = 0
            self._last_failure_reset_triggered = False
            return False

        self._consecutive_solver_failure_count += 1
        reset_threshold = int(self.reference_consecutive_solver_failure_reset_threshold)
        should_reset = bool(
            reset_threshold > 0
            and int(self._consecutive_solver_failure_count) >= int(reset_threshold)
        )
        self._last_failure_reset_triggered = bool(should_reset)
        return bool(should_reset)

    def _record_clean_restart_result(self, solved: bool) -> None:
        self._consecutive_solver_failure_count = 0 if bool(solved) else 1
        self._last_was_stop_goal = False

    def get_last_lane_keeping_diagnostics(self) -> Dict[str, object]:
        """Return d_perp, U_lane, and J_lane for the last planned horizon."""

        return self._last_lane_keeping_profile.as_dict()

    def get_current_lateral_offset_m(self) -> float | None:
        """Perpendicular distance from lane center at the ego's current position (stage-0 of last plan)."""
        profile = self._last_lane_keeping_profile
        if profile is None or len(profile.stage_metrics) == 0:
            return None
        return float(profile.stage_metrics[0].d_perp_m)

    def get_current_heading_error_rad(self) -> float | None:
        """Heading error (ego yaw minus lane reference heading) at the ego's current position."""
        profile = self._last_lane_keeping_profile
        x = self._last_x_solution
        if profile is None or len(profile.stage_metrics) == 0 or x is None or x.shape[0] == 0:
            return None
        return float(self._wrap_angle(float(x[0, 3]) - float(profile.stage_metrics[0].lane_heading_rad)))

    def get_last_control_sequence(self, max_steps: int | None = None) -> List[Dict[str, float]]:
        """
        Return the most recent MPC-planned control sequence.

        This exposes the optimizer/fallback plan controls used for the latest
        candidate trajectory generation. It does not return the applied ego
        controls from the downstream PID tracker.
        """

        if self._last_u_solution is None:
            return []

        controls = np.asarray(self._last_u_solution, dtype=float)
        step_limit = int(controls.shape[0])
        if max_steps is not None:
            step_limit = max(0, min(step_limit, int(max_steps)))

        output: List[Dict[str, float]] = []
        for step_idx in range(step_limit):
            output.append(
                {
                    "step_index": int(step_idx),
                    "time_from_plan_start_s": float(step_idx) * float(self.dt_s),
                    "acceleration_mps2": float(controls[step_idx, 0]),
                    "steering_angle_rad": float(controls[step_idx, 1]),
                }
            )
        return output

    def _normalize_destination_state(self, destination_state: Sequence[float]) -> np.ndarray:
        """
        Intent:
            Normalize destination input to shape (4,) = [x,y,v,psi].

        PDF rule implemented:
            If destination provides only [x,y], default to v_ref = 0 and
            psi_ref = 0.
        """

        if len(destination_state) >= 4:
            x_ref = float(destination_state[0])
            y_ref = float(destination_state[1])
            v_ref = float(destination_state[2])
            psi_ref = float(destination_state[3])
            return np.array([x_ref, y_ref, v_ref, self._wrap_angle(psi_ref)], dtype=float)

        if len(destination_state) >= 2:
            x_ref = float(destination_state[0])
            y_ref = float(destination_state[1])
            v_ref = 0.0
            psi_ref = 0.0
            return np.array([x_ref, y_ref, v_ref, psi_ref], dtype=float)

        raise ValueError("destination_state must have at least [x, y].")

    def _compute_active_speed_upper_bound_mps(
        self,
        current_state: Sequence[float],
        destination_state: Sequence[float],
        force_stop_goal: bool = False,
    ) -> float:
        """
        Compute the active speed upper bound for this MPC replan.

        When the active destination is a stop goal (destination speed near zero),
        use a braking-distance cap:
            v_cap(d) = min(v_max, sqrt(2 * a_brake * max(d - stop_buffer, 0)))

        where a_brake is taken from abs(min_acceleration_mps2).
        """

        base_max_velocity_mps = float(self.constraints.max_velocity_mps)
        destination_speed_mps = (
            abs(float(destination_state[2]))
            if len(destination_state) >= 3
            else 0.0
        )
        if not bool(force_stop_goal):
            if len(destination_state) < 3:
                return float(base_max_velocity_mps)
            if destination_speed_mps > float(
                self.final_stop_speed_cap_activation_threshold_mps
            ):
                return float(min(
                    base_max_velocity_mps,
                    max(
                        float(self.constraints.min_velocity_mps),
                        float(destination_speed_mps)
                        + float(self.reference_speed_upper_bound_margin_mps),
                    ),
                ))
        if not bool(self.final_stop_speed_cap_enabled):
            return float(base_max_velocity_mps)
        if len(destination_state) < 2:
            return float(base_max_velocity_mps)

        current_x_m = float(current_state[0]) if len(current_state) >= 1 else 0.0
        current_y_m = float(current_state[1]) if len(current_state) >= 2 else 0.0
        destination_x_m = float(destination_state[0]) if len(destination_state) >= 1 else current_x_m
        destination_y_m = float(destination_state[1]) if len(destination_state) >= 2 else current_y_m
        distance_to_destination_m = math.hypot(destination_x_m - current_x_m, destination_y_m - current_y_m)
        remaining_stop_distance_m = max(
            0.0,
            float(distance_to_destination_m) - float(self.final_stop_speed_cap_stop_buffer_m),
        )
        braking_deceleration_mps2 = max(1e-6, abs(float(self.constraints.min_acceleration_mps2)))
        speed_cap_mps = math.sqrt(2.0 * braking_deceleration_mps2 * remaining_stop_distance_m)
        return float(min(base_max_velocity_mps, speed_cap_mps))

    def _minimum_reachable_speed_profile_mps(
        self,
        current_speed_mps: float,
        current_acceleration_mps2: float,
        braking_deceleration_mps2: float | None = None,
    ) -> List[float]:
        """
        Compute the minimum reachable speed profile under bounded jerk when
        applying a sustained braking deceleration.

        Defaults to the strongest allowed braking (``min_acceleration_mps2``)
        when ``braking_deceleration_mps2`` is not given; callers that need a
        milder, comfort-braking profile (e.g. the MPC fail-safe fallback) can
        pass a smaller magnitude instead.
        """

        profile = [max(float(self.constraints.min_velocity_mps), float(current_speed_mps))]
        min_velocity_mps = float(self.constraints.min_velocity_mps)
        target_a_mps2 = (
            float(self.constraints.min_acceleration_mps2)
            if braking_deceleration_mps2 is None
            else -abs(float(braking_deceleration_mps2))
        )
        jerk_delta_limit = float(self.constraints.max_jerk_mps3) * float(self.dt_s)

        v_k_mps = float(profile[0])
        a_prev_mps2 = float(current_acceleration_mps2)
        for _ in range(self.horizon_steps):
            a_k_mps2 = max(target_a_mps2, a_prev_mps2 - jerk_delta_limit)
            v_k_mps = max(min_velocity_mps, float(v_k_mps) + float(self.dt_s) * float(a_k_mps2))
            profile.append(float(v_k_mps))
            a_prev_mps2 = float(a_k_mps2)
        return profile

    def _fail_safe_fallback_trajectory(
        self,
        *,
        x0: np.ndarray,
        rollout_x: np.ndarray,
        rollout_u: np.ndarray,
        current_acceleration_mps2: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Deterministic fallback trajectory for a cycle where every QP solve
        attempt failed.

        The plain rollout returned before this method existed is only an
        open-loop kinematic guess toward the destination -- it is not
        guaranteed to decelerate, so handing it straight to the controller on
        solver failure could keep the vehicle moving at an unvalidated speed.
        This escalates with consecutive failure count instead, per the
        architecture proposal's Phase 4 fail-safe requirement:
          - below ``fail_safe_emergency_stop_failure_threshold``: keep the
            rollout's path (x, y, heading) but override its speed with a
            comfortable braking profile ("brake gently, hold last safe
            path").
          - at/above that threshold: switch to the strongest allowed braking
            (``min_acceleration_mps2``), i.e. a true emergency stop.
        """
        emergency = (
            int(self._consecutive_solver_failure_count)
            >= int(self.fail_safe_emergency_stop_failure_threshold)
        )
        speed_profile_mps = self._minimum_reachable_speed_profile_mps(
            current_speed_mps=float(x0[2]),
            current_acceleration_mps2=float(current_acceleration_mps2),
            braking_deceleration_mps2=(
                None if emergency else float(self.fail_safe_gentle_brake_deceleration_mps2)
            ),
        )

        x_solution = np.array(rollout_x, dtype=float)
        u_solution = np.array(rollout_u, dtype=float)
        min_velocity_mps = float(self.constraints.min_velocity_mps)
        for k in range(self.horizon_steps + 1):
            x_solution[k, 2] = max(min_velocity_mps, float(speed_profile_mps[k]))
        for k in range(int(u_solution.shape[0])):
            v_before_mps = float(x_solution[k, 2])
            v_after_mps = float(x_solution[k + 1, 2])
            u_solution[k, 0] = self._clamp(
                (v_after_mps - v_before_mps) / float(self.dt_s),
                float(self.constraints.min_acceleration_mps2),
                float(self.constraints.max_acceleration_mps2),
            )
        event_count = int(getattr(self, "_solver_failure_log_event_count", 0)) + 1
        self._solver_failure_log_event_count = int(event_count)
        emergency_first_report = bool(
            emergency
            and not bool(
                getattr(self, "_solver_failure_emergency_logged", False)
            )
        )
        log_every_n = max(
            1,
            int(getattr(self, "solver_failure_log_every_n", 50)),
        )
        if event_count == 1 or emergency_first_report or event_count % log_every_n == 0:
            print(
                "[MPC] "
                + ("EMERGENCY STOP" if emergency else "brake-gently")
                + " fail-safe fallback trajectory "
                + f"(consecutive_failures={int(self._consecutive_solver_failure_count)}, "
                + f"fallback_events={int(event_count)})"
            )
        if emergency:
            self._solver_failure_emergency_logged = True
        return x_solution, u_solution

    def _future_speed_upper_bound_mps(
        self,
        active_speed_upper_bound_mps: float,
        future_state_index: int,
        reachable_speed_floor_profile_mps: Sequence[float] | None = None,
    ) -> float:
        """
        Compute a feasible per-stage future speed upper bound.

        If the current speed is already above the active cap, the QP must still
        remain feasible under bounded braking. This upper bound therefore follows
        the fastest physically achievable deceleration envelope down toward the
        active cap.
        """

        future_state_index = max(1, int(future_state_index))
        base_max_velocity_mps = float(self.constraints.max_velocity_mps)
        min_velocity_mps = float(self.constraints.min_velocity_mps)
        active_speed_upper_bound_mps = min(float(base_max_velocity_mps), max(min_velocity_mps, float(active_speed_upper_bound_mps)))
        if reachable_speed_floor_profile_mps is None or len(reachable_speed_floor_profile_mps) == 0:
            reachable_speed_floor_mps = float(min_velocity_mps)
        else:
            profile_idx = min(int(future_state_index), len(reachable_speed_floor_profile_mps) - 1)
            reachable_speed_floor_mps = max(
                min_velocity_mps,
                float(reachable_speed_floor_profile_mps[profile_idx]),
            )
        return float(min(base_max_velocity_mps, max(active_speed_upper_bound_mps, reachable_speed_floor_mps)))

    def _cg_slip_angle_beta(self, delta_rad: float) -> float:
        """
        Intent:
            Compute the CG-reference kinematic bicycle slip angle beta.

        Equation:
            beta = atan((l_r / L) * tan(delta))
        """

        k_ratio = float(self.l_r_m / max(1e-9, self.wheelbase_m))
        return float(math.atan(k_ratio * math.tan(float(delta_rad))))

    def _cg_slip_angle_beta_derivative(self, delta_rad: float) -> float:
        """
        Intent:
            Compute d(beta)/d(delta) for linearizing the CG-reference model.
        """

        delta_rad = float(delta_rad)
        k_ratio = float(self.l_r_m / max(1e-9, self.wheelbase_m))
        cos_delta = math.cos(delta_rad)
        sec_delta_sq = 1.0 / max(1e-9, cos_delta * cos_delta)
        tan_delta = math.tan(delta_rad)
        return float((k_ratio * sec_delta_sq) / (1.0 + (k_ratio * tan_delta) ** 2))

    def _reference_rollout(
        self,
        x0: np.ndarray,
        x_ref_target: np.ndarray,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
        object_snapshots: Sequence[Mapping[str, object]],
        speed_upper_bound_mps: float | None = None,
        reachable_speed_floor_profile_mps: Sequence[float] | None = None,
        seed_state_traj: np.ndarray | None = None,
        seed_control_traj: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intent:
            Build a deterministic nonlinear rollout used as the linearization
            reference for the LTV-MPC QP.

        Logic:
            - Prefer stage-wise lane-center targets when available.
            - Blend path heading with line-of-sight heading to the next path point.
            - Reduce rollout speed when a lead obstacle is predicted ahead.
            - Propagate with the nonlinear CG-reference kinematic bicycle model.

        Output:
            x_ref_traj:
                np.ndarray, shape (N+1,4)
            u_ref_traj:
                np.ndarray, shape (N,2)
        """

        effective_speed_upper_bound_mps = float(self.constraints.max_velocity_mps)
        if speed_upper_bound_mps is not None:
            effective_speed_upper_bound_mps = min(
                float(effective_speed_upper_bound_mps),
                max(float(self.constraints.min_velocity_mps), float(speed_upper_bound_mps)),
            )
        if (
            seed_state_traj is not None
            and seed_control_traj is not None
            and seed_state_traj.shape == (self.horizon_steps + 1, self.nx)
            and seed_control_traj.shape == (self.horizon_steps, self.nu)
        ):
            x_seed = np.asarray(seed_state_traj, dtype=float).copy()
            u_seed = np.asarray(seed_control_traj, dtype=float).copy()
            x_seed[0] = np.asarray(x0, dtype=float)
            seed_speed_soft_active = bool(getattr(self, "speed_soft_constraint_enabled", False))
            for k in range(1, self.horizon_steps + 1):
                stage_speed_upper_bound_mps = self._future_speed_upper_bound_mps(
                    active_speed_upper_bound_mps=float(effective_speed_upper_bound_mps),
                    future_state_index=int(k),
                    reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
                )
                seed_clamp_upper_mps = float(stage_speed_upper_bound_mps)
                if seed_speed_soft_active:
                    seed_clamp_upper_mps += float(self.speed_soft_max_slack_mps)
                x_seed[k, 2] = self._clamp(
                    float(x_seed[k, 2]),
                    float(self.constraints.min_velocity_mps),
                    seed_clamp_upper_mps,
                )
            x_seed[:, 3] = np.asarray([self._wrap_angle(float(angle)) for angle in x_seed[:, 3]], dtype=float)
            return x_seed, u_seed

        x_ref_traj = np.zeros((self.horizon_steps + 1, self.nx), dtype=float)
        u_ref_traj = np.zeros((self.horizon_steps, self.nu), dtype=float)
        x_ref_traj[0] = x0

        x_goal, y_goal, v_goal, psi_goal = [float(v) for v in x_ref_target]
        v_goal = self._clamp(v_goal, self.constraints.min_velocity_mps, effective_speed_upper_bound_mps)
        psi_goal = self._wrap_angle(psi_goal)

        progress_lookup_active = bool(
            getattr(self, "lane_center_follow_use_progress_lookup", False)
        ) and bool(self.reference_prefer_lane_center_path) and bool(lane_center_reference)
        current_progress_m = 0.0
        if progress_lookup_active:
            current_progress_m, _ = self._nearest_progress_along_route(
                route_points=[
                    (float(s.get("x_ref_m", 0.0)), float(s.get("y_ref_m", 0.0)))
                    for s in lane_center_reference
                    if isinstance(s, Mapping)
                ],
                xy=[float(x0[0]), float(x0[1])],
            )

        for k in range(self.horizon_steps):
            x_m, y_m, v_mps, psi_rad = [float(v) for v in x_ref_traj[k]]
            target_x_m = float(x_goal)
            target_y_m = float(y_goal)
            path_heading_rad = float(psi_goal)

            if progress_lookup_active:
                # Query the position this stage's own traveled arc length so
                # far implies for the *next* stage, rather than indexing the
                # reference array by stage number k+1 -- the two only agree
                # on a straight reference built at exactly the nominal step
                # distance; on a curve, array position and true along-path
                # distance diverge.
                next_progress_estimate_m = float(current_progress_m) + max(0.5, float(v_mps) * float(self.dt_s))
                stage_ref = self._get_lane_center_stage_ref_by_progress(
                    lane_center_reference=lane_center_reference,
                    query_progress_m=next_progress_estimate_m,
                )
                if stage_ref is None:
                    stage_ref = self._get_lane_center_stage_ref(
                        lane_center_reference=lane_center_reference,
                        stage_index=int(k + 1),
                    )
            elif bool(self.reference_prefer_lane_center_path):
                stage_ref = self._get_lane_center_stage_ref(
                    lane_center_reference=lane_center_reference,
                    stage_index=int(k + 1),
                )
            else:
                stage_ref = None
            if stage_ref is not None:
                target_x_m = float(stage_ref[0])
                target_y_m = float(stage_ref[1])
                path_heading_rad = float(stage_ref[2])

            dx_target = target_x_m - x_m
            dy_target = target_y_m - y_m
            los_heading_rad = (
                math.atan2(dy_target, dx_target)
                if (abs(dx_target) + abs(dy_target)) > 1e-9
                else float(path_heading_rad)
            )
            desired_heading = self._blend_heading_angles(
                path_heading_rad=float(path_heading_rad),
                los_heading_rad=float(los_heading_rad),
                los_weight=float(self.reference_path_los_heading_blend),
            )
            heading_error = self._wrap_angle(desired_heading - psi_rad)
            # Scale steering authority linearly with speed to prevent the
            # rollout from curving into circular arcs at near-zero speed.
            # Below 0.5 m/s the gain tapers to 0; above 1.5 m/s it is full.
            _low_spd_lo = 0.5   # [m/s] gain = 0 at or below this speed
            _low_spd_hi = 1.5   # [m/s] gain = 1 at or above this speed
            _spd_scale = min(
                1.0,
                max(0.0, (v_mps - _low_spd_lo) / max(1e-9, _low_spd_hi - _low_spd_lo)),
            )
            delta_des = self._clamp(
                self.reference_heading_gain * heading_error * _spd_scale,
                self.constraints.min_steer_rad,
                self.constraints.max_steer_rad,
            )

            stage_speed_target_mps = self._compute_reference_rollout_speed_limit(
                stage_x_m=x_m,
                stage_y_m=y_m,
                stage_heading_rad=float(path_heading_rad),
                stage_index=int(k),
                base_speed_mps=float(v_goal),
                object_snapshots=object_snapshots,
            )
            next_stage_speed_upper_bound_mps = self._future_speed_upper_bound_mps(
                active_speed_upper_bound_mps=float(effective_speed_upper_bound_mps),
                future_state_index=int(k + 1),
                reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
            )
            if bool(getattr(self, "speed_soft_constraint_enabled", False)):
                # Keep the rollout's own linearization reference consistent
                # with what _build_qp now permits the solved trajectory to
                # reach, so the QP doesn't linearize around an artificially
                # conservative point.
                next_stage_speed_upper_bound_mps = float(
                    next_stage_speed_upper_bound_mps
                ) + float(self.speed_soft_max_slack_mps)
            stage_speed_target_mps = min(float(stage_speed_target_mps), float(next_stage_speed_upper_bound_mps))
            accel_des = self.reference_speed_gain * (stage_speed_target_mps - v_mps)
            accel_des = self._clamp(
                accel_des,
                self.constraints.min_acceleration_mps2,
                self.constraints.max_acceleration_mps2,
            )

            u_ref_traj[k] = np.array([accel_des, delta_des], dtype=float)

            beta_rad = self._cg_slip_angle_beta(delta_des)
            x_next = x_m + self.dt_s * v_mps * math.cos(psi_rad + beta_rad)
            y_next = y_m + self.dt_s * v_mps * math.sin(psi_rad + beta_rad)
            v_next = self._clamp(
                v_mps + self.dt_s * accel_des,
                self.constraints.min_velocity_mps,
                float(next_stage_speed_upper_bound_mps),
            )
            psi_next = self._wrap_angle(psi_rad + self.dt_s * (v_mps / self.l_r_m) * math.sin(beta_rad))
            x_ref_traj[k + 1] = np.array([x_next, y_next, v_next, psi_next], dtype=float)
            if progress_lookup_active:
                current_progress_m += math.hypot(float(x_next) - x_m, float(y_next) - y_m)

        return x_ref_traj, u_ref_traj

    def _speed_tracking_reference(
        self,
        *,
        x0: np.ndarray,
        x_ref_target: np.ndarray,
        linearization_rollout: np.ndarray,
        object_snapshots: Sequence[Mapping[str, object]],
        current_acceleration_mps2: float,
        speed_upper_bound_mps: float | None = None,
        reachable_speed_floor_profile_mps: Sequence[float] | None = None,
    ) -> np.ndarray:
        """Build the QP velocity reference independently of ``speed_gain``.

        ``_reference_rollout`` remains a nonlinear warm start and
        linearization trajectory.  This profile is the longitudinal contract
        the QP actually tracks: it approaches the SpeedPlanner target through
        the same acceleration and jerk limits enforced by the optimization.
        """

        rollout = np.asarray(linearization_rollout, dtype=float)
        if rollout.shape != (self.horizon_steps + 1, self.nx):
            raise ValueError("linearization rollout must be shaped (N+1,nx)")
        effective_upper_mps = float(self.constraints.max_velocity_mps)
        if speed_upper_bound_mps is not None:
            effective_upper_mps = min(
                float(effective_upper_mps),
                max(
                    float(self.constraints.min_velocity_mps),
                    float(speed_upper_bound_mps),
                ),
            )
        base_target_mps = self._clamp(
            float(x_ref_target[2]),
            float(self.constraints.min_velocity_mps),
            float(effective_upper_mps),
        )
        profile = np.zeros(self.horizon_steps + 1, dtype=float)
        profile[0] = self._clamp(
            float(x0[2]),
            float(self.constraints.min_velocity_mps),
            float(self.constraints.max_velocity_mps),
        )
        previous_accel_mps2 = self._clamp(
            float(current_acceleration_mps2),
            float(self.constraints.min_acceleration_mps2),
            float(self.constraints.max_acceleration_mps2),
        )
        jerk_step_mps2 = (
            float(self.constraints.max_jerk_mps3) * float(self.dt_s)
        )
        for k in range(self.horizon_steps):
            stage_target_mps = self._compute_reference_rollout_speed_limit(
                stage_x_m=float(rollout[k, 0]),
                stage_y_m=float(rollout[k, 1]),
                stage_heading_rad=float(rollout[k, 3]),
                stage_index=int(k),
                base_speed_mps=float(base_target_mps),
                object_snapshots=object_snapshots,
            )
            stage_upper_mps = self._future_speed_upper_bound_mps(
                active_speed_upper_bound_mps=float(effective_upper_mps),
                future_state_index=int(k + 1),
                reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
            )
            if bool(getattr(self, "speed_soft_constraint_enabled", False)):
                stage_upper_mps += float(self.speed_soft_max_slack_mps)
            stage_target_mps = min(float(stage_target_mps), float(stage_upper_mps))
            desired_accel_mps2 = self._clamp(
                (float(stage_target_mps) - float(profile[k]))
                / max(1.0e-9, float(self.dt_s)),
                float(self.constraints.min_acceleration_mps2),
                float(self.constraints.max_acceleration_mps2),
            )
            accel_mps2 = self._clamp(
                float(desired_accel_mps2),
                float(previous_accel_mps2) - float(jerk_step_mps2),
                float(previous_accel_mps2) + float(jerk_step_mps2),
            )
            accel_mps2 = self._clamp(
                float(accel_mps2),
                float(self.constraints.min_acceleration_mps2),
                float(self.constraints.max_acceleration_mps2),
            )
            profile[k + 1] = self._clamp(
                float(profile[k]) + float(self.dt_s) * float(accel_mps2),
                float(self.constraints.min_velocity_mps),
                float(stage_upper_mps),
            )
            previous_accel_mps2 = float(accel_mps2)
        return profile

    def _linearize_dynamics(self, x_bar: np.ndarray, u_bar: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Intent:
            Linearize Euler-discretized CG-reference kinematic bicycle dynamics
            around one reference point (x_bar, u_bar).

        QP form used:
            X_{k+1} = A_k X_k + B_k U_k + c_k
        """

        x_m, y_m, v_mps, psi_rad = [float(v) for v in x_bar]
        a_mps2, delta_rad = [float(v) for v in u_bar]
        _ = x_m, y_m, a_mps2

        dt_s = float(self.dt_s)
        l_r_m = float(self.l_r_m)
        beta_rad = self._cg_slip_angle_beta(delta_rad)
        beta_delta = self._cg_slip_angle_beta_derivative(delta_rad)
        cos_sum = math.cos(psi_rad + beta_rad)
        sin_sum = math.sin(psi_rad + beta_rad)
        sin_beta = math.sin(beta_rad)
        cos_beta = math.cos(beta_rad)

        A_k = np.array(
            [
                [1.0, 0.0, dt_s * cos_sum, -dt_s * v_mps * sin_sum],
                [0.0, 1.0, dt_s * sin_sum, dt_s * v_mps * cos_sum],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, dt_s * sin_beta / l_r_m, 1.0],
            ],
            dtype=float,
        )
        B_k = np.array(
            [
                [0.0, -dt_s * v_mps * sin_sum * beta_delta],
                [0.0, dt_s * v_mps * cos_sum * beta_delta],
                [dt_s, 0.0],
                [0.0, dt_s * (v_mps / l_r_m) * cos_beta * beta_delta],
            ],
            dtype=float,
        )

        f_bar = np.array(
            [
                float(x_bar[0] + dt_s * v_mps * cos_sum),
                float(x_bar[1] + dt_s * v_mps * sin_sum),
                float(x_bar[2] + dt_s * u_bar[0]),
                float(self._wrap_angle(x_bar[3] + dt_s * (v_mps / l_r_m) * sin_beta)),
            ],
            dtype=float,
        )
        c_k = f_bar - A_k @ x_bar - B_k @ u_bar
        return A_k, B_k, c_k

    def _build_qp(
        self,
        x0: np.ndarray,
        x_ref_target: np.ndarray,
        object_snapshots: Sequence[Mapping[str, object]],
        current_acceleration_mps2: float,
        current_steering_rad: float,
        x_ref_rollout: np.ndarray,
        u_ref_rollout: np.ndarray,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
        speed_upper_bound_mps: float | None,
        reachable_speed_floor_profile_mps: Sequence[float] | None,
        road_envelope_blocks: Mapping[str, object] | None = None,
        speed_tracking_reference_mps: Sequence[float] | None = None,
    ) -> Tuple[sp.csc_matrix, np.ndarray, sp.csc_matrix, np.ndarray, np.ndarray, QPIndex]:
        """
        Intent:
            Build the full convex QP matrices for OSQP.

        Important implementation note:
            Collision-checker auxiliary constraints are disabled in this mode.
            Obstacle handling is done through repulsive-potential cost shaping.
        """

        object_count = len(object_snapshots)
        envelope_blocks_list = (
            list(road_envelope_blocks.get("blocks", []) or [])
            if isinstance(road_envelope_blocks, Mapping)
            else []
        )
        road_envelope_term_active = (
            bool(getattr(self, "road_envelope_enabled", False))
            and float(getattr(self, "road_envelope_weight", 0.0)) > 0.0
            and len(envelope_blocks_list) > 0
        )
        road_boundary_term_active = (
            bool(getattr(self, "road_boundary_enabled", True))
            and float(getattr(self, "road_boundary_weight", self.lane_keep_boundary_weight)) > 0.0
            and not road_envelope_term_active
        )
        speed_soft_term_active = bool(getattr(self, "speed_soft_constraint_enabled", False))
        index = QPIndex(
            nx=self.nx,
            nu=self.nu,
            horizon_steps=self.horizon_steps,
            road_boundary_slack_pair_count=(
                int(self.horizon_steps) if road_boundary_term_active else 0
            ),
            speed_slack_count=(int(self.horizon_steps) if speed_soft_term_active else 0),
            road_envelope_slack_count=(
                int(self.horizon_steps) if road_envelope_term_active else 0
            ),
        )
        n_var = index.total_variables
        effective_speed_upper_bound_mps = float(self.constraints.max_velocity_mps)
        if speed_upper_bound_mps is not None:
            effective_speed_upper_bound_mps = min(
                float(effective_speed_upper_bound_mps),
                max(float(self.constraints.min_velocity_mps), float(speed_upper_bound_mps)),
            )

        # --- Helpers for sparse QP assembly ---
        q = np.zeros(n_var, dtype=float)
        p_entries: Dict[Tuple[int, int], float] = {}
        a_row: List[int] = []
        a_col: List[int] = []
        a_data: List[float] = []
        lower_bounds: List[float] = []
        upper_bounds: List[float] = []

        def add_p_entry(i: int, j: int, value: float) -> None:
            if abs(value) < 1e-12:
                return
            ii, jj = (i, j) if i <= j else (j, i)
            p_entries[(ii, jj)] = p_entries.get((ii, jj), 0.0) + float(value)

        def add_quadratic(var_idx: int, weight: float) -> None:
            if weight <= 0.0:
                return
            add_p_entry(var_idx, var_idx, 2.0 * float(weight))

        def add_tracking(var_idx: int, weight: float, ref_value: float) -> None:
            if weight <= 0.0:
                return
            add_quadratic(var_idx, weight)
            q[var_idx] += -2.0 * float(weight) * float(ref_value)

        def add_constraint(coeffs: Mapping[int, float], lower: float, upper: float) -> None:
            row_idx = len(lower_bounds)
            for col_idx, coeff in coeffs.items():
                if abs(float(coeff)) < 1e-12:
                    continue
                a_row.append(row_idx)
                a_col.append(int(col_idx))
                a_data.append(float(coeff))
            lower_bounds.append(float(lower))
            upper_bounds.append(float(upper))

        # X_0 is fixed, so tracking starts at stage 1. Each predicted state must
        # track its matching time-parameterized rollout sample. Applying the
        # terminal destination to every stage creates a receding-horizon
        # accelerate/brake limit cycle as the endpoint moves forward each tick.

        # --- Objective: control term Cost_Control ---
        # Penalizes rapid changes in acceleration and steering across horizon.
        comfort_scale = float(self.comfort_cost.w_comf)

        # J_ctrl rate terms: ((a_k-a_{k-1})/dt)^2 + ((delta_k-delta_{k-1})/dt)^2
        qa_eff = comfort_scale * float(self.comfort_cost.qa) / max(1e-9, self.dt_s * self.dt_s)
        qd_eff = comfort_scale * float(self.comfort_cost.qdelta) / max(1e-9, self.dt_s * self.dt_s)

        def add_rate_penalty(var_idx: int, prev_idx: int | None, prev_value: float, weight: float) -> None:
            if weight <= 0.0:
                return
            if prev_idx is None:
                # (u - u_prev_const)^2
                add_quadratic(var_idx, weight)
                q[var_idx] += -2.0 * float(weight) * float(prev_value)
                return
            # (u_k - u_{k-1})^2 = u_k^2 + u_{k-1}^2 - 2 u_k u_{k-1}
            add_quadratic(var_idx, weight)
            add_quadratic(prev_idx, weight)
            add_p_entry(var_idx, prev_idx, -2.0 * float(weight))

        for k in range(self.horizon_steps):
            a_idx = index.control_index(k, 0)
            d_idx = index.control_index(k, 1)
            if k == 0:
                add_rate_penalty(a_idx, None, float(current_acceleration_mps2), qa_eff)
                add_rate_penalty(d_idx, None, float(current_steering_rad), qd_eff)
            else:
                add_rate_penalty(a_idx, index.control_index(k - 1, 0), 0.0, qa_eff)
                add_rate_penalty(d_idx, index.control_index(k - 1, 1), 0.0, qd_eff)

        # --- Objective: attractive term Cost_ref ---
        # Quadratic pull to destination reference state.
        attractive_scale = float(self.safety_cost.w_safe)
        w_qx_safe = attractive_scale * float(self.comfort_cost.qx)
        w_qy_safe = attractive_scale * float(self.comfort_cost.qy)
        w_qv_safe = attractive_scale * float(self.comfort_cost.qv)
        w_qpsi_safe = attractive_scale * float(self.comfort_cost.qpsi)

        qp_progress_lookup_active = bool(
            getattr(self, "lane_center_follow_use_progress_lookup", False)
        ) and bool(lane_center_reference)
        qp_stage_progress_m: List[float] = []
        if qp_progress_lookup_active:
            ego_progress_m, _ = self._nearest_progress_along_route(
                route_points=[
                    (float(s.get("x_ref_m", 0.0)), float(s.get("y_ref_m", 0.0)))
                    for s in lane_center_reference
                    if isinstance(s, Mapping)
                ],
                xy=[float(x0[0]), float(x0[1])],
            )
            cumulative_distance_m = 0.0
            qp_stage_progress_m = [float(ego_progress_m)]
            for j in range(self.horizon_steps):
                cumulative_distance_m += math.hypot(
                    float(x_ref_rollout[j + 1, 0]) - float(x_ref_rollout[j, 0]),
                    float(x_ref_rollout[j + 1, 1]) - float(x_ref_rollout[j, 1]),
                )
                qp_stage_progress_m.append(float(ego_progress_m) + float(cumulative_distance_m))

        for k in range(1, self.horizon_steps + 1):
            stage_reference = self._tracking_reference_at_stage(
                x_ref_rollout=x_ref_rollout,
                stage_index=int(k),
            )
            x_ref_value = float(stage_reference[0])
            y_ref_value = float(stage_reference[1])
            v_ref_value = float(stage_reference[2])
            if (
                speed_tracking_reference_mps is not None
                and len(speed_tracking_reference_mps) > int(k)
            ):
                v_ref_value = float(speed_tracking_reference_mps[int(k)])
            psi_ref_value = self._wrap_angle(float(stage_reference[3]))
            x_k_idx = index.state_index(k, 0)
            y_k_idx = index.state_index(k, 1)
            add_tracking(x_k_idx, w_qx_safe, x_ref_value)
            add_tracking(y_k_idx, w_qy_safe, y_ref_value)
            add_tracking(index.state_index(k, 2), w_qv_safe, v_ref_value)
            add_tracking(index.state_index(k, 3), w_qpsi_safe, psi_ref_value)

            # Lane-center-follow term:
            #   P_att_lane = w_lane * e_y^2 + w_lane * q_psi_lane * e_psi^2,
            # where
            #   e_y   = -(x-x_ref)sin(theta_ref) + (y-y_ref)cos(theta_ref)
            #   e_psi = wrap(psi - theta_ref)
            lane_sample = None
            if qp_progress_lookup_active:
                lane_sample = self._get_lane_center_stage_sample_by_progress(
                    lane_center_reference=lane_center_reference,
                    query_progress_m=qp_stage_progress_m[k],
                )
            if lane_sample is None:
                lane_sample = self._get_lane_center_stage_sample(
                    lane_center_reference=lane_center_reference,
                    stage_index=int(k),
                    query_x_m=float(x_ref_rollout[k, 0]),
                    query_y_m=float(x_ref_rollout[k, 1]),
                )
            lane_reference = normalize_lane_reference_sample(
                lane_sample,
                default_lane_width_m=float(getattr(self, "lane_width_m", 4.0)),
            )
            if lane_reference is not None:
                centerline_xy_weight = float(getattr(self, "lane_center_follow_xy_weight", 0.0))
                if bool(self.lane_center_follow_enabled) and float(centerline_xy_weight) > 0.0:
                    if bool(getattr(self, "lane_center_follow_xy_uses_frenet_decomposition", False)):
                        # Decomposed form: penalize along-track deviation
                        # from the reference point in the lane frame instead
                        # of pulling raw world x and y toward
                        # (x_center_m, y_center_m) independently -- both are
                        # zero exactly at the reference point, but this form
                        # doesn't couple world-frame x/y errors together and
                        # rotates with lane heading the way the lateral term
                        # below already does.
                        long_affine = signed_longitudinal_progress_affine_form(lane_reference)
                        p_coef = float(long_affine.x_coef)
                        q_coef = float(long_affine.y_coef)
                        r_coef = float(long_affine.constant)
                        add_quadratic(x_k_idx, centerline_xy_weight * p_coef * p_coef)
                        add_quadratic(y_k_idx, centerline_xy_weight * q_coef * q_coef)
                        add_p_entry(x_k_idx, y_k_idx, 2.0 * centerline_xy_weight * p_coef * q_coef)
                        q[x_k_idx] += 2.0 * centerline_xy_weight * p_coef * r_coef
                        q[y_k_idx] += 2.0 * centerline_xy_weight * q_coef * r_coef
                    else:
                        add_tracking(x_k_idx, centerline_xy_weight, float(lane_reference.x_center_m))
                        add_tracking(y_k_idx, centerline_xy_weight, float(lane_reference.y_center_m))

                lane_affine = signed_lateral_offset_affine_form(lane_reference)
                a_coef = float(lane_affine.x_coef)
                b_coef = float(lane_affine.y_coef)
                c_coef = float(lane_affine.constant)
                lane_heading_ref = float(lane_reference.heading_rad)

                if bool(self.lane_center_follow_enabled) and float(self.lane_center_follow_weight) > 0.0:
                    lane_weight = float(self.lane_center_follow_weight)
                    lane_heading_weight = lane_weight * float(self.lane_center_follow_qpsi)
                    lane_heading_ref_aligned = self._align_angle_near(
                        angle_rad=float(lane_heading_ref),
                        around_rad=float(x_ref_rollout[k, 3]),
                    )

                    add_quadratic(x_k_idx, lane_weight * a_coef * a_coef)
                    add_quadratic(y_k_idx, lane_weight * b_coef * b_coef)
                    add_p_entry(x_k_idx, y_k_idx, 2.0 * lane_weight * a_coef * b_coef)
                    q[x_k_idx] += 2.0 * lane_weight * a_coef * c_coef
                    q[y_k_idx] += 2.0 * lane_weight * b_coef * c_coef
                    if lane_heading_weight > 0.0:
                        add_tracking(index.state_index(k, 3), lane_heading_weight, lane_heading_ref_aligned)

                if road_boundary_term_active:
                    left_slack_idx = index.road_boundary_left_slack_index(k)
                    right_slack_idx = index.road_boundary_right_slack_index(k)
                    road_weight = float(getattr(self, "road_boundary_weight", self.lane_keep_boundary_weight))
                    road_margin_m = float(getattr(self, "road_boundary_margin_m", 0.5))
                    road_center_offset_m = float(lane_reference.road_center_offset_m)
                    road_left_width_m = float(lane_reference.left_road_width_m)
                    road_right_width_m = float(lane_reference.right_road_width_m)
                    add_quadratic(left_slack_idx, road_weight)
                    add_quadratic(right_slack_idx, road_weight)
                    add_constraint(
                        {
                            x_k_idx: -a_coef,
                            y_k_idx: -b_coef,
                            left_slack_idx: 1.0,
                        },
                        float(road_margin_m) - float(road_left_width_m) + float(c_coef) - float(road_center_offset_m),
                        np.inf,
                    )
                    add_constraint(
                        {
                            x_k_idx: a_coef,
                            y_k_idx: b_coef,
                            right_slack_idx: 1.0,
                        },
                        float(road_margin_m) - float(road_right_width_m) - float(c_coef) + float(road_center_offset_m),
                        np.inf,
                    )
                    road_max_slack_m = float(getattr(self, "road_boundary_max_slack_m", np.inf))
                    road_slack_upper = (
                        float(road_max_slack_m)
                        if math.isfinite(float(road_max_slack_m)) and float(road_max_slack_m) > 0.0
                        else np.inf
                    )
                    add_constraint({left_slack_idx: 1.0}, 0.0, road_slack_upper)
                    add_constraint({right_slack_idx: 1.0}, 0.0, road_slack_upper)

                if road_envelope_term_active:
                    # Road-envelope block-union hard constraint (Yu et al.,
                    # arXiv:2509.18506, Sec. III-B1), gated in mutually
                    # exclusive to road_boundary_term_active above. Unlike
                    # the line-based road_boundary constraint (tied to
                    # whichever lane_reference sample is active this tick,
                    # which can jump when the tracked reference switches
                    # lanes mid-maneuver), this constraint is linearized
                    # around the current rollout point against a UNION of
                    # static blocks that never move for the duration of the
                    # locked maneuver -- so it stays satisfiable even when
                    # the reference itself jumps.
                    stage_x0_m = float(x_ref_rollout[k, 0])
                    stage_y0_m = float(x_ref_rollout[k, 1])
                    envelope_rho = float(
                        road_envelope_blocks.get(
                            "rho", getattr(self, "road_envelope_rho", -8.0)
                        )
                    )
                    envelope_epsilon0 = float(road_envelope_blocks.get("epsilon0", 0.0))
                    g_lse0, dg_dx, dg_dy, _weights = road_envelope_union_logsumexp(
                        blocks=envelope_blocks_list,
                        rho=envelope_rho,
                        x_m=stage_x0_m,
                        y_m=stage_y0_m,
                    )
                    h0 = float(g_lse0) - float(envelope_epsilon0)
                    envelope_slack_idx = index.road_envelope_slack_index(k)
                    envelope_weight = float(
                        getattr(self, "road_envelope_weight", 0.0)
                    )
                    add_quadratic(envelope_slack_idx, envelope_weight)
                    add_constraint(
                        {
                            x_k_idx: float(dg_dx),
                            y_k_idx: float(dg_dy),
                            envelope_slack_idx: -1.0,
                        },
                        -np.inf,
                        float(dg_dx * stage_x0_m + dg_dy * stage_y0_m - h0),
                    )
                    # A rolling turn tube may need more recovery room than a
                    # locked lane-change envelope.  Let the payload override
                    # only the slack ceiling; the same large quadratic weight
                    # still drives the solution back into the road tube.
                    envelope_max_slack_m = float(
                        road_envelope_blocks.get(
                            "max_slack_m",
                            getattr(self, "road_envelope_max_slack_m", np.inf),
                        )
                    )
                    envelope_slack_upper = (
                        float(envelope_max_slack_m)
                        if math.isfinite(float(envelope_max_slack_m))
                        and float(envelope_max_slack_m) > 0.0
                        else np.inf
                    )
                    add_constraint(
                        {envelope_slack_idx: 1.0}, 0.0, envelope_slack_upper
                    )

        # --- Objective: repulsive potential field Cost_Repulsive ---
        # Super-ellipsoid obstacle cost from `super_ellipsoid.py`, approximated
        # by a local quadratic Taylor model in [x, y, v, psi] for each stage.
        if bool(self.repulsive_cost.enabled) and object_count > 0:
            for k in range(1, self.horizon_steps + 1):
                stage_idx = k - 1
                x_idx = index.state_index(k, 0)
                y_idx = index.state_index(k, 1)
                v_idx = index.state_index(k, 2)
                psi_idx = index.state_index(k, 3)

                ego_state_ref = np.array(
                    [
                        float(x_ref_rollout[k, 0]),
                        float(x_ref_rollout[k, 1]),
                        float(x_ref_rollout[k, 2]),
                        float(self._wrap_angle(float(x_ref_rollout[k, 3]))),
                    ],
                    dtype=float,
                )
                state_indices = [x_idx, y_idx, v_idx, psi_idx]

                for object_snapshot in object_snapshots:
                    obj_state = self._get_object_state_at_stage(
                        object_snapshot=object_snapshot,
                        stage_index=stage_idx,
                        dt_s=float(self.dt_s),
                    )

                    repulsive_weight = float(object_snapshot.get("repulsive_class_weight", 1.0))
                    if repulsive_weight <= 0.0:
                        continue

                    obstacle_length_m = float(object_snapshot.get("length_m", 4.5))
                    obstacle_width_m = float(object_snapshot.get("width_m", 2.0))
                    _p0, gradient, hessian = self._superellipsoid_cost_taylor_terms(
                        ego_state_ref=ego_state_ref,
                        obstacle_state=obj_state,
                        obstacle_length_m=obstacle_length_m,
                        obstacle_width_m=obstacle_width_m,
                    )

                    gradient = float(repulsive_weight) * np.asarray(gradient, dtype=float)
                    hessian = float(repulsive_weight) * np.asarray(hessian, dtype=float)

                    if bool(self.repulsive_cost.cross_track_suppression_enabled):
                        stage_heading_rad = float(ego_state_ref[3])
                        cos_h = math.cos(stage_heading_rad)
                        sin_h = math.sin(stage_heading_rad)
                        cross_track_m = (
                            -(float(obj_state[0]) - float(ego_state_ref[0])) * sin_h
                            + (float(obj_state[1]) - float(ego_state_ref[1])) * cos_h
                        )
                        lateral_scale = self._cross_track_lateral_scale(
                            cross_track_abs_m=abs(cross_track_m),
                            full_suppression_m=float(
                                self.repulsive_cost.cross_track_full_suppression_m
                            ),
                            full_response_m=float(
                                self.repulsive_cost.cross_track_full_response_m
                            ),
                        )
                        if lateral_scale < 1.0:
                            # R is orthogonal (rotation by the stage heading),
                            # so rotating into (along, cross), scaling only
                            # what touches cross-track, and rotating back is
                            # an exact change of basis -- the along-track
                            # (braking) contribution is left untouched.
                            rotation = np.array(
                                [[cos_h, sin_h], [-sin_h, cos_h]], dtype=float
                            )
                            g_rot = rotation @ gradient[0:2]
                            h_rot = rotation @ hessian[0:2, 0:2] @ rotation.T
                            g_rot[1] *= lateral_scale
                            h_rot[0, 1] *= lateral_scale
                            h_rot[1, 0] *= lateral_scale
                            h_rot[1, 1] *= lateral_scale
                            gradient[0:2] = rotation.T @ g_rot
                            hessian[0:2, 0:2] = rotation.T @ h_rot @ rotation

                    if bool(self.repulsive_cost.project_hessian_psd):
                        hessian = self._project_symmetric_hessian_to_psd(hessian=hessian)

                    linear_term = np.asarray(gradient - hessian @ ego_state_ref, dtype=float)

                    for row_local, row_idx in enumerate(state_indices):
                        q[row_idx] += float(linear_term[row_local])
                        for col_local in range(row_local, len(state_indices)):
                            add_p_entry(
                                row_idx,
                                state_indices[col_local],
                                float(hessian[row_local, col_local]),
                            )

        # Optional tiny regularization on controls to improve numerical conditioning.
        # This does not change the problem meaningfully but stabilizes OSQP.
        tiny_reg = 1e-6
        for k in range(self.horizon_steps):
            add_quadratic(index.control_index(k, 0), tiny_reg)
            add_quadratic(index.control_index(k, 1), tiny_reg)
        if road_boundary_term_active:
            for k in range(1, self.horizon_steps + 1):
                add_quadratic(index.road_boundary_left_slack_index(k), tiny_reg)
                add_quadratic(index.road_boundary_right_slack_index(k), tiny_reg)
        if speed_soft_term_active:
            for k in range(1, self.horizon_steps + 1):
                add_quadratic(index.speed_slack_index(k), tiny_reg)
        if road_envelope_term_active:
            for k in range(1, self.horizon_steps + 1):
                add_quadratic(index.road_envelope_slack_index(k), tiny_reg)

        # --- Constraints ---
        # Initial state equality X_0 = current state.
        for i in range(self.nx):
            add_constraint({index.state_index(0, i): 1.0}, float(x0[i]), float(x0[i]))

        # LTV dynamics equality constraints.
        for k in range(self.horizon_steps):
            A_k, B_k, c_k = self._linearize_dynamics(x_ref_rollout[k], u_ref_rollout[k])
            for i in range(self.nx):
                coeffs: Dict[int, float] = {index.state_index(k + 1, i): 1.0}
                for j in range(self.nx):
                    coeffs[index.state_index(k, j)] = coeffs.get(index.state_index(k, j), 0.0) - float(A_k[i, j])
                for j in range(self.nu):
                    coeffs[index.control_index(k, j)] = coeffs.get(index.control_index(k, j), 0.0) - float(B_k[i, j])
                add_constraint(coeffs, float(c_k[i]), float(c_k[i]))

        # Speed constraints for future states. The lower bound (near-zero
        # floor) always stays hard. The upper bound (the "speed profile
        # limit cap", e.g. curve/traffic-light/lead-obstacle speed caps) is
        # hard by default, matching prior behavior exactly. When
        # speed_soft_term_active, it becomes a slack-penalized soft bound
        # instead -- mirrors the road-boundary slack pattern above: v_k is
        # allowed to exceed the posted cap only by paying a quadratic
        # penalty per unit overshoot, itself hard-capped at
        # speed_soft_max_slack_mps so the softened bound still cannot be
        # violated without limit.
        for k in range(1, self.horizon_steps + 1):
            stage_speed_upper_bound_mps = self._future_speed_upper_bound_mps(
                active_speed_upper_bound_mps=float(effective_speed_upper_bound_mps),
                future_state_index=int(k),
                reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
            )
            v_idx = index.state_index(k, 2)
            if speed_soft_term_active:
                add_constraint({v_idx: 1.0}, self.constraints.min_velocity_mps, np.inf)
                slack_idx = index.speed_slack_index(k)
                add_quadratic(slack_idx, float(self.speed_soft_constraint_weight))
                add_constraint(
                    {v_idx: 1.0, slack_idx: -1.0},
                    -np.inf,
                    float(stage_speed_upper_bound_mps),
                )
                add_constraint(
                    {slack_idx: 1.0},
                    0.0,
                    float(self.speed_soft_max_slack_mps),
                )
            else:
                add_constraint(
                    {v_idx: 1.0},
                    self.constraints.min_velocity_mps,
                    float(stage_speed_upper_bound_mps),
                )
        # Optional hard terminal-speed constraint. Apply it only for stop-like
        # destinations (destination speed near zero), otherwise every rolling
        # temporary goal would incorrectly force the horizon-end speed to zero.
        terminal_speed_constraint_active = bool(self.constraints.enforce_terminal_velocity_constraint) and (
            abs(float(x_ref_target[2])) <= float(self.final_stop_speed_cap_activation_threshold_mps)
        )
        if terminal_speed_constraint_active:
            add_constraint(
                {index.state_index(self.horizon_steps, 2): 1.0},
                float(self.constraints.terminal_velocity_mps),
                float(self.constraints.terminal_velocity_mps),
            )

        # Acceleration and steering bounds.
        for k in range(self.horizon_steps):
            add_constraint(
                {index.control_index(k, 0): 1.0},
                self.constraints.min_acceleration_mps2,
                self.constraints.max_acceleration_mps2,
            )
            add_constraint(
                {index.control_index(k, 1): 1.0},
                self.constraints.min_steer_rad,
                self.constraints.max_steer_rad,
            )
        # Jerk bounds |a_k - a_{k-1}| <= j_max * dt.
        jerk_delta_limit = float(self.constraints.max_jerk_mps3) * float(self.dt_s)
        for k in range(self.horizon_steps):
            a_k_idx = index.control_index(k, 0)
            if k == 0:
                add_constraint(
                    {a_k_idx: 1.0},
                    float(current_acceleration_mps2) - jerk_delta_limit,
                    float(current_acceleration_mps2) + jerk_delta_limit,
                )
            else:
                a_km1_idx = index.control_index(k - 1, 0)
                add_constraint({a_k_idx: 1.0, a_km1_idx: -1.0}, -jerk_delta_limit, jerk_delta_limit)

        # Steering-rate bounds:
        #   min_rate <= (delta_k - delta_{k-1}) / dt <= max_rate
        # where delta_{-1} is the currently applied steering angle.
        steer_delta_min = float(self.constraints.min_steer_rate_rps) * float(self.dt_s)
        steer_delta_max = float(self.constraints.max_steer_rate_rps) * float(self.dt_s)
        for k in range(self.horizon_steps):
            d_k_idx = index.control_index(k, 1)
            if k == 0:
                add_constraint(
                    {d_k_idx: 1.0},
                    float(current_steering_rad) + steer_delta_min,
                    float(current_steering_rad) + steer_delta_max,
                )
            else:
                d_km1_idx = index.control_index(k - 1, 1)
                add_constraint(
                    {d_k_idx: 1.0, d_km1_idx: -1.0},
                    steer_delta_min,
                    steer_delta_max,
                )
        # Collision-checker constraints removed per configuration.

        # Assemble sparse matrices.
        if len(p_entries) == 0:
            P = sp.csc_matrix((n_var, n_var), dtype=float)
        else:
            p_rows = [idx_pair[0] for idx_pair in p_entries.keys()]
            p_cols = [idx_pair[1] for idx_pair in p_entries.keys()]
            p_vals = [val for val in p_entries.values()]
            P = sp.csc_matrix((p_vals, (p_rows, p_cols)), shape=(n_var, n_var), dtype=float)

        A = sp.csc_matrix((a_data, (a_row, a_col)), shape=(len(lower_bounds), n_var), dtype=float)
        l = np.asarray(lower_bounds, dtype=float)
        u = np.asarray(upper_bounds, dtype=float)
        return P, q, A, l, u, index

    @staticmethod
    def _tracking_reference_at_stage(
        *,
        x_ref_rollout: np.ndarray,
        stage_index: int,
    ) -> np.ndarray:
        """Return the time-matched state reference for one MPC stage."""

        rollout = np.asarray(x_ref_rollout, dtype=float)
        if rollout.ndim != 2 or rollout.shape[0] == 0 or rollout.shape[1] < 4:
            raise ValueError("MPC tracking rollout must contain Nx4 states")
        index = max(0, min(int(stage_index), int(rollout.shape[0]) - 1))
        return np.asarray(rollout[index, :4], dtype=float)

    def _solve_qp(
        self,
        P: sp.csc_matrix,
        q: np.ndarray,
        A: sp.csc_matrix,
        l: np.ndarray,
        u: np.ndarray,
    ) -> Tuple[np.ndarray | None, str, float]:
        """Solve the QP with OSQP and return solution/status/time."""

        solver = osqp.OSQP()  # type: ignore[union-attr]
        t0 = time.perf_counter()
        solver.setup(
            P=P,
            q=q,
            A=A,
            l=l,
            u=u,
            verbose=False,
            warm_start= True,
            polish=self.qp_polish,
            max_iter=self.qp_max_iter,
            eps_abs=self.qp_eps_abs,
            eps_rel=self.qp_eps_rel,
            adaptive_rho=True,
        )
        result = solver.solve()
        solve_time_ms = (time.perf_counter() - t0) * 1000.0
        status = str(result.info.status).lower()
        if result.x is None or "solved" not in status:
            return None, status, float(solve_time_ms)
        return np.asarray(result.x, dtype=float), status, float(solve_time_ms)

    def _extract_solution(self, solution: np.ndarray, index: QPIndex) -> Tuple[np.ndarray, np.ndarray]:
        x_traj = np.zeros((self.horizon_steps + 1, self.nx), dtype=float)
        u_traj = np.zeros((self.horizon_steps, self.nu), dtype=float)
        for k in range(self.horizon_steps + 1):
            for i in range(self.nx):
                x_traj[k, i] = float(solution[index.state_index(k, i)])
            x_traj[k, 3] = self._wrap_angle(float(x_traj[k, 3]))
        for k in range(self.horizon_steps):
            for i in range(self.nu):
                u_traj[k, i] = float(solution[index.control_index(k, i)])
        return x_traj, u_traj

    def _evaluate_lane_keeping_profile(
        self,
        x_traj: np.ndarray,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
    ) -> LaneKeepingProfile:
        lane_stage_samples: List[Dict[str, float] | None] = []
        state_xy: List[Tuple[float, float]] = []
        stage_count = min(int(x_traj.shape[0]), int(self.horizon_steps) + 1)
        for stage_index in range(stage_count):
            state_xy.append(
                (
                    float(x_traj[stage_index, 0]),
                    float(x_traj[stage_index, 1]),
                )
            )
            lane_stage_samples.append(
                self._get_lane_center_stage_sample(
                    lane_center_reference=lane_center_reference,
                    stage_index=int(stage_index),
                    query_x_m=float(x_traj[stage_index, 0]),
                    query_y_m=float(x_traj[stage_index, 1]),
                )
            )

        lane_center_weight = (
            float(self.lane_center_follow_weight)
            if bool(self.lane_center_follow_enabled)
            else 0.0
        )
        road_boundary_weight = (
            float(getattr(self, "road_boundary_weight", self.lane_keep_boundary_weight))
            if bool(getattr(self, "road_boundary_enabled", True))
            else 0.0
        )
        return evaluate_lane_keeping_profile(
            state_xy=state_xy,
            lane_references=lane_stage_samples,
            centering_weight=float(lane_center_weight),
            boundary_weight=float(road_boundary_weight),
            safe_region_alpha=float(self.lane_keep_safe_region_alpha),
            road_boundary_margin_m=float(getattr(self, "road_boundary_margin_m", 0.5)),
            default_lane_width_m=float(getattr(self, "lane_width_m", 4.0)),
        )

    def _evaluate_plan_cost_terms(
        self,
        x_traj: np.ndarray,
        u_traj: np.ndarray,
        x_ref_target: np.ndarray,
        object_snapshots: Sequence[Mapping[str, object]],
        current_acceleration_mps2: float,
        current_steering_rad: float,
        lane_center_reference: Sequence[Mapping[str, object]] | None,
    ) -> Dict[str, float]:
        """
        Evaluate per-term objective values for the most recent planned trajectory.

        These values are for runtime diagnostics/plotting and match the active
        cost terms used by this MPC implementation.
        """

        attractive_scale = float(self.safety_cost.w_safe)
        qx = float(self.comfort_cost.qx)
        qy = float(self.comfort_cost.qy)
        qv = float(self.comfort_cost.qv)
        qpsi = float(self.comfort_cost.qpsi)

        x_ref = float(x_ref_target[0])
        y_ref = float(x_ref_target[1])
        v_ref = float(x_ref_target[2])
        psi_ref = float(x_ref_target[3])

        cost_attractive_ref = 0.0
        for k in range(1, self.horizon_steps + 1):
            dx = float(x_traj[k, 0]) - x_ref
            dy = float(x_traj[k, 1]) - y_ref
            dv = float(x_traj[k, 2]) - v_ref
            dpsi = self._wrap_angle(float(x_traj[k, 3]) - psi_ref)
            cost_attractive_ref += qx * dx * dx + qy * dy * dy + qv * dv * dv + qpsi * dpsi * dpsi

        lane_keep_profile = self._evaluate_lane_keeping_profile(
            x_traj=x_traj,
            lane_center_reference=lane_center_reference,
        )
        self._last_lane_keeping_profile = lane_keep_profile
        cost_lane_center = 0.0
        cost_centerline_xy = 0.0
        cost_road_boundary = 0.0
        for metric in lane_keep_profile.stage_metrics:
            if int(metric.stage_index) <= 0:
                continue
            lane_sample = self._get_lane_center_stage_sample(
                lane_center_reference=lane_center_reference,
                stage_index=int(metric.stage_index),
                query_x_m=float(x_traj[int(metric.stage_index), 0]),
                query_y_m=float(x_traj[int(metric.stage_index), 1]),
            )
            lane_reference = normalize_lane_reference_sample(
                lane_sample,
                default_lane_width_m=float(getattr(self, "lane_width_m", 4.0)),
            )
            centerline_xy_weight = float(getattr(self, "lane_center_follow_xy_weight", 0.0))
            if (
                lane_reference is not None
                and bool(self.lane_center_follow_enabled)
                and float(centerline_xy_weight) > 0.0
            ):
                if bool(getattr(self, "lane_center_follow_xy_uses_frenet_decomposition", False)):
                    long_affine = signed_longitudinal_progress_affine_form(lane_reference)
                    longitudinal_error = long_affine.evaluate(
                        x_m=float(x_traj[int(metric.stage_index), 0]),
                        y_m=float(x_traj[int(metric.stage_index), 1]),
                    )
                    cost_centerline_xy += float(centerline_xy_weight) * float(longitudinal_error) * float(longitudinal_error)
                else:
                    dx_center = float(x_traj[int(metric.stage_index), 0]) - float(lane_reference.x_center_m)
                    dy_center = float(x_traj[int(metric.stage_index), 1]) - float(lane_reference.y_center_m)
                    cost_centerline_xy += float(centerline_xy_weight) * (
                        float(dx_center) * float(dx_center)
                        + float(dy_center) * float(dy_center)
                    )
            cost_lane_center += float(metric.centering_cost)
            cost_road_boundary += float(metric.boundary_cost)
            if bool(self.lane_center_follow_enabled) and float(self.lane_center_follow_weight) > 0.0:
                e_psi_lane = self._wrap_angle(
                    float(x_traj[int(metric.stage_index), 3]) - float(metric.lane_heading_rad)
                )
                cost_lane_center += (
                    float(self.lane_center_follow_weight)
                    * float(self.lane_center_follow_qpsi)
                    * float(e_psi_lane)
                    * float(e_psi_lane)
                )

        cost_attractive = attractive_scale * cost_attractive_ref
        cost_lane_center = float(cost_lane_center)
        cost_road_boundary = float(cost_road_boundary)
        j_ctrl = 0.0
        a_prev = float(current_acceleration_mps2)
        d_prev = float(current_steering_rad)
        inv_dt = 1.0 / max(1e-9, float(self.dt_s))
        qa = float(self.comfort_cost.qa)
        qd = float(self.comfort_cost.qdelta)
        for k in range(self.horizon_steps):
            a_k = float(u_traj[k, 0])
            d_k = float(u_traj[k, 1])
            da = (a_k - a_prev) * inv_dt
            dd = (d_k - d_prev) * inv_dt
            j_ctrl += qa * da * da + qd * dd * dd
            a_prev = a_k
            d_prev = d_k
        cost_control = float(self.comfort_cost.w_comf) * j_ctrl

        cost_repulsive_safe = 0.0
        cost_repulsive_collision = 0.0
        cost_repulsive_logbarrier = 0.0
        if bool(self.repulsive_cost.enabled) and len(object_snapshots) > 0:
            for k in range(1, self.horizon_steps + 1):
                stage_idx = k - 1
                ego_state = [
                    float(x_traj[k, 0]),
                    float(x_traj[k, 1]),
                    float(x_traj[k, 2]),
                    float(self._wrap_angle(float(x_traj[k, 3]))),
                ]

                for object_snapshot in object_snapshots:
                    obj_state = self._get_object_state_at_stage(
                        object_snapshot=object_snapshot,
                        stage_index=stage_idx,
                        dt_s=float(self.dt_s),
                    )
                    repulsive_weight = float(object_snapshot.get("repulsive_class_weight", 1.0))
                    if repulsive_weight <= 0.0:
                        continue

                    obstacle_length_m = float(object_snapshot.get("length_m", 4.5))
                    obstacle_width_m = float(object_snapshot.get("width_m", 2.0))
                    obstacle_cost_safe, obstacle_cost_collision = self._superellipsoid_obstacle_cost_components(
                        ego_state=ego_state,
                        obstacle_state=obj_state,
                        obstacle_length_m=obstacle_length_m,
                        obstacle_width_m=obstacle_width_m,
                    )
                    if bool(self.repulsive_cost.log_barrier_replace_exponential):
                        obstacle_cost_collision = 0.0
                    cost_repulsive_safe += float(repulsive_weight) * float(obstacle_cost_safe)
                    cost_repulsive_collision += float(repulsive_weight) * float(obstacle_cost_collision)
                    if bool(self.repulsive_cost.log_barrier_enabled):
                        obstacle_geometry = self._superellipsoid_zone_geometry(
                            ego_state=ego_state,
                            obstacle_state=obj_state,
                            obstacle_length_m=obstacle_length_m,
                            obstacle_width_m=obstacle_width_m,
                        )
                        cost_repulsive_logbarrier += float(repulsive_weight) * float(
                            self._log_barrier_obstacle_cost_component(float(obstacle_geometry["rc"]))
                        )
        cost_repulsive = float(cost_repulsive_safe + cost_repulsive_collision + cost_repulsive_logbarrier)

        cost_velocity_slack = 0.0
        if bool(getattr(self, "speed_soft_constraint_enabled", False)):
            # Diagnostic-only approximation: measures overshoot beyond the
            # global max_velocity_mps rather than re-deriving each stage's
            # exact dynamic cap (curve/traffic-light/lead-obstacle caps),
            # which would require threading extra parameters into this
            # evaluation-only function. Sufficient for CSV visibility into
            # whether/how much the soft constraint is engaging.
            velocity_weight = float(self.speed_soft_constraint_weight)
            max_velocity_mps = float(self.constraints.max_velocity_mps)
            for k in range(1, self.horizon_steps + 1):
                overshoot_mps = max(0.0, float(x_traj[k, 2]) - max_velocity_mps)
                cost_velocity_slack += velocity_weight * overshoot_mps * overshoot_mps

        return {
            "Cost_ref": float(cost_attractive),
            "Cost_LaneCenter": float(cost_lane_center),
            "Cost_CenterlineXY": float(cost_centerline_xy),
            "Cost_RoadBoundary": float(cost_road_boundary),
            "Cost_LaneBoundary": float(cost_road_boundary),
            "Cost_Lane": float(cost_lane_center + cost_centerline_xy + cost_road_boundary),
            "Cost_Repulsive_Safe": float(cost_repulsive_safe),
            "Cost_Repulsive_Collision": float(cost_repulsive_collision),
            "Cost_Repulsive_LogBarrier": float(cost_repulsive_logbarrier),
            "Cost_Repulsive": float(cost_repulsive),
            "Cost_Control": float(cost_control),
            "Cost_VelocitySlack": float(cost_velocity_slack),
        }

    def plan_trajectory(
        self,
        current_state: Sequence[float],
        destination_state: Sequence[float],
        object_snapshots: Sequence[Mapping[str, object]],
        current_acceleration_mps2: float,
        current_steering_rad: float,
        lane_center_waypoints: Sequence[Mapping[str, object]] | None = None,
        lane_center_reference_samples: Sequence[Mapping[str, object]] | None = None,
        stop_goal_active: bool = False,
        road_envelope_payload_world: Mapping[str, object] | None = None,
    ) -> List[List[float]]:
        """
        Intent:
            Solve one MPC optimization and return future states [x,y,v,psi].
        """

        if len(current_state) != 4:
            raise ValueError("current_state must be [x, y, v, psi].")

        origin_x_m = float(current_state[0])
        origin_y_m = float(current_state[1])
        x0_world = np.array(
            [
                float(origin_x_m),
                float(origin_y_m),
                self._clamp(float(current_state[2]), self.constraints.min_velocity_mps, self.constraints.max_velocity_mps),
                self._wrap_angle(float(current_state[3])),
            ],
            dtype=float,
        )
        x0 = np.asarray(x0_world, dtype=float).copy()
        x0[0] = 0.0
        x0[1] = 0.0
        planning_current_acceleration_mps2 = float(current_acceleration_mps2)
        destination_world = self._normalize_destination_state(destination_state)
        destination = np.asarray(destination_world, dtype=float).copy()
        destination[0] -= float(origin_x_m)
        destination[1] -= float(origin_y_m)
        object_snapshots = self._translate_object_snapshots(
            object_snapshots=object_snapshots,
            origin_x_m=float(origin_x_m),
            origin_y_m=float(origin_y_m),
        )
        destination_lane_id = (
            int(destination_state[4])
            if len(destination_state) >= 5
            else None
        )
        active_speed_upper_bound_mps = self._compute_active_speed_upper_bound_mps(
            current_state=x0,
            destination_state=destination,
            force_stop_goal=bool(stop_goal_active),
        )
        self._last_active_max_velocity_mps = float(active_speed_upper_bound_mps)

        # Detect a stop goal while destination[2] is still the original value
        # (0.0 for stop goals set by the behavior planner).
        _is_stop_goal = bool(stop_goal_active) or (
            abs(float(destination[2]))
            <= float(self.final_stop_speed_cap_activation_threshold_mps)
        )
        _resume_release_speed_threshold_mps = max(
            float(self.final_stop_speed_cap_activation_threshold_mps),
            1.0,
        )
        _transitioning_from_stationary_hold = (
            not bool(_is_stop_goal)
            and float(x0[2]) <= float(_resume_release_speed_threshold_mps)
            and float(destination[2]) > float(_resume_release_speed_threshold_mps)
            and float(planning_current_acceleration_mps2) < -0.05
        )
        if _transitioning_from_stationary_hold:
            planning_current_acceleration_mps2 = 0.0

        if _is_stop_goal:
            # Replace the static v_ref=0 with a dynamic kinematic profile:
            #   v_ref = active_speed_upper_bound_mps
            #         = sqrt(2 * a_brake * max(dist - stop_buffer, 0))
            # This is the maximum safe speed at the current distance; far from
            # the stop point it equals v_max, tapering smoothly to 0 only
            # within the buffer zone.  The rollout therefore keeps moving until
            # close to the target instead of freezing at v=0 in the middle of
            # the trajectory — eliminating the degenerate QP that caused the
            # circular-arc artefact.
            destination[2] = float(active_speed_upper_bound_mps)
        else:
            destination[2] = self._clamp(
                float(destination[2]),
                self.constraints.min_velocity_mps,
                float(active_speed_upper_bound_mps),
            )

        reachable_speed_floor_profile_mps = self._minimum_reachable_speed_profile_mps(
            current_speed_mps=float(x0[2]),
            current_acceleration_mps2=float(planning_current_acceleration_mps2),
        )

        destination[3] = self._wrap_angle(float(destination[3]))

        lane_center_reference: List[Dict[str, float]] = []
        should_use_lane_center_reference = (
            (bool(self.lane_center_follow_enabled) and float(self.lane_center_follow_weight) > 0.0)
            or (bool(getattr(self, "road_boundary_enabled", True)) and float(getattr(self, "road_boundary_weight", 0.0)) > 0.0)
            or bool(self.reference_prefer_lane_center_path)
        )
        if should_use_lane_center_reference:
            world_lane_center_reference = self._normalize_lane_center_reference_samples(
                lane_center_reference_samples=lane_center_reference_samples,
            )
            if len(world_lane_center_reference) == 0:
                world_destination = np.asarray(destination, dtype=float).copy()
                world_destination[0] += float(origin_x_m)
                world_destination[1] += float(origin_y_m)
                world_lane_center_reference = self._build_lane_center_reference(
                    current_state=x0_world,
                    destination_state=world_destination,
                    lane_center_waypoints=lane_center_waypoints,
                    destination_lane_id=destination_lane_id,
                )
            lane_center_reference = self._translate_lane_reference(
                lane_center_reference=world_lane_center_reference,
                origin_x_m=float(origin_x_m),
                origin_y_m=float(origin_y_m),
            )

        road_envelope_blocks: Mapping[str, object] | None = None
        if isinstance(road_envelope_payload_world, Mapping):
            world_blocks = list(road_envelope_payload_world.get("blocks", []) or [])
            if world_blocks:
                translated_blocks = [
                    RoadEnvelopeBlock(
                        x_center_m=float(block.x_center_m) - float(origin_x_m),
                        y_center_m=float(block.y_center_m) - float(origin_y_m),
                        heading_rad=float(block.heading_rad),
                        half_length_m=float(block.half_length_m),
                        half_width_m=float(block.half_width_m),
                        shape_exponent=float(block.shape_exponent),
                    )
                    for block in world_blocks
                ]
                road_envelope_blocks = {
                    "blocks": translated_blocks,
                    "epsilon0": float(road_envelope_payload_world.get("epsilon0", 0.0)),
                    "rho": float(
                        road_envelope_payload_world.get(
                            "rho", getattr(self, "road_envelope_rho", -8.0)
                        )
                    ),
                    "max_slack_m": float(
                        road_envelope_payload_world.get(
                            "max_slack_m",
                            getattr(self, "road_envelope_max_slack_m", 0.10),
                        )
                    ),
                }

        # During stop-goal mode never reuse the previous QP solution as seed.
        # Previous plans produced under the old v_ref=0 regime may have been
        # circular arcs; reusing them would re-seed the bad linearisation point
        # and perpetuate the instability even after the velocity reference fix.
        #
        # On the first non-stop-goal call after a stop goal (the stop→resume
        # transition), also discard the seed.  The previous solution is a v=0
        # braking trajectory; using it as the linearisation reference gives a
        # degenerate QP where every A/B matrix is evaluated at v=0, making
        # steering effects vanish and cost gradients for acceleration extremely
        # weak.  Starting from a clean rollout instead lets the reference
        # propagate acceleration properly and allows the QP to plan a
        # physically meaningful re-acceleration trajectory.
        _transitioning_from_stop = bool(self._last_was_stop_goal) and not bool(_is_stop_goal)
        if _is_stop_goal or _transitioning_from_stop or _transitioning_from_stationary_hold:
            shifted_seed = None
            if _transitioning_from_stop or _transitioning_from_stationary_hold:
                # Also drop any stored solution so _build_shifted_previous_solution_seed
                # cannot return it on a later call before the ego has moved.
                self._previous_x_solution = None
                self._previous_u_solution = None
        else:
            shifted_seed = self._build_shifted_previous_solution_seed(x0=x0_world)
            if shifted_seed is not None:
                shifted_seed_x = np.asarray(shifted_seed[0], dtype=float).copy()
                shifted_seed_x[:, 0] -= float(origin_x_m)
                shifted_seed_x[:, 1] -= float(origin_y_m)
                shifted_seed = (
                    shifted_seed_x,
                    np.asarray(shifted_seed[1], dtype=float),
                )
        x_ref_rollout, u_ref_rollout = self._reference_rollout(
            x0=x0,
            x_ref_target=destination,
            lane_center_reference=lane_center_reference,
            object_snapshots=object_snapshots,
            speed_upper_bound_mps=float(active_speed_upper_bound_mps),
            reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
            seed_state_traj=shifted_seed[0] if shifted_seed is not None else None,
            seed_control_traj=shifted_seed[1] if shifted_seed is not None else None,
        )
        speed_tracking_reference_mps = self._speed_tracking_reference(
            x0=x0,
            x_ref_target=destination,
            linearization_rollout=x_ref_rollout,
            object_snapshots=object_snapshots,
            current_acceleration_mps2=float(planning_current_acceleration_mps2),
            speed_upper_bound_mps=float(active_speed_upper_bound_mps),
            reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
        )

        def _run_sequential_qp(
            initial_x_ref_rollout: np.ndarray,
            initial_u_ref_rollout: np.ndarray,
            fixed_speed_tracking_reference_mps: Sequence[float],
        ) -> tuple[np.ndarray | None, np.ndarray | None, str, float, np.ndarray, np.ndarray]:
            solve_time_total_ms = 0.0
            current_x_rollout = np.asarray(initial_x_ref_rollout, dtype=float)
            current_u_rollout = np.asarray(initial_u_ref_rollout, dtype=float)
            best_x: np.ndarray | None = None
            best_u: np.ndarray | None = None
            status_text = "not_solved"

            for iteration_idx in range(int(self.reference_sequential_iterations)):
                P, q, A, l, u, index = self._build_qp(
                    x0=x0,
                    x_ref_target=destination,
                    object_snapshots=object_snapshots,
                    current_acceleration_mps2=float(planning_current_acceleration_mps2),
                    current_steering_rad=float(current_steering_rad),
                    x_ref_rollout=current_x_rollout,
                    u_ref_rollout=current_u_rollout,
                    lane_center_reference=lane_center_reference,
                    speed_upper_bound_mps=float(active_speed_upper_bound_mps),
                    reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
                    road_envelope_blocks=road_envelope_blocks,
                    speed_tracking_reference_mps=(
                        fixed_speed_tracking_reference_mps
                    ),
                )
                solution, status, solve_time_ms = self._solve_qp(P=P, q=q, A=A, l=l, u=u)
                solve_time_total_ms += float(solve_time_ms)

                if solution is None:
                    if best_x is None:
                        status_text = str(status)
                    break

                status_text = str(status)
                best_x, best_u = self._extract_solution(solution=solution, index=index)
                if int(iteration_idx) + 1 >= int(self.reference_sequential_iterations):
                    break

                current_x_rollout = np.asarray(best_x, dtype=float)
                current_u_rollout = np.asarray(best_u, dtype=float)

            return best_x, best_u, status_text, float(solve_time_total_ms), current_x_rollout, current_u_rollout

        best_x_solution, best_u_solution, best_status, total_solve_time_ms, current_x_ref_rollout, current_u_ref_rollout = _run_sequential_qp(
            initial_x_ref_rollout=np.asarray(x_ref_rollout, dtype=float),
            initial_u_ref_rollout=np.asarray(u_ref_rollout, dtype=float),
            fixed_speed_tracking_reference_mps=np.asarray(
                speed_tracking_reference_mps,
                dtype=float,
            ),
        )

        solved_initially = best_x_solution is not None and best_u_solution is not None
        if self._record_solver_failure_state(solved=bool(solved_initially)):
            self._clear_all_solution_memory()
            if bool(getattr(self, "log_solution_memory_resets", False)):
                print(
                    "[MPC] Solver failed "
                    f"{int(self.reference_consecutive_solver_failure_reset_threshold)} consecutive replans; "
                    "clearing stored solution and retrying with a fresh rollout."
                )
            clean_x_ref_rollout, clean_u_ref_rollout = self._reference_rollout(
                x0=x0,
                x_ref_target=destination,
                lane_center_reference=lane_center_reference,
                object_snapshots=object_snapshots,
                speed_upper_bound_mps=float(active_speed_upper_bound_mps),
                reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
                seed_state_traj=None,
                seed_control_traj=None,
            )
            clean_speed_tracking_reference_mps = (
                self._speed_tracking_reference(
                    x0=x0,
                    x_ref_target=destination,
                    linearization_rollout=clean_x_ref_rollout,
                    object_snapshots=object_snapshots,
                    current_acceleration_mps2=float(
                        planning_current_acceleration_mps2
                    ),
                    speed_upper_bound_mps=float(active_speed_upper_bound_mps),
                    reachable_speed_floor_profile_mps=(
                        reachable_speed_floor_profile_mps
                    ),
                )
            )
            (
                best_x_solution,
                best_u_solution,
                best_status,
                clean_solve_time_ms,
                current_x_ref_rollout,
                current_u_ref_rollout,
            ) = _run_sequential_qp(
                initial_x_ref_rollout=np.asarray(clean_x_ref_rollout, dtype=float),
                initial_u_ref_rollout=np.asarray(clean_u_ref_rollout, dtype=float),
                fixed_speed_tracking_reference_mps=np.asarray(
                    clean_speed_tracking_reference_mps,
                    dtype=float,
                ),
            )
            total_solve_time_ms += float(clean_solve_time_ms)
            self._record_clean_restart_result(
                solved=bool(best_x_solution is not None and best_u_solution is not None)
            )

        self._last_status = str(best_status)
        self._last_solve_time_ms = float(total_solve_time_ms)

        if best_x_solution is None or best_u_solution is None:
            x_solution, u_solution = self._fail_safe_fallback_trajectory(
                x0=x0,
                rollout_x=current_x_ref_rollout,
                rollout_u=current_u_ref_rollout,
                current_acceleration_mps2=float(planning_current_acceleration_mps2),
            )
        else:
            x_solution = np.asarray(best_x_solution, dtype=float)
            u_solution = np.asarray(best_u_solution, dtype=float)
        x_solution = np.asarray(x_solution, dtype=float)
        u_solution = np.asarray(u_solution, dtype=float)
        speed_soft_active_for_clamp = bool(getattr(self, "speed_soft_constraint_enabled", False))
        for k in range(1, self.horizon_steps + 1):
            stage_speed_upper_bound_mps = self._future_speed_upper_bound_mps(
                active_speed_upper_bound_mps=float(active_speed_upper_bound_mps),
                future_state_index=int(k),
                reachable_speed_floor_profile_mps=reachable_speed_floor_profile_mps,
            )
            # When the velocity upper bound was solved as a soft (slack)
            # constraint in _build_qp, a legitimately-solved overshoot must
            # not be clamped straight back down to the nominal cap here --
            # that would silently erase the whole point of softening it.
            # Allow up to the same speed_soft_max_slack_mps margin the QP
            # itself was allowed to use.
            clamp_upper_mps = float(stage_speed_upper_bound_mps)
            if speed_soft_active_for_clamp:
                clamp_upper_mps += float(self.speed_soft_max_slack_mps)
            x_solution[k, 2] = self._clamp(
                float(x_solution[k, 2]),
                float(self.constraints.min_velocity_mps),
                clamp_upper_mps,
            )
            x_solution[k, 3] = self._wrap_angle(float(x_solution[k, 3]))

        self._last_cost_terms = self._evaluate_plan_cost_terms(
            x_traj=x_solution,
            u_traj=u_solution,
            x_ref_target=destination,
            object_snapshots=object_snapshots,
            current_acceleration_mps2=float(current_acceleration_mps2),
            current_steering_rad=float(current_steering_rad),
            lane_center_reference=lane_center_reference,
        )

        world_x_solution = np.asarray(x_solution, dtype=float).copy()
        world_x_solution[:, 0] += float(origin_x_m)
        world_x_solution[:, 1] += float(origin_y_m)
        self._last_x_solution = np.asarray(world_x_solution, dtype=float)
        self._last_u_solution = np.asarray(u_solution, dtype=float)

        if best_x_solution is not None and best_u_solution is not None:
            self._previous_x_solution = np.asarray(world_x_solution, dtype=float)
            self._previous_u_solution = np.asarray(u_solution, dtype=float)

        # Record whether this call was a stop goal so the next call can detect
        # the stop→resume transition and avoid reusing the v=0 braking seed.
        self._last_was_stop_goal = bool(_is_stop_goal)

        output: List[List[float]] = []
        for k in range(1, self.horizon_steps + 1):
            output.append(
                [
                    float(world_x_solution[k, 0]),
                    float(world_x_solution[k, 1]),
                    float(x_solution[k, 2]),
                    float(self._wrap_angle(float(x_solution[k, 3]))),
                ]
            )
        return output

    def probe_trajectory_feasibility(
        self,
        *,
        current_state: Sequence[float],
        destination_state: Sequence[float],
        object_snapshots: Sequence[Mapping[str, object]],
        current_acceleration_mps2: float,
        current_steering_rad: float,
        lane_center_reference_samples: Sequence[Mapping[str, object]] | None,
        stop_goal_active: bool = False,
        road_envelope_payload_world: Mapping[str, object] | None = None,
    ) -> Dict[str, object]:
        """Solve a candidate without changing MPC warm-start/runtime state."""

        mutable_fields = (
            "_last_status",
            "_last_solve_time_ms",
            "_last_active_max_velocity_mps",
            "_last_cost_terms",
            "_last_lane_keeping_profile",
            "_last_x_solution",
            "_last_u_solution",
            "_previous_x_solution",
            "_previous_u_solution",
            "_consecutive_solver_failure_count",
            "_last_failure_reset_triggered",
            "_last_was_stop_goal",
            "_solver_failure_log_event_count",
            "_solver_failure_emergency_logged",
        )
        snapshot = {
            name: copy.deepcopy(getattr(self, name))
            for name in mutable_fields
            if hasattr(self, name)
        }
        result: Dict[str, object] = {
            "solved": False,
            "status": "probe_not_run",
            "solve_time_ms": 0.0,
            "cost_terms": {},
            "dynamic_cost": 0.0,
        }
        try:
            self.plan_trajectory(
                current_state=current_state,
                destination_state=destination_state,
                object_snapshots=object_snapshots,
                current_acceleration_mps2=float(current_acceleration_mps2),
                current_steering_rad=float(current_steering_rad),
                lane_center_reference_samples=lane_center_reference_samples,
                stop_goal_active=bool(stop_goal_active),
                road_envelope_payload_world=road_envelope_payload_world,
            )
            status = str(getattr(self, "_last_status", "")).strip().lower()
            cost_terms = dict(getattr(self, "_last_cost_terms", {}) or {})
            dynamic_cost = sum(
                max(0.0, float(value))
                for value in cost_terms.values()
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            )
            result = {
                "solved": status in {"solved", "solved inaccurate"},
                "status": str(status or "unknown"),
                "solve_time_ms": float(
                    getattr(self, "_last_solve_time_ms", 0.0) or 0.0
                ),
                "cost_terms": cost_terms,
                # Keep probe cost numerically subordinate to behavior safety
                # and route costs while still breaking ties by trackability.
                "dynamic_cost": 0.001 * float(dynamic_cost),
            }
        except Exception as exc:
            result["status"] = "probe_exception:" + str(exc)
        finally:
            for name, value in snapshot.items():
                setattr(self, name, value)
        return result
