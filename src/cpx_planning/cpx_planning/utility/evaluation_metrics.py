"""Planning-module evaluation metrics.

The recorder is intentionally observational: it consumes ego state, obstacle
snapshots, collision events, and MPC runtime statuses without affecting the
planner decisions.
"""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence, Tuple


_EPS = 1.0e-9


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        numeric_value = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(numeric_value):
        return float(default)
    return float(numeric_value)


def _state_xyvpsi(state: Mapping[str, object] | Sequence[object]) -> Tuple[float, float, float, float]:
    if isinstance(state, Mapping):
        return (
            _finite_float(state.get("x", 0.0)),
            _finite_float(state.get("y", 0.0)),
            max(0.0, _finite_float(state.get("v", 0.0))),
            _finite_float(state.get("psi", 0.0)),
        )
    values = list(state or [])
    return (
        _finite_float(values[0] if len(values) > 0 else 0.0),
        _finite_float(values[1] if len(values) > 1 else 0.0),
        max(0.0, _finite_float(values[2] if len(values) > 2 else 0.0)),
        _finite_float(values[3] if len(values) > 3 else 0.0),
    )


def _snapshot_id(snapshot: Mapping[str, object], fallback_index: int) -> str:
    for key in ("vehicle_id", "id", "message_id", "actor_id"):
        raw_value = str(snapshot.get(key, "")).strip()
        if raw_value:
            return raw_value
    return f"obstacle_{int(fallback_index)}"


def compute_pairwise_ttc_drac(
    ego_state: Mapping[str, object] | Sequence[object],
    obstacle_snapshot: Mapping[str, object],
    *,
    ego_length_m: float = 4.5,
    obstacle_length_m: float | None = None,
    lateral_conflict_width_m: float = 2.5,
    min_ego_speed_for_ttc_mps: float = 0.5,
) -> Tuple[float, float]:
    """Return longitudinal TTC and DRAC for one obstacle.

    TTC is only finite when the obstacle is in front of the ego, laterally
    close enough to conflict, and the ego is closing on it. DRAC is the
    constant deceleration needed to avoid reaching the obstacle gap.
    """
    details = compute_pairwise_ttc_drac_debug(
        ego_state=ego_state,
        obstacle_snapshot=obstacle_snapshot,
        ego_length_m=float(ego_length_m),
        obstacle_length_m=obstacle_length_m,
        lateral_conflict_width_m=float(lateral_conflict_width_m),
        min_ego_speed_for_ttc_mps=float(min_ego_speed_for_ttc_mps),
    )
    return (
        float(details.get("ttc_s", float("inf"))),
        float(details.get("drac_mps2", 0.0)),
    )


def compute_pairwise_ttc_drac_debug(
    ego_state: Mapping[str, object] | Sequence[object],
    obstacle_snapshot: Mapping[str, object],
    *,
    ego_length_m: float = 4.5,
    obstacle_length_m: float | None = None,
    lateral_conflict_width_m: float = 2.5,
    min_ego_speed_for_ttc_mps: float = 0.5,
) -> Dict[str, object]:
    """Return TTC/DRAC plus the geometry that produced the result."""

    ego_x, ego_y, ego_v, ego_psi = _state_xyvpsi(ego_state)
    obs_x = _finite_float(obstacle_snapshot.get("x", 0.0))
    obs_y = _finite_float(obstacle_snapshot.get("y", 0.0))
    obs_v = max(0.0, _finite_float(obstacle_snapshot.get("v", 0.0)))
    obs_psi = _finite_float(obstacle_snapshot.get("psi", ego_psi))
    obs_length = (
        max(0.0, _finite_float(obstacle_length_m))
        if obstacle_length_m is not None
        else max(0.0, _finite_float(obstacle_snapshot.get("length_m", 4.5), 4.5))
    )

    dx_m = float(obs_x) - float(ego_x)
    dy_m = float(obs_y) - float(ego_y)
    ego_cos = math.cos(float(ego_psi))
    ego_sin = math.sin(float(ego_psi))
    longitudinal_gap_m = float(ego_cos * dx_m + ego_sin * dy_m)
    lateral_gap_m = float(-ego_sin * dx_m + ego_cos * dy_m)
    obs_velocity_along_ego_mps = float(obs_v) * math.cos(float(obs_psi) - float(ego_psi))
    closing_speed_mps = float(ego_v) - float(obs_velocity_along_ego_mps)
    bumper_gap_raw_m = (
        float(longitudinal_gap_m)
        - 0.5 * max(0.0, float(ego_length_m))
        - 0.5 * max(0.0, float(obs_length))
    )
    bumper_gap_m = max(0.0, float(bumper_gap_raw_m))

    details: Dict[str, object] = {
        "ttc_s": float("inf"),
        "drac_mps2": 0.0,
        "longitudinal_gap_m": float(longitudinal_gap_m),
        "lateral_gap_m": float(lateral_gap_m),
        "bumper_gap_m": float(bumper_gap_m),
        "bumper_gap_raw_m": float(bumper_gap_raw_m),
        "closing_speed_mps": float(closing_speed_mps),
        "obstacle_x": float(obs_x),
        "obstacle_y": float(obs_y),
        "obstacle_v_mps": float(obs_v),
        "obstacle_psi_rad": float(obs_psi),
        "reason": "",
    }

    if float(ego_v) < max(0.0, float(min_ego_speed_for_ttc_mps)):
        details["reason"] = "ego_speed_below_ttc_threshold"
        return details

    if longitudinal_gap_m <= 0.0 or abs(lateral_gap_m) > max(0.0, float(lateral_conflict_width_m)):
        details["reason"] = (
            "behind_or_lateral_clear"
            if float(longitudinal_gap_m) <= 0.0
            else "lateral_clear"
        )
        return details

    if closing_speed_mps <= 0.0:
        details["reason"] = "not_closing"
        return details
    if bumper_gap_m <= _EPS:
        details["ttc_s"] = 0.0
        details["drac_mps2"] = float("inf")
        details["reason"] = "bumper_overlap"
        return details
    details["ttc_s"] = float(bumper_gap_m / max(_EPS, closing_speed_mps))
    details["drac_mps2"] = float((closing_speed_mps * closing_speed_mps) / max(_EPS, 2.0 * bumper_gap_m))
    details["reason"] = "closing_conflict"
    return details


@dataclass
class EvaluationMetricsRecorder:
    """Accumulate safety, efficiency, and feasibility metrics for one run."""

    ego_length_m: float = 4.5
    collision_rate_distance_epsilon_km: float = 1.0e-6
    lateral_conflict_width_m: float = 2.5
    min_ego_speed_for_ttc_mps: float = 0.5
    pet_conflict_radius_m: float = 3.0
    pet_bin_size_m: float = 3.0
    samples: list[Dict[str, object]] = field(default_factory=list)
    planned_trajectory_samples: list[Dict[str, object]] = field(default_factory=list)
    collision_count: int = 0
    distance_traveled_m: float = 0.0
    mpc_plan_attempts: int = 0
    mpc_plan_successes: int = 0
    min_ttc_s: float = float("inf")
    max_drac_mps2: float = 0.0
    min_pet_s: float = float("inf")
    road_boundary_breach_count: int = 0
    road_boundary_sample_count: int = 0
    _last_ego_xy: Tuple[float, float] | None = None
    _last_ego_bin: Tuple[int, int] | None = None
    _last_ego_bin_time_s: float | None = None
    _last_obstacle_bin_time_by_id: Dict[str, Tuple[Tuple[int, int], float]] = field(default_factory=dict)
    _collision_event_ids: set[str] = field(default_factory=set)
    collision_events: list = field(default_factory=list)

    def record_collision(
        self,
        event_id: object | None = None,
        *,
        sim_time_s: float | None = None,
        ego_x: float | None = None,
        ego_y: float | None = None,
        ego_speed_mps: float | None = None,
        other_actor_type: str | None = None,
        impulse_magnitude: float | None = None,
    ) -> None:
        """Record one collision sensor event, de-duplicating stable ids."""
        normalized_event_id = str(event_id or "").strip()
        if normalized_event_id:
            if normalized_event_id in self._collision_event_ids:
                return
            self._collision_event_ids.add(normalized_event_id)
        self.collision_count += 1
        self.collision_events.append({
            "collision_index": int(self.collision_count - 1),
            "event_id": str(normalized_event_id),
            "sim_time_s": float(sim_time_s) if sim_time_s is not None else None,
            "ego_x": float(ego_x) if ego_x is not None else None,
            "ego_y": float(ego_y) if ego_y is not None else None,
            "ego_speed_mps": float(ego_speed_mps) if ego_speed_mps is not None else None,
            "other_actor_type": str(other_actor_type or "unknown"),
            "impulse_magnitude": float(impulse_magnitude) if impulse_magnitude is not None else None,
        })

    def record_mpc_status(self, runtime_status: Mapping[str, object] | None) -> None:
        self.mpc_plan_attempts += 1
        status = str((runtime_status or {}).get("solver_status", "")).strip().lower()
        if "solved" in status:
            self.mpc_plan_successes += 1

    def record_mpc_extras(
        self,
        *,
        lateral_offset_m: float | None,
        heading_error_rad: float | None,
        cost_terms: Mapping[str, float] | None,
    ) -> None:
        """Patch the most-recently-created sample with MPC diagnostic fields.

        Called once per MPC replan, immediately after record_mpc_status(), so
        the latest sample already exists in self.samples.
        """
        if not self.samples:
            return
        sample = self.samples[-1]
        sample["lateral_offset_m"] = (
            None if lateral_offset_m is None or not math.isfinite(float(lateral_offset_m))
            else float(lateral_offset_m)
        )
        sample["heading_error_rad"] = (
            None if heading_error_rad is None or not math.isfinite(float(heading_error_rad))
            else float(heading_error_rad)
        )
        terms = dict(cost_terms or {})
        for key in ("Cost_ref", "Cost_LaneCenter", "Cost_RoadBoundary", "Cost_Repulsive", "Cost_Control"):
            raw = terms.get(key)
            sample[key] = None if raw is None or not math.isfinite(float(raw)) else float(raw)

    def record_road_boundary(
        self,
        *,
        sample_valid: bool,
        lateral_offset_m: float | None = None,
        lane_width_m: float | None = None,
        ego_half_width_m: float | None = None,
        clearance_m: float | None = None,
        breach: bool = False,
    ) -> None:
        """Patch the latest sample and accumulate physical lane-envelope breaches."""

        if not self.samples:
            return
        sample = self.samples[-1]
        sample["road_boundary_sample_valid"] = bool(sample_valid)
        if not bool(sample_valid):
            return
        self.road_boundary_sample_count += 1
        if bool(breach):
            self.road_boundary_breach_count += 1
        sample.update(
            {
                "road_boundary_lateral_offset_m": lateral_offset_m,
                "road_boundary_lane_width_m": lane_width_m,
                "road_boundary_ego_half_width_m": ego_half_width_m,
                "road_boundary_clearance_m": clearance_m,
                "road_boundary_breach": bool(breach),
                "road_boundary_breach_count": int(
                    self.road_boundary_breach_count
                ),
                "road_boundary_sample_count": int(
                    self.road_boundary_sample_count
                ),
                "road_boundary_breach_rate": (
                    float(self.road_boundary_breach_count)
                    / float(self.road_boundary_sample_count)
                ),
            }
        )

    def record_planned_trajectory(
        self,
        *,
        sim_time_s: float,
        replan_index: int,
        trajectory: Sequence[Sequence[float]],
    ) -> None:
        """Append one MPC-planned horizon to the trajectory log.

        Each entry in ``trajectory`` is [x, y, v, psi] for one future step.
        """
        for step_k, point in enumerate(list(trajectory or [])):
            pt = list(point or [])
            self.planned_trajectory_samples.append({
                "replan_index": int(replan_index),
                "sim_time_s": float(sim_time_s),
                "step_k": int(step_k + 1),
                "x": float(pt[0]) if len(pt) > 0 else None,
                "y": float(pt[1]) if len(pt) > 1 else None,
                "v": float(pt[2]) if len(pt) > 2 else None,
                "psi": float(pt[3]) if len(pt) > 3 else None,
            })

    def update(
        self,
        *,
        ego_state: Mapping[str, object] | Sequence[object],
        obstacle_snapshots: Sequence[Mapping[str, object]] | None,
        sim_time_s: float,
        behavior_decision: str = "",
        fsm_state: str = "",
        blocking_obstacle_id: str = "",
        decision_reason: str = "",
        collision_count: int = 0,
        last_collision_actor_type: str = "",
        cp_provider_source: str = "",
        native_opencda_available: bool = False,
        native_opencda_required: bool = False,
        cp_obstacle_count: int | None = None,
    ) -> None:
        ego_x, ego_y, ego_v, ego_psi = _state_xyvpsi(ego_state)
        if self._last_ego_xy is not None:
            self.distance_traveled_m += math.hypot(
                float(ego_x) - float(self._last_ego_xy[0]),
                float(ego_y) - float(self._last_ego_xy[1]),
            )
        self._last_ego_xy = (float(ego_x), float(ego_y))

        nearest_ttc_s = float("inf")
        max_tick_drac_mps2 = 0.0
        nearest_ttc_debug: Dict[str, object] = {}
        for index, snapshot in enumerate(list(obstacle_snapshots or [])):
            if not isinstance(snapshot, Mapping):
                continue
            obstacle_id = _snapshot_id(snapshot, index)
            ttc_debug = compute_pairwise_ttc_drac_debug(
                ego_state={"x": ego_x, "y": ego_y, "v": ego_v, "psi": ego_psi},
                obstacle_snapshot=snapshot,
                ego_length_m=float(self.ego_length_m),
                lateral_conflict_width_m=float(self.lateral_conflict_width_m),
                min_ego_speed_for_ttc_mps=float(self.min_ego_speed_for_ttc_mps),
            )
            ttc_debug["obstacle_id"] = str(obstacle_id)
            ttc_s = float(ttc_debug.get("ttc_s", float("inf")))
            drac_mps2 = float(ttc_debug.get("drac_mps2", 0.0))
            if float(ttc_s) < float(nearest_ttc_s):
                nearest_ttc_debug = dict(ttc_debug)
            nearest_ttc_s = min(float(nearest_ttc_s), float(ttc_s))
            max_tick_drac_mps2 = max(float(max_tick_drac_mps2), float(drac_mps2))
            self._update_pet(
                obstacle_id=str(obstacle_id),
                obstacle_xy=(
                    _finite_float(snapshot.get("x", 0.0)),
                    _finite_float(snapshot.get("y", 0.0)),
                ),
                ego_xy=(float(ego_x), float(ego_y)),
                sim_time_s=float(sim_time_s),
            )

        self.min_ttc_s = min(float(self.min_ttc_s), float(nearest_ttc_s))
        self.max_drac_mps2 = max(float(self.max_drac_mps2), float(max_tick_drac_mps2))
        self.samples.append(
            {
                "sample_index": int(len(self.samples)),
                "sim_time_s": float(sim_time_s),
                "ego_x": float(ego_x),
                "ego_y": float(ego_y),
                "ego_speed_mps": float(ego_v),
                "obstacle_count": int(len(list(obstacle_snapshots or []))),
                "nearest_ttc_s": None if not math.isfinite(nearest_ttc_s) else float(nearest_ttc_s),
                "nearest_ttc_obstacle_id": str(nearest_ttc_debug.get("obstacle_id", "")),
                "nearest_ttc_reason": str(nearest_ttc_debug.get("reason", "")),
                "nearest_ttc_longitudinal_gap_m": (
                    None
                    if "longitudinal_gap_m" not in nearest_ttc_debug
                    else float(nearest_ttc_debug.get("longitudinal_gap_m", 0.0))
                ),
                "nearest_ttc_lateral_gap_m": (
                    None
                    if "lateral_gap_m" not in nearest_ttc_debug
                    else float(nearest_ttc_debug.get("lateral_gap_m", 0.0))
                ),
                "nearest_ttc_bumper_gap_m": (
                    None
                    if "bumper_gap_m" not in nearest_ttc_debug
                    else float(nearest_ttc_debug.get("bumper_gap_m", 0.0))
                ),
                "nearest_ttc_bumper_gap_raw_m": (
                    None
                    if "bumper_gap_raw_m" not in nearest_ttc_debug
                    else float(nearest_ttc_debug.get("bumper_gap_raw_m", 0.0))
                ),
                "nearest_ttc_closing_speed_mps": (
                    None
                    if "closing_speed_mps" not in nearest_ttc_debug
                    else float(nearest_ttc_debug.get("closing_speed_mps", 0.0))
                ),
                "nearest_ttc_obstacle_x": (
                    None
                    if "obstacle_x" not in nearest_ttc_debug
                    else float(nearest_ttc_debug.get("obstacle_x", 0.0))
                ),
                "nearest_ttc_obstacle_y": (
                    None
                    if "obstacle_y" not in nearest_ttc_debug
                    else float(nearest_ttc_debug.get("obstacle_y", 0.0))
                ),
                "nearest_ttc_obstacle_v_mps": (
                    None
                    if "obstacle_v_mps" not in nearest_ttc_debug
                    else float(nearest_ttc_debug.get("obstacle_v_mps", 0.0))
                ),
                "max_drac_mps2": None if not math.isfinite(max_tick_drac_mps2) else float(max_tick_drac_mps2),
                "min_pet_s": None if not math.isfinite(self.min_pet_s) else float(self.min_pet_s),
                "behavior_decision": str(behavior_decision),
                "fsm_state": str(fsm_state),
                "blocking_obstacle_id": str(blocking_obstacle_id),
                "decision_reason": str(decision_reason),
                "collision_count": int(collision_count),
                "last_collision_actor_type": str(last_collision_actor_type),
                "cp_provider_source": str(cp_provider_source),
                "native_opencda_available": int(bool(native_opencda_available)),
                "native_opencda_required": int(bool(native_opencda_required)),
                "cp_obstacle_count": (
                    int(len(list(obstacle_snapshots or [])))
                    if cp_obstacle_count is None
                    else int(cp_obstacle_count)
                ),
                # MPC diagnostic fields — filled in by record_mpc_extras() on replan ticks
                "lateral_offset_m": None,
                "heading_error_rad": None,
                "Cost_ref": None,
                "Cost_LaneCenter": None,
                "Cost_RoadBoundary": None,
                "Cost_Repulsive": None,
                "Cost_Control": None,
            }
        )

    def _grid_bin(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        bin_size_m = max(_EPS, float(self.pet_bin_size_m))
        return (
            int(math.floor(float(xy[0]) / bin_size_m)),
            int(math.floor(float(xy[1]) / bin_size_m)),
        )

    def _update_pet(
        self,
        *,
        obstacle_id: str,
        obstacle_xy: Tuple[float, float],
        ego_xy: Tuple[float, float],
        sim_time_s: float,
    ) -> None:
        ego_bin = self._grid_bin(ego_xy)
        obstacle_bin = self._grid_bin(obstacle_xy)
        if self._last_ego_bin is not None and obstacle_bin == self._last_ego_bin:
            pet_s = float(sim_time_s) - float(self._last_ego_bin_time_s or sim_time_s)
            if pet_s >= 0.0:
                self.min_pet_s = min(float(self.min_pet_s), float(pet_s))
        last_obstacle_bin_time = self._last_obstacle_bin_time_by_id.get(str(obstacle_id))
        if last_obstacle_bin_time is not None and ego_bin == last_obstacle_bin_time[0]:
            pet_s = float(sim_time_s) - float(last_obstacle_bin_time[1])
            if pet_s >= 0.0:
                self.min_pet_s = min(float(self.min_pet_s), float(pet_s))

        if math.hypot(float(obstacle_xy[0]) - float(ego_xy[0]), float(obstacle_xy[1]) - float(ego_xy[1])) <= float(self.pet_conflict_radius_m):
            self._last_obstacle_bin_time_by_id[str(obstacle_id)] = (obstacle_bin, float(sim_time_s))
        self._last_ego_bin = ego_bin
        self._last_ego_bin_time_s = float(sim_time_s)

    def summary(self) -> Dict[str, object]:
        distance_km = float(self.distance_traveled_m) / 1000.0
        success_rate = (
            float(self.mpc_plan_successes) / float(self.mpc_plan_attempts)
            if int(self.mpc_plan_attempts) > 0
            else 0.0
        )
        return {
            "collision_count": int(self.collision_count),
            "distance_traveled_m": float(self.distance_traveled_m),
            "collision_rate_per_km": float(self.collision_count)
            / max(float(self.collision_rate_distance_epsilon_km), float(distance_km)),
            "min_ttc_s": None if not math.isfinite(self.min_ttc_s) else float(self.min_ttc_s),
            "min_pet_s": None if not math.isfinite(self.min_pet_s) else float(self.min_pet_s),
            "max_drac_mps2": None if not math.isfinite(self.max_drac_mps2) else float(self.max_drac_mps2),
            "road_boundary_breach_count": int(self.road_boundary_breach_count),
            "road_boundary_sample_count": int(self.road_boundary_sample_count),
            "road_boundary_breach_rate": (
                float(self.road_boundary_breach_count)
                / float(self.road_boundary_sample_count)
                if int(self.road_boundary_sample_count) > 0
                else None
            ),
            "mpc_plan_attempts": int(self.mpc_plan_attempts),
            "mpc_plan_successes": int(self.mpc_plan_successes),
            "mpc_plan_success_rate": float(success_rate),
        }


def write_planning_metrics_artifacts(
    *,
    artifact_dir: str,
    recorder: EvaluationMetricsRecorder,
    scenario_name: str = "scenario",
) -> Dict[str, str]:
    os.makedirs(str(artifact_dir), exist_ok=True)
    json_path = os.path.join(str(artifact_dir), "planning_metrics.json")
    csv_path = os.path.join(str(artifact_dir), "planning_metrics_timeseries.csv")
    payload = {
        "scenario_name": str(scenario_name),
        "summary": recorder.summary(),
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    fieldnames = [
        "sample_index",
        "sim_time_s",
        "ego_x",
        "ego_y",
        "ego_speed_mps",
        "obstacle_count",
        "nearest_ttc_s",
        "nearest_ttc_obstacle_id",
        "nearest_ttc_reason",
        "nearest_ttc_longitudinal_gap_m",
        "nearest_ttc_lateral_gap_m",
        "nearest_ttc_bumper_gap_m",
        "nearest_ttc_bumper_gap_raw_m",
        "nearest_ttc_closing_speed_mps",
        "nearest_ttc_obstacle_x",
        "nearest_ttc_obstacle_y",
        "nearest_ttc_obstacle_v_mps",
        "max_drac_mps2",
        "min_pet_s",
        "behavior_decision",
        "fsm_state",
        "blocking_obstacle_id",
        "decision_reason",
        "collision_count",
        "last_collision_actor_type",
        "cp_provider_source",
        "native_opencda_available",
        "native_opencda_required",
        "cp_obstacle_count",
        "lateral_offset_m",
        "heading_error_rad",
        "Cost_ref",
        "Cost_LaneCenter",
        "Cost_RoadBoundary",
        "Cost_Repulsive",
        "Cost_Control",
        "road_boundary_sample_valid",
        "road_boundary_lateral_offset_m",
        "road_boundary_lane_width_m",
        "road_boundary_ego_half_width_m",
        "road_boundary_clearance_m",
        "road_boundary_breach",
        "road_boundary_breach_count",
        "road_boundary_sample_count",
        "road_boundary_breach_rate",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in recorder.samples:
            writer.writerow({name: sample.get(name, "") for name in fieldnames})

    collision_csv_path = os.path.join(str(artifact_dir), "collision_events.csv")
    collision_fields = [
        "collision_index", "event_id", "sim_time_s",
        "ego_x", "ego_y", "ego_speed_mps",
        "other_actor_type", "impulse_magnitude",
    ]
    with open(collision_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=collision_fields)
        writer.writeheader()
        for event in recorder.collision_events:
            writer.writerow({k: event.get(k, "") for k in collision_fields})

    traj_csv_path = os.path.join(str(artifact_dir), "planned_trajectory.csv")
    traj_fields = ["replan_index", "sim_time_s", "step_k", "x", "y", "v", "psi"]
    with open(traj_csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=traj_fields)
        writer.writeheader()
        for entry in recorder.planned_trajectory_samples:
            writer.writerow({k: entry.get(k, "") for k in traj_fields})

    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "collision_csv_path": collision_csv_path,
        "traj_csv_path": traj_csv_path,
    }
