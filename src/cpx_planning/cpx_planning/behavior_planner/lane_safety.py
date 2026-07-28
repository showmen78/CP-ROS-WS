"""
Lane safety scoring module.

For each drivable lane (same direction), computes a safety score in [0, 1]
from the ego position using the nearest relevant front/rear obstacle.

The live score is intentionally smooth:
    - free lane -> 1.0
    - obstacle far away -> close to 1.0
    - obstacle near / fast-closing -> low

Lane score:
    - ego lane uses front-obstacle score only
    - other lanes use min(front score, rear score)
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------- #
# Defaults                                                                #
# --------------------------------------------------------------------- #
DEFAULT_D_SAFE_M = 8.0
DEFAULT_TTC_SAFE_S = 3.0
DEFAULT_SIGMOID_K = 5.0
DEFAULT_TTC_HISTORY_SIZE = 10
_EPS = 1e-6
DEFAULT_REAR_D_SAFE_M = DEFAULT_D_SAFE_M
DEFAULT_REAR_TTC_SAFE_S = DEFAULT_TTC_SAFE_S
DEFAULT_TTC_EPSILON_MPS = 0.1
DEFAULT_INFINITE_TTC_CAP_S = 15.0


# --------------------------------------------------------------------- #
# Scorer (pure computation, no threading)                                 #
# --------------------------------------------------------------------- #
class LaneSafetyScorer:
    """Compute per-lane safety scores from obstacle data."""

    def __init__(
        self,
        d_safe_m: float = DEFAULT_D_SAFE_M,
        rear_d_safe_m: float = DEFAULT_REAR_D_SAFE_M,
        ttc_safe_s: float = DEFAULT_TTC_SAFE_S,
        rear_ttc_safe_s: float = DEFAULT_REAR_TTC_SAFE_S,
        sigmoid_k: float = DEFAULT_SIGMOID_K,
        ttc_history_size: int = DEFAULT_TTC_HISTORY_SIZE,
        ttc_epsilon_mps: float = DEFAULT_TTC_EPSILON_MPS,
        infinite_ttc_cap_s: float = DEFAULT_INFINITE_TTC_CAP_S,
    ) -> None:
        self._front_d_safe_m = max(0.0, float(d_safe_m))
        self._rear_d_safe_m = max(0.0, float(rear_d_safe_m))
        self._front_ttc_safe_s = max(0.0, float(ttc_safe_s))
        self._rear_ttc_safe_s = max(0.0, float(rear_ttc_safe_s))
        self._sigmoid_k = float(sigmoid_k)
        self._ttc_history_size = int(ttc_history_size)
        self._ttc_epsilon_mps = max(_EPS, float(ttc_epsilon_mps))
        self._infinite_ttc_cap_s = max(1.0, float(infinite_ttc_cap_s))
        # obstacle_id -> deque of (timestamp_s, ttc_s)
        self._ttc_history: Dict[str, deque] = {}

    # ----------------------------------------------------------------- #
    # TTC helpers                                                         #
    # ----------------------------------------------------------------- #
    @staticmethod
    def _finite_or_default(value: Any, default: float) -> float:
        try:
            numeric_value = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(float(numeric_value)):
            return float(default)
        return float(numeric_value)

    @staticmethod
    def _clamp_unit_interval(value: Any, default: float = 0.0) -> float:
        numeric_value = LaneSafetyScorer._finite_or_default(value, default)
        return float(max(0.0, min(1.0, float(numeric_value))))

    def _compute_ttc(self, distance_m: float, delta_v_mps: float) -> float:
        """TTC = d / max(eps, dv).  If dv <= 0 the obstacle is not approaching."""
        if not math.isfinite(float(distance_m)) or not math.isfinite(float(delta_v_mps)):
            return float("inf")
        if delta_v_mps <= 0.0:
            return float("inf")
        return float(distance_m) / max(
            float(self._ttc_epsilon_mps), float(delta_v_mps)
        )

    def _update_ttc_history(self, obstacle_id: str, ttc_s: float, timestamp_s: float) -> None:
        if not math.isfinite(float(ttc_s)):
            ttc_s = float(self._infinite_ttc_cap_s)
        if obstacle_id not in self._ttc_history:
            self._ttc_history[obstacle_id] = deque(maxlen=self._ttc_history_size)
        self._ttc_history[obstacle_id].append((float(timestamp_s), float(ttc_s)))

    def _compute_ttc_slope(self, obstacle_id: str) -> float:
        """Linear-regression slope of TTC vs time.  Positive = improving."""
        history = self._ttc_history.get(obstacle_id)
        if history is None or len(history) < 3:
            return float("nan")

        times = np.array([e[0] for e in history], dtype=np.float64)
        ttcs = np.array([e[1] for e in history], dtype=np.float64)
        # Cap infinite TTCs for numerical stability
        ttcs = np.clip(ttcs, 0.0, float(self._infinite_ttc_cap_s))

        t_mean = np.mean(times)
        ttc_mean = np.mean(ttcs)
        numerator = float(np.sum((times - t_mean) * (ttcs - ttc_mean)))
        denominator = float(np.sum((times - t_mean) ** 2))
        if abs(denominator) < _EPS:
            return float("nan")
        return numerator / denominator

    # ----------------------------------------------------------------- #
    # Single-obstacle score                                               #
    # ----------------------------------------------------------------- #
    def _distance_term(self, distance_m: float, safe_distance_m: float) -> float:
        if not math.isfinite(float(distance_m)):
            return 0.0
        if float(distance_m) <= 0.0:
            return 0.0
        safe_distance_m = max(_EPS, float(safe_distance_m))
        distance_pow4 = float(distance_m) ** 4
        safe_pow4 = float(safe_distance_m) ** 4
        return self._clamp_unit_interval(
            float(distance_pow4 / max(_EPS, distance_pow4 + safe_pow4)),
            default=0.0,
        )

    def _ttc_term(self, ttc_s: float, safe_ttc_s: float) -> float:
        if not math.isfinite(float(ttc_s)):
            return 1.0
        if float(ttc_s) <= 0.0:
            return 0.0
        safe_ttc_s = max(_EPS, float(safe_ttc_s))
        ttc_s = min(float(ttc_s), float(self._infinite_ttc_cap_s))
        ttc_sq = float(ttc_s) ** 2
        safe_sq = float(safe_ttc_s) ** 2
        return self._clamp_unit_interval(
            float(ttc_sq / max(_EPS, ttc_sq + safe_sq)),
            default=0.0,
        )

    def _trend_term(self, ttc_slope: float) -> float:
        if not math.isfinite(float(ttc_slope)):
            return 1.0
        bounded_exponent = max(-20.0, min(20.0, -self._sigmoid_k * float(ttc_slope)))
        trend_sigmoid = 1.0 / (1.0 + math.exp(float(bounded_exponent)))
        # Trend is only a small correction; it should not dominate distance/TTC.
        return self._clamp_unit_interval(0.85 + 0.15 * float(trend_sigmoid), default=1.0)

    def _obstacle_score(
        self,
        distance_m: float,
        ttc_s: float,
        ttc_slope: float,
        *,
        safe_distance_m: float,
        safe_ttc_s: float,
    ) -> float:
        """S_obs in [0, 1]."""
        if not math.isfinite(float(distance_m)):
            return 1.0

        safe_distance_m = max(_EPS, float(safe_distance_m))
        safe_ttc_s = max(_EPS, float(safe_ttc_s))

        # Only an imminent collision risk should clamp to zero.
        if float(distance_m) <= max(0.5, 0.15 * float(safe_distance_m)):
            return 0.0
        if math.isfinite(float(ttc_s)) and float(ttc_s) <= max(0.25, 0.15 * float(safe_ttc_s)):
            return 0.0

        distance_term = self._distance_term(distance_m, float(safe_distance_m))
        ttc_term = self._ttc_term(ttc_s, float(safe_ttc_s))
        trend_term = self._trend_term(ttc_slope)

        return self._clamp_unit_interval(
            0.98 * float(distance_term) * float(ttc_term) * float(trend_term),
            default=0.0,
        )

    # ----------------------------------------------------------------- #
    # Per-lane scoring                                                    #
    # ----------------------------------------------------------------- #
    def compute_lane_scores(
        self,
        ego_snapshot: Mapping[str, float],
        obstacle_snapshots: Sequence[Mapping[str, Any]],
        lane_assignments: Mapping[str, int],
        ego_lane_id: int,
        available_lane_ids: Sequence[int],
        timestamp_s: float,
    ) -> Dict[int, float]:
        """
        Compute safety score for every available lane.

        Parameters
        ----------
        ego_snapshot        : {x, y, v, psi}
        obstacle_snapshots  : list of {vehicle_id, x, y, v, psi, ...}
        lane_assignments    : vehicle_id -> internal lane_id
        ego_lane_id         : ego's current internal lane_id
        available_lane_ids  : [1, 2, ..., N]
        timestamp_s         : simulation clock (for TTC history)

        Returns
        -------
        {lane_id: score}  where score in [0, 1].
        """
        ego_x = self._finite_or_default(ego_snapshot.get("x", 0.0), 0.0)
        ego_y = self._finite_or_default(ego_snapshot.get("y", 0.0), 0.0)
        ego_v = max(0.0, self._finite_or_default(ego_snapshot.get("v", 0.0), 0.0))
        ego_psi = self._finite_or_default(ego_snapshot.get("psi", 0.0), 0.0)
        cos_h = math.cos(ego_psi)
        sin_h = math.sin(ego_psi)

        # Group obstacles by lane
        lane_obstacles: Dict[int, List[Mapping[str, Any]]] = {
            lid: [] for lid in available_lane_ids
        }
        for obs in obstacle_snapshots:
            obs_id = str(obs.get("vehicle_id", ""))
            obs_lane = lane_assignments.get(obs_id, -1)
            if obs_lane in lane_obstacles:
                lane_obstacles[obs_lane].append(obs)

        scores: Dict[int, float] = {}

        for lane_id in available_lane_ids:
            is_ego_lane = int(lane_id) == int(ego_lane_id)
            obstacles = lane_obstacles.get(lane_id, [])
            if len(obstacles) == 0:
                scores[lane_id] = 1.0
                continue

            lane_score = 1.0
            for obs in obstacles:
                obs_x = self._finite_or_default(obs.get("x", 0.0), float("nan"))
                obs_y = self._finite_or_default(obs.get("y", 0.0), float("nan"))
                if not math.isfinite(float(obs_x)) or not math.isfinite(float(obs_y)):
                    continue

                dx = float(obs_x) - ego_x
                dy = float(obs_y) - ego_y
                longitudinal = cos_h * dx + sin_h * dy
                if not math.isfinite(float(longitudinal)):
                    continue

                if float(longitudinal) < 0.0 and bool(is_ego_lane):
                    # Rear obstacles on the ego lane must not affect lane safety.
                    continue

                obs_v = max(0.0, self._finite_or_default(obs.get("v", 0.0), 0.0))
                obs_id = str(obs.get("vehicle_id", f"_lane_{lane_id}"))
                distance_m = abs(float(longitudinal))
                if float(longitudinal) >= 0.0:
                    delta_v = ego_v - obs_v
                    ttc = self._compute_ttc(distance_m, delta_v)
                    self._update_ttc_history(obs_id, ttc, timestamp_s)
                    slope = self._compute_ttc_slope(obs_id)
                    obstacle_score = self._obstacle_score(
                        distance_m,
                        ttc,
                        slope,
                        safe_distance_m=float(self._front_d_safe_m),
                        safe_ttc_s=float(self._front_ttc_safe_s),
                    )
                else:
                    delta_v = obs_v - ego_v
                    ttc = self._compute_ttc(distance_m, delta_v)
                    self._update_ttc_history(obs_id, ttc, timestamp_s)
                    slope = self._compute_ttc_slope(obs_id)
                    obstacle_score = self._obstacle_score(
                        distance_m,
                        ttc,
                        slope,
                        safe_distance_m=float(self._rear_d_safe_m),
                        safe_ttc_s=float(self._rear_ttc_safe_s),
                    )

                lane_score = min(float(lane_score), float(obstacle_score))

            scores[lane_id] = self._clamp_unit_interval(lane_score, default=0.0)

        return scores

    def cleanup_stale_obstacles(self, active_ids: set[str]) -> None:
        """Remove TTC history for obstacles that disappeared."""
        stale = [oid for oid in self._ttc_history if oid not in active_ids]
        for oid in stale:
            del self._ttc_history[oid]
