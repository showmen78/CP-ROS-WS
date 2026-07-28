"""
Obstacle trajectory tracking and prediction module for MPC_custom.

This file defines a `Tracker` class that:
1. Stores recent state history for each non-ego vehicle.
2. Fits time-parameterized polynomials x(t), y(t) at each timestep.
3. Projects obstacle future trajectories over a short horizon (default 3 s).
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Deque, Dict, List, Mapping, Sequence
import math
import warnings

import numpy as np


@dataclass
class TrackerConfig:
    """
    Intent:
        Store tracker hyperparameters used for obstacle trajectory fitting.

    Fields:
        prediction_horizon_s:
            float [s], how far in the future obstacle trajectories are projected.
        polynomial_degree:
            int scalar, requested polynomial degree for x(t) and y(t).
            The effective degree is auto-limited by available data points.
        fit_window_s:
            float [s], only samples newer than `now - fit_window_s` are used.
        max_history_points:
            int scalar, ring-buffer cap for samples per obstacle.
        min_points_for_polyfit:
            int scalar, minimum points before using polynomial fit.
            Below this count, constant/linear fallback is used.
    """

    prediction_horizon_s: float
    polynomial_degree: int
    fit_window_s: float
    max_history_points: int
    min_points_for_polyfit: int
    signal_context_timeout_s: float
    signal_history_points: int


class Tracker:
    """
    Intent:
        Track obstacle motion and predict future obstacle trajectories.

    Logic and Method:
        - At each timestep, the tracker ingests current obstacle snapshots.
        - For each obstacle ID, it stores time-stamped history:
            [timestamp_s, x, y, v, psi].
        - To predict, it fits independent least-squares polynomials:
            x(t) and y(t), where t is relative time from latest sample.
        - It evaluates these polynomials for future times t = dt, 2*dt, ..., H,
          and also derives velocity/heading from polynomial derivatives.

    Inputs / Outputs:
        - `update(obstacle_snapshots, timestamp_s)`:
            obstacle_snapshots: sequence[dict], each containing at least:
                vehicle_id: str
                x, y, v, psi: float
            timestamp_s: float [s], current simulation timestamp.
        - `predict(step_dt_s, horizon_s=None)`:
            step_dt_s: float [s], sampling interval of requested prediction.
            horizon_s: float [s] | None, optional override of horizon.
            Returns:
                dict[str, list[dict[str, float]]]
                key: obstacle vehicle_id
                value: predicted sequence with fields:
                    t_s, x, y, v, psi
                shape per obstacle: (N, 5), N = round(horizon / step_dt_s).
    """

    def __init__(self, tracker_cfg: Mapping[str, object] | None = None) -> None:
        """
        Intent:
            Initialize tracker configuration and internal history buffers.

        Inputs:
            tracker_cfg:
                optional mapping with keys:
                    prediction_horizon_s, polynomial_degree, fit_window_s,
                    max_history_points, min_points_for_polyfit.
        """

        cfg = dict(tracker_cfg or {})
        self.config = TrackerConfig(
            prediction_horizon_s=max(0.1, float(cfg.get("prediction_horizon_s", 3.0))),
            polynomial_degree=max(1, int(cfg.get("polynomial_degree", 2))),
            fit_window_s=max(0.2, float(cfg.get("fit_window_s", 4.0))),
            max_history_points=max(8, int(cfg.get("max_history_points", 240))),
            min_points_for_polyfit=max(2, int(cfg.get("min_points_for_polyfit", 3))),
            signal_context_timeout_s=max(0.05, float(cfg.get("signal_context_timeout_s", 1.0))),
            signal_history_points=max(2, int(cfg.get("signal_history_points", 20))),
        )

        # obstacle_id -> deque of historical samples.
        # Each sample dict stores:
        #   timestamp_s, x, y, v, psi
        self._history_by_obstacle_id: Dict[str, Deque[Dict[str, float]]] = {}
        self._signal_history_by_actor_id: Dict[str, Deque[Dict[str, object]]] = {}
        self._latest_signal_context: Dict[str, object] | None = None
        self._last_timestamp_s: float | None = None

    def update(
        self,
        obstacle_snapshots: Sequence[Mapping[str, object]],
        timestamp_s: float,
        next_signal_context: Mapping[str, object] | None = None,
        next_stop_target: Mapping[str, object] | None = None,
    ) -> None:
        """
        Intent:
            Ingest current obstacle states and refresh per-obstacle histories.

        Logic:
            1. Append one timestamped sample per obstacle.
            2. Keep only active obstacle IDs from this frame.
            3. Remove stale samples older than fit_window_s.

        Inputs:
            obstacle_snapshots:
                sequence[Mapping[str, object]] with fields:
                    vehicle_id: str
                    x, y, v, psi: float
            timestamp_s:
                float [s], current simulation time.

        Output:
            None.
        """

        now_s = float(timestamp_s)
        self._last_timestamp_s = now_s

        active_ids: set[str] = set()
        min_keep_time_s = now_s - self.config.fit_window_s

        for snapshot in obstacle_snapshots:
            obstacle_id = str(snapshot.get("vehicle_id", ""))
            if obstacle_id == "":
                continue

            active_ids.add(obstacle_id)
            if obstacle_id not in self._history_by_obstacle_id:
                self._history_by_obstacle_id[obstacle_id] = deque(maxlen=self.config.max_history_points)

            sample = {
                "timestamp_s": now_s,
                "x": float(snapshot.get("x", 0.0)),
                "y": float(snapshot.get("y", 0.0)),
                "v": float(snapshot.get("v", 0.0)),
                "psi": float(snapshot.get("psi", 0.0)),
            }
            self._history_by_obstacle_id[obstacle_id].append(sample)

        # Drop removed obstacles.
        stale_ids = set(self._history_by_obstacle_id.keys()) - active_ids
        for stale_id in stale_ids:
            self._history_by_obstacle_id.pop(stale_id, None)

        # Window old samples out of active histories.
        for obstacle_id, history_deque in self._history_by_obstacle_id.items():
            while len(history_deque) > 0 and float(history_deque[0]["timestamp_s"]) < min_keep_time_s:
                history_deque.popleft()

        self.update_next_signal_context(
            signal_context=next_signal_context,
            stop_target=next_stop_target,
            timestamp_s=now_s,
        )

    def update_next_signal_context(
        self,
        signal_context: Mapping[str, object] | None,
        stop_target: Mapping[str, object] | None,
        timestamp_s: float,
    ) -> None:
        now_s = float(timestamp_s)
        normalized_context = self._normalize_signal_context(
            signal_context=signal_context,
            stop_target=stop_target,
            timestamp_s=now_s,
        )
        if normalized_context is None:
            self._latest_signal_context = None
            return

        actor_id = normalized_context.get("signal_actor_id", None)
        actor_key = str(actor_id) if actor_id is not None else "__none__"
        if actor_key not in self._signal_history_by_actor_id:
            self._signal_history_by_actor_id[actor_key] = deque(
                maxlen=self.config.signal_history_points
            )
        self._signal_history_by_actor_id[actor_key].append(dict(normalized_context))
        self._latest_signal_context = dict(normalized_context)

        min_keep_time_s = now_s - self.config.signal_context_timeout_s
        stale_actor_keys = []
        for signal_actor_key, history_deque in self._signal_history_by_actor_id.items():
            while len(history_deque) > 0 and float(history_deque[0].get("timestamp_s", 0.0)) < min_keep_time_s:
                history_deque.popleft()
            if len(history_deque) == 0:
                stale_actor_keys.append(signal_actor_key)
        for stale_actor_key in stale_actor_keys:
            self._signal_history_by_actor_id.pop(stale_actor_key, None)

    @staticmethod
    def _normalize_signal_context(
        *,
        signal_context: Mapping[str, object] | None,
        stop_target: Mapping[str, object] | None,
        timestamp_s: float,
    ) -> Dict[str, object] | None:
        has_signal_context = isinstance(signal_context, Mapping)
        has_stop_target = isinstance(stop_target, Mapping)
        if not has_signal_context and not has_stop_target:
            return None

        stop_target_distance_m = None
        normalized_stop_target = None
        if has_stop_target:
            normalized_stop_target = dict(stop_target)
            try:
                stop_target_distance_m = float(stop_target.get("distance_m", 0.0))
            except Exception:
                stop_target_distance_m = None

        signal_found = bool(signal_context.get("signal_found", False)) if has_signal_context else False
        signal_state = str(signal_context.get("signal_state", "unknown")) if has_signal_context else "unknown"
        raw_signal_distance_m = signal_context.get("signal_distance_m", None) if has_signal_context else None
        try:
            signal_distance_m = None if raw_signal_distance_m is None else float(raw_signal_distance_m)
        except Exception:
            signal_distance_m = None
        if stop_target_distance_m is not None:
            signal_distance_m = float(stop_target_distance_m)

        signal_actor_id = signal_context.get("signal_actor_id", None) if has_signal_context else None
        signal_actor_name = str(signal_context.get("signal_actor_name", "")) if has_signal_context else ""
        signal_source = str(signal_context.get("signal_source", "none")) if has_signal_context else "none"

        return {
            "timestamp_s": float(timestamp_s),
            "signal_found": bool(signal_found),
            "signal_state": str(signal_state or "unknown"),
            "phase": str(signal_state or "unknown"),
            "signal_distance_m": signal_distance_m,
            "signal_actor_id": signal_actor_id,
            "signal_actor_name": str(signal_actor_name),
            "signal_source": str(signal_source),
            "stop_target_distance_m": stop_target_distance_m,
            "stop_target": normalized_stop_target,
        }

    def get_next_signal_context(
        self,
        max_age_s: float | None = None,
    ) -> Dict[str, object]:
        empty_context: Dict[str, object] = {
            "signal_found": False,
            "signal_state": "unknown",
            "phase": "unknown",
            "signal_distance_m": None,
            "signal_actor_id": None,
            "signal_actor_name": "",
            "signal_source": "none",
            "stop_target_distance_m": None,
            "stop_target": None,
        }
        if self._latest_signal_context is None:
            return dict(empty_context)

        age_limit_s = float(
            self.config.signal_context_timeout_s if max_age_s is None else max_age_s
        )
        context_timestamp_s = float(self._latest_signal_context.get("timestamp_s", 0.0))
        if (
            self._last_timestamp_s is not None
            and float(self._last_timestamp_s) - float(context_timestamp_s) > float(age_limit_s)
        ):
            return dict(empty_context)

        return deepcopy(dict(self._latest_signal_context))

    @staticmethod
    def _fit_position_polynomial(
        time_rel_s: np.ndarray,
        values: np.ndarray,
        requested_degree: int,
    ) -> np.ndarray:
        """
        Intent:
            Fit one polynomial value(t) using least squares.

        Inputs:
            time_rel_s:
                np.ndarray, shape (M,), relative timestamps.
            values:
                np.ndarray, shape (M,), samples corresponding to time_rel_s.
            requested_degree:
                int scalar, desired polynomial degree.

        Output:
            np.ndarray polynomial coefficients (highest order first),
            shape (degree + 1,).
        """

        point_count = int(values.shape[0])
        degree = int(max(1, min(requested_degree, point_count - 1)))
        return np.polyfit(time_rel_s, values, deg=degree)

    @staticmethod
    def _fallback_predict(
        latest_sample: Mapping[str, float],
        future_time_s: float,
    ) -> Dict[str, float]:
        """Predict one future sample with constant velocity + heading (CV)."""
        x0 = float(latest_sample["x"])
        y0 = float(latest_sample["y"])
        v0 = float(latest_sample["v"])
        psi0 = float(latest_sample["psi"])
        x_pred = x0 + v0 * math.cos(psi0) * future_time_s
        y_pred = y0 + v0 * math.sin(psi0) * future_time_s
        return {
            "t_s": future_time_s,
            "x": x_pred,
            "y": y_pred,
            "v": v0,
            "psi": psi0,
            "yaw_rate_rad_s": 0.0,
        }

    @staticmethod
    def _estimate_yaw_rate(history: List[Dict[str, float]]) -> float:
        """Estimate yaw rate [rad/s] from history using finite differences.

        Uses the two most recent samples so the estimate reflects current
        turning state.  Wraps the angle delta to (-pi, pi] before dividing
        by the time interval to handle heading wrap-around correctly.
        """
        if len(history) < 2:
            return 0.0
        s0 = history[-2]
        s1 = history[-1]
        dt = float(s1["timestamp_s"]) - float(s0["timestamp_s"])
        if dt < 1.0e-6:
            return 0.0
        dpsi = float(s1["psi"]) - float(s0["psi"])
        dpsi = math.atan2(math.sin(dpsi), math.cos(dpsi))
        return float(dpsi / dt)

    @staticmethod
    def _ctrv_predict(
        latest_sample: Mapping[str, float],
        future_time_s: float,
        *,
        yaw_rate_rad_s: float = 0.0,
    ) -> Dict[str, float]:
        """Predict one future sample with CTRV (Constant Turn Rate & Velocity).

        Degenerates to straight-line CV when |yaw_rate| < 1e-4 rad/s to avoid
        divide-by-zero.  This is the standard model used in production tracking
        systems (Apollo, Autoware, SUMO) for short-horizon obstacle prediction.
        """
        x0 = float(latest_sample["x"])
        y0 = float(latest_sample["y"])
        v0 = float(latest_sample["v"])
        psi0 = float(latest_sample["psi"])
        omega = float(yaw_rate_rad_s)
        t = float(future_time_s)

        if abs(omega) < 1.0e-4:
            x_pred = x0 + v0 * math.cos(psi0) * t
            y_pred = y0 + v0 * math.sin(psi0) * t
            psi_pred = psi0
        else:
            x_pred = x0 + (v0 / omega) * (math.sin(psi0 + omega * t) - math.sin(psi0))
            y_pred = y0 + (v0 / omega) * (-math.cos(psi0 + omega * t) + math.cos(psi0))
            psi_pred = psi0 + omega * t

        return {
            "t_s": t,
            "x": x_pred,
            "y": y_pred,
            "v": v0,
            "psi": float(psi_pred),
            "yaw_rate_rad_s": float(omega),
        }

    def predict(
        self,
        step_dt_s: float,
        horizon_s: float | None = None,
    ) -> Dict[str, List[Dict[str, float]]]:
        """
        Intent:
            Generate future obstacle trajectories from tracked history.

        Method:
            For each obstacle:
            - If insufficient samples, use constant-velocity fallback.
            - Else fit x(t), y(t) polynomials and evaluate future samples.
            - Derive v, psi from first derivatives dx/dt, dy/dt.

        Inputs:
            step_dt_s:
                float [s], temporal spacing between predicted points.
            horizon_s:
                optional float [s], prediction horizon override.

        Output:
            dict[str, list[dict[str, float]]], predicted trajectory per obstacle.
        """

        if step_dt_s <= 0.0:
            raise ValueError("step_dt_s must be > 0.")

        horizon_value_s = float(self.config.prediction_horizon_s if horizon_s is None else horizon_s)
        horizon_value_s = max(step_dt_s, horizon_value_s)
        steps = max(1, int(round(horizon_value_s / step_dt_s)))

        prediction_by_obstacle_id: Dict[str, List[Dict[str, float]]] = {}

        for obstacle_id, history_deque in self._history_by_obstacle_id.items():
            history = list(history_deque)
            if len(history) == 0:
                continue

            latest = history[-1]
            prediction: List[Dict[str, float]] = []

            if len(history) < self.config.min_points_for_polyfit:
                if len(history) >= 2:
                    yaw_rate = self._estimate_yaw_rate(history)
                    for step_idx in range(steps):
                        t_future_s = (step_idx + 1) * step_dt_s
                        prediction.append(self._ctrv_predict(latest, t_future_s, yaw_rate_rad_s=yaw_rate))
                else:
                    for step_idx in range(steps):
                        t_future_s = (step_idx + 1) * step_dt_s
                        prediction.append(self._fallback_predict(latest, t_future_s))
                prediction_by_obstacle_id[obstacle_id] = prediction
                continue

            t0 = float(latest["timestamp_s"])
            time_rel_s = np.array([float(sample["timestamp_s"]) - t0 for sample in history], dtype=float)
            x_values = np.array([float(sample["x"]) for sample in history], dtype=float)
            y_values = np.array([float(sample["y"]) for sample in history], dtype=float)

            # Polyfit can emit RankWarning for near-singular matrices when all
            # timestamps or positions are almost constant; fallback handles that.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    coeff_x = self._fit_position_polynomial(
                        time_rel_s=time_rel_s,
                        values=x_values,
                        requested_degree=self.config.polynomial_degree,
                    )
                    coeff_y = self._fit_position_polynomial(
                        time_rel_s=time_rel_s,
                        values=y_values,
                        requested_degree=self.config.polynomial_degree,
                    )
                dcoeff_x = np.polyder(coeff_x)
                dcoeff_y = np.polyder(coeff_y)

                for step_idx in range(steps):
                    t_future_s = (step_idx + 1) * step_dt_s
                    x_pred = float(np.polyval(coeff_x, t_future_s))
                    y_pred = float(np.polyval(coeff_y, t_future_s))
                    vx_pred = float(np.polyval(dcoeff_x, t_future_s))
                    vy_pred = float(np.polyval(dcoeff_y, t_future_s))
                    v_pred = float(math.hypot(vx_pred, vy_pred))

                    if v_pred > 1e-5:
                        psi_pred = float(math.atan2(vy_pred, vx_pred))
                    else:
                        psi_pred = float(latest["psi"])

                    prediction.append(
                        {
                            "t_s": t_future_s,
                            "x": x_pred,
                            "y": y_pred,
                            "v": v_pred,
                            "psi": psi_pred,
                        }
                    )
            except Exception:
                prediction = []
                yaw_rate_fallback = self._estimate_yaw_rate(history) if len(history) >= 2 else 0.0
                for step_idx in range(steps):
                    t_future_s = (step_idx + 1) * step_dt_s
                    prediction.append(self._ctrv_predict(latest, t_future_s, yaw_rate_rad_s=yaw_rate_fallback))

            prediction_by_obstacle_id[obstacle_id] = prediction

        return prediction_by_obstacle_id

    def get_histories(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Intent:
            Return deep-copied obstacle histories for debugging/inspection.

        Output:
            dict[str, list[dict[str, float]]], obstacle history map.
        """

        return {key: deepcopy(list(value)) for key, value in self._history_by_obstacle_id.items()}

    def get_signal_histories(self) -> Dict[str, List[Dict[str, object]]]:
        return {key: deepcopy(list(value)) for key, value in self._signal_history_by_actor_id.items()}
