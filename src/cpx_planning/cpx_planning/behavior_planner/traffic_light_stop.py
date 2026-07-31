"""CARLA-independent traffic-light decisions used by the copied behavior planner."""

from __future__ import annotations

from typing import Mapping


def normalize_signal_state(signal_state: object) -> str:
    """Normalize the signal value with the same rules used by OpenCDA."""
    raw_name = (str(signal_state) if signal_state is not None else "").strip().upper()
    if "." in raw_name:
        raw_name = raw_name.rsplit(".", 1)[-1]
    if raw_name == "2":
        return "green"
    if raw_name == "1":
        return "yellow"
    if raw_name == "0":
        return "red"
    if raw_name in {"GREEN", "GO"}:
        return "green"
    if raw_name in {"YELLOW", "AMBER"}:
        return "yellow"
    if raw_name in {"RED", "STOP"}:
        return "red"
    return "unknown"


def should_stop_for_signal(*, signal_state: object, stop_target: Mapping[str, object] | None, ego_velocity_mps: float, ego_max_deceleration_mps2: float, ego_in_junction: bool, stop_buffer_m: float = 2.0) -> bool:
    """Use the unchanged OpenCDA braking-distance rule to decide whether the ego should stop."""
    if bool(ego_in_junction):
        return False
    normalized_signal_state = normalize_signal_state(signal_state)
    if normalized_signal_state == "green" or normalized_signal_state not in {"yellow", "red"} or not isinstance(stop_target, Mapping):
        return False
    try:
        stop_distance_m = max(0.0, float(stop_target.get("distance_m", 0.0)))
    except Exception:
        return False
    if stop_distance_m <= 1.0e-3:
        return False
    ego_speed_mps = max(0.0, float(ego_velocity_mps))
    max_deceleration_mps2 = max(1.0e-6, float(ego_max_deceleration_mps2))
    required_stop_distance_m = ego_speed_mps * ego_speed_mps / (2.0 * max_deceleration_mps2) + max(0.0, float(stop_buffer_m))
    return stop_distance_m >= required_stop_distance_m
