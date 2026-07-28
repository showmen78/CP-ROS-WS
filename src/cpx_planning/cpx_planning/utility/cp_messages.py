"""Shared cooperative-perception message helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CP_MESSAGE_PATH = os.path.join(PROJECT_ROOT, "behavior_planner", "cp_message.json")
_CONTROL_MESSAGE_TYPES = {"traffic_light", "intersection", "stop"}
_OBSTACLE_MESSAGE_TYPES = {"vehicle", "vru"}


def empty_cp_payload(schema_version: int = 1, timestamp_s: float = 0.0) -> Dict[str, Any]:
    return {
        "schema_version": int(schema_version),
        "sequence": 0,
        "timestamp_s": float(timestamp_s),
        "obstacles": [],
        "lane_events": [],
        "control": [],
    }


def _is_mapping_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def normalize_cp_payload(
    payload: object,
    *,
    schema_version: int = 1,
) -> Dict[str, Any]:
    normalized_payload = empty_cp_payload(schema_version=schema_version)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        lane_events, control = _partition_cp_messages(
            [dict(item) for item in list(payload) if isinstance(item, Mapping)]
        )
        normalized_payload["lane_events"] = lane_events
        normalized_payload["control"] = control
        return normalized_payload
    if not isinstance(payload, Mapping):
        return normalized_payload

    try:
        normalized_payload["schema_version"] = int(payload.get("schema_version", schema_version))
    except Exception:
        normalized_payload["schema_version"] = int(schema_version)
    try:
        normalized_payload["sequence"] = int(payload.get("sequence", 0))
    except Exception:
        normalized_payload["sequence"] = 0
    try:
        normalized_payload["timestamp_s"] = float(payload.get("timestamp_s", 0.0))
    except Exception:
        normalized_payload["timestamp_s"] = 0.0

    for key in ("obstacles", "lane_events", "control"):
        raw_items = payload.get(key, [])
        if _is_mapping_sequence(raw_items):
            normalized_payload[key] = [
                dict(item)
                for item in list(raw_items)
                if isinstance(item, Mapping)
            ]
    return normalized_payload


def ensure_cp_message_file_exists(message_path: str = CP_MESSAGE_PATH) -> None:
    normalized_path = str(message_path).strip() or CP_MESSAGE_PATH
    os.makedirs(os.path.dirname(os.path.abspath(normalized_path)), exist_ok=True)
    if os.path.exists(normalized_path):
        return
    with open(normalized_path, "w", encoding="utf-8") as message_file:
        json.dump(empty_cp_payload(), message_file, indent=2)


def load_cp_message_payload(
    message_path: str = CP_MESSAGE_PATH,
    *,
    schema_version: int = 1,
) -> Dict[str, Any]:
    ensure_cp_message_file_exists(message_path=message_path)
    try:
        with open(message_path, "r", encoding="utf-8") as message_file:
            payload = json.load(message_file)
    except Exception:
        return empty_cp_payload(schema_version=schema_version)
    return normalize_cp_payload(payload, schema_version=schema_version)


def write_cp_message_payload(
    payload: Mapping[str, object],
    message_path: str = CP_MESSAGE_PATH,
    *,
    schema_version: int = 1,
) -> None:
    ensure_cp_message_file_exists(message_path=message_path)
    normalized_payload = normalize_cp_payload(payload, schema_version=schema_version)
    with open(message_path, "w", encoding="utf-8") as message_file:
        json.dump(normalized_payload, message_file, indent=2)


def reset_cp_message_payload(
    message_path: str = CP_MESSAGE_PATH,
    *,
    schema_version: int = 1,
) -> None:
    write_cp_message_payload(
        empty_cp_payload(schema_version=schema_version, timestamp_s=0.0),
        message_path=message_path,
        schema_version=schema_version,
    )


def _partition_cp_messages(
    messages: Sequence[Mapping[str, object]] | None,
) -> tuple[List[dict], List[dict]]:
    lane_events: List[dict] = []
    control: List[dict] = []
    for message in list(messages or []):
        if not isinstance(message, Mapping):
            continue
        normalized_message = dict(message)
        message_type = str(normalized_message.get("type", "")).strip().lower()
        if message_type in _CONTROL_MESSAGE_TYPES:
            control.append(normalized_message)
        else:
            lane_events.append(normalized_message)
    return lane_events, control


def _normalized_id_set(message_ids: Sequence[object]) -> set[str]:
    return {
        str(message_id).strip()
        for message_id in list(message_ids or [])
        if str(message_id).strip()
    }


def replace_cp_list(
    *,
    message_path: str,
    schema_version: int,
    list_name: str,
    items: Sequence[Mapping[str, object]] | None,
    timestamp_s: float,
) -> None:
    payload = load_cp_message_payload(message_path=message_path, schema_version=schema_version)
    payload["schema_version"] = int(schema_version)
    payload[str(list_name)] = [
        dict(item)
        for item in list(items or [])
        if isinstance(item, Mapping)
    ]
    payload["sequence"] = int(payload.get("sequence", 0) or 0) + 1
    payload["timestamp_s"] = float(timestamp_s)
    write_cp_message_payload(payload, message_path=message_path, schema_version=schema_version)


def upsert_cp_item(
    *,
    message_path: str,
    schema_version: int,
    list_name: str,
    item: Mapping[str, object],
    timestamp_s: float,
) -> bool:
    item_id = str(item.get("id", "")).strip()
    if not item_id:
        return False

    payload = load_cp_message_payload(message_path=message_path, schema_version=schema_version)
    items = [dict(existing) for existing in list(payload.get(list_name, []) or [])]
    changed = True
    replaced = False
    next_items: List[dict] = []
    for existing in items:
        if str(existing.get("id", "")).strip() != item_id:
            next_items.append(existing)
            continue
        replaced = True
        changed = dict(existing) != dict(item)
        next_items.append(dict(item))
    if not replaced:
        next_items.append(dict(item))
    if replaced and not changed:
        return False

    payload["schema_version"] = int(schema_version)
    payload[str(list_name)] = next_items
    payload["sequence"] = int(payload.get("sequence", 0) or 0) + 1
    payload["timestamp_s"] = float(timestamp_s)
    write_cp_message_payload(payload, message_path=message_path, schema_version=schema_version)
    return True


def remove_cp_item(
    *,
    message_path: str,
    schema_version: int,
    list_name: str,
    item_id: object,
    timestamp_s: float,
) -> bool:
    normalized_item_id = str(item_id).strip()
    if not normalized_item_id:
        return False

    payload = load_cp_message_payload(message_path=message_path, schema_version=schema_version)
    items = [dict(existing) for existing in list(payload.get(list_name, []) or [])]
    next_items = [
        existing
        for existing in items
        if str(existing.get("id", "")).strip() != normalized_item_id
    ]
    if len(next_items) == len(items):
        return False

    payload["schema_version"] = int(schema_version)
    payload[str(list_name)] = next_items
    payload["sequence"] = int(payload.get("sequence", 0) or 0) + 1
    payload["timestamp_s"] = float(timestamp_s)
    write_cp_message_payload(payload, message_path=message_path, schema_version=schema_version)
    return True


def remove_cp_messages_by_id(
    message_ids: Sequence[object],
    message_path: str = CP_MESSAGE_PATH,
    *,
    schema_version: int = 1,
) -> List[dict]:
    remove_ids = _normalized_id_set(message_ids)
    if len(remove_ids) == 0:
        return load_cp_messages(message_path=message_path, schema_version=schema_version)

    payload = load_cp_message_payload(message_path=message_path, schema_version=schema_version)
    retained_payload = dict(payload)
    for key in ("obstacles", "lane_events", "control"):
        retained_payload[key] = [
            dict(message)
            for message in list(payload.get(key, []))
            if isinstance(message, Mapping)
            and (
                not str(message.get("id", "")).strip()
                or str(message.get("id", "")).strip() not in remove_ids
            )
        ]
    write_cp_message_payload(
        retained_payload,
        message_path=message_path,
        schema_version=schema_version,
    )
    return load_cp_messages(message_path=message_path, schema_version=schema_version)


def load_cp_messages(
    message_path: str = CP_MESSAGE_PATH,
    *,
    schema_version: int = 1,
) -> List[dict]:
    payload = load_cp_message_payload(message_path=message_path, schema_version=schema_version)
    messages: List[dict] = []
    for key in ("lane_events", "control"):
        messages.extend(
            [
                dict(item)
                for item in list(payload.get(key, []))
                if isinstance(item, Mapping)
            ]
        )
    return messages


def write_cp_messages(
    messages: Sequence[Mapping[str, object]],
    message_path: str = CP_MESSAGE_PATH,
    *,
    schema_version: int = 1,
) -> None:
    payload = load_cp_message_payload(message_path=message_path, schema_version=schema_version)
    lane_events, control = _partition_cp_messages(messages)
    payload["lane_events"] = lane_events
    payload["control"] = control
    write_cp_message_payload(payload, message_path=message_path, schema_version=schema_version)


def lane_closure_messages(messages: Sequence[Mapping[str, object]]) -> List[dict]:
    valid_messages: List[dict] = []
    for message in list(messages or []):
        if not isinstance(message, Mapping):
            continue
        if str(message.get("type", "")).strip().lower() != "lane_closure":
            continue
        if not str(message.get("id", "")).strip():
            continue
        valid_messages.append(dict(message))
    return valid_messages


def load_lane_closure_messages(
    message_path: str = CP_MESSAGE_PATH,
    *,
    schema_version: int = 1,
) -> List[dict]:
    payload = load_cp_message_payload(message_path=message_path, schema_version=schema_version)
    return lane_closure_messages(payload.get("lane_events", []))


def control_messages(messages: Sequence[Mapping[str, object]]) -> List[dict]:
    valid_messages: List[dict] = []
    for message in list(messages or []):
        if not isinstance(message, Mapping):
            continue
        message_id = str(message.get("id", "")).strip()
        message_type = str(message.get("type", "")).strip().lower()
        if not message_id or message_type not in _CONTROL_MESSAGE_TYPES:
            continue
        valid_messages.append(dict(message))
    return valid_messages


def load_control_messages(
    message_path: str = CP_MESSAGE_PATH,
    *,
    schema_version: int = 1,
) -> List[dict]:
    payload = load_cp_message_payload(message_path=message_path, schema_version=schema_version)
    return control_messages(payload.get("control", []))


def pop_lane_closure_messages(
    message_path: str = CP_MESSAGE_PATH,
    *,
    schema_version: int = 1,
) -> List[dict]:
    payload = load_cp_message_payload(message_path=message_path, schema_version=schema_version)
    closure_messages = lane_closure_messages(payload.get("lane_events", []))
    if len(closure_messages) == 0:
        return []
    closure_ids = _normalized_id_set(
        [message.get("id", "") for message in closure_messages]
    )
    payload["lane_events"] = [
        dict(message)
        for message in list(payload.get("lane_events", []))
        if isinstance(message, Mapping)
        and (
            str(message.get("type", "")).strip().lower() != "lane_closure"
            or str(message.get("id", "")).strip() not in closure_ids
        )
    ]
    write_cp_message_payload(payload, message_path=message_path, schema_version=schema_version)
    return [dict(message) for message in closure_messages]


def obstacle_messages(messages: Sequence[Mapping[str, object]]) -> List[dict]:
    valid_messages: List[dict] = []
    for message in list(messages or []):
        if not isinstance(message, Mapping):
            continue
        message_type = str(message.get("type", "")).strip().lower()
        if message_type not in _OBSTACLE_MESSAGE_TYPES:
            continue
        if message.get("id", None) is None:
            continue
        valid_messages.append(dict(message))
    return valid_messages


def load_obstacle_messages(
    message_path: str = CP_MESSAGE_PATH,
    *,
    schema_version: int = 1,
) -> List[dict]:
    payload = load_cp_message_payload(message_path=message_path, schema_version=schema_version)
    return obstacle_messages(payload.get("obstacles", []))


def replace_obstacle_messages(
    *,
    message_path: str,
    schema_version: int,
    obstacles: Sequence[Mapping[str, object]] | None,
    timestamp_s: float,
) -> None:
    replace_cp_list(
        message_path=message_path,
        schema_version=schema_version,
        list_name="obstacles",
        items=obstacles,
        timestamp_s=float(timestamp_s),
    )


def _coerce_xyvp_state(raw_state: object) -> List[float] | None:
    if not _is_mapping_sequence(raw_state):
        return None
    if len(raw_state) < 4:
        return None
    try:
        return [
            float(raw_state[0]),
            float(raw_state[1]),
            float(raw_state[2]),
            float(raw_state[3]),
        ]
    except Exception:
        return None


def obstacle_message_to_snapshot(message: Mapping[str, object]) -> Dict[str, object] | None:
    state = _coerce_xyvp_state(message.get("state", None))
    if state is None:
        return None

    raw_shape = message.get("shape", None)
    length_m = 4.5
    width_m = 2.0
    height_m = 2.0
    if _is_mapping_sequence(raw_shape) and len(raw_shape) >= 2:
        try:
            length_m = float(raw_shape[0])
            width_m = float(raw_shape[1])
        except Exception:
            pass
    elif isinstance(raw_shape, Mapping):
        try:
            length_m = float(raw_shape.get("length_m", raw_shape.get("length", length_m)))
        except Exception:
            pass
        try:
            width_m = float(raw_shape.get("width_m", raw_shape.get("width", width_m)))
        except Exception:
            pass
        try:
            height_m = float(raw_shape.get("height_m", raw_shape.get("height", height_m)))
        except Exception:
            pass
    try:
        height_m = float(message.get("height_m", height_m))
    except Exception:
        pass

    trajectory: List[List[float]] = []
    raw_trajectory = message.get("trajectory", [])
    if _is_mapping_sequence(raw_trajectory):
        for raw_state in list(raw_trajectory):
            normalized_state = _coerce_xyvp_state(raw_state)
            if normalized_state is not None:
                trajectory.append(list(normalized_state))

    try:
        z_m = float(message.get("z", 0.0))
    except Exception:
        z_m = 0.0
    try:
        road_id = int(message.get("road_id", -1))
    except Exception:
        road_id = -1
    try:
        lane_id = int(message.get("lane_id", 0))
    except Exception:
        lane_id = 0

    snapshot = {
        "vehicle_id": str(message.get("id")),
        "x": float(state[0]),
        "y": float(state[1]),
        "z": float(z_m),
        "v": float(state[2]),
        "psi": float(state[3]),
        "length_m": float(length_m),
        "width_m": float(width_m),
        "height_m": float(height_m),
        "type": str(message.get("type", "vehicle")).strip().lower() or "vehicle",
        "road_id": int(road_id),
        "lane_id": int(lane_id),
        "predicted_trajectory": [list(item) for item in trajectory],
    }
    return snapshot


def obstacle_messages_to_snapshots(messages: Sequence[Mapping[str, object]]) -> List[dict]:
    snapshots: List[dict] = []
    for message in list(messages or []):
        if not isinstance(message, Mapping):
            continue
        snapshot = obstacle_message_to_snapshot(message)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


def load_obstacle_snapshots(
    message_path: str = CP_MESSAGE_PATH,
    *,
    schema_version: int = 1,
) -> List[dict]:
    return obstacle_messages_to_snapshots(
        load_obstacle_messages(message_path=message_path, schema_version=schema_version)
    )
