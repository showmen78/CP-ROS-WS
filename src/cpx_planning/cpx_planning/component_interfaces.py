"""ROS communication helpers for the separated CP-X planner components.

The planner still works with its original Python dataclasses and dictionaries.
This module only moves those values across ROS topics as JSON. It
does not change any planning calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, field
import importlib
import itertools
import json
import math
from pathlib import Path
from types import SimpleNamespace
import threading
import time
from typing import Any, Dict, List, Mapping

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


@dataclass(frozen=True)
class PlannerLocation:
    """Numeric position used by the planner without a simulator object."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass(frozen=True)
class PlannerRotation:
    """Numeric world rotation used by the planner without a simulator object."""

    yaw: float = 0.0


@dataclass(frozen=True)
class PlannerTransform:
    """Numeric pose compatible with the existing planner method inputs."""

    location: PlannerLocation = field(default_factory=PlannerLocation)
    rotation: PlannerRotation = field(default_factory=PlannerRotation)


@dataclass(frozen=True)
class PlannerControl:
    """Simulator-independent throttle, brake, and normalized steering values."""

    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0


class PlannerRuntime:
    """Factory names expected by copied actuator logic, backed only by plain values."""

    Location = PlannerLocation
    Transform = PlannerTransform
    VehicleControl = PlannerControl
    Rotation = PlannerRotation


class PlannerSafetyManager:
    """Primitive view of the latest OpenCDA safety-manager status."""

    def __init__(self, timestamp_s: float, status: Dict[str, bool]):
        self.status_queue = [(float(timestamp_s), dict(status))]


@dataclass(frozen=True)
class ROSInputSnapshot:
    """Plain values collected from the latest ROS input messages."""

    timestamp_s: float
    ego_pose: Dict[str, float]
    ego_speed_mps: float
    perception_objects: List[Dict[str, object]] = field(default_factory=list)
    v2x_objects: List[Dict[str, object]] = field(default_factory=list)
    traffic_lights: List[Dict[str, object]] = field(default_factory=list)
    lane_events: List[Dict[str, object]] = field(default_factory=list)
    final_goal: Dict[str, float] = field(default_factory=dict)


def _qualified_name(value: Any) -> str:
    return "{}.{}".format(value.__class__.__module__, value.__class__.__qualname__)


def to_wire(value: Any) -> Any:
    """Convert a planner value to JSON data without losing tuples, map keys, or dataclass types."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__cpx_type__": "float", "value": str(value)}
    if isinstance(value, np.ndarray):
        return {"__cpx_type__": "ndarray", "dtype": str(value.dtype), "value": value.tolist()}
    if isinstance(value, RemoteWaypoint):
        return {"__cpx_type__": "waypoint", "value": to_wire(value.to_wire_dict())}
    if value.__class__.__name__ == "Waypoint" and callable(getattr(value, "to_dict", None)):
        return {"__cpx_type__": "waypoint", "value": to_wire(value.to_dict())}
    if isinstance(value, PlannerSafetyManager):
        timestamp_s, status = value.status_queue[-1]
        return {"__cpx_type__": "safety_manager", "timestamp_s": float(timestamp_s), "status": to_wire(status)}
    if is_dataclass(value):
        return {"__cpx_type__": "dataclass", "class": _qualified_name(value), "fields": {item.name: to_wire(getattr(value, item.name)) for item in fields(value)}}
    if isinstance(value, Mapping):
        return {"__cpx_type__": "mapping", "items": [[to_wire(key), to_wire(item)] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"__cpx_type__": "tuple", "items": [to_wire(item) for item in value]}
    if isinstance(value, (list, set)):
        return [to_wire(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return to_wire(item_method())
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return {"__cpx_type__": "plain_object", "class": _qualified_name(value), "fields": to_wire(as_dict())}
    raise TypeError("Unsupported CP-X wire value: {}".format(type(value).__name__))


def _load_class(name: str):
    module_name, class_name = str(name).rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def from_wire(value: Any, *, waypoint_client=None) -> Any:
    """Restore JSON data to the planner's original Python values."""
    if isinstance(value, list):
        return [from_wire(item, waypoint_client=waypoint_client) for item in value]
    if not isinstance(value, dict) or "__cpx_type__" not in value:
        return value
    value_type = str(value["__cpx_type__"])
    if value_type == "float":
        return float(value["value"])
    if value_type == "ndarray":
        return np.asarray(value["value"], dtype=str(value.get("dtype", "float64")))
    if value_type == "tuple":
        return tuple(from_wire(item, waypoint_client=waypoint_client) for item in value.get("items", []))
    if value_type == "mapping":
        return {from_wire(pair[0], waypoint_client=waypoint_client): from_wire(pair[1], waypoint_client=waypoint_client) for pair in value.get("items", [])}
    if value_type == "safety_manager":
        return PlannerSafetyManager(float(value.get("timestamp_s", 0.0)), from_wire(value.get("status", {}), waypoint_client=waypoint_client))
    if value_type == "waypoint":
        waypoint_data = from_wire(value.get("value", {}), waypoint_client=waypoint_client)
        return RemoteWaypoint(waypoint_data, waypoint_client) if waypoint_client is not None else waypoint_data
    if value_type == "dataclass":
        cls = _load_class(value["class"])
        kwargs = {name: from_wire(item, waypoint_client=waypoint_client) for name, item in value.get("fields", {}).items()}
        return cls(**kwargs)
    if value_type == "plain_object":
        cls = _load_class(value["class"])
        kwargs = from_wire(value.get("fields", {}), waypoint_client=waypoint_client)
        return cls(**kwargs)
    return value


def encode_json(value: Any) -> str:
    """Encode one planner value for a typed ROS JSON envelope."""
    return json.dumps(to_wire(value), allow_nan=False, separators=(",", ":"))


def decode_json(payload: str, *, waypoint_client=None) -> Any:
    """Decode one planner value from a typed ROS JSON envelope."""
    return from_wire(json.loads(str(payload or "null")), waypoint_client=waypoint_client)


class LocalComponentBus:
    """Fast transport used when the ROS component nodes share one process.

    The separated nodes still own and lock their original component objects.
    This bus only avoids sending a local request through DDS and then blocking
    for the matching local result. The normal fast launch passes the existing
    CP-X objects directly, matching the monolithic planner's call behavior.
    When full topic debugging is enabled, values still cross the wire
    conversion boundary. Standalone node executables do not receive this bus
    and keep using the ROS request/result topics.
    """

    def __init__(self, mirror_payloads: bool = False, mirror_markers: bool = False, timing_recorder=None):
        self._servers = {}
        self._input_frames = {}
        self._frame_order = []
        self._request_ids = itertools.count(1)
        self.mirror_payloads = bool(mirror_payloads)
        self.mirror_markers = bool(mirror_markers)
        self.timing_recorder = timing_recorder
        self._mirrored_cycles = set()
        self._mirrored_cycle_order = []
        self._map_query_cache_cycle_id = None
        self._map_query_cache = {}
        self._lock = threading.RLock()

    def register_server(self, request_topic: str, server) -> None:
        with self._lock:
            self._servers[str(request_topic)] = server

    def unregister_server(self, request_topic: str, server) -> None:
        with self._lock:
            if self._servers.get(str(request_topic)) is server:
                self._servers.pop(str(request_topic), None)

    def has_server(self, request_topic: str) -> bool:
        with self._lock:
            return str(request_topic) in self._servers

    def store_input_frame(self, message, max_frames: int = 8) -> None:
        cycle_id = int(message.cycle_id)
        with self._lock:
            if cycle_id not in self._input_frames:
                self._frame_order.append(cycle_id)
            self._input_frames[cycle_id] = message
            while len(self._frame_order) > max(2, int(max_frames)):
                old_cycle_id = self._frame_order.pop(0)
                self._input_frames.pop(old_cycle_id, None)

    def get_input_frame(self, cycle_id: int):
        with self._lock:
            return self._input_frames.get(int(cycle_id))

    def cached_map_call(self, cycle_id: int, cache_key, callback):
        """Run one exact map query at most once during a planning cycle.

        The copied planner is allowed to ask the same map question in several
        stages. The map is static, so returning the first exact answer again
        preserves the planner result while avoiding another AD-map lookup.
        """
        cycle_id = int(cycle_id)
        if cycle_id <= 0:
            started = time.perf_counter()
            return callback(), False, (time.perf_counter() - started) * 1000.0
        with self._lock:
            if self._map_query_cache_cycle_id != cycle_id:
                self._map_query_cache_cycle_id = cycle_id
                self._map_query_cache.clear()
            if cache_key in self._map_query_cache:
                return self._map_query_cache[cache_key], True, 0.0
        started = time.perf_counter()
        result = callback()
        execution_ms = (time.perf_counter() - started) * 1000.0
        with self._lock:
            if self._map_query_cache_cycle_id == cycle_id:
                self._map_query_cache[cache_key] = result
        return result, False, execution_ms

    def call(self, request_topic: str, operation: str, payload: Any, *, cycle_id: int, timestamp_s: float, waypoint_client=None, component_client=None) -> Any:
        call_started = time.perf_counter()
        stage_prefix = "component.{}.{}".format(str(request_topic).strip("/").replace("/", "."), str(operation))
        forwarding_prefix = "inter_node_forwarding.{}.{}".format(str(request_topic).strip("/").replace("/", "."), str(operation))
        scheduling_prefix = "node_scheduling.component.{}.{}".format(str(request_topic).strip("/").replace("/", "."), str(operation))
        with self._lock:
            server = self._servers.get(str(request_topic))
        if server is None:
            raise RuntimeError("No local component owns '{}'.".format(request_topic))
        from cpx_interfaces.msg import ComponentPacket

        request = ComponentPacket()
        fill_header(request.header, float(timestamp_s))
        request.cycle_id = int(cycle_id)
        request.request_id = int(next(self._request_ids))
        request.requester = str(getattr(component_client, "_requester", "local_component_bus"))
        request.operation = str(operation)
        request.success = True
        request.reason = ""
        # The composed launch keeps all ROS nodes in one process. Pass the
        # original CP-X value directly, just as the monolithic OpenCDA planner
        # does, instead of recursively copying it to and from a JSON-shaped
        # value. Standalone nodes still use the normal topic serialization in
        # ComponentClient and ComponentServer below.
        use_wire_copy = bool(self.mirror_payloads)
        encode_started = time.perf_counter()
        wire_payload = to_wire(payload) if use_wire_copy else payload
        self._record_timing(cycle_id, stage_prefix + ".request_copy", encode_started)
        request.payload_json = encode_json(payload) if self.mirror_payloads else ""
        mirror_key = (str(request_topic), int(cycle_id))
        with self._lock:
            mirror_request = bool(self.mirror_markers) and mirror_key not in self._mirrored_cycles
            if mirror_request:
                self._mirrored_cycles.add(mirror_key)
                self._mirrored_cycle_order.append(mirror_key)
                while len(self._mirrored_cycle_order) > 128:
                    old_key = self._mirrored_cycle_order.pop(0)
                    self._mirrored_cycles.discard(old_key)
        if mirror_request:
            server.reserve_local_request(request.requester, request.request_id)
        if mirror_request and component_client is not None:
            component_client.publisher.publish(request)
        response = ComponentPacket()
        response.header = request.header
        response.cycle_id = int(cycle_id)
        response.request_id = int(request.request_id)
        response.requester = str(request.requester)
        response.operation = str(operation)
        try:
            server_payload = from_wire(wire_payload, waypoint_client=server.waypoint_client) if use_wire_copy else wire_payload
            request_forwarded = time.perf_counter()
            self._record_timing(cycle_id, forwarding_prefix + ".request", call_started)
            dispatch_ready = time.perf_counter()
            dispatch_started = time.perf_counter()
            self._record_timing(cycle_id, scheduling_prefix + ".direct_dispatch", dispatch_ready)
            result = server.dispatch(str(operation), server_payload, int(cycle_id), request.header)
            self._record_timing(cycle_id, stage_prefix + ".node_execution", dispatch_started)
            response_copy_started = time.perf_counter()
            wire_result = to_wire(result) if use_wire_copy else result
            self._record_timing(cycle_id, stage_prefix + ".response_copy", response_copy_started)
            response.success = True
            response.reason = ""
            response.payload_json = encode_json(result) if self.mirror_payloads else ""
        except Exception as exc:
            response.success = False
            response.reason = str(exc)
            response.payload_json = encode_json(None)
        if mirror_request:
            server.publisher.publish(response)
        if not bool(response.success):
            raise RuntimeError("{} failed: {}".format(request_topic, response.reason))
        decode_started = time.perf_counter()
        decoded_result = from_wire(wire_result, waypoint_client=waypoint_client) if use_wire_copy else wire_result
        self._record_timing(cycle_id, stage_prefix + ".return_decode", decode_started)
        response_returned = time.perf_counter()
        self._record_timing(cycle_id, forwarding_prefix + ".response", response_copy_started)
        recorder = self.timing_recorder
        if recorder is not None:
            request_forwarding_ms = max(0.0, (request_forwarded - call_started) * 1000.0)
            response_forwarding_ms = max(0.0, (response_returned - response_copy_started) * 1000.0)
            recorder.record_duration(int(cycle_id), forwarding_prefix + ".total", request_forwarding_ms + response_forwarding_ms)
        self._record_timing(cycle_id, stage_prefix + ".total", call_started)
        return decoded_result

    def _record_timing(self, cycle_id: int, stage: str, started: float) -> None:
        """Record transport overhead without changing a component request or result."""
        recorder = self.timing_recorder
        if recorder is not None:
            recorder.record_duration(int(cycle_id), str(stage), (time.perf_counter() - float(started)) * 1000.0)


def cycle_id_from_timestamp(timestamp_s: float) -> int:
    """Use nanoseconds as the stable ID shared by all six nodes for one simulation cycle."""
    return max(0, int(round(float(timestamp_s) * 1000000000.0)))


def fill_header(header, timestamp_s: float) -> None:
    """Fill the common map-frame header carried by every component request."""
    timestamp_s = max(0.0, float(timestamp_s))
    header.frame_id = "map"
    header.stamp.sec = int(timestamp_s)
    header.stamp.nanosec = int(round((timestamp_s - header.stamp.sec) * 1000000000.0))
    if header.stamp.nanosec >= 1000000000:
        header.stamp.sec += 1
        header.stamp.nanosec -= 1000000000


_CLIENT_IDS = itertools.count(1)
_COMPONENT_QOS = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100, reliability=ReliabilityPolicy.RELIABLE)


class PlannerInputFrameCache:
    """Keep a few complete input frames so each stage can match work by cycle ID."""

    def __init__(self, node, topic_name: str = "/cpx/planning/input_frame", max_frames: int = 8, local_bus=None):
        from cpx_interfaces.msg import PlannerInputFrame

        self.max_frames = max(2, int(max_frames))
        self.local_bus = local_bus
        self._frames = {}
        self._order = []
        self._condition = threading.Condition()
        input_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.subscription = node.create_subscription(PlannerInputFrame, str(topic_name), self._receive, input_qos, callback_group=ReentrantCallbackGroup())

    def _receive(self, message):
        cycle_id = int(message.cycle_id)
        with self._condition:
            if cycle_id not in self._frames:
                self._order.append(cycle_id)
            self._frames[cycle_id] = message
            while len(self._order) > self.max_frames:
                old_cycle_id = self._order.pop(0)
                self._frames.pop(old_cycle_id, None)
            self._condition.notify_all()

    def wait_for(self, cycle_id: int, timeout_s: float = 0.5):
        """Return the frame for this cycle, allowing DDS time to deliver it first."""
        if self.local_bus is not None:
            local_frame = self.local_bus.get_input_frame(int(cycle_id))
            if local_frame is not None:
                return local_frame
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while int(cycle_id) not in self._frames:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    return None
                self._condition.wait(timeout=remaining_s)
            return self._frames[int(cycle_id)]


class ComponentClient:
    """Send a component request and wait for its matching result using ROS topics only."""

    def __init__(self, node, request_topic: str, result_topic: str, timeout_s: float = 5.0, cache_operations=None, local_bus=None):
        from cpx_interfaces.msg import ComponentPacket

        self.node = node
        self.request_topic = str(request_topic)
        self.result_topic = str(result_topic)
        self.timeout_s = max(0.1, float(timeout_s))
        self.local_bus = local_bus
        self.active_cycle_id = 0
        self.active_timestamp_s = 0.0
        self._requester = "{}:{}".format(node.get_fully_qualified_name(), next(_CLIENT_IDS))
        self._next_request_id = itertools.count(1)
        self._pending = {}
        self._pending_lock = threading.RLock()
        self._component_ready = False
        self._cache_operations = {str(name) for name in (cache_operations or ())}
        self._response_cache = {}
        self._cache_cycle_id = None
        self.publisher = node.create_publisher(ComponentPacket, self.request_topic, _COMPONENT_QOS)
        self.subscription = node.create_subscription(ComponentPacket, self.result_topic, self._receive_result, _COMPONENT_QOS, callback_group=ReentrantCallbackGroup())

    def _receive_result(self, message):
        if str(message.requester) != self._requester:
            return
        with self._pending_lock:
            pending = self._pending.get(int(message.request_id))
            if pending is None:
                return
            pending["message"] = message
            pending["event"].set()

    def _wait_for_component(self):
        if self._component_ready and self.publisher.get_subscription_count() >= 1 and self.node.count_publishers(self.result_topic) >= 1:
            return
        deadline = time.monotonic() + self.timeout_s
        while self.publisher.get_subscription_count() < 1 or self.node.count_publishers(self.result_topic) < 1:
            if time.monotonic() >= deadline:
                raise TimeoutError("Topic component '{}' is unavailable.".format(self.request_topic))
            if self.node.executor is None:
                rclpy.spin_once(self.node, timeout_sec=0.01)
            else:
                time.sleep(0.01)
        time.sleep(0.02)
        self._component_ready = True

    def call(self, operation: str, payload: Any, *, cycle_id: int = 0, timestamp_s: float = 0.0, waypoint_client=None) -> Any:
        from cpx_interfaces.msg import ComponentPacket

        request_cycle_id = int(cycle_id or self.active_cycle_id)
        request_timestamp_s = float(timestamp_s or self.active_timestamp_s)
        if self.local_bus is not None and self.local_bus.has_server(self.request_topic):
            return self.local_bus.call(self.request_topic, str(operation), payload, cycle_id=request_cycle_id, timestamp_s=request_timestamp_s, waypoint_client=waypoint_client, component_client=self)
        payload_json = encode_json(payload)
        cache_key = None
        if str(operation) in self._cache_operations and request_cycle_id > 0:
            if self._cache_cycle_id != request_cycle_id:
                self._response_cache.clear()
                self._cache_cycle_id = request_cycle_id
            cache_key = (str(operation), payload_json)
            if cache_key in self._response_cache:
                return self._response_cache[cache_key]
        self._wait_for_component()
        request_id = int(next(self._next_request_id))
        request = ComponentPacket()
        fill_header(request.header, request_timestamp_s)
        request.cycle_id = request_cycle_id
        request.request_id = request_id
        request.requester = self._requester
        request.operation = str(operation)
        request.success = True
        request.reason = ""
        request.payload_json = payload_json
        pending = {"event": threading.Event(), "message": None}
        with self._pending_lock:
            self._pending[request_id] = pending
        self.publisher.publish(request)
        deadline = time.monotonic() + self.timeout_s
        while not pending["event"].is_set() and time.monotonic() < deadline:
            if self.node.executor is None:
                rclpy.spin_once(self.node, timeout_sec=min(0.01, max(0.0, deadline - time.monotonic())))
            else:
                pending["event"].wait(timeout=min(0.01, max(0.0, deadline - time.monotonic())))
        with self._pending_lock:
            response = self._pending.pop(request_id, {}).get("message")
        if response is None:
            raise TimeoutError("Topic component '{}' timed out.".format(self.request_topic))
        if not bool(response.success):
            raise RuntimeError("{} failed: {}".format(self.request_topic, str(response.reason)))
        if int(response.cycle_id) != request_cycle_id:
            raise RuntimeError("Stale {} result: expected cycle {}, received {}.".format(self.request_topic, request_cycle_id, int(response.cycle_id)))
        result = decode_json(response.payload_json, waypoint_client=waypoint_client)
        if cache_key is not None:
            self._response_cache[cache_key] = result
        return result


class ComponentServer:
    """Run an existing component method for each request topic and publish its result."""

    def __init__(self, node, request_topic: str, result_topic: str, dispatch, waypoint_client=None, local_bus=None):
        from cpx_interfaces.msg import ComponentPacket

        self.node = node
        self.request_topic = str(request_topic)
        self.local_bus = local_bus
        self.dispatch = dispatch
        self.waypoint_client = waypoint_client
        self._local_reservations = set()
        self._local_reservations_lock = threading.RLock()
        self.publisher = node.create_publisher(ComponentPacket, str(result_topic), _COMPONENT_QOS)
        self.subscription = node.create_subscription(ComponentPacket, str(request_topic), self._receive_request, _COMPONENT_QOS, callback_group=ReentrantCallbackGroup())
        if self.local_bus is not None:
            self.local_bus.register_server(self.request_topic, self)

    def reserve_local_request(self, requester: str, request_id: int) -> None:
        """Mark a request already executed by the composed-process fast path."""
        with self._local_reservations_lock:
            self._local_reservations.add((str(requester), int(request_id)))

    def _consume_local_reservation(self, requester: str, request_id: int) -> bool:
        key = (str(requester), int(request_id))
        with self._local_reservations_lock:
            if key not in self._local_reservations:
                return False
            self._local_reservations.remove(key)
            return True

    def _receive_request(self, request):
        from cpx_interfaces.msg import ComponentPacket

        if self._consume_local_reservation(request.requester, request.request_id):
            return

        response = ComponentPacket()
        response.header = request.header
        response.header.frame_id = "map"
        response.cycle_id = int(request.cycle_id)
        response.request_id = int(request.request_id)
        response.requester = str(request.requester)
        response.operation = str(request.operation)
        try:
            payload = decode_json(request.payload_json, waypoint_client=self.waypoint_client)
            result = self.dispatch(str(request.operation), payload, int(request.cycle_id), request.header)
            response.success = True
            response.reason = ""
            response.payload_json = encode_json(result)
        except Exception as exc:
            response.success = False
            response.reason = str(exc)
            response.payload_json = encode_json(None)
        self.publisher.publish(response)


class RemoteWaypoint:
    """Waypoint-shaped proxy whose traversal operations run inside global_planner_node."""

    def __init__(self, data: Mapping[str, Any], client: ComponentClient | None):
        self._data = dict(data or {})
        self._client = client
        for name, value in self._data.items():
            if name != "lane_width":
                setattr(self, name, value)

    @property
    def lane_width(self):
        return self._data.get("lane_width", self._data.get("lane_width_m"))

    def _step(self, operation: str, distance_m: float | None = None):
        if self._client is None:
            return [] if operation in {"next", "previous"} else None
        embedded_key = "__cpx_{}__".format(operation)
        if operation in {"left", "right"} and embedded_key in self._data:
            embedded = self._data.get(embedded_key)
            return RemoteWaypoint(embedded, self._client) if embedded is not None else None
        if operation in {"next", "previous"}:
            embedded_distance = self._data.get("__cpx_{}_distance_m__".format(operation))
            if embedded_key in self._data and embedded_distance is not None and abs(float(embedded_distance) - float(distance_m)) <= 1.0e-6:
                return [RemoteWaypoint(item, self._client) for item in list(self._data.get(embedded_key) or [])]
        payload = {"waypoint": self.to_dict()}
        if distance_m is not None:
            payload["distance_m"] = float(distance_m)
        result = self._client.call("waypoint_" + operation, payload, waypoint_client=self._client)
        return result

    def left(self):
        return self._step("left")

    def right(self):
        return self._step("right")

    def next(self, distance_m: float):
        return self._step("next", distance_m)

    def previous(self, distance_m: float):
        return self._step("previous", distance_m)

    def to_dict(self):
        return {name: value for name, value in self._data.items() if not str(name).startswith("__cpx_")}

    def to_wire_dict(self):
        """Keep prefetched neighbors only while this waypoint crosses a ROS topic."""
        return dict(self._data)


class RemoteGlobalPlannerProxy:
    """Expose the existing custom-planner method names through global-planner topics."""

    def __init__(self, node, *, request_topic="/cpx/global_planner/request", result_topic="/cpx/global_planner/result", local_map_planner=None, local_bus=None):
        cache_operations = {"get_waypoint", "get_local_lane_context", "waypoint_left", "waypoint_right", "waypoint_next", "waypoint_previous"}
        self.context_client = ComponentClient(node, request_topic, result_topic, cache_operations=cache_operations, local_bus=local_bus)
        self.reference_client = ComponentClient(node, request_topic, result_topic, cache_operations=cache_operations, local_bus=local_bus)
        self.local_map_planner = local_map_planner
        self.local_bus = local_bus
        self.timing_recorder = getattr(node, "timing_recorder", None)
        self.name = str(getattr(local_map_planner, "name", "Town10HD_Opt.xodr"))
        self.blocked_lanes = []
        self.active_cycle_id = 0
        self.active_timestamp_s = 0.0

    def _timed_local_call(self, operation, callback):
        """Measure direct in-process map calls that do not cross the component topic bus."""
        started = time.perf_counter()
        try:
            return callback()
        finally:
            duration_ms = (time.perf_counter() - started) * 1000.0
            recorder = self.timing_recorder or getattr(self.local_bus, "timing_recorder", None)
            if recorder is not None:
                prefix = "component.cpx.global_planner.local.{}".format(operation)
                recorder.record_duration(int(self.active_cycle_id), prefix + ".node_execution", duration_ms)
                recorder.record_duration(int(self.active_cycle_id), prefix + ".total", duration_ms)

    def _timed_cached_local_call(self, operation, cache_key, callback):
        """Reuse an identical static-map answer inside one cycle and record whether it was reused."""
        if self.local_bus is None or int(self.active_cycle_id) <= 0:
            return self._timed_local_call(operation, callback)
        started = time.perf_counter()
        result, cache_hit, execution_ms = self.local_bus.cached_map_call(int(self.active_cycle_id), cache_key, callback)
        total_ms = (time.perf_counter() - started) * 1000.0
        recorder = self.timing_recorder or getattr(self.local_bus, "timing_recorder", None)
        if recorder is not None:
            prefix = "component.cpx.global_planner.local.{}".format(operation)
            recorder.record_duration(int(self.active_cycle_id), prefix + (".cache_hit" if cache_hit else ".cache_miss"), total_ms)
            if not cache_hit:
                recorder.record_duration(int(self.active_cycle_id), prefix + ".node_execution", execution_ms)
            recorder.record_duration(int(self.active_cycle_id), prefix + ".total", total_ms)
        return result

    def _call(self, operation, payload, *, cycle_id=0, timestamp_s=0.0, context=False):
        client = self.context_client if context else self.reference_client
        request_cycle_id = int(cycle_id or self.active_cycle_id)
        request_timestamp_s = float(timestamp_s or self.active_timestamp_s)
        client.active_cycle_id = request_cycle_id
        client.active_timestamp_s = request_timestamp_s
        return client.call(operation, payload, cycle_id=request_cycle_id, timestamp_s=request_timestamp_s, waypoint_client=client)

    def get_waypoint(self, point):
        if self.local_map_planner is not None:
            point_mapping = _point_mapping(point)
            cache_key = ("get_waypoint", float(point_mapping["x"]), float(point_mapping["y"]), float(point_mapping["z"]))
            return self._timed_cached_local_call("get_waypoint", cache_key, lambda: self.local_map_planner.get_waypoint(point_mapping))
        return self._call("get_waypoint", {"point": _point_mapping(point)}, context=True)

    def get_local_lane_context(self, **kwargs):
        if self.local_map_planner is not None:
            return self._timed_local_call("get_local_lane_context", lambda: self.local_map_planner.get_local_lane_context(**kwargs))
        return self._call("get_local_lane_context", kwargs, context=True)

    def plan_route_from_locations(self, **kwargs):
        return self._call("plan_route_from_locations", kwargs)

    def trace_route(self, *args, **kwargs):
        return self._call("trace_route", {"args": list(args), "kwargs": kwargs})

    def get_current_route_info(self, **kwargs):
        return self._call("get_current_route_info", kwargs)

    def get_waypoint_candidates(self, point):
        if self.local_map_planner is not None:
            point_mapping = _point_mapping(point)
            cache_key = ("get_waypoint_candidates", float(point_mapping["x"]), float(point_mapping["y"]), float(point_mapping["z"]))
            return self._timed_cached_local_call("get_waypoint_candidates", cache_key, lambda: self.local_map_planner.get_waypoint_candidates(point_mapping))
        return self._call("get_waypoint_candidates", {"point": _point_mapping(point)}, context=True)

    def get_local_lane_graph(self, x_m, y_m, **kwargs):
        payload = {"x_m": float(x_m), "y_m": float(y_m), **dict(kwargs)}
        if self.local_map_planner is not None:
            return self._timed_local_call("get_local_lane_graph", lambda: self.local_map_planner.get_local_lane_graph(**payload))
        return self._call("get_local_lane_graph", payload, context=True)

    def block_ad_lane_id(self, ad_lane_id):
        result = self._call("block_ad_lane_id", {"ad_lane_id": int(ad_lane_id)})
        if int(ad_lane_id) not in self.blocked_lanes:
            self.blocked_lanes.append(int(ad_lane_id))
        return result

    def block_lane_at_position(self, position):
        return self._call("block_lane_at_position", {"position": _point_mapping(position)})

    def close(self):
        return None


class RemoteRouteManagerProxy:
    """Keep CPXRouteManager state in global_planner_node while preserving its public method names."""

    _PROPERTY_NAMES = {"last_status", "active_route_summary", "carla_route_debug_reason", "carla_route_sync_reason", "carla_route_progress_index"}

    def __init__(self, node, request_topic="/cpx/global_planner/request", result_topic="/cpx/global_planner/result", local_bus=None):
        self.client = ComponentClient(node, request_topic, result_topic, local_bus=local_bus)
        self.active_cycle_id = 0
        self.active_timestamp_s = 0.0

    def __getattr__(self, name):
        if name in self._PROPERTY_NAMES:
            return self.client.call("route_manager_property", {"name": name}, cycle_id=self.active_cycle_id, timestamp_s=self.active_timestamp_s, waypoint_client=self.client)

        def method(*args, **kwargs):
            return self.client.call("route_manager_call", {"name": name, "args": list(args), "kwargs": kwargs}, cycle_id=self.active_cycle_id, timestamp_s=self.active_timestamp_s, waypoint_client=self.client)

        return method


class RemoteMPCProxy:
    """Expose the unchanged MPC object through mpc_node topics."""

    def __init__(self, node, request_topic="/cpx/mpc/request", result_topic="/cpx/mpc/result", local_bus=None):
        self.client = ComponentClient(node, request_topic, result_topic, timeout_s=20.0, local_bus=local_bus)
        self.active_cycle_id = 0
        self.active_timestamp_s = 0.0
        self._load_local_metadata()

    def _load_local_metadata(self):
        """Read immutable MPC dimensions locally so node construction never waits for a topic reply."""
        import yaml

        config_path = Path(__file__).resolve().parent / "MPC" / "mpc.yaml"
        with config_path.open("r", encoding="utf-8") as config_file:
            payload = yaml.safe_load(config_file) or {}
        config = dict(payload.get("mpc", payload) or {})
        constraints = dict(config.get("constraints", {}) or {})
        self.horizon_s = float(config.get("horizon_s", 5.0))
        self.dt_s = float(config.get("plan_dt_s", 0.05))
        self.horizon_steps = max(1, int(round(self.horizon_s / self.dt_s)))
        self.horizon_s = float(self.horizon_steps * self.dt_s)
        self.wheelbase_m = float(config.get("wheelbase_m", 2.7))
        frequency_hz = max(1.0e-3, float(config.get("trajectory_generation_frequency_hz", 2.0)))
        self.trajectory_generation_period_s = 1.0 / frequency_hz
        self.lane_width_m = float(dict(payload.get("road", {}) or {}).get("lane_width_m", 3.5))
        self.adaptive_horizon_enabled = bool(config.get("adaptive_horizon_enabled", False))
        self.active_cost_profile_name = "lane_follow"
        self._last_status = ""
        self.constraints = SimpleNamespace(
            min_velocity_mps=float(constraints.get("min_velocity_mps", 0.0)),
            max_velocity_mps=float(constraints.get("max_velocity_mps", 15.0)),
            min_acceleration_mps2=float(constraints.get("min_acceleration_mps2", -3.0)),
            max_acceleration_mps2=float(constraints.get("max_acceleration_mps2", 3.0)),
            max_jerk_mps3=abs(float(constraints.get("max_jerk_mps3", 10.0))),
            min_steer_rad=float(constraints.get("min_steer_rad", -0.3)),
            max_steer_rad=float(constraints.get("max_steer_rad", 0.3)),
            min_steer_rate_rps=float(constraints.get("min_steer_rate_rps", -0.02)),
            max_steer_rate_rps=float(constraints.get("max_steer_rate_rps", 0.02)),
            enforce_terminal_velocity_constraint=bool(constraints.get("enforce_terminal_velocity_constraint", True)),
            terminal_velocity_mps=float(constraints.get("terminal_velocity_mps", 0.0)),
        )

    def _apply_state(self, state):
        for name, value in dict(state or {}).items():
            if name == "constraints" and isinstance(value, Mapping):
                value = SimpleNamespace(**dict(value))
            setattr(self, name, value)

    def _call(self, operation, payload):
        result = self.client.call(operation, payload, cycle_id=self.active_cycle_id, timestamp_s=self.active_timestamp_s)
        if isinstance(result, Mapping) and "state" in result:
            self._apply_state(result.get("state", {}))
            return result.get("result")
        return result

    def plan_trajectory(self, **kwargs):
        return self._call("final", kwargs)

    def probe_trajectory_feasibility(self, **kwargs):
        return self._call("probe", kwargs)

    def apply_mode_cost_profile(self, profile_name, blend_alpha=None):
        return self._call("apply_mode_cost_profile", {"profile_name": profile_name, "blend_alpha": blend_alpha})

    def blend_toward_horizon_s(self, *args, **kwargs):
        return self._call("blend_toward_horizon_s", {"args": list(args), "kwargs": kwargs})

    def get_runtime_status(self):
        return self._call("get_runtime_status", {})

    def get_last_cost_terms(self):
        return self._call("get_last_cost_terms", {})


class PrebuiltInputAdapter:
    """Return the input frame already built by planner_input_node without rebuilding it."""

    def __init__(self, adapter_output):
        self.adapter_output = adapter_output

    def build(self, **_kwargs):
        return self.adapter_output

    def latest_timestamp_s(self):
        return float(self.adapter_output.frame.planning.sim_time_s)


class DistributedInputAdapter(PrebuiltInputAdapter):
    """Give planner_node both the received adapter output and the original per-cycle raw values."""

    def __init__(self, adapter_output, runtime_inputs):
        super().__init__(adapter_output)
        self._runtime_inputs = dict(runtime_inputs)

    def runtime_inputs(self):
        return dict(self._runtime_inputs)


class _RemoteStageMethod:
    """Present one remote operation with the same method name used by CP-X."""

    def __init__(self, owner, operation: str):
        self.owner = owner
        self.operation = str(operation)

    def update(self, **kwargs):
        return self.owner._call(self.operation + "_update", kwargs)

    def reset(self, **kwargs):
        return self.owner._call(self.operation + "_reset", kwargs)


class RemoteBehaviorContextProxy:
    """Keep traffic-light memory and scenario state in behavior_context_node."""

    def __init__(self, node, local_bus=None):
        self.traffic_memory_client = ComponentClient(node, "/cpx/behavior/context/traffic_memory/request", "/cpx/behavior/context/traffic_memory/result", timeout_s=5.0, local_bus=local_bus)
        self.scenario_client = ComponentClient(node, "/cpx/behavior/context/scenario/request", "/cpx/behavior/context/scenario/result", timeout_s=5.0, local_bus=local_bus)
        self.active_cycle_id = 0
        self.active_timestamp_s = 0.0
        self.traffic_memory = _RemoteStageMethod(self, "traffic_memory")
        self.scenario_manager = _RemoteStageMethod(self, "scenario")

    def _call(self, operation, payload):
        client = self.traffic_memory_client if str(operation).startswith("traffic_memory_") else self.scenario_client
        return client.call(operation, payload, cycle_id=self.active_cycle_id, timestamp_s=self.active_timestamp_s)


class RemoteBehaviorDecisionProxy:
    """Keep the unchanged RuleBasedBehaviorPlanner FSM in behavior_decision_node."""

    def __init__(self, node, request_topic="/cpx/behavior/decision/request", result_topic="/cpx/behavior/decision/result", local_bus=None):
        self.client = ComponentClient(node, request_topic, result_topic, timeout_s=5.0, local_bus=local_bus)
        self.active_cycle_id = 0
        self.active_timestamp_s = 0.0

    def update(self, **kwargs):
        return self.client.call("behavior_update", kwargs, cycle_id=self.active_cycle_id, timestamp_s=self.active_timestamp_s)

    def _reset_lane_change_state(self, **kwargs):
        return self.client.call("behavior_reset_lane_change", kwargs, cycle_id=self.active_cycle_id, timestamp_s=self.active_timestamp_s)


class RemoteReferencePipeline:
    """Keep final reference conditioning and maneuver memory in reference_planner_node."""

    def __init__(self, node, local_bus=None):
        self.maneuver_client = ComponentClient(node, "/cpx/behavior/reference/maneuver/request", "/cpx/behavior/reference/maneuver/result", timeout_s=20.0, local_bus=local_bus)
        self.finalize_client = ComponentClient(node, "/cpx/behavior/reference/finalize/request", "/cpx/behavior/reference/finalize/result", timeout_s=20.0, local_bus=local_bus)
        self.active_cycle_id = 0
        self.active_timestamp_s = 0.0

    def finalize(self, request):
        return self.finalize_client.call("finalize", request, cycle_id=self.active_cycle_id, timestamp_s=self.active_timestamp_s)

    def condition(self, request):
        return self.finalize_client.call("condition", request, cycle_id=self.active_cycle_id, timestamp_s=self.active_timestamp_s)

    def update(self, **kwargs):
        return self.maneuver_client.call("maneuver_update", kwargs, cycle_id=self.active_cycle_id, timestamp_s=self.active_timestamp_s)

    def reset(self, **kwargs):
        return self.maneuver_client.call("maneuver_reset", kwargs, cycle_id=self.active_cycle_id, timestamp_s=self.active_timestamp_s)


def _point_mapping(point: Any) -> Dict[str, float]:
    if isinstance(point, Mapping):
        return {"x": float(point.get("x", 0.0)), "y": float(point.get("y", 0.0)), "z": float(point.get("z", 0.0))}
    return {"x": float(getattr(point, "x", 0.0)), "y": float(getattr(point, "y", 0.0)), "z": float(getattr(point, "z", 0.0))}


def load_planner_configuration(package_root: Path) -> Dict[str, Any]:
    """Load the shared planner configuration used by every separated node."""
    from cpx_planning.utility.config_loader import deep_merge_dicts, load_yaml_file

    planner_payload = load_yaml_file(str(package_root / "config" / "planner.yaml"))
    global_payload = load_yaml_file(str(package_root / "Global_Planner" / "global_planner.yaml"))
    planner_config = deep_merge_dicts(dict(planner_payload.get("planner", planner_payload)), dict(global_payload.get("global_planner", global_payload)))
    planner_config["global_planner_mode"] = "dij"
    planner_config["mpc_config_path"] = str(package_root / "MPC" / "mpc.yaml")
    planner_config["cp_message_path"] = ""
    return planner_config
