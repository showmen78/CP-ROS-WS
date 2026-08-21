"""ROS boundary for the copied CP-X planning pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
import math
from pathlib import Path
import threading
import time

from autoware_control_msgs.msg import Control
from autoware_perception_msgs.msg import TrackedObjects
from builtin_interfaces.msg import Time
from cpx_interfaces.msg import CooperativeMessageArray, PlannerInputFrame, TrafficLightObservationArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from cpx_planning.component_interfaces import DistributedInputAdapter, PlannerLocation, RemoteBehaviorContextProxy, RemoteBehaviorDecisionProxy, RemoteGlobalPlannerProxy, RemoteMPCProxy, RemoteReferencePipeline, RemoteRouteManagerProxy, cycle_id_from_timestamp, encode_json, fill_header, load_planner_configuration
from cpx_planning.planner_core.cpx_mpc_planner import CPXMPCPlannerBridge
from cpx_planning.ros_input_adapter import ROSInputAdapter
from cpx_planning.ros_output_adapter import ROSOutputAdapter


def _default_planner_input_log_path():
    """Keep the comparison log in the workspace root for a symlink build."""
    source_path = Path(__file__).resolve()
    for parent in source_path.parents:
        if parent.name == "src":
            return parent.parent / "ros_planner_input_adapter_output.jsonl"
    return Path.cwd() / "ros_planner_input_adapter_output.jsonl"


def _json_safe(value):
    """Convert planner contracts and custom waypoints into JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if value.__class__.__name__ == "Waypoint" and callable(getattr(value, "to_dict", None)):
        return _json_safe(value.to_dict())
    if is_dataclass(value):
        return {field.name: _json_safe(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except Exception:
            pass
    list_method = getattr(value, "tolist", None)
    if callable(list_method):
        try:
            return _json_safe(list_method())
        except Exception:
            pass
    return str(value)


def _time_message(timestamp_s):
    """Use simulation time in both the planner output and the returned TCP control."""
    timestamp_s = max(0.0, float(timestamp_s))
    message = Time()
    message.sec = int(timestamp_s)
    message.nanosec = int(round((timestamp_s - message.sec) * 1000000000.0))
    if message.nanosec >= 1000000000:
        message.sec += 1
        message.nanosec -= 1000000000
    return message


class CPXPlannerNode(Node):
    """Receive raw ROS data and run the same CP-X bridge used by OpenCDA."""

    def __init__(self, local_map_planner=None, local_bus=None):
        super().__init__("planner_node")
        package_root = Path(__file__).resolve().parent
        self.package_root = package_root
        self.local_bus = local_bus
        self.declare_parameter("planner_input_log_path", str(_default_planner_input_log_path()))
        self.declare_parameter("control_topic", "/control/command/control_cmd")
        self.declare_parameter("debug", False)
        self.declare_parameter("planning_period_s", 0.05)
        self.debug = bool(self.get_parameter("debug").value)
        planner_config = load_planner_configuration(package_root)
        planner_config["debug"] = bool(self.debug)
        planner_config["record_debug"] = bool(self.debug)
        planner_config["record_evaluation_metrics"] = bool(self.debug)
        self.global_planner = RemoteGlobalPlannerProxy(self, local_map_planner=local_map_planner, local_bus=local_bus)
        self.route_manager = RemoteRouteManagerProxy(self, local_bus=local_bus)
        self.mpc = RemoteMPCProxy(self, local_bus=local_bus)
        self.planner_bridge = CPXMPCPlannerBridge(vehicle_manager=None, config=planner_config, map_planner=self.global_planner, mpc_instance=self.mpc, route_manager_instance=self.route_manager, behavior_components_enabled=False)
        self.behavior_context = RemoteBehaviorContextProxy(self, local_bus=local_bus)
        self.behavior_decision = RemoteBehaviorDecisionProxy(self, local_bus=local_bus)
        self.reference_proxy = RemoteReferencePipeline(self, local_bus=local_bus)
        self.planner_bridge._full_traffic_memory = self.behavior_context.traffic_memory
        self.planner_bridge._scenario_manager = self.behavior_context.scenario_manager
        self.planner_bridge.behavior_planner = self.behavior_decision
        self.planner_bridge.maneuver_manager = self.reference_proxy
        self.planner_bridge.reference_pipeline = self.reference_proxy
        self.input_adapter = ROSInputAdapter(bridge=self.planner_bridge)
        self.latest_planner_output = None
        self.latest_control_message = None
        self.latest_adapter_output = None
        self.latest_planning_cycle_time_ms = 0.0
        self._last_frame_timestamp_s = None
        self._waiting_message_printed = False
        self._planning_lock = threading.RLock()
        self._planning_callback_group = MutuallyExclusiveCallbackGroup()

        self.ros_output_adapter = ROSOutputAdapter()
        self.control_publisher = self.create_publisher(Control, str(self.get_parameter("control_topic").value), 10)
        self.tcp_control_output_publisher = self.create_publisher(String, "/cpx/planner_control_output", 10)
        self.debug_output_publisher = self.create_publisher(String, "/cpx/debug_output", 10) if self.debug else None
        input_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2, reliability=ReliabilityPolicy.RELIABLE)
        self.input_frame_publisher = self.create_publisher(PlannerInputFrame, "/cpx/planning/input_frame", input_qos)
        self.localization_subscription = self.create_subscription(Odometry, "/cpx/localization", lambda message: self._receive("localization", message), 10)
        self.perception_subscription = self.create_subscription(TrackedObjects, "/cpx/perception", lambda message: self._receive("perception", message), 10)
        self.v2x_subscription = self.create_subscription(TrackedObjects, "/cpx/v2x", lambda message: self._receive("v2x", message), 10)
        self.cp_obstacles_subscription = self.create_subscription(TrackedObjects, "/cpx/cp_obstacles", lambda message: self._receive("cp_obstacles", message), 10)
        self.traffic_light_subscription = self.create_subscription(TrafficLightObservationArray, "/cpx/traffic_light", lambda message: self._receive("traffic_lights", message), 10)
        self.cooperative_subscription = self.create_subscription(CooperativeMessageArray, "/cpx/cooperative_messages", lambda message: self._receive("cooperative", message), 10)
        self.safety_status_subscription = self.create_subscription(String, "/cpx/safety_status", lambda message: self._receive("safety_status", message), 10)
        self.destination_subscription = self.create_subscription(PoseStamped, "/cpx/final_destination", lambda message: self._receive("final_destination", message), 10)

        self.planner_input_log_path = Path(str(self.get_parameter("planner_input_log_path").value)).expanduser().resolve()
        if self.debug:
            self.planner_input_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.planner_input_log_path.write_text("", encoding="utf-8")
        self.create_timer(float(self.get_parameter("planning_period_s").value), self.run_planning_cycle, callback_group=self._planning_callback_group)
        self.get_logger().info("CP-X planner coordinator is waiting for ROS inputs at 20 Hz.")

    def _receive(self, input_name, message):
        """Forward each raw ROS message to the matching input-adapter method."""
        self._apply_message(input_name, message)

    def _apply_message(self, input_name, message):
        """Call the matching ROS adapter update method without changing the data."""
        methods = {
            "localization": self.input_adapter.update_localization,
            "perception": self.input_adapter.update_perception,
            "v2x": self.input_adapter.update_v2x,
            "cp_obstacles": self.input_adapter.update_cp_obstacles,
            "traffic_lights": self.input_adapter.update_traffic_lights,
            "cooperative": self.input_adapter.update_cooperative_messages,
            "safety_status": self.input_adapter.update_safety_status,
            "final_destination": self.input_adapter.update_final_destination,
        }
        methods[input_name](message)

    def run_planning_cycle(self):
        """Run exactly one copied CP-X cycle for each synchronized ROS input frame."""
        with self._planning_lock:
            if self.input_adapter is None or not self.input_adapter.ready():
                if self.debug and not self._waiting_message_printed:
                    self.get_logger().info("Waiting for localization, perception, CP/V2X, traffic-light, safety, and destination data.")
                    self._waiting_message_printed = True
                return
            timestamp_s = float(self.input_adapter.latest_timestamp_s())
            if timestamp_s == self._last_frame_timestamp_s:
                return
            planning_started_monotonic = time.perf_counter()
            cycle_id = cycle_id_from_timestamp(timestamp_s)
            self._set_component_cycle(cycle_id, timestamp_s)
            try:
                runtime_inputs = self.input_adapter.runtime_inputs()
                ego_pose = dict(runtime_inputs["ego_pose"])
                ego_location = PlannerLocation(x=float(ego_pose["x"]), y=float(ego_pose["y"]), z=float(ego_pose.get("z", 0.0)))
                cp_payload = dict(runtime_inputs["cp_payload"] or {})
                object_snapshots = self.planner_bridge._fused_planning_object_snapshots(local_object_snapshots=runtime_inputs["local_object_snapshots"], cp_obstacles=list(cp_payload.get("obstacles", []) or []), ego_location=ego_location, sim_time_s=timestamp_s)
                adapter_output = self.input_adapter.build(ego_location=ego_location, ego_yaw_rad=float(ego_pose["heading_rad"]), ego_speed_mps=float(runtime_inputs["ego_speed_mps"]), object_snapshots=object_snapshots, cp_payload=cp_payload)
                self._publish_input_frame(cycle_id, timestamp_s, adapter_output, runtime_inputs)
                self.planner_bridge._prediction_lane_step_resolved_count = int(getattr(self.planner_bridge, "_prediction_lane_step_resolved_count", 0))
                self.planner_bridge._prediction_lane_step_none_count = int(getattr(self.planner_bridge, "_prediction_lane_step_none_count", 0))
                self.planner_bridge.input_adapter = DistributedInputAdapter(adapter_output, runtime_inputs)
                planner_output = self.planner_bridge.run_step()
            except Exception as exc:
                self.get_logger().error("Could not run the copied CP-X planning cycle: {}".format(exc))
                return
            self._last_frame_timestamp_s = timestamp_s
            self._waiting_message_printed = False
            self.latest_adapter_output = adapter_output
            self.latest_planner_output = planner_output
            if self.debug:
                self.write_planner_input_adapter_output(adapter_output)
                self.publish_debug_output(adapter_output, planner_output)
            control_message = self.ros_output_adapter.build_control_message(planner_output=planner_output, stamp=_time_message(timestamp_s))
            planning_cycle_time_ms = (time.perf_counter() - planning_started_monotonic) * 1000.0
            self.latest_planning_cycle_time_ms = float(planning_cycle_time_ms)
            tcp_control_output = String()
            tcp_control_output.data = json.dumps({"cycle_time_s": float(timestamp_s), "target_speed_mps": float(control_message.longitudinal.velocity), "acceleration_mps2": float(control_message.longitudinal.acceleration), "steering_rad": float(control_message.lateral.steering_tire_angle), "planning_cycle_time_ms": float(planning_cycle_time_ms)}, allow_nan=False, separators=(",", ":"))
            self.tcp_control_output_publisher.publish(tcp_control_output)
            self.control_publisher.publish(control_message)
            self.latest_control_message = control_message
            if self.debug:
                self._log_cycle(adapter_output, planner_output)

    def _set_component_cycle(self, cycle_id, timestamp_s):
        """Give every component request the same simulation-cycle identity."""
        for component in (self.global_planner, self.route_manager, self.mpc, self.behavior_context, self.behavior_decision):
            component.active_cycle_id = int(cycle_id)
            component.active_timestamp_s = float(timestamp_s)
        self.global_planner.context_client.active_cycle_id = int(cycle_id)
        self.global_planner.context_client.active_timestamp_s = float(timestamp_s)
        self.global_planner.reference_client.active_cycle_id = int(cycle_id)
        self.global_planner.reference_client.active_timestamp_s = float(timestamp_s)
        self.reference_proxy.active_cycle_id = int(cycle_id)
        self.reference_proxy.active_timestamp_s = float(timestamp_s)

    def _publish_input_frame(self, cycle_id, timestamp_s, adapter_output, runtime_inputs):
        """Publish the complete existing input contract once per 20 Hz cycle for recording."""
        message = PlannerInputFrame()
        fill_header(message.header, timestamp_s)
        message.cycle_id = int(cycle_id)
        message.sim_time_s = float(timestamp_s)
        message.planner_input_adapter_output_json = encode_json(adapter_output)
        message.runtime_inputs_json = encode_json(runtime_inputs)
        message.metadata_json = encode_json({"v2x_nearby_count": int(runtime_inputs.get("v2x_nearby_count", 0) or 0), "prediction_lane_step_resolved_count": int(getattr(self.planner_bridge, "_prediction_lane_step_resolved_count", 0)), "prediction_lane_step_none_count": int(getattr(self.planner_bridge, "_prediction_lane_step_none_count", 0))})
        if self.local_bus is not None:
            self.local_bus.store_input_frame(message)
        self.input_frame_publisher.publish(message)

    def publish_debug_output(self, adapter_output, planner_output):
        """Publish the complete input/output pair used for shadow comparison."""
        frame = adapter_output.frame
        diagnostics = planner_output.diagnostics.as_dict()
        planning_objects = [{key: item.get(key) for key in ("x", "y", "v", "psi", "length_m", "width_m", "confidence", "track_stale", "prediction_valid")} for item in list(frame.perception.planning_objects or []) if isinstance(item, dict)]
        planning_objects.sort(key=lambda item: (float(item.get("x", 0.0) or 0.0), float(item.get("y", 0.0) or 0.0)))
        reference_xy = [[float(dict(sample).get("x_ref_m", dict(sample).get("x", 0.0))), float(dict(sample).get("y_ref_m", dict(sample).get("y", 0.0)))] for sample in list(planner_output.reference_trajectory or []) if isinstance(sample, dict)]
        planned_trajectory = [[float(value) for value in list(state)[:4]] for state in list(planner_output.planned_trajectory or [])]
        payload = {
            "schema_version": 1,
            "cycle_time_s": float(frame.planning.sim_time_s),
            "planner_input_adapter_output": _json_safe(adapter_output),
            "input": {
                "ego_state": [float(frame.planning.ego.x_m), float(frame.planning.ego.y_m), float(frame.planning.ego.speed_mps), float(frame.planning.ego.heading_rad)],
                "current_lane_id": int(frame.map_lane.lane_id),
                "object_count": int(frame.perception.planning_count),
                "planning_objects": planning_objects,
                "predicted_object_count": int(frame.prediction.predicted_object_count),
                "obstacle_future_trajectories": dict(frame.prediction.obstacle_future_trajectories),
                "v2x_obstacle_count": int(frame.cp_messages.obstacle_count),
                "traffic_signal_state": str(frame.planning.traffic_control.signal_state),
                "traffic_control": _json_safe(frame.planning.traffic_control),
                "prediction_horizon_s": float(frame.prediction.horizon_s),
                "prediction_dt_s": float(frame.prediction.dt_s),
                "route_point_count": len(adapter_output.route_points),
            },
            "behavior": planner_output.behavior_command.as_dict(),
            "mpc": {
                "acceleration_mps2": float(planner_output.acceleration_mps2),
                "steering_rad": float(planner_output.steering_rad),
                "status": str(diagnostics.get("mpc_status", "")),
                "fallback_reason": str(diagnostics.get("mpc_fallback_reason", "")),
                "replan_executed": bool(diagnostics.get("mpc_replan_executed", False)),
            },
            "reference_xy": reference_xy,
            "planned_trajectory": planned_trajectory,
        }
        message = String()
        message.data = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        if self.debug_output_publisher is not None:
            self.debug_output_publisher.publish(message)

    def write_planner_input_adapter_output(self, adapter_output):
        """Append every field of the input contract without changing the planner."""
        record = {"schema_version": 1, "source": "ros", "cycle_time_s": float(adapter_output.frame.planning.sim_time_s), "planner_input_adapter_output": _json_safe(adapter_output)}
        with self.planner_input_log_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def _log_cycle(self, adapter_output, planner_output):
        """Print a short status line for the cycle that was just published."""
        frame = adapter_output.frame
        behavior = planner_output.behavior_command
        self.get_logger().info("CP-X cycle {:.3f}: ego=({:.2f},{:.2f}) lane={} objects={} decision={} fsm={} accel={:.3f} steer={:.4f}".format(float(frame.planning.sim_time_s), float(frame.planning.ego.x_m), float(frame.planning.ego.y_m), int(frame.map_lane.lane_id), int(frame.perception.planning_count), str(behavior.decision), str(behavior.fsm_state), float(planner_output.acceleration_mps2), float(planner_output.steering_rad)))

    def destroy_node(self):
        """Close the copied planner and AD-map runtime before ROS exits."""
        if self.planner_bridge is not None:
            self.planner_bridge.destroy()
        self.global_planner.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CPXPlannerNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
