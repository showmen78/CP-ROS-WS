"""ROS boundary for the copied CP-X planning pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import json
import math
import os
from pathlib import Path

from autoware_control_msgs.msg import Control
from autoware_perception_msgs.msg import TrackedObjects
from builtin_interfaces.msg import Time
from cpx_interfaces.msg import CooperativeMessageArray, TrafficLightObservationArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from cpx_planning.planner_core.cpx_mpc_planner import CPXMPCPlannerBridge
from cpx_planning.ros_input_adapter import ROSInputAdapter
from cpx_planning.ros_output_adapter import ROSOutputAdapter
from cpx_planning.utility.config_loader import deep_merge_dicts, load_yaml_file
from cpx_planning.utility.global_planner import CustomGlobalPlannerAdapter


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

    def __init__(self):
        super().__init__("cpx_planner")
        package_root = Path(__file__).resolve().parent
        self.package_root = package_root
        self.declare_parameter("xodr_path", str(package_root / "Global_Planner" / "maps" / "Town10HD_Opt.xodr"))
        self.declare_parameter("cache_root", str(Path.home() / ".cache" / "cpx_planning" / "global_planner"))
        self.declare_parameter("ad_map_install_root", os.environ.get("GLOBAL_PLANNER_AD_MAP_INSTALL", ""))
        self.declare_parameter("planner_input_log_path", str(_default_planner_input_log_path()))
        self.declare_parameter("control_topic", "/control/command/control_cmd")

        xodr_path = str(self.get_parameter("xodr_path").value)
        if not Path(xodr_path).is_file():
            raise FileNotFoundError("OpenDRIVE map not found: {}".format(xodr_path))
        ad_map_install_root = str(self.get_parameter("ad_map_install_root").value).strip()
        self.global_planner = CustomGlobalPlannerAdapter(xodr_path=xodr_path, cache_root=str(self.get_parameter("cache_root").value), route_sample_distance_m=2.0, ad_map_install_root=ad_map_install_root or None)
        self.global_planner.load()

        planner_config = self._load_planner_configuration()
        self.planner_bridge = CPXMPCPlannerBridge(vehicle_manager=None, config=planner_config, map_planner=self.global_planner)
        self.input_adapter = ROSInputAdapter(bridge=self.planner_bridge)
        self.planner_bridge.input_adapter = self.input_adapter
        self.latest_planner_output = None
        self.latest_control_message = None
        self.latest_adapter_output = None
        self._last_frame_timestamp_s = None
        self._waiting_message_printed = False

        self.ros_output_adapter = ROSOutputAdapter()
        self.control_publisher = self.create_publisher(Control, str(self.get_parameter("control_topic").value), 10)
        self.debug_output_publisher = self.create_publisher(String, "/cpx/debug_output", 10)
        self.localization_subscription = self.create_subscription(Odometry, "/cpx/localization", lambda message: self._receive("localization", message), 10)
        self.perception_subscription = self.create_subscription(TrackedObjects, "/cpx/perception", lambda message: self._receive("perception", message), 10)
        self.v2x_subscription = self.create_subscription(TrackedObjects, "/cpx/v2x", lambda message: self._receive("v2x", message), 10)
        self.cp_obstacles_subscription = self.create_subscription(String, "/cpx/cp_obstacles", lambda message: self._receive("cp_obstacles", message), 10)
        self.traffic_light_subscription = self.create_subscription(TrafficLightObservationArray, "/cpx/traffic_light", lambda message: self._receive("traffic_lights", message), 10)
        self.cooperative_subscription = self.create_subscription(CooperativeMessageArray, "/cpx/cooperative_messages", lambda message: self._receive("cooperative", message), 10)
        self.safety_status_subscription = self.create_subscription(String, "/cpx/safety_status", lambda message: self._receive("safety_status", message), 10)
        self.destination_subscription = self.create_subscription(PoseStamped, "/cpx/final_destination", lambda message: self._receive("final_destination", message), 10)

        self.planner_input_log_path = Path(str(self.get_parameter("planner_input_log_path").value)).expanduser().resolve()
        self.planner_input_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.planner_input_log_path.write_text("", encoding="utf-8")
        self.create_timer(0.02, self.run_planning_cycle)
        self.get_logger().info("CP-X planner node loaded its local configuration and is waiting for ROS inputs.")

    def _load_planner_configuration(self):
        """Load the copied OpenCDA defaults from local ROS package YAML files."""
        planner_payload = load_yaml_file(str(self.package_root / "config" / "planner.yaml"))
        global_payload = load_yaml_file(str(self.package_root / "Global_Planner" / "global_planner.yaml"))
        planner_config = dict(planner_payload.get("planner", planner_payload))
        global_config = dict(global_payload.get("global_planner", global_payload))
        planner_config = deep_merge_dicts(planner_config, global_config)
        planner_config["mpc_config_path"] = str(self.package_root / "MPC" / "mpc.yaml")
        planner_config["cp_message_path"] = ""
        return planner_config

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
        if self.input_adapter is None or not self.input_adapter.ready():
            if not self._waiting_message_printed:
                self.get_logger().info("Waiting for localization, perception, CP/V2X, traffic-light, safety, and destination data.")
                self._waiting_message_printed = True
            return
        timestamp_s = float(self.input_adapter.latest_timestamp_s())
        if timestamp_s == self._last_frame_timestamp_s:
            return
        try:
            planner_output = self.planner_bridge.run_step()
        except Exception as exc:
            self.get_logger().error("Could not run the copied CP-X planning cycle: {}".format(exc))
            return
        adapter_output = self.planner_bridge.last_adapter_output
        if adapter_output is None:
            self.get_logger().error("The copied pipeline did not produce PlannerInputAdapterOutput.")
            return
        self._last_frame_timestamp_s = timestamp_s
        self._waiting_message_printed = False
        self.latest_adapter_output = adapter_output
        self.latest_planner_output = planner_output
        self.write_planner_input_adapter_output(adapter_output)
        self.publish_debug_output(adapter_output, planner_output)
        control_message = self.ros_output_adapter.build_control_message(planner_output=planner_output, stamp=_time_message(timestamp_s))
        self.control_publisher.publish(control_message)
        self.latest_control_message = control_message
        self._log_cycle(adapter_output, planner_output)

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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
