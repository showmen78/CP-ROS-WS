"""ROS node that builds a CP-X PlannerInputFrame from ROS topics."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import os
from pathlib import Path
import json
import math
from collections.abc import Mapping
from std_msgs.msg import String

from autoware_perception_msgs.msg import TrackedObjects
from autoware_control_msgs.msg import Control
from cpx_interfaces.msg import CooperativeMessageArray, TrafficLightObservationArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node

from cpx_planning.behavior_planner.lane_safety import LaneSafetyScorer
from cpx_planning.behavior_planner import RuleBasedBehaviorPlanner
from cpx_planning.planner_core.cpx_mpc_planner import CPXMPCPlannerBridge

from cpx_planning.pipeline.route_manager import CPXRouteManager
from cpx_planning.pipeline.tracker import CPXObstacleTracker
from cpx_planning.ros_input_adapter import ROSInputAdapter
from cpx_planning.utility.global_planner import CustomGlobalPlannerAdapter

from cpx_planning.MPC import MPC
from cpx_planning.pipeline.control_buffer import MPCControlBuffer
from cpx_planning.pipeline.mpc_feedback import BehaviorMPCFeedback
from cpx_planning.utility.config_loader import load_yaml_file
from cpx_planning.ros_output_adapter import ROSOutputAdapter


def _default_planner_input_log_path():
    """Keep the ROS input log in the workspace root when this is a source or symlink build."""
    source_path = Path(__file__).resolve()
    for parent in source_path.parents:
        if parent.name == "src":
            return parent.parent / "ros_planner_input_adapter_output.jsonl"
    return Path.cwd() / "ros_planner_input_adapter_output.jsonl"


def _json_safe(value):
    """Convert the complete adapter output, including its custom waypoint, into JSON-safe values."""
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


class CPXPlannerNode(Node):
    """Receive ROS inputs and build the planner input frame."""

    def __init__(self):
        """Create the custom map planner, input adapter, ROS subscribers, and input-building timer."""
        super().__init__("cpx_planner")

        package_root = Path(__file__).resolve().parent
        default_xodr_path = package_root / "Global_Planner" / "maps" / "Town10HD_Opt.xodr"
        default_cache_root = Path.home() / ".cache" / "cpx_planning" / "global_planner"
        default_mpc_config_path = package_root / "MPC" / "mpc.yaml"

        self.declare_parameter("xodr_path", str(default_xodr_path))
        self.declare_parameter("cache_root", str(default_cache_root))
        self.declare_parameter("ad_map_install_root", os.environ.get("GLOBAL_PLANNER_AD_MAP_INSTALL", ""))
        self.declare_parameter("planner_input_log_path", str(_default_planner_input_log_path()))
        self.declare_parameter("route_sample_distance_m", 2.0)
        self.declare_parameter("route_reached_distance_m", 3.0)
        self.declare_parameter("route_stale_lateral_m", 12.0)
        self.declare_parameter("min_front_gap_m", 8.0)
        self.declare_parameter("prediction_min_ttc_s", 2.0)
        self.declare_parameter("communication_range_m", 80.0)
        self.declare_parameter("cooperative_message_check_frequency_hz", 5.0)
        self.declare_parameter("tracker_max_stale_s", 0.5)
        self.declare_parameter("tracker_max_speed_mps", 45.0)
        self.declare_parameter("tracker_max_acceleration_mps2", 12.0)
        self.declare_parameter("tracker_max_position_jump_m", 12.0)
        
        
        self.declare_parameter("mpc_config_path", str(default_mpc_config_path))
        self.declare_parameter("lane_count", 3)
        self.declare_parameter("lane_width_m", 3.5)
        self.declare_parameter("max_mpc_obstacles", 4)
        self.declare_parameter("control_buffer_enabled", True)
        self.declare_parameter("control_buffer_max_reuse_s", 0.35)
        self.declare_parameter("mpc_feedback_enabled", True)
        self.declare_parameter("mpc_feedback_hold_s", 1.5)
        self.declare_parameter("mpc_feedback_min_failures", 1)
        
        self.declare_parameter("target_speed_mps", 10.0)
        self.declare_parameter("ego_max_deceleration_mps2", 3.0)
        self.declare_parameter("full_traffic_unknown_hold_s", 0.25)
        self.declare_parameter("full_traffic_green_confirm_s", 0.5)
        self.declare_parameter("full_lane_change_start_lock_s", 8.0)
        self.declare_parameter("full_dense_traffic_lane_change_lock_enabled", True)
        self.declare_parameter("full_dense_traffic_object_count", 8)
        self.declare_parameter("full_dense_traffic_risky_lane_count", 2)
        self.declare_parameter("full_prepare_lane_change_reference_lock", True)
        self.declare_parameter("full_allow_opportunistic_lane_change", False)
        self.declare_parameter("route_lane_change_preparation_start_distance_m", 45.0)
        self.declare_parameter("route_lane_change_latest_start_distance_m", 12.0)
        self.declare_parameter("route_lane_change_target_safety_threshold", 0.65)
        self.declare_parameter("route_lane_change_require_adjacent", True)
        self.declare_parameter("mpc_feedback_candidate_weight", 80.0)
        self.declare_parameter("full_intersection_turn_speed_cap_mps", 2.2)
        self.declare_parameter("full_intersection_turn_prepare_speed_cap_mps", 2.2)
        self.declare_parameter("full_intersection_turn_prepare_enabled", False)
        self.declare_parameter("full_intersection_turn_latch_enabled", True)
        self.declare_parameter("full_intersection_turn_latch_hold_s", 6.0)
        self.declare_parameter("full_latched_virtual_stop_distance_m", 12.0)
        self.declare_parameter("full_stop_guard_destination_forward_m", 6.0)
        
        
        self.declare_parameter("full_candidate_pipeline_enabled", True)
        self.declare_parameter("full_candidate_reference_min_object_distance_m", 2.0)
        self.declare_parameter("strict_decision_ownership_enabled", True)

        copied_pipeline_defaults = {
            "lookahead_m": 18.0,
            "safety_supervisor_enabled": True,
            "safety_max_steer_delta": 0.25,
            "safety_max_throttle_delta": 0.45,
            "safety_max_brake_delta": 0.60,
            "full_reference_memory_enabled": True,
            "full_reference_memory_max_first_jump_m": 0.85,
            "full_reference_memory_max_destination_jump_m": 2.0,
            "full_reference_memory_max_reuse_age_s": 0.8,
            "full_trajectory_memory_enabled": True,
            "full_trajectory_memory_max_accel_jump_mps2": 0.9,
            "full_trajectory_memory_max_steer_jump_rad": 0.08,
            "full_trajectory_memory_blend_alpha": 0.35,
            "full_trajectory_memory_max_reuse_age_s": 0.5,
            "overspeed_guard_enabled": True,
            "overspeed_margin_mps": 0.5,
            "overspeed_brake_gain": 0.10,
            "overspeed_min_brake": 0.2,
            "overspeed_max_brake": 0.65,
            "low_speed_lateral_recovery_enabled": True,
            "low_speed_lateral_recovery_speed_mps": 0.8,
            "low_speed_lateral_recovery_threshold_m": 1.2,
            "low_speed_lateral_recovery_target_speed_mps": 1.0,
            "low_speed_lateral_recovery_max_steer_rad": 0.18,
            "full_low_speed_launch_enabled": True,
            "full_low_speed_launch_speed_mps": 0.35,
            "full_low_speed_launch_min_accel_mps2": 0.8,
            "full_low_speed_launch_ramp_enabled": True,
            "full_low_speed_launch_ramp_speed_mps": 0.25,
            "full_low_speed_launch_ramp_trigger_accel_mps2": 0.6,
            "full_low_speed_launch_max_accel_mps2": 1.0,
            "full_low_speed_launch_stuck_s": 0.8,
            "full_low_speed_launch_stuck_distance_m": 0.15,
            "full_lane_follow_max_destination_lateral_m": 1.2,
            "full_lane_follow_max_reference_first_lateral_m": 0.65,
            "full_stop_max_destination_lateral_m": 1.0,
            "full_stop_max_reference_first_lateral_m": 0.55,
            "full_mpc_reference_stabilizer_enabled": True,
            "full_reference_stabilizer_min_forward_m": -0.25,
            "full_reference_stabilizer_min_spacing_m": 0.35,
            "full_reference_stabilizer_max_heading_step_rad": 0.75,
            "full_stop_reference_buffer_m": 1.5,
            "full_stop_reference_decel_mps2": 2.0,
            "full_stop_reference_speed_cap_mps": 2.0,
            "reference_contract_lane_follow_min_first_forward_m": 0.5,
            "reference_contract_lane_follow_max_first_lateral_abs_m": 0.75,
            "reference_contract_lane_follow_max_destination_lane_error_m": 0.75,
            "reference_contract_lane_follow_max_destination_body_lateral_abs_m": 1.5,
            "reference_contract_stop_min_first_forward_m": 0.2,
            "reference_contract_stop_max_first_lateral_abs_m": 0.75,
            "reference_contract_stop_max_destination_lane_error_m": 0.75,
            "reference_contract_stop_max_destination_body_lateral_abs_m": 1.5,
            "strict_lane_follow_reference": False,
            "strict_reference_validator_veto_enabled": True,
            "strict_explicit_fallback_speed_mps": 0.8,
            "opencda_style_reference_conditioning_enabled": True,
            "opencda_style_reference_min_node_spacing_m": 0.45,
            "opencda_style_reference_ego_anchor_forward_m": 0.35,
            "opencda_style_reference_max_lateral_accel_mps2": 3.0,
            "opencda_style_reference_min_speed_mps": 0.6,
            "opencda_style_turn_min_speed_mps": 0.45,
            "opencda_style_turn_speed_cap_mps": 1.35,
            "red_yellow_stop_guard_buffer_m": 1.5,
            "red_yellow_stop_guard_comfort_decel_mps2": 1.6,
            "red_yellow_stop_guard_approach_speed_mps": 1.5,
            "red_yellow_stop_guard_speed_kp": 0.8,
            "red_yellow_stop_guard_approach_max_accel_mps2": 0.45,
            "red_yellow_stop_guard_max_decel_mps2": 2.2,
        }
        for parameter_name, default_value in copied_pipeline_defaults.items():
            if not self.has_parameter(parameter_name):
                self.declare_parameter(parameter_name, default_value)
        
        self.declare_parameter("control_topic", "/control/command/control_cmd")

        xodr_path = str(self.get_parameter("xodr_path").value)
        cache_root = str(self.get_parameter("cache_root").value)
        ad_map_install_root = str(self.get_parameter("ad_map_install_root").value).strip()
        
        mpc_config_path = str(self.get_parameter("mpc_config_path").value)
        mpc_payload = load_yaml_file(mpc_config_path)
        mpc_cfg = dict(mpc_payload.get("mpc", mpc_payload))
        road_cfg = dict(mpc_payload.get("road", {}))
        road_cfg.setdefault("lane_count", int(self.get_parameter("lane_count").value))
        road_cfg.setdefault("lane_width_m", float(self.get_parameter("lane_width_m").value))
        
        self.latest_planner_output = None
        self.latest_control_message = None

        if not Path(xodr_path).is_file():
            raise FileNotFoundError("OpenDRIVE map not found: {}".format(xodr_path))

        self.global_planner = CustomGlobalPlannerAdapter(
            xodr_path=xodr_path,
            cache_root=cache_root,
            route_sample_distance_m=float(self.get_parameter("route_sample_distance_m").value),
            ad_map_install_root=ad_map_install_root or None,
        )
        self.global_planner.load()
        
        self.mpc = MPC(mpc_cfg=mpc_cfg, road_cfg=road_cfg)
        self.behavior_runtime_cfg = dict(mpc_cfg.get("behavior_planner_runtime", {}))
        self.control_buffer = MPCControlBuffer(enabled=bool(self.get_parameter("control_buffer_enabled").value), replan_period_s=float(self.mpc.trajectory_generation_period_s), max_reuse_s=float(self.get_parameter("control_buffer_max_reuse_s").value))
        self.mpc_feedback = BehaviorMPCFeedback(enabled=bool(self.get_parameter("mpc_feedback_enabled").value), hold_s=float(self.get_parameter("mpc_feedback_hold_s").value), min_failures=int(self.get_parameter("mpc_feedback_min_failures").value))

        self.route_manager = CPXRouteManager(global_planner=self.global_planner, reached_distance_m=float(self.get_parameter("route_reached_distance_m").value), stale_route_lateral_m=float(self.get_parameter("route_stale_lateral_m").value))
        self.tracker = CPXObstacleTracker(max_stale_s=float(self.get_parameter("tracker_max_stale_s").value), max_speed_mps=float(self.get_parameter("tracker_max_speed_mps").value), max_acceleration_mps2=float(self.get_parameter("tracker_max_acceleration_mps2").value), max_position_jump_m=float(self.get_parameter("tracker_max_position_jump_m").value))
        self.lane_safety_scorer = LaneSafetyScorer()
        self.input_adapter = ROSInputAdapter(
            map_planner=self.global_planner,
            route_manager=self.route_manager,
            tracker=self.tracker,
            lane_safety_scorer=self.lane_safety_scorer,
            mpc=self.mpc,
            min_front_gap_m=float(self.get_parameter("min_front_gap_m").value),
            min_ttc_s=float(self.get_parameter("prediction_min_ttc_s").value),
            communication_range_m=float(self.get_parameter("communication_range_m").value),
        )
        
        
        behavior_config = dict(self.behavior_runtime_cfg)
        behavior_config.update({name: self.get_parameter(name).value for name in copied_pipeline_defaults})
        behavior_config.update({
            "target_speed_mps": float(self.get_parameter("target_speed_mps").value),
            "min_front_gap_m": float(self.get_parameter("min_front_gap_m").value),
            "ego_max_deceleration_mps2": float(self.get_parameter("ego_max_deceleration_mps2").value),
            "full_traffic_unknown_hold_s": float(self.get_parameter("full_traffic_unknown_hold_s").value),
            "full_traffic_green_confirm_s": float(self.get_parameter("full_traffic_green_confirm_s").value),
            "full_lane_change_start_lock_s": float(self.get_parameter("full_lane_change_start_lock_s").value),
            "full_dense_traffic_lane_change_lock_enabled": bool(self.get_parameter("full_dense_traffic_lane_change_lock_enabled").value),
            "full_dense_traffic_object_count": int(self.get_parameter("full_dense_traffic_object_count").value),
            "full_dense_traffic_risky_lane_count": int(self.get_parameter("full_dense_traffic_risky_lane_count").value),
            "full_prepare_lane_change_reference_lock": bool(self.get_parameter("full_prepare_lane_change_reference_lock").value),
            "full_allow_opportunistic_lane_change": bool(self.get_parameter("full_allow_opportunistic_lane_change").value),
            "route_lane_change_preparation_start_distance_m": float(self.get_parameter("route_lane_change_preparation_start_distance_m").value),
            "route_lane_change_latest_start_distance_m": float(self.get_parameter("route_lane_change_latest_start_distance_m").value),
            "route_lane_change_target_safety_threshold": float(self.get_parameter("route_lane_change_target_safety_threshold").value),
            "route_lane_change_require_adjacent": bool(self.get_parameter("route_lane_change_require_adjacent").value),
            "mpc_feedback_candidate_weight": float(self.get_parameter("mpc_feedback_candidate_weight").value),
            "full_intersection_turn_speed_cap_mps": float(self.get_parameter("full_intersection_turn_speed_cap_mps").value),
            "full_intersection_turn_prepare_speed_cap_mps": float(self.get_parameter("full_intersection_turn_prepare_speed_cap_mps").value),
            "full_intersection_turn_prepare_enabled": bool(self.get_parameter("full_intersection_turn_prepare_enabled").value),
            "full_intersection_turn_latch_enabled": bool(self.get_parameter("full_intersection_turn_latch_enabled").value),
            "full_intersection_turn_latch_hold_s": float(self.get_parameter("full_intersection_turn_latch_hold_s").value),
            "full_latched_virtual_stop_distance_m": float(self.get_parameter("full_latched_virtual_stop_distance_m").value),
            "full_stop_guard_destination_forward_m": float(self.get_parameter("full_stop_guard_destination_forward_m").value),
            "max_mpc_obstacles": int(self.get_parameter("max_mpc_obstacles").value),

            "full_candidate_pipeline_enabled": bool(self.get_parameter("full_candidate_pipeline_enabled").value),
            "full_candidate_reference_min_object_distance_m": float(self.get_parameter("full_candidate_reference_min_object_distance_m").value),
            "strict_decision_ownership_enabled": bool(self.get_parameter("strict_decision_ownership_enabled").value),
                
        })

        self.behavior_planner = RuleBasedBehaviorPlanner(cp_message_path=None, cooperative_message_check_frequency_hz=float(self.get_parameter("cooperative_message_check_frequency_hz").value))
        self.planning_pipeline = CPXMPCPlannerBridge(
            behavior_planner=self.behavior_planner,
            route_manager=self.route_manager,
            global_planner=self.global_planner,
            mpc=self.mpc,
            control_buffer=self.control_buffer,
            mpc_feedback=self.mpc_feedback,
            behavior_runtime_cfg=self.behavior_runtime_cfg,
            config=behavior_config,
        )
        
        
        control_topic = str(self.get_parameter("control_topic").value)
        self.ros_output_adapter = ROSOutputAdapter()
        self.control_publisher = self.create_publisher(Control, control_topic, 10)
        self.shadow_output_publisher = self.create_publisher(String, "/cpx/planner_shadow_output", 10)

        self.localization_subscription = self.create_subscription(
            Odometry, "/cpx/localization", self.input_adapter.update_localization, 10
        )
        self.perception_subscription = self.create_subscription(
            TrackedObjects, "/cpx/perception", self.input_adapter.update_perception, 10
        )
        self.v2x_subscription = self.create_subscription(TrackedObjects, "/cpx/v2x", self.input_adapter.update_v2x, 10)
        self.traffic_light_subscription = self.create_subscription(
            TrafficLightObservationArray, "/cpx/traffic_light", self.input_adapter.update_traffic_lights, 10
        )
        self.cooperative_subscription = self.create_subscription(
            CooperativeMessageArray,
            "/cpx/cooperative_messages",
            self.input_adapter.update_cooperative_messages,
            10,
        )
        self.destination_subscription = self.create_subscription(
            PoseStamped, "/cpx/final_destination", self.input_adapter.update_final_destination, 10
        )
        
        # Step 7 builds the planner input and runs behavior planning. Reference generation and MPC are added in Step 8.
        self.latest_adapter_output = None
        self.latest_behavior_command = None
        self.latest_destination_state = None
        self.latest_lane_center_reference = []
        self.latest_reference_debug = {}
        self._last_frame_timestamp_s = None
        self._waiting_message_printed = False

        self.planner_input_log_path = Path(str(self.get_parameter("planner_input_log_path").value)).expanduser().resolve()
        self.planner_input_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.planner_input_log_path.open("w", encoding="utf-8"):
            pass
        self.get_logger().info("ROS PlannerInputAdapterOutput log: {}".format(self.planner_input_log_path))


        self.create_timer(0.05, self.build_planner_input)
        self.get_logger().info("CP-X planner node is waiting for ROS inputs.")
        
        
    def publish_shadow_output(self, adapter_output, planner_output):
        """Send the ROS planner input summary and output back for comparison."""
        frame = adapter_output.frame
        diagnostics = planner_output.diagnostics.as_dict()

        object_ids = sorted(
            str(dict(item).get("id", dict(item).get("vehicle_id", "")))
            for item in list(frame.perception.planning_objects or [])
            if isinstance(item, dict)
        )
        planning_objects = []
        for item in list(frame.perception.planning_objects or []):
            if not isinstance(item, dict):
                continue
            planning_objects.append({key: item.get(key) for key in ("x", "y", "v", "psi", "length_m", "width_m", "confidence", "track_stale", "prediction_valid")})
        planning_objects.sort(key=lambda item: (float(item.get("x", 0.0) or 0.0), float(item.get("y", 0.0) or 0.0)))

        reference_xy = [
            [
                float(dict(sample).get("x_ref_m", dict(sample).get("x", 0.0))),
                float(dict(sample).get("y_ref_m", dict(sample).get("y", 0.0))),
            ]
            for sample in list(planner_output.reference_trajectory or [])
            if isinstance(sample, dict)
        ]

        planned_trajectory = [
            [float(value) for value in list(state)[:4]]
            for state in list(planner_output.planned_trajectory or [])
        ]

        payload = {
            "schema_version": 1,
            "cycle_time_s": float(frame.planning.sim_time_s),
            "planner_input_adapter_output": _json_safe(adapter_output),
            "input": {
                "ego_state": [
                    float(frame.planning.ego.x_m),
                    float(frame.planning.ego.y_m),
                    float(frame.planning.ego.speed_mps),
                    float(frame.planning.ego.heading_rad),
                ],
                "current_lane_id": int(frame.map_lane.lane_id),
                "object_ids": object_ids,
                "object_count": int(frame.perception.planning_count),
                "planning_objects": planning_objects,
                "predicted_object_count": int(frame.prediction.predicted_object_count),
                "obstacle_future_trajectories": dict(frame.prediction.obstacle_future_trajectories),
                "v2x_obstacle_count": int(frame.cp_messages.obstacle_count),
                "traffic_signal_state": str(frame.planning.traffic_control.signal_state),
                "traffic_control": {
                    "signal_state": str(frame.planning.traffic_control.signal_state),
                    "source": str(frame.planning.traffic_control.source),
                    "control_id": str(frame.planning.traffic_control.control_id),
                    "provider_source": str(frame.planning.traffic_control.provider_source),
                    "from_cp": bool(frame.planning.traffic_control.from_cp),
                    "confidence": float(frame.planning.traffic_control.confidence),
                    "ego_passed_stop_line": bool(frame.planning.traffic_control.ego_passed_stop_line),
                    "stop_target": frame.planning.traffic_control.stop_target.as_dict(),
                },
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
        self.shadow_output_publisher.publish(message)

    def write_planner_input_adapter_output(self, adapter_output):
        """Append every field of one ROS PlannerInputAdapterOutput without changing the planning cycle."""
        record = {
            "schema_version": 1,
            "source": "ros",
            "cycle_time_s": float(adapter_output.frame.planning.sim_time_s),
            "planner_input_adapter_output": _json_safe(adapter_output),
        }
        with self.planner_input_log_path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")

    def build_planner_input(self):
        """Build and print one new PlannerInputFrame after all required ROS inputs have arrived."""
        if not self.input_adapter.ready():
            if not self._waiting_message_printed:
                self.get_logger().info("Waiting for localization, perception, V2X, traffic-light, and destination data.")
                self._waiting_message_printed = True
            return

        try:
            adapter_output = self.input_adapter.build()
        except Exception as error:
            self.get_logger().error("Could not build PlannerInputFrame: {}".format(error))
            return

        timestamp_s = float(adapter_output.frame.planning.sim_time_s)

        # The timer may run more frequently than localization updates, so do not print the same frame twice.
        if timestamp_s == self._last_frame_timestamp_s:
            return

        self._last_frame_timestamp_s = timestamp_s
        self.latest_adapter_output = adapter_output
        frame = adapter_output.frame
        self.write_planner_input_adapter_output(adapter_output)
        
        try:
            planner_output = self.planning_pipeline._run_full_cpx_pipeline_step(adapter_output)
        except Exception as error:
            self.get_logger().error("Could not run the planning cycle: {}".format(error))
            return
        
        self.latest_planner_output = planner_output
        self.publish_shadow_output(adapter_output, planner_output)
        control_message = self.ros_output_adapter.build_control_message(planner_output=planner_output, stamp=self.get_clock().now().to_msg())
        self.control_publisher.publish(control_message)
        self.latest_control_message = control_message
        
        
        
        
        behavior_command = planner_output.behavior_command.as_dict()
        destination_state = list(self.planning_pipeline.last_destination_state or [])
        lane_center_reference = [dict(sample) for sample in planner_output.reference_trajectory]
        reference_debug = planner_output.diagnostics.as_dict()

        self.latest_behavior_command = behavior_command
        self.latest_destination_state = list(destination_state)
        self.latest_lane_center_reference = [dict(sample) for sample in lane_center_reference]
        self.latest_reference_debug = dict(reference_debug)

        self.get_logger().info(
            "PlannerInputFrame: ego=({:.2f}, {:.2f}), speed={:.2f}, lane={}, objects={}, predictions={}, "
            "v2x={}, lane_events={}, signal={}, route_points={}".format(
                frame.planning.ego.x_m,
                frame.planning.ego.y_m,
                frame.planning.ego.speed_mps,
                frame.map_lane.lane_id,
                frame.perception.planning_count,
                frame.prediction.predicted_object_count,
                frame.cp_messages.obstacle_count,
                frame.cp_messages.lane_closure_count,
                frame.planning.traffic_control.signal_state,
                len(adapter_output.route_points),
            )
        )
    
        self.get_logger().info(
            "================================================================\n Behavior: decision={}, target_lane={}, fsm={}, target_speed={:.2f}, scenario={} \n==========================================================================".format(
                behavior_command.get("decision", "lane_follow"),
                behavior_command.get("target_lane_id", 0),
                behavior_command.get("lc_state", "LANE_KEEP"),
                float(behavior_command.get("target_speed_mps", 0.0)),
                behavior_command.get("scenario_state", ""),
            )
)
        
        self.get_logger().info("Reference: points={}, destination=({:.2f}, {:.2f}), target_speed={:.2f}, source={}, fallback={}".format(len(lane_center_reference), float(destination_state[0]), float(destination_state[1]), float(destination_state[2]), reference_debug.get("reference_source", ""), reference_debug.get("fallback_reason", "")))
        self.get_logger().info("MPC: status={}, trajectory_points={}, acceleration={:.3f} m/s^2, steering={:.4f} rad, fallback={}".format(reference_debug.get("mpc_status", ""), len(planner_output.planned_trajectory), float(planner_output.acceleration_mps2), float(planner_output.steering_rad), reference_debug.get("mpc_fallback_reason", "")))
   
    def destroy_node(self):
        """Close the AD-map runtime before ROS destroys this node."""
        self.global_planner.close()
        super().destroy_node()


def main(args=None):
    """Start the planner node and keep it running until ROS or the user stops it."""
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
