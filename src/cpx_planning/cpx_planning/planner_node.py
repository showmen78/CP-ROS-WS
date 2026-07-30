"""ROS node that builds a CP-X PlannerInputFrame from ROS topics."""

from __future__ import annotations

import os
from pathlib import Path

from autoware_perception_msgs.msg import TrackedObjects
from cpx_interfaces.msg import CooperativeMessageArray, TrafficLightObservationArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node

from cpx_planning.behavior_planner.lane_safety import LaneSafetyScorer
from cpx_planning.behavior_planner import RuleBasedBehaviorPlanner
from cpx_planning.pipeline.planner_pipeline import CPXPlanningPipeline

from cpx_planning.pipeline.route_manager import CPXRouteManager
from cpx_planning.pipeline.tracker import CPXObstacleTracker
from cpx_planning.ros_input_adapter import ROSInputAdapter
from cpx_planning.utility.global_planner import CustomGlobalPlannerAdapter

from cpx_planning.MPC import MPC
from cpx_planning.pipeline.control_buffer import MPCControlBuffer
from cpx_planning.pipeline.mpc_feedback import BehaviorMPCFeedback
from cpx_planning.utility.config_loader import load_yaml_file


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
        self.declare_parameter("route_sample_distance_m", 2.0)
        self.declare_parameter("prediction_horizon_s", 3.0)
        self.declare_parameter("prediction_dt_s", 0.2)
        self.declare_parameter("min_front_gap_m", 8.0)
        self.declare_parameter("min_rear_gap_m", 8.0)
        self.declare_parameter("min_ttc_s", 2.0)
        
        
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

        self.route_manager = CPXRouteManager(global_planner=self.global_planner)
        self.tracker = CPXObstacleTracker()
        self.lane_safety_scorer = LaneSafetyScorer()
        self.input_adapter = ROSInputAdapter(
            map_planner=self.global_planner,
            route_manager=self.route_manager,
            tracker=self.tracker,
            lane_safety_scorer=self.lane_safety_scorer,
            prediction_horizon_s=float(self.get_parameter("prediction_horizon_s").value),
            prediction_dt_s=float(self.get_parameter("prediction_dt_s").value),
            min_front_gap_m=float(self.get_parameter("min_front_gap_m").value),
            min_rear_gap_m=float(self.get_parameter("min_rear_gap_m").value),
            min_ttc_s=float(self.get_parameter("min_ttc_s").value),
        )
        
        
        behavior_config = dict(self.behavior_runtime_cfg)
        behavior_config.update({
            "target_speed_mps": float(self.get_parameter("target_speed_mps").value),
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

        self.behavior_planner = RuleBasedBehaviorPlanner(cp_message_path=None, cooperative_message_check_frequency_hz=0.0)
        self.planning_pipeline = CPXPlanningPipeline(
            behavior_planner=self.behavior_planner,
            route_manager=self.route_manager,
            global_planner=self.global_planner,
            mpc=self.mpc,
            control_buffer=self.control_buffer,
            mpc_feedback=self.mpc_feedback,
            behavior_runtime_cfg=self.behavior_runtime_cfg,
            config=behavior_config,
        )

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


        self.create_timer(0.05, self.build_planner_input)
        self.get_logger().info("CP-X planner node is waiting for ROS inputs.")

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
        
        try:
            planner_output = self.planning_pipeline.run_planning_cycle(adapter_output)
        except Exception as error:
            self.get_logger().error("Could not run the planning cycle: {}".format(error))
            return
        
        self.latest_planner_output = planner_output
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
