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
from cpx_planning.pipeline.route_manager import CPXRouteManager
from cpx_planning.pipeline.tracker import CPXObstacleTracker
from cpx_planning.ros_input_adapter import ROSInputAdapter
from cpx_planning.utility.global_planner import CustomGlobalPlannerAdapter


class CPXPlannerNode(Node):
    """Receive ROS inputs and build the planner input frame."""

    def __init__(self):
        """Create the custom map planner, input adapter, ROS subscribers, and input-building timer."""
        super().__init__("cpx_planner")

        package_root = Path(__file__).resolve().parent
        default_xodr_path = package_root / "Global_Planner" / "maps" / "Town10HD_Opt.xodr"
        default_cache_root = Path.home() / ".cache" / "cpx_planning" / "global_planner"

        self.declare_parameter("xodr_path", str(default_xodr_path))
        self.declare_parameter("cache_root", str(default_cache_root))
        self.declare_parameter("ad_map_install_root", os.environ.get("GLOBAL_PLANNER_AD_MAP_INSTALL", ""))
        self.declare_parameter("route_sample_distance_m", 2.0)
        self.declare_parameter("prediction_horizon_s", 3.0)
        self.declare_parameter("prediction_dt_s", 0.2)
        self.declare_parameter("min_front_gap_m", 8.0)
        self.declare_parameter("min_rear_gap_m", 8.0)
        self.declare_parameter("min_ttc_s", 2.0)

        xodr_path = str(self.get_parameter("xodr_path").value)
        cache_root = str(self.get_parameter("cache_root").value)
        ad_map_install_root = str(self.get_parameter("ad_map_install_root").value).strip()

        if not Path(xodr_path).is_file():
            raise FileNotFoundError("OpenDRIVE map not found: {}".format(xodr_path))

        self.global_planner = CustomGlobalPlannerAdapter(
            xodr_path=xodr_path,
            cache_root=cache_root,
            route_sample_distance_m=float(self.get_parameter("route_sample_distance_m").value),
            ad_map_install_root=ad_map_install_root or None,
        )
        self.global_planner.load()

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

        self.latest_adapter_output = None
        self._last_frame_timestamp_s = None
        self._waiting_message_printed = False

        # Step 6 only builds the input. It does not run behavior or MPC.
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
