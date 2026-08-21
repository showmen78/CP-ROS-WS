"""ROS owner for the existing final reference-conditioning pipeline."""

from __future__ import annotations

from pathlib import Path
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from cpx_planning.component_interfaces import ComponentServer, PlannerInputFrameCache, RemoteGlobalPlannerProxy, RemoteMPCProxy, RemoteRouteManagerProxy, load_planner_configuration
from cpx_planning.pipeline.maneuver_manager import ManeuverManager
from cpx_planning.planner_core.cpx_mpc_planner import CPXMPCPlannerBridge


class ReferencePlannerNode(Node):
    """Own final reference conditioning and persistent maneuver geometry."""

    def __init__(self, local_map_planner=None, local_bus=None):
        super().__init__("reference_planner_node")
        self.declare_parameter("debug", False)
        package_root = Path(__file__).resolve().parent
        planner_config = load_planner_configuration(package_root)
        planner_config["debug"] = bool(self.get_parameter("debug").value)
        self.global_planner = RemoteGlobalPlannerProxy(self, local_map_planner=local_map_planner, local_bus=local_bus)
        self.route_manager = RemoteRouteManagerProxy(self, local_bus=local_bus)
        self.mpc = RemoteMPCProxy(self, local_bus=local_bus)
        self.bridge = CPXMPCPlannerBridge(vehicle_manager=None, config=planner_config, map_planner=self.global_planner, mpc_instance=self.mpc, route_manager_instance=self.route_manager, behavior_components_enabled=False)
        self.maneuver_manager = ManeuverManager(planner_config)
        self.input_frames = PlannerInputFrameCache(self, local_bus=local_bus)
        self._reference_lock = threading.RLock()
        self.maneuver_server = ComponentServer(self, "/cpx/behavior/reference/maneuver/request", "/cpx/behavior/reference/maneuver/result", self.dispatch, waypoint_client=self.global_planner.reference_client, local_bus=local_bus)
        self.finalize_server = ComponentServer(self, "/cpx/behavior/reference/finalize/request", "/cpx/behavior/reference/finalize/result", self.dispatch, waypoint_client=self.global_planner.reference_client, local_bus=local_bus)
        self.get_logger().info("Reference planner node owns final conditioning and maneuver-reference memory.")

    def dispatch(self, operation, payload, cycle_id, header):
        """Run the existing reference or maneuver method for the matching cycle."""
        if self.input_frames.wait_for(cycle_id) is None:
            raise RuntimeError("Planner input frame is unavailable for cycle {}.".format(cycle_id))
        with self._reference_lock:
            timestamp_s = float(header.stamp.sec) + float(header.stamp.nanosec) / 1000000000.0
            for component in (self.global_planner, self.route_manager, self.mpc):
                component.active_cycle_id = int(cycle_id)
                component.active_timestamp_s = timestamp_s
            self.global_planner.context_client.active_cycle_id = int(cycle_id)
            self.global_planner.context_client.active_timestamp_s = timestamp_s
            self.global_planner.reference_client.active_cycle_id = int(cycle_id)
            self.global_planner.reference_client.active_timestamp_s = timestamp_s
            if str(operation) == "condition":
                return self.bridge.reference_pipeline.condition(payload)
            if str(operation) == "finalize":
                return self.bridge.reference_pipeline.finalize(payload)
            if str(operation) == "maneuver_update":
                return self.maneuver_manager.update(**dict(payload))
            if str(operation) == "maneuver_reset":
                return self.maneuver_manager.reset(**dict(payload))
            raise ValueError("Unsupported reference operation: {}".format(operation))

def main(args=None):
    rclpy.init(args=args)
    node = ReferencePlannerNode()
    executor = MultiThreadedExecutor(num_threads=4)
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
