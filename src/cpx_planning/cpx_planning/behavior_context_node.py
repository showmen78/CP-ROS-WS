"""ROS owner for traffic-light memory and high-level scenario context."""

from __future__ import annotations

from pathlib import Path
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from cpx_planning.component_interfaces import ComponentServer, PlannerInputFrameCache, load_planner_configuration
from cpx_planning.pipeline.scenario_manager import CPXScenarioManager
from cpx_planning.pipeline.traffic_light_memory import TrafficLightMemory


class BehaviorContextNode(Node):
    """Own state used to understand traffic signals and the current scenario."""

    def __init__(self, local_bus=None):
        super().__init__("behavior_context_node")
        self.declare_parameter("debug", False)
        self.debug = bool(self.get_parameter("debug").value)
        config = load_planner_configuration(Path(__file__).resolve().parent)
        self.traffic_memory = TrafficLightMemory(hold_unknown_s=float(config.get("full_traffic_unknown_hold_s", 1.5)), hold_green_unknown_s=float(config.get("full_traffic_green_unknown_hold_s", 0.25)), green_confirm_s=float(config.get("full_traffic_green_confirm_s", 0.15)), hold_stop_unknown_until_green=bool(config.get("full_traffic_hold_stop_unknown_until_green", False)))
        self.scenario_manager = CPXScenarioManager(config)
        self.input_frames = PlannerInputFrameCache(self, local_bus=local_bus)
        self._lock = threading.RLock()
        self.traffic_memory_server = ComponentServer(self, "/cpx/behavior/context/traffic_memory/request", "/cpx/behavior/context/traffic_memory/result", self.dispatch, local_bus=local_bus)
        self.scenario_server = ComponentServer(self, "/cpx/behavior/context/scenario/request", "/cpx/behavior/context/scenario/result", self.dispatch, local_bus=local_bus)
        self.get_logger().info("Behavior context node owns traffic-light memory and scenario state.")

    def dispatch(self, operation, payload, cycle_id, _header):
        """Run the original context component method for one matching input cycle."""
        if self.input_frames.wait_for(cycle_id) is None:
            raise RuntimeError("Planner input frame is unavailable for cycle {}.".format(cycle_id))
        with self._lock:
            if operation == "traffic_memory_update":
                result = self.traffic_memory.update(**dict(payload))
            elif operation == "scenario_update":
                result = self.scenario_manager.update(**dict(payload))
            elif operation == "scenario_reset":
                result = self.scenario_manager.reset()
            else:
                raise ValueError("Unsupported behavior-context operation: {}".format(operation))
        if self.debug:
            self.get_logger().info("cycle={} operation={}".format(cycle_id, operation))
        return result


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorContextNode()
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
