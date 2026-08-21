"""ROS owner for the unchanged CP-X rule-based behavior state machine."""

from __future__ import annotations

from pathlib import Path
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from cpx_planning.behavior_planner.planner import RuleBasedBehaviorPlanner
from cpx_planning.component_interfaces import ComponentServer, PlannerInputFrameCache, load_planner_configuration


class BehaviorDecisionNode(Node):
    """Own lane-change, stop, yield and reroute behavior state across cycles."""

    def __init__(self, local_bus=None):
        super().__init__("behavior_decision_node")
        self.declare_parameter("debug", False)
        self.debug = bool(self.get_parameter("debug").value)
        config = load_planner_configuration(Path(__file__).resolve().parent)
        self.behavior_planner = RuleBasedBehaviorPlanner(cp_message_path=str(config.get("cp_message_path", "")), cooperative_message_check_frequency_hz=float(config.get("cooperative_message_check_frequency_hz", 5.0)))
        self.input_frames = PlannerInputFrameCache(self, local_bus=local_bus)
        self._lock = threading.RLock()
        self.server = ComponentServer(self, "/cpx/behavior/decision/request", "/cpx/behavior/decision/result", self.dispatch, local_bus=local_bus)
        self.get_logger().info("Behavior decision node owns the unchanged CP-X behavior FSM.")

    def dispatch(self, operation, payload, cycle_id, _header):
        """Run the original behavior-planner method for one matching input cycle."""
        if self.input_frames.wait_for(cycle_id) is None:
            raise RuntimeError("Planner input frame is unavailable for cycle {}.".format(cycle_id))
        with self._lock:
            if operation == "behavior_update":
                result = self.behavior_planner.update(**dict(payload))
            elif operation == "behavior_reset_lane_change":
                result = self.behavior_planner._reset_lane_change_state(**dict(payload))
            else:
                raise ValueError("Unsupported behavior-decision operation: {}".format(operation))
        if self.debug:
            self.get_logger().info("cycle={} operation={}".format(cycle_id, operation))
        return result


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorDecisionNode()
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
