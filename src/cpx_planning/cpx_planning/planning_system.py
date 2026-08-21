"""Run the six topic-connected CP-X ROS nodes in one Python process."""

import rclpy
from rclpy.executors import MultiThreadedExecutor

from cpx_planning.behavior_context_node import BehaviorContextNode
from cpx_planning.behavior_decision_node import BehaviorDecisionNode
from cpx_planning.component_interfaces import LocalComponentBus
from cpx_planning.global_planner_node import GlobalPlannerNode
from cpx_planning.mpc_node import MPCNode
from cpx_planning.planner_node import CPXPlannerNode
from cpx_planning.reference_planner_node import ReferencePlannerNode


def main(args=None):
    """Keep the ROS node boundaries while avoiding slow cross-process DDS round trips."""
    rclpy.init(args=args)
    local_bus = LocalComponentBus()
    global_planner_node = GlobalPlannerNode(local_bus=local_bus)
    mpc_node = MPCNode(local_bus=local_bus)
    behavior_context_node = BehaviorContextNode(local_bus=local_bus)
    behavior_decision_node = BehaviorDecisionNode(local_bus=local_bus)
    reference_planner_node = ReferencePlannerNode(local_map_planner=global_planner_node.map_planner, local_bus=local_bus)
    planner_node = CPXPlannerNode(local_map_planner=global_planner_node.map_planner, local_bus=local_bus)
    local_bus.mirror_markers = bool(planner_node.debug)
    local_bus.mirror_payloads = bool(planner_node.debug)
    nodes = [global_planner_node, mpc_node, behavior_context_node, behavior_decision_node, reference_planner_node, planner_node]
    executor = MultiThreadedExecutor(num_threads=16)
    for node in nodes:
        executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        for node in reversed(nodes):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
