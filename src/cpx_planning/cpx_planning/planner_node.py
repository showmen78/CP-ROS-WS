"""ROS node that will run the CP-X planner."""

import rclpy
from rclpy.node import Node


class CPXPlannerNode(Node):
    """Main ROS node for the planning module."""

    def __init__(self):
        super().__init__("cpx_planner")
        self.get_logger().info("CP-X planning package is ready.")


def main(args=None):
    rclpy.init(args=args)
    node = CPXPlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()