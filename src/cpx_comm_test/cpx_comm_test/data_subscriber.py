#!/usr/bin/env python3

"""Subscribe to and print all OpenCDA data published through ROS 2."""

from autoware_perception_msgs.msg import PredictedObjects
from autoware_perception_msgs.msg import TrafficLightGroupArray
from autoware_perception_msgs.msg import TrackedObjects
from cpx_interfaces.msg import CooperativeMessageArray
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node


# Keeping the topic name and its message type together makes it easy to add
# another input later without copying the subscription code.
TOPICS = {
    "localization": (Odometry, "/cpx/localization"),
    "perception": (PredictedObjects, "/cpx/perception"),
    "traffic_light": (
        TrafficLightGroupArray,
        "/cpx/traffic_light",
    ),
    "v2x": (TrackedObjects, "/cpx/v2x"),
    "cooperative_messages": (
        CooperativeMessageArray,
        "/cpx/cooperative_messages",
    ),
}


class DataSubscriber(Node):
    """Print every OpenCDA data stream published through ROS."""

    def __init__(self):
        super().__init__("opencda_data_subscriber")

        # Create one subscription for every data stream listed above.
        for data_type, (message_class, topic_name) in TOPICS.items():
            self.create_subscription(
                message_class,
                topic_name,
                # Remember the matching name so the printed output is clear.
                lambda message, name=data_type: self.print_message(
                    name,
                    message,
                ),
                # Keep up to ten messages if this node is briefly busy.
                10,
            )

        self.get_logger().info("Listening to all five OpenCDA topics.")

    def print_message(self, data_type, message):
        """Print one typed ROS message."""
        # ROS already formats typed messages in a readable field-by-field form.
        self.get_logger().info(
            "{} data:\n{}".format(data_type, message)
        )


def main(args=None):
    rclpy.init(args=args)
    node = DataSubscriber()
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
