#!/usr/bin/env python3

"""Subscribe to and print all OpenCDA data published through ROS 2."""

from autoware_perception_msgs.msg import PredictedObjects
from autoware_perception_msgs.msg import TrafficLightGroupArray
from autoware_perception_msgs.msg import TrackedObjects
from cpx_interfaces.msg import CooperativeMessageArray
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node

from .tcp_json_sender import ros_message_to_dict
from .tcp_json_sender import TcpJsonSender


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

        # Port 5060 carries the typed ROS data back to Python 3.7/OpenCDA.
        self.sender = TcpJsonSender(5060, self.get_logger())
        self.sequence = 0

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
        """Print one typed ROS message and send a JSON copy to OpenCDA."""
        # ROS already formats typed messages in a readable field-by-field form.
        self.get_logger().info(
            "{} data:\n{}".format(data_type, message)
        )

        # Give each forwarded update a number so its order is easy to check.
        self.sequence += 1
        forwarded_data = {
            "schema_version": 1,
            "sequence": self.sequence,
            "timestamp_s": (
                self.get_clock().now().nanoseconds / 1000000000.0
            ),
            "message_type": data_type,
            "data": ros_message_to_dict(message),
        }
        self.sender.send(forwarded_data)

    def destroy_node(self):
        """Close the TCP connection before shutting down the ROS node."""
        self.sender.close()
        super().destroy_node()


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
