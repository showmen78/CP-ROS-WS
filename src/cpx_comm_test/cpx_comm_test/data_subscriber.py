#!/usr/bin/env python3

"""Subscribe to and print all OpenCDA data published through ROS 2."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


TOPICS = {
    "localization": "/cpx/localization",
    "perception": "/cpx/perception",
    "traffic_light": "/cpx/traffic_light",
}


class DataSubscriber(Node):
    """Print localization, perception, and traffic-light messages."""

    def __init__(self):
        super().__init__("opencda_data_subscriber")

        # Node.create_subscription() keeps each subscription internally.
        for data_type, topic_name in TOPICS.items():
            self.create_subscription(
                String,
                topic_name,
                lambda message, name=data_type: self.print_message(
                    name,
                    message,
                ),
                10,
            )

        self.get_logger().info("Listening to all three OpenCDA topics.")

    def print_message(self, data_type, message):
        """Decode one JSON message and print it clearly."""
        try:
            data = json.loads(message.data)
            self.get_logger().info(
                "{} data:\n{}".format(
                    data_type,
                    json.dumps(data, indent=2),
                )
            )
        except ValueError as error:
            self.get_logger().warning(
                "Invalid {} JSON: {}".format(data_type, error)
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
