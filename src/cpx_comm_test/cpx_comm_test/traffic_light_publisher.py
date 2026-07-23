"""Publish OpenCDA traffic-light output on a ROS 2 topic."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .tcp_json_receiver import TcpJsonReceiver


class TrafficLightPublisher(Node):
    """Receive traffic-light JSON on port 5053 and publish it."""

    def __init__(self):
        super().__init__("traffic_light_publisher")
        self.publisher = self.create_publisher(
            String,
            "/cpx/traffic_light",
            10,
        )
        self.receiver = TcpJsonReceiver(5053, self.get_logger())
        self.create_timer(0.02, self.publish_messages)

    def publish_messages(self):
        for data in self.receiver.get_messages():
            message = String()
            message.data = json.dumps(data)
            self.publisher.publish(message)

    def destroy_node(self):
        self.receiver.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
