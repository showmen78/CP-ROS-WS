"""Publish the latest OpenCDA safety-manager status as plain ROS JSON."""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .tcp_json_receiver import TcpJsonReceiver


class SafetyStatusPublisher(Node):
    """Forward the OpenCDA safety flags without recreating safety logic in ROS."""

    def __init__(self):
        super().__init__("safety_status_publisher")
        self.publisher = self.create_publisher(String, "/cpx/safety_status", 10)
        self.receiver = TcpJsonReceiver(5056, self.get_logger())
        self.create_timer(0.02, self.publish_messages)

    def publish_messages(self):
        """Publish every received safety status with its simulation timestamp."""
        for data in self.receiver.get_messages():
            payload = data.get("data", data)
            message = String()
            message.data = json.dumps({"timestamp_s": float(data.get("timestamp_s", 0.0)), "status": dict(payload.get("status", {}) or {})}, allow_nan=False, separators=(",", ":"))
            self.publisher.publish(message)

    def destroy_node(self):
        self.receiver.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SafetyStatusPublisher()
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
