"""Publish the latest OpenCDA safety-manager status as plain ROS JSON."""

import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .tcp_json_receiver import TcpJsonReceiver
from .timing import configure_timing, publish_transport_timing


class SafetyStatusPublisher(Node):
    """Forward the OpenCDA safety flags without recreating safety logic in ROS."""

    def __init__(self):
        super().__init__("safety_status_publisher")
        self.publisher = self.create_publisher(String, "/cpx/safety_status", 10)
        self.debug_time, self.timing_publisher, self.timing_stream = configure_timing(self, "safety_status")
        self.message_guard = self.create_guard_condition(self.publish_messages)
        self.receiver = TcpJsonReceiver(5056, self.get_logger(), on_message=self.message_guard.trigger)

    def publish_messages(self):
        """Publish every received safety status with its simulation timestamp."""
        for data in self.receiver.get_messages():
            publish_started_ns = time.time_ns()
            payload = data.get("data", data)
            message = String()
            message.data = json.dumps({"timestamp_s": float(data.get("timestamp_s", 0.0)), "status": dict(payload.get("status", {}) or {}), "cycle_send_started_wall_time_ns": int(data.get("cycle_send_started_wall_time_ns", 0) or 0), "tcp_received_wall_time_ns": int(dict(data.get("_timing", {}) or {}).get("tcp_received_wall_time_ns", 0) or 0)}, allow_nan=False, separators=(",", ":"))
            self.publisher.publish(message)
            publish_transport_timing(self.timing_publisher, data, self.timing_stream, publish_started_ns, time.time_ns())

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
