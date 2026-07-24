"""Publish OpenCDA traffic-light output as Autoware light groups."""

import zlib

from autoware_perception_msgs.msg import TrafficLightElement
from autoware_perception_msgs.msg import TrafficLightGroup
from autoware_perception_msgs.msg import TrafficLightGroupArray
import rclpy
from rclpy.node import Node

from .tcp_json_receiver import TcpJsonReceiver


def _set_stamp(stamp, timestamp_s):
    """Convert a floating-point Unix time to a ROS time."""
    seconds = int(timestamp_s)
    nanoseconds = int((float(timestamp_s) - seconds) * 1000000000)
    stamp.sec = seconds
    stamp.nanosec = nanoseconds


def _group_id(value):
    """Return a stable integer traffic-light group ID."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(zlib.crc32(str(value).encode("utf-8")))


class TrafficLightPublisher(Node):
    """Receive traffic-light JSON and publish Autoware light groups."""

    def __init__(self):
        super().__init__("traffic_light_publisher")
        self.publisher = self.create_publisher(
            TrafficLightGroupArray,
            "/cpx/traffic_light",
            10,
        )
        self.receiver = TcpJsonReceiver(5053, self.get_logger())
        self.create_timer(0.02, self.publish_messages)

    def publish_messages(self):
        for data in self.receiver.get_messages():
            payload = data.get("data", data)
            message = TrafficLightGroupArray()
            timestamp_s = data.get(
                "timestamp_s",
                self.get_clock().now().nanoseconds / 1e9,
            )
            _set_stamp(message.stamp, timestamp_s)

            colors = {
                "red": TrafficLightElement.RED,
                "yellow": TrafficLightElement.AMBER,
                "amber": TrafficLightElement.AMBER,
                "green": TrafficLightElement.GREEN,
                "white": TrafficLightElement.WHITE,
            }

            for item in payload.get("traffic_lights", []):
                element = TrafficLightElement()
                element.color = colors.get(
                    str(item.get("state", "")).lower(),
                    TrafficLightElement.UNKNOWN,
                )
                element.shape = TrafficLightElement.CIRCLE
                element.status = TrafficLightElement.SOLID_ON
                element.confidence = float(
                    item.get("confidence", 1.0)
                )

                group = TrafficLightGroup()
                group.traffic_light_group_id = _group_id(
                    item.get("id", 0)
                )
                group.elements = [element]
                message.traffic_light_groups.append(group)

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
