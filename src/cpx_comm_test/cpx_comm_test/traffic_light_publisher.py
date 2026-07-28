"""Publish OpenCDA traffic-light output as Autoware light groups."""

import zlib

from autoware_perception_msgs.msg import TrafficLightElement
from autoware_perception_msgs.msg import TrafficLightGroup
from cpx_interfaces.msg import TrafficLightObservation
from cpx_interfaces.msg import TrafficLightObservationArray
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
    """Publish OpenCDA traffic lights without dropping their position."""

    def __init__(self):
        super().__init__("traffic_light_publisher")
        self.publisher = self.create_publisher(
            TrafficLightObservationArray,
            "/cpx/traffic_light",
            10,
        )
        self.receiver = TcpJsonReceiver(5053, self.get_logger())
        self.create_timer(0.02, self.publish_messages)

    def publish_messages(self):
        for data in self.receiver.get_messages():
            payload = data.get("data", data)
            message = TrafficLightObservationArray()
            timestamp_s = data.get(
                "timestamp_s",
                self.get_clock().now().nanoseconds / 1e9,
            )
            _set_stamp(message.header.stamp, timestamp_s)
            message.header.frame_id = str(data.get("frame_id", "map"))

            colors = {
                "red": TrafficLightElement.RED,
                "yellow": TrafficLightElement.AMBER,
                "amber": TrafficLightElement.AMBER,
                "green": TrafficLightElement.GREEN,
                "white": TrafficLightElement.WHITE,
            }

            for item in payload.get("traffic_lights", []):
                observation = TrafficLightObservation()
                observation.source_id = str(item.get("id", ""))
                observation.type = str(item.get("type", "traffic_light"))
                observation.state = str(item.get("state", "unknown"))
                position = item.get("position", {})
                if isinstance(position, dict):
                    observation.position.x = float(position.get("x", 0.0))
                    observation.position.y = float(position.get("y", 0.0))
                    observation.position.z = float(position.get("z", 0.0))

                element = TrafficLightElement()
                element.color = colors.get(
                    observation.state.strip().lower(),
                    TrafficLightElement.UNKNOWN,
                )
                element.shape = TrafficLightElement.CIRCLE
                element.status = TrafficLightElement.SOLID_ON
                observation.confidence = float(item.get("confidence", 1.0))
                element.confidence = observation.confidence

                group = TrafficLightGroup()
                group.traffic_light_group_id = _group_id(
                    observation.source_id
                )
                group.elements = [element]
                observation.signal = group
                message.traffic_lights.append(observation)

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
