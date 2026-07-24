"""Publish CP lane events and traffic controls as typed ROS messages."""

import zlib

from autoware_perception_msgs.msg import TrafficLightElement
from autoware_perception_msgs.msg import TrafficLightGroup
from cpx_interfaces.msg import CooperativeMessageArray
from cpx_interfaces.msg import LaneEvent
from cpx_interfaces.msg import TrafficControl
import rclpy
from rclpy.node import Node

from .tcp_json_receiver import TcpJsonReceiver


# These tables turn the simple text received from OpenCDA into the
# numbered values used by the ROS message types.
EVENT_TYPES = {
    "lane_closure": LaneEvent.LANE_CLOSURE,
    "hazard": LaneEvent.ROAD_HAZARD,
    "road_hazard": LaneEvent.ROAD_HAZARD,
    "work_zone": LaneEvent.WORK_ZONE,
}

SIGNAL_COLORS = {
    "red": TrafficLightElement.RED,
    "yellow": TrafficLightElement.AMBER,
    "amber": TrafficLightElement.AMBER,
    "green": TrafficLightElement.GREEN,
    "white": TrafficLightElement.WHITE,
}


def _set_time(message_time, timestamp):
    """Copy seconds represented as a float into a ROS Time field."""
    value = max(0.0, float(timestamp or 0.0))

    # ROS keeps the whole seconds and the small fraction in separate fields.
    # For example, 20.5 becomes 20 seconds and 500,000,000 nanoseconds.
    message_time.sec = int(value)
    message_time.nanosec = int((value - int(value)) * 1000000000)


def _set_duration(duration, seconds):
    """Copy seconds represented as a float into a ROS Duration field."""
    value = max(0.0, float(seconds or 0.0))

    # Duration uses the same two-part format as a ROS timestamp.
    duration.sec = int(value)
    duration.nanosec = int((value - int(value)) * 1000000000)


def _stable_group_id(value):
    """Convert a string or integer control ID into a stable integer."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(zlib.crc32(str(value).encode("utf-8")))


def _fill_position(point, raw_position):
    """Copy either a dictionary or an [x, y, z] list into a ROS Point."""
    if isinstance(raw_position, dict):
        point.x = float(raw_position.get("x", 0.0))
        point.y = float(raw_position.get("y", 0.0))
        point.z = float(raw_position.get("z", 0.0))
        return

    if isinstance(raw_position, (list, tuple)):
        if len(raw_position) > 0:
            point.x = float(raw_position[0])
        if len(raw_position) > 1:
            point.y = float(raw_position[1])
        if len(raw_position) > 2:
            point.z = float(raw_position[2])


class CooperativeMessagePublisher(Node):
    """Receive CP JSON and publish lane-event and traffic-control arrays."""

    def __init__(self):
        super().__init__("cooperative_message_publisher")

        # Both lane events and traffic controls travel together on this topic.
        self.publisher = self.create_publisher(
            CooperativeMessageArray,
            "/cpx/cooperative_messages",
            10,
        )

        # Port 5055 is reserved for cooperative messages from OpenCDA.
        self.receiver = TcpJsonReceiver(5055, self.get_logger())

        # Check the TCP queue every 20 milliseconds without blocking ROS.
        self.create_timer(0.02, self.publish_messages)

    def publish_messages(self):
        """Convert every waiting CP JSON payload and publish it."""
        for data in self.receiver.get_messages():
            # The transmitter may wrap the useful part inside a "data" field.
            payload = data.get("data", data)

            # One ROS message represents one complete cooperative update.
            message = CooperativeMessageArray()
            timestamp = float(
                payload.get(
                    "timestamp_s",
                    data.get(
                        "timestamp_s",
                        self.get_clock().now().nanoseconds / 1e9,
                    ),
                )
            )

            _set_time(message.header.stamp, timestamp)
            message.header.frame_id = str(data.get("frame_id", "map"))

            # The version explains the message layout. The sequence tells us
            # which update came first and helps us notice a missing update.
            message.schema_version = int(
                payload.get("schema_version", 1)
            )
            message.sequence = int(payload.get("sequence", 0))

            # Convert every lane event into the small custom LaneEvent format.
            for item in payload.get("lane_events", []):
                event = LaneEvent()
                event.id = str(item.get("id", ""))
                event.event_type = EVENT_TYPES.get(
                    str(item.get("type", "")).strip().lower(),
                    LaneEvent.UNKNOWN,
                )
                event.source = str(item.get("source", ""))
                _set_time(
                    event.source_stamp,
                    item.get("timestamp_s", timestamp),
                )
                _set_duration(event.ttl, item.get("ttl_s", 0.0))

                # The planner can use this position to find the affected lane.
                _fill_position(event.position, item.get("position", []))
                event.confidence = float(item.get("confidence", 1.0))
                message.lane_events.append(event)

            # The current CP payload calls this list "control". We also accept
            # "traffic_controls" so the JSON can use the clearer ROS name.
            controls = payload.get(
                "traffic_controls",
                payload.get("control", []),
            )
            for item in controls:
                control = TrafficControl()
                control.id = str(item.get("id", ""))
                control.source = str(item.get("source", ""))
                _set_time(
                    control.source_stamp,
                    item.get("timestamp_s", timestamp),
                )
                _set_duration(control.ttl, item.get("ttl_s", 0.0))
                control.confidence = float(
                    item.get("confidence", 1.0)
                )

                # Reuse Autoware's traffic-light format inside our custom
                # wrapper instead of defining another red/yellow/green format.
                signal = TrafficLightGroup()
                signal.traffic_light_group_id = _stable_group_id(
                    item.get("control_id", item.get("id", 0))
                )

                element = TrafficLightElement()
                state = str(
                    item.get(
                        "signal_state",
                        item.get("state", ""),
                    )
                ).strip().lower()
                element.color = SIGNAL_COLORS.get(
                    state,
                    TrafficLightElement.UNKNOWN,
                )
                element.shape = TrafficLightElement.CIRCLE
                element.status = TrafficLightElement.SOLID_ON
                element.confidence = control.confidence
                signal.elements = [element]
                control.signal = signal
                message.traffic_controls.append(control)

            # Subscribers receive the lane and traffic-control update together.
            self.publisher.publish(message)

    def destroy_node(self):
        self.receiver.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CooperativeMessagePublisher()
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
