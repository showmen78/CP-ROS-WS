#!/usr/bin/env python3

"""Subscribe to ROS data, print it, and forward planner output to OpenCDA."""

import json
import time

from autoware_control_msgs.msg import Control
from autoware_perception_msgs.msg import TrackedObjects
from cpx_interfaces.msg import CooperativeMessageArray
from cpx_interfaces.msg import TrafficLightObservationArray
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from .tcp_json_sender import ros_message_to_dict
from .tcp_json_sender import TcpJsonSender
from .timing import configure_timing


# Keeping the topic name and its message type together makes it easy to add
# another input later without copying the subscription code.
TOPICS = {
    "localization": (Odometry, "/cpx/localization"),
    "perception": (TrackedObjects, "/cpx/perception"),
    "traffic_light": (
        TrafficLightObservationArray,
        "/cpx/traffic_light",
    ),
    "v2x": (TrackedObjects, "/cpx/v2x"),
    "cp_obstacles": (TrackedObjects, "/cpx/cp_obstacles"),
    "cooperative_messages": (
        CooperativeMessageArray,
        "/cpx/cooperative_messages",
    ),
    "safety_status": (String, "/cpx/safety_status"),
    "debug_output": (String, "/cpx/debug_output"),
    "planner_control": (Control, "/control/command/control_cmd"),
    "planner_control_output": (String, "/cpx/planner_control_output"),
}


class DataSubscriber(Node):
    """Print ROS data and send planner results back to OpenCDA over TCP."""

    def __init__(self):
        super().__init__("opencda_data_subscriber")
        self.declare_parameter("debug", False)
        self.debug = bool(self.get_parameter("debug").value)
        self.debug_time, self.timing_publisher, self.timing_stream = configure_timing(self, "output_forwarder")

        # Port 5060 carries the typed ROS data back to Python 3.7/OpenCDA.
        self.sender = TcpJsonSender(5060, self.get_logger())
        self.sequence = 0

        # Normal operation only forwards control. Extra subscriptions are for debugging.
        active_topics = TOPICS if self.debug else {"planner_control_output": TOPICS["planner_control_output"]}
        for data_type, (message_class, topic_name) in active_topics.items():
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

        self.get_logger().info("Listening to OpenCDA input topics and ROS planner output topics.")

    def print_message(self, data_type, message):
        """Print one typed ROS message and send a JSON copy to OpenCDA."""
        output_topic_received_wall_time_ns = time.time_ns()
        # ROS already formats typed messages in a readable field-by-field form.
        if self.debug:
            if data_type == "debug_output":
                self.get_logger().info("ROS planner debug output received.")
            elif data_type == "planner_control":
                self.get_logger().info("ROS planner control received: acceleration={:.3f} m/s^2, steering={:.3f} rad.".format(message.longitudinal.acceleration, message.lateral.steering_tire_angle))
            elif data_type == "planner_control_output":
                try:
                    compact_control = json.loads(str(message.data))
                    self.get_logger().info("ROS planner timed control received: acceleration={:.3f} m/s^2, steering={:.3f} rad, cycle={:.3f} ms.".format(float(compact_control["acceleration_mps2"]), float(compact_control["steering_rad"]), float(compact_control["planning_cycle_time_ms"])))
                except (KeyError, TypeError, ValueError):
                    self.get_logger().warning("Invalid timed planner control received.")
            else:
                self.get_logger().info("{} data:\n{}".format(data_type, message))

        # Give each forwarded update a number so its order is easy to check.
        self.sequence += 1
        forwarded_data = {
            "schema_version": 1,
            "sequence": self.sequence,
            "timestamp_s": self.get_clock().now().nanoseconds / 1000000000.0,
            "message_type": data_type,
            "data": ros_message_to_dict(message),
        }

        # Send a small, direct control payload so OpenCDA can later convert it to carla.VehicleControl.
        if data_type == "planner_control_output":
            try:
                compact_control = json.loads(str(message.data))
                cycle_time_s = float(compact_control["cycle_time_s"])
                compact_control["ros_output_subscriber_received_wall_time_ns"] = int(output_topic_received_wall_time_ns)
                forwarded_data["message_type"] = "planner_control"
                forwarded_data["data"] = compact_control
                forwarded_data["timestamp_s"] = cycle_time_s
            except (KeyError, TypeError, ValueError):
                return

        # Keep the comparison stream and the real control stream available at the same time.
        if data_type in {"debug_output", "planner_control_output"}:
            if data_type == "planner_control_output":
                output_tcp_send_started_wall_time_ns = time.time_ns()
                forwarded_data["data"]["ros_output_tcp_send_started_wall_time_ns"] = int(output_tcp_send_started_wall_time_ns)
            else:
                output_tcp_send_started_wall_time_ns = 0
            self.sender.send(forwarded_data)
            if self.timing_publisher is not None and data_type == "planner_control_output":
                output_tcp_sent_wall_time_ns = time.time_ns()
                event = String()
                event.data = json.dumps({"cycle_id": max(0, int(round(float(forwarded_data["timestamp_s"]) * 1000000000.0))), "stream": "output_forwarder", "output_topic_received_wall_time_ns": int(output_topic_received_wall_time_ns), "output_tcp_send_started_wall_time_ns": int(output_tcp_send_started_wall_time_ns), "output_tcp_sent_wall_time_ns": int(output_tcp_sent_wall_time_ns), "output_topic_to_tcp_send_ms": (output_tcp_sent_wall_time_ns - output_topic_received_wall_time_ns) / 1000000.0, "output_callback_to_tcp_send_started_ms": (output_tcp_send_started_wall_time_ns - output_topic_received_wall_time_ns) / 1000000.0, "output_tcp_send_duration_ms": (output_tcp_sent_wall_time_ns - output_tcp_send_started_wall_time_ns) / 1000000.0}, allow_nan=False, separators=(",", ":"))
                self.timing_publisher.publish(event)

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
