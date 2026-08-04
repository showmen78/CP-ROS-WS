#!/usr/bin/env python3

"""Subscribe to ROS data, print it, and forward planner output to OpenCDA."""

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
    "cooperative_messages": (
        CooperativeMessageArray,
        "/cpx/cooperative_messages",
    ),
    "planner_shadow_output": (String, "/cpx/planner_shadow_output"),
    "planner_control": (Control, "/control/command/control_cmd"),
}


class DataSubscriber(Node):
    """Print ROS data and send planner results back to OpenCDA over TCP."""

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

        self.get_logger().info("Listening to OpenCDA input topics and ROS planner output topics.")

    def print_message(self, data_type, message):
        """Print one typed ROS message and send a JSON copy to OpenCDA."""
        # ROS already formats typed messages in a readable field-by-field form.
        if data_type == "planner_shadow_output":
            self.get_logger().info("ROS planner shadow output received.")
        elif data_type == "planner_control":
            self.get_logger().info("ROS planner control received: acceleration={:.3f} m/s^2, steering={:.3f} rad.".format(message.longitudinal.acceleration, message.lateral.steering_tire_angle))
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
        if data_type == "planner_control":
            forwarded_data["data"] = {
                "target_speed_mps": float(message.longitudinal.velocity),
                "acceleration_mps2": float(message.longitudinal.acceleration),
                "steering_rad": float(message.lateral.steering_tire_angle),
            }

        # Keep the comparison stream and the real control stream available at the same time.
        if data_type in {"planner_shadow_output", "planner_control"}:
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
