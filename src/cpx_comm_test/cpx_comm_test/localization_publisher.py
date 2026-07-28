"""Publish OpenCDA localization output as ROS 2 odometry."""

import math

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node

from .tcp_json_receiver import TcpJsonReceiver


def _set_stamp(stamp, timestamp_s):
    """Convert a floating-point Unix time to a ROS time."""
    seconds = int(timestamp_s)
    nanoseconds = int((float(timestamp_s) - seconds) * 1000000000)
    stamp.sec = seconds
    stamp.nanosec = nanoseconds


class LocalizationPublisher(Node):
    """Receive localization JSON and publish ROS odometry."""

    def __init__(self):
        super().__init__("localization_publisher")
        self.publisher = self.create_publisher(
            Odometry,
            "/cpx/localization",
            10,
        )
        
        # The destination is a separate ROS topic because it is a mission input,
        # not part of the ego odometry message.
        self.destination_publisher = self.create_publisher(
            PoseStamped,
            "/cpx/final_destination",
            10,
        )
        self.receiver = TcpJsonReceiver(5051, self.get_logger())
        self.create_timer(0.02, self.publish_messages)

    def publish_messages(self):
        for data in self.receiver.get_messages():
            payload = data.get("data", data)
            ego_state = payload.get("ego_state", {})

            message = Odometry()
            timestamp_s = data.get(
                "timestamp_s",
                self.get_clock().now().nanoseconds / 1e9,
            )
            _set_stamp(message.header.stamp, timestamp_s)
            message.header.frame_id = str(data.get("frame_id", "map"))
            message.child_frame_id = "base_link"

            message.pose.pose.position.x = float(
                ego_state.get("x", 0.0)
            )
            message.pose.pose.position.y = float(
                ego_state.get("y", 0.0)
            )
            message.pose.pose.position.z = float(
                ego_state.get("z", 0.0)
            )

            yaw = float(ego_state.get("psi", 0.0))
            message.pose.pose.orientation.z = math.sin(yaw / 2.0)
            message.pose.pose.orientation.w = math.cos(yaw / 2.0)
            message.twist.twist.linear.x = float(
                ego_state.get("v", 0.0)
            )

            self.publisher.publish(message)
            
            #publishing the final destination
            # OpenCDA includes the current final destination in the localization TCP
            # payload. Publish it separately so the ROS planner can create its route.
            final_destination = payload.get("final_destination")
            
            
            # OpenCDA includes the current final destination in the localization TCP
            # payload. Publish it separately so the ROS planner can create its route.
            final_destination = payload.get("final_destination")

            if isinstance(final_destination, dict):
                destination_message = PoseStamped()

                destination_message.header.stamp.sec = message.header.stamp.sec
                destination_message.header.stamp.nanosec = (
                    message.header.stamp.nanosec
                )
                destination_message.header.frame_id = message.header.frame_id

                destination_message.pose.position.x = float(
                    final_destination.get("x", 0.0)
                )
                destination_message.pose.position.y = float(
                    final_destination.get("y", 0.0)
                )
                destination_message.pose.position.z = float(
                    final_destination.get("z", 0.0)
                )

                # The global planner currently needs only the goal position.
                destination_message.pose.orientation.w = 1.0

                self.destination_publisher.publish(destination_message)

    def destroy_node(self):
        self.receiver.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
