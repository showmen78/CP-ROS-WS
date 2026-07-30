"""Start all OpenCDA-to-ROS communication nodes."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Start all data publishers and the subscriber together."""
    return LaunchDescription([
        Node(
            package="cpx_comm_test",
            executable="localization_publisher",
            output="screen",
        ),
        Node(
            package="cpx_comm_test",
            executable="perception_publisher",
            output="screen",
        ),
        Node(
            package="cpx_comm_test",
            executable="traffic_light_publisher",
            output="screen",
        ),
        Node(
            package="cpx_comm_test",
            executable="v2x_publisher",
            output="screen",
        ),
        Node(
            package="cpx_comm_test",
            executable="cooperative_message_publisher",
            output="screen",
        ),
        Node(
            package="cpx_comm_test",
            executable="data_subscriber",
            output="screen",
        ),
    ])