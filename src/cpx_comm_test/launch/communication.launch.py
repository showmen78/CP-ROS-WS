"""Start all OpenCDA-to-ROS communication nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Start all data publishers and the subscriber together."""
    debug = LaunchConfiguration("debug")
    return LaunchDescription([
        DeclareLaunchArgument("debug", default_value="false"),
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
            executable="safety_status_publisher",
            output="screen",
        ),
        Node(
            package="cpx_comm_test",
            executable="data_subscriber",
            output="screen",
            parameters=[{"debug": debug}],
        ),
    ])
