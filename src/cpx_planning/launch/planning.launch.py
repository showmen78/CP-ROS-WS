"""Start the six topic-connected CP-X planning nodes in one fast process."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    debug = LaunchConfiguration("debug")
    debug_time = LaunchConfiguration("debug_time")
    return LaunchDescription([
        DeclareLaunchArgument("debug", default_value="false"),
        DeclareLaunchArgument("debug_time", default_value="false"),
        Node(package="cpx_planning", executable="planning_system", output="screen", parameters=[{"debug": debug, "debug_time": debug_time}]),
    ])
