"""Convert the numeric CP-X planner output into an Autoware ROS control message."""

from autoware_control_msgs.msg import Control

from cpx_planning.pipeline.output import PlannerOutput


class ROSOutputAdapter:
    """Convert PlannerOutput without changing the planner's control values."""

    def build_control_message(self, *, planner_output: PlannerOutput, stamp):
        """Create one Autoware control message from the MPC output."""
        message = Control()

        message.stamp = stamp

        message.lateral.stamp = stamp
        message.lateral.steering_tire_angle = float(planner_output.steering_rad)
        message.lateral.steering_tire_rotation_rate = 0.0
        message.lateral.is_defined_steering_tire_rotation_rate = False

        message.longitudinal.stamp = stamp
        message.longitudinal.velocity = max(0.0, float(planner_output.behavior_command.target_speed_mps))
        message.longitudinal.acceleration = float(planner_output.acceleration_mps2)
        message.longitudinal.jerk = 0.0
        message.longitudinal.is_defined_acceleration = True
        message.longitudinal.is_defined_jerk = False

        return message