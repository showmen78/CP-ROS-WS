"""Publish OpenCDA perception output as Autoware tracked objects."""

import math
import time
import uuid

from autoware_perception_msgs.msg import ObjectClassification
from autoware_perception_msgs.msg import Shape
from autoware_perception_msgs.msg import TrackedObject
from autoware_perception_msgs.msg import TrackedObjectKinematics
from autoware_perception_msgs.msg import TrackedObjects
import rclpy
from rclpy.node import Node

from .tcp_json_receiver import TcpJsonReceiver
from .timing import configure_timing, publish_transport_timing


def _set_stamp(stamp, timestamp_s):
    """Convert a floating-point Unix time to a ROS time."""
    seconds = int(timestamp_s)
    nanoseconds = int((float(timestamp_s) - seconds) * 1000000000)
    stamp.sec = seconds
    stamp.nanosec = nanoseconds


def _classification_label(type_name):
    """Convert an OpenCDA type name to an Autoware class."""
    name = str(type_name).lower()
    if "pedestrian" in name or "walker" in name:
        return ObjectClassification.PEDESTRIAN
    if "motorcycle" in name:
        return ObjectClassification.MOTORCYCLE
    if "bicycle" in name or "bike" in name:
        return ObjectClassification.BICYCLE
    if "truck" in name:
        return ObjectClassification.TRUCK
    if "trailer" in name:
        return ObjectClassification.TRAILER
    if "bus" in name:
        return ObjectClassification.BUS
    if "vehicle" in name or "car" in name:
        return ObjectClassification.CAR
    return ObjectClassification.UNKNOWN


class PerceptionPublisher(Node):
    """Receive perception JSON and publish tracked objects."""

    def __init__(self):
        super().__init__("perception_publisher")
        self.publisher = self.create_publisher(
            TrackedObjects,
            "/cpx/perception",
            10,
        )
        self.debug_time, self.timing_publisher, self.timing_stream = configure_timing(self, "perception")
        self.message_guard = self.create_guard_condition(self.publish_messages)
        self.receiver = TcpJsonReceiver(5052, self.get_logger(), on_message=self.message_guard.trigger)

    def publish_messages(self):
        for data in self.receiver.get_messages():
            publish_started_ns = time.time_ns()
            payload = data.get("data", data)
            message = TrackedObjects()
            timestamp_s = data.get(
                "timestamp_s",
                self.get_clock().now().nanoseconds / 1e9,
            )
            _set_stamp(message.header.stamp, timestamp_s)
            message.header.frame_id = str(data.get("frame_id", "map"))

            for item in payload.get("objects", []):
                tracked = TrackedObject()
                
                # Use the original vehicle ID when possible. Perception and V2X then
                # generate the same UUID when they report the same vehicle.
                object_id = str(
                    item.get("vehicle_id", item.get("id", ""))
                )

                tracked.object_id.uuid = list(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        "cpx:object:{}".format(object_id),
                    ).bytes
                )

                confidence = float(item.get("confidence", 1.0))
                tracked.existence_probability = confidence

                classification = ObjectClassification()
                classification.label = _classification_label(
                    item.get("type", "")
                )
                classification.probability = confidence
                tracked.classification = [classification]

                pose = tracked.kinematics.pose_with_covariance.pose
                pose.position.x = float(item.get("x", 0.0))
                pose.position.y = float(item.get("y", 0.0))
                pose.position.z = float(item.get("z", 0.0))

                yaw = item.get("psi")
                if yaw is None:
                    pose.orientation.w = 1.0
                else:
                    yaw = float(yaw)
                    pose.orientation.z = math.sin(yaw / 2.0)
                    pose.orientation.w = math.cos(yaw / 2.0)
                    tracked.kinematics.orientation_availability = (
                        TrackedObjectKinematics.AVAILABLE
                    )

                speed = float(item.get("v", 0.0))
                twist = tracked.kinematics.twist_with_covariance.twist
                twist.linear.x = speed
                tracked.kinematics.is_stationary = abs(speed) < 0.1

                # This topic carries the measured/tracked state only. A
                # separate tracker/predictor will later turn these objects
                # into PredictedObjects for the planner.
                tracked.shape.type = Shape.BOUNDING_BOX
                tracked.shape.dimensions.x = float(
                    item.get("length_m") or 0.0
                )
                tracked.shape.dimensions.y = float(
                    item.get("width_m") or 0.0
                )
                tracked.shape.dimensions.z = float(
                    item.get("height_m") or 0.0
                )
                message.objects.append(tracked)

            self.publisher.publish(message)
            publish_transport_timing(self.timing_publisher, data, self.timing_stream, publish_started_ns, time.time_ns())

    def destroy_node(self):
        self.receiver.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
