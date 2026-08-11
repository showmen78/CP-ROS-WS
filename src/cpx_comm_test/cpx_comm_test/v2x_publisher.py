"""Publish nearby OpenCDA V2X CAVs as Autoware tracked objects."""

import json
import math
import uuid

from autoware_perception_msgs.msg import ObjectClassification
from autoware_perception_msgs.msg import Shape
from autoware_perception_msgs.msg import TrackedObject
from autoware_perception_msgs.msg import TrackedObjectKinematics
from autoware_perception_msgs.msg import TrackedObjects
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from .tcp_json_receiver import TcpJsonReceiver


def _set_stamp(stamp, timestamp_s):
    """Convert a floating-point Unix time to a ROS time."""
    seconds = int(timestamp_s)
    nanoseconds = int((float(timestamp_s) - seconds) * 1000000000)
    stamp.sec = seconds
    stamp.nanosec = nanoseconds


def _classification_label(type_name):
    """Convert an OpenCDA type name to an Autoware class."""
    name = str(type_name).lower()
    if "truck" in name:
        return ObjectClassification.TRUCK
    if "trailer" in name:
        return ObjectClassification.TRAILER
    if "bus" in name:
        return ObjectClassification.BUS
    if "motorcycle" in name:
        return ObjectClassification.MOTORCYCLE
    if "bicycle" in name or "bike" in name:
        return ObjectClassification.BICYCLE
    if "pedestrian" in name or "walker" in name:
        return ObjectClassification.PEDESTRIAN
    if "vehicle" in name or "car" in name:
        return ObjectClassification.CAR
    return ObjectClassification.UNKNOWN


class V2XPublisher(Node):
    """Receive nearby-CAV JSON and publish tracked objects."""

    def __init__(self):
        super().__init__("v2x_publisher")
        self.publisher = self.create_publisher(
            TrackedObjects,
            "/cpx/v2x",
            10,
        )
        # Preserve the complete CP dictionaries as well as their typed object view.
        self.cp_obstacles_publisher = self.create_publisher(String, "/cpx/cp_obstacles", 10)
        self.receiver = TcpJsonReceiver(5054, self.get_logger())
        self.create_timer(0.02, self.publish_messages)

    def publish_messages(self):
        for data in self.receiver.get_messages():
            payload = data.get("data", data)
            message = TrackedObjects()
            timestamp_s = data.get(
                "timestamp_s",
                self.get_clock().now().nanoseconds / 1e9,
            )
            _set_stamp(message.header.stamp, timestamp_s)
            message.header.frame_id = str(data.get("frame_id", "map"))

            for item in payload.get("nearby_cavs", []):
                tracked = TrackedObject()
                
                # Use the original vehicle ID so that this UUID matches the perception
                # UUID when both sources report the same vehicle.
                cav_id = str(
                    item.get("vehicle_id", item.get("id", ""))
                )

                tracked.object_id.uuid = list(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        "cpx:object:{}".format(cav_id),
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

                yaw = float(item.get("psi", 0.0))
                pose.orientation.z = math.sin(yaw / 2.0)
                pose.orientation.w = math.cos(yaw / 2.0)
                tracked.kinematics.orientation_availability = (
                    TrackedObjectKinematics.AVAILABLE
                )

                speed = float(item.get("v", 0.0))
                tracked.kinematics.twist_with_covariance.twist.linear.x = (
                    speed
                )
                tracked.kinematics.is_stationary = abs(speed) < 0.1

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

            cp_message = String()
            cp_message.data = json.dumps({"schema_version": int(payload.get("schema_version", 1) or 1), "timestamp_s": float(payload.get("timestamp_s", timestamp_s) or timestamp_s), "obstacles": [dict(item) for item in list(payload.get("cp_obstacles", []) or []) if isinstance(item, dict)]}, allow_nan=False, separators=(",", ":"))
            self.cp_obstacles_publisher.publish(cp_message)

    def destroy_node(self):
        self.receiver.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = V2XPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
