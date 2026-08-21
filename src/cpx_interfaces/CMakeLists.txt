cmake_minimum_required(VERSION 3.8)
project(cpx_interfaces)

find_package(ament_cmake REQUIRED)
find_package(autoware_perception_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LaneEvent.msg"
  "msg/TrafficControl.msg"
  "msg/CooperativeMessageArray.msg"
  "msg/TrafficLightObservation.msg"
  "msg/TrafficLightObservationArray.msg"
  "msg/ComponentPacket.msg"
  "msg/RoutePoint.msg"
  "msg/GlobalRoute.msg"
  "msg/PlannerInputFrame.msg"
  DEPENDENCIES
    autoware_perception_msgs
    builtin_interfaces
    geometry_msgs
    std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
