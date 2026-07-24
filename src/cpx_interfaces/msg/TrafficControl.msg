# Cooperative traffic-light state sent to the planner.

# Unique traffic-control message identifier.
string id

# Name of the vehicle, roadside unit, or scenario that sent the state.
string source

# Time at which the source observed the state.
builtin_interfaces/Time source_stamp

# How long the state remains valid.
builtin_interfaces/Duration ttl

# Autoware traffic-light group containing the light color and condition.
autoware_perception_msgs/TrafficLightGroup signal

# Overall reliability of this cooperative message from 0.0 to 1.0.
float32 confidence
