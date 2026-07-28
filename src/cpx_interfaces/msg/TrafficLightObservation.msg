# One traffic-light observation received from OpenCDA perception.

# Original CARLA/OpenCDA actor ID. This is kept as text so no information is
# lost when it is different from the integer ID used by Autoware.
string source_id

# Original object type reported by OpenCDA, normally "traffic_light".
string type

# Original normalized state text, such as "red", "yellow", or "green".
string state

# Position supplied by OpenCDA. In the current OpenCDA implementation this is
# a trigger-derived road/intersection position, not the physical lamp position.
geometry_msgs/Point position

# Reliability reported by OpenCDA, from 0.0 to 1.0.
float32 confidence

# The same ID, state, and confidence represented in Autoware's standard type.
autoware_perception_msgs/TrafficLightGroup signal
