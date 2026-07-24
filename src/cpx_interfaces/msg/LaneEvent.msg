# A cooperative event that affects a road lane.

uint8 UNKNOWN=0
uint8 LANE_CLOSURE=1
uint8 ROAD_HAZARD=2
uint8 WORK_ZONE=3

# Unique event identifier.
string id

# Event category. Use one of the constants above.
uint8 event_type

# Name of the vehicle, roadside unit, or scenario that sent the event.
string source

# Time at which the source created or observed the event.
builtin_interfaces/Time source_stamp

# How long the event remains valid.
builtin_interfaces/Duration ttl

# Event location in the message header's coordinate frame.
geometry_msgs/Point position

# Reliability from 0.0 to 1.0.
float32 confidence
