# One synchronized CP/V2X update containing lane events and traffic controls.

std_msgs/Header header
uint16 schema_version
uint64 sequence
cpx_interfaces/LaneEvent[] lane_events
cpx_interfaces/TrafficControl[] traffic_controls
