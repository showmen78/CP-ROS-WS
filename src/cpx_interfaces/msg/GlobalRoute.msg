# A route snapshot published for recording and visualization.

std_msgs/Header header
uint64 cycle_id
bool route_found
string reason
int32 optimal_lane_id
float64 remaining_distance_m
string next_macro_maneuver
string current_road_option
cpx_interfaces/RoutePoint[] points
