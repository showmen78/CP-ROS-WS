# One request or result exchanged between separated CP-X planner components.

std_msgs/Header header
uint64 cycle_id
uint64 request_id
string requester
string operation
bool success
string reason
string payload_json
