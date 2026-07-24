# OpenCDA-to-ROS 2 Data Bridge

## Overview

This project receives data produced by the OpenCDA localization and
perception and V2X modules. It publishes the data on five ROS topics and provides one
subscriber node that prints all received messages.

OpenCDA runs with Python 3.7, while ROS 2 Jazzy uses Python 3.12. The two
processes therefore communicate through local TCP connections using JSON.

```text
OpenCDA LocalizationManager
        └── TCP 5051 ──> localization_publisher ──> /cpx/localization

OpenCDA PerceptionManager (nearby objects)
        └── TCP 5052 ──> perception_publisher ───> /cpx/perception

OpenCDA PerceptionManager (traffic lights)
        └── TCP 5053 ──> traffic_light_publisher -> /cpx/traffic_light

OpenCDA V2XManager (nearby CAVs)
        └── TCP 5054 ──> v2x_publisher ──────────> /cpx/v2x

OpenCDA CP messages (lane events and traffic controls)
        └── TCP 5055 ──> cooperative_message_publisher
                         └────────────────────────> /cpx/cooperative_messages

                                                all five topics
                                                       │
                                                       v
                                                data_subscriber
                                                (prints the data)
```

The ROS nodes do not read CARLA directly. They only receive outputs already
produced by the OpenCDA Localization, Perception, and V2X modules.

## Files

The ROS package is located in `src/cpx_comm_test`.

| File | Purpose |
|---|---|
| `tcp_json_receiver.py` | Shared method for receiving newline-separated JSON over TCP. |
| `localization_publisher.py` | Receives localization data on port 5051 and publishes `/cpx/localization`. |
| `perception_publisher.py` | Receives nearby-object data on port 5052 and publishes `/cpx/perception`. |
| `traffic_light_publisher.py` | Receives traffic-light data on port 5053 and publishes `/cpx/traffic_light`. |
| `v2x_publisher.py` | Receives nearby-CAV V2X data on port 5054 and publishes `/cpx/v2x`. |
| `cooperative_message_publisher.py` | Receives lane events and traffic controls on port 5055 and publishes `/cpx/cooperative_messages`. |
| `data_subscriber.py` | Subscribes to all five topics and prints the data. |

Custom ROS messages are defined in `src/cpx_interfaces`.

| Message | Purpose |
|---|---|
| `LaneEvent.msg` | Describes a lane closure, roadway hazard, or work zone. |
| `TrafficControl.msg` | Describes a cooperative traffic-light state. |
| `CooperativeMessageArray.msg` | Carries lane events and traffic controls in one synchronized update. |

## ROS topics and message types

| Topic | ROS message type |
|---|---|
| `/cpx/localization` | `nav_msgs/msg/Odometry` |
| `/cpx/perception` | `autoware_perception_msgs/msg/PredictedObjects` |
| `/cpx/traffic_light` | `autoware_perception_msgs/msg/TrafficLightGroupArray` |
| `/cpx/v2x` | `autoware_perception_msgs/msg/TrackedObjects` |
| `/cpx/cooperative_messages` | `cpx_interfaces/msg/CooperativeMessageArray` |

## Custom message format

`LaneEvent.msg` contains:

```text
uint8 UNKNOWN=0
uint8 LANE_CLOSURE=1
uint8 ROAD_HAZARD=2
uint8 WORK_ZONE=3

string id
uint8 event_type
string source
builtin_interfaces/Time source_stamp
builtin_interfaces/Duration ttl
geometry_msgs/Point position
float32 confidence
```

`TrafficControl.msg` contains:

```text
string id
string source
builtin_interfaces/Time source_stamp
builtin_interfaces/Duration ttl
autoware_perception_msgs/TrafficLightGroup signal
float32 confidence
```

 The
`signal` field reuses Autoware's traffic-light format for the light ID, color,
shape, status, and confidence.

`CooperativeMessageArray.msg` puts both custom message lists into one update:

```text
std_msgs/Header header
uint16 schema_version
uint64 sequence
cpx_interfaces/LaneEvent[] lane_events
cpx_interfaces/TrafficControl[] traffic_controls
```



## Data format


### Localization

```json
{
  "ego_state": {
    "x": 100.5,
    "y": 25.2,
    "z": 0.3,
    "v": 5.5,
    "psi": 1.57
  }
}
```

### Perception

```json
{
  "objects": [
    {
      "id": "25",
      "vehicle_id": "25",
      "type": "vehicle.tesla.model3",
      "x": 110.2,
      "y": 25.4,
      "z": 0.3,
      "v": 4.8,
      "psi": 1.57,
      "length_m": 4.5,
      "width_m": 1.8,
      "height_m": 1.5,
      "confidence": 1.0
    }
  ]
}
```


### Traffic light

```json
{
  "traffic_lights": [
    {
      "id": "12",
      "type": "traffic_light",
      "state": "red",
      "position": {
        "x": 125.0,
        "y": 25.0,
        "z": 3.0
      },
      "confidence": 1.0
    }
  ]
}
```

### V2X

```json
{
  "nearby_cavs": [
    {
      "id": "cav-id",
      "vehicle_id": "42",
      "type": "vehicle.lincoln.mkz_2017",
      "source": "opencda_v2x",
      "x": 100.0,
      "y": 25.0,
      "z": 0.3,
      "v": 5.0,
      "psi": 1.57,
      "length_m": 4.9,
      "width_m": 1.9,
      "height_m": 1.5,
      "confidence": 1.0
    }
  ],
  "communication_range_m": 100.0
}
```

### Cooperative lane events and traffic controls

This is the JSON input expected on TCP port `5055`. The
`cooperative_message_publisher` converts it into
`cpx_interfaces/msg/CooperativeMessageArray`.

```json
{
  "schema_version": 1,
  "sequence": 4,
  "timestamp_s": 20.5,
  "lane_events": [
    {
      "id": "closure-1",
      "type": "lane_closure",
      "source": "roadside_unit",
      "timestamp_s": 20.5,
      "ttl_s": 10.0,
      "position": [100.0, 25.0, 0.0],
      "confidence": 1.0
    }
  ],
  "control": [
    {
      "id": "signal-12",
      "control_id": "12",
      "type": "traffic_light",
      "signal_state": "red",
      "source": "roadside_unit",
      "timestamp_s": 20.5,
      "ttl_s": 2.0,
      "confidence": 1.0
    }
  ]
}
```



## Requirements

- Ubuntu with ROS 2 Jazzy installed.
- ROS interface generation tools:
  `sudo apt install ros-jazzy-rosidl-default-generators`
- OpenCDA project configured with Python 3.7 and CARLA 0.9.12.
- The OpenCDA `data_transmitter.py` connected to
  `VehicleManager.update_info()`.
- The ROS workspace and OpenCDA must run on the same computer because the TCP
  receiver uses `127.0.0.1`.

## Run OpenCDA

First, start the CARLA 0.9.12 server in a separate terminal (change the path):

```bash
export UE4_ROOT=/home/umd-user/carla_source/UnrealEngine_4.26

"$UE4_ROOT/Engine/Binaries/Linux/UE4Editor" \
  /home/umd-user/carla_source/carla/Unreal/CarlaUE4/CarlaUE4.uproject \
  /Game/Carla/Maps/Town10HD_Opt \
  -game -vulkan -quality-level=Low -ResX=800 -ResY=600
```

Wait until the CARLA server is ready on port 2000. Then open another terminal
and run the scenario:

```bash
conda activate carla307
cd /path/to/CP-X-planning-module
python opencda.py -t cpx_town10_scenario_1 -v 0.9.12
```
Note: Change the name of the conda environment.
## Build this ROS workspace

```bash
cd ~/cp_ros_ws
source /opt/ros/jazzy/setup.zsh
colcon build --symlink-install --packages-select cpx_interfaces cpx_comm_test
source ~/cp_ros_ws/install/setup.zsh
```

Rebuild the workspace after changing any ROS source file.

## Run

Start each command in a separate terminal. In every ROS terminal, first run:

```bash
source /opt/ros/jazzy/setup.zsh
source ~/cp_ros_ws/install/setup.zsh
```

### 1. Start the localization publisher

```bash
ros2 run cpx_comm_test localization_publisher
```

### 2. Start the perception publisher

```bash
ros2 run cpx_comm_test perception_publisher
```

### 3. Start the traffic-light publisher

```bash
ros2 run cpx_comm_test traffic_light_publisher
```

### 4. Start the V2X publisher

```bash
ros2 run cpx_comm_test v2x_publisher
```

### 5. Start the cooperative-message publisher

```bash
ros2 run cpx_comm_test cooperative_message_publisher
```

### 6. Start the subscriber

```bash
ros2 run cpx_comm_test data_subscriber
```

### 7. Start CARLA and run the OpenCDA scenario

After the CARLA 0.9.12 server is ready, open the OpenCDA terminal:

```bash
conda activate carla307
cd /path/to/CP-X-planning-module
python opencda.py -t cpx_town10_scenario_1 -v 0.9.12
```

The subscriber terminal should print localization, nearby-object, traffic-light,
nearby-CAV V2X, and cooperative lane-event/traffic-control messages.
