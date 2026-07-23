# OpenCDA-to-ROS 2 Data Bridge

## Overview

This project receives data produced by the OpenCDA localization and
perception modules. It publishes the data on three ROS topics and provides one
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

                                                all three topics
                                                       │
                                                       v
                                                data_subscriber
                                                (prints the data)
```

The ROS nodes do not read CARLA directly. They only receive the outputs already
produced by OpenCDA Perception and Localization module.

## Files

The ROS package is located in `src/cpx_comm_test`.

| File | Purpose |
|---|---|
| `tcp_json_receiver.py` | Shared method for receiving newline-separated JSON over TCP. |
| `localization_publisher.py` | Receives localization data on port 5051 and publishes `/cpx/localization`. |
| `perception_publisher.py` | Receives nearby-object data on port 5052 and publishes `/cpx/perception`. |
| `traffic_light_publisher.py` | Receives traffic-light data on port 5053 and publishes `/cpx/traffic_light`. |
| `data_subscriber.py` | Subscribes to all three topics and prints the data. |


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
  },
  "units": {
    "position": "m",
    "speed": "m/s",
    "heading": "rad"
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

## Requirements

- Ubuntu with ROS 2 Jazzy installed.
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
colcon build --symlink-install --packages-select cpx_comm_test
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

### 4. Start the subscriber

```bash
ros2 run cpx_comm_test data_subscriber
```

### 5. Start CARLA and run the OpenCDA-wrapper scenario

After the CARLA 0.9.12 server is ready, open the OpenCDA terminal:

```bash
conda activate carla307
cd /path/to/CP-X-planning-module
python opencda.py -t cpx_town10_scenario_1 -v 0.9.12
```

The subscriber terminal should print localization, nearby-object, and
traffic-light messages.

