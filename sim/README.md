# Simulator Design
This document describes the (probable) design of the DiDi dispatch simulator
based on the contest description, historical data provided, and snippets of info 
hidden in forum posts. It will serve as a tech spec for the simulator that I 
implement.

## Agent interface
At each time step, the simulator provides the following to the agent:

* Candidate dispatches (order-driver pairs)
* Repositionable drivers

The agent returns:

* Matched order-driver pairs
* Destinations (hexid) for repositionable drivers

With this info, the simulator can update the state and provide the next set of 
inputs to the agent.

### Candidate Dispatch Data
```
{"order_id": 0, 
"driver_id": 36, 
"order_driver_distance": 1126.2238477454885, 
"order_start_location": [104.17213000000001, 30.65868], 
"order_finish_location": [104.07704, 30.68109], 
"driver_location": [104.17223539384607, 30.6485633653061], 
"timestamp": 1488330000, 
"order_finish_timestamp": 1488335000, 
"day_of_week": 2, 
"reward_units": 2.620967741935484, 
"pick_up_eta": 375.40794924849615}
```

### Repositionable Driver Data
```
{"timestamp": 1477962000, 
"driver_info": [{"driver_id": 0, "grid_id": "8f2a0ba14e0965b7"}, ... ] 
"day_of_week": 2}
```

## Historical Data
Historical data is provided which is used to power the simulator.

#### Hex Grid System
Lat/long pairs for the vertices of each hexagon (along with an ID) are provided.
With this data, we can translate between lat/long and hex grids.

#### Ride Trajectories
Timestamp/location pairs are provided for the duration of each historical ride 
request that was dispatched by DiDi. The trajectory is provided for P1, P2, P3 
time (TODO: Check that P1 time is in the data. Check that drivers in trajectory
data all serve ride requests).

#### Ride request data
This data describes all actual ride requests that were dispatched by DiDi. The 
ride start/stop time represent the bounds of ride P3 time.

| Field | Description |
|-------|-------------|
| Order ID        | Anonymized              |
| Ride Start time | Unix timestamp, seconds |
| Ride stop time  | Unix timestamp, seconds |
| Pickup Lat      |                         |
| Pickup Long     |                         |
| Dropoff Lat     |                         |
| Dropoff Long    |                         |
| Reward units    | Model prediction value  |

#### Idle Transition Probability
Idle drivers move around according to this transition probability data based on 
grid id.

| Field | Description |
|-------|-------------|
| Hour                   | Hour of day                                         |
| Origin grid ID         | Hex grid ID                                         |
| Destination grid ID    | Hex grid ID                                         |
| Transition probability | Probability of transition from origin to destination|

#### Order Cancellation probability
Orders are canceled based on a distance-based probability. Unique probabilities 
are provided for each order.

| Field | Description |
|-------|-------------|
| Order ID | Anonymized |
| Cancellation Probability (200m) | Cancellation probability when pickup distance is 200m |
| Cancellation Probability (400m) | Cancellation probability when pickup distance is 400m |
| ...     
| Cancellation Probability (2000m) | Cancellation probability when pickup distance is 2000m |

## Simulator

### Data Preprocessing

#### Requests
Build simulated requests using ride request data. Treat `ride start time` as 
 `ride request time` (this is an approximation, but probably close enough). 
 Batch orders over 2 second dispatch batch window, joining with cancellation 
 probability data.
 
#### Drivers
Build driver go online time/location and go offline time from driver trajectory 
data. Use first data point of each driver as go online time/location. Use last
trajectory timestamp as go offline time.
 
### Algorithm

1. Advance one batch window (2 sec)
    
    1. For any routes that completed, mark drivers as available.
    
    1. Remove any drivers that went offline in this batch.
    
    2. Add drivers that went online in this batch. Maintain a set number of 
    repositionable drivers by making new drivers eligible until the requirement 
    is met.
    
    3. Add orders that were made in this batch.
    
    4. Create order/driver candidate pairs using 2km great circle distance 
    cutoff.
    
    5. Check if any reposition-eligble drivers have been idle for >= 5min.
    
    6. Send order/driver candidates and reposition-eligible drivers to agent.
    
2. Update state
    
    1. P2/P3: 
        1. If new P2: Cancel order according to probability. If canceled, move 
        driver to P1.
        2. Complete route, using calculated duration:
            - P2: Straight line distance divided by 3 m/s
            - P3: According to historical data
            
    2. P1: 
        1. Choose random hex to drive towards if not assigned. 
        2. Advance towards random hex.

