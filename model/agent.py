from collections import defaultdict
import numpy as np


GAMMA = 0.9              # Future reward discount
ALPHA = 0.25             # SARSA learning rate
STATE_VALUE_INIT = 4.22  # State values initialized to this (average ride reward in offline data)


def cancel_probability(order_driver_distance):
    """Determined in cancel_prob.ipynb

    This technically goes above 100% starting at 7599 m. Doesn't cause any critical issues though, so we'll save on
    compute by NOT doing max(cancel_prob, 1.0)
    """
    return 1 / (np.exp(4.39349586 - 0.00109042 * order_driver_distance) + 1) + 0.02


class TileMap:
    """Stores state values by tiling a (lng, lat) grid

    Longitudes and latitudes are rounded to the nearest 100th (potentially with some offsetting).
    """

    def __init__(self, lng_offset, lat_offset):
        self.map = defaultdict(lambda: STATE_VALUE_INIT)  # Average reward in offline data
        self.lng_offset = lng_offset
        self.lat_offset = lat_offset

    def _loc_key(self, location):
        """Get the key for the map associated with the given location"""
        return int(location[0] * 100 + self.lng_offset), int(location[1] * 100 + self.lat_offset)

    def get_state_value(self, location):
        """Get the state value for the given location"""
        return self.map[self._loc_key(location)]

    def update_state_value(self, location, new_value):
        """Update the state value for the given location based on the given new value

        Updates by ALPHA * (new_value - old_value)
        """
        key = self._loc_key(location)
        self.map[key] += ALPHA * (new_value - self.map[key])


class StateModel:
    """A state value map that uses a coarse tiling for 4 TileMaps"""

    def __init__(self):
        self.tile_maps = [
            TileMap(0.0, 0.0),
            TileMap(0.5, 0.0),
            TileMap(0.0, 0.5),
            TileMap(0.5, 0.5),
        ]

    def get_state_value(self, location):
        return sum(self.tile_maps[i].get_state_value(location) for i in range(4)) / 4

    def update_state_value(self, location, new_value):
        for tile_map in self.tile_maps:
            tile_map.update_state_value(location, new_value)


class Agent(object):
    """ Agent for dispatching and reposition """

    def __init__(self):
        """ Load your trained model and initialize the parameters """
        self.state_model = StateModel()

    def calc_order_assignment_value(self, order):
        completion_prob = 1.0 - cancel_probability(order['order_driver_distance'])
        order_finish_state_value = self.state_model.get_state_value(order['order_finish_location'])
        return completion_prob * order['reward_units'] + GAMMA * order_finish_state_value

    def calc_current_driver_state_value(self, order):
        return self.state_model.get_state_value(order['driver_location'])

    def dispatch(self, dispatch_observ):
        """ Compute the assignment between drivers and passengers at each time step
        :param dispatch_observ: a list of dict, the key in the dict includes:
                order_id, int
                driver_id, int
                order_driver_distance, float
                order_start_location, a list as [lng, lat], float
                order_finish_location, a list as [lng, lat], float
                driver_location, a list as [lng, lat], float
                timestamp, int
                order_finish_timestamp, int
                day_of_week, int
                reward_units, float
                pick_up_eta, float

        :return: a list of dict, the key in the dict includes:
                order_id and driver_id, the pair indicating the assignment
        """
        for order in dispatch_observ:
            order['current_value'] = self.calc_current_driver_state_value(order)
            order['order_value'] = self.calc_order_assignment_value(order)
        dispatch_observ.sort(key=lambda o_dict: o_dict['order_value'] - o_dict['current_value'], reverse=True)
        assigned_order = set()
        assigned_driver = set()
        all_driver_locs = {}
        dispatch_action = []
        for od in dispatch_observ:
            all_driver_locs[od['driver_id']] = (od['driver_location'], od['current_value'])
            if od['order_value'] < od['current_value']:
                # Stop once driver orders are negative value
                break
            # make sure each order is assigned to one driver, and each driver is assigned with one order
            if (od['order_id'] in assigned_order) or (od['driver_id'] in assigned_driver):
                continue
            assigned_order.add(od['order_id'])
            assigned_driver.add(od['driver_id'])
            dispatch_action.append(dict(order_id=od['order_id'], driver_id=od['driver_id']))
            self.state_model.update_state_value(od['driver_location'], od['order_value'])

        for driver_id, (driver_loc, state_value) in all_driver_locs.items():
            if driver_id not in assigned_driver:
                self.state_model.update_state_value(driver_loc, GAMMA * state_value)
        return dispatch_action

    def reposition(self, repo_observ):
        """ Compute the reposition action for the given drivers
        :param repo_observ: a dict, the key in the dict includes:
                timestamp: int
                driver_info: a list of dict, the key in the dict includes:
                        driver_id: driver_id of the idle driver in the treatment group, int
                        grid_id: id of the grid the driver is located at, str
                day_of_week: int

        :return: a list of dict, the key in the dict includes:
                driver_id: corresponding to the driver_id in the od_list
                destination: id of the grid the driver is repositioned to, str
        """
        repo_action = []
        for driver in repo_observ['driver_info']:
            # the default reposition is to let drivers stay where they are
            repo_action.append({'driver_id': driver['driver_id'], 'destination': driver['grid_id']})
        return repo_action
