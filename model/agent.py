import numpy as np
from itertools import product
import os
import pickle


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tile_maps.pickle')


DEFAULT_GAMMA = 0.90       # Future reward discount
DEFAULT_UP = 0.90          # Penalty applied to states containing unassigned drivers
DEFAULT_ALPHA = 0.25       # SARSA learning rate
DEFAULT_SVI = 4.22         # State values initialized to this (average ride reward in offline data)


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

    def __init__(self, lng_offset, lat_offset, state_value_init=DEFAULT_SVI, alpha=DEFAULT_ALPHA):
        self.state_value_init = state_value_init
        self.alpha = alpha
        self.map = {}
        self.lng_offset = lng_offset
        self.lat_offset = lat_offset

    def _loc_key(self, location):
        """Get the key for the map associated with the given location"""
        return int(location[0] * 100 + self.lng_offset), int(location[1] * 100 + self.lat_offset)

    def get_state_value(self, location):
        """Get the state value for the given location"""
        return self.map.get(self._loc_key(location), self.state_value_init)

    def update_state_value(self, location, new_value):
        """Update the state value for the given location based on the given new value

        Updates by self.alpha * (new_value - old_value)
        """
        key = self._loc_key(location)
        old_value = self.map.get(key, self.state_value_init)
        self.map[key] = old_value + self.alpha * (new_value - old_value)


class StateModel:
    """A state value map that uses a coarse tiling for 4 TileMaps"""

    def __init__(self):
        self.tile_maps = [TileMap(lng, lat) for (lng, lat) in product([0.0, 0.25, 0.5, 0.75], [0.0, 0.25, 0.5, 0.75])]
        self.num_maps = len(self.tile_maps)

    def get_state_value(self, location):
        return sum(self.tile_maps[i].get_state_value(location) for i in range(self.num_maps)) / self.num_maps

    def update_state_value(self, location, new_value):
        for tile_map in self.tile_maps:
            tile_map.update_state_value(location, new_value)


class Agent(object):
    """ Agent for dispatching and reposition """

    @staticmethod
    def _load_state_model():
        with open(MODEL_PATH, 'rb') as f:
            loaded_tile_maps = pickle.load(f)
        sm = StateModel()
        for (tm, loaded) in zip(sm.tile_maps, loaded_tile_maps):
            tm.map = loaded
        return sm

    def save_state_model(self, output_file_name):
        map_list = [tm.map for tm in self.state_model.tile_maps]
        with open(output_file_name, 'wb') as f:
            pickle.dump(map_list, f)

    def __init__(self, gamma=DEFAULT_GAMMA, unassigned_penalty=DEFAULT_UP, load_state_model=True):
        """ Load your trained model and initialize the parameters """
        self.gamma = gamma
        self.unassigned_penalty = unassigned_penalty
        if load_state_model:
            self.state_model = self._load_state_model()
        else:
            self.state_model = StateModel()

    def calc_order_assignment_value(self, order):
        completion_prob = 1.0 - cancel_probability(order['order_driver_distance'])
        order_finish_state_value = self.state_model.get_state_value(order['order_finish_location'])
        return completion_prob * order['reward_units'] + self.gamma * order_finish_state_value

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
        # TODO: Consider using 0.9 * current_value here?
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
                self.state_model.update_state_value(driver_loc, self.unassigned_penalty * state_value)
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
