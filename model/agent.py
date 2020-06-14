from collections import defaultdict
import numpy as np


GAMMA = 0.9
ALPHA = 0.5


def get_loc_key(location):
    return int(location[0] * 100), int(location[1] * 100)


def cancel_probability(order_driver_distance):
    """Determined in cancel_prob.ipynb"""
    return 1 / (np.exp(4.39349586 - 0.00109042 * order_driver_distance) + 1) + 0.02


class Agent(object):
    """ Agent for dispatching and reposition """

    def __init__(self):
        """ Load your trained model and initialize the parameters """
        self.state_values = defaultdict(lambda: 0.0)
        self.max_distance = 0.0

    def calc_order_value(self, order):
        loc_key = get_loc_key(order['order_finish_location'])
        cancel_prob = cancel_probability(order['order_driver_distance'])
        return (1 - cancel_prob) * order['reward_units'] + GAMMA * self.state_values[loc_key]

    def calc_current_value(self, order):
        loc_key = get_loc_key(order['driver_location'])
        return self.state_values[loc_key]

    def update_state_value(self, order):
        s0 = get_loc_key(order['driver_location'])
        self.state_values[s0] += ALPHA * (order['order_value'] - order['current_value'])

    def update_state_value_not_assigned(self, driver_location):
        loc_key = get_loc_key(driver_location)
        self.state_values[loc_key] *= (1 + ALPHA * GAMMA - ALPHA)

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
            order['current_value'] = self.calc_current_value(order)
            order['order_value'] = self.calc_order_value(order)
        dispatch_observ.sort(key=lambda o_dict: o_dict['order_value'] - o_dict['current_value'], reverse=True)
        assigned_order = set()
        assigned_driver = set()
        all_driver_locs = {}
        dispatch_action = []
        # stop_at = int(0.9 * len(dispatch_observ))
        for i, od in enumerate(dispatch_observ):
            all_driver_locs[od['driver_id']] = od['driver_location']
            # if od['order_value'] < od['current_value'] and i > stop_at:
            #     # Stop once driver orders are negative value and 90% of orders have been considered
            #     break
            # make sure each order is assigned to one driver, and each driver is assigned with one order
            if (od['order_id'] in assigned_order) or (od['driver_id'] in assigned_driver):
                continue
            assigned_order.add(od['order_id'])
            assigned_driver.add(od['driver_id'])
            dispatch_action.append(dict(order_id=od['order_id'], driver_id=od['driver_id']))
            self.update_state_value(od)

        for driver_id, driver_loc in all_driver_locs.items():
            if driver_id not in assigned_driver:
                self.update_state_value_not_assigned(driver_loc)
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
