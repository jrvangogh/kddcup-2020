import numpy as np
import os
import warnings
import pandas as pd
from collections import deque
from sklearn.tree import DecisionTreeRegressor


warnings.simplefilter(action='ignore', category=FutureWarning)


MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tile_maps.pickle')


DEFAULT_GAMMA = 0.90            # Future reward discount
DEFAULT_UP = 0.90               # Penalty applied to states containing unassigned drivers
DEFAULT_MAX_DEPTH = 5           # Max depth of each tree
DEFAULT_MIN_SAMPLES_LEAF = 5    # Min number of samples in each leaf in each tree
DEFAULT_NUM_TREES = 100         # The number of trees in the forest
DEFAULT_MIN_X_LEN = 100         # Minimum number of instances to train a tree on
DEFAULT_EXP_DECAY = 0.97        # The decay used to weight older trees
DEFAULT_SVI = 4.22              # State values initialized to this (average ride reward in offline data)


def cancel_probability(order_driver_distance):
    """Determined in cancel_prob.ipynb

    This technically goes above 100% starting at 7599 m. Doesn't cause any critical issues though, so we'll save on
    compute by NOT doing max(cancel_prob, 1.0)
    """
    return 1 / (np.exp(4.39349586 - 0.00109042 * order_driver_distance) + 1) + 0.02


class FakeTree:

    def __init__(self, predict_return=DEFAULT_SVI):
        self.predict_return = predict_return

    def predict(self, X):
        arr = np.empty(X.shape[0])
        arr.fill(self.predict_return)
        return arr


class StateModel:
    """A state value map that uses a coarse tiling for 4 TileMaps"""

    def __init__(
            self,
            max_depth=DEFAULT_MAX_DEPTH,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
            init_predict=DEFAULT_SVI,
            num_trees=DEFAULT_NUM_TREES,
            exp_decay=DEFAULT_EXP_DECAY,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.q = deque([FakeTree(predict_return=init_predict) for _ in range(num_trees)], maxlen=num_trees)
        self.exp_decay = exp_decay
        self.tree_weights = np.power(self.exp_decay, np.arange(0, num_trees)).reshape(num_trees, 1)
        self.total_weight = self.tree_weights.sum()

    def predict(self, X):
        return (self.tree_weights * np.stack([tree.predict(X) for tree in self.q])).sum(axis=0) / self.total_weight

    def update(self, X, y):
        new_tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        new_tree.fit(X, y)
        self.q.append(new_tree)


class Agent(object):
    """ Agent for dispatching and reposition """

    # @staticmethod
    # def _load_state_model():
    #     with open(MODEL_PATH, 'rb') as f:
    #         loaded_tile_maps = pickle.load(f)
    #     sm = StateModel()
    #     for (tm, loaded) in zip(sm.tile_maps, loaded_tile_maps):
    #         tm.map = loaded
    #     return sm
    #
    # def save_state_model(self, output_file_name):
    #     map_list = [tm.map for tm in self.state_model.tile_maps]
    #     with open(output_file_name, 'wb') as f:
    #         pickle.dump(map_list, f)

    def __init__(
            self,
            gamma=DEFAULT_GAMMA,
            unassigned_penalty=DEFAULT_UP,
            min_x_len=DEFAULT_MIN_X_LEN,
            num_trees=DEFAULT_NUM_TREES,
            max_depth=DEFAULT_MAX_DEPTH,
    ):
        """ Load your trained model and initialize the parameters """
        self.gamma = gamma
        self.unassigned_penalty = unassigned_penalty
        self.state_model = StateModel(num_trees=num_trees, max_depth=max_depth)
        self.min_x_len = min_x_len
        self.x_prep = []
        self.x_prep_len = 0

    def calc_order_assignment_value(self, order_df):
        completion_prob = 1.0 - cancel_probability(order_df['order_driver_distance'])
        order_finish_state_value = self.state_model.predict(order_df[['finish_lng', 'finish_lat']].to_numpy())
        return completion_prob * order_df['reward_units'] + self.gamma * order_finish_state_value

    def calc_current_driver_state_value(self, order_df):
        return self.state_model.predict(order_df[['driver_lng', 'driver_lat']].to_numpy())

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
        if not dispatch_observ:
            return []
        order_df = pd.DataFrame.from_records(dispatch_observ)
        order_df['driver_lng'], order_df['driver_lat'] = order_df['driver_location'].str
        order_df['finish_lng'], order_df['finish_lat'] = order_df['order_finish_location'].str
        order_df['current_value'] = self.calc_current_driver_state_value(order_df)
        order_df['order_value'] = self.calc_order_assignment_value(order_df)
        order_df['net_value'] = order_df['order_value'] - self.gamma * order_df['current_value']
        order_df = order_df.sort_values(by='net_value', ascending=False)

        assign_df = order_df[['driver_id', 'driver_lng', 'driver_lat', 'current_value']]\
            .drop_duplicates(subset=['driver_id'])\
            .set_index('driver_id')
        assign_df['assign_value'] = self.gamma * assign_df['current_value']
        assigned_order = set()
        assigned_driver = set()
        dispatch_action = []
        for _, row in order_df.iterrows():
            order_id = row['order_id']
            driver_id = row['driver_id']
            if row['order_value'] < row['current_value']:
                # Stop once driver orders are negative value
                break
            # make sure each order is assigned to one driver, and each driver is assigned with one order
            if (order_id in assigned_order) or (driver_id in assigned_driver):
                continue
            assigned_order.add(order_id)
            assigned_driver.add(driver_id)
            dispatch_action.append(dict(order_id=order_id, driver_id=driver_id))
            assign_df.at[driver_id, 'assign_value'] = row['order_value']

        self.x_prep.append(assign_df)
        self.x_prep_len += assign_df.shape[0]
        if self.x_prep_len >= self.min_x_len:
            x_df = pd.concat(self.x_prep, ignore_index=True)
            self.state_model.update(x_df[['driver_lng', 'driver_lat']].to_numpy(),
                                    x_df['assign_value'].to_numpy().reshape(-1, 1))
            self.x_prep = []
            self.x_prep_len = 0
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
