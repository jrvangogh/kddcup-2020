import heapq
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
import time
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm

from model.agent import Agent
from sim.geo import (METERS_PER_DEG_LAT, METERS_PER_DEG_LNG, local_projection_distance,
                     local_projection_intermediate_point)
from sim.grid import Grid

STEP_SEC = 2
MAX_PICKUP_RADIUS_METERS = 2000
DRIVER_SPEED_SEC_PER_METER = 1.0 / 3
DRIVER_STEP_DISTANCE = STEP_SEC / DRIVER_SPEED_SEC_PER_METER
DRIVER_POWERMODE_REPOSITION_TIMEOUT = 60 * 5  # Amount of time powermode driver must be idle before repositioning

CANCEL_PROB_DISTANCES = list(range(200, 2001, 200))


PROCESSED_DATA_PATH = Path(__file__).parent.parent / 'data' / 'processed'


class DriverState(Enum):
    FREE = auto()
    IDLE_MOVING = auto()
    IN_RIDE = auto()


@dataclass
class Driver:
    """A driver."""
    driver_id: str
    lat: float
    lng: float
    start_time: int
    end_time: int  # Shift end time
    available_time: int = 0  # Next time that driver is available (not in a ride)
    destination_lat: float = 0  # Destination used for idle movement
    destination_lng: float = 0
    idle_duration: int = 0
    state: DriverState = DriverState.FREE
    score: float = 0

    def __lt__(self, other):
        return self.end_time < other.end_time

    def assign_ride(self, lat: float, lng: float, available_time: int, score: float):
        """Assign a ride to driver.

        Store location as ride destination. We don't need to track the driver's location until
        they arrive.
        """
        self.lat = lat
        self.lng = lng
        self.available_time = available_time
        self.idle_duration = 0
        self.state = DriverState.IN_RIDE
        self.score += score


@dataclass
class Order:
    """An order."""
    order_id: str
    start_time: int
    end_time: int
    pickup_lat: float
    pickup_lng: float
    dropoff_lat: float
    dropoff_lng: float
    reward: float
    cancel_prob: List[float]

    def create_matches(self, drivers: List[Driver], day_of_week: int):
        """Create potential match with drivers."""
        if not drivers:
            return []
        coords = np.array([(d.lat, d.lng, self.pickup_lat, self.pickup_lng) for d in drivers])
        distances = local_projection_distance(coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3])
        matches = [{
            'order_id': self.order_id,
            'driver_id': d.driver_id,
            'order_driver_distance': distance,
            'order_start_location': [self.pickup_lng, self.pickup_lat],
            'order_finish_location': [self.dropoff_lng, self.dropoff_lat],
            'driver_location': [d.lng, d.lat],
            'timestamp': self.start_time,
            'order_finish_timestamp': self.end_time,
            'day_of_week': day_of_week,
            'reward_units': self.reward,
            'pick_up_eta': distance * DRIVER_SPEED_SEC_PER_METER
        } for d, distance in zip(drivers, distances)]
        return matches


class Simulator:
    def __init__(self,
                 agent: Agent,
                 ds: str,
                 num_powermode_drivers: int = 5,
                 driver_warmup_time_sec: int = 60,
                 order_limit: Optional[int] = None,
                 disable_progress_bar: bool = False
                 ):
        """Initialize simulator.

        All the data is loaded into memory during init. The simulator state can be reset and re-run so the data load
        only has to happen once.

        Args:
            agent: Agent to use which handles dispatches and repositions.
            ds: String in YYYYMMDD format to choose the day to simulate.
            num_powermode_drivers: Number of powermode (aka repositionable) drivers.
            driver_warmup_time_sec: Warmup time occurs before first order. During this period, drivers can go online
                and offline, and they move according to idle transition rules. This warms up the simulation state.
            order_limit: Limit simulation to N orders. If none, all orders are used.
            disable_progress_bar: Disables the tqdm progress bar.
        """
        self.agent = agent
        self.ds = ds
        self.num_powermode_drivers = num_powermode_drivers
        self.order_limit = order_limit
        self.driver_limit = None
        self.driver_warmup_time_sec = driver_warmup_time_sec
        self.disable_progress_bar = disable_progress_bar

        # Spatial info
        self.grid = Grid()

        #########
        # Drivers
        self.driver_idx = 0
        self.drivers = self._init_drivers(ds, self.driver_limit)  # All drivers. Natural order: start_time
        self.drivers_online = dict()  # All online drivers, keyed by driver_id.

        # These are secondary data structures to enable fast lookup for driver updates.
        #
        # Online but not in ride. Keyed by driver_id
        self.drivers_available = dict()
        # Powermode drivers. Keyed by driver_id
        self.drivers_powermode = dict()
        # Powermode offline drivers. Keyed by driver_id
        self.drivers_powermode_offline = dict()
        # Online and in ride. Natural order: available_time. Min-heap of tuples of (available_time, driver_id)
        self.drivers_busy = []

        ########
        # Orders
        self.order_idx = 0
        self.orders = self._init_orders(ds, self.order_limit)
        self.orders_active = dict()

        ##################
        # Simulation state
        self.day_of_week = datetime.strptime(self.ds, '%Y%m%d').weekday()  # Monday = 0, Sunday = 6
        self.sim_start_time = max(self.drivers[0].start_time, self.orders[0].start_time)
        self.time = self.sim_start_time - self.driver_warmup_time_sec
        self.steps = 0

        self.num_fulfilled = 0
        self.num_cancelled = 0
        self.num_unfulfilled = 0
        self.score = 0
        self.score_cancelled = 0
        self.score_unfulfilled = 0

    def run(self):
        """Run simulation."""
        start_time = time.time()
        while self.time < self.sim_start_time:
            self._warmup_step()
        with tqdm(total=len(self.orders), unit=' orders', unit_scale=True, disable=self.disable_progress_bar) as pbar:
            self.pbar = pbar
            while self.order_idx < len(self.orders):
                self._step()

        end_time = time.time()
        print(f'Run time: {end_time-start_time:.2f} sec')

        repo_score = np.mean([driver.score / (driver.end_time-driver.start_time)
                              for driver in self.drivers_powermode_offline.values()])
        total_orders = self.num_fulfilled + self.num_cancelled + self.num_unfulfilled

        metrics = {
            'dispatch_score': self.score,
            'reposition_score': repo_score,
            'orders_completed': self.num_fulfilled / total_orders,
            'orders_cancelled': self.num_cancelled / total_orders,
            'orders_unfulfilled': self.num_unfulfilled / total_orders
        }
        print(f'Dispatch score: {metrics["dispatch_score"]:.4f}')
        print(f'Reposition score: {metrics["reposition_score"]:.4f}')
        print(f'Completed orders: {metrics["orders_completed"]:.2f} | '
              f'Cancelled orders: {metrics["orders_cancelled"]:.2f} | '
              f'Unfulfilled orders: {metrics["orders_unfulfilled"]:.2f}')
        return metrics

    def reset(self):
        """Reset simulation.

        Call this after a run to prepare the simulator state for a new run.
        """
        self.driver_idx = 0
        self.drivers_available = dict()
        self.drivers_powermode = dict()
        self.drivers_powermode_offline = dict()
        self.drivers_busy = []
        self.order_idx = 0

        self.sim_start_time = max(self.drivers[0].start_time, self.orders[0].start_time)
        self.time = self.sim_start_time - self.driver_warmup_time_sec
        self.steps = 0

        self.num_fulfilled = 0
        self.num_cancelled = 0
        self.num_unfulfilled = 0
        self.score = 0
        self.score_cancelled = 0
        self.score_unfulfilled = 0

    @staticmethod
    def _init_drivers(ds: str, driver_limit: Optional[int] = None) -> List[Driver]:
        print('Initializing drivers')
        df = pd.read_parquet(PROCESSED_DATA_PATH / 'drivers' / f'{ds}.parquet')
        if driver_limit:
            df = df.head(driver_limit)
        drivers = []
        for _, row in df.iterrows():
            driver = Driver(row.driver_id, row.start_lat, row.start_lng, row.start_time, row.end_time)
            drivers.append(driver)
        drivers = sorted(drivers, key=lambda d: d.start_time)
        print(f'{len(drivers)} drivers initialized')
        return drivers

    @staticmethod
    def _init_orders(ds: str, order_limit: Optional[int] = None) -> List[Order]:
        print('Initializing orders')
        df = pd.read_parquet(PROCESSED_DATA_PATH / 'orders' / f'{ds}.parquet')
        df = df.sort_values(by='start_time', ignore_index=True)
        if order_limit:
            df = df.head(order_limit)
        orders = []
        for _, row in df.iterrows():
            order = Order(
                row.order_id,
                row.start_time,
                row.stop_time,
                row.pickup_lat,
                row.pickup_lng,
                row.dropoff_lat,
                row.dropoff_lng,
                row.reward,
                row.cancel_prob)
            orders.append(order)
        orders = sorted(orders, key=lambda x: x.start_time)
        print(f'{len(orders)} orders initialized')
        return orders

    def _step(self):
        self.time += STEP_SEC
        self.steps += 1

        if (self.steps % 20) == 0:
            metrics = {
                'd_online': len(self.drivers_online),
                'd_busy': len(self.drivers_busy),
                'o_fulfilled': self.num_fulfilled,
                'o_unfulfilled': self.num_unfulfilled,
                'o_cancelled': self.num_cancelled,
                'd_score': self.score
            }
            self.pbar.set_postfix(**metrics, refresh=False)

        # Complete routes, moving finished drivers back to available.
        self._complete_routes()
        # Remove drivers that went offline
        self._remove_offline_drivers()
        # Add new drivers that went online
        self._add_online_drivers()

        # Add orders that were made
        self._add_new_orders()

        # Build candidate order-driver pairs
        candidates = self._build_candidates()
        matched = self.agent.dispatch(candidates)

        self._process_new_matches(matched)
        self._process_unfulfilled_orders()
        self._position_powermode_drivers()
        self._move_idle_drivers()

    def _warmup_step(self):
        """Warmup steps occur before orders happen and only simulate driver online/offline/idle movement."""
        self.time += STEP_SEC
        self.steps += 1

        if (self.steps % 100) == 0:
            print(f'DRIVER WARMUP | Time: {self.time} | '
                  f'Drivers online: {len(self.drivers_online)} | '
                  f'Drivers available: {len(self.drivers_available)} |')

        # Remove drivers that went offline
        self._remove_offline_drivers()
        # Add new drivers that went online
        self._add_online_drivers()

        self._move_idle_drivers()

    def _complete_routes(self):
        """Complete finished routes by marking drivers as free."""
        while self.drivers_busy and self.drivers_busy[0][0] <= self.time:
            _, driver_id = heapq.heappop(self.drivers_busy)
            driver = self.drivers_online[driver_id]
            driver.state = DriverState.FREE
            self.drivers_available[driver.driver_id] = driver

    def _remove_offline_drivers(self):
        """Remove drivers who went offline from being active."""
        # Track which drivers to remove from available
        remove = [driver_id for driver_id, driver in self.drivers_available.items() if driver.end_time <= self.time]
        for driver_id in remove:
            if driver_id in self.drivers_powermode.keys():
                self.drivers_powermode_offline[driver_id] = self.drivers_available[driver_id]
                del self.drivers_powermode[driver_id]
            del self.drivers_available[driver_id]
            del self.drivers_online[driver_id]

    def _add_online_drivers(self):
        """Add drivers who went online."""
        while self.driver_idx < len(self.drivers) and self.drivers[self.driver_idx].start_time <= self.time:
            driver = self.drivers[self.driver_idx]
            self.driver_idx += 1
            self.drivers_online[driver.driver_id] = driver  # Main data structure
            self.drivers_available[driver.driver_id] = driver  # Secondary data structure
            if len(self.drivers_powermode) < self.num_powermode_drivers:
                self.drivers_powermode[driver.driver_id] = driver

    def _add_new_orders(self):
        """Add new orders which occurred."""
        order_count = 0
        while self.order_idx < len(self.orders) and self.orders[self.order_idx].start_time <= self.time:
            order = self.orders[self.order_idx]
            self.order_idx += 1
            order_count += 1
            self.orders_active[order.order_id] = order
        self.pbar.update(order_count)

    def _build_candidates(self):
        """Build match candidates.

        This is accomplished by putting order pickup coordinates and driver coordinates into KDTrees, then finding all
        neighbors within 2km of each order.
        """
        drivers = [(METERS_PER_DEG_LAT * d.lat, METERS_PER_DEG_LNG * d.lng) for d in self.drivers_available.values()]
        driver_ids = [d.driver_id for d in self.drivers_available.values()]
        orders = [(METERS_PER_DEG_LAT * order.pickup_lat, METERS_PER_DEG_LNG * order.pickup_lng)
                  for order in self.orders_active.values()]
        if len(drivers) == 0 or len(orders) == 0:
            return []
        order_tree = KDTree(np.array(orders))
        driver_tree = KDTree(np.array(drivers))

        all_order_matches = order_tree.query_ball_tree(driver_tree, r=MAX_PICKUP_RADIUS_METERS)

        candidates = []
        for order_matches, order in zip(all_order_matches, self.orders_active.values()):
            drivers_matched = [self.drivers_available[driver_ids[driver_idx]] for driver_idx in order_matches]
            candidates.extend(order.create_matches(drivers_matched, self.day_of_week))
        return candidates

    def _process_new_matches(self, matched):
        """Process new matches created by agent.

        Immediately after match, the cancellation random event is calculated. If a ride is cancelled, the driver remains
        free for this batch cycle and will do idle movement.
        """
        for match in matched:
            driver_id = match['driver_id']
            order_id = match['order_id']

            driver = self.drivers_available[driver_id]
            order = self.orders_active[order_id]

            pickup_distance = local_projection_distance(driver.lat, driver.lng, order.pickup_lat, order.pickup_lng)
            cancel_prob = np.interp(pickup_distance, CANCEL_PROB_DISTANCES, order.cancel_prob)
            if cancel_prob > np.random.rand():  # Canceled
                self.num_cancelled += 1
                self.score_cancelled += order.reward
            else:  # Not cancelled
                self.num_fulfilled += 1
                self.score += order.reward

                # Update driver
                # Ride duration calculation:
                # - P2 is calculated by great circle distance and 3 m/s moving speed
                # - P3 comes directly from order duration
                available_time = (
                        self.time +
                        int(pickup_distance * DRIVER_SPEED_SEC_PER_METER) +
                        (order.end_time - order.start_time))
                driver.assign_ride(order.dropoff_lat, order.dropoff_lng, available_time, order.reward)

                del self.drivers_available[driver_id]
                del self.orders_active[order_id]
                heapq.heappush(self.drivers_busy, (driver.available_time, driver.driver_id))

    def _process_unfulfilled_orders(self):
        """Remaining orders are unfulfilled, so remove them and track stats."""
        self.num_unfulfilled += len(self.orders_active)
        self.score_unfulfilled += np.sum([order.reward for order in self.orders_active.values()])
        self.orders_active.clear()

    def _position_powermode_drivers(self):
        """Position idle powermode drivers using agent reposition function."""
        driver_ids = [driver_id for driver_id, driver in self.drivers_powermode.items()
                      if driver.state == DriverState.FREE and
                      driver.idle_duration > DRIVER_POWERMODE_REPOSITION_TIMEOUT]
        if not driver_ids:
            return
        driver_coords = np.array([(self.drivers[driver_id].lat, self.drivers[driver_id].lng)
                                  for driver_id in driver_ids])
        grid_ids = self.grid.lookup_grid_ids(driver_coords)
        repo_observ = {'timestamp': self.time,
                       'day_of_week': self.day_of_week,
                       'driver_info': [{'driver_id': driver_id, 'grid_id': grid_id}
                                       for driver_id, grid_id in zip(driver_ids, grid_ids)]}
        repositions = self.agent.reposition(repo_observ)
        for repo in repositions:
            self._reposition_driver_to_grid_id(repo['driver_id'], repo['destination'])

    def _move_idle_drivers(self):
        """Idle driver movement."""
        for driver in self.drivers_available.values():
            driver.idle_duration += STEP_SEC

        # Driver who just idly reached their idle destination are now (temporarily) free
        for driver in self.drivers_available.values():
            # Driver who just idly reached their idle destination are now (temporarily) free
            if driver.state == DriverState.IDLE_MOVING and driver.available_time <= self.time:
                driver.lat = driver.destination_lat
                driver.lng = driver.destination_lng
                driver.state = DriverState.FREE

        # Assign new idle destinations to free drivers
        drivers = [d for d in self.drivers_available.values() if d.state == DriverState.FREE]
        if drivers:
            driver_coords = np.array([(d.lat, d.lng) for d in drivers])
            driver_grid_ids = self.grid.lookup_grid_ids(driver_coords)

            all_transitions = [self.grid.idle_transitions(self.time, grid_id) for grid_id in driver_grid_ids]
            destination_grid_ids = [np.random.choice(list(transitions.keys()), p=list(transitions.values()))
                                    for transitions in all_transitions]
            for driver, grid_id in zip(drivers, destination_grid_ids):
                self._reposition_driver_to_grid_id(driver, grid_id)

        # Advance drivers towards their idle destinations
        drivers = [d for d in self.drivers_available.values()
                   if d.state == DriverState.IDLE_MOVING and not
                   (d.lat == d.destination_lat and d.lng == d.destination_lng)]
        if drivers:
            driver_coords = np.array([(d.lat, d.lng, d.destination_lat, d.destination_lng) for d in drivers])
            lats, lngs = local_projection_intermediate_point(driver_coords[:, 0], driver_coords[:, 1],
                                                             driver_coords[:, 2], driver_coords[:, 3],
                                                             DRIVER_STEP_DISTANCE)
            for driver, lat, lng in zip(drivers, lats, lngs):
                driver.lat = lat
                driver.lng = lng

    def _reposition_driver_to_grid_id(self, driver, grid_id):
        """Reposition a driver to chosen grid_id.

        This sets the driver's state so that the logic to advance drivers towards their destination will pick up
        this driver.
        """
        lat, lng = self.grid.lookup_coord(grid_id)
        driver.destination_lat = lat
        driver.destination_lng = lng
        dist = local_projection_distance(driver.lat, driver.lng, driver.destination_lat, driver.destination_lng)
        driver.available_time = self.time + int(DRIVER_SPEED_SEC_PER_METER * dist)
        driver.state = DriverState.IDLE_MOVING


if __name__ == '__main__':
    sim = Simulator(Agent(), '20161102', order_limit=None)
    sim.run()
