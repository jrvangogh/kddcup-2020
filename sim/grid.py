import collections
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree as KDTree

from sim.geo import great_circle_distance, local_projection_distance, METERS_PER_DEG_LAT, METERS_PER_DEG_LNG


class Grid:
    def __init__(self):
        self.grids = collections.OrderedDict()  # type: Dict[str, Tuple[float, float]]
        self.transitions = dict()  # type: Dict[int, Dict[start_grid_id, Dict[str, float]]]

        grid_path = Path(__file__).parent.parent / 'data' / 'interim' / 'hexagon_grid_table.csv'
        with grid_path.open('r') as csvfile:
            for row in csv.reader(csvfile):
                if len(row) != 13:
                    continue
                grid_id = row[0]

                # Use centroid for simplicity
                lng = sum([float(row[i]) for i in range(1, 13, 2)]) / 6
                lat = sum([float(row[i]) for i in range(2, 13, 2)]) / 6
                self.grids[grid_id] = (lat, lng)

        assert len(self.grids) == 8518
        self.grid_ids = list(self.grids.keys())  # type: List[str]
        self.kdtree = KDTree([(METERS_PER_DEG_LAT*lat, METERS_PER_DEG_LNG*lng)
                              for lat, lng in self.grids.values()])

        transitions_path = Path(__file__).parent.parent / 'data' / 'interim' / 'idle_transition_probability'
        with transitions_path.open('r') as csvfile:
            for row in csv.reader(csvfile):
                # TODO: verify hour in GMT
                hour, start_grid_id, end_grid_id, probability = row
                hour = int(hour)
                if hour not in self.transitions:
                    self.transitions[hour] = dict()

                hour_dict = self.transitions[hour]
                if start_grid_id not in hour_dict:
                    hour_dict[start_grid_id] = dict()

                start_dict = hour_dict[start_grid_id]
                if end_grid_id not in start_dict:
                    start_dict[end_grid_id] = float(probability)
        assert len(self.transitions) == 24

    def lookup_grid_id(self, lat: float, lng: float) -> str:
        """Get grid id from coordinates."""
        _, i = self.kdtree.query([METERS_PER_DEG_LAT*lat, METERS_PER_DEG_LNG*lng])
        return self.grid_ids[i]

    def lookup_grid_ids(self, coords: np.ndarray) -> List[str]:
        """Lookup multiple grid_ids.

        Input coords is an array of lat,lng pairs.
        """
        coords[:, 0] *= METERS_PER_DEG_LAT
        coords[:, 1] *= METERS_PER_DEG_LNG
        _, idx = self.kdtree.query(coords)
        return [self.grid_ids[i] for i in idx]

    def lookup_coord(self, grid_id: str) -> Tuple[float, float]:
        """Return lat/lng coordinates of grid."""
        return self.grids[grid_id]

    def distance(self, x: str, y: str, fast=True) -> float:
        """Return distance between two grids in meters.

        When fast is true, a local projection distance is used. Otherwise, great circle distance is used.
        """
        if x not in self.grids or y not in self.grids:
            return 1e12

        lat_x, lng_x = self.grids[x]
        lat_y, lng_y = self.grids[y]

        if fast:
            return local_projection_distance(lat_x, lng_x, lat_y, lng_y)
        else:
            return great_circle_distance(lat_x, lng_x, lat_y, lng_y)

    def idle_transitions(self, timestamp: int, start_grid_id: str) -> Dict[str, float]:
        hour = time.gmtime(timestamp).tm_hour
        if hour in self.transitions and start_grid_id in self.transitions[hour]:
            return self.transitions[hour][start_grid_id]
        return {start_grid_id: 1.}
