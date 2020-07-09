import os
import pickle
from math import cos

import numpy as np
from scipy import interpolate

###########################################################
# Calculate approximate projection based on region latitude

# We set the latitude here for the region
LAT_DEG = 30.6
LAT_RAD = np.deg2rad(LAT_DEG)
# Source:
# https://en.wikipedia.org/wiki/Geographic_coordinate_system#Latitude_and_longitude
METERS_PER_DEG_LAT = 111132.92 - 559.82 * cos(2 * LAT_RAD) + 1.175 * cos(4 * LAT_RAD) - 0.0023 * cos(6*LAT_RAD)
DEG_PER_METER_LAT = 1 / METERS_PER_DEG_LAT
METERS_PER_DEG_LNG = 111412.84 * cos(LAT_RAD) - 93.5 * cos(3*LAT_RAD) + 0.118 * cos(5*LAT_RAD)
DEG_PER_METER_LNG = 1 / METERS_PER_DEG_LNG


def local_projection_distance(lat1, lng1, lat2, lng2):
    """Return distance between two points in meters using a local projection."""
    return np.sqrt((METERS_PER_DEG_LAT * (lat1 - lat2)) ** 2 + (METERS_PER_DEG_LNG * (lng1 - lng2)) ** 2)


# These are the median pickup lat/lng from all available historical order data
LAT_CENTER = 30.67135
LNG_CENTER = 104.073643

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cancel_prob_splines.pickle')
CANCEL_PROB_DISTANCES = list(range(200, 2001, 200))
MAX_SPLINE_DISTANCE = 20000


class CancelProb:
    def __init__(self):
        self.splines = self._load_cancel_splines()

    def cancel_probability(self, order_driver_distance, pickup_loc):
        """Calculate cancel probability based on spline fit to city center distance."""
        if order_driver_distance > MAX_SPLINE_DISTANCE:
            # Fallback to original method
            return self._cancel_probability_static(order_driver_distance)

        city_dist = self._distance_from_city_center(pickup_loc[0], pickup_loc[1])
        cancel_probs = [spl(city_dist) for spl in self.splines]

        f = interpolate.interp1d(CANCEL_PROB_DISTANCES, cancel_probs)
        return f(order_driver_distance)

    @staticmethod
    def _load_cancel_splines():
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _distance_from_city_center(lng, lat):
        return local_projection_distance(LAT_CENTER, LNG_CENTER, lat, lng)

    @staticmethod
    def _cancel_probability_static(order_driver_distance):
        """Determined in cancel_prob.ipynb

        This technically goes above 100% starting at 7599 m. Doesn't cause any critical issues though, so we'll save on
        compute by NOT doing max(cancel_prob, 1.0)
        """
        return 1 / (np.exp(4.39349586 - 0.00109042 * order_driver_distance) + 1) + 0.02
