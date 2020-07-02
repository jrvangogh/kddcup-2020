from math import radians, sin, cos, acos, sqrt, atan2

import numpy as np

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


def local_projection_intermediate_point(lat1, lng1, lat2, lng2, distance_meters):
    """Return lat,lng of intermediate point a fixed distance along the line between points 1 and 2.

    Uses local projection coordinate system.
    """
    x1, x2 = METERS_PER_DEG_LAT * lat1, METERS_PER_DEG_LAT * lat2
    y1, y2 = METERS_PER_DEG_LNG * lng1, METERS_PER_DEG_LNG * lng2

    u = x2 - x1
    v = y2 - y1
    norm = np.sqrt(u**2 + v**2)
    x_int = x1 + u/norm * distance_meters
    y_int = y1 + u/norm * distance_meters

    return x_int * DEG_PER_METER_LAT, y_int * DEG_PER_METER_LNG


def great_circle_distance(lat1, lng1, lat2, lng2):
    """Returns great circle distance in meters."""
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    x = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lng1 - lng2)
    x = max(min(x, 1.0), -1.0)
    return 6371008.8 * (acos(x))


def intermediate_point(lat1, lng1, lat2, lng2, frac):
    """Return lat,lng of intermediate point a fraction of the way between points 1 and 2.

    This uses great circle lines.
    """
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    d = acos(sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lng1 - lng2))  # Angular distance
    a = sin((1-frac)*d) / sin(d)
    b = sin(frac*d) / sin(d)

    x = a * cos(lat1) * cos(lng1) + b * cos(lat2) * cos(lng2)
    y = a * cos(lat1) * sin(lng1) + b * cos(lat2) * sin(lng2)
    z = a * sin(lat1) + b * sin(lat2)
    lat = atan2(z, sqrt(x ** 2 + y ** 2))
    lng = atan2(y, x)

    return lat, lng
