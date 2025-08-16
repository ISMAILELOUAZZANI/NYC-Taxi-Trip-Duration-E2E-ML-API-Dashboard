from math import radians, sin, cos, atan2, sqrt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth surface.
    Input in decimal degrees. Output in kilometers.
    Vectorize at caller if necessary (pandas.apply / numpy).
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    earth_radius_km = 6371.0
    return earth_radius_km * c

def bearing(lat1, lon1, lat2, lon2):
    """
    Bearing from point 1 to point 2 in degrees.
    """
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    initial_bearing = atan2(x, y)
    # convert from radians to degrees and normalize
    bearing_deg = (initial_bearing * 180.0 / 3.141592653589793 + 360.0) % 360.0
    return bearing_deg