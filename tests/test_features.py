import math
from src.features.geo import haversine_distance

def test_haversine_zero():
    assert math.isclose(haversine_distance(0, 0, 0, 0), 0.0, rel_tol=1e-9)

def test_haversine_known():
    # distance between roughly New York City coords (approx)
    d = haversine_distance(40.7128, -74.0060, 40.730610, -73.935242)
    # expected ~6-7 km (approx)
    assert 0 < d < 20