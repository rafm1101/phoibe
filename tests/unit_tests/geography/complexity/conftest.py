import dataclasses

import pytest

from phoibe.geography.complexity.rix.geometry import RayGeometry


@dataclasses.dataclass(frozen=True)
class Location:
    easting: float
    northing: float


@pytest.fixture
def dummy_location():
    return Location(easting=-2.5, northing=-7.4)


@pytest.fixture
def origin():
    return Location(easting=0.0, northing=0.0)


@pytest.fixture
def ray_1km_100m(dummy_location):
    return RayGeometry.from_compass_regular(location=dummy_location, theta=0.0, R_km=1.0, dr_km=0.1)


@pytest.fixture
def ray_01km(origin, request):
    dr_km = request.param
    return RayGeometry.from_compass_regular(location=origin, theta=0.0, R_km=0.1, dr_km=dr_km)
