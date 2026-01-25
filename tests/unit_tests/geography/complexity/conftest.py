import dataclasses

import numpy as np
import pytest
import xarray

from phoibe.geography.complexity.rix import RayGeometry

# from phoibe.geography.complexity.sampler import FieldSampler


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
def planar_field():
    x = np.arange(-100, 100)
    y = np.arange(-100, 100)
    xx, yy = np.meshgrid(x, y)
    z = 0.5 * xx
    return xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="field")


@pytest.fixture
def ray_1km_100m(dummy_location):
    return RayGeometry(location=dummy_location, theta=0.0, R_km=1.0, dr_km=0.1)


@pytest.fixture
def ray_01km(origin, request):
    dr_km = request.param
    return RayGeometry(location=origin, theta=0.0, R_km=0.1, dr_km=dr_km)
