import numpy as np
import pytest
import xarray


@pytest.fixture
def planar_field():
    x = np.arange(0, 100)
    y = np.arange(0, 100)
    xx, yy = np.meshgrid(x, y)
    z = 0.5 * xx
    return xarray.DataArray(data=z, coords={"x": x, "y": y}, dims=("y", "x"), name="field")
