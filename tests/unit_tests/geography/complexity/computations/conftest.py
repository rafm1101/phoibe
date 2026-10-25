import numpy as np
import pytest
import shapely.geometry

from phoibe.geography.complexity.rix.computations.fieldsampler import FieldInfo
from phoibe.geography.complexity.rix.computations.geometry import RayGeometry
from phoibe.geography.complexity.rix.computations.profiles import RayProfile, SamplingInfo


@pytest.fixture
def make_profile():
    """Profile heading to `theta` from the origin, without a sampler. `n_points` defaults to the profile length."""

    def make(r_m, z, theta=0.0, nan_count=0, n_points=None):
        r_m = np.asarray(r_m, dtype=float)
        ray = RayGeometry.from_compass(location=shapely.geometry.Point(0.0, 0.0), theta=theta, r_m=r_m, crs=None)
        info = SamplingInfo(
            field_info=FieldInfo(None, None, None),
            crs=None,
            n_points=len(r_m) if n_points is None else n_points,
            nan_count=nan_count,
            message="",
        )
        return RayProfile(ray_=ray, r_m=r_m, z=np.asarray(z, dtype=float), sampling_info=info)

    return make
