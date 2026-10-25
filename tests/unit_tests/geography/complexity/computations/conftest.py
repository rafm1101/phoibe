import types

import numpy as np
import pyproj
import pytest
import shapely.geometry

from phoibe.geography.complexity.rix.computations.fieldsampler import FieldInfo
from phoibe.geography.complexity.rix.computations.geometry import RayGeometry
from phoibe.geography.complexity.rix.computations.profiles import RayProfile, SamplingInfo
from phoibe.geography.complexity.rix.computations.results import RadialRuggedness, RayRuggedness


@pytest.fixture
def make_profile():
    """Profile heading to `theta` from the origin, without a sampler. `n_points` defaults to the profile length."""

    def make(r_m, z, theta=0.0, nan_count=0, n_points=None, crs=None, field_info=None):
        r_m = np.asarray(r_m, dtype=float)
        ray = RayGeometry.from_compass(location=shapely.geometry.Point(0.0, 0.0), theta=theta, r_m=r_m, crs=crs)
        info = SamplingInfo(
            field_info=field_info or FieldInfo(None, None, None),
            crs=None if crs is None else pyproj.CRS.from_user_input(crs).to_string(),
            n_points=len(r_m) if n_points is None else n_points,
            nan_count=nan_count,
            message="",
        )
        return RayProfile(ray_=ray, r_m=r_m, z=np.asarray(z, dtype=float), sampling_info=info)

    return make


@pytest.fixture
def make_ray(make_profile):
    def make(z, theta=0.0, slope_critical=0.1, dr_m=100.0, **kwargs):
        return RayRuggedness(make_profile(np.arange(len(z)) * dr_m, z, theta=theta, **kwargs), slope_critical)

    return make


@pytest.fixture
def make_radial(make_ray):
    """Radial result with rays at 0°, 10°, ...; a spec is `z` or `(z, nan_count, n_points)`."""

    def make(*specs):
        rays = []
        for k, spec in enumerate(specs):
            z, nan_count, n_points = spec if isinstance(spec, tuple) else (spec, 0, None)
            rays.append(make_ray(z, theta=10.0 * k, nan_count=nan_count, n_points=n_points))
        return RadialRuggedness(rays=tuple(rays))

    return make


@pytest.fixture
def make_sampler():
    class DummySampler:
        def __init__(self, z, crs=None, field_info=None):
            self.z = np.asarray(z, dtype=float)
            self.crs = crs
            self.field = types.SimpleNamespace(info=field_info or FieldInfo(None, None, None))
            self.calls = []

        def sample(self, xs, ys):
            self.calls.append((np.asarray(xs), np.asarray(ys)))
            z = self.z[: len(xs)].copy()
            return z, int(np.isnan(z).sum())

    return DummySampler
