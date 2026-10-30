import dataclasses
import datetime

import geopandas as gpd
import numpy as np
import pytest
import shapely.geometry
import xarray

from phoibe.geography.complexity.rix.computations.fieldsampler import Field
from phoibe.geography.complexity.rix.pipelines.analyzer import TRIXAnalyzer
from phoibe.geography.complexity.rix.pipelines.ports import WrittenArtifacts
from phoibe.geography.complexity.rix.pipelines.product import (
    PRODUCT_SPECIFICATION,
    RunParameters,
    SampleParameters,
)

NOW = datetime.datetime(2026, 10, 24, 12, 0, tzinfo=datetime.UTC)


@pytest.fixture
def make_field():
    """Field of the plane z = a x + b y on a 50 m grid from -half to +half, without georeferencing."""

    def make(a=1.0, b=0.0, half=2000, step=50):
        x = np.arange(-half, half + 1, step, dtype=float)
        da = xarray.DataArray(a * x[None, :] + b * x[:, None], coords={"x": x, "y": x}, dims=("y", "x"), name="dem")
        assert not hasattr(da, "rio"), "rioxarray was imported in this process: run with -m 'not rio'"
        return Field.from_array(da)

    return make


@pytest.fixture
def make_locations():
    def make(ids=("a", "b", "c"), xs=(0, 100, -100), ys=(0, 0, 100), crs=None, index_name=None):
        points = [shapely.geometry.Point(x, y) for x, y in zip(xs, ys, strict=True)]
        if ids is None:
            frame = gpd.GeoDataFrame(geometry=points, crs=crs)
            return frame if index_name is None else frame.rename_axis(index_name)
        return gpd.GeoDataFrame({"site_id": list(ids)}, geometry=points, crs=crs)

    return make


class InMemoryWriter:
    """Implements `ResultWriter` without files. Reports what a correct writer would, and remembers the calls.

    `tamper` changes what is reported, `fail` is raised instead of writing.
    """

    def __init__(self, spec, tamper=None, fail=None):
        self.spec, self.tamper, self.fail = spec, tamper, fail
        self.calls, self.manifest = [], None

    def write_artifacts(self, result, target, profile):
        self.calls.append(("artifacts", profile))
        if self.fail is not None:
            raise self.fail
        present = result.record.present_inputs
        files, table_columns, table_records = {}, {}, {}
        layer_columns, layer_geometry_types, layer_records = {}, {}, {}
        for artifact in self.spec.artifacts_for(profile, present):
            if artifact.key == "manifest":
                continue
            files[artifact.key] = artifact.filename
            if artifact.table is not None:
                frame = result.tables[artifact.table].reset_index()
                table_columns[artifact.key], table_records[artifact.key] = tuple(frame.columns), len(frame)
            for layer in self.spec.layers_for(artifact.key, present):
                frame = result.tables[layer.table].reset_index()
                layer_columns[layer.key] = tuple(
                    c for c in frame.columns if layer.geometry_type is None or c != "geometry"
                )
                layer_geometry_types[layer.key], layer_records[layer.key] = layer.geometry_type, len(frame)
        written = WrittenArtifacts(
            profile, files, table_columns, table_records, layer_columns, layer_geometry_types, layer_records
        )
        return self.tamper(written) if self.tamper is not None else written

    def write_manifest(self, manifest, target):
        self.calls.append(("manifest",))
        self.manifest = manifest


@pytest.fixture
def make_writer(product_spec):
    return lambda **kwargs: InMemoryWriter(product_spec, **kwargs)


@pytest.fixture
def make_analyzer(product_spec):
    """Analyzer with small parameters (4 rays of 0.5 km, steps of 50 m). Other values via `parameters`."""

    def make(spec=None, writer=None, method="linear", **parameters):
        values = {"n_angles": 4, "R_km": 0.5, "dr_km": 0.05, "slope_critical": 0.1, "crs": None} | parameters
        return TRIXAnalyzer(
            spec or product_spec, RunParameters(**values), SampleParameters(method), writer, clock=lambda: NOW
        )

    return make


@pytest.fixture
def product_spec():
    return PRODUCT_SPECIFICATION


@pytest.fixture
def make_spec():
    """The product specification with other severities or parameters of its checks."""

    def make(severities=None, check_parameters=None):
        checks = dict(PRODUCT_SPECIFICATION.checks)
        for name, severity in (severities or {}).items():
            checks[name] = dataclasses.replace(checks[name], severity=severity)
        for name, parameters in (check_parameters or {}).items():
            checks[name] = dataclasses.replace(checks[name], parameters=parameters)
        return dataclasses.replace(PRODUCT_SPECIFICATION, checks=checks)

    return make
