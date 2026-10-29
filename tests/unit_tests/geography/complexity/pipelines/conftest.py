import copy
import dataclasses

import pytest

from phoibe.geography.complexity.rix.pipelines.product.schema import PRODUCT_SPECIFICATION
from phoibe.geography.complexity.rix.pipelines.product.specification import ProductSpecification


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


MINIMAL_DEFINITION = {
    "name": "Demo",
    "description": "A demo product.",
    "version": "1.2.3",
    "references": {"REF": {"title": "A reference", "year": 2020}},
    "parameters": {
        "ray": {
            "n": {"description": "Count.", "value": 3, "dtype": "int", "unit": None, "source": "REF", "locked": True},
            "r": {"description": "Radius.", "value": 2.5, "unit": "km", "range": [0, 10]},
            "crs": {"description": "CRS.", "value": None, "dtype": "str"},
        },
        "mode": {"method": {"description": "Method.", "value": "linear"}},
    },
    "inputs": {
        "dem": {
            "description": "DEM.",
            "required": True,
            "type": "array",
            "notes": ["a note"],
            "conditions": ["a rule"],
        },
        "refs": {"description": "References.", "required": False},
    },
    "checks": {
        "c1": {"description": "A check.", "severity": "warn", "parameters": {"max": 0.1}},
        "c2": {"description": "Another check.", "severity": "abort"},
    },
    "tables": {
        "t_a": {
            "description": "Table A.",
            "columns": {
                "id": {"description": "Identifier."},
                "v": {"description": "Value.", "unit": "m", "dtype": "float", "range": [0, 1]},
            },
        },
        "t_b": {
            "description": "Table B.",
            "columns": {"code": {"description": "Code.", "dtype": "int", "values": {0: "No", 1: "Yes"}}},
        },
        "t_geo": {
            "description": "Geodata.",
            "columns": {"id": {"description": "Identifier."}, "geometry": {"description": "Geometry."}},
        },
    },
    "artifacts": {
        "profiles": {"small": ["table_a", "manifest"], "big": ["table_a", "table_b", "manifest", "container"]},
        "files": {
            "manifest": {"filename": "manifest.yaml"},
            "table_a": {"filename": "a.csv", "table": "t_a"},
            "table_b": {"filename": "b.csv", "table": "t_b", "requires_input": "refs"},
            "container": {"filename": "c.gpkg"},
        },
        "layers": {
            "geo": {"artifact": "container", "table": "t_geo", "name": "geo", "geometry_type": "Point"},
            "attrs": {
                "artifact": "container",
                "table": "t_b",
                "name": "attrs",
                "geometry_type": None,
                "requires_input": "refs",
            },
        },
    },
    "manifest": {
        "description": "Manifest.",
        "keys": {
            "meta": {"description": "Meta.", "keys": {"name": {"description": "Name."}}},
            "run": {"description": "Run.", "keys": {"n": {"description": "Count."}}},
        },
    },
}


@pytest.fixture
def make_definition():
    """Copy of a small valid definition. `changes` maps dotted paths to new values, `deleted` lists paths to drop."""

    def make(changes=None, deleted=()):
        definition = copy.deepcopy(MINIMAL_DEFINITION)
        for path, value in (changes or {}).items():
            *parents, last = path.split(".")
            node = definition
            for key in parents:
                node = node[key]
            node[last] = value
        for path in deleted:
            *parents, last = path.split(".")
            node = definition
            for key in parents:
                node = node[key]
            del node[last]
        return definition

    return make


@pytest.fixture
def spec(make_definition):
    return ProductSpecification.from_dict(make_definition())
