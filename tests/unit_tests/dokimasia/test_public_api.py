"""The public names of each package. A new public name has to be added here on purpose."""

import importlib
import types

import pytest

EXPECTED = {
    "phoibe.dokimasia.core": {
        "Catalog",
        "Dataset",
        "FINGERPRINT_ALGORITHM",
        "Finding",
        "Fingerprint",
        "Measure",
        "Measured",
        "MeasurementError",
        "Metric",
        "NotEvaluable",
        "NotEvaluableReason",
        "Outcome",
        "Profile",
        "Signal",
        "Statistic",
        "Subject",
        "build_finding_mask",
        "compute_fingerprint",
        "profile_dataset",
        "run_measure",
    },
    "phoibe.dokimasia.rules": {"GapMeasure", "MissingRunsMeasure"},
}


@pytest.mark.parametrize("package", EXPECTED)
def test_public_api(package):
    module = importlib.import_module(package)

    assert set(module.__all__) == EXPECTED[package]
    assert all(hasattr(module, name) for name in module.__all__)


@pytest.mark.parametrize("package", EXPECTED)
def test_public_api_given_attribute_access_exposes_no_other_public_symbol(package):
    module = importlib.import_module(package)
    exposed = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and not isinstance(value, types.ModuleType)
    }

    assert exposed == EXPECTED[package]
