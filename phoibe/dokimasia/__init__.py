"""Dokimasia: layered profiling and validation of time series (ADR-01 to ADR-12).

The packages follow one direction of dependency (ADR-07): `core` imports nothing of ours, `rules` imports `core`.

This module is the surface for the calls every run makes. A run reads like this::

    contract = dokimasia.load_contract(path)
    detection = dokimasia.wind.detect_wind_signals(dataset.signal_names)
    validation = dokimasia.validate_layer(dataset, subject, contract, "raw")
    attestation = dokimasia.attest_validation(validation, created_at=now, software_version=version)
    store.save_run(dokimasia.build_run(validation, attestation), created_at=now)
    bronze = dokimasia.advance_dataset(dataset, attestation, transform, software_version=version)

Everything else is reached through the packages: `core`, `contract`, `rules`, `wind`, `standardize`, `adapters` and
`compose`. The package configures no logging and writes no file by itself.
"""

import logging

from phoibe.dokimasia import adapters, compose, contract, core, rules, standardize, wind
from phoibe.dokimasia.adapters import FlatStore, load_contract
from phoibe.dokimasia.compose import attest_validation, build_run, profile_layer, validate_layer
from phoibe.dokimasia.core import Dataset, Subject, advance_dataset

# A library configures no logging (conventions): records go nowhere until the application sets handlers up.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Dataset",
    "FlatStore",
    "Subject",
    "adapters",
    "advance_dataset",
    "attest_validation",
    "build_run",
    "compose",
    "contract",
    "core",
    "load_contract",
    "profile_layer",
    "rules",
    "standardize",
    "validate_layer",
    "wind",
]
