"""The composition: the only place that knows all packages (ADR-07).

It builds the registry from an explicit list of rules, checks rule parameters against it, builds measures from a
contract and offers profile and validate.
"""

from phoibe.dokimasia.compose.defaults import create_default_registry
from phoibe.dokimasia.compose.registry import Builder, ParamKind, ParamSpec, Registry, Rule, SignalUse
from phoibe.dokimasia.compose.rules_generic import list_generic_rules
from phoibe.dokimasia.compose.rules_wind import list_wind_rules
from phoibe.dokimasia.compose.run import (
    Validation,
    attest_validation,
    build_measures,
    build_run,
    profile_layer,
    validate_layer,
)
from phoibe.dokimasia.compose.validation import validate_layer_rules

__all__ = [
    "Builder",
    "ParamKind",
    "ParamSpec",
    "Registry",
    "Rule",
    "SignalUse",
    "Validation",
    "attest_validation",
    "build_measures",
    "build_run",
    "create_default_registry",
    "list_generic_rules",
    "list_wind_rules",
    "profile_layer",
    "validate_layer",
    "validate_layer_rules",
]
