"""The composition: the only place that knows all packages (ADR-07).

It builds the registry from an explicit list of rules, checks rule parameters against it, builds measures from a
contract and offers profile and validate.
"""

from phoibe.dokimasia.compose.registry import Builder, ParamKind, ParamSpec, Registry, Rule, SignalUse
from phoibe.dokimasia.compose.rules_generic import list_generic_rules
from phoibe.dokimasia.compose.run import Validation, build_measures, profile_layer, validate_layer
from phoibe.dokimasia.compose.validation import validate_layer_rules


def create_default_registry() -> Registry:
    """Create the registry with the generic rules.

    Returns
    -------
    Registry
        A new registry; there is no shared state between calls.
    """
    return Registry(list_generic_rules())


__all__ = [
    "Builder",
    "ParamKind",
    "ParamSpec",
    "Registry",
    "Rule",
    "SignalUse",
    "Validation",
    "build_measures",
    "create_default_registry",
    "list_generic_rules",
    "profile_layer",
    "validate_layer",
    "validate_layer_rules",
]
