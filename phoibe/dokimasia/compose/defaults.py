"""The registry that a plain call uses."""

from phoibe.dokimasia.compose import registry as registry_module
from phoibe.dokimasia.compose import rules_generic, rules_wind


def create_default_registry() -> registry_module.Registry:
    """Create the registry with the generic rules and the rules of the wind pack.

    Returns
    -------
    Registry
        A new registry; there is no shared state between calls.
    """
    return registry_module.Registry((*rules_generic.list_generic_rules(), *rules_wind.list_wind_rules()))
