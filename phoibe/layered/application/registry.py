import typing

from phoibe.layered.logging.logging import get_logger
from phoibe.layered.rules.rule import ValidationRule

logger = get_logger(__name__)


class RuleRegistry:
    """Central registry for validation rules.

    Rules self-register using the @RuleRegistry.register decorator.

    Example
    -------
    1. Register some rule:

       > @RuleRegistry.register("temporal_resolution")
       > class TemporalResolutionRule(ValidationRule):
       >     ...
    """

    _rules: dict[str, typing.Type[ValidationRule]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a validation rule."""

        def wrapper(rule_class: typing.Type[ValidationRule]):
            if name in cls._rules:
                logger.warning(f"Rule '{name}' already registered, overwriting.")
            cls._rules[name] = rule_class
            logger.debug(f"Registered rule: {name} -> {rule_class.__name__}")
            return rule_class

        return wrapper

    @classmethod
    def get(cls, name: str) -> typing.Type[ValidationRule]:
        """Get rule class by name.

        Parameters
        ----------
        name
            Rule identifier.

        Return
        ------
        ValidationRule
            Rule class.

        Raises
        ------
        KeyError
            If rule not found.
        """
        if name not in cls._rules:
            available = ", ".join(sorted(cls._rules.keys()))
            raise KeyError(f"Rule '{name}' not found in registry. " f"Available rules: {available}.")
        return cls._rules[name]

    @classmethod
    def list_rules(cls) -> list[str]:
        """Get list of all registered rule names."""
        return sorted(cls._rules.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check whether rule is registered.

        Parameters
        ----------
        name
            Identifier under which the rule is registered.

        Return
        ------
        bool
            True if rule exists in registry.
        """
        return name in cls._rules

    @classmethod
    def clear(cls):
        """Clear all registered rules (for testing)."""
        cls._rules.clear()


__all__ = ["RuleRegistry"]
