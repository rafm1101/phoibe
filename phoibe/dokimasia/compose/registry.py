"""The registry of rules: one name, its parameters, its metrics and how to build it (ADR-04, ADR-07).

The registry is an instance, not class-level state. It is built from an explicit list of rules, so there is one
source for rule names and nothing to clear between tests.
"""

import dataclasses
import enum
import types
from collections.abc import Callable, Iterable, Mapping

from phoibe.dokimasia import core


class ParamKind(enum.StrEnum):
    """The type of a rule parameter."""

    NUMBER = "number"
    NUMBER_OR_SUBJECT = "number_or_subject"
    TEXT = "text"
    TEXT_LIST = "text_list"


class SignalUse(enum.StrEnum):
    """Whether a rule is about one signal of the measure."""

    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


@dataclasses.dataclass(frozen=True, slots=True)
class ParamSpec:
    """One parameter of a rule."""

    kind: ParamKind
    """Its type."""
    required: bool = False
    """Whether the contract must give it."""


type Builder = Callable[[str, str | None, Mapping[str, object]], core.Measure]


@dataclasses.dataclass(frozen=True, slots=True)
class Rule:
    """A rule that a contract can name in a measure."""

    name: str
    """Name used in contracts."""
    build: Builder
    """Builds the measure from its identifier, its signal and its parameters, after subject references are resolved."""
    metrics: tuple[str, ...]
    """Keys of the metrics the measure can report; checks may only read these."""
    parameters: Mapping[str, ParamSpec] = dataclasses.field(default_factory=dict)
    """Parameters by name. A read-only copy."""
    signal: SignalUse = SignalUse.NONE
    """Whether the measure takes a signal."""
    at_least_one_of: tuple[str, ...] = ()
    """Parameters of which the contract must give at least one."""

    def __post_init__(self) -> None:
        """Replace the parameters by a read-only copy."""
        object.__setattr__(self, "parameters", types.MappingProxyType(dict(self.parameters)))


class Registry:
    """The rules a composition knows."""

    def __init__(self, rules: Iterable[Rule]) -> None:
        """Create a registry.

        Parameters
        ----------
        rules
            The rules; names must be unique.

        Raises
        ------
        ValueError
            If two rules have the same name.
        """
        self._rules: dict[str, Rule] = {}
        for rule in rules:
            if rule.name in self._rules:
                raise ValueError(f"rule '{rule.name}' is registered twice")
            self._rules[rule.name] = rule

    @property
    def names(self) -> tuple[str, ...]:
        """Rule names, sorted."""
        return tuple(sorted(self._rules))

    def find_rule(self, name: str) -> Rule | None:
        """Look a rule up.

        Parameters
        ----------
        name
            Rule name.

        Returns
        -------
        Rule | None
            The rule, or `None`.
        """
        return self._rules.get(name)
