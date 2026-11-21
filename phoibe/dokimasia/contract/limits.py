"""Resolve limits that depend on the subject (ADR-04)."""

import dataclasses
from collections.abc import Mapping

from phoibe.dokimasia import core
from phoibe.dokimasia.contract import build, schema


@dataclasses.dataclass(frozen=True, slots=True)
class ResolvedLimit:
    """A limit as a number, with where it came from."""

    value: float
    """The number."""
    text: str
    """How it came about, for example `5880 = 1.05 × rated_power_kw 5600`."""


def resolve_limit(limit: schema.Limit, specification: Mapping[str, object]) -> ResolvedLimit:
    """Resolve a limit against the specification of a subject.

    Parameters
    ----------
    limit
        A number or a reference to a subject attribute.
    specification
        The specification of the subject.

    Returns
    -------
    ResolvedLimit
        The number and a text that says where it came from.

    Raises
    ------
    KeyError
        If the attribute is not in the specification.
    TypeError
        If the attribute is not a number.

    """
    if isinstance(limit, schema.SubjectRef):
        attribute = specification[limit.attribute]
        if not isinstance(attribute, int | float) or isinstance(attribute, bool):
            raise TypeError(f"subject attribute '{limit.attribute}' is not a number: {attribute!r}")
        value = limit.factor * attribute + limit.offset
        offset = f" + {limit.offset:g}" if limit.offset else ""
        return ResolvedLimit(value, f"{value:g} = {limit.factor:g} × {limit.attribute} {attribute:g}{offset}")
    return ResolvedLimit(float(limit), f"{limit:g}")


def _walk_limits(contract: schema.Contract) -> list[schema.SubjectRef]:
    references: list[schema.SubjectRef] = []
    for layer in contract.layers:
        for measure in layer.measures:
            references.extend(value for value in measure.params.values() if isinstance(value, schema.SubjectRef))
        for check in layer.checks:
            for band in (check.good, check.acceptable):
                if band is not None:
                    references.extend(limit for limit in (band.min, band.max) if isinstance(limit, schema.SubjectRef))
    return references


def list_subject_attributes(contract: schema.Contract) -> tuple[str, ...]:
    """List the subject attributes that a contract refers to.

    Parameters
    ----------
    contract
        The contract.

    Returns
    -------
    tuple
        The attribute names, sorted and without duplicates.

    """
    return tuple(sorted({reference.attribute for reference in _walk_limits(contract)}))


def validate_subject_references(contract: schema.Contract, subject: core.Subject) -> None:
    """Check that every subject attribute that the contract refers to exists and is a number.

    Parameters
    ----------
    contract
        The contract.
    subject
        The subject whose specification the contract is applied to.

    Raises
    ------
    ContractError
        If one or more attributes are missing or not numbers; all are named.

    """
    problems = []
    for attribute in list_subject_attributes(contract):
        value = subject.specification.get(attribute)
        if attribute not in subject.specification:
            problems.append(f"subject '{subject.id}' has no attribute '{attribute}'")
        elif not isinstance(value, int | float) or isinstance(value, bool):
            problems.append(f"subject '{subject.id}': attribute '{attribute}' is not a number: {value!r}")
    if problems:
        raise build.ContractError(tuple(problems))
