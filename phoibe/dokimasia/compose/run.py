"""Compose profile and judge from a contract: the only place that knows both (ADR-07)."""

import dataclasses
import datetime

from phoibe.dokimasia import adapters, contract, core
from phoibe.dokimasia.compose import defaults, validation
from phoibe.dokimasia.compose import registry as registry_module


@dataclasses.dataclass(frozen=True, slots=True)
class Validation:
    """A profile and the judgment of it, with the contract that was applied."""

    contract_id: str
    """Identifier of the contract."""
    contract_version: str
    """Version of the contract."""
    layer: str
    """Layer name."""
    profile: core.Profile
    """What was measured."""
    result: core.GateResult
    """Verdicts and gate decision."""


def build_measures(
    layer: contract.LayerSpec, registry: registry_module.Registry, subject: core.Subject
) -> tuple[core.Measure, ...]:
    """Build the measures of a layer, with limits that depend on the subject resolved.

    Parameters
    ----------
    layer
        The layer.
    registry
        The rules that are known.
    subject
        The subject; its specification resolves subject references.

    Returns
    -------
    tuple
        The measures in contract order, each with the contract key as its id.

    Raises
    ------
    ContractError
        If a rule or parameter is wrong, or a subject attribute is missing or not a number; all problems are listed.
    """
    validation.validate_layer_rules(layer, registry)
    problems: list[str] = []
    measures: list[core.Measure] = []
    for spec in layer.measures:
        rule = registry.find_rule(spec.rule)
        assert rule is not None
        params: dict[str, object] = {}
        for name, value in spec.params.items():
            if isinstance(value, contract.SubjectRef):
                try:
                    params[name] = contract.resolve_limit(value, subject.specification).value
                except (KeyError, TypeError) as error:
                    problems.append(f"layers.{layer.name}.measures.{spec.id}.params.{name}: cannot resolve: {error!s}")
            else:
                params[name] = value
        if not problems:
            measures.append(rule.build(spec.id, spec.signal, params))
    if problems:
        raise contract.ContractError(tuple(problems))
    return tuple(measures)


def profile_layer(
    dataset: core.Dataset,
    subject: core.Subject,
    layer: contract.LayerSpec,
    registry: registry_module.Registry,
) -> core.Profile:
    """Measure a dataset with the measures of a layer. No verdicts are made.

    Parameters
    ----------
    dataset
        The data.
    subject
        The subject.
    layer
        The layer of the contract.
    registry
        The rules that are known.

    Returns
    -------
    Profile
        The profile.

    Raises
    ------
    ContractError
        As `build_measures`.
    """
    return core.profile_dataset(dataset, subject, layer.name, build_measures(layer, registry, subject))


def validate_layer(
    dataset: core.Dataset,
    subject: core.Subject,
    contract_: contract.Contract,
    layer_name: str,
    registry: registry_module.Registry | None = None,
    *,
    policy: core.GatePolicy | None = None,
) -> Validation:
    """Profile and judge in one step. The profile and the judgment stay separate objects.

    Parameters
    ----------
    dataset
        The data.
    subject
        The subject.
    contract_
        The contract.
    layer_name
        Name of the layer to apply.
    registry
        The rules that are known; the generic rules if `None`.
    policy
        Gate policy; the worst effect decides if `None`.

    Returns
    -------
    Validation
        Profile, verdicts and gate decision, with the contract id and version.

    Raises
    ------
    KeyError
        If the contract has no such layer.
    ContractError
        As `build_measures`.
    """
    layer = contract_.get_layer(layer_name)
    profile = profile_layer(
        dataset, subject, layer, defaults.create_default_registry() if registry is None else registry
    )
    result = contract.judge_layer(profile, layer, policy=policy)
    return Validation(contract_.id, contract_.version, layer_name, profile, result)


def attest_validation(
    validation: Validation, *, created_at: datetime.datetime, software_version: str
) -> core.Attestation:
    """Attest a passed gate. The parent is the attestation of the layer the data came from, if it advanced.

    Parameters
    ----------
    validation
        The profile and judgment. The gate must not have failed.
    created_at
        Time of the attestation, timezone-aware UTC.
    software_version
        Version of the software that judged.

    Returns
    -------
    Attestation
        The attestation, bound to the fingerprint of the data that was measured.

    Raises
    ------
    ValueError
        If the gate failed or `created_at` is not UTC.
    """
    profile = validation.profile
    return core.build_attestation(
        subject=profile.subject,
        layer=validation.layer,
        fingerprint=profile.fingerprint,
        contract_id=validation.contract_id,
        contract_version=validation.contract_version,
        result=validation.result,
        created_at=created_at,
        software_version=software_version,
        parent_id=profile.provenance.parent_attestation_id,
    )


def build_run(validation: Validation, attestation: core.Attestation | None = None) -> adapters.Run:
    """Collect profile, verdicts and attestation of one validation as a run, ready for a store.

    Parameters
    ----------
    validation
        The profile and judgment.
    attestation
        The attestation of the passed gate, if there is one.

    Returns
    -------
    Run
        The run.

    Raises
    ------
    ValueError
        If the artifacts do not belong together, for example an attestation of another dataset version.
    """
    profile = validation.profile
    verdicts = adapters.VerdictsRecord(
        subject_id=profile.subject.id,
        layer=validation.layer,
        fingerprint=profile.fingerprint,
        contract_id=validation.contract_id,
        contract_version=validation.contract_version,
        result=validation.result,
    )
    return adapters.Run(profile, verdicts, attestation)
