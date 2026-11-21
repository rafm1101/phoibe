"""Validate the rules and parameters of a contract against the registry, at load time (ADR-04)."""

from phoibe.dokimasia import contract
from phoibe.dokimasia.compose import registry as registry_module


def _check_parameter(path: str, name: str, value: object, spec: registry_module.ParamSpec, problems: list[str]) -> None:
    kind = registry_module.ParamKind
    is_number = isinstance(value, int | float) and not isinstance(value, bool)
    if spec.kind == kind.NUMBER and not is_number:
        problems.append(f"{path}.{name}: must be a number, got {value!r}")
    elif spec.kind == kind.NUMBER_OR_SUBJECT and not (is_number or isinstance(value, contract.SubjectRef)):
        problems.append(f"{path}.{name}: must be a number or a subject reference, got {value!r}")
    elif spec.kind == kind.TEXT and not isinstance(value, str):
        problems.append(f"{path}.{name}: must be text, got {value!r}")
    elif spec.kind == kind.TEXT_LIST and not (
        isinstance(value, list | tuple) and value and all(isinstance(item, str) for item in value)
    ):
        problems.append(f"{path}.{name}: must be a non-empty list of text, got {value!r}")


def _check_measure(
    path: str, measure: contract.MeasureSpec, registry: registry_module.Registry, problems: list[str]
) -> registry_module.Rule | None:
    rule = registry.find_rule(measure.rule)
    if rule is None:
        problems.append(f"{path}.rule: unknown rule '{measure.rule}'; known: {', '.join(registry.names)}")
        return None
    if rule.signal == registry_module.SignalUse.REQUIRED and measure.signal is None:
        problems.append(f"{path}: rule '{rule.name}' needs a 'signal'")
    if rule.signal == registry_module.SignalUse.NONE and measure.signal is not None:
        problems.append(f"{path}.signal: rule '{rule.name}' takes no signal")
    for name in sorted(set(measure.params) - set(rule.parameters)):
        known = ", ".join(sorted(rule.parameters)) or "none"
        problems.append(f"{path}.params.{name}: unknown parameter of rule '{rule.name}'; known: {known}")
    for name, spec in rule.parameters.items():
        if name in measure.params:
            _check_parameter(f"{path}.params", name, measure.params[name], spec, problems)
        elif spec.required:
            problems.append(f"{path}.params: missing parameter '{name}' of rule '{rule.name}'")
    if rule.at_least_one_of and not any(name in measure.params for name in rule.at_least_one_of):
        problems.append(f"{path}.params: rule '{rule.name}' needs at least one of {', '.join(rule.at_least_one_of)}")
    return rule


def validate_layer_rules(layer: contract.LayerSpec, registry: registry_module.Registry) -> None:
    """Check that the rules, parameters and read values of a layer exist and have the right types.

    Parameters
    ----------
    layer
        The layer.
    registry
        The rules that are known.

    Raises
    ------
    ContractError
        If anything is wrong; the error lists all problems. A check may only read a metric that the rule of its
        measure reports.
    """
    problems: list[str] = []
    rules = {}
    for measure in layer.measures:
        path = f"layers.{layer.name}.measures.{measure.id}"
        rules[measure.id] = _check_measure(path, measure, registry, problems)
    for check in layer.checks:
        rule = rules.get(check.measure)
        if rule is not None and check.value not in rule.metrics:
            path = f"layers.{layer.name}.checks.{check.id}.value"
            problems.append(
                f"{path}: rule '{rule.name}' reports no '{check.value}'; it reports: {', '.join(rule.metrics)}"
            )
    if problems:
        raise contract.ContractError(tuple(problems))
