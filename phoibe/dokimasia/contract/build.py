"""Build a contract from decoded data, collecting every problem before failing (ADR-04, ADR-12).

The data is what a YAML or JSON loader returns. It is untrusted: a hand-edited file may be wrong in many places, and
a person should see them all at once.
"""

from collections.abc import Mapping

from phoibe.dokimasia import core
from phoibe.dokimasia.contract import schema


class ContractError(ValueError):
    """The contract is invalid. `problems` holds every problem found, each with the path where it was found."""

    def __init__(self, problems: tuple[str, ...]) -> None:
        """Create the error from all problems."""
        self.problems = problems
        super().__init__("invalid contract:\n" + "\n".join(f"- {problem}" for problem in problems))


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _describe(value: object) -> str:
    return f"{value!r}" if isinstance(value, str | int | float | bool | type(None)) else type(value).__name__


class _Collector:
    """Collects problems with their paths."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def add(self, path: str, text: str) -> None:
        self.problems.append(f"{path}: {text}")

    def expect_keys(self, path: str, data: Mapping[str, object], required: set[str], optional: set[str]) -> None:
        for key in sorted(required - set(data)):
            self.add(path, f"missing key '{key}'")
        for key in sorted(set(data) - required - optional):
            self.add(path, f"unknown key '{key}'; allowed: {', '.join(sorted(required | optional))}")

    def read_mapping(self, path: str, value: object) -> Mapping[str, object] | None:
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            self.add(path, f"must be a mapping with text keys, got {_describe(value)}")
            return None
        return value

    def read_text(self, path: str, value: object) -> str | None:
        if not isinstance(value, str) or not value.strip():
            self.add(path, f"must be non-empty text, got {_describe(value)}")
            return None
        return value

    def read_number(self, path: str, value: object) -> float | None:
        if not _is_number(value):
            self.add(path, f"must be a number, got {_describe(value)}")
            return None
        return float(value)  # type: ignore[arg-type]


def _read_limit(collector: _Collector, path: str, value: object) -> schema.Limit | None:
    if _is_number(value):
        return float(value)  # type: ignore[arg-type]
    if isinstance(value, Mapping) and "subject" in value:
        mapping = collector.read_mapping(path, value)
        if mapping is None:
            return None
        collector.expect_keys(path, mapping, {"subject"}, {"factor", "offset"})
        attribute = collector.read_text(f"{path}.subject", mapping.get("subject"))
        factor = collector.read_number(f"{path}.factor", mapping["factor"]) if "factor" in mapping else 1.0
        offset = collector.read_number(f"{path}.offset", mapping["offset"]) if "offset" in mapping else 0.0
        if attribute is None or factor is None or offset is None:
            return None
        return schema.SubjectRef(attribute, factor, offset)
    collector.add(path, f"must be a number or {{subject: <attribute>, factor, offset}}, got {_describe(value)}")
    return None


def _read_band(collector: _Collector, path: str, value: object) -> schema.Band | None:
    mapping = collector.read_mapping(path, value)
    if mapping is None:
        return None
    collector.expect_keys(path, mapping, set(), {"min", "max"})
    if not mapping:
        collector.add(path, "needs at least one of 'min' and 'max'")
        return None
    lower = _read_limit(collector, f"{path}.min", mapping["min"]) if "min" in mapping else None
    upper = _read_limit(collector, f"{path}.max", mapping["max"]) if "max" in mapping else None
    if ("min" in mapping and lower is None) or ("max" in mapping and upper is None):
        return None
    if isinstance(lower, float) and isinstance(upper, float) and lower > upper:
        collector.add(path, f"min {lower:g} is above max {upper:g}")
        return None
    return schema.Band(lower, upper)


def _read_param(collector: _Collector, path: str, value: object) -> object:
    if isinstance(value, Mapping) and "subject" in value:
        return _read_limit(collector, path, value)
    return value


def _read_measure(collector: _Collector, path: str, key: str, value: object) -> schema.MeasureSpec | None:
    mapping = collector.read_mapping(path, {} if value is None else value)
    if mapping is None:
        return None
    collector.expect_keys(path, mapping, set(), {"rule", "signal", "params"})
    rule = collector.read_text(f"{path}.rule", mapping["rule"]) if "rule" in mapping else key
    signal = collector.read_text(f"{path}.signal", mapping["signal"]) if "signal" in mapping else None
    params: dict[str, object] = {}
    if "params" in mapping:
        raw = collector.read_mapping(f"{path}.params", mapping["params"])
        if raw is not None:
            params = {name: _read_param(collector, f"{path}.params.{name}", item) for name, item in raw.items()}
    if rule is None or ("signal" in mapping and signal is None):
        return None
    return schema.MeasureSpec(id=key, rule=rule, signal=signal, params=params)


def _read_check(
    collector: _Collector, path: str, key: str, value: object, measure_keys: set[str] | None
) -> schema.CheckSpec | None:
    mapping = collector.read_mapping(path, value)
    if mapping is None:
        return None
    collector.expect_keys(path, mapping, {"measure", "value", "severity", "good"}, {"acceptable", "clause", "signal"})
    measure = collector.read_text(f"{path}.measure", mapping["measure"]) if "measure" in mapping else None
    if measure is not None and measure_keys is not None and measure not in measure_keys:
        collector.add(f"{path}.measure", f"unknown measure '{measure}'")
    metric = collector.read_text(f"{path}.value", mapping["value"]) if "value" in mapping else None
    severity = None
    if "severity" in mapping:
        raw = mapping["severity"]
        allowed = ", ".join(item.value for item in core.Severity)
        if isinstance(raw, str) and raw in {item.value for item in core.Severity}:
            severity = core.Severity(raw)
        else:
            collector.add(f"{path}.severity", f"must be one of {allowed}, got {_describe(raw)}")
    good = _read_band(collector, f"{path}.good", mapping["good"]) if "good" in mapping else None
    acceptable = _read_band(collector, f"{path}.acceptable", mapping["acceptable"]) if "acceptable" in mapping else None
    clause = collector.read_text(f"{path}.clause", mapping["clause"]) if "clause" in mapping else None
    signal = collector.read_text(f"{path}.signal", mapping["signal"]) if "signal" in mapping else None
    if good is None or measure is None or metric is None or severity is None:
        return None
    if "acceptable" in mapping and acceptable is None:
        return None
    if acceptable is not None:
        _check_acceptable_contains_good(collector, path, good, acceptable)
    return schema.CheckSpec(key, measure, metric, severity, good, acceptable, clause, signal)


def _check_acceptable_contains_good(
    collector: _Collector, path: str, good: schema.Band, acceptable: schema.Band
) -> None:
    sides = (("min", acceptable.min, good.min, 1.0), ("max", acceptable.max, good.max, -1.0))
    for name, wide, narrow, direction in sides:
        if narrow is not None and wide is None:
            continue
        if narrow is None and wide is not None:
            collector.add(path, f"the acceptable band has a '{name}' but the good band has none, so it is narrower")
        elif isinstance(wide, float) and isinstance(narrow, float) and direction * (wide - narrow) > 0:
            collector.add(path, f"the good band lies outside the acceptable band at '{name}'")


def _read_layer(collector: _Collector, path: str, name: str, value: object) -> schema.LayerSpec | None:
    mapping = collector.read_mapping(path, value)
    if mapping is None:
        return None
    collector.expect_keys(path, mapping, {"measures", "checks"}, set())
    measures_data = (
        collector.read_mapping(f"{path}.measures", mapping.get("measures")) if "measures" in mapping else None
    )
    checks_data = collector.read_mapping(f"{path}.checks", mapping.get("checks")) if "checks" in mapping else None
    measure_keys = None if measures_data is None else set(measures_data)
    measures = [
        measure
        for key, item in (measures_data or {}).items()
        if (measure := _read_measure(collector, f"{path}.measures.{key}", key, item)) is not None
    ]
    checks = [
        check
        for key, item in (checks_data or {}).items()
        if (check := _read_check(collector, f"{path}.checks.{key}", key, item, measure_keys)) is not None
    ]
    if measures_data is not None and not measures_data:
        collector.add(f"{path}.measures", "a layer needs at least one measure")
    if checks_data is not None and not checks_data:
        collector.add(f"{path}.checks", "a layer needs at least one check; a layer without criteria is not a pass")
    return schema.LayerSpec(name=name, measures=tuple(measures), checks=tuple(checks))


def build_contract(data: object) -> schema.Contract:
    """Build a contract from decoded YAML or JSON data.

    Every problem is collected with its path, so that a hand-edited file is corrected in one pass. The structure is
    checked here; rule names and parameters are checked against the registry where the contract is composed with
    rules, and subject references against a subject by `validate_subject_references`.

    Parameters
    ----------
    data
        A mapping with the keys `id`, `version`, `domain` and `layers`. A measure may omit `rule` (the key is the
        rule). A check has `measure`, `value`, `severity` and `good`, and may have `acceptable`, `clause` and
        `signal`. A limit is a number or `{subject: <attribute>, factor: <number>, offset: <number>}`.

    Returns
    -------
    Contract
        The contract.

    Raises
    ------
    ContractError
        If anything is wrong; the error lists all problems.

    """
    collector = _Collector()
    mapping = collector.read_mapping("contract", data)
    layers: list[schema.LayerSpec] = []
    contract_id = version = domain = None
    if mapping is not None:
        collector.expect_keys("contract", mapping, {"id", "version", "domain", "layers"}, set())
        contract_id = collector.read_text("id", mapping["id"]) if "id" in mapping else None
        version = collector.read_text("version", mapping["version"]) if "version" in mapping else None
        domain = collector.read_text("domain", mapping["domain"]) if "domain" in mapping else None
        layers_data = collector.read_mapping("layers", mapping.get("layers")) if "layers" in mapping else None
        for name, item in (layers_data or {}).items():
            layer = _read_layer(collector, f"layers.{name}", name, item)
            if layer is not None:
                layers.append(layer)
        if layers_data is not None and not layers_data:
            collector.add("layers", "a contract needs at least one layer")
    if collector.problems or contract_id is None or version is None or domain is None:
        raise ContractError(tuple(collector.problems))
    return schema.Contract(id=contract_id, version=version, domain=domain, layers=tuple(layers))
