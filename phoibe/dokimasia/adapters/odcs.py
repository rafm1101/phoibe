"""One-way export of a contract to the Open Data Contract Standard (ADR-10).

The export is lossy by design. ODCS can say which signals a layer has and attach a quality rule to the layer, but it
executes nothing and has no good and acceptable bands. So each check becomes one rule of `type: custom` with the engine
`dokimasia`, which names its measure in `implementation`. The standard's schema allows no comparison operator on a
custom rule, so the thresholds are in the description for people and in the `dokimasia.good` and
`dokimasia.acceptable` custom properties for tools. Everything else ODCS cannot hold travels as `dokimasia.*` custom
properties too. An ODCS tool can read the file; it cannot run the time-series rules, and nothing here reads the
file back. The contract in Dokimasia's own schema stays the source.
"""

import pathlib

from phoibe.dokimasia import contract as contract_module
from phoibe.dokimasia import core
from phoibe.dokimasia.adapters import _text, contract_files

ODCS_API_VERSION = "v3.2.0"
"""Version of the standard the export is written for."""
EXPORT_VERSION = 1
"""Version of the `dokimasia.*` custom properties; it is written into the file."""
ENGINE = "dokimasia"
"""Value of `engine` in the quality rules."""

_REQUIRED_SIGNALS_RULE = "required_signals"
_SEVERITY = {"critical": "error", "warning": "warning", "info": "info"}


def _custom(property_name: str, value: object) -> dict[str, object]:
    return {"property": f"dokimasia.{property_name}", "value": value}


def _describe_limit(limit: object) -> str:
    if isinstance(limit, dict):
        text = str(limit["subject"])
        if "factor" in limit:
            text += f" * {limit['factor']:g}"
        if "offset" in limit:
            text += f" + {limit['offset']:g}"
        return text
    return f"{limit:g}" if isinstance(limit, int | float) else str(limit)


def _describe_band(band: dict[str, object]) -> str:
    return ", ".join(f"{key} {_describe_limit(limit)}" for key, limit in band.items()) or "no limit"


def _build_rule(check_id: str, check: dict[str, object]) -> dict[str, object]:
    good = check["good"]
    assert isinstance(good, dict)
    severity = str(check["severity"])
    rule: dict[str, object] = {
        "name": check_id,
        "type": "custom",
        "engine": ENGINE,
        "implementation": str(check["measure"]),
        "description": f"{check['value']} of {check['measure']}; good: {_describe_band(good)}",
        "severity": _SEVERITY[severity],
    }
    properties = [
        _custom("measure", check["measure"]),
        _custom("value", check["value"]),
        _custom("severity", severity),
        _custom("good", good),
    ]
    properties.extend(_custom(key, check[key]) for key in ("acceptable", "clause", "signal") if key in check)
    rule["customProperties"] = properties
    return rule


def _list_signals(measures: dict[str, object], checks: dict[str, object]) -> tuple[list[str], set[str]]:
    """List the signals a layer names, in order of first appearance, and the ones it requires."""
    names: dict[str, None] = {}
    required: set[str] = set()
    for measure_id, entry in measures.items():
        spec = entry if isinstance(entry, dict) else {}
        found: list[str] = []
        if isinstance(spec.get("signal"), str):
            found.append(str(spec["signal"]))
        params = spec.get("params")
        if isinstance(params, dict) and isinstance(params.get("signals"), list | tuple):
            found.extend(str(name) for name in params["signals"])
        names.update(dict.fromkeys(found))
        if spec.get("rule", measure_id) == _REQUIRED_SIGNALS_RULE:
            required.update(found)
    for check in checks.values():
        if isinstance(check, dict) and isinstance(check.get("signal"), str):
            names[str(check["signal"])] = None
    return list(names), required


def _build_property(name: str, *, required: bool, catalog: core.Catalog | None) -> dict[str, object]:
    schema_property: dict[str, object] = {"name": name, "logicalType": "number", "required": required}
    signal = None if catalog is None else catalog.find_signal(name)
    if signal is not None:
        custom = [_custom("quantity", signal.quantity), _custom("statistic", signal.statistic.value)]
        if signal.unit is not None:
            custom.append(_custom("unit", signal.unit))
        schema_property["customProperties"] = custom
    return schema_property


def export_odcs(
    contract_: contract_module.Contract, *, catalog: core.Catalog | None = None, status: str = "draft"
) -> dict[str, object]:
    """Export a contract as ODCS plain data.

    Each layer becomes a schema object. Its properties are the signals the layer names (in a measure's `signal` or
    `params.signals`, or in a check), required where a `required_signals` measure asks for them. Each check becomes a
    custom quality rule of the layer.

    Parameters
    ----------
    contract_
        The contract.
    catalog
        If given, signals that are in it carry quantity, statistic and unit as `dokimasia.*` custom properties.
    status
        The `status` of the ODCS contract. Nothing in a contract says whether it is in force, so the default is the
        modest `draft`.

    Returns
    -------
    dict
        Plain data in a fixed key order, ready for YAML or JSON. The bands, severity names, clause, signal and measure
        definitions are kept in `dokimasia.*` custom properties. The file is valid against the JSON schema of ODCS.
    """
    layers = contract_files.contract_to_data(contract_)["layers"]
    assert isinstance(layers, dict)
    schema: list[dict[str, object]] = []
    for layer_name, layer in layers.items():
        measures, checks = layer["measures"], layer["checks"]
        names, required = _list_signals(measures, checks)
        schema.append(
            {
                "name": layer_name,
                "logicalType": "object",
                "properties": [_build_property(name, required=name in required, catalog=catalog) for name in names],
                "quality": [_build_rule(check_id, check) for check_id, check in checks.items()],
                "customProperties": [_custom("measures", measures)],
            }
        )
    return {
        "apiVersion": ODCS_API_VERSION,
        "kind": "DataContract",
        "id": contract_.id,
        "name": contract_.id,
        "version": contract_.version,
        "status": status,
        "domain": contract_.domain,
        "schema": schema,
        "customProperties": [_custom("export_version", EXPORT_VERSION)],
    }


def format_odcs(
    contract_: contract_module.Contract,
    *,
    catalog: core.Catalog | None = None,
    status: str = "draft",
    fmt: _text.Format = "yaml",
) -> str:
    """Export a contract as ODCS text.

    Parameters
    ----------
    contract_
        The contract.
    catalog
        See `export_odcs`.
    status
        See `export_odcs`.
    fmt
        `yaml` (default) or `json`.

    Returns
    -------
    str
        The text.
    """
    return _text.format_text(export_odcs(contract_, catalog=catalog, status=status), fmt)


def save_odcs(
    contract_: contract_module.Contract,
    path: pathlib.Path,
    *,
    catalog: core.Catalog | None = None,
    status: str = "draft",
) -> None:
    """Write a contract as an ODCS file; the format follows the suffix, `.json` or else YAML.

    Parameters
    ----------
    contract_
        The contract.
    path
        The file. An existing file is replaced; this is an export, not an artifact of a run.
    catalog
        See `export_odcs`.
    status
        See `export_odcs`.

    Raises
    ------
    OSError
        If the file cannot be written.
    """
    fmt: _text.Format = "json" if path.suffix == ".json" else "yaml"
    path.write_text(format_odcs(contract_, catalog=catalog, status=status, fmt=fmt), encoding="utf-8")
