"""Read and write contracts as YAML (ADR-04, ADR-10)."""

import pathlib

import yaml

from phoibe.dokimasia import contract


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe YAML loader that refuses a key that appears twice, instead of silently keeping the last."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[object, object]:  # noqa: FBT001, FBT002
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def parse_contract_yaml(text: str) -> contract.Contract:
    """Parse a contract from YAML text.

    Parameters
    ----------
    text
        YAML text. A key that appears twice is an error.

    Returns
    -------
    Contract
        The contract.

    Raises
    ------
    ContractError
        If the text is not valid YAML or the contract is invalid; all contract problems are listed.
    """
    try:
        data = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as error:
        raise contract.ContractError((f"yaml: {error!s}",)) from error
    return contract.build_contract(data)


def load_contract(path: pathlib.Path) -> contract.Contract:
    """Load a contract from a YAML file.

    Parameters
    ----------
    path
        Path of the file.

    Returns
    -------
    Contract
        The contract.

    Raises
    ------
    ContractError
        As `parse_contract_yaml`.
    OSError
        If the file cannot be read.
    """
    return parse_contract_yaml(path.read_text(encoding="utf-8"))


def _limit_to_data(limit: contract.Limit) -> float | dict[str, object]:
    if isinstance(limit, contract.SubjectRef):
        data: dict[str, object] = {"subject": limit.attribute}
        if limit.factor != 1.0:
            data["factor"] = limit.factor
        if limit.offset != 0.0:
            data["offset"] = limit.offset
        return data
    return limit


def _band_to_data(band: contract.Band) -> dict[str, object]:
    data: dict[str, object] = {}
    if band.min is not None:
        data["min"] = _limit_to_data(band.min)
    if band.max is not None:
        data["max"] = _limit_to_data(band.max)
    return data


def _param_to_data(value: object) -> object:
    if isinstance(value, contract.SubjectRef):
        return _limit_to_data(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def contract_to_data(contract_: contract.Contract) -> dict[str, object]:
    """Map a contract to plain data, the inverse of `build_contract`.

    Parameters
    ----------
    contract_
        The contract.

    Returns
    -------
    dict
        Plain data with text keys, in contract order, ready for YAML or JSON.
    """
    layers: dict[str, object] = {}
    for layer in contract_.layers:
        measures: dict[str, object] = {}
        for measure in layer.measures:
            entry: dict[str, object] = {}
            if measure.rule != measure.id:
                entry["rule"] = measure.rule
            if measure.signal is not None:
                entry["signal"] = measure.signal
            if measure.params:
                entry["params"] = {name: _param_to_data(value) for name, value in measure.params.items()}
            measures[measure.id] = entry or None
        checks: dict[str, object] = {}
        for check in layer.checks:
            item: dict[str, object] = {"measure": check.measure, "value": check.value}
            if check.signal is not None:
                item["signal"] = check.signal
            item["severity"] = check.severity.value
            item["good"] = _band_to_data(check.good)
            if check.acceptable is not None:
                item["acceptable"] = _band_to_data(check.acceptable)
            if check.clause is not None:
                item["clause"] = check.clause
            checks[check.id] = item
        layers[layer.name] = {"measures": measures, "checks": checks}
    return {"id": contract_.id, "version": contract_.version, "domain": contract_.domain, "layers": layers}


def dump_contract_yaml(contract_: contract.Contract) -> str:
    """Write a contract as YAML text.

    Parameters
    ----------
    contract_
        The contract.

    Returns
    -------
    str
        YAML text that `parse_contract_yaml` reads back to an equal contract.
    """
    return yaml.safe_dump(contract_to_data(contract_), sort_keys=False, allow_unicode=True)


def save_contract(contract_: contract.Contract, path: pathlib.Path) -> None:
    """Write a contract to a YAML file.

    Parameters
    ----------
    contract_
        The contract.
    path
        Path of the file; an existing file is replaced.
    """
    path.write_text(dump_contract_yaml(contract_), encoding="utf-8")
