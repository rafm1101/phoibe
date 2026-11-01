"""D-A3 and D-A5: the contract is deterministic and lossless in YAML, each release has a snapshot.

A change of the contract needs a version bump. Regenerate a snapshot deliberately with UPDATE_SNAPSHOTS=1 and review
the diff. `required_bump` below is a helper of these tests: extend it when the contract grows (ADR 0008).
"""

import copy
import itertools
import os
import pathlib

import yaml

from phoibe.geography.complexity.rix.pipelines.publishing.contract import render_contract

SNAPSHOTS = pathlib.Path(__file__).parent / "snapshots"
ORDER = {"none": 0, "patch": 1, "minor": 2, "major": 3}


def required_bump(old, new):
    """ "major", "minor", "patch" or "none" for the change from contract `old` to `new`."""
    major, minor = [], []
    for name, item in old["inputs"].items():
        if name not in new["inputs"]:
            major.append(f"input {name} removed")
        elif new["inputs"][name]["required"] and not item["required"]:
            major.append(f"input {name} became required")
    minor += [f"input {n} added" for n in new["inputs"] if n not in old["inputs"]]

    for group, parameters in old["parameters"].items():
        for name, item in parameters.items():
            now = new["parameters"].get(group, {}).get(name)
            if now is None:
                major.append(f"parameter {group}.{name} removed")
            elif item.get("value") != now.get("value") or item["locked"] != now["locked"]:
                major.append(f"parameter {group}.{name}: value or lock changed")
    for group, parameters in new["parameters"].items():
        minor += [f"parameter {group}.{n} added" for n in parameters if n not in old["parameters"].get(group, {})]

    for key, output in old["outputs"].items():
        now = new["outputs"].get(key)
        if now is None or now["file"] != output["file"]:
            major.append(f"output {key} removed or renamed")
            continue
        before, after = [c["name"] for c in output["columns"]], [c["name"] for c in now["columns"]]
        if [c for c in before if c not in after] or [c for c in before if c in after] != [
            c for c in after if c in before
        ]:
            major.append(f"output {key}: column removed or reordered")
        elif after != before:
            minor.append(f"output {key}: column added")
        for name, layer in output.get("layers", {}).items():
            layer_now = now.get("layers", {}).get(name)
            if layer_now is None or layer_now["name"] != layer["name"]:
                major.append(f"output {key}: layer {name} removed or renamed")
        minor += [f"output {key}: layer {n} added" for n in now.get("layers", {}) if n not in output.get("layers", {})]
    minor += [f"output {k} added" for k in new["outputs"] if k not in old["outputs"]]

    if major:
        return "major"
    if minor:
        return "minor"
    return "patch" if old != new else "none"


def bump_between(old_version, new_version):
    old, new = (tuple(int(part) for part in version.split(".")) for version in (old_version, new_version))
    for level, name in enumerate(("major", "minor", "patch")):
        if old[level : level + 1] != new[level : level + 1]:
            return name
    return "none"


def test_contract_is_deterministic_and_lossless(product_spec):
    contract = render_contract(product_spec)
    assert contract == render_contract(product_spec)
    assert yaml.safe_load(yaml.safe_dump(contract, sort_keys=False)) == contract
    assert isinstance(contract["version"], str)


def test_contract_equals_the_snapshot_of_its_version(product_spec):
    """Regenerate deliberately with UPDATE_SNAPSHOTS=1 and review the diff."""
    path = SNAPSHOTS / f"contract_{product_spec.version}.yaml"
    contract = render_contract(product_spec)
    if os.environ.get("UPDATE_SNAPSHOTS"):
        SNAPSHOTS.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8")
    assert path.exists(), f"no snapshot for version {product_spec.version}; create it with UPDATE_SNAPSHOTS=1"
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == contract, "the contract changed without a new version"


def test_each_version_raises_its_number_at_least_as_much_as_its_contract_requires():
    snapshots = sorted(
        SNAPSHOTS.glob("contract_*.yaml"),
        key=lambda p: tuple(int(x) for x in p.stem.removeprefix("contract_").split(".")),
    )
    loaded = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in snapshots]
    for old, new in itertools.pairwise(loaded):
        needed, actual = required_bump(old, new), bump_between(old["version"], new["version"])
        assert (
            ORDER[actual] >= ORDER[needed]
        ), f"{old['version']} -> {new['version']} is a {actual} bump, the change needs {needed}"


def test_bump_between_names_the_first_part_that_differs():
    assert [bump_between("2.0.0", v) for v in ("3.0.0", "2.1.0", "2.0.1", "2.0.0")] == [
        "major",
        "minor",
        "patch",
        "none",
    ]


def test_required_bump(product_spec):
    base = render_contract(product_spec)

    def changed(edit):
        contract = copy.deepcopy(base)
        edit(contract)
        return required_bump(base, contract)

    layer = next(iter(next(o for o in base["outputs"].values() if o.get("layers"))["layers"]))
    container = next(key for key, o in base["outputs"].items() if o.get("layers"))
    assert required_bump(base, base) == "none"
    assert changed(lambda c: c["outputs"]["rix_table"]["columns"].pop()) == "major"
    assert changed(lambda c: c["outputs"]["rix_table"]["columns"].append({"name": "x"})) == "minor"
    assert changed(lambda c: c["outputs"]["rix_table"]["columns"].reverse()) == "major"
    assert changed(lambda c: c["outputs"]["rix_table"].update(file="other.csv")) == "major"
    assert changed(lambda c: c["outputs"][container]["layers"][layer].update(name="renamed")) == "major"
    assert changed(lambda c: c["outputs"][container]["layers"].pop(layer)) == "major"
    assert (
        changed(lambda c: c["outputs"][container]["layers"].update(extra={"name": "extra", "columns": []})) == "minor"
    )
    assert changed(lambda c: c["outputs"].update(extra={"file": "e.csv", "columns": []})) == "minor"
    assert changed(lambda c: c["outputs"].pop("rix_table")) == "major"
    assert changed(lambda c: c["parameters"]["ray"]["R_km"].update(value=5.0)) == "major"
    assert changed(lambda c: c["parameters"]["ray"]["R_km"].update(locked=False)) == "major"
    assert changed(lambda c: c["parameters"]["ray"].pop("R_km")) == "major"
    assert changed(lambda c: c["parameters"]["ray"].update(extra={"locked": False})) == "minor"
    assert changed(lambda c: c["inputs"]["dem"].update(description="reworded")) == "patch"
    assert changed(lambda c: c["inputs"].pop("dem")) == "major"
    assert changed(lambda c: c["inputs"]["locations_reference"].update(required=True)) == "major"
    assert changed(lambda c: c["inputs"].update(extra={"required": False})) == "minor"
