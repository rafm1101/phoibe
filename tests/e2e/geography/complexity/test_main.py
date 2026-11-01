"""`main.main(argv)` without reading rasters: usage, help, the contract command and the translation of errors."""

import pytest
import yaml

from phoibe.geography.complexity.rix import main as main_module
from phoibe.geography.complexity.rix.pipelines import errors
from phoibe.geography.complexity.rix.pipelines.publishing.contract import render_contract

RUN = ["run", "--dem", "dem.tif", "--sites", "sites.gpkg", "--output", "out"]


@pytest.mark.parametrize(
    "argv",
    [[], ["run"], [*RUN, "--profile", "nope"], [*RUN, "--references-layer", "r"], ["nonsense"]],
    ids=["no command", "missing arguments", "unknown profile", "layer without references", "unknown command"],
)
def test_main_returns_two_given_invalid_usage(argv, capsys):
    assert main_module.main(argv) == 2
    assert capsys.readouterr().err


def test_main_returns_zero_and_prints_the_usage_given_help(capsys):
    assert main_module.main(["--help"]) == 0
    assert "usage: trix" in capsys.readouterr().out


def test_main_returns_the_contract_of_the_specification_on_the_standard_output(product_spec, capsys):
    assert main_module.main(["contract"]) == 0
    assert yaml.safe_load(capsys.readouterr().out) == render_contract(product_spec)


def test_main_returns_the_contract_in_a_file_given_an_output(product_spec, tmp_path, capsys):
    path = tmp_path / "contract.yaml"
    assert main_module.main(["contract", "--output", str(path)]) == 0
    assert (
        yaml.safe_load(path.read_text(encoding="utf-8")) == render_contract(product_spec)
        and capsys.readouterr().out == ""
    )


def test_main_returns_the_contract_of_the_specification_it_was_given(make_spec, capsys):
    spec = make_spec(severities={"nan_count": "abort"})
    assert main_module.main(["contract"], spec=spec) == 0
    checks = {check["name"]: check["on_failure"] for check in yaml.safe_load(capsys.readouterr().out)["checks"]}
    assert checks["nan_count"] == "abort"


@pytest.mark.parametrize(
    "error, code",
    [
        (errors.InputRequirementError("a required input is missing"), 1),
        (errors.ConfigurationError("a parameter is unknown"), 1),
        (errors.CheckAborted([]), 1),
        (errors.PortError("a file cannot be written"), 3),
        (errors.ContractViolation(["a column is missing"]), 4),
    ],
    ids=lambda value: value if isinstance(value, int) else type(value).__name__,
)
def test_main_returns_the_exit_code_of_a_pipeline_error_and_logs_its_message(error, code, monkeypatch, caplog):
    def fail(arguments, spec):
        raise error

    monkeypatch.setattr(main_module, "_run", fail)
    assert main_module.main(RUN) == code
    assert [record.levelname for record in caplog.records] == ["ERROR"] and str(error) in caplog.text


def test_main_returns_seventy_given_an_unexpected_error_and_logs_the_traceback(monkeypatch, caplog):
    def fail(arguments, spec):
        raise RuntimeError("a defect")

    monkeypatch.setattr(main_module, "_run", fail)
    assert main_module.main(RUN) == 70
    record = caplog.records[0]
    assert (record.getMessage(), record.exc_info[0]) == ("Unexpected error", RuntimeError)
