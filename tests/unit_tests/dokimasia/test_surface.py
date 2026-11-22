"""The surface of `phoibe.dokimasia`: what a run calls, and that importing it has no side effects."""

import logging
import os
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

from phoibe import dokimasia

ROOT = pathlib.Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks" / "02-demonstrate-dokimasia.ipynb"


def test_the_package_logs_to_a_null_handler_only():
    handlers = logging.getLogger("phoibe.dokimasia").handlers

    assert [type(handler) for handler in handlers] == [logging.NullHandler]


def test_importing_the_package_prints_nothing_and_configures_no_logging():
    code = (
        "import logging, phoibe.dokimasia\n"
        "assert logging.getLogger().handlers == [], logging.getLogger().handlers\n"
        "assert logging.getLogger().level == logging.WARNING\n"
    )
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}

    completed = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=False, cwd=ROOT
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_validate_layer_without_a_registry_uses_the_generic_rules():
    contract = dokimasia.adapters.parse_contract_yaml(
        """\
id: c
version: 1.0.0
domain: d
layers:
  raw:
    measures:
      time_gaps:
        params: {expected_step_s: 600}
    checks:
      availability:
        measure: time_gaps
        value: share_missing_steps
        severity: critical
        good: {max: 0}
"""
    )
    dataset = dokimasia.Dataset.from_frame(
        pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=pd.date_range("2026-01-01", periods=3, freq="10min"))
    )

    validation = dokimasia.validate_layer(dataset, dokimasia.Subject("t", "d"), contract, "raw")

    assert validation.result.outcome == dokimasia.core.GateOutcome.PASS


def test_the_notebook_walks_from_raw_to_bronze_and_nothing_else_happens(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    notebook = nbformat.read(NOTEBOOK, as_version=4)

    nbclient.NotebookClient(
        notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    ).execute()

    streams = [
        (output["name"], output["text"])
        for cell in notebook.cells
        if cell.cell_type == "code"
        for output in cell.outputs
        if output["output_type"] == "stream"
    ]
    assert streams == [], "a cell printed or logged something"
    assert list(tmp_path.iterdir()) == [], "the notebook wrote files next to itself"
