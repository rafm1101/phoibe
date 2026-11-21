import os
import pathlib
import subprocess
import sys

import pytest

import phoibe.layered


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 1: registry is empty on a fresh import")
def test_factory_finds_rules_on_fresh_import(tmp_path):
    """A user who imports only the factory can build a configured rule.

    Runs in a fresh interpreter, because other tests in this process populate or clear the registry.
    """
    repository_root = pathlib.Path(phoibe.layered.__file__).parents[2]
    code = (
        "from phoibe.layered.application.factory import ValidatorFactory\n"
        "ValidatorFactory._create_rules([{'name': 'data_gaps', 'params': {}}])\n"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(repository_root)},
        capture_output=True,
        text=True,
    )

    assert process.returncode == 0, process.stderr[-500:]
