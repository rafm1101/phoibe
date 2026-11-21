"""Shared helpers of the characterisation tests.

The characterisation tests are a temporary scaffold. Each one states the desired behaviour of a defect that the
current code violates and is marked `xfail(strict=True)`. Fixing the defect turns the test into an unexpected pass,
which fails the run until the `xfail` mark is removed. Once all marks are gone, the files move to the regular suite.
"""

import datetime
import importlib

import pytest

from phoibe.layered.application.registry import RuleRegistry
from phoibe.layered.core.entities import FileMetadata, LayerReport, RuleExecutionResult

RULE_MODULES = (
    "phoibe.layered.rules.rules_columns",
    "phoibe.layered.rules.rules_index",
    "phoibe.layered.rules.rules_power",
    "phoibe.layered.rules.rules_values",
)


@pytest.fixture
def fresh_registry():
    """Registry holding exactly the rules the package defines.

    Existing registry tests clear the registry and never restore it. This fixture clears it, re-imports the rule
    modules so that they register again, and restores the previous content afterwards. It does not hide defect 1,
    which is tested separately in a fresh interpreter.
    """
    saved = dict(RuleRegistry._rules)
    RuleRegistry.clear()
    for module_name in RULE_MODULES:
        importlib.reload(importlib.import_module(module_name))
    yield RuleRegistry
    RuleRegistry.clear()
    RuleRegistry._rules.update(saved)


@pytest.fixture
def report_factory():
    """Build a `LayerReport` with sensible defaults."""

    def create(
        *,
        layer_name: str = "bronze",
        turbine_id: str = "WEA_01",
        results: list[RuleExecutionResult] | None = None,
        detected_variables: dict[str, str | None] | None = None,
        size_bytes: int = 3 * 1024**2,
    ) -> LayerReport:
        return LayerReport(
            layer_name=layer_name,
            device_type="wtg",
            version="0.1.0",
            turbine_id=turbine_id,
            timestamp=datetime.datetime(2026, 1, 2, 3, 4, 5),
            file_metadata=FileMetadata(
                filename="scada.csv",
                size_bytes=size_bytes,
                format="csv",
                modified_at=datetime.datetime(2026, 1, 1, 12, 0, 0),
            ),
            detected_variables=detected_variables or {},
            rule_execution_results=results or [],
        )

    return create
