import pandas as pd
import pytest

from phoibe.layered.application.validator import LayerValidator
from phoibe.layered.core.entities import RuleExecutionResult, Severity, Status
from phoibe.layered.infrastructure.detector import RegexVariableDetector
from phoibe.layered.infrastructure.io import InMemoryDataLoader
from phoibe.layered.rules.rule import ValidationRule


class _CrashingRule(ValidationRule):
    @property
    def name(self) -> str:
        return "crashing"

    def execute(self, df, context) -> RuleExecutionResult:
        raise RuntimeError("boom")


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 10: a crashed rule is always recorded as CRITICAL")
@pytest.mark.parametrize("severity", [Severity.WARNING, Severity.INFO])
def test_crashed_rule_keeps_its_configured_severity(severity):
    validator = LayerValidator(
        layer_name="raw",
        device_type="wtg",
        version="0.1.0",
        data_loader=InMemoryDataLoader(pd.DataFrame({"x": [1, 2]})),
        variable_detector=RegexVariableDetector({}),
        rules=[_CrashingRule(points=10, severity=severity)],
    )

    report = validator.validate("", "WEA_01")

    (result,) = report.rule_execution_results
    assert result.status == Status.ERROR
    assert result.severity == severity
