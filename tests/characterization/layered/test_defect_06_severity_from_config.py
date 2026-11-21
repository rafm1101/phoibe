import pandas as pd
import pytest

from phoibe.layered.application.config import ValidationConfig
from phoibe.layered.application.factory import ValidatorFactory
from phoibe.layered.core.entities import Severity
from phoibe.layered.infrastructure.io import YAMLReportRepository

RULE_CONFIG = {"name": "required_variable", "params": {"variable_name": "wind_speed", "severity": "warning"}}


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 6: a severity from config stays a plain string")
def test_severity_from_config_becomes_enum(fresh_registry):
    (rule,) = ValidatorFactory._create_rules([RULE_CONFIG])

    assert isinstance(rule.severity, Severity)
    assert rule.severity is Severity.WARNING


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 6: saving a report raises AttributeError for a config severity")
def test_report_with_severity_from_config_can_be_saved(fresh_registry, tmp_path):
    config = ValidationConfig(
        layer_name="raw",
        version="0.1.0",
        device_type="wtg",
        variable_patterns={"wind_speed": ["ws"]},
        rules=[RULE_CONFIG],
    )
    validator = ValidatorFactory.profiling(config, data=pd.DataFrame({"ws": [1.0, 2.0]}))
    report = validator.validate("", "WEA_01")

    YAMLReportRepository().save([report], tmp_path / "report.yaml")
