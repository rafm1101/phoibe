import datetime

import pytest

from phoibe.layered.core.entities import FileMetadata
from phoibe.layered.core.entities import LayerReport
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status


@pytest.fixture
def sample_reports():
    reports = []

    for i in range(3):
        turbine_id = f"WEA_{i+1:02d}"

        rule_execution_results = [
            RuleExecutionResult(
                rule_name="signal_timestamp",
                status=Status.PASSED,
                severity=Severity.CRITICAL,
                required=True,
                actual="Zeitstempel",
                points_max=10,
                points_achieved=10,
                message="Signal detected",
                details={},
            ),
            RuleExecutionResult(
                rule_name="temporal_resolution",
                status=Status.PASSED if i < 2 else Status.FAILED,
                severity=Severity.CRITICAL,
                required="10min",
                actual="10.0min" if i < 2 else "15.0min",
                points_max=10,
                points_achieved=10 if i < 2 else 0,
                message="Resolution check",
                details={"median_seconds": 600 if i < 2 else 900},
            ),
        ]

        report = LayerReport(
            layer_name="raw",
            turbine_id=turbine_id,
            timestamp=datetime.datetime(2024, 1, 1, 10, 0, 0),
            file_metadata=FileMetadata(
                filename=f"{turbine_id.lower()}_data.csv",
                size_bytes=1024 * 100,
                format="csv",
                modified_at=datetime.datetime(2024, 1, 1, 9, 0, 0),
            ),
            detected_variables={"timestamp": "Zeitstempel", "wind_speed": "ws_gondel"},
            rule_execution_results=rule_execution_results,
        )

        reports.append(report)

    return reports
