import datetime

import pytest

from phoibe.layered.core.entities import FileMetadata
from phoibe.layered.core.entities import LayerReport
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status


@pytest.fixture
def file_metadata():
    return FileMetadata("test.csv", 1024, "csv", datetime.datetime.now())


class TestLayerReportScoring:

    @pytest.fixture
    def sample_checks(self):
        return [
            RuleExecutionResult(
                rule_name="check1",
                status=Status.PASSED,
                severity=Severity.CRITICAL,
                required=True,
                actual=True,
                points_max=10,
                points_achieved=10,
            ),
            RuleExecutionResult(
                rule_name="check2",
                status=Status.FAILED,
                severity=Severity.CRITICAL,
                required=True,
                actual=False,
                points_max=20,
                points_achieved=0,
            ),
            RuleExecutionResult(
                rule_name="check3",
                status=Status.WARNING,
                severity=Severity.WARNING,
                required=True,
                actual=True,
                points_max=5,
                points_achieved=3,
            ),
        ]

    def test_score_max_sums_all_points(self, sample_checks, file_metadata):
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=sample_checks,
        )

        assert report.score_max == 35

    def test_score_achieved_sums_achieved_points(self, sample_checks, file_metadata):
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=sample_checks,
        )

        assert report.score_achieved == 13  # 10 + 0 + 3

    def test_percentage_computes_correcly(self, sample_checks, file_metadata):
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=sample_checks,
        )

        expected = (13 / 35) * 100
        assert abs(report.percentage - expected) < 0.01

    def test_percentage_is_zero_when_max_is_zero(self, file_metadata):
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=[],
        )

        assert report.percentage == 0

    def test_critical_failures_counts_only_critical_failed(self, file_metadata):
        checks = [
            RuleExecutionResult("c1", Status.FAILED, Severity.CRITICAL, True, False, 10, 0),
            RuleExecutionResult("c2", Status.FAILED, Severity.WARNING, True, False, 10, 0),
            RuleExecutionResult("c3", Status.PASSED, Severity.CRITICAL, True, True, 10, 10),
            RuleExecutionResult("c4", Status.FAILED, Severity.CRITICAL, True, False, 10, 0),
        ]
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=checks,
        )

        assert report.critical_failures == 2

    def test_warnings_counts_warning_status(self, file_metadata):
        checks = [
            RuleExecutionResult("c1", Status.WARNING, Severity.WARNING, True, True, 10, 5),
            RuleExecutionResult("c2", Status.WARNING, Severity.CRITICAL, True, True, 10, 8),
            RuleExecutionResult("c3", Status.PASSED, Severity.WARNING, True, True, 10, 10),
        ]
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=checks,
        )

        assert report.warnings == 2


class TestLayerReportOverallStatus:
    """Unit tests for overall_status business logic"""

    def test_status_error_if_any_error(self, file_metadata):
        checks = [
            RuleExecutionResult("c1", Status.PASSED, Severity.CRITICAL, True, True, 10, 10),
            RuleExecutionResult("c2", Status.ERROR, Severity.CRITICAL, True, None, 10, 0),
        ]
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=checks,
        )

        assert report.overall_status == Status.ERROR

    def test_status_failed_if_critical_failures(self, file_metadata):
        checks = [
            RuleExecutionResult("c1", Status.FAILED, Severity.CRITICAL, True, False, 10, 0),
        ]
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=checks,
        )

        assert report.overall_status == Status.FAILED

    def test_status_warning_if_warnings(self, file_metadata):
        checks = [
            RuleExecutionResult("c1", Status.PASSED, Severity.CRITICAL, True, True, 10, 10),
            RuleExecutionResult("c2", Status.WARNING, Severity.WARNING, True, True, 10, 5),
        ]
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=checks,
        )

        assert report.overall_status == Status.WARNING

    def test_status_passed_if_all_passed(self, file_metadata):
        checks = [
            RuleExecutionResult("c1", Status.PASSED, Severity.CRITICAL, True, True, 10, 10),
            RuleExecutionResult("c2", Status.PASSED, Severity.WARNING, True, True, 10, 10),
        ]
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=checks,
        )

        assert report.overall_status == Status.PASSED

    def test_status_priority_error_over_failed(self, file_metadata):
        checks = [
            RuleExecutionResult("c1", Status.FAILED, Severity.CRITICAL, True, False, 10, 0),
            RuleExecutionResult("c2", Status.ERROR, Severity.CRITICAL, True, None, 10, 0),
        ]
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=checks,
        )

        assert report.overall_status == Status.ERROR

    def test_status_priority_failed_over_warning(self, file_metadata):
        checks = [
            RuleExecutionResult("c1", Status.WARNING, Severity.WARNING, True, True, 10, 5),
            RuleExecutionResult("c2", Status.FAILED, Severity.CRITICAL, True, False, 10, 0),
        ]
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA 01",
            timestamp=datetime.datetime.now(),
            file_metadata=file_metadata,
            detected_variables={},
            rule_execution_results=checks,
        )

        assert report.overall_status == Status.FAILED
