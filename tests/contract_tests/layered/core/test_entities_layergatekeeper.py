import datetime

from phoibe.layered.core.entities import FileMetadata
from phoibe.layered.core.entities import LayerGateFailureError
from phoibe.layered.core.entities import LayerGateKeeper
from phoibe.layered.core.entities import LayerReport
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status


class TestLayerGateKeeper:

    def test_has_from_report_classmethod(self):
        assert hasattr(LayerGateKeeper, "from_report")
        assert callable(LayerGateKeeper.from_report)

    def test_from_report_returns_gatekeeper_instance(self):
        report = self._create_report([])
        gate_decision = LayerGateKeeper.from_report(report)

        assert isinstance(gate_decision, LayerGateKeeper)

    def test_has_attributes(self):
        report = self._create_report([])
        gate_decision = LayerGateKeeper.from_report(report)

        assert hasattr(gate_decision, "passed")
        assert isinstance(gate_decision.passed, bool)
        assert hasattr(gate_decision, "failed_critical_rules")
        assert isinstance(gate_decision.failed_critical_rules, list)
        assert hasattr(gate_decision, "report")
        assert gate_decision.report is report
        assert hasattr(gate_decision, "statistics")
        assert isinstance(gate_decision.statistics, dict)

    def test_statistics_contains_required_keys(self):
        report = self._create_report([])
        gate_decision = LayerGateKeeper.from_report(report)
        required_keys = [
            "total_checks",
            "passed",
            "failed",
            "warning",
            "error",
            "not_checked",
            "critical_failed",
            "critical_error",
            "warning_failed",
            "info_failed",
        ]
        for key in required_keys:
            assert key in gate_decision.statistics, f"Missing key: {key}"

    def test_statistics_values_are_integers(self):
        report = self._create_report([])
        gate_decision = LayerGateKeeper.from_report(report)

        for key, value in gate_decision.statistics.items():
            assert isinstance(value, int), f"Non-integer value for {key}: {value}"

    def test_statistics_values_non_negative(self):
        report = self._create_report([])
        gate_decision = LayerGateKeeper.from_report(report)

        for key, value in gate_decision.statistics.items():
            assert value >= 0, f"Negative value for {key}: {value}"

    def test_gate_passes_with_no_results(self):
        report = self._create_report([])
        gate_decision = LayerGateKeeper.from_report(report)

        assert gate_decision.passed is True
        assert gate_decision.failed_critical_rules == []

    def test_gate_passes_with_all_passed(self):
        results = [
            self._create_result("rule1", Status.PASSED, Severity.CRITICAL),
            self._create_result("rule2", Status.PASSED, Severity.WARNING),
            self._create_result("rule3", Status.PASSED, Severity.INFO),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)

        assert gate_decision.passed is True
        assert gate_decision.failed_critical_rules == []

    def test_gate_blocks_on_critical_failed(self):
        results = [
            self._create_result("rule1", Status.PASSED, Severity.CRITICAL),
            self._create_result("rule2", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)

        assert gate_decision.passed is False
        assert "rule2" in gate_decision.failed_critical_rules

    def test_gate_blocks_on_critical_error(self):
        results = [
            self._create_result("rule1", Status.PASSED, Severity.CRITICAL),
            self._create_result("rule2", Status.ERROR, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)

        assert gate_decision.passed is False
        assert "rule2" in gate_decision.failed_critical_rules

    def test_gate_passes_with_warning_failed(self):
        results = [
            self._create_result("rule1", Status.PASSED, Severity.CRITICAL),
            self._create_result("rule2", Status.FAILED, Severity.WARNING),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)

        assert gate_decision.passed is True
        assert gate_decision.failed_critical_rules == []

    def test_gate_passes_with_info_failed(self):
        results = [
            self._create_result("rule1", Status.PASSED, Severity.CRITICAL),
            self._create_result("rule2", Status.FAILED, Severity.INFO),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)

        assert gate_decision.passed is True

    def test_gate_passes_with_warning_error(self):
        results = [
            self._create_result("rule1", Status.ERROR, Severity.WARNING),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)

        assert gate_decision.passed is True

    def test_failed_critical_rules_empty_when_passed(self):
        results = [
            self._create_result("rule1", Status.PASSED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)

        assert isinstance(gate_decision.failed_critical_rules, list)
        assert gate_decision.failed_critical_rules == []

    def test_failed_critical_rules_contains_rule_names(self):
        results = [
            self._create_result("critical_rule_1", Status.FAILED, Severity.CRITICAL),
            self._create_result("critical_rule_2", Status.ERROR, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)

        assert "critical_rule_1" in gate_decision.failed_critical_rules
        assert "critical_rule_2" in gate_decision.failed_critical_rules
        assert len(gate_decision.failed_critical_rules) == 2

    def test_failed_critical_rules_only_critical_failed_or_error(self):
        results = [
            self._create_result("critical_failure", Status.FAILED, Severity.CRITICAL),
            self._create_result("critical_error", Status.ERROR, Severity.CRITICAL),
            self._create_result("warning_failure", Status.FAILED, Severity.WARNING),
            self._create_result("info_failure", Status.FAILED, Severity.INFO),
            self._create_result("critical_pass", Status.PASSED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)

        assert gate_decision.failed_critical_rules == ["critical_failure", "critical_error"]

    def test_gate_failure_error_is_exception(self):
        assert issubclass(LayerGateFailureError, Exception)

    def test_gate_failure_error_has_gate_decision_attribute(self):
        results = [
            self._create_result("rule", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)
        error = LayerGateFailureError(gate_decision)

        assert hasattr(error, "decision")
        assert error.decision is gate_decision

    def test_gate_failure_error_message_contains_layer(self):
        results = [
            self._create_result("rule", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)
        error = LayerGateFailureError(gate_decision)

        assert "bronze" in str(error).lower()

    def test_gate_failure_error_message_contains_rule_names(self):
        results = [
            self._create_result("failed_rule", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)
        error = LayerGateFailureError(gate_decision)

        assert "failed_rule" in str(error)

    def test_gate_failure_error_message_contains_score(self):
        results = [
            self._create_result("rule", Status.FAILED, Severity.CRITICAL, points_max=10),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)
        error = LayerGateFailureError(gate_decision)

        message = str(error)
        assert "score" in message.lower() or "/" in message

    def test_gate_failure_error_message_proper_pluralization_single(self):
        results = [
            self._create_result("rule", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)
        error = LayerGateFailureError(gate_decision)

        message = str(error)
        assert "1 CRITICAL failure" in message or "1 failure" in message.lower()
        assert "1 failures" not in message.lower()

    def test_gate_failure_error_message_proper_pluralization_multiple(self):
        results = [
            self._create_result("rule1", Status.FAILED, Severity.CRITICAL),
            self._create_result("rule2", Status.ERROR, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        gate_decision = LayerGateKeeper.from_report(report)
        error = LayerGateFailureError(gate_decision)

        message = str(error)
        assert "2 CRITICAL failures" in message or "failures" in message.lower()

    def _create_result(
        self,
        rule_name: str,
        status: Status,
        severity: Severity,
        points_max: int = 10,
        points_achieved: int | None = None,
    ) -> RuleExecutionResult:
        """Create a RuleExecutionResult for testing."""
        if points_achieved is None:
            points_achieved = points_max if status == Status.PASSED else 0

        return RuleExecutionResult(
            rule_name=rule_name,
            status=status,
            severity=severity,
            required="test",
            actual="test",
            points_max=points_max,
            points_achieved=points_achieved,
            message=f"{rule_name} {status.value}",
        )

    def _create_report(self, rule_execution_results: list[RuleExecutionResult]) -> LayerReport:
        """Create a LayerReport for testing."""
        return LayerReport(
            turbine_id="WEA 01",
            layer_name="bronze",
            timestamp=datetime.datetime.now(),
            file_metadata=FileMetadata(
                filename="test.csv", size_bytes=1024**2, format="csv", modified_at=datetime.datetime(1498, 3, 14, 13, 0)
            ),
            detected_variables={},
            rule_execution_results=rule_execution_results,
        )
