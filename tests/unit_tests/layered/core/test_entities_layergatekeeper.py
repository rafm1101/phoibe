import datetime

from phoibe.layered.core.entities import FileMetadata
from phoibe.layered.core.entities import LayerGateFailureError
from phoibe.layered.core.entities import LayerGateKeeper
from phoibe.layered.core.entities import LayerReport
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status


class TestLayerGateKeeper:

    def test_empty_report_passes_gate(self):
        report = self._create_report([])
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True
        assert decision.failed_critical_rules == []
        assert decision.statistics["total_checks"] == 0

    def test_all_passed_passes_gate(self):
        results = [
            self._result("rule1", Status.PASSED, Severity.CRITICAL),
            self._result("rule2", Status.PASSED, Severity.WARNING),
            self._result("rule3", Status.PASSED, Severity.INFO),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True
        assert decision.statistics["passed"] == 3

    def test_critical_failed_blocks_gate(self):
        results = [
            self._result("good_rule", Status.PASSED, Severity.CRITICAL),
            self._result("bad_rule", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is False
        assert decision.failed_critical_rules == ["bad_rule"]
        assert decision.statistics["critical_failed"] == 1

    def test_critical_error_blocks_gate(self):
        results = [
            self._result("good_rule", Status.PASSED, Severity.CRITICAL),
            self._result("crashed_rule", Status.ERROR, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is False
        assert decision.failed_critical_rules == ["crashed_rule"]
        assert decision.statistics["critical_error"] == 1

    def test_warning_failed_does_not_block(self):
        results = [
            self._result("warn_rule", Status.FAILED, Severity.WARNING),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True
        assert decision.failed_critical_rules == []
        assert decision.statistics["warning_failed"] == 1

    def test_info_failed_does_not_block(self):
        results = [
            self._result("info_rule", Status.FAILED, Severity.INFO),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True
        assert decision.statistics["info_failed"] == 1

    def test_warning_error_does_not_block(self):
        results = [
            self._result("warn_error", Status.ERROR, Severity.WARNING),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True

    def test_critical_passed_does_not_block(self):
        """Unit: CRITICAL + PASSED does NOT block (rule succeeded)"""
        results = [
            self._result("crit_pass", Status.PASSED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True

    def test_critical_warning_does_not_block(self):
        results = [
            self._result("crit_warn", Status.WARNING, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True

    def test_critical_not_checked_does_not_block(self):
        results = [
            self._result("crit_skip", Status.NOT_CHECKED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True

    def test_multiple_critical_failed_all_block(self):
        results = [
            self._result("fail1", Status.FAILED, Severity.CRITICAL),
            self._result("fail2", Status.FAILED, Severity.CRITICAL),
            self._result("fail3", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is False
        assert len(decision.failed_critical_rules) == 3
        assert "fail1" in decision.failed_critical_rules
        assert "fail2" in decision.failed_critical_rules
        assert "fail3" in decision.failed_critical_rules

    def test_mixed_critical_failed_and_error(self):
        results = [
            self._result("failed", Status.FAILED, Severity.CRITICAL),
            self._result("errored", Status.ERROR, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is False
        assert len(decision.failed_critical_rules) == 2
        assert "failed" in decision.failed_critical_rules
        assert "errored" in decision.failed_critical_rules

    def test_mixed_passed_and_critical_failed(self):
        results = [
            self._result("good1", Status.PASSED, Severity.CRITICAL),
            self._result("good2", Status.PASSED, Severity.WARNING),
            self._result("bad", Status.FAILED, Severity.CRITICAL),
            self._result("good3", Status.PASSED, Severity.INFO),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is False
        assert decision.failed_critical_rules == ["bad"]
        assert decision.statistics["passed"] == 3
        assert decision.statistics["critical_failed"] == 1

    def test_mixed_warnings_and_critical_failed(self):
        results = [
            self._result("warn1", Status.FAILED, Severity.WARNING),
            self._result("crit", Status.FAILED, Severity.CRITICAL),
            self._result("warn2", Status.FAILED, Severity.WARNING),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is False
        assert decision.failed_critical_rules == ["crit"]
        assert decision.statistics["warning_failed"] == 2
        assert decision.statistics["critical_failed"] == 1

    def test_all_failed_but_no_critical(self):
        results = [
            self._result("warn", Status.FAILED, Severity.WARNING),
            self._result("info", Status.FAILED, Severity.INFO),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True
        assert decision.failed_critical_rules == []

    def test_all_error_but_no_critical(self):
        results = [
            self._result("err1", Status.ERROR, Severity.WARNING),
            self._result("err2", Status.ERROR, Severity.INFO),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True

    def test_all_not_checked(self):
        results = [
            self._result("skip1", Status.NOT_CHECKED, Severity.CRITICAL),
            self._result("skip2", Status.NOT_CHECKED, Severity.WARNING),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True
        assert decision.statistics["not_checked"] == 2

    def test_statistics_total_checks_correct(self):
        results = [
            self._result("r1", Status.PASSED, Severity.CRITICAL),
            self._result("r2", Status.FAILED, Severity.WARNING),
            self._result("r3", Status.ERROR, Severity.INFO),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.statistics["total_checks"] == 3

    def test_statistics_counts_by_status(self):
        results = [
            self._result("p1", Status.PASSED, Severity.CRITICAL),
            self._result("p2", Status.PASSED, Severity.WARNING),
            self._result("f1", Status.FAILED, Severity.CRITICAL),
            self._result("w1", Status.WARNING, Severity.WARNING),
            self._result("e1", Status.ERROR, Severity.INFO),
            self._result("n1", Status.NOT_CHECKED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.statistics["passed"] == 2
        assert decision.statistics["failed"] == 1
        assert decision.statistics["warning"] == 1
        assert decision.statistics["error"] == 1
        assert decision.statistics["not_checked"] == 1

    def test_statistics_counts_critical_combinations(self):
        results = [
            self._result("cf1", Status.FAILED, Severity.CRITICAL),
            self._result("cf2", Status.FAILED, Severity.CRITICAL),
            self._result("ce1", Status.ERROR, Severity.CRITICAL),
            self._result("wf1", Status.FAILED, Severity.WARNING),
            self._result("if1", Status.FAILED, Severity.INFO),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.statistics["critical_failed"] == 2
        assert decision.statistics["critical_error"] == 1
        assert decision.statistics["warning_failed"] == 1
        assert decision.statistics["info_failed"] == 1

    def test_statistics_empty_report(self):
        report = self._create_report([])
        decision = LayerGateKeeper.from_report(report)
        for key, value in decision.statistics.items():
            assert value == 0

    def test_single_critical_failed(self):
        results = [
            self._result("only_fail", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is False
        assert decision.failed_critical_rules == ["only_fail"]

    def test_ten_critical_all_passed(self):
        results = [self._result(f"crit{i}", Status.PASSED, Severity.CRITICAL) for i in range(10)]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is True
        assert decision.statistics["passed"] == 10

    def test_ten_critical_one_failed(self):
        results = [self._result(f"pass{i}", Status.PASSED, Severity.CRITICAL) for i in range(9)]
        results.append(self._result("fail", Status.FAILED, Severity.CRITICAL))
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.passed is False
        assert decision.failed_critical_rules == ["fail"]

    def test_gate_failure_error_contains_decision(self):
        results = [
            self._result("fail", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)
        error = LayerGateFailureError(decision)

        assert error.decision is decision

    def test_gate_failure_error_message_format(self):
        results = [
            self._result("temporal_check", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)
        error = LayerGateFailureError(decision)
        message = str(error)

        assert "bronze" in message.lower()
        assert "temporal_check" in message
        assert "WEA_01" in message
        assert "/" in message

    def test_gate_failure_error_message_single_failure(self):
        results = [
            self._result("rule", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)
        error = LayerGateFailureError(decision)

        assert "1 CRITICAL failure" in str(error)

    def test_gate_failure_error_message_multiple_failures(self):
        results = [
            self._result("rule1", Status.FAILED, Severity.CRITICAL),
            self._result("rule2", Status.ERROR, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)
        error = LayerGateFailureError(decision)

        assert "2 CRITICAL failures" in str(error)

    def test_gate_failure_error_lists_all_blocking_rules(self):
        results = [
            self._result("check_a", Status.FAILED, Severity.CRITICAL),
            self._result("check_b", Status.ERROR, Severity.CRITICAL),
            self._result("check_c", Status.FAILED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)
        error = LayerGateFailureError(decision)
        message = str(error)

        assert "check_a" in message
        assert "check_b" in message
        assert "check_c" in message

    def test_decision_contains_original_report(self):
        results = [
            self._result("rule", Status.PASSED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.report is report

    def test_decision_preserves_report_metadata(self):
        results = [
            self._result("rule", Status.PASSED, Severity.CRITICAL),
        ]
        report = self._create_report(results)
        decision = LayerGateKeeper.from_report(report)

        assert decision.report.turbine_id == "WEA_01"
        assert decision.report.layer_name == "bronze"
        assert len(decision.report.rule_execution_results) == 1

    def _result(self, name: str, status: Status, severity: Severity, points_max: int = 10) -> RuleExecutionResult:
        """Create test result."""
        points = points_max if status == Status.PASSED else 0
        return RuleExecutionResult(
            rule_name=name,
            status=status,
            severity=severity,
            required="test",
            actual="test",
            points_max=points_max,
            points_achieved=points,
            message=f"{name} {status.value}",
        )

    def _create_report(self, rule_execution_results: list[RuleExecutionResult]) -> LayerReport:
        """Create test report."""
        return LayerReport(
            turbine_id="WEA_01",
            layer_name="bronze",
            timestamp=datetime.datetime.now(),
            file_metadata=FileMetadata(
                filename="test.csv", size_bytes=1024**2, format="csv", modified_at=datetime.datetime.now()
            ),
            detected_variables={},
            rule_execution_results=rule_execution_results,
        )
