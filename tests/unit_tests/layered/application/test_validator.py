import datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from phoibe.layered.application.validator import LayerValidator
from phoibe.layered.core.entities import FileMetadata
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.rules.rule import ValidationRule


class MockRule(ValidationRule):
    def __init__(self, name: str, behavior: str = "pass", points: int = 10):
        self._name = name
        self.behavior = behavior
        self.points = points

    @property
    def name(self):
        return self._name

    def execute(self, df, context):
        if self.behavior == "crash":
            raise RuntimeError(f"{self.name} intentionally crashed")
        elif self.behavior == "fail":
            return RuleExecutionResult(self.name, Status.FAILED, Severity.CRITICAL, True, False, self.points, 0)
        else:  # pass
            return RuleExecutionResult(
                self.name, Status.PASSED, Severity.CRITICAL, True, True, self.points, self.points
            )


class TestLayerValidatorErrorRecovery:

    @pytest.fixture
    def mock_loader(self):
        loader = Mock()
        loader.load.return_value = pd.DataFrame({"col": [1, 2, 3]})
        loader.get_metadata.return_value = FileMetadata(
            filename="test.csv", size_bytes=1024, format="csv", modified_at=datetime.datetime.now()
        )
        return loader

    @pytest.fixture
    def mock_detector(self):
        detector = Mock()
        detector.detect.return_value = {"variable": "col"}
        return detector

    def test_single_crashing_rule_creates_error_result(self, mock_loader, mock_detector):
        rules = [MockRule("crasher", behavior="crash")]
        validator = LayerValidator("raw", mock_loader, mock_detector, rules)
        report = validator.validate("test.csv", "WEA 01")

        assert len(report.rule_execution_results) == 1
        assert report.rule_execution_results[0].status == Status.ERROR
        assert report.rule_execution_results[0].rule_name == "crasher"

    def test_crashed_rule_has_zero_points(self, mock_loader, mock_detector):
        rules = [MockRule("crasher", behavior="crash", points=50)]
        validator = LayerValidator("raw", mock_loader, mock_detector, rules)
        report = validator.validate("test.csv", "WEA 01")

        assert report.rule_execution_results[0].points_achieved == 0
        assert report.rule_execution_results[0].points_max == 50

    def test_crashed_rule_error_message_contains_exception(self, mock_loader, mock_detector):
        rules = [MockRule("crasher", behavior="crash")]
        validator = LayerValidator("raw", mock_loader, mock_detector, rules)
        report = validator.validate("test.csv", "WEA 01")

        message = report.rule_execution_results[0].message
        assert "crashed" in message.lower()
        assert "crasher" in message

    def test_crash_does_not_stop_other_rules(self, mock_loader, mock_detector):
        rules = [
            MockRule("rule1", behavior="pass"),
            MockRule("crasher", behavior="crash"),
            MockRule("rule3", behavior="pass"),
        ]
        validator = LayerValidator("raw", mock_loader, mock_detector, rules)
        report = validator.validate("test.csv", "WEA 01")
        assert len(report.rule_execution_results) == 3

        rule1 = next(record for record in report.rule_execution_results if record.rule_name == "rule1")
        rule3 = next(record for record in report.rule_execution_results if record.rule_name == "rule3")
        assert rule1.status == Status.PASSED
        assert rule3.status == Status.PASSED

    def test_multiple_crashes_each_get_error_results(self, mock_loader, mock_detector):
        rules = [
            MockRule("crasher1", behavior="crash"),
            MockRule("crasher2", behavior="crash"),
            MockRule("survivor", behavior="pass"),
        ]
        validator = LayerValidator("raw", mock_loader, mock_detector, rules)
        report = validator.validate("test.csv", "WEA 01")
        errors = [record for record in report.rule_execution_results if record.status == Status.ERROR]
        assert len(errors) == 2
        assert {e.rule_name for e in errors} == {"crasher1", "crasher2"}

    def test_overall_status_error_with_any_crash(self, mock_loader, mock_detector):
        rules = [MockRule("passer", behavior="pass"), MockRule("crasher", behavior="crash")]
        validator = LayerValidator("raw", mock_loader, mock_detector, rules)
        report = validator.validate("test.csv", "WEA 01")
        assert report.overall_status == Status.ERROR

    def test_overall_status_failed_without_crashes(self, mock_loader, mock_detector):
        rules = [MockRule("passer", behavior="pass"), MockRule("failer", behavior="fail")]
        validator = LayerValidator("raw", mock_loader, mock_detector, rules)
        report = validator.validate("test.csv", "WEA 01")
        assert report.overall_status == Status.FAILED

    def test_score_max_includes_crashed_rule_points(self, mock_loader, mock_detector):
        rules = [
            MockRule("rule1", behavior="pass", points=10),
            MockRule("crasher", behavior="crash", points=20),
            MockRule("rule3", behavior="pass", points=30),
        ]
        validator = LayerValidator("raw", mock_loader, mock_detector, rules)
        report = validator.validate("test.csv", "WEA 01")
        assert report.score_max == 60

    def test_score_achieved_excludes_crashed_rule_points(self, mock_loader, mock_detector):
        rules = [
            MockRule("rule1", behavior="pass", points=10),
            MockRule("crasher", behavior="crash", points=20),
            MockRule("rule3", behavior="pass", points=30),
        ]
        validator = LayerValidator("raw", mock_loader, mock_detector, rules)
        report = validator.validate("test.csv", "WEA 01")
        assert report.score_achieved == 40

    def test_handles_value_error(self, mock_loader, mock_detector):
        class ValueErrorRule(ValidationRule):
            @property
            def name(self):
                return "value_error"

            def execute(self, df, context):
                raise ValueError("Invalid value")

        validator = LayerValidator("raw", mock_loader, mock_detector, [ValueErrorRule()])
        report = validator.validate("test.csv", "WEA 01")
        assert report.rule_execution_results[0].status == Status.ERROR
        assert "Invalid value" in report.rule_execution_results[0].message

    def test_handles_key_error(self, mock_loader, mock_detector):
        class KeyErrorRule(ValidationRule):
            @property
            def name(self):
                return "key_error"

            def execute(self, df, context):
                raise KeyError("missing_column")

        validator = LayerValidator("raw", mock_loader, mock_detector, [KeyErrorRule()])
        report = validator.validate("test.csv", "WEA 01")
        assert report.rule_execution_results[0].status == Status.ERROR
        assert "missing_column" in report.rule_execution_results[0].message

    def test_handles_attribute_error(self, mock_loader, mock_detector):
        class AttrErrorRule(ValidationRule):
            @property
            def name(self):
                return "attr_error"

            def execute(self, df, context):
                raise AttributeError("missing attribute")

        validator = LayerValidator("raw", mock_loader, mock_detector, [AttrErrorRule()])
        report = validator.validate("test.csv", "WEA 01")
        assert report.rule_execution_results[0].status == Status.ERROR


class TestLayerValidatorEdgeCases:

    @pytest.fixture
    def mock_loader(self):
        loader = Mock()
        loader.load.return_value = pd.DataFrame({"col": [1]})
        loader.get_metadata.return_value = FileMetadata("test.csv", 1024, "csv", datetime.datetime.now())
        return loader

    @pytest.fixture
    def mock_detector(self):
        detector = Mock()
        detector.detect.return_value = {}
        return detector

    def test_validates_empty_dataframe(self, mock_detector):
        loader = Mock()
        loader.load.return_value = pd.DataFrame()
        loader.get_metadata.return_value = FileMetadata("empty.csv", 0, "csv", datetime.datetime.now())

        rules = [MockRule("rule1")]
        validator = LayerValidator("raw", loader, mock_detector, rules)
        report = validator.validate("empty.csv", "WEA 01")

        assert isinstance(report.rule_execution_results, list)

    def test_validates_single_row_dataframe(self, mock_detector):
        loader = Mock()
        loader.load.return_value = pd.DataFrame({"col": [42]})
        loader.get_metadata.return_value = FileMetadata("single.csv", 10, "csv", datetime.datetime.now())

        rules = [MockRule("rule1")]
        validator = LayerValidator("raw", loader, mock_detector, rules)
        report = validator.validate("single.csv", "WEA 01")

        assert len(report.rule_execution_results) == 1

    def test_works_when_no_variables_detected(self, mock_loader):
        detector = Mock()
        detector.detect.return_value = {"timestamp": None, "power": None, "wind_speed": None}

        rules = [MockRule("rule1")]
        validator = LayerValidator("raw", mock_loader, detector, rules)
        report = validator.validate("test.csv", "WEA 01")

        assert len(report.rule_execution_results) == 1
        assert all(v is None for v in report.detected_variables.values())

    def test_rule_execution_order_matches_list_order(self, mock_loader, mock_detector):
        execution_order = []

        class OrderTrackingRule(ValidationRule):
            def __init__(self, name):
                self._name = name

            @property
            def name(self):
                return self._name

            def execute(self, df, context):
                execution_order.append(self._name)
                return RuleExecutionResult(self._name, Status.PASSED, Severity.INFO, True, True, 0, 0)

        rules = [OrderTrackingRule("first"), OrderTrackingRule("second"), OrderTrackingRule("third")]
        validator = LayerValidator("raw", mock_loader, mock_detector, rules)
        validator.validate("test.csv", "WEA 01")

        assert execution_order == ["first", "second", "third"]
