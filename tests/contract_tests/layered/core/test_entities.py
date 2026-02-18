import dataclasses
import datetime

import pytest

from phoibe.layered.core.entities import FileMetadata
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status


class TestFileMetadataContract:

    def test_file_metadata_is_frozen(self):
        metadata = FileMetadata(filename="test.csv", size_bytes=1024, format="csv", modified_at=datetime.datetime.now())

        with pytest.raises(dataclasses.FrozenInstanceError):
            metadata.filename = "changed.csv"

    def test_size_mb_property(self):
        metadata = FileMetadata(
            filename="test.csv", size_bytes=5 * 1024 * 1024, format="csv", modified_at=datetime.datetime.now()
        )

        assert metadata.size_mb == 5.0

    def test_size_mb_rounds_to_2_decimals(self):
        metadata = FileMetadata(
            filename="test.csv", size_bytes=1234567, format="csv", modified_at=datetime.datetime.now()
        )

        assert metadata.size_mb == 1.18


class TestRuleExecutionResultContract:

    def test_is_frozen(self):
        result = RuleExecutionResult(
            rule_name="test",
            status=Status.PASSED,
            severity=Severity.CRITICAL,
            required=True,
            actual=True,
            points_max=10,
            points_achieved=10,
        )

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.status = Status.FAILED

    def test_raises_if_points_achieved_exceeds_max(self):
        with pytest.raises(ValueError, match="Points achieved .* > max"):
            RuleExecutionResult(
                rule_name="test",
                status=Status.PASSED,
                severity=Severity.CRITICAL,
                required=True,
                actual=True,
                points_max=10,
                points_achieved=15,
            )

    def test_raises_if_points_negative(self):
        with pytest.raises(ValueError, match="Points cannot be negative"):
            RuleExecutionResult(
                rule_name="test",
                status=Status.FAILED,
                severity=Severity.CRITICAL,
                required=True,
                actual=False,
                points_max=10,
                points_achieved=-5,
            )

    def test_default_message_is_empty_string(self):
        result = RuleExecutionResult(
            rule_name="test",
            status=Status.PASSED,
            severity=Severity.CRITICAL,
            required=True,
            actual=True,
            points_max=10,
            points_achieved=10,
        )

        assert result.message == ""

    def test_default_details_is_empty_dict(self):
        result = RuleExecutionResult(
            rule_name="test",
            status=Status.PASSED,
            severity=Severity.CRITICAL,
            required=True,
            actual=True,
            points_max=10,
            points_achieved=10,
        )

        assert result.details == {}
