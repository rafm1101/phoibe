import datetime

import pytest
import yaml

from phoibe.layered.core.entities import FileMetadata
from phoibe.layered.core.entities import LayerReport
from phoibe.layered.core.entities import RuleExecutionResult
from phoibe.layered.core.entities import Severity
from phoibe.layered.core.entities import Status
from phoibe.layered.infrastructure.io import YAMLReportRepository


class TestYAMLReportRepositoryEdgeCases:

    @pytest.fixture
    def repository(self):
        return YAMLReportRepository()

    @pytest.fixture
    def minimal_report(self):
        """Create minimal valid LayerReport"""
        return LayerReport(
            layer_name="raw",
            turbine_id="WEA_01",
            timestamp=datetime.datetime(2024, 1, 1, 10, 0, 0),
            file_metadata=FileMetadata(
                filename="test.csv",
                size_bytes=1024,
                format="csv",
                modified_at=datetime.datetime(2024, 1, 1, 9, 0, 0),
            ),
            detected_variables={"timestamp": "Zeitstempel"},
            rule_execution_results=[],
        )

    def test_save_stores_empty_report_list(self, repository, tmp_path):
        output = tmp_path / "empty.yaml"
        repository.save([], str(output))
        assert output.exists()
        with open(output) as f:
            data = yaml.safe_load(f)
        assert data["run_metadata"]["total_turbines"] == 0
        assert data["turbines"] == {}

    def test_save_stores_single_report(self, repository, minimal_report, tmp_path):
        output = tmp_path / "single.yaml"
        repository.save([minimal_report], str(output))
        with open(output) as f:
            data = yaml.safe_load(f)
        assert data["run_metadata"]["total_turbines"] == 1
        assert "WEA_01" in data["turbines"]

    def test_save_stores_report_with_no_rules(self, repository, minimal_report, tmp_path):
        output = tmp_path / "no_rules.yaml"
        repository.save([minimal_report], str(output))
        with open(output) as f:
            data = yaml.safe_load(f)
        turbine = data["turbines"]["WEA_01"]
        assert turbine["rules"] == []
        assert turbine["scoring"]["achieved"] == 0
        assert turbine["scoring"]["max"] == 0

    def test_save_stores_many_reports(self, repository, tmp_path):
        reports = []
        for i in range(150):
            report = LayerReport(
                layer_name="raw",
                turbine_id=f"WEA_{i:03d}",
                timestamp=datetime.datetime(2024, 1, 1, 10, 0, 0),
                file_metadata=FileMetadata(
                    filename=f"wea_{i:03d}.csv",
                    size_bytes=1024,
                    format="csv",
                    modified_at=datetime.datetime(2024, 1, 1, 9, 0, 0),
                ),
                detected_variables={},
                rule_execution_results=[],
            )
            reports.append(report)
        output = tmp_path / "many.yaml"
        repository.save(reports, str(output))
        with open(output) as f:
            data = yaml.safe_load(f)
        assert data["run_metadata"]["total_turbines"] == 150
        assert len(data["turbines"]) == 150

    def test_save_creates_parent_directories(self, repository, minimal_report, tmp_path):
        output = tmp_path / "level1" / "level2" / "level3" / "report.yaml"
        assert not output.parent.exists()
        repository.save([minimal_report], str(output))
        assert output.exists()
        assert output.parent.exists()

    def test_save_accepts_pathlib_path(self, repository, minimal_report, tmp_path):
        output = tmp_path / "report.yaml"
        repository.save([minimal_report], output)
        assert output.exists()

    def test_save_overwrites_existing_file(self, repository, minimal_report, tmp_path):
        output = tmp_path / "report.yaml"
        output.write_text("old content")
        old_size = output.stat().st_size
        repository.save([minimal_report], str(output))
        new_size = output.stat().st_size
        assert new_size != old_size
        with open(output) as f:
            data = yaml.safe_load(f)
        assert "turbines" in data

    def test_save_handles_special_characters_in_path(self, repository, minimal_report, tmp_path):
        subdir = tmp_path / "my reports" / "data (2024)"
        subdir.mkdir(parents=True)
        output = subdir / "validation report.yaml"
        repository.save([minimal_report], str(output))
        assert output.exists()

    def test_yaml_is_valid_parseable(self, repository, minimal_report, tmp_path):
        output = tmp_path / "report.yaml"
        repository.save([minimal_report], str(output))
        with open(output) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)

    def test_yaml_preserves_unicode(self, repository, tmp_path):
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA_Müller_Äpfel",
            timestamp=datetime.datetime(2024, 1, 1, 10, 0, 0),
            file_metadata=FileMetadata(
                filename="Außentemperatur.csv",
                size_bytes=1024,
                format="csv",
                modified_at=datetime.datetime(2024, 1, 1, 9, 0, 0),
            ),
            detected_variables={"temp": "Außentemperatur_Gondel"},
            rule_execution_results=[],
        )
        output = tmp_path / "unicode.yaml"
        repository.save([report], str(output))
        with open(output, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert "WEA_Müller_Äpfel" in data["turbines"]
        assert data["turbines"]["WEA_Müller_Äpfel"]["file_info"]["filename"] == "Außentemperatur.csv"

    def test_yaml_handles_special_yaml_characters(self, repository, tmp_path):
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA:01",  # Colon is special in YAML
            timestamp=datetime.datetime(2024, 1, 1, 10, 0, 0),
            file_metadata=FileMetadata(
                filename="data-2024.csv",
                size_bytes=1024,
                format="csv",
                modified_at=datetime.datetime(2024, 1, 1, 9, 0, 0),
            ),
            detected_variables={"variable": "column-name"},
            rule_execution_results=[],
        )
        output = tmp_path / "special.yaml"
        repository.save([report], str(output))
        with open(output) as f:
            data = yaml.safe_load(f)
        assert "WEA:01" in data["turbines"]

    def test_timestamps_are_iso_format_strings(self, repository, minimal_report, tmp_path):
        output = tmp_path / "report.yaml"
        repository.save([minimal_report], str(output))
        with open(output) as f:
            data = yaml.safe_load(f)

        run_ts = data["run_metadata"]["timestamp"]
        assert isinstance(run_ts, str)
        assert "T" in run_ts
        turbine_ts = data["turbines"]["WEA_01"]["timestamp"]
        assert isinstance(turbine_ts, str)
        assert "T" in turbine_ts
        file_ts = data["turbines"]["WEA_01"]["file_info"]["modified"]
        assert isinstance(file_ts, str)
        assert "T" in file_ts

    def test_serializes_rule_with_none_values(self, repository, tmp_path):
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA_01",
            timestamp=datetime.datetime(2024, 1, 1, 10, 0, 0),
            file_metadata=FileMetadata(
                filename="test.csv", size_bytes=1024, format="csv", modified_at=datetime.datetime(2024, 1, 1, 9, 0, 0)
            ),
            detected_variables={},
            rule_execution_results=[
                RuleExecutionResult(
                    rule_name="test",
                    status=Status.NOT_CHECKED,
                    severity=Severity.CRITICAL,
                    required=True,
                    actual=None,
                    points_max=10,
                    points_achieved=0,
                )
            ],
        )
        output = tmp_path / "none_values.yaml"
        repository.save([report], str(output))
        with open(output) as f:
            data = yaml.safe_load(f)
        rule = data["turbines"]["WEA_01"]["rules"][0]
        assert rule["actual"] == "None"

    def test_serializes_rule_with_dict_details(self, repository, tmp_path):
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA_01",
            timestamp=datetime.datetime(2024, 1, 1, 10, 0, 0),
            file_metadata=FileMetadata(
                filename="test.csv", size_bytes=1024, format="csv", modified_at=datetime.datetime(2024, 1, 1, 9, 0, 0)
            ),
            detected_variables={},
            rule_execution_results=[
                RuleExecutionResult(
                    rule_name="test",
                    status=Status.PASSED,
                    severity=Severity.CRITICAL,
                    required=True,
                    actual=True,
                    points_max=10,
                    points_achieved=10,
                    details={"nested": {"key": "value"}, "list": [1, 2, 3], "mixed": {"a": [1, 2], "b": "text"}},
                )
            ],
        )
        output = tmp_path / "complex_details.yaml"
        repository.save([report], str(output))
        with open(output) as f:
            data = yaml.safe_load(f)
        details = data["turbines"]["WEA_01"]["rules"][0]["details"]
        assert details["nested"]["key"] == "value"
        assert details["list"] == [1, 2, 3]

    def test_serializes_rule_with_empty_message(self, repository, tmp_path):
        report = LayerReport(
            layer_name="raw",
            turbine_id="WEA_01",
            timestamp=datetime.datetime(2024, 1, 1, 10, 0, 0),
            file_metadata=FileMetadata(
                filename="test.csv", size_bytes=1024, format="csv", modified_at=datetime.datetime(2024, 1, 1, 9, 0, 0)
            ),
            detected_variables={},
            rule_execution_results=[
                RuleExecutionResult(
                    rule_name="test",
                    status=Status.PASSED,
                    severity=Severity.CRITICAL,
                    required=True,
                    actual=True,
                    points_max=10,
                    points_achieved=10,
                    message="",
                )
            ],
        )
        output = tmp_path / "empty_message.yaml"
        repository.save([report], str(output))
        with open(output) as f:
            data = yaml.safe_load(f)
        rule = data["turbines"]["WEA_01"]["rules"][0]
        assert rule["message"] == ""

    def test_serializes_empty_detected_variables(self, repository, minimal_report, tmp_path):
        minimal_report.detected_variables = {}
        output = tmp_path / "no_variables.yaml"
        repository.save([minimal_report], str(output))
        with open(output) as f:
            data = yaml.safe_load(f)
        variables = data["turbines"]["WEA_01"]["detected_variables"]
        assert variables == {}

    def test_serializes_variables_with_none_values(self, repository, minimal_report, tmp_path):
        minimal_report.detected_variables = {
            "timestamp": "Zeitstempel",
            "power": None,
            "wind_speed": "ws_gondel",
            "missing": None,
        }
        output = tmp_path / "variables_with_none.yaml"
        repository.save([minimal_report], str(output))
        with open(output) as filestream:
            data = yaml.safe_load(filestream)
        variables = data["turbines"]["WEA_01"]["detected_variables"]
        assert variables["timestamp"] == "Zeitstempel"
        assert variables["power"] is None
        assert variables["wind_speed"] == "ws_gondel"
        assert variables["missing"] is None

    def test_counts_all_status_types(self, repository, tmp_path):
        reports = [
            self._create_report("WEA_01", Status.PASSED),
            self._create_report("WEA_02", Status.FAILED),
            self._create_report("WEA_03", Status.WARNING),
            self._create_report("WEA_04", Status.ERROR),
            self._create_report("WEA_05", Status.PASSED),
        ]
        output = tmp_path / "all_statuses.yaml"
        repository.save(reports, str(output))
        with open(output) as filestream:
            data = yaml.safe_load(filestream)
        metadata = data["run_metadata"]
        assert metadata["total_turbines"] == 5
        assert metadata["passed"] == 2
        assert metadata["failed"] == 1
        assert metadata["warnings"] == 1
        assert metadata["errors"] == 1

    def test_counts_only_passed(self, repository, tmp_path):
        reports = [self._create_report(f"WEA_{i:02d}", Status.PASSED) for i in range(1, 6)]
        output = tmp_path / "all_passed.yaml"
        repository.save(reports, str(output))
        with open(output) as filestream:
            data = yaml.safe_load(filestream)

        metadata = data["run_metadata"]
        assert metadata["passed"] == 5
        assert metadata["failed"] == 0
        assert metadata["warnings"] == 0
        assert metadata["errors"] == 0

    def test_save_load_roundtrip(self, repository, minimal_report, tmp_path):
        output = tmp_path / "report.yaml"
        repository.save([minimal_report], str(output))
        loaded = repository.load(str(output))
        assert len(loaded) == 1
        assert loaded[0].turbine_id == "WEA_01"

    def test_load_raises_file_not_found_error_given_nonexistent_file_raises(self, repository, tmp_path):
        nonexistent = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError):
            repository.load(str(nonexistent))

    def test_load_raises_parsererror_given_invalid_yaml_raises(self, repository, tmp_path):
        invalid = tmp_path / "invalid.yaml"
        invalid.write_text("{ invalid yaml content:")
        with pytest.raises(yaml.parser.ParserError):
            repository.load(str(invalid))

    def _create_report(self, turbine_id: str, status: Status) -> LayerReport:
        if status == Status.PASSED:
            rule_execution_results = [
                RuleExecutionResult("check1", Status.PASSED, Severity.CRITICAL, True, True, 10, 10)
            ]
        elif status == Status.FAILED:
            rule_execution_results = [
                RuleExecutionResult("check1", Status.FAILED, Severity.CRITICAL, True, False, 10, 0)
            ]
        elif status == Status.WARNING:
            rule_execution_results = [
                RuleExecutionResult("check1", Status.WARNING, Severity.WARNING, True, True, 10, 5)
            ]
        elif status == Status.ERROR:
            rule_execution_results = [RuleExecutionResult("check1", Status.ERROR, Severity.CRITICAL, True, None, 10, 0)]
        else:
            rule_execution_results = []

        return LayerReport(
            layer_name="raw",
            turbine_id=turbine_id,
            timestamp=datetime.datetime(2024, 1, 1, 10, 0, 0),
            file_metadata=FileMetadata(
                filename=f"{turbine_id.lower()}.csv",
                size_bytes=1024,
                format="csv",
                modified_at=datetime.datetime(2024, 1, 1, 9, 0, 0),
            ),
            detected_variables={},
            rule_execution_results=rule_execution_results,
        )
