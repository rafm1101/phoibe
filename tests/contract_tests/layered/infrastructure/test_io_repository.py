import pytest
import yaml

from phoibe.layered.core.entities import Status
from phoibe.layered.infrastructure.io import YAMLReportRepository


class ReportRepositoryContract:

    def test_save_creates_nonempty_file(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "report.yaml"
        repository.save(sample_reports, str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_save_creates_parent_directories(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "subdir1" / "subdir2" / "report.yaml"
        assert not output_path.parent.exists()
        repository.save(sample_reports, str(output_path))

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_overwrites_existing_file(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "report.yaml"

        output_path.write_text("old content")
        old_size = output_path.stat().st_size

        repository.save(sample_reports, str(output_path))
        new_size = output_path.stat().st_size

        assert new_size != old_size

    def test_save_handles_empty_report_list(self, repository, tmp_path):
        output_path = tmp_path / "empty_report.yaml"

        repository.save([], str(output_path))

        assert output_path.exists()


class TestYAMLReportRepositoryContract(ReportRepositoryContract):

    @pytest.fixture
    def repository(self):
        return YAMLReportRepository()

    def test_saves_valid_yaml(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "report.yaml"
        repository.save(sample_reports, str(output_path))
        with open(output_path) as filestream:
            parsed = yaml.safe_load(filestream)

        assert isinstance(parsed, dict)

    def test_yaml_has_required_structure(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "report.yaml"
        repository.save(sample_reports, str(output_path))
        with open(output_path) as filestream:
            parsed = yaml.safe_load(filestream)

        assert "run_metadata" in parsed
        assert isinstance(parsed["run_metadata"], dict)
        assert "timestamp" in parsed["run_metadata"]
        assert "total_turbines" in parsed["run_metadata"]

        assert "turbines" in parsed
        assert isinstance(parsed["turbines"], dict)

    def test_yaml_exposes_all_turbines(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "report.yaml"
        repository.save(sample_reports, str(output_path))
        with open(output_path) as filestream:
            parsed = yaml.safe_load(filestream)

        for report in sample_reports:
            assert report.turbine_id in parsed["turbines"]

    def test_yaml_metadata_counts_are_correct(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "report.yaml"
        repository.save(sample_reports, str(output_path))
        with open(output_path) as filestream:
            parsed = yaml.safe_load(filestream)

        metadata = parsed["run_metadata"]
        passed = sum(1 for record in sample_reports if record.overall_status == Status.PASSED)
        failed = sum(1 for record in sample_reports if record.overall_status == Status.FAILED)

        assert metadata["total_turbines"] == len(sample_reports)
        assert metadata["passed"] == passed
        assert metadata["failed"] == failed

    def test_turbine_data_includes_rules(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "report.yaml"
        repository.save(sample_reports, str(output_path))
        with open(output_path) as filestream:
            parsed = yaml.safe_load(filestream)

        first_turbine = parsed["turbines"]["WEA_01"]

        assert "rules" in first_turbine
        assert isinstance(first_turbine["rules"], list)
        assert len(first_turbine["rules"]) > 0

    def test_turbine_data_includes_scoring(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "report.yaml"
        repository.save(sample_reports, str(output_path))
        with open(output_path) as filestream:
            parsed = yaml.safe_load(filestream)

        first_turbine = parsed["turbines"]["WEA_01"]
        assert "scoring" in first_turbine
        assert "achieved" in first_turbine["scoring"]
        assert "max" in first_turbine["scoring"]
        assert "percentage" in first_turbine["scoring"]

    def test_timestamps_are_iso_format(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "report.yaml"
        repository.save(sample_reports, str(output_path))
        with open(output_path) as f:
            parsed = yaml.safe_load(f)

        metadata_timestamp = parsed["run_metadata"]["timestamp"]
        assert isinstance(metadata_timestamp, str)
        assert "T" in metadata_timestamp

        turbine_timestamp = parsed["turbines"]["WEA_01"]["timestamp"]
        assert isinstance(turbine_timestamp, str)
        assert "T" in turbine_timestamp

    def test_detected_variables_are_dict(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "report.yaml"
        repository.save(sample_reports, str(output_path))
        with open(output_path) as f:
            parsed = yaml.safe_load(f)

        signals = parsed["turbines"]["WEA_01"]["detected_variables"]
        assert isinstance(signals, dict)
        assert "timestamp" in signals


class TestReportRepositoryRoundTrip:

    @pytest.fixture
    def repository(self):
        return YAMLReportRepository()

    def test_save_load_round_trip(self, repository, sample_reports, tmp_path):
        output_path = tmp_path / "report.yaml"
        repository.save(sample_reports, str(output_path))
        loaded_reports = repository.load(str(output_path))
        assert len(loaded_reports) == len(sample_reports)

        original_ids = {record.turbine_id for record in sample_reports}
        loaded_ids = {record.turbine_id for record in loaded_reports}
        assert original_ids == loaded_ids
