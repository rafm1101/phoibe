import pytest

from phoibe.layered.core.entities import RuleExecutionResult, Severity, Status
from phoibe.layered.infrastructure.io import YAMLReportRepository


def _result(name: str = "rule", points: int = 10) -> RuleExecutionResult:
    return RuleExecutionResult(
        rule_name=name,
        status=Status.PASSED,
        severity=Severity.WARNING,
        required="required",
        actual="actual",
        points_max=10,
        points_achieved=points,
        message="message",
        details={"count": 3},
    )


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 5: size_mb is reloaded as size_bytes")
def test_roundtrip_preserves_file_size(tmp_path, report_factory):
    repository = YAMLReportRepository()
    report = report_factory(results=[_result()], size_bytes=3 * 1024**2)

    repository.save([report], tmp_path / "report.yaml")
    (loaded,) = repository.load(tmp_path / "report.yaml")

    assert loaded.file_metadata.size_bytes == report.file_metadata.size_bytes


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 5: reports keyed by turbine_id overwrite each other")
def test_roundtrip_keeps_reports_of_one_turbine_in_different_layers(tmp_path, report_factory):
    repository = YAMLReportRepository()
    reports = [
        report_factory(layer_name="raw", turbine_id="WEA_01", results=[_result()]),
        report_factory(layer_name="bronze", turbine_id="WEA_01", results=[_result()]),
    ]

    repository.save(reports, tmp_path / "report.yaml")
    loaded = repository.load(tmp_path / "report.yaml")

    assert sorted(report.layer_name for report in loaded) == ["bronze", "raw"]


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 5: the report does not survive save and load unchanged")
def test_roundtrip_returns_equal_report(tmp_path, report_factory):
    repository = YAMLReportRepository()
    report = report_factory(results=[_result("a"), _result("b", points=5)], detected_variables={"power_kw": "P"})

    repository.save([report], tmp_path / "report.yaml")
    (loaded,) = repository.load(tmp_path / "report.yaml")

    assert loaded == report
