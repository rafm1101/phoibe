import pytest

from phoibe.geography.complexity.rix.computations import checks
from phoibe.geography.complexity.rix.pipelines import errors
from phoibe.geography.complexity.rix.pipelines.checks.findings import LocatedFinding


@pytest.mark.parametrize("error", [errors.InputRequirementError, errors.ConfigurationError, errors.CheckAborted])
def test_input_problems_derive_from_value_error_so_that_existing_handlers_keep_working(error):
    assert issubclass(error, errors.PipelineError) and issubclass(error, ValueError)


@pytest.mark.parametrize("error", [errors.PortError, errors.ContractViolation])
def test_technical_problems_do_not_derive_from_value_error_so_that_they_do_not_look_like_bad_input(error):
    assert issubclass(error, errors.PipelineError) and not issubclass(error, ValueError)


def test_contract_violation_derives_from_runtime_error():
    assert issubclass(errors.ContractViolation, RuntimeError)


def test_check_aborted_returns_the_findings_and_a_message_listing_them():
    findings = [
        LocatedFinding(checks.Finding(checks.NAN_COUNT, "some NaN"), "warn"),
        LocatedFinding(checks.Finding(checks.UNIQUE_IDS, "dup"), "abort"),
    ]
    error = errors.CheckAborted(findings)
    assert error.findings == tuple(findings)
    assert str(error) == "Run aborted by 2 finding(s):\n  - [warn] nan_count: some NaN\n  - [abort] unique_ids: dup"


def test_contract_violation_returns_the_problems_and_a_message_listing_them():
    error = errors.ContractViolation(["first", "second"])
    assert error.problems == ("first", "second")
    assert str(error) == "Artifacts violate the contract:\n  - first\n  - second"
