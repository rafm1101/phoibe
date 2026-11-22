"""The public names of each package. A new public name has to be added here on purpose."""

import importlib
import types

import pytest

EXPECTED = {
    "phoibe.dokimasia": {
        "adapters",
        "compose",
        "contract",
        "core",
        "rules",
        "standardize",
        "wind",
        "Dataset",
        "FlatStore",
        "Subject",
        "advance_dataset",
        "attest_validation",
        "build_run",
        "load_contract",
        "profile_layer",
        "validate_layer",
    },
    "phoibe.dokimasia.core": {
        "Catalog",
        "Dataset",
        "FINGERPRINT_ALGORITHM",
        "Finding",
        "Fingerprint",
        "Measure",
        "Measured",
        "MeasurementError",
        "Metric",
        "NotEvaluable",
        "NotEvaluableReason",
        "Outcome",
        "Profile",
        "Signal",
        "Statistic",
        "Subject",
        "build_finding_mask",
        "Score",
        "compute_score",
        "score_findings",
        "build_verdict_table",
        "compute_fingerprint",
        "profile_dataset",
        "run_measure",
        "ProvenanceMismatchError",
        "ProvenanceReport",
        "report_provenance",
        "Transform",
        "Transformed",
        "TransformRefused",
        "TransformRefusedError",
        "advance_dataset",
        "Attestation",
        "ProvenanceRecord",
        "TransformDescription",
        "build_attestation",
        "Effect",
        "GateFailedError",
        "GateOutcome",
        "GatePolicy",
        "GateReason",
        "GateResult",
        "WorstEffectPolicy",
        "get_effect",
        "CriterionVerdict",
        "Severity",
        "Verdict",
    },
    "phoibe.dokimasia.contract": {
        "Band",
        "CheckSpec",
        "Contract",
        "ContractError",
        "LayerSpec",
        "Limit",
        "MeasureSpec",
        "ResolvedLimit",
        "SubjectRef",
        "build_contract",
        "judge_layer",
        "list_subject_attributes",
        "resolve_limit",
        "validate_subject_references",
    },
    "phoibe.dokimasia.standardize": {"Chain", "LocalizeTimeToUtc", "RenameToCatalog"},
    "phoibe.dokimasia.rules": {
        "Ambiguity",
        "Detection",
        "DetectionError",
        "FrozenRunsMeasure",
        "GapMeasure",
        "MissingRunsMeasure",
        "RangesMeasure",
        "RequiredSignalsMeasure",
        "SignalPattern",
        "StatisticOrderMeasure",
        "detect_signals",
        "find_runs",
        "list_statistic_triples",
        "read_numeric_signals",
    },
    "phoibe.dokimasia.wind": {
        "PATTERNS",
        "LevelMethod",
        "NightLimitMeasure",
        "PowerLevelsMeasure",
        "create_catalog",
        "detect_wind_signals",
    },
    "phoibe.dokimasia.compose": {
        "Builder",
        "ParamKind",
        "ParamSpec",
        "Registry",
        "Rule",
        "SignalUse",
        "Validation",
        "attest_validation",
        "build_run",
        "build_measures",
        "create_default_registry",
        "list_generic_rules",
        "list_wind_rules",
        "profile_layer",
        "validate_layer",
        "validate_layer_rules",
    },
    "phoibe.dokimasia.adapters": {
        "ODCS_API_VERSION",
        "SCHEMA_VERSION",
        "export_odcs",
        "format_odcs",
        "save_odcs",
        "ArtifactError",
        "FlatStore",
        "Run",
        "RunEntry",
        "Store",
        "StoreError",
        "build_run_table",
        "find_entries",
        "VerdictsRecord",
        "attestation_from_data",
        "attestation_to_data",
        "format_artifact",
        "parse_artifact",
        "profile_from_data",
        "profile_to_data",
        "verdicts_from_data",
        "verdicts_to_data",
        "contract_to_data",
        "dump_contract_yaml",
        "load_contract",
        "parse_contract_yaml",
        "save_contract",
    },
}


@pytest.mark.parametrize("package", EXPECTED)
def test_public_api(package):
    module = importlib.import_module(package)

    assert set(module.__all__) == EXPECTED[package]
    assert all(hasattr(module, name) for name in module.__all__)


@pytest.mark.parametrize("package", EXPECTED)
def test_public_api_given_attribute_access_exposes_no_other_public_symbol(package):
    module = importlib.import_module(package)
    exposed = {
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and not isinstance(value, types.ModuleType)
    }

    expected = {name for name in EXPECTED[package] if not isinstance(getattr(module, name), types.ModuleType)}
    assert exposed == expected
