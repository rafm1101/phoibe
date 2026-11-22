"""Adapters: files in and out (ADR-07). Imports `core` and `contract`."""

from phoibe.dokimasia.adapters.artifacts import (
    SCHEMA_VERSION,
    ArtifactError,
    VerdictsRecord,
    attestation_from_data,
    attestation_to_data,
    format_artifact,
    parse_artifact,
    profile_from_data,
    profile_to_data,
    verdicts_from_data,
    verdicts_to_data,
)
from phoibe.dokimasia.adapters.contract_files import (
    contract_to_data,
    dump_contract_yaml,
    load_contract,
    parse_contract_yaml,
    save_contract,
)
from phoibe.dokimasia.adapters.store import (
    FlatStore,
    Run,
    RunEntry,
    Store,
    StoreError,
    build_run_table,
    find_entries,
)

__all__ = [
    "SCHEMA_VERSION",
    "ArtifactError",
    "FlatStore",
    "Run",
    "RunEntry",
    "Store",
    "StoreError",
    "VerdictsRecord",
    "attestation_from_data",
    "attestation_to_data",
    "build_run_table",
    "contract_to_data",
    "dump_contract_yaml",
    "find_entries",
    "format_artifact",
    "load_contract",
    "parse_artifact",
    "parse_contract_yaml",
    "profile_from_data",
    "profile_to_data",
    "save_contract",
    "verdicts_from_data",
    "verdicts_to_data",
]
