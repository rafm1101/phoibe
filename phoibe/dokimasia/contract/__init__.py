"""The contract: schema, building it from decoded data, and judging a profile against it (ADR-04, ADR-05).

Imports `core` only. Reading files is the business of `adapters`.
"""

from phoibe.dokimasia.contract.build import ContractError, build_contract
from phoibe.dokimasia.contract.judge import judge_layer
from phoibe.dokimasia.contract.limits import (
    ResolvedLimit,
    list_subject_attributes,
    resolve_limit,
    validate_subject_references,
)
from phoibe.dokimasia.contract.schema import (
    Band,
    CheckSpec,
    Contract,
    LayerSpec,
    Limit,
    MeasureSpec,
    SubjectRef,
)

__all__ = [
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
]
