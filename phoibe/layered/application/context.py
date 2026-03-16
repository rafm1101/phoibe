from __future__ import annotations

import dataclasses
from typing import Any

from phoibe.layered.core.entities import ValidationMode


@dataclasses.dataclass(frozen=True)
class ValidationContext:
    """Value object of a layer validation configuration."""

    layer_name: str
    """Name of the validation layer (raw, bronze, silver, gold)."""
    detected_variables: dict[str, str | None]
    """Mapping of standard variable names to actual column keys."""
    turbine_id: str
    """Identifier of the turbine whose data is being validated."""
    validation_mode: ValidationMode = ValidationMode.PROFILING
    """Pure descriptive (PROFILING) or prescriptive (CONTRACT) rule handling."""
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Additional layer-specific metadata"""

    def get_column_key(self, variable: str) -> str | None:
        """Get actual column name for a variable.

        Parameters
        ----------
        variable
            Standard variable name.

        Return
        ------
        str | None
            Actual column key or `None` if not detected.
        """
        return self.detected_variables.get(variable)

    def has_variable(self, variable: str) -> bool:
        """Verify if variable was detected.

        Parameters
        ----------
        variable
            Standard variable name.

        Return
        ------
        bool
            True if variable was detected.
        """
        return self.detected_variables.get(variable) is not None

    @property
    def is_contract_mode(self) -> bool:
        """Check whether validation is in contract mode.

        Return
        ------
        bool
            True if in CONTRACT mode.
        """
        return self.validation_mode == ValidationMode.CONTRACT

    @property
    def is_profiling_mode(self) -> bool:
        """Check whether validation is in profiling mode.

        Return
        ------
        bool
            True if in PROFILING mode.
        """
        return self.validation_mode == ValidationMode.PROFILING


__all__ = ["ValidationContext"]
