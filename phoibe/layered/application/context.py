from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any


@dataclass(frozen=True)
class ValidationContext:
    """Value object of a layer validation configuration."""

    layer_name: str
    """Name of the validation layer (raw, bronze, silver, gold)."""
    detected_variables: dict[str, str | None]
    """Mapping of standard variable names to actual column keys."""
    turbine_id: str
    """Identifier of the turbine whose data is being validated."""
    metadata: dict[str, Any] = field(default_factory=dict)
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


__all__ = ["ValidationContext"]
