"""Errors of the pipeline.

Inbound adapters translate them into their outer form, e.g. an exit code (E1). Input problems also derive from
`ValueError`, so existing ``except ValueError`` code keeps working.
"""

from __future__ import annotations

import typing
from collections.abc import Sequence

if typing.TYPE_CHECKING:
    from .checks.findings import LocatedFinding


class PipelineError(Exception):
    """Base of all errors raised on purpose by the pipeline."""


class InputRequirementError(PipelineError, ValueError):
    """A precondition on the inputs is not met, e.g. a required input is missing."""


class ConfigurationError(PipelineError, ValueError):
    """A configuration is invalid: unreadable, unknown or fixed parameters, wrong types, values out of range."""


class CheckAborted(PipelineError, ValueError):
    """The run was stopped by findings whose check has severity "abort"."""

    def __init__(self, findings: Sequence[LocatedFinding]) -> None:
        self.findings = tuple(findings)
        lines = "\n".join(f"  - {finding}" for finding in self.findings)
        super().__init__(f"Run aborted by {len(self.findings)} finding(s):\n{lines}")


class PortError(PipelineError):
    """A technical failure of an adapter, e.g. a file that cannot be written. Not a problem of the inputs."""


class ContractViolation(PipelineError, RuntimeError):
    """Written artifacts do not match the product specification. Nothing may be published (DA4)."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        lines = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(f"Artifacts violate the contract:\n{lines}")
