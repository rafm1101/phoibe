"""The contract schema as plain dataclasses (ADR-04). Loading and validation are in `build.py`."""

import dataclasses
import types
from collections.abc import Mapping

from phoibe.dokimasia import core


@dataclasses.dataclass(frozen=True, slots=True)
class SubjectRef:
    """A limit that depends on the subject: `factor * attribute + offset`. One contract then serves several types."""

    attribute: str
    """Key in the specification of the subject, for example `rated_power_kw`."""
    factor: float = 1.0
    """Multiplier of the attribute."""
    offset: float = 0.0
    """Value added after the multiplication."""


type Limit = float | SubjectRef


@dataclasses.dataclass(frozen=True, slots=True)
class Band:
    """An interval of values, bounds included. At least one bound is set."""

    min: Limit | None = None
    """Lower bound, or `None` for none."""
    max: Limit | None = None
    """Upper bound, or `None` for none."""


@dataclasses.dataclass(frozen=True, slots=True)
class MeasureSpec:
    """What to measure: a rule with its parameters, and no thresholds."""

    id: str
    """Key of the measure in its layer; unique there."""
    rule: str
    """Name of the rule; the key if the contract gives none."""
    signal: str | None = None
    """Catalog name of the signal the measure is about, if it is about one."""
    params: Mapping[str, object] = dataclasses.field(default_factory=dict)
    """Parameters of the rule. Limits that depend on the subject are `SubjectRef`. A read-only copy."""

    def __post_init__(self) -> None:
        """Replace the parameters by a read-only copy."""
        object.__setattr__(self, "params", types.MappingProxyType(dict(self.params)))


@dataclasses.dataclass(frozen=True, slots=True)
class CheckSpec:
    """What to judge: one value of one measurement against one or two bands, with a severity."""

    id: str
    """Key of the check in its layer; unique there."""
    measure: str
    """Key of the one measure whose result the check reads."""
    value: str
    """Key of the one metric of that result that the check reads."""
    severity: core.Severity
    """What a violation does to the gate. Required; there is no default."""
    good: Band
    """Values that are good."""
    acceptable: Band | None = None
    """Values that are acceptable but make the pass conditional if the severity is critical; `None` for one band."""
    clause: str | None = None
    """Reference to the source clause, as free text."""
    signal: str | None = None
    """Signal whose metric is read, where the measure reports the metric for several signals."""


@dataclasses.dataclass(frozen=True, slots=True)
class LayerSpec:
    """One layer of the contract: the measures and, separately, the checks."""

    name: str
    """Name of the layer; the contract defines names and count."""
    measures: tuple[MeasureSpec, ...]
    """Measures, in contract order."""
    checks: tuple[CheckSpec, ...]
    """Checks, in contract order."""


@dataclasses.dataclass(frozen=True, slots=True)
class Contract:
    """A versioned, serializable agreement about what the data of each layer has to satisfy."""

    id: str
    """Identifier of the contract."""
    version: str
    """Version of the contract; it appears in every verdict and attestation."""
    domain: str
    """Domain pack the contract is written for, for example `wind`."""
    layers: tuple[LayerSpec, ...]
    """Layers, in contract order."""

    def get_layer(self, name: str) -> LayerSpec:
        """Return a layer by name.

        Parameters
        ----------
        name
            Name of the layer.

        Returns
        -------
        LayerSpec
            The layer.

        Raises
        ------
        KeyError
            If the contract has no such layer.

        """
        for layer in self.layers:
            if layer.name == name:
                return layer
        raise KeyError(name)
