"""The subject of a run (ADR-01)."""

import dataclasses
import types
from collections.abc import Mapping


@dataclasses.dataclass(frozen=True, slots=True)
class Subject:
    """The thing that is observed: one device with its specification. One run covers one subject and one dataset.

    The specification is a plain mapping so that no domain term enters the model (rated power and rotor diameter for
    a turbine, for example). It is copied and read-only after construction.
    """

    id: str
    """Identifier of the subject, for example a turbine id."""
    device_type: str
    """Kind of device, for example `wind_turbine`; a domain pack gives it meaning."""
    specification: Mapping[str, object] = dataclasses.field(default_factory=dict)
    """Properties of the device, such as rated power or the site time zone."""

    def __post_init__(self) -> None:
        """Replace the specification by a read-only copy."""
        object.__setattr__(self, "specification", types.MappingProxyType(dict(self.specification)))
