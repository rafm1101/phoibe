"""Localize the time axis to UTC (ADR-11)."""

import dataclasses
import zoneinfo

from phoibe.dokimasia import core


@dataclasses.dataclass(frozen=True, slots=True)
class LocalizeTimeToUtc:
    """Convert the time axis to UTC.

    A naive axis is read as local time of `timezone`. A repeated hour at the end of daylight saving time is resolved
    by order, which holds when the data is sorted and complete around the change; the rule is recorded. If it cannot
    be applied, for example for unsorted data or a time that does not exist, the transform refuses and never guesses.
    An axis that already has a time zone is converted.
    """

    timezone: str
    """IANA name of the site time zone, for example `Europe/Berlin`."""

    name = "localize_time_to_utc"

    def __post_init__(self) -> None:
        """Reject a time zone that does not exist."""
        try:
            zoneinfo.ZoneInfo(self.timezone)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError) as error:
            raise ValueError(f"unknown time zone '{self.timezone}'") from error

    @classmethod
    def from_subject(cls, subject: core.Subject, attribute: str = "site_timezone") -> "LocalizeTimeToUtc":
        """Create the transform from the site time zone in the specification of a subject.

        Parameters
        ----------
        subject
            The subject.
        attribute
            Key of the time zone in the specification.

        Returns
        -------
        LocalizeTimeToUtc
            The transform.

        Raises
        ------
        KeyError
            If the subject has no such attribute.
        ValueError
            If the attribute is not a known time zone.
        """
        value = subject.specification[attribute]
        if not isinstance(value, str):
            raise ValueError(f"subject attribute '{attribute}' is not a time zone name: {value!r}")
        return cls(value)

    def apply(self, dataset: core.Dataset) -> core.Transformed | core.TransformRefused:
        """Convert the time axis.

        Parameters
        ----------
        dataset
            The data.

        Returns
        -------
        Transformed | TransformRefused
            The data with a UTC time axis, or a refusal with the reason.
        """
        axis = dataset.time_axis
        parameters = {"timezone": self.timezone}
        if axis.tz is not None:
            converted = axis.tz_convert("UTC")
            rule = f"axis already had time zone {axis.tz}; converted"
        else:
            try:
                converted = axis.tz_localize(self.timezone, ambiguous="infer", nonexistent="raise").tz_convert("UTC")
            except ValueError as error:
                return core.TransformRefused(
                    f"local time cannot be placed in {self.timezone} without guessing: {error}"
                )
            rule = "naive local time; repeated hour resolved by order (ambiguous=infer); missing hour refused"
        frame = dataset.frame.copy()
        frame.index = converted
        return core.Transformed(frame, parameters, rule)
