"""Measures of power curtailment: power held at a level below what the wind allows."""

import dataclasses
import enum
import typing
import zoneinfo

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.signal
import scipy.stats

from phoibe.dokimasia import core, rules


def _round_to_grid(values: npt.NDArray[np.float64], resolution_kw: float) -> npt.NDArray[np.float64]:
    """Round to multiples of the resolution; the power of a held level repeats exactly only on such a grid."""
    return np.round(values / resolution_kw) * resolution_kw


def _find_modal_level(rounded: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Find the most frequent value and its share. The lowest value wins a tie."""
    values, counts = np.unique(rounded, return_counts=True)
    position = int(np.argmax(counts))
    return float(values[position]), float(counts[position]) / len(rounded)


class LevelMethod(enum.StrEnum):
    """How power levels are found."""

    COUNT = "count"
    """A value that at least `min_count` rows share is a level."""
    KDE = "kde"
    """The peaks of a Gaussian kernel density estimate are the levels; the method of `layered`."""


def _is_near(
    power: npt.NDArray[np.float64], levels: npt.NDArray[np.float64], resolution_kw: float
) -> npt.NDArray[np.bool_]:
    """Mark the values within one resolution step of a level, so that noise around a held level stays in its run."""
    if len(levels) == 0:
        return np.zeros(len(power), dtype=bool)
    with np.errstate(invalid="ignore"):
        return np.asarray(np.abs(power[:, None] - levels[None, :]).min(axis=1) <= resolution_kw, dtype=bool)


def _find_levels_by_count(rounded: npt.NDArray[np.float64], min_count: int) -> npt.NDArray[np.float64]:
    values, counts = np.unique(rounded, return_counts=True)
    return values[counts >= min_count]


def _find_levels_by_kde(
    power: npt.NDArray[np.float64], resolution_kw: float, prominence_threshold: float
) -> npt.NDArray[np.float64]:
    """Find the peaks of the density of the power values on a grid; the algorithm of `layered`'s curtailment rule.

    The density is evaluated on multiples of the resolution from five steps below the lowest to six above the highest
    value, and a peak is a level if its prominence is above the threshold. If all values are equal, the density is
    degenerate, which `layered` could not handle; that value is the level.
    """
    lowest = np.floor(power.min() / resolution_kw) * resolution_kw
    highest = np.ceil(power.max() / resolution_kw) * resolution_kw
    if power.min() == power.max():
        return np.array([lowest])
    grid = np.arange(lowest - 5 * resolution_kw, highest + 6 * resolution_kw, resolution_kw)
    density = scipy.stats.gaussian_kde(power)(grid)
    peaks, properties = scipy.signal.find_peaks(density, prominence=(None, None))
    return np.asarray(grid[peaks[properties["prominences"] > prominence_threshold]], dtype=float)


def _check_resolution(resolution_kw: float) -> None:
    if not resolution_kw > 0.0:
        raise ValueError(f"resolution_kw must be above zero, got {resolution_kw:g}")


@dataclasses.dataclass(frozen=True, slots=True)
class PowerLevelsMeasure:
    """Find the power levels that the turbine holds at high wind speed, and the rows that sit on a curtailed one.

    At wind speeds from `min_wind_speed` on, a turbine produces its rated power. Power is rounded to multiples of
    `resolution_kw`. With the method `count`, a value that at least `min_count` of these rows share is a level; with
    `kde`, the levels are the prominent peaks of a Gaussian density estimate of the power at these rows, as in
    `layered` (there the threshold of 14 m/s was fixed; here it is `min_wind_speed`). A level more than
    `resolution_kw` below `rated_power_kw` is a curtailed level. Every row at high wind speed within `resolution_kw` of
    a curtailed level is a finding of kind `curtailment_level`, so that noise around a held level does not break its
    run. Rows below `min_wind_speed` are not looked at: low power is normal there.

    Metrics, on the power signal: `n_high_wind_rows`, `n_levels`, `n_curtailed_levels`, `highest_level_kw` (zero if
    there is no level), `n_curtailed_rows` and `share_curtailed_rows` (of the high wind rows).
    """

    ASPECT: typing.ClassVar[str] = "validity"
    METRICS: typing.ClassVar[tuple[str, ...]] = (
        "n_high_wind_rows",
        "n_levels",
        "n_curtailed_levels",
        "highest_level_kw",
        "n_curtailed_rows",
        "share_curtailed_rows",
    )

    power: str
    """Catalog name of the power signal."""
    wind_speed: str
    """Catalog name of the wind speed signal."""
    rated_power_kw: float
    """Rated power of the turbine."""
    min_wind_speed: float
    """Wind speed from which on the turbine is expected to produce rated power, usually the rated wind speed."""
    resolution_kw: float = 10.0
    """Grid that power is rounded to."""
    method: LevelMethod = LevelMethod.COUNT
    """How levels are found."""
    min_count: int = 5
    """Fewest rows at one value for it to be a level; used by `count`."""
    prominence_threshold: float = 1e-7
    """Prominence of a density peak for it to be a level; used by `kde`."""
    min_rows: int = 20
    """Fewest rows at high wind speed that the measure needs."""
    measure_id: str = "power_levels"
    """Identifier of this measure in a profile; the contract key when built from a contract."""

    def __post_init__(self) -> None:
        """Require a positive resolution and positive counts."""
        _check_resolution(self.resolution_kw)
        if self.min_count < 1 or self.min_rows < 1:
            raise ValueError(f"min_count and min_rows must be at least 1, got {self.min_count} and {self.min_rows}")
        if not self.prominence_threshold >= 0.0:
            raise ValueError(f"prominence_threshold must be zero or more, got {self.prominence_threshold:g}")

    @property
    def id(self) -> str:
        """Identifier of the measure."""
        return self.measure_id

    @property
    def aspect(self) -> str:
        """Aspect of data quality."""
        return self.ASPECT

    def measure(self, dataset: core.Dataset) -> core.Measured | core.NotEvaluable:
        """Measure the power levels at high wind speed.

        Parameters
        ----------
        dataset
            The data.

        Returns
        -------
        Measured | NotEvaluable
            Metrics and `curtailment_level` findings; not evaluable if a signal is missing or ambiguous or fewer than
            `min_rows` rows have power and wind speed at high wind speed.
        """
        numbers = rules.read_numeric_signals(dataset, (self.power, self.wind_speed), self.id)
        if isinstance(numbers, core.NotEvaluable):
            return numbers
        power, wind = numbers.iloc[:, 0].to_numpy(), numbers.iloc[:, 1].to_numpy()
        with np.errstate(invalid="ignore"):
            high = (wind >= self.min_wind_speed) & ~np.isnan(power)
        n_high = int(high.sum())
        if n_high < self.min_rows:
            return core.NotEvaluable(
                self.id,
                core.NotEvaluableReason.INSUFFICIENT_DATA,
                f"{n_high} rows at wind speeds from {self.min_wind_speed:g}, {self.min_rows} are needed",
            )

        rounded = _round_to_grid(power, self.resolution_kw)
        if self.method is LevelMethod.KDE:
            levels = _find_levels_by_kde(power[high], self.resolution_kw, self.prominence_threshold)
        else:
            levels = _find_levels_by_count(rounded[high], self.min_count)
        curtailed_levels = levels[levels < self.rated_power_kw - self.resolution_kw]
        curtailed = high & _is_near(power, curtailed_levels, self.resolution_kw)
        runs = rules.find_runs(curtailed)
        index = numbers.index
        n_curtailed = int(curtailed.sum())
        metrics = (
            core.Metric("n_high_wind_rows", float(n_high), signal=self.power),
            core.Metric("n_levels", float(len(levels)), signal=self.power),
            core.Metric("n_curtailed_levels", float(len(curtailed_levels)), signal=self.power),
            core.Metric("highest_level_kw", float(levels.max()) if len(levels) else 0.0, unit="kW", signal=self.power),
            core.Metric("n_curtailed_rows", float(n_curtailed), signal=self.power),
            core.Metric("share_curtailed_rows", n_curtailed / n_high, signal=self.power),
        )
        findings = tuple(
            core.Finding("curtailment_level", (self.power,), index[start], index[end - 1], end - start)
            for start, end in runs
        )
        return core.Measured(self.id, metrics, findings)


@dataclasses.dataclass(frozen=True, slots=True)
class NightLimitMeasure:
    """Find a power limit that applies at night and not by day.

    At wind speeds from `min_wind_speed` on, the most frequent power value (rounded to multiples of `resolution_kw`)
    is found for the night hours and for the rest of the day. If the night value is more than `resolution_kw` below the
    day value and at least `min_share` of the high wind rows at night sit on it, there is a night limit. Then every
    night row on the night value, at any wind speed, is a finding of kind `night_limit`.

    Night runs from `night_start_hour` (inclusive) to `night_end_hour` (exclusive) and wraps over midnight if the start
    is later. The hours are those of the time axis as given, unless `time_zone` is set: then the time axis, which must
    have a time zone, is converted to it first, so that the night stays at the same local hours when daylight saving
    time shifts the offset from UTC.

    Metrics, on the power signal: `n_night_high_wind_rows`, `n_day_high_wind_rows`, `night_level_kw`, `day_level_kw`,
    `share_night_at_level`, `night_limit_depth_kw` (day level less night level; zero or less means no limit) and
    `n_limited_rows`.
    """

    ASPECT: typing.ClassVar[str] = "validity"
    METRICS: typing.ClassVar[tuple[str, ...]] = (
        "n_night_high_wind_rows",
        "n_day_high_wind_rows",
        "night_level_kw",
        "day_level_kw",
        "share_night_at_level",
        "night_limit_depth_kw",
        "n_limited_rows",
    )

    power: str
    """Catalog name of the power signal."""
    wind_speed: str
    """Catalog name of the wind speed signal."""
    min_wind_speed: float
    """Wind speed from which on the turbine is expected to produce the same power day and night."""
    night_start_hour: int
    """First hour of the night, 0 to 23."""
    night_end_hour: int
    """First hour after the night, 0 to 23."""
    time_zone: str | None = None
    """Time zone of the night hours, for example `Europe/Berlin`; `None` for the hours of the time axis as given."""
    resolution_kw: float = 10.0
    """Grid that power is rounded to."""
    min_rows: int = 10
    """Fewest rows at high wind speed that the night and the day each need."""
    min_share: float = 0.5
    """Share of the high wind rows at night that must sit on the night value."""
    measure_id: str = "night_limit"
    """Identifier of this measure in a profile; the contract key when built from a contract."""

    def __post_init__(self) -> None:
        """Require valid hours, a positive resolution and sensible counts."""
        _check_resolution(self.resolution_kw)
        for hour in (self.night_start_hour, self.night_end_hour):
            if not 0 <= hour <= 23:
                raise ValueError(f"night hours must be from 0 to 23, got {hour}")
        if self.time_zone is not None:
            try:
                zoneinfo.ZoneInfo(self.time_zone)
            except (zoneinfo.ZoneInfoNotFoundError, ValueError) as error:
                raise ValueError(f"unknown time zone '{self.time_zone}'") from error
        if self.night_start_hour == self.night_end_hour:
            raise ValueError(f"the night needs different start and end hours, got {self.night_start_hour} twice")
        if self.min_rows < 1 or not 0.0 < self.min_share <= 1.0:
            raise ValueError(
                f"min_rows must be at least 1 and min_share in (0, 1], got {self.min_rows}, {self.min_share}"
            )

    @property
    def id(self) -> str:
        """Identifier of the measure."""
        return self.measure_id

    @property
    def aspect(self) -> str:
        """Aspect of data quality."""
        return self.ASPECT

    def measure(self, dataset: core.Dataset) -> core.Measured | core.NotEvaluable:
        """Measure the night limit.

        Parameters
        ----------
        dataset
            The data.

        Returns
        -------
        Measured | NotEvaluable
            Metrics and `night_limit` findings; not evaluable if a signal is missing or ambiguous or the night or the
            day has fewer than `min_rows` rows at high wind speed.
        """
        numbers = rules.read_numeric_signals(dataset, (self.power, self.wind_speed), self.id)
        if isinstance(numbers, core.NotEvaluable):
            return numbers
        power, wind = numbers.iloc[:, 0].to_numpy(), numbers.iloc[:, 1].to_numpy()
        axis = pd.DatetimeIndex(numbers.index)
        if self.time_zone is not None:
            if axis.tz is None:
                return core.NotEvaluable(
                    self.id,
                    core.NotEvaluableReason.MISSING_STRUCTURE,
                    f"the time axis has no time zone, which the night hours in '{self.time_zone}' need",
                )
            axis = axis.tz_convert(self.time_zone)
        hours = axis.hour.to_numpy()
        if self.night_start_hour < self.night_end_hour:
            night = (hours >= self.night_start_hour) & (hours < self.night_end_hour)
        else:
            night = (hours >= self.night_start_hour) | (hours < self.night_end_hour)
        with np.errstate(invalid="ignore"):
            high = (wind >= self.min_wind_speed) & ~np.isnan(power)
        night_high, day_high = high & night, high & ~night
        n_night, n_day = int(night_high.sum()), int(day_high.sum())
        if n_night < self.min_rows or n_day < self.min_rows:
            return core.NotEvaluable(
                self.id,
                core.NotEvaluableReason.INSUFFICIENT_DATA,
                f"{n_night} night rows and {n_day} day rows at wind speeds from {self.min_wind_speed:g}, "
                f"{self.min_rows} of each are needed",
            )

        rounded = _round_to_grid(power, self.resolution_kw)
        night_level, night_share = _find_modal_level(rounded[night_high])
        day_level, _ = _find_modal_level(rounded[day_high])
        depth = day_level - night_level
        limited = np.zeros(len(power), dtype=bool)
        if depth > self.resolution_kw and night_share >= self.min_share:
            limited = night & ~np.isnan(power) & (rounded == night_level)
        runs = rules.find_runs(limited)
        index = numbers.index
        metrics = (
            core.Metric("n_night_high_wind_rows", float(n_night), signal=self.power),
            core.Metric("n_day_high_wind_rows", float(n_day), signal=self.power),
            core.Metric("night_level_kw", night_level, unit="kW", signal=self.power),
            core.Metric("day_level_kw", day_level, unit="kW", signal=self.power),
            core.Metric("share_night_at_level", night_share, signal=self.power),
            core.Metric("night_limit_depth_kw", depth, unit="kW", signal=self.power),
            core.Metric("n_limited_rows", float(limited.sum()), signal=self.power),
        )
        findings = tuple(
            core.Finding(
                "night_limit",
                (self.power,),
                index[start],
                index[end - 1],
                end - start,
                magnitude=depth,
                magnitude_unit="kW",
            )
            for start, end in runs
        )
        return core.Measured(self.id, metrics, findings)
