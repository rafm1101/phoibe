import dataclasses

import numpy as np
import pandas as pd

from ._turbine_noise import (
    CurtailmentNight,
    CurtailmentToZero,
    DeleteEntries,
    Freeze,
    GeometricSegments,
    MessLabel,
    RowDrop,
    Spike,
    UniformSegments,
)
from ._turbine_scada import (
    Time,
    WtgType,
    _generate_weibull_timeseries,
    _wind_speed_to_pitch_angle,
    _wind_speed_to_power,
    _wind_speed_to_rotor_speed,
)

DEFAULT_WTG = WtgType(nominal_power=5600, rotor_diameter=150, cut_in=3.0, cut_out=25.0, rated=13.0, tsr=8.0)
DEFAULT_TIME = Time(start="2026-02-14T04:00:00", freq="10min", periods=576)


def generate_wtg_scada(
    A=10,
    k=2,
    time: Time = DEFAULT_TIME,
    wtg_type: WtgType = DEFAULT_WTG,
    latent_freq: str | None = None,
    seed: int | None = None,
):
    """Generate a timeseries of WTG Scada data for the given `Time` period, a given `WtgType` configuration.

    Parameters
    ----------
    A
        Scale parameter of the Distribution of the underlying wind speed.
    k
        Shape parameter of the Distribution of the underlying wind speed.
    time
        Sampling window.
    wtg_type
        Turbine information.
    latent_freq
        If not `None`, a finer sampling frequency from which the output average, minimum and maximum are determnined.
        `time.freq` must be a multiple of `latent_freq`. Ignored if `None`.
    seed
        Seed of the wind speed process. The data is reproducible if given and differs from call to call if `None`.

    Returns
    -------
    df
        Dataframe holding a generated timeseries on the index generated from `time` w/ columns
        `power`, `wind_speed`, `rotor_speed`, `pitch_angle`.
        If `latent_freq` is not `None`, for each column minimum and maximum are added w/ suffices `_min` and `_max`.
    """
    if latent_freq is None:
        index = pd.date_range(start=time.start, freq=time.freq, periods=time.periods, name="datetime")
        latent_index = index
        delta_t = pd.to_timedelta(time.freq).seconds
    else:
        index = pd.date_range(start=time.start, freq=time.freq, periods=time.periods, name="datetime")
        factor = int(pd.to_timedelta(time.freq).seconds / pd.to_timedelta(latent_freq).seconds)
        latent_index = pd.date_range(start=time.start, freq=latent_freq, periods=time.periods * factor, name="datetime")
        delta_t = pd.to_timedelta(latent_freq).seconds

    wind_speed = _generate_weibull_timeseries(
        A=A, k=k, n_steps=len(latent_index), delta_t=delta_t, theta=1 / 7200, random_state=seed
    )
    latent_df = pd.DataFrame(data={"wind_speed": wind_speed}, index=latent_index)

    power = _wind_speed_to_power(wind_speeds=latent_df["wind_speed"].to_numpy(), wtg_type=wtg_type)
    latent_df["power"] = power[:, 1]

    latent_df["rotor_speed"] = _wind_speed_to_rotor_speed(latent_df["wind_speed"].to_numpy(), wtg_type=wtg_type)
    latent_df["pitch_angle"] = _wind_speed_to_pitch_angle(latent_df["wind_speed"].to_numpy(), wtg_type=wtg_type)

    column_order = ["power", "wind_speed", "rotor_speed", "pitch_angle"]
    if latent_freq is None:
        df = latent_df.loc[:, column_order]
    else:
        resampler = latent_df.resample(rule=time.freq)
        min_df, mean_df, max_df = resampler.min(), resampler.mean(), resampler.max()
        df = pd.concat([mean_df, min_df.add_suffix("_min"), max_df.add_suffix("_max")], axis="columns")
        columns = [key + suffix for key in column_order for suffix in ["", "_min", "_max"]]
        df = df.loc[:, columns]

    return df


class MessUpPipeline:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.steps = []

    def add(self, step):
        self.steps.append(step)

    def apply(self, df):
        for step in self.steps:
            df = step.apply(df, self.rng)
        return df

    def apply_labelled(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[MessLabel]]:
        """Apply the steps and return the messed-up data together with the ground truth of the injected faults.

        The data is identical to what `apply` returns for the same seed. A step without `apply_labelled`, such as
        a custom one, is applied as before and contributes no labels.
        """
        labels: list[MessLabel] = []
        for index, step in enumerate(self.steps):
            if hasattr(step, "apply_labelled"):
                df, step_labels = step.apply_labelled(df, self.rng)
                labels.extend(dataclasses.replace(label, step=index) for label in step_labels)
            else:
                df = step.apply(df, self.rng)
        return df, labels


def messlabel_to_frame(labels: list[MessLabel]) -> pd.DataFrame:
    """Tabulate fault labels, one row per injected segment."""
    columns = ["step", "kind", "columns", "start", "end", "n_rows", "n_changed", "detail"]
    return pd.DataFrame([{key: getattr(label, key) for key in columns} for label in labels], columns=columns)


def messlabel_mask(
    labels: list[MessLabel],
    index: pd.Index,
    *,
    kind: str | None = None,
    column: str | None = None,
    effective_only: bool = True,
) -> pd.Series:
    """Mark the timestamps of `index` that lie in labelled fault segments.

    Parameters
    ----------
    labels
        Ground truth as returned by `MessUpPipeline.apply_labelled`.
    index
        Timestamps to mark, usually the index of the messed-up data. Dropped rows are not part of it.
    kind
        Restrict to faults of this step class, e.g. `"Freeze"`.
    column
        Restrict to faults addressing this column.
    effective_only
        Skip segments that did not alter any value, e.g. a freeze of an already constant signal.

    Returns
    -------
    pd.Series
        Boolean series on `index`.
    """
    mask = pd.Series(False, index=index)
    for label in labels:
        if kind is not None and label.kind != kind:
            continue
        if column is not None and column not in label.columns:
            continue
        if effective_only and label.n_changed == 0:
            continue
        mask |= (index >= label.start) & (index <= label.end)
    return mask


pipeline = MessUpPipeline(seed=23)


def create_default_messup_pipeline(
    pipeline: MessUpPipeline | None = None, incidence: float = 1.0, level: float = 1.0
) -> MessUpPipeline:
    """Create a basic messup pipeline for scada data w/ four columns."""
    pipeline = pipeline if pipeline is not None else MessUpPipeline()

    gsegments_frequent_durable = GeometricSegments(n=int(11 * incidence), p=0.1 / level)
    gsegments_medium_short = GeometricSegments(n=int(5 * incidence), p=0.5 / level)
    gsegments_rare_medium = GeometricSegments(n=int(incidence), p=0.3 / level)
    usegments_medium_medium = UniformSegments(n=int(5 * incidence), min_len=int(5 * level), max_len=int(11 * level))
    usegments_frequent_short = UniformSegments(n=int(11 * incidence), min_len=int(1 * level), max_len=int(3 * level))
    usegments_frequent_unique = UniformSegments(n=int(11 * incidence), min_len=1, max_len=1)

    pipeline.add(CurtailmentNight(limit=3200, column="power"))
    pipeline.add(CurtailmentToZero(segments=gsegments_frequent_durable, columns_to_keep=["wind_speed"]))
    pipeline.add(Freeze(segments=gsegments_medium_short, column="power"))
    pipeline.add(Freeze(segments=gsegments_rare_medium, column="rotor_speed"))
    pipeline.add(RowDrop(segments=usegments_medium_medium))
    pipeline.add(DeleteEntries(segments=usegments_frequent_short, columns=["pitch_angle"]))
    pipeline.add(Spike(segments=usegments_frequent_unique, column="power", magnitude=2000))
    pipeline.add(Spike(segments=usegments_frequent_unique, column="wind_speed", magnitude=5))
    return pipeline


def create_extended_messup_pipeline(
    pipeline: MessUpPipeline | None = None, incidence: float = 1.0, level: float = 1.0
) -> MessUpPipeline:
    """Create a basic messup pipeline for scada data."""
    pipeline = pipeline if pipeline is not None else MessUpPipeline()

    gsegments_frequent_durable = GeometricSegments(n=int(11 * incidence), p=0.1 / level)
    gsegments_medium_short = GeometricSegments(n=int(5 * incidence), p=0.5 / level)
    gsegments_rare_medium = GeometricSegments(n=int(incidence), p=0.3 / level)
    usegments_medium_medium = UniformSegments(n=int(5 * incidence), min_len=int(5 * level), max_len=int(11 * level))
    usegments_frequent_short = UniformSegments(n=int(11 * incidence), min_len=int(1 * level), max_len=int(3 * level))
    usegments_frequent_unique = UniformSegments(n=int(11 * incidence), min_len=1, max_len=1)

    pipeline.add(CurtailmentNight(limit=3200, column="power"))
    pipeline.add(CurtailmentNight(limit=3200, column="power_min"))
    pipeline.add(CurtailmentNight(limit=3200, column="power_max"))
    pipeline.add(
        CurtailmentToZero(
            segments=gsegments_frequent_durable, columns_to_keep=["wind_speed", "wind_speed_min", "wind_speed_max"]
        )
    )
    pipeline.add(Freeze(segments=gsegments_medium_short, column="power"))
    pipeline.add(Freeze(segments=gsegments_medium_short, column="power_min"))
    pipeline.add(Freeze(segments=gsegments_medium_short, column="power_max"))
    pipeline.add(Freeze(segments=gsegments_rare_medium, column="rotor_speed"))
    pipeline.add(Freeze(segments=gsegments_rare_medium, column="rotor_speed_max"))
    pipeline.add(RowDrop(segments=usegments_medium_medium))
    pipeline.add(
        DeleteEntries(segments=usegments_frequent_short, columns=["pitch_angle", "pitch_angle_min", "pitch_angle_max"])
    )
    pipeline.add(Spike(segments=usegments_frequent_unique, column="power", magnitude=2000))
    pipeline.add(Spike(segments=usegments_frequent_unique, column="power_max", magnitude=2000))
    pipeline.add(Spike(segments=usegments_frequent_unique, column="wind_speed", magnitude=5))
    return pipeline
