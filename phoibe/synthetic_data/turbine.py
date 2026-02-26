import numpy as np
import pandas as pd

from ._turbine_noise import CurtailmentNight
from ._turbine_noise import CurtailmentToZero
from ._turbine_noise import DeleteEntries
from ._turbine_noise import Freeze
from ._turbine_noise import GeometricSegments
from ._turbine_noise import RowDrop
from ._turbine_noise import Spike
from ._turbine_noise import UniformSegments
from ._turbine_scada import Time
from ._turbine_scada import WtgType
from ._turbine_scada import _generate_weibull_timeseries
from ._turbine_scada import _wind_speed_to_pitch_angle
from ._turbine_scada import _wind_speed_to_power
from ._turbine_scada import _wind_speed_to_rotor_speed

DEFAULT_WTG = WtgType(nominal_power=5600, rotor_diameter=150, cut_in=3.0, cut_out=25.0, rated=13.0, tsr=8.0)
DEFAULT_TIME = Time(start="2026-02-14T04:00:00", freq="10min", periods=576)


def generate_wtg_scada(A=10, k=2, time: Time = DEFAULT_TIME, wtg_type: WtgType = DEFAULT_WTG):
    """Generate a timeseries of WTG Scada data for the given `Time` period, a given `WtgType` configuration."""
    index = pd.date_range(start=time.start, freq=time.freq, periods=time.periods)

    wind_speed = _generate_weibull_timeseries(A=A, k=k, n_steps=len(index), delta_t=600, theta=1 / 7200)
    df = pd.DataFrame(data={"wind_speed": wind_speed}, index=index)

    power = _wind_speed_to_power(wind_speeds=df["wind_speed"].to_numpy(dtype=np.floating), wtg_type=wtg_type)
    df["power"] = power[:, 1]

    df["rotor_speed"] = _wind_speed_to_rotor_speed(df["wind_speed"].to_numpy(dtype=np.floating), wtg_type=wtg_type)
    df["pitch_angle"] = _wind_speed_to_pitch_angle(df["wind_speed"].to_numpy(dtype=np.floating), wtg_type=wtg_type)
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


pipeline = MessUpPipeline(seed=23)


def create_default_messup_pipeline(pipeline: MessUpPipeline = MessUpPipeline()) -> MessUpPipeline:
    """Create a basic messup pipeline for scada data."""
    pipeline.add(CurtailmentNight(limit=3200, column="power"))
    pipeline.add(CurtailmentToZero(segments=GeometricSegments(n=5, p=0.1), columns_to_keep=["wind_speed"]))
    pipeline.add(Freeze(segments=GeometricSegments(n=1, p=0.3), column="power"))
    pipeline.add(Freeze(segments=GeometricSegments(n=3, p=0.2), column="rotor_speed"))
    pipeline.add(RowDrop(segments=UniformSegments(n=3, min_len=3, max_len=7)))
    pipeline.add(DeleteEntries(segments=UniformSegments(n=5, min_len=1, max_len=1), columns=["pitch_angle"]))
    pipeline.add(Spike(segments=UniformSegments(n=5, min_len=1, max_len=1), column="power", magnitude=2000))
    pipeline.add(Spike(segments=UniformSegments(n=5, min_len=1, max_len=1), column="wind_speed", magnitude=5))
    return pipeline
