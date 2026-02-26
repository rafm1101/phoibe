import dataclasses

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats
from numpy.typing import NDArray


@dataclasses.dataclass
class WtgType:
    nominal_power: float
    rotor_diameter: float
    cut_in: float
    rated: float
    cut_out: float
    tsr: float


@dataclasses.dataclass
class Time:
    start: str
    freq: str | pd.Timedelta
    periods: int


def _generate_weibull_timeseries(
    A: float, k: float, n_steps: int, delta_t: int, theta=0.1, mu=0.0, random_state=None
) -> NDArray[np.floating]:
    """Generate a sequence of Weibull(A, k)-distributed, correlated random variables.

    The resulting sequence is generated from an Ornstein-Uhlenbeck process via quantile transformations
    on the marginals. Any correlation structure is defined on the level of the latent process, and passed on
    through these transformations.
    Note that the strengh of the driving noise is set to `sqrt(2theta)` to balance out noise and `delta_t`.
    """
    rng = np.random.default_rng(random_state)
    decay = np.exp(-theta * delta_t)

    ornstein_uhlenbeck = np.zeros(n_steps, dtype=np.float64)
    ornstein_uhlenbeck[0] = mu

    # noise_level = sigma * np.sqrt((1 - decay**2) / (2 * theta))
    noise_level = np.sqrt(1 - decay**2)

    for t in range(1, n_steps):
        ornstein_uhlenbeck[t] = ornstein_uhlenbeck[t - 1] * decay + mu * (1 - decay) + noise_level * rng.normal()

    u = scipy.stats.norm.cdf(ornstein_uhlenbeck)
    x = np.float_power(-np.log(1 - u), 1 / k) * A

    return x


def _wind_speed_to_power(
    wtg_type: WtgType, wind_speeds: NDArray[np.floating] = np.linspace(0, 30, num=61)
) -> NDArray[np.floating]:
    """Convert given wind speeds to powers for the `WtgType` configuration.

    The powercurve features cut-in, rated and cut-out wind speed, and remains very basic.
    """
    t = np.array(
        [0, wtg_type.cut_in - 0.5, wtg_type.cut_in, wtg_type.rated, wtg_type.cut_out, wtg_type.cut_out + 0.5, 30]
    )
    p = np.array([0, 0, 0.04 * wtg_type.nominal_power, wtg_type.nominal_power, wtg_type.nominal_power, 0, 0])
    values = scipy.interpolate.make_interp_spline(t, p, k=1)(wind_speeds)
    return np.column_stack([wind_speeds, values])


def _wind_speed_to_rotor_speed(wind_speeds: NDArray[np.floating], wtg_type: WtgType) -> NDArray[np.floating]:
    """Convert given wind speeds to rotor speeds for the `WtgType` configuration.

    The conversion features a proportional relation for partial loads and a cap for full loads.
    """
    rotor_speed = wtg_type.tsr / (np.pi * wtg_type.rotor_diameter) * wind_speeds
    rotor_speed_before_cut_in = wtg_type.tsr / (np.pi * wtg_type.rotor_diameter) * (wtg_type.cut_in - 0.5)
    rotor_speed_at_rated_wind_speed = wtg_type.tsr / (np.pi * wtg_type.rotor_diameter) * wtg_type.rated

    rotor_speed = np.where(
        wind_speeds < wtg_type.cut_in, (rotor_speed - rotor_speed_before_cut_in) / wtg_type.cut_in / 0.5, rotor_speed
    )
    rotor_speed = np.where(wind_speeds < wtg_type.cut_in - 0.5, 0, rotor_speed)

    rotor_speed = np.where(wind_speeds > wtg_type.rated, rotor_speed_at_rated_wind_speed, rotor_speed)
    rotor_speed = np.where(wind_speeds > wtg_type.cut_out, 0, rotor_speed)
    return rotor_speed * 60


def _wind_speed_to_pitch_angle(wind_speeds: NDArray[np.floating], wtg_type: WtgType) -> NDArray[np.floating]:
    """Convert given wind speeds to pitch for the `WtgType` configuration.

    The conversion features a heuristic increase for full loads and is constant below the rated wind speed.
    """
    wind_speeds = np.asarray(wind_speeds, dtype=np.float64)
    rated_ratio = wtg_type.rated / wind_speeds
    rated_ratio = np.where(rated_ratio > 1, 1.0, rated_ratio**3)
    pitch_angle = np.rad2deg(np.arccos(rated_ratio)) * 0.3

    pitch_angle = np.where(wind_speeds < wtg_type.rated, 1, pitch_angle)
    pitch_angle = np.where(wind_speeds > wtg_type.cut_out, 90, pitch_angle)
    return pitch_angle
