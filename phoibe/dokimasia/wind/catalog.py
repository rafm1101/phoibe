"""The signals of a wind turbine SCADA data set."""

from phoibe.dokimasia import core

_QUANTITIES = (
    ("power_kw", "power", "kW"),
    ("wind_speed", "wind_speed", "m/s"),
    ("rotor_speed", "rotor_speed", "rpm"),
    ("pitch_angle", "pitch_angle", "deg"),
)
_STATISTICS = (
    ("", core.Statistic.MEAN),
    ("_min", core.Statistic.MIN),
    ("_max", core.Statistic.MAX),
)


def create_catalog() -> core.Catalog:
    """Create the catalog of wind turbine signals.

    Every quantity comes as the mean over a time step, with the minimum and maximum as `_min` and `_max`. The time
    axis is a property of the dataset and not a signal.

    Returns
    -------
    Catalog
        Twelve signals: power, wind speed, rotor speed and pitch angle, each as mean, minimum and maximum.
    """
    return core.Catalog(
        tuple(
            core.Signal(name=f"{name}{suffix}", quantity=quantity, statistic=statistic, unit=unit)
            for name, quantity, unit in _QUANTITIES
            for suffix, statistic in _STATISTICS
        )
    )
