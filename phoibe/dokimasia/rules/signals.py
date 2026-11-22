"""Reading signals for measures that look at several of them at once."""

import pandas as pd

from phoibe.dokimasia import core


def read_numeric_signals(
    dataset: core.Dataset, names: tuple[str, ...], measure_id: str
) -> pd.DataFrame | core.NotEvaluable:
    """Read signals as numbers, in time order.

    Text that is a number is read as a number; anything that cannot be read becomes NaN. Rows without a valid timestamp
    are left out. Equal timestamps keep their order.

    Parameters
    ----------
    dataset
        The data.
    names
        Catalog names of the signals; each must be in the dataset exactly once.
    measure_id
        Identifier of the measure that asks, for the outcome.

    Returns
    -------
    pandas.DataFrame | NotEvaluable
        A frame of floats with one column per name, in the order of `names`, on the time axis; or why a name cannot be
        used: missing, or more than one column of that name.
    """
    absent = [name for name in names if name not in dataset.signal_names]
    if absent:
        return core.NotEvaluable(
            measure_id, core.NotEvaluableReason.MISSING_SIGNAL, f"signals not in the dataset: {', '.join(absent)}"
        )
    repeated = [name for name in names if dataset.signal_names.count(name) > 1]
    if repeated:
        return core.NotEvaluable(
            measure_id,
            core.NotEvaluableReason.MISSING_STRUCTURE,
            f"signals with duplicate columns: {', '.join(repeated)}",
        )
    positions = [dataset.signal_names.index(name) for name in names]
    numbers = dataset.frame.iloc[:, positions].apply(pd.to_numeric, errors="coerce").astype(float)
    numbers.index = dataset.time_axis
    ordered: pd.DataFrame = numbers[numbers.index.notna()].sort_index(kind="stable")
    return ordered
