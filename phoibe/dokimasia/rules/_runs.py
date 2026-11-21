"""Run-length helper shared by the generic measures."""

import numpy as np
import numpy.typing as npt


def find_runs(mask: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Find the runs of consecutive `True` values.

    Parameters
    ----------
    mask
        Boolean array.

    Returns
    -------
    list
        Half-open position intervals `(start, end)`, in order.

    """
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(end)) for start, end in zip(edges[::2], edges[1::2], strict=True)]
