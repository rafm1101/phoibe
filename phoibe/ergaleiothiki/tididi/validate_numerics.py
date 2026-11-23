import numpy as np
from numpy.typing import NDArray


def _validate_non_negative(array: NDArray[np.floating], prefix: str) -> None:
    """Assumption: `array` has been validated notna and finite."""
    if np.any(array < 0.0):
        raise ValueError(f"{prefix} contains negative values.")


def _validate_positive(array: NDArray[np.floating], prefix: str) -> None:
    """Assumption: `array` has been validated notna and finite."""
    if np.any(array <= 0.0):
        raise ValueError(f"{prefix} contains non-positive values.")


def _validate_notna_finite(array: NDArray[np.floating], prefix: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{prefix} contains NaN or infinite values.")


def _validate_broadcastable(left: NDArray, right: NDArray, prefix: str) -> None:
    try:
        np.broadcast(left, right)
    except ValueError as exception:
        raise ValueError(f"{prefix} are not broadcast-compatible: {left.shape} <> {right.shape}.") from exception
