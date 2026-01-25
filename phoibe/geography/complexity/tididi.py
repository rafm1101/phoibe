import numpy as np
from numpy.typing import NDArray


def _validate_positive(array: NDArray[np.floating], prefix: str) -> None:
    """Assumption: `array` has been validated notna and finite."""
    if np.any(array <= 0.0):
        raise ValueError(f"{prefix} contains non-positive values.")
