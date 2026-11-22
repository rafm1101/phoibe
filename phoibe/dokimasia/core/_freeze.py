"""Deep immutability for free-form data (a subject specification, transform parameters)."""

import types
from collections.abc import Mapping


def freeze(value: object) -> object:
    """Freeze free-form data: sequences become tuples and mappings become read-only views, recursively.

    Parameters
    ----------
    value
        Any value. Text, numbers and other scalars are returned as they are.

    Returns
    -------
    object
        The frozen value. A frozen value equals the original where the original was a mapping, and equals a tuple
        in place of a list, which is what makes a file round trip exact.
    """
    if isinstance(value, Mapping):
        return types.MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(freeze(item) for item in value)
    return value
