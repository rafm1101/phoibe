"""Transforms between layers and the move to the next layer (ADR-06, ADR-11)."""

import dataclasses
import typing
from collections.abc import Mapping

import pandas as pd

from phoibe.dokimasia.core import attestation as attestation_module
from phoibe.dokimasia.core import dataset as dataset_module


@dataclasses.dataclass(frozen=True, slots=True)
class Transformed:
    """The result of a transform that could be applied."""

    frame: pd.DataFrame
    """The next version of the data, indexed by time."""
    parameters: Mapping[str, object]
    """The parameters the transform ran with."""
    rule: str | None = None
    """A rule applied where the data left a choice, such as how a repeated hour was resolved."""


@dataclasses.dataclass(frozen=True, slots=True)
class TransformRefused:
    """The result of a transform that did not apply, because the data left a choice that it will not guess."""

    reason: str
    """Why, for a person."""


class TransformRefusedError(Exception):
    """Raised when a layer cannot be advanced because the transform refused."""

    def __init__(self, transform: str, reason: str) -> None:
        """Create the error."""
        self.transform = transform
        self.reason = reason
        super().__init__(f"transform '{transform}' refused: {reason}")


class Transform(typing.Protocol):
    """A deterministic, parameterized function from one dataset version to the next (ADR-11).

    It changes the form of the data, not its content. Cleaning and imputation are not transforms.
    """

    @property
    def name(self) -> str:
        """Name recorded in the provenance."""
        ...

    def apply(self, dataset: dataset_module.Dataset) -> Transformed | TransformRefused:
        """Apply the transform, or refuse where it would have to guess."""
        ...


def advance_dataset(
    dataset: dataset_module.Dataset,
    attestation: attestation_module.Attestation,
    transform: Transform | None = None,
    *,
    software_version: str,
) -> dataset_module.Dataset:
    """Move a dataset to the next layer: apply an optional transform and record where the data came from.

    Parameters
    ----------
    dataset
        The dataset that passed the gate.
    attestation
        The attestation of that gate. Its fingerprint must be the fingerprint of `dataset`.
    transform
        The transform to apply on the way, or `None` to hand the data on unchanged.
    software_version
        Version of the software, recorded in the description of the transform.

    Returns
    -------
    Dataset
        The data of the next layer, with a provenance record: the parent attestation, the transform that was applied
        and the output fingerprint.

    Raises
    ------
    ValueError
        If the attestation is not the attestation of this version of the data.
    TransformRefusedError
        If the transform refused; nothing is guessed.
    """
    if attestation.fingerprint != dataset.fingerprint:
        raise ValueError("the attestation belongs to another version of the data than the dataset given")
    description = None
    frame = dataset.frame
    if transform is not None:
        outcome = transform.apply(dataset)
        if isinstance(outcome, TransformRefused):
            raise TransformRefusedError(transform.name, outcome.reason)
        frame = outcome.frame
        description = dataset_module.TransformDescription(
            transform.name, outcome.parameters, software_version, outcome.rule
        )
    advanced = dataset_module.Dataset.from_frame(frame)
    record = dataset_module.ProvenanceRecord(attestation.id, description, advanced.fingerprint)
    return dataclasses.replace(advanced, provenance=record)
