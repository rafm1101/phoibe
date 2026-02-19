import pathlib
import typing

import pandas as pd

from .entities import FileMetadata
from .entities import LayerReport


class DataLoader(typing.Protocol):
    """Protocil for loading data from various sources, and retrieving a dataframe."""

    def load(self, file_path: str | pathlib.Path) -> pd.DataFrame:
        """Load data from `file_path` and return as a dataframe.

        Parameters
        ----------
        file_path
            Path to the requested file.

        Returns
        -------
        pd.DataFrame
            Dataframe holding loaded data.

        Raises
        ------
        ValueError
            If fileformat is unsupported.
        FileNotFoundError
            If file does not exist.
        """
        ...

    def get_metadata(self, file_path: str | pathlib.Path) -> FileMetadata:
        """Get information about the file.

        Parameters
        ----------
        file_path
            Path to the requested file.

        Returns
        -------
        FileMetadata
            Information about the source.
        """
        ...


class VariableDetector(typing.Protocol):
    """Protocol for detecting variable-to-column mappings in dataframes.

    Implementations must detect which dataframe column corresponds to expected variables.
    """

    def detect(self, df: pd.DataFrame) -> dict[str, str | None]:
        """Detect variable-to-column mapping.

        Parameters
        ----------
        df
            Dataframe to analyse.

        Returns
        -------
        dict
            Variable-to-column key mapping. Value is `None` if variable is not found.
        """
        ...


class ReportRepository(typing.Protocol):
    """Protocol for persisting validation reports.

    Implementations must provide methods to save and load reports.
    """

    def save(self, reported: list[LayerReport], output_path: str | pathlib.Path) -> None:
        """Save the reported results.

        Parameters
        ----------
        reported
            List of `LayerReport`s to save.
        output_path
            Path to the output file.

        Raises
        ------
        IOError
            If writing fails.
        """
        ...

    def load(self, input_path: str | pathlib.Path) -> list[LayerReport]:
        """Load a stored report.

        Parameters
        ----------
        input_path
            Path to the output file.

        Returns
        -------
        list[LayerReport]
            List of `LayerReport`s loaded.

        Raises
        ------
        ValueError
            If fileformat is unsupported.
        FileNotFoundError
            If file does not exist.
        """
        ...
