"""Rename columns to catalog names (ADR-11)."""

import dataclasses
from collections.abc import Mapping

from phoibe.dokimasia import core


@dataclasses.dataclass(frozen=True, slots=True)
class RenameToCatalog:
    """Rename source columns to catalog names, so that later layers address signals directly.

    It refuses where a rename would collide with another column or with another rename, and it refuses a source
    column that is not in the data. It never picks one of several candidates.
    """

    mapping: Mapping[str, str]
    """Source column name to catalog name."""

    name = "rename_to_catalog"

    def apply(self, dataset: core.Dataset) -> core.Transformed | core.TransformRefused:
        """Rename the columns.

        Parameters
        ----------
        dataset
            The data.

        Returns
        -------
        Transformed | TransformRefused
            The data with renamed columns, or a refusal that names every problem.
        """
        present = list(dataset.signal_names)
        problems = [f"column '{source}' is not in the data" for source in self.mapping if source not in present]
        targets = list(self.mapping.values())
        problems.extend(
            f"'{target}' is the target of several renames"
            for target in sorted({t for t in targets if targets.count(t) > 1})
        )
        kept = [name for name in present if name not in self.mapping]
        problems.extend(
            f"'{target}' is already the name of another column" for target in sorted(set(targets) & set(kept))
        )
        duplicated = sorted({name for name in present if present.count(name) > 1} & set(self.mapping))
        problems.extend(f"column '{name}' appears several times" for name in duplicated)
        if problems:
            return core.TransformRefused("; ".join(problems))
        frame = dataset.frame.rename(columns=dict(self.mapping))
        return core.Transformed(frame, {"mapping": dict(self.mapping)})
