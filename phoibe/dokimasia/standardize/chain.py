"""Apply several transforms in order."""

import dataclasses

from phoibe.dokimasia import core


@dataclasses.dataclass(frozen=True, slots=True)
class Chain:
    """Apply transforms one after the other; the first refusal stops the chain."""

    steps: tuple[core.Transform, ...]
    """The transforms, in order."""

    name = "chain"

    def __post_init__(self) -> None:
        """Require at least one step."""
        if not self.steps:
            raise ValueError("a chain needs at least one transform")

    def apply(self, dataset: core.Dataset) -> core.Transformed | core.TransformRefused:
        """Apply all steps.

        Parameters
        ----------
        dataset
            The data.

        Returns
        -------
        Transformed | TransformRefused
            The data after the last step, with the parameters and rules of every step recorded, or the first refusal
            with the name of the step.
        """
        current = dataset
        steps: list[dict[str, object]] = []
        rules: list[str] = []
        for transform in self.steps:
            outcome = transform.apply(current)
            if isinstance(outcome, core.TransformRefused):
                return core.TransformRefused(f"{transform.name}: {outcome.reason}")
            steps.append({"name": transform.name, "parameters": dict(outcome.parameters)})
            if outcome.rule is not None:
                rules.append(f"{transform.name}: {outcome.rule}")
            current = core.Dataset.from_frame(outcome.frame)
        return core.Transformed(current.frame, {"steps": steps}, "; ".join(rules) or None)
