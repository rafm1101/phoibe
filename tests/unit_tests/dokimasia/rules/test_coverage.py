"""The fault coverage table, executable (ADR-09, Phase 4 exit gate).

One row per class of faults that the generator injects, and one per fault that the tests inject themselves. A row with
measures is scored by rows, pooled over seeds and over a grid of incidence and level, and on clean data for false
alarms. A row without measures says that nothing detects the class yet. When a rule is added, its measures go into
`FAULT_CLASSES`, and the row turns from a gap into numbers.

Print the table with `PYTHONPATH=. python tests/unit_tests/dokimasia/rules/test_coverage.py` from the repository root.
"""

import dataclasses
import itertools
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

from phoibe.dokimasia import core, rules
from phoibe.synthetic_data import turbine

FREQ = pd.Timedelta("10min")
TIME = turbine.Time(start="2026-02-14T00:00:00", freq="10min", periods=1008)
SEEDS = range(4)
GRID = list(itertools.product([1, 3], [1, 2]))  # (incidence, level)
QUANTITIES = ("power", "wind_speed", "rotor_speed", "pitch_angle")


@dataclasses.dataclass(frozen=True)
class Injection:
    """Faulty data with the rows that are faulty."""

    messy: pd.DataFrame
    truth: npt.NDArray[np.bool_]
    grid: pd.DatetimeIndex
    """The rows that `truth` and the findings refer to."""


type Injector = Callable[[pd.DataFrame, float, float, int], Injection]


def _inject_with_pipeline(kind: str, column: str | None, *, create=turbine.create_default_messup_pipeline) -> Injector:
    """Inject with the generator and take the truth from its labels."""

    def inject(clean: pd.DataFrame, incidence: float, level: float, seed: int) -> Injection:
        messy, labels = create(turbine.MessUpPipeline(seed=seed), incidence, level).apply_labelled(clean)
        truth = turbine.messlabel_mask(labels, messy.index, kind=kind, column=column).to_numpy(copy=True)
        return Injection(messy, truth, messy.index)

    return inject


def _inject_dropped_rows(clean: pd.DataFrame, incidence: float, level: float, seed: int) -> Injection:
    """Inject with the generator. Dropped rows are gone, so truth and findings refer to the clean time axis."""
    messy, labels = turbine.create_default_messup_pipeline(
        turbine.MessUpPipeline(seed=seed), incidence, level
    ).apply_labelled(clean)
    truth = turbine.messlabel_mask(labels, clean.index, kind="RowDrop").to_numpy(copy=True)
    truth &= (clean.index >= messy.index[0]) & (clean.index <= messy.index[-1])  # a drop at an end leaves no trace
    return Injection(messy, truth, clean.index)


def _inject_joint_freeze(clean: pd.DataFrame, incidence: float, level: float, seed: int) -> Injection:
    """Hold the state of all signals at once, which the generator does not do: it freezes one column per step."""
    rng = np.random.default_rng(seed)
    messy = clean.copy()
    truth = np.zeros(len(clean), dtype=bool)
    for start, end in turbine.GeometricSegments(n=int(5 * incidence), p=0.5 / level).generate(len(clean), rng):
        if end - start >= 2:
            messy.iloc[start:end] = messy.iloc[start].to_numpy()
            truth[start:end] = True
    return Injection(messy, truth, messy.index)


def _inject_statistic_order(clean: pd.DataFrame, incidence: float, level: float, seed: int) -> Injection:
    """Inject with the extended pipeline. Truth is computed from the values, with no use of the measure."""
    pipeline = turbine.create_extended_messup_pipeline(turbine.MessUpPipeline(seed=seed), incidence, level)
    messy, _ = pipeline.apply_labelled(clean)
    truth = np.zeros(len(messy), dtype=bool)
    for name in QUANTITIES:
        low, mid, high = (messy[column].to_numpy(dtype=float) for column in (f"{name}_min", name, f"{name}_max"))
        with np.errstate(invalid="ignore"):
            truth |= (mid < low) | (mid > high)
    return Injection(messy, truth, messy.index)


@dataclasses.dataclass(frozen=True)
class FaultClass:
    """A class of faults and, if there are any, the measures that are meant to detect it."""

    name: str
    note: str
    kind: str | None = None
    """Label kind of the generator step, `None` for a fault that the tests inject themselves."""
    measures: Callable[[], list[core.Measure]] | None = None
    """Builds the measures that detect it; `None` while nothing does."""
    inject: Injector | None = None
    source: str = "basic"
    """`basic` for the four mean signals, `extended` for the mean, minimum and maximum of each."""


FAULT_CLASSES = [
    FaultClass(
        "RowDrop",
        "dropped rows",
        kind="RowDrop",
        measures=lambda: [rules.GapMeasure(expected_step=FREQ)],
        inject=_inject_dropped_rows,
    ),
    FaultClass(
        "DeleteEntries (pitch_angle)",
        "NaN blocks",
        kind="DeleteEntries",
        measures=lambda: [rules.MissingRunsMeasure(signals=("pitch_angle",))],
        inject=_inject_with_pipeline("DeleteEntries", "pitch_angle"),
    ),
    FaultClass(
        "Joint freeze (all signals)",
        "the state of all signals held for 2 or more steps",
        measures=lambda: [rules.FrozenRunsMeasure(signals=QUANTITIES, min_steps=2)],
        inject=_inject_joint_freeze,
    ),
    FaultClass(
        "Mean outside min and max",
        "a mean above its maximum or below its minimum",
        measures=lambda: [rules.StatisticOrderMeasure(f"{name}_min", name, f"{name}_max") for name in QUANTITIES],
        inject=_inject_statistic_order,
        source="extended",
    ),
    FaultClass("Freeze (power)", "one signal held; a stopped turbine does that too", kind="Freeze"),
    FaultClass("Freeze (rotor_speed)", "one signal held; a stopped turbine does that too", kind="Freeze"),
    FaultClass("Spike (power)", "spikes; waits for the domain experts", kind="Spike"),
    FaultClass("Spike (wind_speed)", "spikes; waits for the domain experts", kind="Spike"),
    FaultClass("CurtailmentNight (power)", "night power limit", kind="CurtailmentNight"),
    FaultClass("CurtailmentToZero", "whole-turbine zeroing; waits for the users", kind="CurtailmentToZero"),
]


def _create_clean(source: str, seed: int) -> pd.DataFrame:
    latent_freq = "5min" if source == "extended" else None
    return turbine.generate_wtg_scada(time=TIME, seed=seed, latent_freq=latent_freq)


def _find(measures: list[core.Measure], frame: pd.DataFrame) -> tuple[core.Finding, ...]:
    findings: list[core.Finding] = []
    for measure in measures:
        outcome = core.run_measure(measure, core.Dataset.from_frame(frame))
        assert isinstance(outcome, core.Measured), outcome
        findings.extend(outcome.findings)
    return tuple(findings)


def build_coverage_table() -> pd.DataFrame:
    """Score every covered fault class over seeds and the incidence-level grid, and on clean data.

    Returns
    -------
    pandas.DataFrame
        One row per fault class with `covered`, `recall`, `precision`, the pooled row counts, and
        `false_alarms_clean`, the number of rows flagged on clean data. A class without measures has no numbers.
    """
    rows = []
    for fault in FAULT_CLASSES:
        row: dict[str, object] = {"fault": fault.name, "note": fault.note, "covered": fault.measures is not None}
        if fault.measures is not None and fault.inject is not None:
            pooled = core.Score(0, 0, 0)
            false_alarms_clean = 0
            for seed in SEEDS:
                clean = _create_clean(fault.source, seed)
                no_truth = np.zeros(len(clean), dtype=bool)
                false_alarms_clean += core.score_findings(
                    _find(fault.measures(), clean), no_truth, clean.index
                ).n_flagged
                for incidence, level in GRID:
                    injection = fault.inject(clean, incidence, level, seed)
                    findings = _find(fault.measures(), injection.messy)
                    pooled += core.score_findings(findings, injection.truth, injection.grid)
            row |= {
                "recall": pooled.recall,
                "precision": pooled.precision,
                "n_truth": pooled.n_truth,
                "n_flagged": pooled.n_flagged,
                "false_alarms_clean": false_alarms_clean,
            }
        rows.append(row)
    return pd.DataFrame(rows)


COVERED = [fault for fault in FAULT_CLASSES if fault.measures is not None]


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return build_coverage_table()


def test_the_table_has_a_row_for_every_fault_class_that_the_generator_injects():
    injected = set()
    for source, create in (
        ("basic", turbine.create_default_messup_pipeline),
        ("extended", turbine.create_extended_messup_pipeline),
    ):
        _, labels = create(turbine.MessUpPipeline(seed=0), 3, 1).apply_labelled(_create_clean(source, 0))
        injected |= {label.kind for label in labels}

    assert injected <= {fault.kind for fault in FAULT_CLASSES}


def test_every_row_says_whether_it_is_covered(table):
    assert table["covered"].tolist() == [fault.measures is not None for fault in FAULT_CLASSES]
    assert table.loc[~table["covered"], ["recall", "precision"]].isna().all(axis=None)


@pytest.mark.parametrize("fault", COVERED, ids=lambda fault: fault.name)
def test_a_covered_class_has_faulty_rows_to_find_and_is_found_exactly(table, fault):
    row = table.set_index("fault").loc[fault.name]

    assert row["n_truth"] > 0, "the grid must inject this fault"
    assert row["recall"] == 1.0
    assert row["precision"] == 1.0


@pytest.mark.parametrize("fault", COVERED, ids=lambda fault: fault.name)
def test_a_covered_class_raises_no_false_alarm_on_clean_data(table, fault):
    assert table.set_index("fault").loc[fault.name, "false_alarms_clean"] == 0


def test_the_uncovered_classes_are_listed_until_a_rule_covers_them(table):
    assert table.loc[~table["covered"], "fault"].tolist() == [
        "Freeze (power)",
        "Freeze (rotor_speed)",
        "Spike (power)",
        "Spike (wind_speed)",
        "CurtailmentNight (power)",
        "CurtailmentToZero",
    ]


if __name__ == "__main__":
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(build_coverage_table().round(3).to_string(index=False))
