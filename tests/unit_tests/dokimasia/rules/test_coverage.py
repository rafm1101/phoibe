"""The fault coverage table, executable (ADR-09, Phase 4 exit gate).

One row per fault class that the generator injects. A row with a detector is scored by rows, pooled over seeds and
over a grid of incidence and level, and on clean data for false alarms. A row without a detector says that nothing
detects the class yet. When a rule is added, its detector goes into `DETECTORS`, and the row turns from a gap into
numbers.

Print the table with `PYTHONPATH=. python tests/unit_tests/dokimasia/rules/test_coverage.py` from the repository root.
"""

import dataclasses
import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import core, rules
from phoibe.synthetic_data import turbine

FREQ = pd.Timedelta("10min")
TIME = turbine.Time(start="2026-02-14T00:00:00", freq="10min", periods=1008)
SEEDS = range(4)
GRID = list(itertools.product([1, 3], [1, 2]))  # (incidence, level)


@dataclasses.dataclass(frozen=True)
class FaultClass:
    """A class of injected faults and, if there is one, the detector for it."""

    kind: str | None
    """Label kind of the generator step; `None` for a fault that has no step of its own."""
    column: str | None
    """Column the step addresses, `None` if it addresses several or none."""
    note: str
    """What the fault is, for the table."""
    measure: Callable[[], core.Measure] | None = None
    """Builds the measure that is meant to detect it; `None` while nothing does."""
    grid: str = "messy"
    """`clean` if the faulty rows are no longer in the data (dropped rows), else `messy`."""

    @property
    def name(self) -> str:
        return f"{self.kind or 'statistic order'}{'' if self.column is None else f' ({self.column})'}"


DETECTORS = [
    FaultClass("RowDrop", None, "dropped rows", lambda: rules.GapMeasure(expected_step=FREQ), grid="clean"),
    FaultClass(
        "DeleteEntries", "pitch_angle", "NaN blocks", lambda: rules.MissingRunsMeasure(signals=("pitch_angle",))
    ),
    FaultClass("Freeze", "power", "frozen values"),
    FaultClass("Freeze", "rotor_speed", "frozen values"),
    FaultClass("Spike", "power", "spikes"),
    FaultClass("Spike", "wind_speed", "spikes"),
    FaultClass("CurtailmentNight", "power", "night power limit"),
    FaultClass("CurtailmentToZero", None, "whole-turbine zeroing"),
    FaultClass(None, None, "mean outside its min and max (extended pipeline)"),
]


def _measure_findings(measure: core.Measure, frame: pd.DataFrame) -> tuple[core.Finding, ...]:
    outcome = core.run_measure(measure, core.Dataset.from_frame(frame))
    assert isinstance(outcome, core.Measured), outcome
    return outcome.findings


def _score_on_faulty_data(
    fault: FaultClass, clean: pd.DataFrame, incidence: float, level: float, seed: int
) -> core.Score:
    assert fault.measure is not None
    pipeline = turbine.create_default_messup_pipeline(turbine.MessUpPipeline(seed=seed), incidence, level)
    messy, labels = pipeline.apply_labelled(clean)
    grid = clean.index if fault.grid == "clean" else messy.index
    truth = turbine.messlabel_mask(labels, grid, kind=fault.kind, column=fault.column).to_numpy(copy=True)
    if fault.grid == "clean":  # a dropped row at either end of the series leaves no trace
        truth &= (grid >= messy.index[0]) & (grid <= messy.index[-1])
    return core.score_findings(_measure_findings(fault.measure(), messy), truth, grid)


def build_coverage_table() -> pd.DataFrame:
    """Score every detector over seeds and the incidence-level grid, and on clean data.

    Returns
    -------
    pandas.DataFrame
        One row per fault class with `recall`, `precision`, the pooled row counts, and `false_alarms_clean`, the
        number of rows flagged on clean data. A class without a detector has `covered` false and no numbers.
    """
    rows = []
    for fault in DETECTORS:
        row: dict[str, object] = {"fault": fault.name, "note": fault.note, "covered": fault.measure is not None}
        if fault.measure is not None:
            pooled = core.Score(0, 0, 0)
            false_alarms_clean = 0
            for seed in SEEDS:
                clean = turbine.generate_wtg_scada(time=TIME, seed=seed)
                false_alarms_clean += core.score_findings(
                    _measure_findings(fault.measure(), clean), np.zeros(len(clean), dtype=bool), clean.index
                ).n_flagged
                for incidence, level in GRID:
                    pooled += _score_on_faulty_data(fault, clean, incidence, level, seed)
            row |= {
                "recall": pooled.recall,
                "precision": pooled.precision,
                "n_truth": pooled.n_truth,
                "n_flagged": pooled.n_flagged,
                "false_alarms_clean": false_alarms_clean,
            }
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return build_coverage_table()


def test_the_table_has_a_row_for_every_fault_class_that_the_generator_injects():
    basic = turbine.generate_wtg_scada(time=TIME, seed=0)
    extended = turbine.generate_wtg_scada(time=TIME, seed=0, latent_freq="5min")
    injected = set()
    for create, frame in (
        (turbine.create_default_messup_pipeline, basic),
        (turbine.create_extended_messup_pipeline, extended),
    ):
        _, labels = create(turbine.MessUpPipeline(seed=0), 3, 1).apply_labelled(frame)
        injected |= {label.kind for label in labels}

    assert injected <= {fault.kind for fault in DETECTORS}


def test_every_row_says_whether_it_is_covered(table):
    assert table["covered"].tolist() == [fault.measure is not None for fault in DETECTORS]
    assert not table.loc[~table["covered"], "recall"].notna().any() if "recall" in table else True


@pytest.mark.parametrize("fault", [f for f in DETECTORS if f.measure is not None], ids=lambda fault: fault.name)
def test_a_covered_class_has_faulty_rows_to_find_and_is_found_exactly(table, fault):
    row = table.set_index("fault").loc[fault.name]

    assert row["n_truth"] > 0, "the grid must inject this fault"
    assert row["recall"] == 1.0
    assert row["precision"] == 1.0


@pytest.mark.parametrize("fault", [f for f in DETECTORS if f.measure is not None], ids=lambda fault: fault.name)
def test_a_covered_class_raises_no_false_alarm_on_clean_data(table, fault):
    assert table.set_index("fault").loc[fault.name, "false_alarms_clean"] == 0


def test_the_uncovered_classes_are_listed_until_a_rule_covers_them(table):
    uncovered = table.loc[~table["covered"], "fault"].tolist()

    assert uncovered == [
        "Freeze (power)",
        "Freeze (rotor_speed)",
        "Spike (power)",
        "Spike (wind_speed)",
        "CurtailmentNight (power)",
        "CurtailmentToZero",
        "statistic order",
    ]


if __name__ == "__main__":
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(build_coverage_table().round(3).to_string(index=False))
