"""The draft contract of `contracts/wind-scada-10min.yaml` against generator data (Phase 4 exit gate).

Clean data must pass without a single verdict short of good. Faults that the generator injects must be seen by the
check that is meant to see them, and the messed-up data of the notebook (incidence 7, level 5) must not pass.
"""

import pathlib

import pandas as pd
import pytest

from phoibe import dokimasia
from phoibe.dokimasia import compose, contract, core
from phoibe.synthetic_data import turbine

ROOT = pathlib.Path(__file__).resolve().parents[3]
CONTRACT_FILE = ROOT / "contracts" / "wind-scada-10min.yaml"
SUBJECT = core.Subject("WEA 01", "wind_turbine", {"rated_power_kw": 5600, "rated_wind_speed_mps": 13.0})
TIME = turbine.Time(start="2026-02-14T00:00:00", freq="10min", periods=1008)
SEEDS = range(4)


@pytest.fixture(scope="module")
def draft() -> contract.Contract:
    return dokimasia.load_contract(CONTRACT_FILE)


def _create_clean(seed: int, *, extended: bool = True) -> pd.DataFrame:
    return turbine.generate_wtg_scada(time=TIME, seed=seed, latent_freq="5min" if extended else None)


def _to_bronze(frame: pd.DataFrame, *, utc: bool = True) -> pd.DataFrame:
    """Stand in for the standardizing: catalog names, and local (Berlin) generator times as UTC."""
    bronze = frame.copy()
    bronze.columns = [f"power_kw{name.removeprefix('power')}" if name.startswith("power") else name for name in frame]
    if utc:
        bronze.index = bronze.index.tz_localize("Europe/Berlin").tz_convert("UTC")
    return bronze


def _judge(draft: contract.Contract, frame: pd.DataFrame, *, utc: bool = True) -> core.GateResult:
    dataset = core.Dataset.from_frame(_to_bronze(frame, utc=utc))
    return compose.validate_layer(dataset, SUBJECT, draft, "bronze").result


def _verdicts(result: core.GateResult) -> dict[str, core.Verdict]:
    return {verdict.check_id: verdict.verdict for verdict in result.verdicts}


class TestTheDraft:
    def test_every_layer_uses_rules_and_metrics_that_exist(self, draft):
        registry = compose.create_default_registry()

        for layer in draft.layers:
            compose.validate_layer_rules(layer, registry)

    def test_names_the_layers_of_the_pipeline(self, draft):
        assert [layer.name for layer in draft.layers] == ["raw", "bronze"]

    def test_names_no_clause_because_none_is_known(self, draft):
        checks = draft.get_layer("bronze").checks

        assert all(check.clause is None for check in checks)


class TestCleanData:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_passes_with_every_check_good(self, draft, seed):
        result = _judge(draft, _create_clean(seed))

        assert result.outcome == core.GateOutcome.PASS
        assert {check: verdict for check, verdict in _verdicts(result).items() if verdict != core.Verdict.GOOD} == {}


class TestEachFaultIsSeenByItsCheck:
    """A fault is injected alone; the check that is meant to see it must not say good."""

    def _messy(self, seed: int, *steps) -> pd.DataFrame:
        pipeline = turbine.MessUpPipeline(seed=seed)
        for step in steps:
            pipeline.add(step)
        return pipeline.apply_labelled(_create_clean(seed))[0]

    def _cap_at_night(self, limit: float = 3200) -> list:
        return [turbine.CurtailmentNight(limit=limit, column=name) for name in ("power", "power_min", "power_max")]

    def test_dropped_rows_are_seen_by_availability(self, draft):
        step = turbine.RowDrop(segments=turbine.UniformSegments(n=5, min_len=5, max_len=11))

        result = _judge(draft, self._messy(0, step))

        assert _verdicts(result)["availability"] != core.Verdict.GOOD
        assert result.outcome != core.GateOutcome.PASS

    def test_deleted_pitch_angles_are_seen_by_the_completeness_of_the_pitch_angle(self, draft):
        step = turbine.DeleteEntries(
            segments=turbine.UniformSegments(n=40, min_len=1, max_len=3),
            columns=["pitch_angle", "pitch_angle_min", "pitch_angle_max"],
        )

        verdicts = _verdicts(_judge(draft, self._messy(0, step)))

        assert verdicts["pitch_angle_complete"] != core.Verdict.GOOD
        assert verdicts["power_complete"] == core.Verdict.GOOD

    def test_a_night_limit_is_seen_by_the_night_check_and_the_curtailed_share(self, draft):
        verdicts = _verdicts(_judge(draft, self._messy(0, *self._cap_at_night())))

        assert verdicts["no_night_limit"] == core.Verdict.BAD
        assert verdicts["share_of_curtailed_rows"] != core.Verdict.GOOD

    def test_a_cap_of_the_mean_alone_puts_it_out_of_order_with_the_unchanged_minimum(self, draft):
        verdicts = _verdicts(_judge(draft, self._messy(0, turbine.CurtailmentNight(limit=3200, column="power"))))

        assert verdicts["power_order"] == core.Verdict.BAD

    def test_a_night_limit_alone_does_not_stop_the_gate(self, draft):
        result = _judge(draft, self._messy(0, *self._cap_at_night()))

        assert result.outcome == core.GateOutcome.PASS

    def test_a_mean_above_its_maximum_is_seen_by_the_order_check(self, draft):
        frame = _create_clean(0)
        frame.iloc[100:110, frame.columns.get_loc("power_max")] = -1.0
        frame.iloc[100:110, frame.columns.get_loc("power_min")] = -1.0

        result = _judge(draft, frame)

        assert _verdicts(result)["power_order"] == core.Verdict.BAD

    def test_a_state_held_by_all_signals_is_seen_by_the_frozen_check(self, draft):
        frame = _create_clean(0)
        frame.iloc[200:210] = frame.iloc[200].to_numpy()

        verdicts = _verdicts(_judge(draft, frame))

        assert verdicts["no_frozen_state"] == core.Verdict.BAD

    def test_a_time_axis_without_zone_is_seen_by_the_zone_check(self, draft):
        verdicts = _verdicts(_judge(draft, _create_clean(0), utc=False))

        assert verdicts["time_axis_in_utc"] == core.Verdict.BAD

    def test_a_constant_wind_speed_is_seen_by_the_spread_check(self, draft):
        frame = _create_clean(0)
        frame["wind_speed"] = 8.0
        frame["wind_speed_min"] = 8.0
        frame["wind_speed_max"] = 8.0

        verdicts = _verdicts(_judge(draft, frame))

        assert verdicts["wind_speed_not_constant"] == core.Verdict.BAD


class TestMessedUpData:
    """The notebook run (incidence 7, level 5) passed the legacy bronze gate with 30 of 30 points."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_extended_pipeline_fails_the_gate_by_measurement(self, draft, seed):
        messy, _ = turbine.create_extended_messup_pipeline(turbine.MessUpPipeline(seed=seed), 7, 5).apply_labelled(
            _create_clean(seed)
        )

        result = _judge(draft, messy)
        verdicts = _verdicts(result)

        assert result.outcome == core.GateOutcome.FAIL
        assert verdicts["availability"] == core.Verdict.BAD
        assert verdicts["power_order"] == core.Verdict.BAD
        assert core.Verdict.ERROR not in verdicts.values()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_the_default_pipeline_fails_the_gate_by_measurement(self, draft, seed):
        messy, _ = turbine.create_default_messup_pipeline(turbine.MessUpPipeline(seed=seed), 7, 5).apply_labelled(
            _create_clean(seed, extended=False)
        )

        result = _judge(draft, messy)
        verdicts = _verdicts(result)

        assert result.outcome == core.GateOutcome.FAIL
        assert verdicts["availability"] == core.Verdict.BAD
        assert verdicts["pitch_angle_complete"] == core.Verdict.BAD

    def test_data_without_minimum_and_maximum_cannot_pass_the_order_checks(self, draft):
        result = _judge(draft, _create_clean(0, extended=False))

        verdicts = _verdicts(result)
        assert verdicts["power_order"] == core.Verdict.NOT_EVALUABLE
        assert result.outcome == core.GateOutcome.FAIL
