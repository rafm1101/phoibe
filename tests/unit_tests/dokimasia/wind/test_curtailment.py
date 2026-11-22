import numpy as np
import pandas as pd
import pytest

from phoibe.dokimasia import compose, contract, core, wind
from phoibe.layered.application.context import ValidationContext
from phoibe.layered.core.entities import Status
from phoibe.layered.rules.rules_power import CurtailmentRule
from phoibe.synthetic_data import turbine

RATED_KW = 5600.0
RATED_WIND = 13.0
STEPS = 288  # two days of 10 minutes
GRID = pd.date_range("2026-03-01", periods=STEPS, freq="10min", name="datetime")


def _dataset(power, wind_speed, index=GRID) -> core.Dataset:
    return core.Dataset.from_frame(pd.DataFrame({"power_kw": power, "wind_speed": wind_speed}, index=index))


def _metrics(outcome: core.Measured) -> dict[str, float]:
    return {metric.key: metric.value for metric in outcome.metrics}


def _windy() -> np.ndarray:
    return np.full(STEPS, 15.0)


class TestPowerLevelsMeasure:
    def _measure(self, **kwargs) -> wind.PowerLevelsMeasure:
        return wind.PowerLevelsMeasure("power_kw", "wind_speed", RATED_KW, RATED_WIND, **kwargs)

    def test_the_rated_level_alone_is_no_curtailment(self):
        outcome = self._measure().measure(_dataset(np.full(STEPS, RATED_KW), _windy()))

        assert isinstance(outcome, core.Measured)
        assert outcome.findings == ()
        assert _metrics(outcome) == {
            "n_high_wind_rows": 288.0,
            "n_levels": 1.0,
            "n_curtailed_levels": 0.0,
            "highest_level_kw": RATED_KW,
            "n_curtailed_rows": 0.0,
            "share_curtailed_rows": 0.0,
        }

    def test_finds_a_curtailed_level_and_locates_its_runs(self):
        power = np.full(STEPS, RATED_KW)
        power[20:50] = 3200.0
        power[100:110] = 3200.0

        outcome = self._measure().measure(_dataset(power, _windy()))

        assert [(f.kind, f.signals, f.start, f.end, f.n_steps) for f in outcome.findings] == [
            ("curtailment_level", ("power_kw",), GRID[20], GRID[49], 30),
            ("curtailment_level", ("power_kw",), GRID[100], GRID[109], 10),
        ]
        metrics = _metrics(outcome)
        assert (metrics["n_levels"], metrics["n_curtailed_levels"], metrics["n_curtailed_rows"]) == (2.0, 1.0, 40.0)
        assert metrics["highest_level_kw"] == RATED_KW

    def test_a_power_that_scatters_is_no_level(self):
        power = RATED_KW - np.arange(STEPS) * 7.0  # no value repeats at the resolution of 10 kW often enough

        outcome = self._measure(min_count=10).measure(_dataset(power, _windy()))

        assert outcome.findings == ()
        assert _metrics(outcome)["n_levels"] == 0.0
        assert _metrics(outcome)["highest_level_kw"] == 0.0

    def test_values_within_the_resolution_share_a_level(self):
        power = np.full(STEPS, RATED_KW)
        power[:40] = 3200.0 + np.tile([0.0, 4.0, -4.0, 2.0], 10)

        outcome = self._measure().measure(_dataset(power, _windy()))

        assert [(f.start, f.n_steps) for f in outcome.findings] == [(GRID[0], 40)]

    def test_leaves_out_low_wind_where_low_power_is_normal(self):
        wind_speed = np.r_[np.full(100, 5.0), np.full(STEPS - 100, 15.0)]
        power = np.r_[np.zeros(100), np.full(STEPS - 100, RATED_KW)]

        outcome = self._measure().measure(_dataset(power, wind_speed))

        assert outcome.findings == ()
        assert _metrics(outcome)["n_high_wind_rows"] == 188.0

    def test_a_level_at_the_rated_power_within_the_resolution_is_not_curtailed(self):
        outcome = self._measure().measure(_dataset(np.full(STEPS, RATED_KW - 10.0), _windy()))

        assert outcome.findings == ()

    def test_a_stop_at_high_wind_is_a_curtailed_level_of_zero(self):
        power = np.full(STEPS, RATED_KW)
        power[30:60] = 0.0

        outcome = self._measure().measure(_dataset(power, _windy()))

        assert [(f.start, f.n_steps) for f in outcome.findings] == [(GRID[30], 30)]

    def test_too_few_rows_at_high_wind_are_not_evaluable(self):
        outcome = self._measure().measure(_dataset(np.full(STEPS, RATED_KW), np.full(STEPS, 5.0)))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason is core.NotEvaluableReason.INSUFFICIENT_DATA

    def test_names_a_missing_signal(self):
        frame = pd.DataFrame({"power_kw": np.zeros(STEPS)}, index=GRID)

        outcome = self._measure().measure(core.Dataset.from_frame(frame))

        assert outcome == core.NotEvaluable(
            "power_levels", core.NotEvaluableReason.MISSING_SIGNAL, "signals not in the dataset: wind_speed"
        )

    @pytest.mark.parametrize(
        "kwargs", [{"resolution_kw": 0.0}, {"resolution_kw": -1.0}, {"min_count": 0}, {"min_rows": 0}]
    )
    def test_rejects_an_unusable_definition(self, kwargs):
        with pytest.raises(ValueError, match=r"resolution_kw|min_count"):
            self._measure(**kwargs)


class TestPowerLevelsMeasureGivenKde:
    def _measure(self, **kwargs) -> wind.PowerLevelsMeasure:
        return wind.PowerLevelsMeasure(
            "power_kw", "wind_speed", RATED_KW, RATED_WIND, method=wind.LevelMethod.KDE, **kwargs
        )

    def test_finds_the_levels_as_peaks_of_the_density(self):
        power = np.full(STEPS, RATED_KW)
        power[20:80] = 3200.0

        outcome = self._measure().measure(_dataset(power, _windy()))

        assert [(f.start, f.n_steps) for f in outcome.findings] == [(GRID[20], 60)]
        assert _metrics(outcome)["n_levels"] == 2.0
        assert _metrics(outcome)["highest_level_kw"] == RATED_KW

    def test_a_single_level_is_found_where_the_legacy_density_fails(self):
        outcome = self._measure().measure(_dataset(np.full(STEPS, RATED_KW), _windy()))

        assert _metrics(outcome)["n_levels"] == 1.0
        assert outcome.findings == ()

    def test_a_noisy_plateau_is_one_level(self):
        rng = np.random.default_rng(1)
        power = np.full(STEPS, RATED_KW) + rng.normal(0, 3, STEPS)
        power[:100] = 3200.0 + rng.normal(0, 3, 100)

        outcome = self._measure().measure(_dataset(power, _windy()))

        assert _metrics(outcome)["n_levels"] == 2.0
        first = outcome.findings[0]
        assert (first.start, first.n_steps) == (GRID[0], 100)

    def test_rejects_a_negative_prominence_threshold(self):
        with pytest.raises(ValueError, match="prominence_threshold"):
            self._measure(prominence_threshold=-1.0)


def _create_clean_with_levels(seed: int, *, night: bool, limit: float = 3200.0) -> pd.DataFrame:
    time = turbine.Time(start="2026-02-14T00:00:00", freq="10min", periods=1008)
    clean = turbine.generate_wtg_scada(time=time, seed=seed)
    if night:
        pipeline = turbine.MessUpPipeline(seed=seed)
        pipeline.add(turbine.CurtailmentNight(limit=limit, column="power"))
        return pipeline.apply_labelled(clean)[0]
    return clean


class TestPowerLevelsMeasureAgainstLayered:
    """The `kde` method finds the levels that the curtailment rule of `layered` finds, at the same threshold."""

    THRESHOLD = 14.0

    def _run_layered(self, frame: pd.DataFrame) -> dict:
        rule = CurtailmentRule(wind_speed_threshold=self.THRESHOLD, prominence_threshold=1e-7)
        context = ValidationContext(
            layer_name="test", detected_variables={"power_kw": "power", "wind_speed": "wind_speed"}, turbine_id="t"
        )
        result = rule.execute(frame.reset_index(drop=True), context)
        assert result.status == Status.PASSED
        return result.details

    def _run_kde(self, frame: pd.DataFrame) -> core.Measured:
        measure = wind.PowerLevelsMeasure(
            "power", "wind_speed", RATED_KW, self.THRESHOLD, method=wind.LevelMethod.KDE, min_rows=2
        )
        outcome = measure.measure(core.Dataset.from_frame(frame))
        assert isinstance(outcome, core.Measured)
        return outcome

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_finds_the_same_levels_on_data_with_a_night_limit(self, seed):
        frame = _create_clean_with_levels(seed, night=True)

        legacy = self._run_layered(frame)
        metrics = _metrics(self._run_kde(frame))

        assert metrics["n_levels"] == legacy["n_curtailments"]
        assert metrics["highest_level_kw"] == max(legacy["power"])
        assert metrics["n_curtailed_levels"] == sum(level < RATED_KW - 10.0 for level in legacy["power"])

    @pytest.mark.parametrize("limit", [2000.0, 4100.0])
    def test_finds_the_same_levels_for_other_limits(self, limit):
        frame = _create_clean_with_levels(0, night=True, limit=limit)

        legacy = self._run_layered(frame)
        metrics = _metrics(self._run_kde(frame))

        assert metrics["n_levels"] == legacy["n_curtailments"]
        assert metrics["highest_level_kw"] == max(legacy["power"])

    def test_both_find_one_level_on_clean_data_or_layered_fails_where_kde_does_not(self):
        frame = _create_clean_with_levels(1, night=False)

        metrics = _metrics(self._run_kde(frame))

        assert metrics["n_levels"] == 1.0
        try:
            legacy = self._run_layered(frame)
        except np.linalg.LinAlgError:
            return  # the known failure of `layered` for a single level
        assert legacy["n_curtailments"] == 1


class TestNightLimitMeasure:
    def _measure(self, **kwargs) -> wind.NightLimitMeasure:
        return wind.NightLimitMeasure("power_kw", "wind_speed", RATED_WIND, 22, 6, **kwargs)

    def _night(self, index=GRID) -> np.ndarray:
        return (index.hour >= 22) | (index.hour < 6)

    def test_the_same_level_day_and_night_is_no_limit(self):
        outcome = self._measure().measure(_dataset(np.full(STEPS, RATED_KW), _windy()))

        assert isinstance(outcome, core.Measured)
        assert outcome.findings == ()
        metrics = _metrics(outcome)
        assert (metrics["night_level_kw"], metrics["day_level_kw"], metrics["night_limit_depth_kw"]) == (
            RATED_KW,
            RATED_KW,
            0.0,
        )
        assert metrics["n_limited_rows"] == 0.0

    def test_finds_the_night_rows_on_the_limit_at_any_wind_speed(self):
        night = self._night()
        wind_speed = np.where(np.arange(STEPS) % 3 == 0, 8.0, 15.0)
        power = np.where(night, 3200.0, RATED_KW)

        outcome = self._measure().measure(_dataset(power, wind_speed))

        assert [(f.kind, f.start, f.end, f.n_steps, f.magnitude, f.magnitude_unit) for f in outcome.findings] == [
            ("night_limit", GRID[0], GRID[35], 36, 2400.0, "kW"),
            ("night_limit", GRID[132], GRID[179], 48, 2400.0, "kW"),
            ("night_limit", GRID[276], GRID[287], 12, 2400.0, "kW"),
        ]
        metrics = _metrics(outcome)
        assert (metrics["night_level_kw"], metrics["day_level_kw"], metrics["night_limit_depth_kw"]) == (
            3200.0,
            RATED_KW,
            2400.0,
        )
        assert metrics["share_night_at_level"] == 1.0
        assert metrics["n_limited_rows"] == float(night.sum())

    def test_leaves_unlimited_night_rows_alone(self):
        night = self._night()
        power = np.where(night, np.minimum(3200.0, np.linspace(0, 6000, STEPS)), RATED_KW)
        power[night & (np.arange(STEPS) % 2 == 0)] = 3200.0

        outcome = self._measure().measure(_dataset(power, _windy()))

        limited = core.build_finding_mask(outcome.findings, GRID)
        assert limited[night & (power == 3200.0)].all()
        assert not limited[~night].any()

    def test_a_night_that_wraps_midnight_and_one_that_does_not(self):
        power = np.where((GRID.hour >= 23) | (GRID.hour < 3), 3200.0, RATED_KW)

        wrapped = wind.NightLimitMeasure("power_kw", "wind_speed", RATED_WIND, 23, 3).measure(_dataset(power, _windy()))
        plain = wind.NightLimitMeasure("power_kw", "wind_speed", RATED_WIND, 3, 23).measure(_dataset(power, _windy()))

        assert [(f.start, f.n_steps) for f in wrapped.findings] == [(GRID[0], 18), (GRID[138], 24), (GRID[282], 6)]
        assert plain.findings == ()
        assert _metrics(plain)["night_level_kw"] == RATED_KW

    def test_a_night_level_that_few_rows_share_is_no_limit(self):
        night = self._night()
        rng = np.random.default_rng(0)
        power = np.where(night, rng.uniform(1000, 3000, STEPS).round(-1) + rng.integers(0, 3, STEPS) * 1000, RATED_KW)

        outcome = self._measure(min_share=0.9).measure(_dataset(power, _windy()))

        assert outcome.findings == ()
        assert _metrics(outcome)["n_limited_rows"] == 0.0

    def test_a_night_that_is_stronger_than_the_day_is_no_limit(self):
        power = np.where(self._night(), RATED_KW, 3200.0)

        outcome = self._measure().measure(_dataset(power, _windy()))

        assert outcome.findings == ()
        assert _metrics(outcome)["night_limit_depth_kw"] == -2400.0

    def test_not_enough_rows_at_high_wind_at_night_or_by_day_is_not_evaluable(self):
        wind_speed = np.where(self._night(), 5.0, 15.0)

        outcome = self._measure().measure(_dataset(np.full(STEPS, RATED_KW), wind_speed))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason is core.NotEvaluableReason.INSUFFICIENT_DATA
        assert "0 night rows" in outcome.detail

    def test_names_a_missing_signal(self):
        frame = pd.DataFrame({"wind_speed": np.zeros(STEPS)}, index=GRID)

        outcome = self._measure().measure(core.Dataset.from_frame(frame))

        assert isinstance(outcome, core.NotEvaluable)
        assert outcome.reason is core.NotEvaluableReason.MISSING_SIGNAL

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"night_start_hour": 24},
            {"night_end_hour": -1},
            {"night_start_hour": 6},
            {"resolution_kw": 0.0},
            {"min_rows": 0},
            {"min_share": 0.0},
            {"min_share": 1.5},
        ],
    )
    def test_rejects_an_unusable_definition(self, kwargs):
        arguments = {"night_start_hour": 22, "night_end_hour": 6, **kwargs}

        with pytest.raises(ValueError, match=r"night|resolution_kw|min_rows"):
            wind.NightLimitMeasure("power_kw", "wind_speed", RATED_WIND, **arguments)


class TestWindRulesInAContract:
    def _layer(self, measures: dict, checks: dict) -> contract.LayerSpec:
        data = {
            "id": "c",
            "version": "1",
            "domain": "wind",
            "layers": {"raw": {"measures": measures, "checks": checks}},
        }
        return contract.build_contract(data).get_layer("raw")

    def test_the_registry_builds_both_measures_with_subject_numbers(self):
        registry = compose.create_default_registry()
        layer = self._layer(
            {
                "levels": {
                    "rule": "power_levels",
                    "signal": "power_kw",
                    "params": {
                        "wind_speed": "wind_speed",
                        "rated_power_kw": {"subject": "rated_power_kw"},
                        "min_wind_speed": {"subject": "rated_wind_speed_mps"},
                    },
                },
                "night": {
                    "rule": "night_limit",
                    "signal": "power_kw",
                    "params": {
                        "wind_speed": "wind_speed",
                        "min_wind_speed": {"subject": "rated_wind_speed_mps"},
                        "night_start_hour": 22,
                        "night_end_hour": 6,
                    },
                },
            },
            {
                "no_curtailment": {
                    "measure": "levels",
                    "value": "n_curtailed_rows",
                    "severity": "warning",
                    "good": {"max": 0},
                },
                "no_night_limit": {
                    "measure": "night",
                    "value": "n_limited_rows",
                    "severity": "warning",
                    "good": {"max": 0},
                },
            },
        )

        compose.validate_layer_rules(layer, registry)

    def test_a_night_limit_without_its_hours_is_refused(self):
        layer = self._layer(
            {"night": {"rule": "night_limit", "signal": "power_kw", "params": {"wind_speed": "wind_speed"}}},
            {"c": {"measure": "night", "value": "n_limited_rows", "severity": "info", "good": {"max": 0}}},
        )

        with pytest.raises(contract.ContractError, match=r"min_wind_speed|night_start_hour"):
            compose.validate_layer_rules(layer, compose.create_default_registry())

    def test_the_rule_builds_the_kde_method_from_its_parameters(self):
        rule = compose.create_default_registry().find_rule("power_levels")

        measure = rule.build(
            "levels",
            "power_kw",
            {
                "wind_speed": "wind_speed",
                "rated_power_kw": RATED_KW,
                "min_wind_speed": RATED_WIND,
                "method": "kde",
                "prominence_threshold": 1e-5,
            },
        )

        assert measure == wind.PowerLevelsMeasure(
            "power_kw",
            "wind_speed",
            RATED_KW,
            RATED_WIND,
            method=wind.LevelMethod.KDE,
            prominence_threshold=1e-5,
            measure_id="levels",
        )

    def test_the_rule_refuses_an_unknown_method(self):
        rule = compose.create_default_registry().find_rule("power_levels")
        params = {"wind_speed": "wind_speed", "rated_power_kw": 1.0, "min_wind_speed": 1.0, "method": "fft"}

        with pytest.raises(ValueError, match="fft"):
            rule.build("levels", "power_kw", params)
