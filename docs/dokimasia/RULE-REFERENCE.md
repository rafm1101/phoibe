# Rule reference

A **rule** is a named, registered kind of measure. A contract instantiates it as a *measure* with parameters ([ADR-0004](adr/0004-contract.md)); a *check* then reads one metric of that measure ([ADR-0005](adr/0005-gate-is-a-policy-over-verdicts.md)). This page lists the ten rules of the default registry, then the rules that are missing or still to be discussed. It is the starting point for deciding what to add, including for new domain packs ([extending with a pack](extending-with-a-pack.md)).

The rules are listed as the code reports them (`compose.create_default_registry()`, at project close). "Proposed layer" is a proposal: the layers are named by the contract, and the draft contract `contracts/wind-scada-10min.yaml` is the only one so far.

## Conventions

- **Signals.** A rule asks for catalog names, never column names. Raw data carries detected names (`power`); after the standardizing transform ([ADR-0011](adr/0011-standardizing-transforms.md)) later layers use the catalog names of the pack (`power_kw`, `wind_speed`, `rotor_speed`, `pitch_angle`, each with `_min` and `_max`). Signal use in the tables: *none* (the rule names its signals in a parameter or needs none), *optional* (no signal means all signals), *required* (the measure's `signal` key).
- **Metrics.** Numbers a check can read. A *share* is a fraction between 0 and 1. A *step* is one time step of the dataset. Metrics are per signal where the rule says so.
- **Findings.** Located observations as run-length intervals ([ADR-0003](adr/0003-findings-are-located-score-is-informational.md)). A rule that only describes the data returns none.
- **Not evaluable.** A rule never raises for bad input ([ADR-0012](adr/0012-hostile-input-total-measurements.md)). It returns *not evaluable* with a reason: `missing_signal` (a named signal is not in the dataset), `missing_structure` (the signal is ambiguous or the time axis lacks what the rule needs) or `insufficient_data` (too few usable rows). The gate treats it like *bad* at the check's severity.
- **Aspect.** The aspect string the code assigns. It differs in places from the aspect names used in the fault analysis; see [aspect assignment](#aspect-assignment).
- **Coverage.** Whether the generator injects a matching fault and the rule finds it ([architecture](architecture.md#fault-coverage)). Generator data only.

## Overview

| Rule | Package | Aspect (code) | Signal | Proposed layer | Findings | Coverage |
|---|---|---|---|---|---|---|
| [`required_signals`](#required_signals) | `rules` | structure | none | raw | none | n/a |
| [`time_gaps`](#time_gaps) | `rules` | temporal | none | raw, bronze | `gap` | dropped rows: 1.00 / 1.00 |
| [`time_span`](#time_span) | `rules` | temporal | none | bronze | `time_span`, `short_step` | none (no injected fault) |
| [`missing_runs`](#missing_runs) | `rules` | completeness | optional | raw | `missing` | deleted entries: 1.00 / 1.00 |
| [`ranges`](#ranges) | `rules` | validity | required | bronze | `out_of_range` | none; spikes wait (OQ-02) |
| [`essential_range`](#essential_range) | `rules` | distribution | required | bronze | none | n/a (describes) |
| [`statistic_order`](#statistic_order) | `rules` | consistency | required | raw or bronze | `statistic_order` | mean outside min and max: 1.00 / 1.00 |
| [`frozen_runs`](#frozen_runs) | `rules` | consistency | none | bronze | `frozen` | joint freeze: 1.00 / 1.00 |
| [`power_levels`](#power_levels) | `wind` | validity | required | bronze | `curtailment_level` | held level: count 0.93 / 1.00, kde 1.00 / 0.98 |
| [`night_limit`](#night_limit) | `wind` | validity | required | bronze | `night_limit` | night limit: 1.00 / 1.00 |

Coverage reads recall / precision, pooled over four seeds and the incidence × level grid.

## Rules

### `required_signals`

**Definition.** Reports which of the required signals are in the dataset.
**Package, aspect.** `rules`, structure. **Proposed layer.** raw: it comes before everything else, because later measures declare what they need.
**Signals.** none; the signals are a parameter.

| Parameter | Type | Required | Meaning |
|---|---|---|---|
| `signals` | list of names | yes | Names that must be present. At least one. |

| Metric | Meaning |
|---|---|
| `n_missing_signals` | How many required signals are absent. |
| `present` | Per required signal: 1 if present, 0 if not. |

**Not evaluable.** Never; the measure is always evaluable.
**Typical check.** `n_missing_signals` good max 0, critical.

### `time_gaps`

**Definition.** Finds the intervals of the time axis in which whole time steps are missing. It works on the distinct, valid timestamps in time order. A difference of `m` steps between neighbors is a gap of `m − 1` missing steps; the finding spans the missing timestamps. Missing steps at the very start or end have no neighbor and are not seen. NaT, repeated and out-of-order timestamps are counted, not repaired.
**Package, aspect.** `rules`, temporal. **Proposed layer.** raw and bronze (the draft contract measures both).
**Signals.** none.

| Parameter | Type | Required | Meaning |
|---|---|---|---|
| `expected_step_s` | number (seconds) | no | Length of one step. Default: the median difference of the data, reported as `step_s`. Must be positive. |

| Metric | Meaning |
|---|---|
| `step_s` | The step that was used, in seconds. |
| `n_gaps` | Number of gaps. |
| `n_missing_steps` | Steps missing in all gaps. |
| `longest_gap_steps` | Missing steps in the longest gap. |
| `share_missing_steps` | Missing steps as a share of the steps expected. |
| `n_missing_timestamps` | Rows whose timestamp is missing (NaT). |
| `n_duplicate_timestamps` | Repeated timestamps. |
| `n_decreasing_steps` | Steps where time goes backwards. |

**Not evaluable.** Fewer than two distinct valid timestamps.
**Finding.** `gap`, with start and end of the missing timestamps.
**Typical checks.** `share_missing_steps` as availability of the time axis; `n_duplicate_timestamps` and `n_decreasing_steps` at max 0.

### `time_span`

**Definition.** Reports the period of the time axis, its time zone form, and steps shorter than they should be. One `time_span` finding runs from the first to the last timestamp. A step of less than half the expected step is oversampled; consecutive ones are one `short_step` finding.
**Package, aspect.** `rules`, temporal. **Proposed layer.** bronze, after localization to UTC. At raw the naive axis makes start, end and offsets ambiguous.
**Signals.** none.

| Parameter | Type | Required | Meaning |
|---|---|---|---|
| `expected_step_s` | number (seconds) | no | As in `time_gaps`. |

| Metric | Meaning |
|---|---|
| `n_timestamps` | Distinct, valid timestamps. |
| `span_s` | Length of the period in seconds. |
| `start_epoch_s`, `end_epoch_s` | Start and end, seconds since 1970 in UTC. A naive axis is read as UTC. |
| `is_timezone_aware` | 1 if the axis carries a time zone, else 0. |
| `n_utc_offsets` | Distinct offsets from UTC: 1 for UTC or a naive axis, more where daylight saving time shifts it. |
| `n_short_steps` | Steps shorter than half the expected step. |

**Not evaluable.** No row has a valid timestamp.
**Typical checks.** `is_timezone_aware` equal to 1; `n_utc_offsets` equal to 1 after standardizing; `start_epoch_s` and `end_epoch_s` inside the agreed period (open: [OQ-11](open-questions.md)).

### `missing_runs`

**Definition.** Finds runs of missing values (NaN, None, NaT) in each signal. Rows are taken in time order; rows without a valid timestamp are left out. Infinite values and text are not missing; other measures look at them.
**Package, aspect.** `rules`, completeness. **Proposed layer.** raw (completeness per signal). The draft contract measures it at bronze, on the four mean signals after renaming.
**Signals.** optional: one signal, or all signals if none is given.

| Parameter | Type | Required | Meaning |
|---|---|---|---|
| none | | | |

| Metric (per signal) | Meaning |
|---|---|
| `n_missing` | Missing values. |
| `share_missing` | Missing values as a share of the rows. |
| `n_runs` | Runs of missing values. |
| `longest_run_steps` | Steps in the longest run. |

**Not evaluable.** A named signal is absent, or the dataset has no signal, or there are too few rows.
**Finding.** `missing`, one per run, with the signal and the first and last timestamp.
**Typical checks.** `share_missing` per signal; `longest_run_steps` where a long outage matters more than the total.

### `ranges`

**Definition.** Counts the values of one signal outside `[min, max]` and locates the runs. Text is read as numbers where possible; what cannot be read is counted as `n_unreadable` and not checked. Missing values are not checked. Infinite values lie outside any finite limit.
**Package, aspect.** `rules`, validity. **Proposed layer.** bronze.
**Signals.** required.

| Parameter | Type | Required | Meaning |
|---|---|---|---|
| `min` | number or subject reference | one of `min`, `max` | Lower limit, inclusive. |
| `max` | number or subject reference | one of `min`, `max` | Upper limit, inclusive. At least one of the two. Lower not above upper. |

A subject reference is `{subject: <attribute>, factor, offset}` ([ADR-0004](adr/0004-contract.md)), for example `max: {subject: rated_power_kw, factor: 1.05}`.

| Metric | Meaning |
|---|---|
| `n_out_of_range` | Values outside the limits. |
| `share_out_of_range` | As a share of the checked values. |
| `n_checked` | Values that were checked. |
| `n_unreadable` | Text values that could not be read as numbers. |
| `limit_min`, `limit_max` | The limits as applied, so the profile records them. |

**Not evaluable.** Signal missing, ambiguous (duplicate columns), or nothing to check.
**Finding.** `out_of_range`.
**Typical checks.** `share_out_of_range` with a good band at 0. The limits come from the TR ([OQ-06](open-questions.md)).
**Note.** The only rule that counts unreadable values; see [rule-level gaps](#rule-level-gaps).

### `essential_range`

**Definition.** Finds the shortest interval that holds a proportion of the values of one signal. It describes where the bulk of the data lies and finds no rows. `ranges` is the check against limits. Missing and infinite values are left out; numeric text is read as a number. The algorithm follows the legacy `EssentialRange` (an interval holds ⌈p·n⌉ + 1 values; the lowest interval wins ties).
**Package, aspect.** `rules`, distribution. **Proposed layer.** bronze, as information that a person reads or that a check compares with an expected spread.
**Signals.** required.

| Parameter | Type | Required | Meaning |
|---|---|---|---|
| `proportion` | number | no | Share the interval holds; above 0 and up to 1. Default 0.995. |

| Metric (on the signal) | Meaning |
|---|---|
| `essential_min`, `essential_max` | The interval. |
| `essential_width` | Its width. |
| `n_values` | Finite values used. |
| `proportion` | The proportion applied. |

**Not evaluable.** Signal missing or ambiguous, or no finite value.
**Findings.** none.
**Typical checks.** `essential_width` or the bounds, as warning or info; an essential range far wider than physics allows hints at outliers that `ranges` did not catch.

### `statistic_order`

**Definition.** Finds rows where a mean lies outside its minimum and maximum, or the minimum lies above the maximum. A row is checked if all three values are numbers. A row violates if `lower − tolerance > middle` or `middle > upper + tolerance`; a minimum above the maximum is caught by one of the two. Consecutive violating rows are one finding.
**Package, aspect.** `rules`, consistency. **Proposed layer.** raw or bronze. The draft contract uses bronze, with one measure per quantity.
**Signals.** required: the measure's `signal` is the middle (mean) signal.

| Parameter | Type | Required | Meaning |
|---|---|---|---|
| `lower` | name | yes | Signal of the minimum. |
| `upper` | name | yes | Signal of the maximum. |
| `tolerance` | number | no | Rounding tolerance, default 0. |

| Metric (on the middle signal) | Meaning |
|---|---|
| `n_violations` | Violating rows. |
| `share_violations` | As a share of the checked rows. |
| `n_checked` | Rows with all three values. |
| `n_runs` | Findings. |
| `tolerance` | The tolerance applied. |

**Not evaluable.** A signal is missing or ambiguous, or no row has all three values.
**Finding.** `statistic_order`, naming the three signals.
**Typical checks.** `n_violations` or `share_violations` at max 0, critical. Real data may need a tolerance ([OQ-07](open-questions.md)).

### `frozen_runs`

**Definition.** Finds runs in which all given signals hold the same values step after step. A turbine that logs the state of its detectors instead of tracking current values holds several signals constant at once. One signal alone staying constant says little, because a stopped turbine does that anyway. A row repeats the previous one if every signal equals its previous value; a missing value never repeats. A run of `n` repeating rows is one finding of `n + 1` steps, and only runs of at least `min_steps` steps are findings.
**Package, aspect.** `rules`, consistency. **Proposed layer.** bronze (sensor plausibility).
**Signals.** none; the signals are a parameter.

| Parameter | Type | Required | Meaning |
|---|---|---|---|
| `signals` | list of names | yes | Signals that must all repeat. No default. |
| `min_steps` | whole number | yes | Shortest run that counts. No default. |

| Metric | Meaning |
|---|---|
| `n_frozen_steps` | Steps inside findings. |
| `share_frozen_steps` | As a share of the rows. |
| `n_runs` | Findings. |
| `longest_run_steps` | Steps in the longest finding. |
| `min_steps` | The value applied. |

**Not evaluable.** A signal is missing or ambiguous, or fewer than two rows have a valid timestamp.
**Finding.** `frozen`.
**Typical checks.** `share_frozen_steps` or `longest_run_steps`. Signals and length wait for the TR ([OQ-01](open-questions.md)).

### `power_levels`

**Definition.** Finds the power levels the turbine holds at high wind speed, and the rows that sit on a curtailed level. From `min_wind_speed` on, a turbine produces rated power. Power is rounded to multiples of `resolution_kw`. With method `count`, a value shared by at least `min_count` of these rows is a level; with `kde`, the levels are the prominent peaks of a Gaussian density estimate (the legacy method). A level more than `resolution_kw` below `rated_power_kw` is curtailed. Every high-wind row within one resolution step of a curtailed level is a finding. Rows below `min_wind_speed` are not looked at.
**Package, aspect.** `wind`, validity (the fault analysis calls it operating state). **Proposed layer.** bronze.
**Signals.** required: the power signal; the wind speed signal is a parameter.

| Parameter | Type | Required | Meaning |
|---|---|---|---|
| `wind_speed` | name | yes | Wind speed signal. |
| `rated_power_kw` | number or subject reference | yes | Rated power; usually `{subject: rated_power_kw}`. |
| `min_wind_speed` | number or subject reference | yes | Wind speed from which rated power is expected; usually the rated wind speed. |
| `resolution_kw` | number | no | Rounding grid, default 10. |
| `method` | `count` or `kde` | no | Default `count`. |
| `min_count` | number | no | Rows a level needs under `count`, default 5. |
| `prominence_threshold` | number | no | Peak prominence under `kde`, default 1e-7. |
| `min_rows` | number | no | High-wind rows needed at all, default 20. |

| Metric (on the power signal) | Meaning |
|---|---|
| `n_high_wind_rows` | Rows at or above `min_wind_speed` with power. |
| `n_levels` | Levels found. |
| `n_curtailed_levels` | Levels below rated power by more than the resolution. |
| `highest_level_kw` | Highest level; 0 if none. |
| `n_curtailed_rows` | Rows in findings. |
| `share_curtailed_rows` | As a share of the high-wind rows. |

**Not evaluable.** A signal is missing or ambiguous, or fewer than `min_rows` high-wind rows.
**Finding.** `curtailment_level`.
**Typical checks.** `highest_level_kw` close to rated power (full load is reached); `share_curtailed_rows` at a low maximum. The method decides how the first behaves on real data ([OQ-08](open-questions.md), [OQ-09](open-questions.md)).

### `night_limit`

**Definition.** Finds a power limit that applies at night and not by day. At high wind speed, the most frequent rounded power value is found for the night hours and for the rest of the day. If the night value is more than one resolution step below the day value and at least `min_share` of the night's high-wind rows sit on it, there is a night limit, and every night row on that value, at any wind speed, is a finding. Night runs from `night_start_hour` (inclusive) to `night_end_hour` (exclusive) and wraps over midnight. With `time_zone` the axis is converted first, so the night keeps its local hours across daylight saving time.
**Package, aspect.** `wind`, validity (the fault analysis calls it operating state). **Proposed layer.** bronze.
**Signals.** required: the power signal; the wind speed signal is a parameter.

| Parameter | Type | Required | Meaning |
|---|---|---|---|
| `wind_speed` | name | yes | Wind speed signal. |
| `min_wind_speed` | number or subject reference | yes | High-wind threshold. |
| `night_start_hour`, `night_end_hour` | whole hours | yes | Night, local hours; 22 and 6 for Germany. |
| `time_zone` | IANA name | no | Zone for the hours, for example `Europe/Berlin`. The axis must be time-zone aware then. Unknown zone is refused at load. |
| `resolution_kw` | number | no | Default 10. |
| `min_rows` | number | no | Night rows and day rows at high wind each needed, default 10. |
| `min_share` | number | no | Share of night rows that must sit on the level, default 0.5. |

| Metric (on the power signal) | Meaning |
|---|---|
| `n_night_high_wind_rows`, `n_day_high_wind_rows` | Rows the comparison rests on. |
| `night_level_kw`, `day_level_kw` | Most frequent rounded power at night and by day. |
| `share_night_at_level` | Share of night high-wind rows on the night level. |
| `night_limit_depth_kw` | Day level less night level; zero or less means no limit. |
| `n_limited_rows` | Rows in findings. |

**Not evaluable.** A signal is missing or ambiguous; too few night or day rows at high wind; `time_zone` set and the axis is naive.
**Finding.** `night_limit`.
**Typical checks.** `n_limited_rows` or `night_limit_depth_kw` at max 0. The hours for other sites are open ([OQ-10](open-questions.md)).

## Aspect assignment

The code assigns one aspect string per measure class. The fault analysis in the [legacy review](legacy-review.md) used other names. The two do not yet agree:

| Rule | Aspect in code | Aspect in the fault analysis |
|---|---|---|
| `frozen_runs` | consistency | sensor plausibility |
| `statistic_order` | consistency | statistic consistency |
| `power_levels`, `night_limit` | validity | operating state |
| `ranges` | validity | plausibility |
| zeroing (not built) | – | cross-signal consistency |

Aspects are identifiers; the contract may group checks by them. A decision is needed before silver and gold are written ([open questions](open-questions.md), DP-08).

## Coverage by aspect

Which aspects have rules, and where the holes are. Rows are the aspects in code plus the two the fault analysis adds.

| Aspect | Rules today | Missing or open |
|---|---|---|
| structure | `required_signals` | Structure measurement: form of the time axis (index or column, naive or aware, share unparseable), duplicate column names, empty and constant columns ([ADR-0012](adr/0012-hostile-input-total-measurements.md)) |
| temporal | `time_gaps`, `time_span` | Timestamps off the grid (no rule); start, end and zone expectations are a contract matter (OQ-11) |
| completeness | `missing_runs` | Conditional availability (OQ-05) |
| validity | `ranges`, `power_levels`, `night_limit` | Spikes (OQ-02); sentinel values; unreadable values counted in every rule |
| consistency | `statistic_order`, `frozen_runs` | Duplicate rows with differing values |
| distribution | `essential_range` | none planned |
| operating state | `power_levels`, `night_limit` (filed under validity) | Zeroing of the whole turbine (OQ-04) |
| cross-signal consistency | none | Zeroing; rotor speed against power (proposal) |

## Missing rules and candidates

**Documented** means the design or an answer already names it. **Proposal** means it is a suggestion made here, to be accepted, changed or dropped; nothing of it is decided.

| ID | Rule | Aspect | Origin | Needs | Generator class | Note |
|---|---|---|---|---|---|---|
| R-01 | Spikes | validity | Documented | A definition from the domain experts (OQ-02) | `Spike`, uncovered | No design until the definition exists. |
| R-02 | Whole-turbine zeroing | cross-signal consistency | Documented | The role of cut-in and cut-out speed, and the zeroing window (OQ-04) | `CurtailmentToZero`, uncovered | Needs a turbine that should run: wind between cut-in and cut-out, and power, rotor speed near zero together. Attributes `cut_in_wind_speed_mps`, `cut_out_wind_speed_mps` are read from the subject. |
| R-03 | Conditional availability | completeness | Documented | The context and the measurement (OQ-05) | none | Parked. Probably availability given a state, such as availability when wind is above cut-in. |
| R-04 | Structure measurement | structure | Documented ([ADR-0012](adr/0012-hostile-input-total-measurements.md)) | Nothing external | none | One measure that describes what arrived and is the first of the raw layer. Today `required_signals` and `time_span` cover a part. |
| R-05 | Unreadable values counted | validity | Documented ([ADR-0012](adr/0012-hostile-input-total-measurements.md)) | Nothing external | none | Either every number-reading rule reports `n_unreadable` as `ranges` does, or one measure counts them per signal. The second keeps the rules small. |
| R-06 | Sentinel values (such as −9999), comma decimals, mixed units | validity | Documented ([ADR-0012](adr/0012-hostile-input-total-measurements.md)) | A list of sentinels from the data owner | none | Findings in raw; never repaired by a transform. |
| R-07 | Duplicate rows | consistency | Documented ([ADR-0011](adr/0011-standardizing-transforms.md)) | Nothing external | none | `time_gaps` counts repeated timestamps only. Rows that repeat with different values are a finding, not a transform. |
| R-08 | Single-signal freeze | validity | Decided against as a stand-alone rule | – | `Freeze` on one signal, uncovered | A stopped turbine holds signals constant. Only meaningful if conditioned on an operating state; see R-10. |
| R-09 | Timestamps off the grid | temporal | Proposal | Nothing external | none | `time_gaps` rounds differences to whole steps, so a series on 10:01, 10:11, … is not reported. A measure of the offset to the expected grid would make it visible. |
| R-10 | Operating-state conditioning | operating state | Proposal | Agreement on how state is derived (wind, rotor speed, a status signal) | none | A shared helper that gives other rules a mask such as "turbine should be running". It would serve R-02, R-03 and R-08. A candidate for the wind pack. |
| R-11 | Rate of change | validity | Proposal | Overlaps R-01 | `Spike` | A step limit per signal. May be one form of the answer to R-01. |
| R-12 | Rotor speed against power | cross-signal consistency | Proposal | Domain experts | none | Power above a small value while rotor speed is zero, or the reverse. May be an alternative route to R-02. |
| R-13 | Power against wind speed | operating state | Proposal | Domain experts, a reference curve | none | Distance of power from the expected curve. This is beyond curtailment levels and approaches a performance check, which is a [non-goal](goals.md); keep it to plausibility. |
| R-14 | Silver and gold rules | not defined | Open | The meaning of silver and gold | none | No rule yet. Gold means ready for further processing ([ADR-0004](adr/0004-contract.md)); silver is unspecified. |

## Rule-level gaps

Existing rules whose behavior still needs work:

- Only `ranges` counts the values it cannot read. `missing_runs`, `statistic_order`, `frozen_runs`, `essential_range`, `power_levels` and `night_limit` treat unreadable text as missing without counting it, which goes against "never coerces silently" ([ADR-0012](adr/0012-hostile-input-total-measurements.md)); see R-05.
- No rule enforces declared row or memory limits; `limit_exceeded` exists only as a reason.
- `statistic_order` has no default tolerance and its bronze checks are critical (OQ-07).
- `power_levels` with method `count` can fail healthy real data because scatter of tens of kW leaves no value shared by five rows (OQ-08).
- `night_limit` reads hours and zone from its parameters; the subject attribute `site_timezone` is not read because the registry has no text-valued subject references (OQ-10).
- The registry reports metrics but not a description; this page is written by hand. A test that compares this page with the registry (names, parameters, metrics) would keep them in step; it is not written.
