# Extending with a pack

A **pack** adds a domain to `phoibe.dokimasia` without touching `core` ([ADR-0001](adr/0001-domain-model.md), [ADR-0007](adr/0007-package-structure-and-dependency-direction.md)). `wind` is the only pack today. This page says what a pack consists of, how a new rule is made and accepted, and sketches what a LiDAR pack could contain. Start from the [rule reference](rule-reference.md): most of a new pack is reuse.

## What a pack consists of

| Part | In `wind` today | What a new pack provides |
|---|---|---|
| Catalog | `wind.create_catalog()`: twelve signals (power, wind speed, rotor speed, pitch angle, each as mean, minimum and maximum) with quantity, statistic and unit | A `core.Catalog` of its own signals |
| Detection patterns | `wind.PATTERNS` and `wind.detect_wind_signals(columns)` | Regular expressions that map column names to catalog names; ambiguity is reported, never decided |
| Rules | `PowerLevelsMeasure`, `NightLimitMeasure` in `wind/curtailment.py` | Measure classes that need domain terms. Generic ones stay in `rules`. |
| Registry entries | `compose/rules_wind.py`: `list_wind_rules()` | `compose/rules_<pack>.py` with `list_<pack>_rules()`, added in `compose/defaults.py` |
| Transforms (optional) | none; renaming and localization are generic | Transforms from `core.Transform`, if the data needs a domain-specific reshape |
| Subject attributes | `rated_power_kw`, `rated_wind_speed_mps`, `cut_in_wind_speed_mps`, `cut_out_wind_speed_mps` | The attributes its rules and contract read, in `Subject.specification` |
| Contract | `contracts/wind-scada-10min.yaml` | A contract per product and layer |
| Fault classes | the generator's messup steps and `tests/unit_tests/dokimasia/rules/test_coverage.py` | A way to inject faults with ground truth, so detection can be scored |

A pack imports `core` and `rules` only. The registry entry lives in `compose`, which is the one place that knows all packs; the pack itself never imports the registry.

## Making a new rule

1. **State the fault first.** Which fault does it find, in which aspect, in which layer, from which signals? If no one can define the fault, stop; spikes ([OQ-02](open-questions.md)) are the example.
2. **Decide generic or domain.** If no domain term appears in the definition, it belongs in `rules`. A rule that needs a rated power or a wind-speed threshold belongs in a pack.
3. **Write the measure** as a frozen, slotted dataclass with `ASPECT` and `METRICS` as class variables, an `id`, an `aspect` and `measure(dataset) -> Measured | NotEvaluable`. Metrics are `Metric(key, value, unit, signal)`. Findings are `Finding(kind, signals, start, end, n_steps, magnitude, magnitude_unit)` as run-length intervals, in time order, with NaT rows left out.
4. **Make it total** ([ADR-0012](adr/0012-hostile-input-total-measurements.md)). Expected problems return `NotEvaluable` with a reason; read numbers with `rules.read_numeric_signals`, and count what could not be read. Do not catch `Exception`; `run_measure` does.
5. **Choose defaults sparingly.** A parameter that stands for a domain decision (frozen signals and length) has no default.
6. **Register it** with `Rule(name, build, metrics, parameters, signal, at_least_one_of)`. Parameter kinds are number, number or subject reference, text and list of text.
7. **Test it against injected faults.** Add a row to the coverage table: fault class, injector, expected recall and precision, and the clean-data false-alarm count. Add unit tests for the not-evaluable cases and the boundary values. Measure first and set floors from the measured values.
8. **Update the surface.** Add the name to the package's `__all__`, the `EXPECTED` dict of `test_public_api`, and the vulture whitelist.
9. **Write the entry** in the [rule reference](rule-reference.md): definition, parameters, signals, package, proposed layer, metrics, findings, not-evaluable cases, typical checks.

A rule is accepted when it finds its injected fault at the agreed recall and precision, gives no false alarm on clean data, and is total on the hostile corpus.

## Making a new pack

1. **Subject.** What is observed and which attributes describe it? `Subject.specification` stays a plain mapping; no domain term enters `core`.
2. **Catalog.** List quantities, statistics and units. The signal model has `name`, `quantity`, `statistic` (mean, min, max or std), `unit` and an optional `measurement_point`, such as a source or a height. Signal names must be unique, so a measurement point is part of the name when several exist.
3. **Detection patterns.** Write patterns for the column names of the real exports. Test them against the demonstration data, including near misses; overlapping patterns are reported as ambiguous.
4. **Reuse what exists.** Check each generic rule in the overview table against the new data. The time-axis, completeness, range, order and frozen rules have no wind terms.
5. **Add domain rules** only for what the generic ones cannot say.
6. **Generator.** A pack with no way to inject faults has no oracle ([ADR-0009](adr/0009-testing-strategy-generator-as-oracle.md)). Either write a generator for the domain or restrict the first version to the rules whose faults a generic injector can place.
7. **Contract.** Draft one contract with raw and bronze layers, mark every band as a proposal, and name the technical requirement for each clause.
8. **Wire and enforce.** Add the package to the import-direction test (`core`, `rules`), add `list_<pack>_rules()` to `compose/defaults.py`, and run the full suite.

Limits of the current model that a new domain may meet:

- `Statistic` is a closed list (mean, min, max, std). A count or a quality indicator would need an addition to `core`, which is the first place a domain can force a change.
- One run covers one subject and one dataset in wide form (one column per signal). Data in long form needs a reshape transform before profiling.
- A joint catalog across sources is designed ([ADR-0001](adr/0001-domain-model.md)) but not exercised; the fingerprint does not yet say which source a column came from.

## Sketch: a LiDAR pack

**This is a proposal for discussion.** The signal list, the faults and the rules below come from general knowledge of profiling lidars, not from a LiDAR data set or from the project owner's requirements. Every item needs confirmation by people who work with the instrument and its export format.

### Subject

A LiDAR is a different kind of subject from a turbine: a ground-based or nacelle-mounted instrument, with a site, a measurement height list and a scan mode. Proposed specification attributes: `device_type`, `mounting` (ground or nacelle), `measurement_heights_m`, `site_timezone`, `sampling_interval_s`, `latitude`, `longitude`, `reference_met_mast` (if any).

### Catalog

Most quantities come once per measurement height, so the height is a measurement point and part of the name.

| Quantity | Statistic | Unit | Per height |
|---|---|---|---|
| `wind_speed_<h>m` | mean, min, max, std | m/s | yes |
| `wind_direction_<h>m` | mean | deg | yes |
| `vertical_wind_speed_<h>m` | mean | m/s | yes |
| `cnr_<h>m` (carrier-to-noise ratio) | mean, min | dB | yes |
| `availability_<h>m` (valid data share in the averaging interval) | mean | share | yes |
| `status` or `error_code` | none | none | no |
| `tilt`, `roll` (platform tilt) | mean, max | deg | no |
| `temperature`, `pressure` (instrument or ambient) | mean | °C, hPa | no |

Open on this table: whether availability and status are signals or a column the export already summarizes, and whether `Statistic` needs a count or a "share" kind.

### Reuse of existing rules

| Existing rule | Use for the lidar |
|---|---|
| `required_signals` | The expected heights are present. |
| `time_gaps`, `time_span` | Missing intervals; UTC time axis; start and end. |
| `missing_runs` | Missing values per height; long outages. |
| `ranges` | Wind speed (0 to a limit), direction (0 to 360), CNR, tilt; limits from the instrument's specification. |
| `statistic_order` | Mean between min and max of the wind speed at one height. |
| `frozen_runs` | Several heights held constant at once, which points to a logger fault, as for turbines. |
| `essential_range` | Spread of wind speed and CNR, as information. |

### New rules to consider

| Candidate | Aspect | Idea | Needs |
|---|---|---|---|
| Availability per height | completeness | Share of intervals with enough valid samples at each height. Could be `missing_runs` over per-height availability signals, or a measure that reads them directly. | The export format |
| CNR threshold | validity | Share of values below a CNR floor at each height; low CNR makes a measurement unreliable. | The instrument's CNR floor |
| Shear plausibility | consistency | Wind speed against height within one interval: speeds that fall with height in a way physics does not allow, or a jump between neighbors. | Experts |
| Direction circularity | validity | Direction outside 0 to 360 and jumps across the wrap, which a plain range check does not handle. | none |
| Platform tilt | consistency | Tilt or roll beyond the limit of the instrument; a nacelle lidar also needs the nacelle position. | Instrument manual |
| Reference comparison | cross-source | Agreement with a met mast or a second instrument over the same period. This is a comparison across datasets, which the design calls compensation or a fleet-style check and has not designed ([ADR-0005](adr/0005-gate-is-a-policy-over-verdicts.md)). | A design for cross-dataset checks |
| Vertical wind speed | validity | Vertical speed beyond a physical limit at flat sites. | Experts |

The first four are within the current design. The reference comparison is not: it needs a gate policy or a layer that reads more than one dataset, which is deferred.

### Layers

A first LiDAR contract could mirror the turbine one: **raw** (signals present, time gaps, structure) and **bronze** (UTC time axis, ranges, order, frozen state, availability and CNR per height). Silver and gold are undefined for both packs.

### Faults for a generator

Without a generator the detection tests cannot be written. The faults to inject, with ground truth, would be: dropped intervals, missing values at one height, a CNR dropout at the top heights, frozen values at all heights, mean outside min and max, a direction wrap error, and a tilt excursion. Which of these a generic injector already covers is for the generator cleanup ([open questions](open-questions.md)).

### What to ask before building

1. Which instrument and export format is first (profiling lidar, scanning, nacelle-mounted)?
2. Is there a technical requirement for the data quality, as TR6 and TR10 are for turbines?
3. Is a reference mast available, and is a comparison in scope?
4. May `Statistic` change in `core`, or must the export be flattened into the existing four?
5. Real data or generator data for the first rules?
