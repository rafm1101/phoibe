# Glossary

The design uses fourteen terms. The [ADRs](adr/) use them in exactly this sense.

| Term | Meaning |
|---|---|
| **Subject** | The asset under examination: a turbine now, later a met mast or a PV plant. It carries a specification, such as rated power, rotor diameter and site time zone. |
| **Dataset** | An immutable time series of one subject, with a simple fingerprint. A fix or a transform produces a new dataset version. |
| **Signal** | A column mapped to a quantity and a statistic, for example power / mean or wind speed / max, and, when sources are combined, a measurement point such as source or height. |
| **Layer** | A quality stage with its own contract section and gate (raw, bronze, silver, gold, or as the contract names them). |
| **Aspect** | One quality dimension inside a layer, for example completeness, temporal regularity or plausibility. |
| **Measurement** | A computation for an aspect that yields numbers (metrics) and findings. No judgment. The code calls the computing object a *measure* and its output an *outcome*. |
| **Finding** | A located observation: kind, signals, interval, number of steps, magnitude. |
| **Profile** | All measurements and findings of one dataset for one layer. |
| **Contract** | User-authored and versioned. Per layer it defines measures with their parameters, and checks with thresholds, severity and a free-text reference to the source clause. |
| **Criterion / check** | A testable expectation on one value of one measurement, with one band or two (good, acceptable) and a severity that says what a violation does to the gate. The contract calls it a *check*. |
| **Verdict** | The outcome of a check: good, acceptable, bad, not evaluable or error. |
| **Gate** | The policy that composes verdicts into the outcome of a layer: pass, conditional or fail. |
| **Attestation** | The record that a dataset fingerprint passed the gate of layer L under contract version V, with its conditions and its parent. |
| **Transform** | A deterministic, recorded function from one dataset version to the next, for example renaming to catalog names or localizing time. Cleaning and imputation are not transforms of this package. |

Further terms used in the documents:

| Term | Meaning |
|---|---|
| **Severity** | Critical, warning or info. Stated by the contract on each check; there is no default. |
| **Catalog** | The shared list of signals, units and detection patterns of a domain pack. |
| **Domain pack** | A package holding a domain's catalog and rules. `wind` is the first. |
| **Fingerprint** | A simple hash of a dataset, used as a consistency check, not as tamper protection. |
| **Provenance** | The record on a dataset that names its parent attestation, the transform applied and the output fingerprint. |
| **Medallion** | The convention of naming layers raw, bronze, silver and gold. Gold means ready for further processing, which this package does not do. |
| **Compensation** | An exception to a violated check justified by other measurements or other time series. Deferred, not designed. |
| **TR** | The technical requirements (for example TR6 or TR10) from which the user writes the contract. |
| **Ground-truth label** | The record a generator messup step keeps of the segment it corrupted. Used to score detection. |
| **Legacy package** | `phoibe.layered`, the intermediate package this design replaces. It stays untouched. |

Compensation is deliberately not among the fourteen terms: it is deferred (see [open questions](open-questions.md)).
