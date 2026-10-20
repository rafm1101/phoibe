# 0004 Parameters: contract and run

Status: Accepted

## Context

Some parameters define what the numbers mean (TR6), some are free choices of the caller, and some are choices of this implementation. The rounding of the limit distances A and B happens before they are compared with the distances, so it changes the classification (a pair at 8.48 km with a limit of 8.4565 km is "conditional" without rounding and "unconditional" with one decimal). It is a parameter, not a matter of presentation.

## Decision

Every parameter is defined in the product definition with default, type, optional range, optional source and `locked`.

- Locked and settable (`n_angles`, `R_km`, `slope_critical`): a deviation is allowed, the run reports the finding `locked_parameter_changed` and records the values.
- Fixed in the product, not settable (`trix.decimals`, `sampler.nan_policy`): code reads them with `fixed_parameter`, the configuration file rejects them. 
  TODO: state the source of `nan_policy` as defined (a reference `implementation` where TR6 does not prescribe it).
- Free (`dr_km`, `crs`, `interpolation_method`): defaults from the definition, set in the configuration file.

The contract lists the values of locked parameters. The run record lists every value used.

## Consequences

- `locked` means two things in practice (prescribed by a source, fixed by the implementation). A separate `fixed` attribute would make the contract say which. Deferred until it matters to readers of the contract.
