# Phoibe

## Overview

Phoibe was one of the Titans and as such sister of Themis. Her name means _pure_, _bright_ and _prophet_. She received control of the Oracle of Delphi from Themis before gifting it to her grandson Apollo. She collects high-level tools for tasks that occur in the data science workflow. Typically, these high-level tools require additional dependencies which are not desired in `ergaleiothiki`.

## Structure

### Actual

- `artificial_data`: Collection of articial 2D-fields for demonstration.
- `geography`: 
  - `complexity`: Assessment of terrain complexity.
    - `rix`: Ruggedness index computation.
    - `sampler`: Sample from 2D-fields.

### Guidelines

## Observed code sweets and sours

### Setup

1. Atm, the package is not registered in some registry. Instead tag the package _from main_:

```bash
git tag -a <version> -m "<message>"
git push origin <version>
```
2. Side effect: There is currently no version resolution. Increase versions manually until further notice.
3. Side effect: **Poetry** uses a package that bugs this tagging. Ensure to configure:

```bash
poetry config system-git-client true
```

### Environment

<!--
1. `netcdf4`: The update from `1.7.2` to `1.7.3` made reading `netcdf` files more rigorous causing troubles reading own `netcdf` files. Solve this issue for allowing further updates.
-->

### Linting

1. `@typing.overload`:
   - `mypy` does not distinguish between `np.array` and `pd.DataFrame`, but `mypy` needs a subclassing hierarchy. Resolve (ignore) by `# type: ignore[overload-cannot-match]` at subsequent overloads.
   - `mypy` does not distingiush between `pd.Series` and `pd.DataFrame`. If static types are not that relevant, use `PANDA = typing.TypeVar("PANDA", pd.Series, pd.DataFrame)`.
   - `flake8` does not like ellipses on the same line as the function definition. Resolve (ignore) by `# noqa: E704`. Needs to follow `mypy` ignores.
2. Function signature not correctly identified by `mypy`. Resolve (ignore) by `# type: ignore [call-arg]`.
3. `docsig`: To skip checks on docstrings _temporarily_, use the directive `# docsig: disable`.
4. Slices: `flake8` mentions `E203 whitespace before ':'`, which `black` enforces according to PEP8.
   - Inline: `# noqa: E203`.
   - `pyproject.toml`: `ignore = E203`? Todo: Find general solution.

## Structure

### Package

1. Circular imports:
   - Functions that are used package internally in e.g. fundamental objects _and_ externally may easily create circular imports. Avoid by:
     - Creating private functions that carry the domain logic but rely on native Python objects in private modules.
     - Letting higher-level objects rely on the private functions.
     - Creating public functions that consume higher-level objects, and call the related private funtion.
   - Example: `perdix.LocationGCSFactory` and `atlas._earth._compute_square_around_center`.

### Objects