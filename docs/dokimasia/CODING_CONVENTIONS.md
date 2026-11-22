---
type: Conventions
title: "Coding conventions for phoibe's own Python code"
date: 2026-11-21
revision: 1
---

# Coding conventions

These rules apply to new code and tests, and to code that is rewritten, in `phoibe/`, `scripts/` and spikes. Existing modules are not retrofitted. They are listed in `pyproject.toml` (ruff per-file ignores and mypy overrides) and leave the list when they are rewritten. The list shrinks and never grows.

The tools enforce what they can (`pyproject.toml`, `.pre-commit-config.yaml`). This document covers what they cannot.

Identifiers, comments, docstrings and documents are in American English (`standardize`, `catalog`, `behavior`, `judgment`).

## 1. Tools

| Tool | Role | Setting |
| :--- | :--- | :--- |
| `ruff format` | Formatter | Line length 120, double quotes. The only formatter; Black is not used. |
| `ruff check` | Linter | Rule groups E, W, F, I, B, UP, SIM, C4, NPY, RUF, N, PTH, BLE, PERF, C90 (complexity at most 12), D (docstrings, numpy convention), G (logging format), FBT (boolean arguments). |
| `mypy` | Type checker | `strict = true`, with `pandas-stubs`. |
| `vulture` | Dead code | Confidence 60 (the default) with a whitelist file for names used only from outside the package. At 80, vulture finds only unused imports and arguments, not unused functions, methods or classes. |
| `docsig` | Docstrings against signatures | Defaults: every public function documents all parameters and its return value; private functions are not checked. |

Code is done when all five pass. Config-level per-file ignores are allowed for tests, notebooks, `__init__.py` and the legacy list. An inline suppression is the exception: it names the rule, and the line before it says why (the suppression itself has to stay on the line it suppresses).

Tests follow the same tools with two relaxations: docstrings are not required (`D`), and test functions and fixtures need no annotations.

## 2. Names

### 2.1 Functions start with a verb

The two tables are a recommendation for the common cases. Where none fits, prefer a plain English verb that says what happens over a vague one (`handle`, `process`, `do` and `manage` are not allowed).

| Verb | Use for |
| :--- | :--- |
| `read_` / `write_` | Bytes or text from or to a medium (file, network). |
| `load_` / `save_` | Read or write plus parsing or serializing. Returns an object / None. |
| `parse_` | Text into a structure, no I/O. Raises on invalid input. |
| `format_` | A structure into text, no I/O. |
| `compute_` | A pure calculation. Returns the value. |
| `build_` | Assemble an in-memory object from parts. |
| `create_` / `delete_` | Make or remove an external resource (file, folder, process). |
| `find_` | Search that may fail. Returns the item or None. |
| `get_` | Lookup that must succeed. Returns the item, raises otherwise. |
| `list_` | Enumerate. Returns a collection. |
| `select_` | Choose a subset. |
| `apply_` | Apply a change to a state. Returns the new state. |
| `run_` | Execute a process or a pipeline step. |

The domain verbs of the package:

| Verb | Use for |
| :--- | :--- |
| `profile_` | Run the measurements a layer names on a dataset. Returns a profile. Makes no judgment. |
| `measure_` | Run one measurement. Total: for any input it returns a measurement or a typed "not evaluable". |
| `judge_` | Evaluate a profile against a contract. Returns verdicts and the gate outcome. Never reads the data. |
| `validate_` | Profile and judge in one step. Returns what `judge_` returns. |
| `standardize_` | A recorded dataset-to-dataset transform (rename to catalog names, localize time to UTC). Returns the new dataset with its provenance. |
| `attest_` | Record a gate outcome with its lineage. Returns an attestation. |
| `advance_` | Move a dataset to the next layer, applying a standardizing transform. Returns the next dataset. |

- Predicates are named `is_`, `has_` or `can_` and return `bool`.
- A method named for the role of its object may use the bare verb (`loader.load()`, `detector.detect(frame)`, `rule.measure(frame)`). A free function takes the prefix and a noun.
- Alternate constructors and converters are `from_` and `to_` (`LayerReport.from_yaml`, `report.to_dict`).
- Exceptions to the verb rule: `main`, test functions, properties (nouns, because they read as attributes), and dunder methods.

### 2.2 Variables, parameters, attributes

- Descriptive nouns: `task_ids`, `allowed_files_of`, `estimated_tokens`. Collections are plural; a mapping is named for what it maps (`package_of_task`, `new_files_of`).
- Units go into the name, with one suffix per unit from the list below. Extend the list instead of inventing a variant.

  | Quantity | Suffix | Example |
  | :--- | :--- | :--- |
  | Time | `_s`, `_ms` (milliseconds), `_min`, `_h` | `timeout_s`, `interval_min` |
  | Frequency, rotation | `_hz`, `_rpm` | `frequency_hz` |
  | Power, energy | `_kw`, `_mw` (megawatt), `_kwh` | `rated_power_kw` |
  | Speed | `_mps` | `cut_in_speed_mps` |
  | Angle, temperature | `_deg`, `_degc` | `yaw_deg` |
  | Size | `_bytes` (no `_kb` or `_mb`) | `size_bytes` |

  Other units are written out (`budget_tokens`, `changed_lines`). Signal names in the domain catalog are data, not identifiers; the catalog defines their units.
- Booleans read as statements: `is_passing`, `has_report`. Boolean parameters are keyword-only (after `*`); the `FBT` rules enforce it.
- No abbreviations, with these exceptions:
  - single letters in comprehensions and short loops over an obvious range;
  - single-letter symbols of a distribution or a model (the Weibull scale `A` and shape `k`), defined in the docstring where they are introduced;
  - the aliases `np`, `pd` and `xr`;
  - the project's domain terms and common acronyms: `id`, `url`, `http`, `json`, `yaml`, `csv`, `sha`, `adr`, `scada`, `utc`, `wtg`, `odcs`, `odps`.

  Abbreviations such as `ctx`, `cfg` or `tmp` are spelled out.
- Constants are `UPPER_SNAKE_CASE` at module level; a magic number gets a named constant, and its source goes into a comment on the line before it.
- Module names start with a verb when the module is a command (`lint_plan.py`), and are nouns when it is a library (`bindings.py`).

### 2.3 Imports

- Import modules, not names: `import dataclasses` and `@dataclasses.dataclass`; `from phoibe import contract` and `contract.Layer`. A use then shows the module it comes from.
- Kept as from-imports: `from __future__ import annotations`, `from collections.abc import ...`, `from typing import ...`, and `from phoibe import ` or `from tests import `, which import a module, not a name.
- A local name must not equal the name of an imported module (a fixture called `run_log` hides the module `run_log`).

## 3. Types and data

- Every function has full type hints. `Any` only for decoded, not yet validated data, or at the boundary to a library without types, behind a named alias (`JsonObject`).
- Data that phoibe owns (measurements, verdicts, contracts, attestations) are dataclasses. `frozen=True` is chosen per class where immutability is part of the concept: a value that is shared, compared or used as a key. A mutable dataclass is fine for a record that is built up step by step or that wraps a mutable object such as a DataFrame.
- No function mutates its arguments, DataFrames included. It returns a new object.
- Untrusted input (a hand-edited contract, a file) stays decoded until one function has walked it and collected every problem. Only then is it converted into dataclasses. A converter that fails on the first error would hide the rest.
- A failed criterion, a missing signal or an unparsable timestamp is a value of a closed type (a verdict, "not evaluable"), never a bare `None` or a sentinel string.
- Closed sets of values are `enum.StrEnum` or a `Literal`, not bare strings, inside phoibe. At the boundary the string form is kept.
- `TypedDict` only for dictionaries that pass through phoibe unchanged and are owned by someone else (an API payload).
- Ports (the interfaces `core` defines and `adapters` implement) are `typing.Protocol`. An abstract base class is used only when subclasses share an implementation.
- Keys in files that phoibe defines are `snake_case`. The vocabulary of an external standard (ODCS) appears only in the export adapter.

## 4. Structure

- A module has no side effects on import: it creates no file or folder, configures no logging, and registers nothing. A command has `def main() -> None` and `if __name__ == "__main__": main()`; arguments are parsed with `argparse`.
- Paths use `pathlib.Path`; text files are read and written with `encoding="utf-8"`.
- No silent failures. Exceptions are specific, and there is no bare `except`. Every `except` either re-raises or turns the failure into a typed value that is logged and carries the exception type and message. A failed criterion or a signal that cannot be measured is a value (a verdict, "not evaluable"), and "not evaluable" never counts as a pass. Invalid arguments and programming errors raise.
- `except Exception` appears in exactly one place: the function that runs a measurement and contains a failure as a typed error result. That one line carries the named suppression for `BLE001` with its reason.
- A function does one thing; complexity at most 12 (C90). Split by rule or step, and keep the order of results stable when it is visible to a model or a person.
- A name without a leading underscore is public. `expose_members` ignores symbols with a leading underscore, so a name that other modules of the package need but users do not gets a leading underscore or lives in an underscore-prefixed module. It may also expose names that a module imports; the convention does not rely on that, and `test_public_api` per package (listing the exposed names) keeps growth deliberate.
- Packages import in one direction, as ADR-07 sets out (`core` imports nothing of ours; `compose` imports all). A plain pytest check over the imports (`ast`) enforces it. `core` performs no file I/O.
- Logging: a module that logs has `logger = logging.getLogger(__name__)`; a module that does not log has none. The package attaches only a `NullHandler`. Log calls use lazy formatting (`logger.warning("gap of %s", gap)`), never an f-string.

## 5. Pandas, NumPy and time

- No `inplace=True`. Use `.to_numpy()` rather than `.values`, and copy explicitly when a view would be read after its source changes.
- Parse timestamps with an explicit `format=` and `utc=True`. `errors="coerce"` is used only where the coercion is itself recorded as a measurement. Inside `core`, timestamps are timezone-aware UTC.
- Randomness comes from a `np.random.Generator` or a seed that is passed in. No global state, no legacy `np.random.*` functions (the `NPY` rules check).

## 6. Docstrings and comments

- Docstrings follow the numpy style, without types: the types are in the signature. The summary line is in the imperative ("Measure U+2026", "Judge U+2026").
- Public functions, methods and classes have a full docstring: a summary, what the function does beyond the summary where that matters, and the sections `Parameters`, `Returns` and, where it raises on purpose, `Raises`. Each parameter gets a sentence that says what it means for the caller (its role, its unit, what `None` means), not its type. A domain symbol (a single-letter parameter, a unit such as MW) is defined where it is introduced.
- A class docstring describes the class. Each field of a public dataclass gets an attribute docstring, a string literal on the line after the field (PEP 257 calls these attribute docstrings; Sphinx, mkdocstrings and Pylance show them). The numpy `Attributes` section is not used, so a field and its description cannot drift apart.
- Private functions may have a one-line docstring.
- Comments are avoided. A name or a smaller function says it better. Where a comment is needed (a source, a reason that the code cannot show), it stands on its own line before the code it explains, never at the end of a line.
- Section dividers are not used; the order of the module and its names give the structure. Longer explanations go to the module docstring or the ADR.

## 7. Tests

- Names of plain test functions are `test_` for the plain case, then `[_given_]` and `[_]` as needed, so that the name shows what the test uses: `test_count_bytes`, `test_count_bytes_given_empty_file`, `test_count_bytes_given_empty_file_returns_zero`.
- In a test class the class name carries the subject and the methods are `test_`: `class TestFreeze` with `test_labels_the_frozen_segment`.
- A detector or rule is tested with a valid case and a mutation per rule, and with a control case it must not fire on. The mutation is a messup step of the generator, and its labels are the ground truth; the control is the clean data. The recall and precision a rule must reach are set per rule.
- A rewrite of existing logic is checked for equal results on recorded inputs before the old version is removed. Recorded inputs are seeded synthetic data. The only allowed differences are the listed defects, each with its characterization test.
- Characterization tests are temporary. They state the desired behavior of a known defect, carry `xfail(strict=True)` with the defect number and the marker `characterization`, live in `tests/characterization/`, are named by the behavior they demand, and are removed with the fix.
- Tests are deterministic: seeded data, no network, no real SCADA data, and temporary files through `tmp_path`.
