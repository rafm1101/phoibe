# 0006 Publishing

Status: Accepted

## Context

Results are used in memory (notebook) and written to disk. Consumers rely on the contract, so an artifact that violates it must not count as published.

## Decision

`run()` returns the result in memory. `publish(result, target, profile)` is a separate, optional step:

1. verify that the result has all tables the profile needs, before anything is written;
2. the writer writes the data files and reads them back (columns, geometry types, record counts);
3. the contract check compares what was read back with the specification;
4. the manifest is written last.

A failure before step 4 leaves no manifest, so the artifact counts as unpublished; data files may remain. The writer is generic: it iterates the artifacts and layers the specification lists for the profile and the inputs present. The result holds its tables keyed by the table names of the specification.

## Consequences

- A new table or artifact needs the definition and a builder, not a change of the writer.
- TODO: Every writer implementation must pass the port test suite.
- Cleaning up partial data files after a failed publish is left to the adapter, if wanted.
