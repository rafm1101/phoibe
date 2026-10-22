# 0001 Architecture and package structure

Status: Accepted

## Context

The product is computation-heavy. Real boundaries exist only where data comes in (elevation field, locations) and goes out (files). The scientific rules must be testable without files or libraries for them.

## Decision

Hexagonal architecture in proportion, layers:

- `computations`
- `pipeline` (use case)
- `adapters`
- `entrypoints`
- `main.py` as composition root.

A port exists only at a real boundary; today that is the result writer. The field input has no port (ADR 0002).

Inside `pipelines`, packages are cut by job:

- `product` (what the product is)
- `checks` (findings and inputs)
- `results` (tables, run record, result)
- `publishing` (manifest, contract, contract check)

Dependencies run downwards only: analyzer → publishing → ports → results → checks → product.

## Consequences

- Layers and the order inside `pipelines` are enforced with import-linter; imports used only for type checking are excluded.
- A module that does not fit one job is a design smell.
- `__init__.py` files are explicit. No autoloading: it would load optional dependencies on import and hide imports from the linter and the type checker.
