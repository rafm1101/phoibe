# 0009 Structure of the product definition dictionary

---
type: ADR
id: "0009"
title: "Structure of the product definition dictionary"
status: active
date: 2026-10-23
superseded_by: null
---

## Context

Aligned with the previous `schema.py`. Base 1.1, new 2.0: same nesting, same keys. Only the marked entries are new or changed.

Legend: `<...>` = a name you choose · **req** = required · **opt** = optional · `[NEW]` / `[CHANGED]` = differs from the previous version

## 1. Structure

```
PRODUCT_DEFINITION
│
├─ name                                   str                                  req
├─ description                            str                                  req
├─ version                                str, dotted numbers e.g. "1.1"       req
│
├─ references                                                                  req
│   └─                     {: str, ...}  free-form entry
│
├─ parameters                                                                  req
│   └─ ray | slope | sampler
│       └─ <parameter_identifier>
│           ├─ description                str                                  req
│           ├─ value                      any   fixed value if locked,         req
│           │                                   else default; None = set by user
│           ├─ unit                       str | None                           opt
│           ├─ dtype                      "int" | "float" | "str" | "bool"     opt   inferred from value;
│           │                                                                        req if value is None   [CHANGED: crs]
│           ├─ range                      [lower, upper]                       opt
│           ├─ source                                           opt   req if locked
│           └─ locked                     bool, default False                  opt
│
├─ inputs                                                                      req
│   └─ dem | locations_site | locations_reference
│       ├─ description                    str                                  req
│       ├─ required                       bool                                 req
│       ├─ type                           str                                  opt
│       ├─ notes                          [str, ...]                           opt
│       └─ conditions                     [str, ...]  prose, not enforced      opt
│
├─ checks                                                                      req   [NEW]
│   └─ <check_name>
│       ├─ description                    str                                  req
│       ├─ severity                       "abort" | "warn"                     req
│       └─ parameters                     {: any, ...}                   opt   thresholds, tolerances
│
├─ artifacts                                                                   req
│   ├─ profiles                                                                req
│   │   └─                       [, ...]                    summary | full
│   ├─ filenames                                                               req
│   │   └─                      str  filename                        manifest | rix_table | trix_table | geopackage
│   └─ geopackage_layers                                                       req
│       └─                     str  layer name inside the file      locations_site | locations_reference |
│                                                                              ruggedness | trix
│
└─ schema                                                                      req
    │
    ├─ manifest                                                                req
    │   ├─ description                    str                                  req
    │   └─ keys                                                                req
    │       └─              (recursive node, see 2.)
    │
    ├─ rix_table                          (table, see 2.)                      req
    ├─ trix_table                         (table, see 2.)                      req   + requires_input   [NEW]
    │
    └─ geopackage                                                              req
        └─ layers                                                              req
            └─                 locations_site | locations_reference | ruggedness | trix
                ├─ description            str                                  req
                ├─ requires_input                                  opt   [NEW]
                ├─ columns                {: column, see 2.}           req   includes the geometry column
                ├─ geometry_column        str                                  req
                └─ geometry_type          "Point" | "MultiPoint" | "LineString" | "MultiLineString" |
                                          "Polygon" | "MultiPolygon" | None    req   None = attribute-only layer
```

## 2. Building blocks

```
table                                                       (schema.rix_table, schema.trix_table)
├─ description                              str             req
├─ requires_input                               opt   [NEW]  the file exists only if this input is present
└─ columns                                                  req
    └─                              (column)

column                                                      (also used in geopackage layers)
├─ name                                     str             opt   if given, equals its key
├─ description                              str             req
├─ unit                                     str | None      opt
├─ dtype                                    "int" | "float" | "str" | "bool"   opt
├─ range                                    [lower, upper]  opt
└─ values                                   {: str}   opt   meaning of coded values, e.g. {0: "Not representable."}

manifest node                                               (schema.manifest.keys., nested)
├─ name                                     str             opt   if given, equals its key
├─ description                              str             req
├─ keys                                                     opt   nested nodes of the same kind
│   └─                        (manifest node)
└─ values                                   {: str}  opt   meaning of enumerated values
```

## 3. What changes compared with the previous `schema.py`

| # | Change | Before | After |
|---|---|---|---|
| 1 | `dtype` for the `crs` parameter (its value is `None`, so the type cannot be inferred) | `"crs": {"description": ..., "unit": None, "value": None}` | same, plus `"dtype": "str"` |
| 2 | New top-level section `checks` | none | `"checks": {"nan_count": {"description": ..., "severity": "warn", "parameters": {"max_fraction": 0.0}}, ...}` |
| 3 | `requires_input` on the three entries that exist only with reference locations | none | `schema.trix_table`, `schema.geopackage.layers.locations_reference` and `schema.geopackage.layers.trix` each get `"requires_input": "locations_reference"` |
| 4 | Layer key made identical in both sections | `schema.geopackage.layers.trix_table` (but `artifacts.geopackage_layers.trix`) | `schema.geopackage.layers.trix` |
| 5 | Optional: put the range check from code into the definition | `n_angles` without range | `n_angles` with `"range": [1, 360]` |

Everything else stays exactly as it is, including the duplicated `trix_table` columns in `schema.trix_table` and `schema.geopackage.layers.trix`.

## 4. Relations that must hold

These cross references are validated once, in `ProductSpecification.__post_init__`:

1. The keys of `artifacts.filenames` and the keys of `schema` are the same set (`manifest`, `rix_table`, `trix_table`, `geopackage`).
2. Every entry in `artifacts.profiles.` is a key of `artifacts.filenames`, and every profile contains `manifest`.
3. The keys of `artifacts.geopackage_layers` and of `schema.geopackage.layers` are the same set.
4. Every `source` is a key of `references`, and every `requires_input` is a key of `inputs`.
5. Filenames are unique, and layer names are unique.
6. A locked parameter has a `value` and a `source`; a `value` lies within its `range` and matches its `dtype`.
7. The columns of `schema.trix_table` and of `schema.geopackage.layers.trix` (apart from the geometry column) are identical, so the duplication cannot drift.
