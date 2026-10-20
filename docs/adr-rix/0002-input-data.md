# 0002 Input data: the field is given, not fetched

Status: Accepted

## Context

The real elevation data (CopDem) is delivered in 1°x1° patches. Extracting the region around the locations is a separate project.

## Decision

The use case takes a `Field` (array with CRS, extent and resolution) created by the caller. The DEM reader adapter reads one raster completely, turns nodata into NaN and fills the `Field`. There is no input port and no windowing.

## Consequences

- The DEM must fit into memory.
- The caller supplies a region that covers all rays. Rays leaving the field produce NaN, which the NaN check reports.
- Revisit when the partition extraction is integrated: a port with a window signature (bounding box) becomes necessary then, so that only the corridor around the rays is loaded.
