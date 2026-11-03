"""Reference outputs of the regression test: a synthetic dataset and an independent oracle.

The oracle does NOT import the package. It recomputes RIX and TRIX from the formulas of FGW TR6 Rev. 12 with its own
bilinear interpolation, so the frozen numbers do not depend on the code under test. The central computation did not
change between 1.1 and 2.0, therefore the same numbers are the reference of both.

    python make_reference.py write            rewrite the reference files next to this script
    python make_reference.py export DIR       write the dataset as files (DEM.tif, sites.gpkg, references.gpkg),
                                              e.g. to run an installed 1.1 on it and compare with the reference

Semantics fixed here, as in the changelog 2.0.0 and ADR 0004, 0005, 0010:
- slope = difference of the elevations / step; a segment is steep if |slope| > slope_critical; NaN is never steep,
  its length stays in the total (nan_policy "mask"); a ray outside the DEM gives NaN
- RIX of a ray: steep length / total length; of a location: mean over the rays
- TRIX = 0.9 * 50 * (RIX_site + RIX_reference) + 0.1 * |elevation difference|; A = max(8.5 - 0.087 TRIX, 1.5),
  B = max(15 - 0.14 TRIX, 3), both rounded to one decimal BEFORE the comparison; transferability 2 up to A, 1 up to B,
  else 0, and 0 if anything is NaN
- distances in the x/y plane (a z coordinate is ignored), in km
"""

import math
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

HERE = pathlib.Path(__file__).parent
PARAMETERS = {"n_angles": 72, "R_km": 3.5, "dr_km": 0.05, "slope_critical": 0.033, "decimals": 1}
HALF, STEP = 9000.0, 50.0
CRS = "EPSG:32633"

SITES = {
    "S1": (0.0, 0.0, None),
    "S2": (1500.0, -800.0, 999.0),
    "S3": (-4500.0, 300.0, None),
    "S4": (6500.0, 500.0, None),  # on the plateau
    "S5": (8800.0, 0.0, None),  # 200 m from the edge: the rays to the east leave the DEM
}
REFERENCES = {
    "R1": (200.0, 1000.0, None),
    "R2": (3000.0, 200.0, None),
    "R3": (-3500.0, -1200.0, None),
    "R4": (8000.0, -500.0, None),
    "R5": (
        -7000.0,
        0.0,
        None,
    ),  # on a 600 m mountain: with the hills, TRIX gets so high that both limits sit on their floors
}
RIX_COLUMNS = ["site_id", "elevation", "rix", "rix_std", "rix_min", "rix_max", "n_rays", "slope_critical"]
TRIX_COLUMNS = ["site_id", "reference_id", "transferability", "distance", "trix", "A", "B"]


def terrain(x, y):
    """Gentle hills (slopes up to about 0.1), a flat plateau east of x = 5000 m and a 600 m mountain at x = -7000 m."""
    hills = 40.0 * np.sin(x / 900.0) * np.cos(y / 700.0) + 15.0 * np.sin((x + y) / 300.0)
    weight = np.clip((x - 4000.0) / 1000.0, 0.0, 1.0)
    mountain = 600.0 * np.exp(-((x + 7000.0) ** 2 + y**2) / (2 * 600.0**2))
    return (1.0 - weight) * hills + weight * 100.0 + mountain


def build_dem():
    axis = np.arange(-HALF, HALF + STEP, STEP)
    xx, yy = np.meshgrid(axis, axis)
    return axis, axis.copy(), terrain(xx, yy)  # z[row of y, column of x]


def bilinear(x, y, z, px, py):
    """Elevation at the points, NaN outside the DEM."""
    inside = (px >= x[0]) & (px <= x[-1]) & (py >= y[0]) & (py <= y[-1])
    i = np.clip(np.floor((px - x[0]) / STEP).astype(int), 0, len(x) - 2)
    j = np.clip(np.floor((py - y[0]) / STEP).astype(int), 0, len(y) - 2)
    tx, ty = (px - x[i]) / STEP, (py - y[j]) / STEP
    value = (
        z[j, i] * (1 - tx) * (1 - ty)
        + z[j, i + 1] * tx * (1 - ty)
        + z[j + 1, i] * (1 - tx) * ty
        + z[j + 1, i + 1] * tx * ty
    )
    return np.where(inside, value, np.nan)


def assess_location(dem, x0, y0, parameters=PARAMETERS):
    """RIX summary of one location: elevation, rix, std, min, max, number of NaN samples."""
    x, y, z = dem
    n, dr = parameters["n_angles"], parameters["dr_km"] * 1000.0
    steps = math.floor(parameters["R_km"] / parameters["dr_km"] + 1e-9)
    theta = np.deg2rad(np.arange(n) * 360.0 / n)  # 0 = north, clockwise
    r = np.arange(steps + 1) * dr
    profile = bilinear(x, y, z, x0 + np.sin(theta)[:, None] * r[None, :], y0 + np.cos(theta)[:, None] * r[None, :])
    with np.errstate(invalid="ignore"):
        steep = np.abs(np.diff(profile, axis=1) / dr) > parameters["slope_critical"]  # NaN is never steep
    ruggedness = steep.sum(axis=1) / steps  # equal segments: the share of steep length
    return {
        "elevation": float(np.mean(profile[:, 0])),
        "rix": float(np.mean(ruggedness)),
        "rix_std": float(np.std(ruggedness)),
        "rix_min": float(np.min(ruggedness)),
        "rix_max": float(np.max(ruggedness)),
        "n_rays": n,
        "slope_critical": parameters["slope_critical"],
        "nan_count": int(np.isnan(profile).sum()),
        "with_nans": bool(np.isnan(profile).any()),
    }


def pair(site, reference, decimals=PARAMETERS["decimals"]):
    trix = 0.9 * 50 * (site["rix"] + reference["rix"]) + 0.1 * abs(site["elevation"] - reference["elevation"])
    a = round(max(8.5 - 0.087 * trix, 1.5), decimals)
    b = round(max(15.0 - 0.14 * trix, 3.0), decimals)
    return trix, a, b


def compute():
    dem = build_dem()
    sites = {name: assess_location(dem, x, y) for name, (x, y, _) in SITES.items()}
    references = {name: assess_location(dem, x, y) for name, (x, y, _) in REFERENCES.items()}
    rix_table = pd.DataFrame(
        [{"site_id": name, **{k: v for k, v in s.items() if k in RIX_COLUMNS}} for name, s in sites.items()]
    )[RIX_COLUMNS]
    rows = []
    for name, (x, y, _) in SITES.items():
        for reference_name, (xr, yr, _) in REFERENCES.items():
            trix, a, b = pair(sites[name], references[reference_name])
            distance = math.hypot(x - xr, y - yr) / 1000.0  # the x/y plane, no z
            nan = any(math.isnan(v) for v in (trix, a, b, distance))
            rows.append(
                {
                    "site_id": name,
                    "reference_id": reference_name,
                    "transferability": 0 if nan else (2 if distance <= a else 1 if distance <= b else 0),
                    "distance": distance,
                    "trix": trix,
                    "A": a,
                    "B": b,
                }
            )
    trix_table = pd.DataFrame(rows)[TRIX_COLUMNS]
    counts = trix_table["transferability"].value_counts().sort_index()
    diagnostics = {
        "nan_count": {
            "locations_site": sum(s["nan_count"] for s in sites.values()),
            "locations_reference": sum(s["nan_count"] for s in references.values()),
        },
        "n_locations_with_nans": {
            "locations_site": sum(s["with_nans"] for s in sites.values()),
            "locations_reference": sum(s["with_nans"] for s in references.values()),
        },
        "transferability_counts": {int(k): int(v) for k, v in counts.items()},
    }
    return rix_table, trix_table, diagnostics


def write_reference(directory=HERE):
    rix_table, trix_table, diagnostics = compute()
    shortest = lambda value: repr(float(value))  # noqa: E731
    rix_table.to_csv(directory / "rix_table.csv", index=False, float_format=shortest)
    trix_table.to_csv(directory / "trix_table.csv", index=False, float_format=shortest)
    (directory / "diagnostics.yaml").write_text(yaml.safe_dump(diagnostics, sort_keys=False), encoding="utf-8")


def export(directory):
    """The dataset as files: DEM.tif (float64, y decreasing with the row), sites.gpkg and references.gpkg."""
    import geopandas as gpd
    import rioxarray  # noqa: F401
    import shapely.geometry as sg
    import xarray

    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    x, y, z = build_dem()
    da = xarray.DataArray(z[::-1], coords={"y": y[::-1], "x": x}, dims=("y", "x")).rio.write_crs(CRS)
    da.rio.to_raster(directory / "DEM.tif")
    for name, locations in (("sites", SITES), ("references", REFERENCES)):
        points = [sg.Point(px, py) if pz is None else sg.Point(px, py, pz) for px, py, pz in locations.values()]
        gpd.GeoDataFrame({"site_id": list(locations)}, geometry=points, crs=CRS).to_file(
            directory / f"{name}.gpkg", driver="GPKG"
        )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "write"
    if command == "write":
        write_reference()
    elif command == "export" and len(sys.argv) == 3:
        export(sys.argv[2])
    else:
        sys.exit(__doc__)
