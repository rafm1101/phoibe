import cartopy.crs as ccrs
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import xarray

LAND_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "land_cmap", [(0.00, "#2e7d32"), (0.30, "#66bb6a"), (0.55, "#a1887f"), (0.75, "#9e9e9e"), (1.00, "#ffffff")]
)


def plot_raster(
    da: xarray.DataArray,
    title: str | None = None,
    figsize: tuple[int, int] | None = (13, 11),
    cmap: matplotlib.colors.Colormap = LAND_CMAP,
    clabel: str | None = "Elevation [m]",
):
    """Plot raster data with given CRS in colours.

    Parameters
    ----------
    da
        Raster data, e.g. some elevation map.
    title
        Title to be printed.
    figsize
        Size of the figure.
    cmap
        Colormap for data visualisation.
    clabel
        Label for the colorbar.

    Returns
    -------
    figure, ax
        Figure and ax object of the plot.

    Notes
    -----
    1. Plots dataarrays w/ or w/o CRS. A provided CRS is recognised only if it comes along with its EPSG code.
    2. Supported EPSG codes are 4326 and projected CRS. Unrecognised CRS are ignored.
    """
    plot_crs = _get_crs_from_dataarray(da=da)

    figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(1, 1, 1, projection=plot_crs)

    da_clean = _hide_nodata_points(da=da)
    vmin, vmax = _get_value_ranges(da=da_clean, pct_lower=1, pct_upper=99)
    cnorm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    X, Y = np.meshgrid(da.x.values, da.y.values)

    kwargs_pcolormesh = dict(cmap=cmap, shading="auto", norm=cnorm)
    if plot_crs is not None:
        kwargs_pcolormesh["transform"] = plot_crs

    mesh = ax.pcolormesh(X, Y, da_clean.values, **kwargs_pcolormesh)

    ax.grid(visible=True)
    if hasattr(ax, "gridlines"):
        ax.gridlines(draw_labels=True)
    if title is not None:
        ax.set_title(title)

    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7)
    if clabel is not None:
        cbar.set_label(clabel)

    plt.tight_layout()
    return figure, ax


def _get_value_ranges(da: xarray.DataArray, pct_lower=1, pct_upper=99) -> tuple[float, float]:
    """Determine the essential range of values appearing in `da`."""
    vmin = float(np.nanpercentile(da, pct_lower))
    vmax = float(np.nanpercentile(da, pct_upper))
    return vmin, vmax


def _get_crs_from_dataarray(da: xarray.DataArray) -> ccrs.CRS | None:
    """Determine the cartopy CRS related to `da`s crs.

    Notes
    -----
    1. Designed to run for the standard GCS (EPSG: 4326), or projected CRS (UTM, GK).
    2. Designed to recognise CRS via their EPSG code. Returns `None` if undetected.
    3. Function may behave unexpectedly.
    """
    if hasattr(da, "rio") and da.rio.crs is not None:
        epsg_code = int(da.rio.crs.to_authority()[1])
        if epsg_code == 4326:
            plot_crs = ccrs.PlateCarree()
        elif epsg_code is not None:
            plot_crs = ccrs.epsg(epsg_code)
        else:
            plot_crs = None
    else:
        plot_crs = None

    return plot_crs


def _hide_nodata_points(da: xarray.DataArray) -> xarray.DataArray:
    """Mask rio's nodata-encoded points in `da`. Pass in case it is no rio."""
    if hasattr(da, "rio") and da.rio.nodata is not None:
        mask_nodata = da != da.rio.nodata
        da_clean = da.where(mask_nodata)
    else:
        da_clean = da
    return da_clean
