import numpy as np
import pandas as pd
import pytest

from phoibe.geography.complexity.rix.entrypoints import notebook
from phoibe.geography.complexity.rix.pipelines.results import tables

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

STEEP = [10, 60, 110]  # slope 0.5 over 100 m cells
FLAT = [10, 10, 10]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_ray_metrics_returns_one_row_per_ray_with_the_metrics_in_order(make_radial):
    metrics = notebook.ray_metrics(make_radial(STEEP, FLAT))
    assert list(metrics.columns) == [
        "theta",
        "ruggedness",
        "total_length_m",
        "steep_length_m",
        "max_abs_slope",
        "n_steep_segments",
    ]
    assert metrics.iloc[0].to_dict() == pytest.approx(
        {
            "theta": 0.0,
            "ruggedness": 1.0,
            "total_length_m": 200.0,
            "steep_length_m": 200.0,
            "max_abs_slope": 0.5,
            "n_steep_segments": 1,
        }
    )
    assert metrics.iloc[1].to_dict() == pytest.approx(
        {
            "theta": 10.0,
            "ruggedness": 0.0,
            "total_length_m": 200.0,
            "steep_length_m": 0.0,
            "max_abs_slope": 0.0,
            "n_steep_segments": 0,
        }
    )


def test_steep_segments_geodataframe_returns_the_table_of_the_product_for_one_location(product_spec, make_radial):
    radial = make_radial([0, 50, 50, 100, 150])
    result = notebook.steep_segments_geodataframe(product_spec, radial, location_id="s1")
    expected = tables.steep_segments(product_spec, {"s1": radial})
    assert list(result.index) == ["s1", "s1"]
    pd.testing.assert_frame_equal(pd.DataFrame(result), pd.DataFrame(expected))
    assert list(notebook.steep_segments_geodataframe(product_spec, radial).index) == ["location"] * 2


def test_plot_polar_returns_the_figure_and_the_polar_axes_with_the_circle_closed(make_radial):
    radial = make_radial(STEEP, FLAT, FLAT)
    figure, ax = notebook.plot_polar(radial, label="s1", slope_critical=0.1)
    line = ax.lines[0]
    np.testing.assert_allclose(line.get_xdata(), np.deg2rad([0, 10, 20, 0]))  # the first point is repeated
    np.testing.assert_allclose(line.get_ydata(), [1.0, 0.0, 0.0, 1.0])
    assert (line.get_label(), ax.name, figure is ax.figure) == ("s1", "polar", True)
    assert (ax.get_theta_direction(), ax.get_theta_offset()) == (-1, pytest.approx(np.pi / 2))  # north, clockwise
    assert (ax.get_title(), ax.get_ylabel()) == ("Directional RIX (threshold=0.100)", "RIX")


def test_plot_polar_returns_the_figure_of_the_axes_it_was_given(make_radial):
    figure, ax = plt.subplots(subplot_kw={"projection": "polar"})
    assert notebook.plot_polar(make_radial(STEEP, FLAT), slope_critical=0.1, ax=ax) == (figure, ax)
    assert len(ax.lines) == 1


def test_plot_polar_returns_the_values_of_a_series_over_its_angles():
    series = pd.Series([0.1, 0.2, 0.3], index=[0.0, 10.0, 20.0])
    _, ax = notebook.plot_polar(series, slope_critical=0.1)
    np.testing.assert_allclose(ax.lines[0].get_xdata(), np.deg2rad([0, 10, 20, 0]))
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.1, 0.2, 0.3, 0.1])


def test_plot_polar_returns_the_circle_closed_given_a_series_whose_index_does_not_contain_zero():
    _, ax = notebook.plot_polar(pd.Series([0.1, 0.2, 0.3], index=[10.0, 20.0, 30.0]), slope_critical=0.1)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.1, 0.2, 0.3, 0.1])


def test_plot_polar_returns_a_figure_given_no_slope_critical(make_radial):
    figure, ax = notebook.plot_polar(make_radial(STEEP, FLAT))
    assert figure is ax.figure


def test_plot_polar_raises_given_anything_but_a_radial_result_or_a_series():
    with pytest.raises(ValueError, match=r"either `RadialRuggedness` or `pd.Series`"):
        notebook.plot_polar([0.1, 0.2], slope_critical=0.1)


def test_plot_polar_raises_given_no_matplotlib(make_radial, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.pyplot", None)  # an import then raises ImportError
    with pytest.raises(ImportError, match="requires matplotlib"):
        notebook.plot_polar(make_radial(STEEP, FLAT), slope_critical=0.1)
