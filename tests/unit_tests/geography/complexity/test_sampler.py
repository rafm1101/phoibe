import numpy as np
import pytest

from phoibe.geography.complexity.sampler import FieldSampler
from phoibe.geography.complexity.sampler import RegularGridXYSampler


def test_field_sampler_is_abstract():
    with pytest.raises(TypeError):
        FieldSampler()


@pytest.mark.parametrize(
    "method, xs, ys, expected_z",
    [
        ("linear", [0.0, 10.0, 20.0], [0.0, 0.0, 0.0], [0.0, 5.0, 10.0]),
        ("linear", [0.5, 0.5, 31.5], [0.0, 0.5, 0.5], [0.25, 0.25, 15.75]),
        ("nearest", [0.0, 10.0, 20.0], [0.0, 0.0, 0.0], [0.0, 5.0, 10.0]),
        ("nearest", [0.5, 0.5, 0.6, 31.5], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.5, 15.5]),
    ],
)
def test_regulargridxysampler_returns_expected_values(planar_field, method, xs, ys, expected_z):
    sampler = RegularGridXYSampler(da=planar_field, method=method)
    z = sampler.sample(xs=xs, ys=ys)
    assert np.allclose(z, expected_z)


@pytest.mark.parametrize(
    "method, xs, ys, expected_match",
    [
        ("linear", [0.0, 10.0, 20.0, 200.0], [0.0, 0.0, 0.0], "Shapes do not agree"),
        ("linear", [0.0, 10.0, 20.0, 200.0], [0.0], "Shapes do not agree"),
        ("linear", [[0.0, 10.0, 20.0, 200.0]], [0.0, 10.0, 20.0, 200.0], "1-dimensional"),
    ],
)
def test_regulargridxysampler_rejects_unstackable_inputs(planar_field, method, xs, ys, expected_match):
    sampler = RegularGridXYSampler(da=planar_field, method=method)
    with pytest.raises(ValueError, match=expected_match):
        sampler.sample(xs=xs, ys=ys)


@pytest.mark.parametrize(
    "method, xs, ys, expected_level, expected_message",
    [
        (
            "linear",
            [0.0, 10.0, 20.0, 200.0],
            [0.0, 0.0, 0.0, 0.0],
            "INFO",
            "NaNs encountered during sampling from field",
        ),
    ],
)
def test_regulargridxysampler_logs_on_nan(planar_field, caplog, method, xs, ys, expected_level, expected_message):
    sampler = RegularGridXYSampler(da=planar_field, method=method)
    with caplog.at_level(expected_level):
        sampler.sample(xs=xs, ys=ys)
    assert expected_message in caplog.text
