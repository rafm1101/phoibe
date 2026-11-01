import pytest

from phoibe.geography.complexity.rix import main as main_module
from phoibe.geography.complexity.rix.computations.checks import NAN_COUNT
from phoibe.geography.complexity.rix.entrypoints import default_configuration, load_configuration
from phoibe.geography.complexity.rix.pipelines import errors


def test_build_analyzer_returns_an_analyzer_that_publishes_files_by_default(make_field, make_locations, tmp_path):
    analyzer = main_module.build_analyzer()  # the defaults of the specification: 72 rays of 3.5 km
    result = analyzer.run(make_field(a=0.01, half=4000), make_locations())
    analyzer.publish(result, tmp_path, "summary")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["rix_table.csv", "summary.yaml"]
    assert result.record.findings == ()  # the locked parameters are the ones of the specification


def test_build_analyzer_returns_an_analyzer_with_the_parameters_of_the_configuration(
    product_spec, make_field, make_locations
):
    configuration = load_configuration(
        {
            "parameters": {
                "n_angles": 4,
                "R_km": 0.5,
                "dr_km": 0.05,
                "slope_critical": 0.1,
                "interpolation_method": "nearest",
            }
        },
        product_spec,
    )
    result = main_module.build_analyzer(configuration).run(make_field(), make_locations())
    assert {
        name: result.record.parameters[name] for name in ("n_angles", "R_km", "slope_critical", "interpolation_method")
    } == {
        "n_angles": 4,
        "R_km": 0.5,
        "slope_critical": 0.1,
        "interpolation_method": "nearest",
    }
    assert result.ruggedness_roses.shape == (3, 4)


def test_build_analyzer_returns_an_analyzer_that_uses_the_writer_it_was_given(
    product_spec, make_writer, make_field, make_locations
):
    writer = make_writer()
    analyzer = main_module.build_analyzer(default_configuration(product_spec), writer=writer)
    analyzer.publish(analyzer.run(make_field(a=0.01, half=4000), make_locations()), "unused", "summary")
    assert writer.calls == [("artifacts", "summary"), ("manifest",)]


def test_build_analyzer_returns_an_analyzer_that_uses_the_specification_it_was_given(
    make_spec, make_field, make_locations
):
    spec = make_spec(severities={NAN_COUNT: "abort"}, check_parameters={NAN_COUNT: {"max_fraction": 0.0}})
    analyzer = main_module.build_analyzer(spec=spec)
    with pytest.raises(errors.CheckAborted, match=NAN_COUNT):
        analyzer.run(make_field(a=0.01, half=4000), make_locations(("edge",), (3900,), (0,)))  # the rays leave the DEM
