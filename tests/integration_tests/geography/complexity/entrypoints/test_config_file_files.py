import pytest

from phoibe.geography.complexity.rix.entrypoints.config_file import default_configuration, load_configuration
from phoibe.geography.complexity.rix.pipelines.errors import ConfigurationError


def write(tmp_path, text):
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_configuration_returns_the_parameters_of_a_yaml_file(product_spec, tmp_path):
    path = write(
        tmp_path,
        "project_name: Zürich\nparameters:\n  n_angles: 36\n  R_km: 5.0\n  dr_km: 0.05\n  slope_critical: 0.3\n  "
        'crs: "EPSG:2056"\n',
    )
    loaded = load_configuration(path, product_spec)
    assert (loaded.project_name, loaded.parameters) == (
        "Zürich",
        type(loaded.parameters)(36, 5.0, 0.05, 0.3, "EPSG:2056"),
    )


@pytest.mark.parametrize("text", ["", "# only a comment\n"], ids=["empty", "comment"])
def test_load_configuration_returns_the_defaults_given_a_file_without_content(product_spec, tmp_path, text):
    assert load_configuration(write(tmp_path, text), product_spec) == default_configuration(product_spec)


def test_load_configuration_returns_the_same_given_a_path_as_text_or_as_path(product_spec, tmp_path):
    path = write(tmp_path, "parameters:\n  n_angles: 36\n")
    assert load_configuration(str(path), product_spec) == load_configuration(path, product_spec)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("parameters: [", "Could not read the configuration"),
        ("- a\n- b\n", "must be a mapping"),
        ("parameters:\n  n_angles: three\n", "'n_angles' must be of type int"),
    ],
    ids=["invalid yaml", "not a mapping", "invalid value"],
)
def test_load_configuration_raises_given_an_invalid_file(product_spec, tmp_path, text, expected):
    with pytest.raises(ConfigurationError, match=expected):
        load_configuration(write(tmp_path, text), product_spec)


def test_load_configuration_raises_given_a_missing_file(product_spec, tmp_path):
    with pytest.raises(ConfigurationError, match="Could not read the configuration"):
        load_configuration(tmp_path / "missing.yaml", product_spec)
