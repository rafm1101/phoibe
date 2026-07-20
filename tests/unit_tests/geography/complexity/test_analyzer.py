import pytest
import yaml

from phoibe.geography.complexity.rix.analyzer import _epsg_int, _load_config, _validate_config
from phoibe.geography.complexity.rix.config import ANALYZER_DEFAULTS


def _valid_parameters(**overrides):
    parameters = {"n_angles": 36, "R_km": 5.0, "dr_km": 0.05, "slope_critical": 0.3}
    parameters.update(overrides)
    return {"parameters": parameters}


class TestValidateConfig:
    def test_accept_valid_config(self):
        _validate_config(_valid_parameters())

    def test_raise_given_required_key_missing(self):
        config = _valid_parameters()
        del config["parameters"]["slope_critical"]
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_config(config)

    def test_raise_given_dr_km_not_smaller_than_r_km(self):
        with pytest.raises(ValueError, match="dr_km must be smaller than R_km"):
            _validate_config(_valid_parameters(dr_km=5.0, R_km=5.0))

    @pytest.mark.parametrize("n_angles", [0, 361, -1])
    def test_raise_given_n_angles_out_of_range(self, n_angles):
        with pytest.raises(ValueError, match="n_angles must be between 1 and 360"):
            _validate_config(_valid_parameters(n_angles=n_angles))

    @pytest.mark.parametrize("n_angles", [1, 360])
    def test_accept_n_angles_boundary_values(self, n_angles):
        _validate_config(_valid_parameters(n_angles=n_angles))  # must not raise

    def test_raise_given_slope_critical_not_positive(self):
        with pytest.raises(ValueError, match="slope_critical must be positive"):
            _validate_config(_valid_parameters(slope_critical=0.0))


class TestLoadConfig:
    def test_fill_missing_parameters_from_analyzer_defaults(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump({"parameters": {"n_angles": 8}}))

        config = _load_config(path)

        assert config["parameters"]["n_angles"] == 8
        assert config["parameters"]["R_km"] == ANALYZER_DEFAULTS["parameters"]["R_km"]
        assert config["parameters"]["dr_km"] == ANALYZER_DEFAULTS["parameters"]["dr_km"]
        assert config["parameters"]["slope_critical"] == ANALYZER_DEFAULTS["parameters"]["slope_critical"]

    def test_raise_on_unknown_parameter(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump({"parameters": {"not_a_real_param": 1}}))

        with pytest.raises(ValueError, match="unknown parameter"):
            _load_config(path)

    def test_validate_merged_config_not_just_overrides(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump({"parameters": {"dr_km": 999.0}}))

        with pytest.raises(ValueError, match="dr_km must be smaller than R_km"):
            _load_config(path)


class TestEpsgInt:
    def test_extracts_code_from_standard_format(self):
        assert _epsg_int("EPSG:4326") == 4326

    def test_extracts_code_case_insensitively(self):
        assert _epsg_int("epsg:32633") == 32633

    def test_returns_none_for_non_epsg_prefix(self):
        assert _epsg_int("OGC:CRS84") is None

    def test_returns_none_for_malformed_string(self):
        assert _epsg_int("not-a-crs-string") is None

    def test_returns_none_for_non_string_input(self):
        assert _epsg_int(None) is None
