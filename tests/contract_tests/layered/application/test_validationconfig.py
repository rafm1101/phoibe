from pathlib import Path

import pytest

from phoibe.layered.application.config import ValidationConfig


class TestValidationConfig:

    @pytest.fixture
    def config_file(self, tmp_path):
        """Create sample config"""
        config = tmp_path / "config.yaml"
        config.write_text(
            """
layer_name: raw
variable_patterns:
  timestamp:
    - zeitstempel
  power:
    - leistung
rules:
  - name: rule1
    params:
      param1: value1
"""
        )
        return config

    def test_from_yaml_returns_config(self, config_file):
        config = ValidationConfig.from_yaml(config_file)

        assert isinstance(config, ValidationConfig)

    def test_from_yaml_accepts_string_path(self, config_file):
        config = ValidationConfig.from_yaml(str(config_file))

        assert isinstance(config, ValidationConfig)

    def test_from_yaml_accepts_path_object(self, config_file):
        config = ValidationConfig.from_yaml(Path(config_file))

        assert isinstance(config, ValidationConfig)

    def test_loads_layer_name(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        assert config.layer_name == "raw"

    def test_loads_variable_patterns(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        assert "timestamp" in config.variable_patterns
        assert config.variable_patterns["timestamp"] == ["zeitstempel"]

    def test_loads_rules(self, config_file):
        config = ValidationConfig.from_yaml(config_file)
        assert len(config.rules) == 1
        assert config.rules[0]["name"] == "rule1"
        assert config.rules[0]["params"]["param1"] == "value1"

    def test_raises_for_missing_file(self, tmp_path):
        missing = tmp_path / "missing.yaml"
        with pytest.raises(FileNotFoundError):
            ValidationConfig.from_yaml(missing)

    def test_raises_for_missing_layer_name(self, tmp_path):
        config = tmp_path / "bad.yaml"
        config.write_text("variable_patterns: {}")
        with pytest.raises(ValueError, match="layer_name"):
            ValidationConfig.from_yaml(config)

    def test_handles_empty_variable_patterns(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            """
layer_name: raw
variable_patterns: {}
rules: []
"""
        )
        validation_config = ValidationConfig.from_yaml(config)
        assert validation_config.variable_patterns == {}

    def test_handles_empty_rules(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            """
layer_name: raw
variable_patterns: {}
rules: []
"""
        )
        validation_config = ValidationConfig.from_yaml(config)
        assert validation_config.rules == []
