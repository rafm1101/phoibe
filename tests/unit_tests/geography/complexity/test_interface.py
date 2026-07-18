import dataclasses

import pytest

from phoibe.geography.complexity.rix.interface import Keys, _get_parameter


class TestGetParameter:
    """Unit tests for the nested-dict navigation helper."""

    def test_navigates_nested_dict(self):
        definition = {"a": {"b": {"c": 42}}}
        assert _get_parameter(definition, "a", "b", "c") == 42

    def test_raises_keyerror_on_missing_final_key_by_default(self):
        definition = {"a": {"b": {}}}
        with pytest.raises(KeyError, match="a -> b -> c"):
            _get_parameter(definition, "a", "b", "c")

    def test_returns_none_on_missing_final_key_when_not_strict(self):
        definition = {"a": {"b": {}}}
        assert _get_parameter(definition, "a", "b", "c", strict=False) is None

    def test_raises_keyerror_on_missing_intermediate_key_even_when_not_strict(self):
        definition = {"a": {}}
        with pytest.raises(KeyError, match="a -> b -> c"):
            _get_parameter(definition, "a", "b", "c", strict=False)

    def test_error_message_names_the_missing_key_and_full_path(self):
        definition = {"a": {"b": {}}}
        with pytest.raises(KeyError, match="missing expected key c"):
            _get_parameter(definition, "a", "b", "c")


class TestKeys:
    """Unit tests for the Keys dataclass."""

    def test_is_frozen(self):
        keys = Keys()
        with pytest.raises(dataclasses.FrozenInstanceError):
            keys.site_id = "something_else"

    def test_values_match_expected_snapshot(self):
        """Protect against unoticed key changes. Bump version upon any change."""
        expected = {
            "site_id": "site_id",
            "geometry": "geometry",
            "elevation": "elevation",
            "elevation_std": "elevation_std",
            "n_rays": "n_rays",
            "rix": "rix",
            "rix_std": "rix_std",
            "rix_min": "rix_min",
            "rix_max": "rix_max",
            "slope_critical": "slope_critical",
            "reference_id": "reference_id",
            "transferability": "transferability",
            "distance": "distance",
            "trix": "trix",
            "A": "A",
            "B": "B",
            "ruggedness": "ruggedness",
            "total_length_m": "total_length_m",
            "steep_length_m": "steep_length_m",
            "max_abs_slope": "max_abs_slope",
            "n_steep_segments": "n_steep_segments",
            "segment_id": "segment_id",
            "theta": "theta",
            "manifest": "manifest",
            "rix_summary": "rix_summary",
            "trix_table": "trix_table",
            "geopackage": "geopackage",
            "locations_site_layer": "locations_site",
            "locations_reference_layer": "locations_reference",
            "ruggedness_layer": "ruggedness",
            "trix_layer": "trix",
            "x": "x",
            "y": "y",
            "project_name": "project_name",
            "meta": "meta",
            "parameters": "parameters",
            "spatial_context": "spatial_context",
            "run": "run",
            "artifacts": "artifacts",
            "created_at": "created_at",
            "source_dem": "source_dem",
            "source_ray": "source_ray",
            "n_sites": "n_sites",
            "n_references": "n_references",
            "computed": "computed",
            "diagnostics": "diagnostics",
            "n_sites_with_nans": "n_sites_with_nans",
            "transferability_counts": "transferability_counts",
            "dem": "dem",
            "rays": "rays",
            "alignment": "internal_alignment",
            "crs_dem": "crs_dem",
            "crs_ray": "crs_ray",
            "extent_dem": "extent_dem",
            "resolution_dem": "resolution_dem",
            "message": "message",
            "nan_count": "nan_count",
        }
        actual = dataclasses.asdict(Keys())
        assert actual == expected

    def test_no_two_distinct_concepts_silently_share_a_string_by_accident(self):
        keys = Keys()
        assert keys.ruggedness == keys.ruggedness_layer == "ruggedness"
