import typing

from phoibe.geography.complexity.rix.config import (
    ANALYZER_DEFAULTS,
    COPDEM_METADATA,
    INTERPOLATION_METHODS,
    WRITER_DEFAULTS,
)
from phoibe.geography.complexity.rix.schema import PRODUCT_DEFINITION_TRIX


def test_analyzer_defaults_match_expected_snapshot():
    expected = {
        "name": "T-RIX assessment",
        "version": "1.0",
        "description": "TRIX, TR6 Rev.12",
        "parameters": {"n_angles": 72, "R_km": 3.5, "dr_km": 0.05, "slope_critical": 0.033, "crs": None},
        "sampler": {"interpolation_method": "linear"},
    }
    assert ANALYZER_DEFAULTS == expected


def test_analyzer_defaults_tr6_locked_defaults_match_the_locked_value_exactly():
    ray_params = PRODUCT_DEFINITION_TRIX["parameters"]["ray"]
    slope_params = PRODUCT_DEFINITION_TRIX["parameters"]["slope"]
    assert ANALYZER_DEFAULTS["parameters"]["n_angles"] == ray_params["n_angles"]["value"]
    assert ANALYZER_DEFAULTS["parameters"]["R_km"] == ray_params["R_km"]["value"]
    assert ANALYZER_DEFAULTS["parameters"]["slope_critical"] == slope_params["slope_critical"]["value"]


def test_analyzer_defaults_dr_km_default_is_within_its_declared_range():
    dr_km_definition = PRODUCT_DEFINITION_TRIX["parameters"]["ray"]["dr_km"]
    low, high = dr_km_definition["range"]
    assert low <= ANALYZER_DEFAULTS["parameters"]["dr_km"] <= high


def test_analyzer_defaults_interpolation_method_default_is_a_valid_literal_member():
    allowed = typing.get_args(INTERPOLATION_METHODS)
    assert ANALYZER_DEFAULTS["sampler"]["interpolation_method"] in allowed


def test_writer_defaults_match_expected_snapshot():
    expected = {
        "filenames": {
            "manifest": "summary.yaml",
            "rix_table": "rix_table.csv",
            "trix_table": "trix.csv",
            "geopackage": "rix_details.gpkg",
        },
        "gpkg_layers": {
            "locations_site": "locations_site",
            "locations_reference": "locations_reference",
            "ruggedness": "ruggedness",
            "trix": "trix",
        },
    }
    assert WRITER_DEFAULTS == expected


def test_writer_defaults_filename_keys_match_artifact_profile_members():
    profile_members = {
        name for members in PRODUCT_DEFINITION_TRIX["artifacts"]["profiles"].values() for name in members
    }
    assert set(WRITER_DEFAULTS["filenames"].keys()) == profile_members


def test_copdem_metadata_matches_expected_snapshot():
    expected = {
        "name": "Copernicus DEM GLO-30",
        "source": "Copernicus",
        "description": "Downloaded 2026.",
    }
    assert COPDEM_METADATA == expected
