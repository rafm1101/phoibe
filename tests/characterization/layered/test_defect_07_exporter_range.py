import numpy as np
import pandas as pd
import pytest

from phoibe.layered.application.context import ValidationContext
from phoibe.layered.infrastructure.export import ProfileExporter
from phoibe.layered.rules.rules_values import EssentialRange


@pytest.mark.characterization
@pytest.mark.xfail(strict=True, reason="defect 7: the exporter's discoveredRange branch never matches")
def test_exported_schema_carries_discovered_range(report_factory):
    df = pd.DataFrame({"ws": np.linspace(0.0, 20.0, 1001)})
    context = ValidationContext(layer_name="raw", detected_variables={"wind_speed": "ws"}, turbine_id="WEA_01")
    result = EssentialRange(variable_names=["wind_speed"]).execute(df, context)
    report = report_factory(layer_name="raw", results=[result], detected_variables={"wind_speed": "ws"})

    profile = ProfileExporter().to_odcs_fragment(report)

    assert profile["discoveredSchema"]["wind_speed"]["discoveredRange"] == result.details["checked"]["wind_speed"]
