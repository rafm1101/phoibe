"""Composition root: the only place that connects the use case with concrete adapters (C1).

Entrypoints and notebooks build their analyzer here::

    from trix.main import build_analyzer, read_field
    from trix.entrypoints.config_file import load_configuration

    configuration = load_configuration("run.yaml", PRODUCT_SPECIFICATION)
    analyzer = build_analyzer(configuration)
    result = analyzer.run(read_field("dem.tif"), sites, references, project_name=configuration.project_name)
    analyzer.publish(result, "output/")
"""

from __future__ import annotations

from .adapters.dem_reader import read_field
from .adapters.writer import FileResultWriter
from .entrypoints.config_file import RunConfiguration, default_configuration
from .pipeline.analyzer_refactored import TRIXAnalyzer
from .pipeline.ports import ResultWriter
from .pipeline.schema import PRODUCT_SPECIFICATION
from .pipeline.specification import ProductSpecification

__all__ = ["build_analyzer", "read_field"]


def build_analyzer(
    configuration: RunConfiguration | None = None,
    *,
    spec: ProductSpecification = PRODUCT_SPECIFICATION,
    writer: ResultWriter | None = None,
) -> TRIXAnalyzer:
    """Create the analyzer.

    Parameters
    ----------
    configuration
        Parameters of the run. Without it, the defaults of the specification are used.
    spec
        Product specification.
    writer
        Where results are published. Defaults to writing files into a local directory.
    """
    configuration = configuration or default_configuration(spec)
    return TRIXAnalyzer(
        spec=spec,
        parameters=configuration.parameters,
        sampler_parameters=configuration.sampler_parameters,
        writer=writer or FileResultWriter(spec),
    )
