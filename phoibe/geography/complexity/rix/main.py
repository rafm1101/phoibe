"""Composition root: the only place that connects the use case with concrete adapters (C1).

Command line (see `entrypoints.cli`)::

    trix run --dem dem.tif --sites sites.gpkg --references refs.gpkg --output out/ --profile full
    trix contract --output contract.yaml

Notebook::

    from trix.main import build_analyzer, read_field, read_locations
    from trix.entrypoints.config_file import load_configuration

    configuration = load_configuration("run.yaml", PRODUCT_SPECIFICATION)
    analyzer = build_analyzer(configuration)
    sites, references = read_locations("sites.gpkg"), read_locations("references.gpkg")
    result = analyzer.run(read_field("dem.tif"), sites, references, project_name=configuration.project_name)
    analyzer.publish(result, "output/")

Entry point for installation: ``trix = ".main:main"`` in ``[project.scripts]``.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from collections.abc import Sequence

import yaml

from .adapters.dem_reader import read_field
from .adapters.locations_reader import read_locations
from .adapters.writer import FileResultWriter
from .entrypoints import cli
from .entrypoints.config_file import RunConfiguration, default_configuration, load_configuration
from .pipeline.analyzer_refactored import TRIXAnalyzer
from .pipeline.contract import render_contract
from .pipeline.errors import PipelineError
from .pipeline.ports import ResultWriter
from .pipeline.schema import PRODUCT_SPECIFICATION
from .pipeline.specification import ProductSpecification

__all__ = ["build_analyzer", "main", "read_field", "read_locations"]

LOGGER = logging.getLogger(__name__)


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


def main(argv: Sequence[str] | None = None, *, spec: ProductSpecification = PRODUCT_SPECIFICATION) -> int:
    """Run the command line and return the exit code."""
    try:
        arguments = cli.parse_args(argv, profiles=list(spec.profiles))
    except SystemExit as stop:  # argparse: invalid usage (2) or --help (0)
        return int(stop.code or 0)

    logging.basicConfig(
        level={0: logging.WARNING, 1: logging.INFO}.get(arguments.verbose, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if arguments.command == "contract":
            return _contract(arguments, spec)
        return _run(arguments, spec)
    except PipelineError as error:
        LOGGER.error("%s", error)
        return cli.exit_code_for(error)
    except Exception:
        LOGGER.exception("Unexpected error")
        return cli.EXIT_INTERNAL


def _run(arguments, spec: ProductSpecification) -> int:
    configuration = load_configuration(arguments.config, spec) if arguments.config else default_configuration(spec)
    if arguments.project_name is not None:
        configuration = dataclasses.replace(configuration, project_name=arguments.project_name)

    analyzer = build_analyzer(configuration, spec=spec)
    field = read_field(arguments.dem)
    sites = read_locations(arguments.sites, layer=arguments.sites_layer)
    references = (
        None if arguments.references is None else read_locations(arguments.references, layer=arguments.references_layer)
    )
    result = analyzer.run(
        field,
        sites,
        references,
        dem_metadata=dict(arguments.dem_meta),
        project_name=configuration.project_name,
        include_steep_segments=arguments.profile == "full",
    )
    written = analyzer.publish(result, arguments.output, arguments.profile)
    print(cli.format_summary(result, written, arguments.output))
    return cli.EXIT_OK


def _contract(arguments, spec: ProductSpecification) -> int:
    text = yaml.safe_dump(render_contract(spec), sort_keys=False, allow_unicode=True)
    if arguments.output is None:
        sys.stdout.write(text)
    else:
        arguments.output.write_text(text, encoding="utf-8")
    return cli.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
