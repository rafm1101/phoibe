"""Command line entry point: arguments in, exit code out (DM5).

This module only defines the command line and the translation of errors into exit codes (E1). It imports no adapter,
so both are testable without files. `main.main` connects it with the adapters.

Exit codes
----------
0   success, also with warnings
1   problem with the inputs or the configuration: required input missing, invalid configuration, a check with severity
    "abort"
2   invalid command line (argparse)
3   technical failure of an adapter: file not found, not writable, ...
4   the written artifacts violate the contract; nothing was published
70  unexpected error, i.e. a defect
"""

from __future__ import annotations

import argparse
import pathlib
from collections.abc import Sequence

from ..pipeline.errors import (
    CheckAborted,
    ConfigurationError,
    ContractViolation,
    InputRequirementError,
    PipelineError,
    PortError,
)
from ..pipeline.ports import WrittenArtifacts
from ..pipeline.results.result_summary import ResultSummary

EXIT_OK = 0
EXIT_INPUT = 1
EXIT_USAGE = 2
EXIT_PORT = 3
EXIT_CONTRACT = 4
EXIT_INTERNAL = 70


def build_parser(profiles: Sequence[str]) -> argparse.ArgumentParser:
    """The command line. `profiles` are the output profiles of the product; the first one is the default."""
    parser = argparse.ArgumentParser(prog="trix", description="T-RIX assessment of sites.")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more log output (-v: info, -vv: debug)")
    # The same option after the command, e.g. `trix run ... -v`. SUPPRESS keeps a value given before the command.
    verbosity = argparse.ArgumentParser(add_help=False)
    verbosity.add_argument(
        "-v", "--verbose", action="count", default=argparse.SUPPRESS, help="more log output (-v: info, -vv: debug)"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", parents=[verbosity], help="assess sites and write the result")
    run.add_argument("--dem", required=True, type=pathlib.Path, help="digital elevation model (GeoTIFF)")
    run.add_argument("--sites", required=True, type=pathlib.Path, help="point locations of the sites to assess")
    run.add_argument("--sites-layer", help="layer of the sites file, if it has several")
    run.add_argument("--references", type=pathlib.Path, help="point locations of the reference sites; enables TRIX")
    run.add_argument("--references-layer", help="layer of the references file, if it has several")
    run.add_argument("--output", required=True, type=pathlib.Path, help="directory for the result")
    run.add_argument("--config", type=pathlib.Path, help="YAML file with parameters; defaults come from the product")
    run.add_argument("--profile", choices=list(profiles), default=profiles[0], help="what to write")
    run.add_argument("--project-name", help="label of the run, overrides the configuration file")
    run.add_argument(
        "--dem-meta",
        action="append",
        default=[],
        type=_key_value,
        metavar="KEY=VALUE",
        help="description of the DEM for the manifest, may be repeated, e.g. name=COPDEM",
    )

    contract = commands.add_parser("contract", parents=[verbosity], help="print the contract of this product version")
    contract.add_argument("--output", type=pathlib.Path, help="write to this file instead of the standard output")
    return parser


def parse_args(argv: Sequence[str] | None, profiles: Sequence[str]) -> argparse.Namespace:
    """Parse the command line. Exits with code 2 on invalid usage, as argparse does."""
    parser = build_parser(profiles)
    arguments = parser.parse_args(argv)
    if arguments.command == "run" and arguments.references is None and arguments.references_layer is not None:
        parser.error("--references-layer requires --references")
    return arguments


def exit_code_for(error: BaseException) -> int:
    """Exit code for an error raised by a run."""
    if isinstance(error, (InputRequirementError, ConfigurationError, CheckAborted)):
        return EXIT_INPUT
    if isinstance(error, PortError):
        return EXIT_PORT
    if isinstance(error, ContractViolation):
        return EXIT_CONTRACT
    if isinstance(error, PipelineError):
        return EXIT_INTERNAL
    return EXIT_INTERNAL


def format_summary(result: ResultSummary, written: WrittenArtifacts, output: pathlib.Path) -> str:
    """Short report of a finished run for the console."""
    record = result.record
    lines = [
        f"Assessed {record.collections['locations_site'].n_locations} site(s)"
        + (
            f" against {record.collections['locations_reference'].n_locations} reference location(s)"
            if "locations_reference" in record.collections
            else ""
        ),
        f"Wrote profile '{written.profile}' to {output}: {', '.join(sorted([*written.files, 'manifest']))}",
    ]
    if record.finding_counts:
        counts = ", ".join(f"{check}: {count}" for check, count in sorted(record.finding_counts.items()))
        lines.append(f"Findings (see manifest): {counts}")
    return "\n".join(lines)


def _key_value(text: str) -> tuple[str, str]:
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key, value
