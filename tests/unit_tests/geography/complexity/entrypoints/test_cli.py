import pathlib
import types

import pytest

from phoibe.geography.complexity.rix.entrypoints import cli
from phoibe.geography.complexity.rix.pipelines import errors
from phoibe.geography.complexity.rix.pipelines.ports import WrittenArtifacts

PROFILES = ["summary", "full"]
RUN = ["run", "--dem", "dem.tif", "--sites", "sites.gpkg", "--output", "out"]


def parse(*argv):
    return cli.parse_args(list(argv), PROFILES)


def test_parse_args_returns_the_defaults_of_a_run():
    arguments = parse(*RUN)
    assert (arguments.command, arguments.dem, arguments.sites, arguments.output) == (
        "run",
        pathlib.Path("dem.tif"),
        pathlib.Path("sites.gpkg"),
        pathlib.Path("out"),
    )
    assert (arguments.sites_layer, arguments.references, arguments.references_layer, arguments.config) == (
        None,
        None,
        None,
        None,
    )
    assert (arguments.profile, arguments.project_name, arguments.dem_meta, arguments.verbose) == (
        "summary",
        None,
        [],
        0,
    )


def test_parse_args_returns_all_options_of_a_run():
    arguments = parse(
        *RUN,
        "--sites-layer",
        "s",
        "--references",
        "r.gpkg",
        "--references-layer",
        "r",
        "--config",
        "c.yaml",
        "--profile",
        "full",
        "--project-name",
        "p",
    )
    assert (arguments.sites_layer, arguments.references, arguments.references_layer) == (
        "s",
        pathlib.Path("r.gpkg"),
        "r",
    )
    assert (arguments.config, arguments.profile, arguments.project_name) == (pathlib.Path("c.yaml"), "full", "p")


def test_parse_args_returns_the_first_profile_as_the_default():
    assert cli.parse_args(RUN, ["full", "summary"]).profile == "full"


@pytest.mark.parametrize(
    "argv, verbose",
    [(["-v", *RUN], 1), ([*RUN, "-vv"], 2), (["-v", *RUN, "-v"], 1), (["-vv", *RUN], 2)],
    ids=["before the command", "after the command", "both, the last wins", "before, twice"],
)
def test_parse_args_returns_the_verbosity_before_or_after_the_command(argv, verbose):
    assert parse(*argv).verbose == verbose


def test_parse_args_returns_the_dem_description_as_pairs_in_order():
    arguments = parse(*RUN, "--dem-meta", "name=COPDEM", "--dem-meta", "source=a=b")
    assert arguments.dem_meta == [("name", "COPDEM"), ("source", "a=b")]  # only the first = separates


def test_parse_args_returns_the_contract_command_with_an_optional_file():
    assert (parse("contract").command, parse("contract").output) == ("contract", None)
    assert parse("contract", "--output", "c.yaml").output == pathlib.Path("c.yaml")


@pytest.mark.parametrize(
    "argv, message",
    [
        ([], "the following arguments are required: command"),
        (["run", "--sites", "s", "--output", "o"], "--dem"),
        (["run", "--dem", "d", "--output", "o"], "--sites"),
        (["run", "--dem", "d", "--sites", "s"], "--output"),
        ([*RUN, "--profile", "nope"], "invalid choice"),
        ([*RUN, "--dem-meta", "novalue"], "expected KEY=VALUE"),
        ([*RUN, "--dem-meta", "=value"], "expected KEY=VALUE"),
        ([*RUN, "--references-layer", "r"], "--references-layer requires --references"),
        (["nonsense"], "invalid choice"),
    ],
)
def test_parse_args_exits_with_two_given_invalid_usage(capsys, argv, message):
    with pytest.raises(SystemExit) as stop:
        parse(*argv)
    assert stop.value.code == cli.EXIT_USAGE and message in capsys.readouterr().err


def test_parse_args_exits_with_zero_given_help(capsys):
    with pytest.raises(SystemExit) as stop:
        parse("--help")
    assert stop.value.code == 0 and "usage: trix" in capsys.readouterr().out


def test_exit_codes_are_the_documented_ones():
    assert (cli.EXIT_OK, cli.EXIT_INPUT, cli.EXIT_USAGE, cli.EXIT_PORT, cli.EXIT_CONTRACT, cli.EXIT_INTERNAL) == (
        0,
        1,
        2,
        3,
        4,
        70,
    )


@pytest.mark.parametrize(
    "error, expected",
    [
        (errors.InputRequirementError("x"), cli.EXIT_INPUT),
        (errors.ConfigurationError("x"), cli.EXIT_INPUT),
        (errors.CheckAborted([]), cli.EXIT_INPUT),
        (errors.PortError("x"), cli.EXIT_PORT),
        (errors.ContractViolation(["x"]), cli.EXIT_CONTRACT),
        (errors.PipelineError("x"), cli.EXIT_INTERNAL),
        (ValueError("x"), cli.EXIT_INTERNAL),  # a plain ValueError is a defect, not bad input
        (RuntimeError("x"), cli.EXIT_INTERNAL),
    ],
    ids=lambda value: value if isinstance(value, str) else type(value).__name__,
)
def test_exit_code_for_returns_the_code_of_the_error(error, expected):
    assert cli.exit_code_for(error) == expected


def summary(collections, finding_counts):
    record = types.SimpleNamespace(
        collections={name: types.SimpleNamespace(n_locations=n) for name, n in collections.items()},
        finding_counts=finding_counts,
    )
    return types.SimpleNamespace(record=record)


WRITTEN = WrittenArtifacts("summary", {"trix_table": "t.csv", "rix_table": "r.csv"}, {}, {}, {}, {}, {})


def test_format_summary_returns_the_assessed_locations_the_files_and_the_findings():
    result = summary({"locations_site": 3, "locations_reference": 2}, {"unique_ids": 1, "nan_count": 2})
    assert cli.format_summary(result, WRITTEN, pathlib.Path("out")) == (
        "Assessed 3 site(s) against 2 reference location(s)\n"
        "Wrote profile 'summary' to out: manifest, rix_table, trix_table\n"
        "Findings (see manifest): nan_count: 2, unique_ids: 1"
    )


def test_format_summary_returns_two_lines_given_no_references_and_no_findings():
    text = cli.format_summary(summary({"locations_site": 1}, {}), WRITTEN, pathlib.Path("out"))
    assert text.splitlines() == [
        "Assessed 1 site(s)",
        "Wrote profile 'summary' to out: manifest, rix_table, trix_table",
    ]
