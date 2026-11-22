"""YAML and JSON text for plain data. Duplicate keys and non-finite number literals are refused, never repaired."""

import json
import typing

import yaml

type Format = typing.Literal["yaml", "json"]

FORMATS: tuple[Format, ...] = ("yaml", "json")


class _UniqueKeyLoader(yaml.SafeLoader):
    """A safe YAML loader that refuses a key that appears twice, instead of silently keeping the last."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[object, object]:  # noqa: FBT001, FBT002
        seen: set[object] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def _build_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"duplicate key {duplicates[0]!r}")
    return dict(pairs)


def _refuse_constant(name: str) -> typing.NoReturn:
    raise ValueError(f"the number literal {name} is not allowed; non-finite numbers are written as {{$float: ...}}")


def load_yaml(text: str) -> object:
    """Parse YAML text with the strict loader.

    Parameters
    ----------
    text
        YAML text.

    Returns
    -------
    object
        The plain data.

    Raises
    ------
    yaml.YAMLError
        If the text is not valid YAML or repeats a key.
    """
    return yaml.load(text, Loader=_UniqueKeyLoader)


def parse_text(text: str, fmt: Format) -> object:
    """Parse text in a format into plain data.

    Parameters
    ----------
    text
        The text.
    fmt
        `yaml` or `json`.

    Returns
    -------
    object
        The plain data.

    Raises
    ------
    ValueError
        If the text is not valid in the format, repeats a key or holds a non-finite number literal. A YAML error is
        converted, so the caller handles one exception type.
    """
    if fmt == "yaml":
        try:
            return load_yaml(text)
        except yaml.YAMLError as error:
            raise ValueError(f"invalid yaml: {error!s}") from error
    if fmt == "json":
        try:
            return json.loads(text, object_pairs_hook=_build_json_object, parse_constant=_refuse_constant)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid json: {error!s}") from error
    raise ValueError(f"unknown format {fmt!r}; use one of {', '.join(FORMATS)}")


def format_text(data: object, fmt: Format) -> str:
    """Format plain data as text, keeping the key order.

    Parameters
    ----------
    data
        Plain data: mappings with text keys, lists, text, numbers, booleans and `None`.
    fmt
        `yaml` or `json`.

    Returns
    -------
    str
        The text, ending in a newline.

    Raises
    ------
    ValueError
        For an unknown format, or data that holds a non-finite number (map such numbers to `{$float: ...}` first).
    """
    if fmt == "yaml":
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False)
    if fmt == "json":
        return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    raise ValueError(f"unknown format {fmt!r}; use one of {', '.join(FORMATS)}")
