"""ADR-07: the packages import in one direction. The imports are read with `ast`; nothing is imported."""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).parents[3] / "phoibe" / "dokimasia"
PREFIX = "phoibe.dokimasia"
MAY_IMPORT = {
    "core": set(),
    "contract": {"core"},
    "standardize": {"core"},
    "rules": {"core"},
    "wind": {"core", "rules"},
    "adapters": {"core", "contract"},
    "compose": {"core", "contract", "standardize", "rules", "wind", "adapters"},
}


def _packages_imported_by(package: str) -> set[str]:
    """Return the sibling packages of `phoibe.dokimasia` that any module of `package` imports."""
    imported: set[str] = set()
    for path in (ROOT / package).rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            for name in names:
                if name.startswith(f"{PREFIX}."):
                    imported.add(name.split(".")[2])
    return imported - {package}


def _existing_packages() -> list[str]:
    return sorted(path.name for path in ROOT.iterdir() if path.is_dir() and path.name in MAY_IMPORT)


def test_the_test_knows_every_package():
    unknown = {path.name for path in ROOT.iterdir() if path.is_dir() and not path.name.startswith("_")} - set(
        MAY_IMPORT
    )

    assert unknown == set(), "add the new package to MAY_IMPORT with what ADR-07 allows it to import"


@pytest.mark.parametrize("package", _existing_packages())
def test_package_imports_only_what_adr_07_allows(package):
    assert _packages_imported_by(package) <= MAY_IMPORT[package]


def test_core_imports_nothing_of_ours_but_itself():
    for path in (ROOT / "core").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                assert not node.module.startswith("phoibe") or node.module.startswith(f"{PREFIX}.core"), path


def test_core_does_no_file_io():
    forbidden = {"open", "pathlib", "yaml", "json", "os", "shutil", "tempfile"}
    for path in (ROOT / "core").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        }
        called = {
            node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not (imported | called) & forbidden, path
