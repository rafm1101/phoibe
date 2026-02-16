import pathlib
import tempfile

import pytest


@pytest.fixture
def tmp_path():
    with tempfile.TemporaryDirectory() as temporary_dir:
        yield pathlib.Path(temporary_dir)
