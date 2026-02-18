import datetime

import pandas as pd
import pytest

from phoibe.layered.core.entities import FileMetadata
from phoibe.layered.infrastructure.io import InMemoryDataLoader
from phoibe.layered.infrastructure.io import PandasDataLoader


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="10min"),
            "wind_speed": [5.2, 6.1, 5.8, 7.2, 6.5, 5.9, 6.8, 7.1, 6.4, 5.7],
            "power": [1200, 1500, 1300, 1800, 1400, 1250, 1600, 1750, 1450, 1280],
        }
    )


class DataLoaderContract:

    def test_load_returns_dataframe(self, loader, sample_data):
        result = loader.load(sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_load_returns_dataframe_w_contents(self, loader, sample_data):
        result = loader.load(sample_data)
        assert len(result) > 0
        assert len(result.columns) > 0

    def test_get_metadata_returns_file_metadata(self, loader, sample_data):
        result = loader.get_metadata(sample_data)
        assert isinstance(result, FileMetadata)

    def test_get_metadata_return_has_properties(self, loader, sample_data):
        metadata = loader.get_metadata(sample_data)
        assert isinstance(metadata.filename, str)
        assert len(metadata.filename) > 0
        assert metadata.size_bytes > 0
        assert isinstance(metadata.format, str)
        assert len(metadata.format) > 0
        assert isinstance(metadata.modified_at, datetime.datetime)

    def test_load_is_idempotent(self, loader, sample_data):
        result1 = loader.load(sample_data)
        result2 = loader.load(sample_data)

        assert len(result1) == len(result2)
        assert list(result1.columns) == list(result2.columns)


class TestPandasDataLoaderContract(DataLoaderContract):

    @pytest.fixture
    def loader(self):
        return PandasDataLoader()

    @pytest.fixture
    def sample_data(self, tmp_path, sample_dataframe):
        csv_file = tmp_path / "test_data.csv"
        sample_dataframe.to_csv(csv_file, index=False)
        return csv_file

    def test_loads_csv_files(self, loader, tmp_path):
        csv_file = tmp_path / "data.csv"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(csv_file, index=False)

        result = loader.load(csv_file)
        assert len(result) == 2

    def test_loads_excel_files(self, loader, tmp_path):
        excel_file = tmp_path / "data.xlsx"
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_excel(excel_file, index=False)

        result = loader.load(excel_file)
        assert len(result) == 2

    def test_raises_on_nonexistent_file(self, loader, tmp_path):
        nonexistent = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            loader.load(nonexistent)

    def test_raises_on_unsupported_format(self, loader, tmp_path):
        unsupported = tmp_path / "data.parquet"
        unsupported.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(unsupported)

    def test_handles_multiple_encodings(self, loader, tmp_path):
        utf8_file = tmp_path / "utf8.csv"
        df = pd.DataFrame({"text": ["Hello", "World"]})
        df.to_csv(utf8_file, index=False, encoding="utf-8")
        result = loader.load(utf8_file)
        assert len(result) == 2


class TestInMemoryDataLoaderContract(DataLoaderContract):

    @pytest.fixture
    def loader(self, sample_dataframe):
        return InMemoryDataLoader(sample_dataframe, filename="test_data")

    @pytest.fixture
    def sample_data(self):
        return ""

    def test_loads_dataframe_directly(self, sample_dataframe):
        loader = InMemoryDataLoader(sample_dataframe)
        result = loader.load("")
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_file_path_is_ignored(self, sample_dataframe):
        loader = InMemoryDataLoader(sample_dataframe)
        result1 = loader.load("ignored_path_1")
        result2 = loader.load("ignored_path_2")
        result3 = loader.load("")

        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result2, result3)

    def test_metadata_uses_virtual_filename(self, sample_dataframe):
        loader = InMemoryDataLoader(sample_dataframe, filename="my_virtual_file")
        metadata = loader.get_metadata("")

        assert metadata.filename == "my_virtual_file"

    def test_metadata_format_is_in_memory(self, sample_dataframe):
        loader = InMemoryDataLoader(sample_dataframe)
        metadata = loader.get_metadata("")

        assert metadata.format == "in_memory"

    def test_raises_on_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported data type"):
            InMemoryDataLoader(data=[1, 2, 3])

    def test_validates_type_at_init(self):
        InMemoryDataLoader(pd.DataFrame({"a": [1, 2]}))
        with pytest.raises(TypeError):
            InMemoryDataLoader({"not": "a dataframe"})


class TestDataLoaderInteroperability:

    def test_file_and_memory_loaders_produce_same_data(self, tmp_path):
        df_original = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        csv_file = tmp_path / "data.csv"
        df_original.to_csv(csv_file, index=False)

        df_from_file = PandasDataLoader().load(csv_file)
        df_from_memory = InMemoryDataLoader(df_original).load("")

        pd.testing.assert_frame_equal(df_from_file, df_from_memory)
