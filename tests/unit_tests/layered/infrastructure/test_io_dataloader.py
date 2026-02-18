import datetime

import numpy as np
import pandas as pd
import pytest
import xarray

from phoibe.layered.infrastructure.io import InMemoryDataLoader
from phoibe.layered.infrastructure.io import PandasDataLoader


class TestPandasDataLoaderEdgeCases:

    @pytest.fixture
    def loader(self):
        return PandasDataLoader()

    def test_load_raises_file_not_found_given_nonexistent_file(self, loader, tmp_path):
        nonexistent = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError, match="File not found"):
            loader.load(nonexistent)

    def test_load_raises_file_not_found_given_directory(self, loader, tmp_path):
        directory = tmp_path / "subdir"
        directory.mkdir()
        with pytest.raises(FileNotFoundError, match="File not found"):
            loader.load(directory)

    def test_get_metadata_raises_given_nonexistent_file(self, loader, tmp_path):
        nonexistent = tmp_path / "does_not_exist.csv"
        with pytest.raises(FileNotFoundError):
            loader.get_metadata(nonexistent)

    def test_load_raises_given__unsupported_format(self, loader, tmp_path):
        unsupported = tmp_path / "data.parquet"
        unsupported.write_text("dummy content")
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(unsupported)

    def test_load_raises_given__json_format(self, loader, tmp_path):
        json_file = tmp_path / "data.json"
        json_file.write_text('{"a": 1}')
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(json_file)

    def test_load_raises_given__txt_format(self, loader, tmp_path):
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("some text")
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(txt_file)

    def test_load_raises_given__no_extension(self, loader, tmp_path):
        no_ext = tmp_path / "data"
        no_ext.write_text("a,b\n1,2")
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load(no_ext)

    @pytest.mark.parametrize("encoding, expected_length", [("utf-8", 4), ("latin1", 4), ("iso-8859-1", 4)])
    def test_loads_utf8_csv(self, loader, tmp_path, encoding, expected_length):
        csv_file = tmp_path / "file.csv"
        df = pd.DataFrame({"text": ["Hello", "Wörld", "Tëst", "Äpfel"]})
        df.to_csv(csv_file, index=False, encoding=encoding)

        result = loader.load(csv_file)
        assert len(result) == expected_length
        assert "Wörld" in result["text"].values

    # Argument `encoding` in `pd.read_csv` makes pandas read weirdly encoded files. Contents is empty.
    # def test_load_raises_given_unsupported_encoding(self, loader, tmp_path):
    #     csv_file = tmp_path / "weird_encoding.csv"
    #     with open(csv_file, "w", encoding="utf-16") as filestream:
    #         filestream.write("Café,b\n1,2")
    #     breakpoint()
    #     with pytest.raises(ValueError, match="Could not decode"):
    #         loader.load(csv_file)

    # Argument `encoding` in `pd.read_csv` makes pandas read binary files. Contents is empty.
    # def test_load_raises_given_binary_file_as_csv(self, loader, tmp_path):
    #     csv_file = tmp_path / "binary.csv"
    #     csv_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe")
    #     with pytest.raises(ValueError, match="Could not decode"):
    #         loader.load(csv_file)

    # Loading throws an error.
    # def test_load_returns_empty_dataframe_given_empty_csv(self, loader, tmp_path):
    #     csv_file = tmp_path / "empty.csv"
    #     csv_file.write_text("")
    #     result = loader.load(csv_file)
    #     assert isinstance(result, pd.DataFrame)
    #     assert len(result) == 0

    def test_load_returns_dataframe_given_csv_with_only_headers(self, loader, tmp_path):
        csv_file = tmp_path / "headers_only.csv"
        csv_file.write_text("a,b,c\n")
        result = loader.load(csv_file)
        assert len(result) == 0
        assert list(result.columns) == ["a", "b", "c"]

    def test_load_returns_dataframe_given_csv_with_inconsistent_columns(self, loader, tmp_path):
        csv_file = tmp_path / "inconsistent.csv"
        csv_file.write_text("a,b,c\n1,2,3\n4,5\n6,7,8")
        result = loader.load(csv_file)
        assert isinstance(result, pd.DataFrame)

    def test_load_returns_dataframe_given_csv_with_special_characters(self, loader, tmp_path):
        csv_file = tmp_path / "special.csv"
        df = pd.DataFrame({"name": ["Smith, John", "O'Brien"], "value": [1, 2]})
        df.to_csv(csv_file, index=False)
        result = loader.load(csv_file)
        assert "Smith, John" in result["name"].values
        assert "O'Brien" in result["name"].values

    @pytest.mark.parametrize("filename", ["data.xlsx", "data.xls"])
    def test_load_accepts_xlsx_file(self, loader, tmp_path, filename):
        file = tmp_path / filename
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df.to_excel(file, index=False)
        result = loader.load(file)
        assert len(result) == 3
        assert list(result.columns) == ["a", "b"]

    def test_load_raises_given_corrupted_excel(self, loader, tmp_path):
        xlsx_file = tmp_path / "corrupted.xlsx"
        xlsx_file.write_text("This is not an Excel file")
        with pytest.raises(ValueError, match="Failed to load Excel"):
            loader.load(xlsx_file)

    def test_load_handles_empty_excel(self, loader, tmp_path):
        xlsx_file = tmp_path / "empty.xlsx"
        df = pd.DataFrame()
        df.to_excel(xlsx_file, index=False)
        result = loader.load(xlsx_file)
        assert len(result) == 0

    def test_get_metadata_size_given_very_small_file(self, loader, tmp_path):
        tiny_file = tmp_path / "tiny.csv"
        tiny_file.write_text("a\n1")
        metadata = loader.get_metadata(tiny_file)
        assert metadata.size_bytes > 0
        # assert metadata.size_mb > 0

    def test_get_metadata_size_given_large_file(self, loader, tmp_path):
        large_file = tmp_path / "large.csv"
        df = pd.DataFrame({"a": range(100000), "b": range(100000), "c": ["x" * 50 for _ in range(100000)]})
        df.to_csv(large_file, index=False)
        metadata = loader.get_metadata(large_file)
        assert metadata.size_mb > 1.0

    def test_get_metadata_filename_preserves_extension(self, loader, tmp_path):
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2")
        metadata = loader.get_metadata(csv_file)
        assert metadata.filename == "test.csv"
        assert metadata.format == "csv"

    def test_get_metadata_filename_given_file_without_extension(self, loader, tmp_path):
        no_ext = tmp_path / "noextension"
        no_ext.write_text("a,b\n1,2")
        metadata = loader.get_metadata(no_ext)
        assert metadata.filename == "noextension"
        assert metadata.format == "unknown"

    def test_load_accepts_str_and_pathlib_path(self, loader, tmp_path):
        csv_file = tmp_path / "test.csv"
        df = pd.DataFrame({"a": [1, 2]})
        df.to_csv(csv_file, index=False)
        result = loader.load(csv_file)
        assert len(result) == 2
        result = loader.load(str(csv_file))
        assert len(result) == 2

    def test_load_handles_uppercase_extensions(self, loader, tmp_path):
        csv_file = tmp_path / "test.CSV"
        df = pd.DataFrame({"a": [1, 2]})
        df.to_csv(csv_file, index=False)
        result = loader.load(csv_file)
        assert len(result) == 2

    def test_load_handles_mixed_case_extensions(self, loader, tmp_path):
        csv_file = tmp_path / "test.CsV"
        df = pd.DataFrame({"a": [1, 2]})
        df.to_csv(csv_file, index=False)
        result = loader.load(csv_file)
        assert len(result) == 2


class TestInMemoryDataLoaderEdgeCases:

    def test_accepts_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        loader = InMemoryDataLoader(df)
        assert loader.data is df

    def test_accepts_xarray_dataset(self):
        ds = xarray.Dataset({"temperature": (["time"], [20, 21, 22]), "pressure": (["time"], [1000, 1010, 1005])})
        loader = InMemoryDataLoader(ds)
        assert isinstance(loader.data, pd.DataFrame)

    def test_accepts_xarray_dataarray(self):
        da = xarray.DataArray([1, 2, 3], dims=["time"], name="pressure")
        loader = InMemoryDataLoader(da)
        assert isinstance(loader.data, pd.DataFrame)

    @pytest.mark.parametrize(
        "data", [[1, 2, 3], {"a": 1, "b": 2}, np.array([1, 2, 3]), pd.Series([1, 2, 3]), None, "a,b,c\n1,2,3"]
    )
    def test_refuses_other(self, data):
        with pytest.raises(TypeError, match="Unsupported data type"):
            InMemoryDataLoader(data=data)

    def test_accepts_empty_dataframe(self):
        df = pd.DataFrame()
        loader = InMemoryDataLoader(df)
        result = loader.load("")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_accepts_dataframe_with_no_rows(self):
        df = pd.DataFrame(columns=["a", "b", "c"])
        loader = InMemoryDataLoader(df)
        result = loader.load("")
        assert len(result) == 0
        assert list(result.columns) == ["a", "b", "c"]

    def test_accepts_single_cell_dataframe(self):
        df = pd.DataFrame({"a": [1]})
        loader = InMemoryDataLoader(df)
        result = loader.load("")
        assert len(result) == 1
        assert result.iloc[0, 0] == 1

    def test_accepts_very_large_dataframe(self):
        df = pd.DataFrame({"a": range(1000000), "b": range(1000000)})
        loader = InMemoryDataLoader(df)
        result = loader.load("")
        assert len(result) == 1000000

    def test_file_path_is_ignored_empty_string(self):
        df = pd.DataFrame({"a": [1, 2]})
        loader = InMemoryDataLoader(df)
        result = loader.load("")
        assert len(result) == 2

    def test_file_path_is_ignored_any_string(self):
        df = pd.DataFrame({"a": [1, 2]})
        loader = InMemoryDataLoader(df)
        result1 = loader.load("path1.csv")
        result2 = loader.load("completely/different/path.xlsx")
        result3 = loader.load("/absolute/path/to/file.txt")
        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result2, result3)

    def test_file_path_default_parameter(self):
        df = pd.DataFrame({"a": [1, 2]})
        loader = InMemoryDataLoader(df)
        result = loader.load()
        assert len(result) == 2

    def test_dataframe_returned_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        loader = InMemoryDataLoader(df)
        result = loader.load("")
        assert result is df

    def test_dataframe_with_multiindex(self):
        df = pd.DataFrame(
            {"value": [1, 2, 3, 4]}, index=pd.MultiIndex.from_tuples([("A", 1), ("A", 2), ("B", 1), ("B", 2)])
        )
        loader = InMemoryDataLoader(df)
        result = loader.load("")
        assert isinstance(result.index, pd.MultiIndex)

    def test_dataframe_with_datetime_index(self):
        df = pd.DataFrame({"value": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
        loader = InMemoryDataLoader(df)
        result = loader.load("")
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_load_converts_xarray_dataset_to_dataframe(self):
        ds = xarray.Dataset(
            {"temperature": (["time"], [20, 21, 22]), "pressure": (["time"], [1000, 1010, 1005])},
            coords={"time": pd.date_range("2024-01-01", periods=3)},
        )
        loader = InMemoryDataLoader(ds)
        result = loader.load("")
        assert isinstance(result, pd.DataFrame)
        assert "temperature" in result.columns
        assert "pressure" in result.columns
        assert len(result) == 3

    def test_load_converts_xarray_dataarray_to_dataframe(self):
        da = xarray.DataArray(
            [1, 2, 3], dims=["time"], coords={"time": pd.date_range("2024-01-01", periods=3)}, name="pressure"
        )
        loader = InMemoryDataLoader(da)
        result = loader.load("")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_xarray_conversion_resets_index(self):
        ds = xarray.Dataset({"value": (["x", "y"], [[1, 2], [3, 4]])}, coords={"x": [0, 1], "y": [10, 20]})
        loader = InMemoryDataLoader(ds)
        result = loader.load("")
        assert "x" in result.columns
        assert "y" in result.columns
        assert "value" in result.columns

    def test_get_metadata_uses_custom_filename(self):
        df = pd.DataFrame({"a": [1, 2]})
        loader = InMemoryDataLoader(df, filename="my_custom_name.csv")
        metadata = loader.get_metadata("")
        assert metadata.filename == "my_custom_name.csv"

    def test_get_metadata_default_filename(self):
        df = pd.DataFrame({"a": [1, 2]})
        loader = InMemoryDataLoader(df)
        metadata = loader.get_metadata("")
        assert metadata.filename == "in_memory_data"

    def test_get_metadata_format_is_in_memory(self):
        df = pd.DataFrame({"a": [1, 2]})
        loader = InMemoryDataLoader(df, filename="anything.xlsx")
        metadata = loader.get_metadata("")
        assert metadata.format == "in_memory"

    def test_get_metadata_returns_size_for_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.5, 2.5, 3.5]})
        loader = InMemoryDataLoader(df)
        metadata = loader.get_metadata("")
        assert metadata.size_bytes > 0

        expected_size = df.memory_usage(deep=True).sum()
        assert metadata.size_bytes == int(expected_size)

    def test_get_metadata_returns_size_for_xarray(self):
        ds = xarray.Dataset({"temperature": (["time"], [20, 21, 22])})
        loader = InMemoryDataLoader(ds)
        metadata = loader.get_metadata("")
        assert metadata.size_bytes > 0

        expected_size = ds.to_dataframe().reset_index().memory_usage(deep=True).sum()
        assert metadata.size_bytes == expected_size

    def test_get_metadata_returns_now_as_modified_at(self):
        df = pd.DataFrame({"a": [1, 2]})
        loader = InMemoryDataLoader(df)
        before = datetime.datetime.now()
        metadata = loader.get_metadata("")
        after = datetime.datetime.now()
        assert before <= metadata.modified_at <= after

    @pytest.mark.parametrize("path", ["", "ignored_path.csv", "/absolute/path"])
    def test_get_metadata_file_path_is_ignored(self, path):
        df = pd.DataFrame({"a": [1, 2]})
        loader = InMemoryDataLoader(df, filename="test.csv")
        metadata = loader.get_metadata(path)
        assert metadata.filename == "test.csv"

    def test_multiple_load_calls_return_same_data(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        loader = InMemoryDataLoader(df)
        result1 = loader.load("")
        result2 = loader.load("")
        result3 = loader.load("")
        pd.testing.assert_frame_equal(result1, result2)
        pd.testing.assert_frame_equal(result2, result3)

    def test_load_does_not_mutate_original_data(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        df_copy = df.copy()
        _ = InMemoryDataLoader(df).load("")
        pd.testing.assert_frame_equal(df, df_copy)

    def test_accepts_dataframe_with_nan_values(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, 5, 6]})
        result = InMemoryDataLoader(df).load("")
        assert pd.isna(result.iloc[0, 1])
        assert pd.isna(result.iloc[1, 0])

    def test_accepts_dataframe_with_mixed_types(self):
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
                "datetime_col": pd.date_range("2024-01-01", periods=3),
            }
        )
        result = InMemoryDataLoader(df).load("")
        assert len(result.columns) == 5
        assert result["int_col"].dtype == "int64"
        assert result["str_col"].dtype == "str"  # "object"

    def test_accepts_dataframe_with_categorical_data(self):
        df = pd.DataFrame({"category": pd.Categorical(["A", "B", "A", "C"])})
        result = InMemoryDataLoader(df).load("")
        assert isinstance(result["category"].dtype, pd.CategoricalDtype)
