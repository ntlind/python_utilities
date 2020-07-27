from python_utilities import helpers, testing, dataframes
import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))


def test_format_bytes():
    assert helpers.format_bytes(5368709120) == "5.0GB"
    assert helpers.format_bytes(1024) == "1.0KB"
    assert helpers.format_bytes(124213) == "121.3KB"


def test_is_pandas_df():
    assert helpers.is_pandas_df(testing.get_test_example())
    assert not helpers.is_pandas_df([12, 123])


def test_is_dask_df():
    pandas_df = testing.get_test_example()
    dask_df = dataframes.convert_pandas_to_dask(pandas_df)
    assert helpers.is_dask_df(dask_df)
    assert not helpers.is_dask_df(pandas_df)


def test_is_list():
    assert helpers.is_list([1, 2, 3])
    assert not helpers.is_list("123")


def test_is_string():
    assert helpers.is_string("12321")
    assert not helpers.is_string([1, 2, 3])


def test_is_datetime_series():
    example = pd.Series(
        [
            "2020-01-08",
            "2020-01-09",
            "2020-01-10",
            "2020-01-11",
            "2020-01-06",
            "2020-01-07",
            "2020-01-08",
            "2020-01-09",
        ]
    )

    assert not helpers.is_datetime_series(example)

    example = pd.to_datetime(example)
    assert helpers.is_datetime_series(example)


def test_ensure_is_list():
    assert helpers.ensure_is_list(123) == [123]
    assert helpers.ensure_is_list([123]) == [123]


def test_correct_suffixes_in_list():
    pandas_df = testing.get_test_example()
    answer = list(pandas_df.columns).copy()
    result = helpers.correct_suffixes_in_list(pandas_df, pandas_df.columns)
    assert result == answer

    index_list = [col + "_index" for col in pandas_df.columns]
    result = helpers.correct_suffixes_in_list(pandas_df, index_list)
    assert result == answer


if __name__ == "__main__":
    test_format_bytes()
    test_is_pandas_df()
    test_is_dask_df()
    test_is_list()
    test_is_string()
    test_is_datetime_series()
    test_ensure_is_list()
    test_correct_suffixes_in_list()

