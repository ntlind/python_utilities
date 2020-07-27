import sys 
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import numpy as np
import pandas as pd
import dask.dataframe as dd
from python_utilities import timeseries, testing

def test_update_time():
    test_series = pd.to_datetime(testing.get_test_example()['datetime'])

    result = timeseries.update_time(test_series, 1, 'weeks')
    answer = pd.Series([
        "2020-01-08",
        "2020-01-09",
        "2020-01-10",
        "2020-01-11",
        "2020-01-06",
        "2020-01-07",
        "2020-01-08",
        "2020-01-09",
        ])

    assert (result == pd.to_datetime(answer)).all(), (result == pd.to_datetime(answer))

    result = timeseries.update_time(test_series, 1, 'months')
    answer = pd.Series([
        "2020-02-01",
        "2020-02-02",
        "2020-02-03",
        "2020-02-04",
        "2020-01-30",
        "2020-01-31",
        "2020-02-01",
        "2020-02-02",
        ])
    assert (result == pd.to_datetime(answer)).all(), (result == pd.to_datetime(answer))

    result = timeseries.update_time(test_series, 1, 'days')
    answer = pd.Series([
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
        "2020-01-05",
        "2019-12-31",
        "2020-01-01",
        "2020-01-02",
        "2020-01-03",
        ])
    assert (result == pd.to_datetime(answer)).all(), (result == pd.to_datetime(answer))

    result = timeseries.update_time(test_series, -1, 'months')
    answer = pd.Series([
        "2019-12-01",
        "2019-12-02",
        "2019-12-03",
        "2019-12-04",
        "2019-11-30",
        "2019-11-30",
        "2019-12-01",
        "2019-12-02",
        ])
    assert (result == pd.to_datetime(answer)).all(), (result == pd.to_datetime(answer))


if __name__ == "__main__":
    test_update_time() 