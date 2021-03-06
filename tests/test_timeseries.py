import pandas as pd

from python_utilities import timeseries, testing # noqa


def test_update_time():
    def check_answer():
        assert (result == pd.to_datetime(answer)).all()

    test_series = pd.to_datetime(testing.get_test_example()["datetime"])

    result = timeseries.update_time(test_series, 1, "weeks")
    answer = pd.Series(
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
    check_answer()

    result = timeseries.update_time(test_series, 1, "months")
    answer = pd.Series(
        [
            "2020-02-01",
            "2020-02-02",
            "2020-02-03",
            "2020-02-04",
            "2020-01-30",
            "2020-01-31",
            "2020-02-01",
            "2020-02-02",
        ]
    )
    check_answer()

    result = timeseries.update_time(test_series, 1, "days")
    answer = pd.Series(
        [
            "2020-01-02",
            "2020-01-03",
            "2020-01-04",
            "2020-01-05",
            "2019-12-31",
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
        ]
    )
    check_answer()

    result = timeseries.update_time(test_series, -1, "months")
    answer = pd.Series(
        [
            "2019-12-01",
            "2019-12-02",
            "2019-12-03",
            "2019-12-04",
            "2019-11-30",
            "2019-11-30",
            "2019-12-01",
            "2019-12-02",
        ]
    )
    check_answer()


def test_fill_missing_timeseries():
    test_pd = testing.get_test_example()
    test_pd.loc[3, 'datetime'] = '2020-01-05'
    test_pd['datetime'] = pd.to_datetime(test_pd['datetime'])

    grouping_cols = ['category', 'product', 'state', 'store']
    result = timeseries.fill_missing_timeseries(test_pd, 
                                                grouping_cols=grouping_cols,
                                                datetime_col='datetime')
    answer = pd.to_datetime([
        '2019-12-30', 
        '2019-12-31', 
        '2020-01-01', 
        '2020-01-02', 
        '2020-01-03', 
        '2020-01-04', 
        '2020-01-05'
        ] * 2)

    assert (result['datetime'] == answer).all()


if __name__ == "__main__":
    test_update_time()
    test_fill_missing_timeseries()
