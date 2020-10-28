import pandas as pd
from python_utilities import utils


def update_time(datetime_series, adjustment, time_unit="weeks"):
    """
    Correctly adds or subtracts units from a time period
    """
    assert utils.is_datetime_series(datetime_series)

    adjustment_delta = pd.DateOffset(**{time_unit: adjustment})
    adjusted_series = datetime_series + adjustment_delta
    return adjusted_series


def fill_missing_timeseries(pandas_df: pd.DataFrame, grouping_cols,
                            datetime_col, freq='D'):
    """
    Fills-in missing dates in a pandas df with a time-series index
    """

    assert datetime_col in pandas_df.columns
    assert utils.is_datetime_series(pandas_df[datetime_col])

    filled_pd = (
        pandas_df.set_index(datetime_col)
        .groupby(grouping_cols)
        .apply(
            lambda d: d.reindex(
                pd.date_range(
                    pandas_df[datetime_col].min(),
                    pandas_df[datetime_col].max(),
                    freq=freq
                )
            )
        )
        .drop(grouping_cols, axis=1)
        .reset_index()
    )

    filled_pd[grouping_cols] = filled_pd[grouping_cols].ffill(downcast="infer")

    date_col = [col for col in list(filled_pd.columns) if "level_" in col]
    assert len(date_col) == 1

    filled_pd = filled_pd.rename({date_col[0]: datetime_col}, axis=1)

    return filled_pd


#TODO write test; meant to be used with generators
def check_anomaly(day, day_data, stdev=3):
    """
    We find the mean, standard deviation, and maximum values for the day.
    Using a single-pass mean/standard deviation algorithm allows us to only
    read through the day's data once. If the day's data is anomalous, we
    return the value of the day.
    Source: High Performance Python (O'Reilly book)
    """
    import math

    n, mean, M2 = 0, 0, 0
    max_value = None
    
    for timestamp, value in day_data:
        n += 1
        delta = value - mean
        mean = mean + delta/n
        M2 += delta*(value - mean)
        max_value = max(max_value, value)
    variance = M2/(n - 1)
    standard_deviation = math.sqrt(variance)
    
    if max_value > mean + stdev * standard_deviation:
        return day

    return False