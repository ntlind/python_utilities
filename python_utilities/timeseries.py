import pandas as pd
from python_utilities import helpers


def update_time(datetime_series, adjustment, time_unit="weeks"):
    """
    Correctly adds or subtracts units from a time period
    """
    assert helpers.is_datetime_series(datetime_series)

    adjustment_delta = pd.DateOffset(**{time_unit: adjustment})
    adjusted_series = datetime_series + adjustment_delta
    return adjusted_series


def fill_missing_timeseries(pandas_df: pd.DataFrame, grouping_cols, 
                            freq='D', datetime_col="datetime"):
    """
    Fills-in missing dates in a pandas df with a time-series index
    """

    assert datetime_col in pandas_df.columns
    assert helpers.is_datetime_series(pandas_df[datetime_col])

    filled_pd = (
        pandas_df.set_index(datetime_col)
        .groupby(grouping_cols)
        .apply(
            lambda d: d.reindex(
                pd.date_range(
                    pandas_df.datetime.min(),
                    pandas_df.datetime.max(),
                    freq=freq
                )
            )
        )
        .drop(grouping_cols, axis=1)
        .reset_index()
    )

    filled_pd[grouping_cols] = filled_pd[grouping_cols].ffill(downcast="infer")

    datetime_col = [col for col in list(filled_pd.columns) if "level_" in col]
    assert len(datetime_col) == 1

    filled_pd = filled_pd.rename({datetime_col[0]: "datetime"}, axis=1)

    return filled_pd

