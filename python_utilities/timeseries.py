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

