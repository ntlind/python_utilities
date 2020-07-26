#TODO gross
def update_time(series_or_int, adjustment, time_unit='weeks', datetime_format="%Y%U-%w"):
  """
  Correctly adds or subtracts units from a time period
  """  
  casted_series = pd.Series(series_or_int).astype(str)
  
  adjustment_delta = pd.DateOffset(**{self.TIME_INCREMENT:adjustment})  
  adjusted_series = (datetime_series + adjustment_delta)
  
  final_series = adjusted_series.dt.strftime(self.TIME_FORMAT)\
                                  .astype(int)
  
  return final_series




def update_time(series_or_int, adjustment, time_unit='weeks', datetime_format="%Y%U-%w"):
"""
Correctly adds or subtracts units from a time period
"""  

# we may want to treat time periods as a monotonically increasing integer
# without breaking all of our other functions
if time_unit == "int":
  return series_or_int + adjustment

casted_series = pd.Series(series_or_int).astype(str)

# get datetime col
if datetime_format in ["%Y%U-%w", "%Y%W-%w"]:
  datetime_series = pd.to_datetime(casted_series.astype(str) + '-1', format=datetime_format)
else:
  datetime_series = pd.to_datetime(casted_series.astype(str), format=datetime_format)
  
# make adjustment
adjustment_delta = pd.DateOffset(**{time_unit:adjustment})  
adjusted_series = (datetime_series + adjustment_delta)

# return the series in the original
final_series = adjusted_series.dt.strftime(datetime_format)

if datetime_format in ["%Y%U-%w", "%Y%W-%w"]:
  final_series = final_series.str.extract(r'(.*)-\d+', expand=False).astype(int)
else:
  final_series = final_series.astype(int)
  
if isinstance(series_or_int, (int, np.integer)): 
  assert final_series.shape == (1,)
  return final_series[0]

return final_series