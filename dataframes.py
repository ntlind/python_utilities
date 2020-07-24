 def compress_dataframe(df):
    """
    Downcast dataframe and convert objects to categories to save memory
    """
    import numpy as np

    def handle_numeric_downcast(array, type_):
        return pd.to_numeric(array, downcast=type_)

    numeric_lookup_dict = {
        "integer" : np.integer,
        "float" : np.floating,
        "object" : "object"
    }

    for type_ in ["integer", "float", "object"]:
        column_list = df.select_dtypes(include=numeric_lookup_dict[type_])

        if type_ == 'object':
            df[column_list] = df[column_list].astype('category') 
        else:
            df[column_list] = handle_numeric_downcast(df[column_list], type_)


def filter_using_multiindex(df_to_be_filtered, orig_df, filter_columns):
  """
  Filter one dataframe using a multiindex from another
  """

  new_index = df_to_be_filtered.set_index(filter_columns).index
  original_index = orig_df.set_index(filter_columns).index
  
  return df_to_be_filtered[new_index.isin(original_index)]


def filter_using_dict(df_pd, column_dict):
  """
  Filter a pandas or dask dataframe using conditions supplied via a dictionary
  """
  filter_conditions = tuple(df_pd[column] == column_dict[column] for column in column_dict.keys())
  
  mask = pd.DataFrame(filter_conditions).transpose().all(axis=1)
  
  return df_pd.loc[mask]


def get_memory_usage(pandas_df):
  """
  Returns the number of bytes used by a pandas dataframe
  """
  return pandas_df.memory_usage().sum()


def print_memory_usage(pandas_df):
  """
  Returns the number of bytes used by a pandas dataframe in a formatted string
  """
  return format_bytes(get_memory_usage(pandas_df))


# TODO requires testing
# def merge_by_concat(big_df, small_df, index_columns=None, how='left'):
#   """
#   Merge two dataframes by concatenation. Avoids trying to join two high-dimensionality dataframes in memory
#   by joining them on an index and then adding in the other columns later.
#   """
  
#   if not index_columns:
#     index_columns = big_df[get_hierarchy() + [TIME_VAR]]
  
#   merged_df = big_df[index_columns].merge(small_df, on=index_columns, how=how)
#   merged_df.drop(index_columns, axis=1, inplace=True)
  
#   return pd.concat([big_df, merged_df], axis=1)


def concatenate_dfs(df1, df2):
  """
  Safely concatenates dataframes, handling for the different treatment between panda and dask implementations
  """
  if is_pandas_df(df1) & is_pandas_df(df2):
    return pd.concat([df1, df2])
  
  elif is_dask_df(df1) & is_dask_df(df2):
    return dd.concat([df1, df2])
    
  else:
    raise InputError("DataFrames of the wrong class or of differnet classes")
  

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


#TODO move down
def convert_pandas_to_dask(df, npartitions=4, partition_size="100MB", distributed=True, *ars, **kwargs):
    """
    Convert a pandas dataframe to a distributed dask dataframe, enabling lazy evaluation that can be prompted using .compute() or .persist()
    """
    from dask.distributed import Client

    dask_df = dd.from_pandas(df, npartitions=npartitions, *args, **kwargs)

    if partition_size:
        dask_df = dask_df.repartition(partition_size=partition_size)
    
    if distributed:    
        global DASK_CLIENT
        DASK_CLIENT = Client()
        
        dask_df = DASK_CLIENT.persist(dask_df)

    return dask_df


def profile_dask_client():
    """
    Print scheduler statistics
    """
    assert DASK_CLIENT, "No dask client has been defined globally."
    return DASK_CLIENT.profile()
