from helpers import *

def make_dir(file_path):
    """
    Create a new directory at a given path
    """
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    
def list_files_in_directory(path):
    """
    Get a list of files within a given directory
    """
    import glob
        
    return glob.glob(path + "/*")


#TODO make this work with dask as well
def _get_extension_dict():
    extensions_dict = {
        '.pkl' : {
            'write':pd.to_pickle, 
            'read':pd.read_pickle
            },
        '.parquet' : {
            'write':pd.to_parquet, 
            'read':pd.read_parquet
            },
        '.csv' : {
            'write':pd.to_csv, 
            'read':pd.read_csv
            }
    }
    
    return extensions_dict 


def save_df(df, path, file_format='.pkl', *args, **kwargs):
    """
    Save file to path using the prescribed format using dask or pandas
    """

    extension_dict = _get_extension_dict()
    
    assert file_format in extensions_dict.keys(), raise NotImplementedError

    save_path = path + file_format
    make_dir(save_path)

    save_func = extensions_dict[file_format]['write']
    df.save_func(save_path, *args, **kwargs)        
    
    print("Saved to %s \n" % save_path)

    
def load_df(path, *args, **kwargs):
    """
    Load file from path using the prescribed format using dask or pandas
    """
    extension_dict = _get_extension_dict()
    
    file_format = path.split('.')[-1]

    assert file_format in extensions_dict.keys(), raise NotImplementedError

    load_path = path + file_format

    load_func = extensions_dict[file_format]['write']
    df.load_func(load_path, *args, **kwargs)        

    return compress_dataframe(df)


def quick_save(df, file_format='.pkl', name=__file__):
    """
    Quickly save a file in an subfolder within your working directory
    """
    save_df(df, './quick_saves/' + name, file_format=file_format)
    

def quick_load(df, file_format='.pkl', name=__file__):
    """
    Quickly load a file from your 'quick_saves' subdirectory
    """
    load_df('./quick_saves/' + name, file_format=file_format)


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


#TODO dependendencies dock based on imports
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


# Boolean tests 
def is_pandas_df(obj):
    return isinstance(obj, pd.DataFrame)


def is_dask_df(obj):
    return isinstance(obj, dd.DataFrame)


def ensure_is_list(obj):
    """
    Return an object in a list if not already wrapped
    """
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


def save_dict(dictionary, path):
    """
    Save dictionary as numpy file
    """
    import os
    np.save(os.path.expanduser(path), dict_obj)

  
def load_dict(path):
    """
    Load dictionary from numpy pickle
    """
    return np.load(path, allow_pickle='TRUE').item()


def load_all_dfs_from_dir(path):
  """
  Load multiple pickles into a single dataframe
  """
  paths_list = list_files_in_directory(path)
  
  first_path = paths_list.pop(0)
  
  df = load_df(first_path)
  
  for path in paths_list:
    try:
      temp_df = load_file(path)
      df = concatenate_dfs(df, temp_df)
    except: 
      warnings.warn("Failed to load" + path, UserWarning)
      
  return df


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



def list_root_files_in_directory(path):
  """
  Print all available files within a folder structure using BFS
  """
  import glob

  def any_exist(avalue, bvalue):
    return any(any(x in y for y in bvalue) for x in avalue)
  
  queue = [path]
  file_paths = []
  
  delimiters = _get_extension_dict().keys()
  
  while len(queue) > 0:
    next_path = queue[0]
    files_in_path = list_files_in_directory(next_path)
    
    if any_exist(delimiters, files_in_path):
      print("Found root:" + next_path)
      file_paths.append(next_path)
    
    else:
      queue += files_in_path
          
    queue.pop(0)

  return file_paths

    
def remove_files_in_dir(path):
  """
  Delete all files within a directory
  """
  path = get_local_path(path) + '/*'
  
  files = glob.glob(path)
  for f in files:
      os.remove(f)
      
  

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


def profile_runtime(func):
  """
  Decorator function you can use to profile a bunch of nested functions.
  Docs: https://docs.python.org/2/library/profile.html#module-cProfile
  Example:

    @profile_python_code
    def profileRunModels(*args, **kwargs):
      return run_models(*args, **kwargs)
  
  """
  
  def wrap(*args, **kwargs):
    import cProfile
        
    profiler = cProfile.Profile()
    profiler.enable()

    result = func(*args, **kwargs)

    profiler.disable()
    profiler.print_stats(sort='cumulative')
      
    return result

  return wrap


def profile_memory_usage(func, *args, **kwargs):
  """
  Profile the amount of memory used in a python function
  """
  from memory_profiler import profile
  
  return profile(func(*args, **kwargs))
  

def format_bytes(size):
  """
  Takes a byte size (int) and returns a formatted, human-interpretable string
  """
  # 2**10 = 1024
  power = 2**10
  n = 0
  power_labels = {0 : ' bytes', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
  while size > power:
      size /= power
      n += 1
  return str(__builtins__.round(size, 2)) + power_labels[n]

#TODO standardize the input name for df
def calc_weighted_average(pd_df, tar_var):
  """
  Returns lambda function for calculating weighted average
  """
  mask = pd_df[tar_var] != 0
  return lambda x: np.ma.average(x[mask], weights=pd_df.loc[x[mask].index, tar_var])


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


def run_function_in_parallel(func, t_split):
  """
  Multiprocess a python function
  """
  from multiprocessing import Pool
  import psutil
  
  N_CORES = psutil.cpu_count()
  
  num_cores = np.min([N_CORES,len(t_split)])
  pool = Pool(num_cores)
  df = pd.concat(pool.map(func, t_split), axis=1)
  pool.close()
  pool.join()
  return df


def correct_suffixes_in_list(input_pd, lst, substring='_index'):
  """
  Remove suffixes from a list of column names if they don't appera in a target dataframe
  """
  dataframe_columns = list(input_pd.columns)
  
  missing_from_dataframe = [item for item in lst if item not in dataframe_columns]
  
  corrected = [item for substring in missing_from_dataframe for item in dataframe_columns if item.startswith(substring)]  
  
  assert len(missing_from_dataframe) == len(corrected), "Not able to correct all missing items in grouping list. \n List 1: % s \n List 2: %s" % (missing_from_dataframe, corrected)
  
  if corrected: 
    lst.extend(corrected)
    lst = [item for item in lst if item not in missing_from_dataframe]
  
  return lst   


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



#TODO move to testing
def exp_increase_df_size(df, n):
  """
  Exponentially multiples dataframe size for feasibility testing
  """
  for i in range(n):
    df = pd.concat([df, df], axis=0)
  
  return df