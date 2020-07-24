
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


def is_pandas_df(obj):
    """
    Check if an object is a pandas dataframe
    """
    return isinstance(obj, pd.DataFrame)


def is_dask_df(obj):
    """
    Check if an object is a dask dataframe
    """
    return isinstance(obj, dd.DataFrame)


def ensure_is_list(obj):
    """
    Return an object in a list if not already wrapped. Useful when you want to treat an object as a collection, \
        even when the user passes a string
    """
    if not isinstance(obj, list):
        return [obj]
    else:
        return obj


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


def run_function_in_parallel(func, t_split):
  """
  Multiprocess a python function using Pool
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