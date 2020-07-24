from python_utilities.helpers import *

def _get_extension_dict():
    extensions_dict = {
        'pandas': {
            '.pkl': {
                'write': pd.to_pickle,
                'read': pd.read_pickle
            },
            '.parquet': {
                'write': pd.to_parquet,
                'read': pd.read_parquet
            },
            '.csv': {
                'write': pd.to_csv,
                'read': pd.read_csv
            }
        },

        'dask': {
            '.pkl': {
                'write': dd.to_pickle,
                'read': dd.read_pickle
            },
            '.parquet': {
                'write': dd.to_parquet,
                'read': dd.read_parquet
            },
            '.csv': {
                'write': dd.to_csv,
                'read': dd.read_csv
            }
        }
    }

    return extensions_dict


def save_df(df, path, file_format='.pkl', pkg='pandas', *args, **kwargs):
    """
    Save file to path using the prescribed format using dask or pandas
    """

    extension_dict = _get_extension_dict()
    
    save_path = path + file_format
    make_dir(save_path)

    save_func = extensions_dict[pkg][file_format]['write']
    df.save_func(save_path, *args, **kwargs)        
    
    print("Saved to %s \n" % save_path)

    
def load_df(path, pkg='pandas', *args, **kwargs):
    """
    Load file from path using the prescribed format using dask or pandas
    """
    extension_dict = _get_extension_dict()
    
    file_format = path.split('.')[-1]

    load_path = path + file_format

    load_func = extensions_dict[pkg][file_format]['write']
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
