import os
import dask.dataframe as dd
import pandas as pd  
import numpy as np
from python_utilities import utils


def _get_pd_io_methods():
    """
    Returns a dictionary of functions to save/load files using pandas
    """
    method_dict = {
        '.pkl': {
            'write': pd.DataFrame.to_pickle,
            'read': pd.read_pickle
        },
        '.parquet': {
            'write': pd.DataFrame.to_parquet,
            'read': pd.read_parquet
        },
        '.csv': {
            'write': pd.DataFrame.to_csv,
            'read': pd.read_csv
        }
    }

    return method_dict


def _get_dd_io_methods():
    """
    Returns a dictionary of functions to save/load files using dask
    """
    method_dict = {
        '.pkl': {
            'write': NotImplementedError,
            'read': NotImplementedError
        },
        '.parquet': {
            'write': dd.DataFrame.to_parquet,
            'read': dd.read_parquet
        },
        '.csv': {
            'write': dd.DataFrame.to_csv,
            'read': dd.read_csv
        }
    }

    return method_dict


def save_df(df, path, file_format='.pkl', pkg='pandas', *args, **kwargs):
    """
    Save file to path using the prescribed format using dask or pandas
    """

    methods_dict = {
        'pandas': _get_pd_io_methods,
        'dask': _get_dd_io_methods
    }[pkg]()

    if file_format not in path:
        path = path + file_format
    
    utils.make_dir(path)

    save_func = methods_dict[file_format]['write']
    save_func(df, path, *args, **kwargs)

    print("Saved to %s \n" % path)


def load_df(path, file_format='.pkl', pkg='pandas', *args, **kwargs):
    """
    Load file from path using the prescribed format using dask or pandas
    """
    methods_dict = {
        'pandas': _get_pd_io_methods,
        'dask': _get_dd_io_methods
    }[pkg]()

    assert '.' in path, "Are you missing a file extension in your path?"
    file_format = '.' + path.split('.')[-1]

    load_func = methods_dict[file_format]['read']
    df = load_func(path, *args, **kwargs)

    return df


def quick_save(df, name='quick_save', file_format='.pkl', 
               pkg='pandas', *args, **kwargs):
    """
    Quickly save a file in an subfolder within your working directory
    """
    path = os.path.abspath(os.path.join(os.path.dirname('.'), 
                                        'quick_saves/' + name))
    save_df(df, path, file_format=file_format, pkg=pkg, *args, **kwargs)


def quick_load(name='quick_save', file_format='.pkl', pkg='pandas'):
    """
    Quickly load a file from your 'quick_saves' subdirectory
    """
    path = os.path.abspath(os.path.join(os.path.dirname('.'), 
                                        'quick_saves/' + name + file_format))
    df = load_df(path, file_format=file_format, pkg=pkg)
    return df


def save_dict(dict_obj, path):
    """
    Save dictionary as numpy file
    """
    assert path[-4:] == '.npy', 'Missing the .npy extension!'

    np.save(os.path.expanduser(path), dict_obj)


def load_dict(path):
    """
    Load dictionary from numpy pickle
    """
    assert path[-4:] == '.npy', 'Missing the .npy extension!'

    return np.load(path, allow_pickle='TRUE').item()