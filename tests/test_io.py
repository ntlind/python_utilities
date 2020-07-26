
import pytest
import sys 
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import numpy as np
import pandas as pd
import dask.dataframe as dd
from python_utilities import io, helpers

def test_pandas_io():
    input_path = os.path.abspath(os.path.join(os.path.curdir, 'tests/data/sample_data.pkl'))
    pd_output_path = os.path.abspath(os.path.join(os.path.curdir, 'tests/data/test_sample.pkl'))
    
    # loading
    example_df = io.load_df(input_path, pkg='pandas')
    
    # saving
    io.save_df(example_df, pd_output_path, pkg='pandas')
    os.remove(pd_output_path)
    print(f'Removed from {pd_output_path}')
    
    # quick_saving
    io.quick_save(example_df)
    
    # quick_loading
    quick_df = io.quick_load()


def test_dask_io():
    from python_utilities.dataframes import remove_blank_cols

    input_path = os.path.abspath(os.path.join(os.path.curdir, 'tests/data/sample_data.csv'))
    pd_output_path = os.path.abspath(os.path.join(os.path.curdir, 'tests/data/test_sample.csv'))

    example_df = io.load_df(input_path, pkg='dask', file_format='.csv')
    
    io.save_df(example_df, pd_output_path, pkg='dask', file_format='.csv', single_file=True)

    io.quick_save(example_df, file_format='.csv', pkg='dask', single_file=True)

    quick_df = io.quick_load(file_format='.csv', pkg='dask')
    quick_df = remove_blank_cols(quick_df.compute())

    example_df = remove_blank_cols(example_df.compute())

    assert example_df.shape == quick_df.shape

    os.remove(pd_output_path)
    print(f'Removed from {pd_output_path}')


def test_dict_io():
    path = os.path.abspath(os.path.join(os.path.dirname('.'), 'tests/test_dict.npy'))

    input_dict = {
        'A':1,
        'B':2,
        'C':3
        }

    io.save_dict(input_dict, path)

    result_dict = io.load_dict(path)

    assert input_dict == result_dict 

    os.remove(path)
    

def test_load_all_dfs_from_path():
    pass
    # path = os.path.abspath(os.path.join(os.path.dirname('.'), 'tests/data/'))

    # df = io.load_all_dfs_from_dir(path)


test_pandas_io()
test_dask_io()
test_dict_io()
test_load_all_dfs_from_path()

