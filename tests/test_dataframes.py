
import sys 
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0])))

import numpy as np
import pandas as pd
import dask.dataframe as dd

from python_utilities import dataframes, testing, helpers


def test_remove_blank_cols():
    # tested this function as part of test_dask_io() in test_io.py
    pass


def test_compress_dataframe():
    test_df = testing.get_test_example()    
    initial_memory = dataframes.get_memory_usage(test_df)

    dataframes.compress_dataframe(test_df)
    new_memory = dataframes.get_memory_usage(test_df)
    
    assert new_memory < initial_memory


def test_filter_using_multiindex():
    test_df = testing.get_test_example()    

    filter_df = test_df[test_df['product'] == 'Prod_3']

    result_df = dataframes.filter_using_multiindex(test_df, filter_df, ['product'])

    assert result_df.equals(filter_df)


def test_filter_using_dict():
    test_df = testing.get_test_example()   
    filter_df = test_df[(test_df['product'] == 'Prod_3') & (test_df['category'] == 'Cat_1')]
    result_df = dataframes.filter_using_dict(test_df, {"product":"Prod_3", 'category':'Cat_1'}) 

    assert result_df.equals(filter_df)


def test_merge_by_concat():
    test_df = testing.get_test_example()   
    small_df = pd.DataFrame([
            ['Cat_1', "A"], 
            ['Cat_2', "B"]
        ],
        columns=['category', 'mapping'])

    merged_df = dataframes.merge_by_concat(test_df, small_df, index_columns=['category'])

    answer = ['A']*4 + ['B']*4
    result = list(merged_df['mapping'].values)
    assert  result == answer


def test_concatenate_dfs():
    test_df = testing.get_test_example()   
    duplicate_df = testing.get_test_example()

    concat_df = dataframes.concatenate_dfs(test_df, duplicate_df)

    answer = list(test_df[['week']]) * 2
    result = list(concat_df[['week']])

    assert all([a == b for a, b in zip(answer, result)])


def test_print_memory_usage():
    test_df = testing.get_test_example()   
    memory = dataframes.print_memory_usage(test_df)
    assert helpers.is_string(memory)   
    assert memory[-2:] == "KB"


def test_index_features():
    test_df = testing.get_test_example()   

    indexed_df = dataframes.index_features(test_df)

    answer_df = pd.DataFrame([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [3, 0, 0, 0, 0],
        [4, 1, 1, 0, 0],
        [5, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
    ], columns=['week_index', 'category_index', 'product_index', 'state_index', 'store_index'])

    indexed_columns = [col for col in indexed_df.columns if "_index" in col]
    assert (indexed_df[indexed_columns].values == answer_df.values).all()

  
def test_deindex_features():
    # ensures that the right index mapping exists
    test_index_features()

    initial_df = pd.DataFrame([
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [2, 0, 0, 0, 0],
        [3, 0, 0, 0, 0],
        [4, 1, 1, 0, 0],
        [5, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
    ], columns=['week_index', 'category_index', 'product_index', 'state_index', 'store_index'])

    result_df = dataframes.deindex_features(initial_df)

    answer_df = testing.get_test_example()[['week', 'category', 'product', 'state', 'store']]   

    assert (result_df.values == answer_df.values).all()


if __name__ == "__main__":
    test_remove_blank_cols()
    test_compress_dataframe()
    test_filter_using_multiindex()
    test_filter_using_dict()
    test_merge_by_concat()
    test_concatenate_dfs()
    test_print_memory_usage()
    test_index_features()
    test_deindex_features()
