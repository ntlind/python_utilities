import sys
import pandas as pd
import numpy as np
import os

sys.path.append(os.path.join(os.path.dirname(sys.path[0])))
from python_utilities import dataframes, testing, utils  # noqa


def test_compress_dataframe():
    test_df = testing.get_test_example()
    initial_memory = dataframes.get_memory_usage(test_df)

    dataframes.compress_dataframe(test_df)
    new_memory = dataframes.get_memory_usage(test_df)

    assert new_memory < initial_memory


def test_filter_using_multiindex():
    test_df = testing.get_test_example()

    filter_df = test_df[test_df["product"] == "Prod_3"]

    result_df = dataframes.filter_using_multiindex(test_df, 
                                                   filter_df,
                                                   ["product"])

    assert result_df.equals(filter_df)


def test_filter_using_dict():
    test_df = testing.get_test_example()
    filter_df = test_df[
        (test_df["product"] == "Prod_3") & (test_df["category"] == "Cat_1")
    ]
    result_df = dataframes.filter_using_dict(
        test_df, {"product": "Prod_3", "category": "Cat_1"}
    )

    assert result_df.equals(filter_df)


def test_merge_by_concat():
    test_df = testing.get_test_example()
    small_df = pd.DataFrame(
        [["Cat_1", "A"], ["Cat_2", "B"]], columns=["category", "mapping"]
    )

    merged_df = dataframes.merge_by_concat(test_df, small_df,
                                           index_cols=["category"])

    answer = ["A"] * 4 + ["B"] * 4
    result = list(merged_df["mapping"].values)
    assert result == answer


def test_concatenate_dfs():
    test_df = testing.get_test_example()
    duplicate_df = testing.get_test_example()

    concat_df = dataframes.concatenate_dfs(test_df, duplicate_df)

    answer = list(test_df[["datetime"]]) * 2
    result = list(concat_df[["datetime"]])

    assert all([a == b for a, b in zip(answer, result)])


def test_print_memory_usage():
    test_df = testing.get_test_example()
    memory = dataframes.print_memory_usage(test_df)
    assert utils.is_string(memory)
    assert memory[-2:] == "KB"


def test_index_features():
    test_df = testing.get_test_example()

    indexed_df = dataframes.index_features(test_df)

    answer_df = pd.DataFrame(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        columns=["category_index", "product_index", 
                 "state_index", "store_index"],
    )

    indexed_columns = [col for col in indexed_df.columns if "_index" in col]
    assert (indexed_df[indexed_columns].values == answer_df.values).all()


def test_deindex_features():
    # ensures that the right index mapping exists
    test_index_features()

    initial_df = pd.DataFrame(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        columns=["category_index", "product_index", 
                 "state_index", "store_index"],
    )

    result_df = dataframes.deindex_features(initial_df)

    answer_df = testing.get_test_example()[["category", "product", 
                                            "state", "store"]]

    assert (result_df.values == answer_df.values).all()


def test_convert_pandas_to_dask():
    # also tested as part of integration tests in test_utils.py

    test_pd = testing.get_test_example()

    test_dd = dataframes.convert_pandas_to_dask(test_pd)

    assert utils.is_dask_df(test_dd)


def test_distribute_dask_df():
    test_pd = testing.get_test_example()

    test_dd = dataframes.convert_pandas_to_dask(test_pd)

    test_dd = dataframes.distribute_dask_df(test_dd)
# TODO fix globals so this isn't necessary
    assert isinstance(dataframes.profile_dask_client(), dict)


def test_profile_dask_client():
    assert isinstance(dataframes.profile_dask_client(), dict)


def test_get_memory_usage():
    test_pd = testing.get_test_example()
    assert 3000 > dataframes.get_memory_usage(test_pd) > 1000


def test_auto_convert_datetime():
    test_df = testing.get_test_example(convert_dtypes=False)
    assert test_df["datetime"].dtype in ["object"]

    test_df = dataframes.auto_convert_datetime(test_df)

    assert utils.is_datetime_series(test_df["datetime"])


def test_get_columns_of_type():
    test_pd = testing.get_test_example()
    float_columns = ["float_col"]
    int_columns = ["sales_int"]
    object_cols = ["category", "product", "state", "store"]

    assert dataframes.get_columns_of_type(test_pd, "float") == float_columns
    assert dataframes.get_columns_of_type(test_pd, "integer") == int_columns

    assert set(dataframes.get_columns_of_type(test_pd, "object")) == \
        set(object_cols)


def test_remove_blank_cols():
    test_pd = testing.get_test_example()
    test_pd["Unnamed"] = 0
    assert "Unnamed" in list(test_pd.columns)

    result = dataframes.remove_blank_cols(test_pd)
    assert "Unnamed" not in list(result.columns)


def test_calc_rolling_agg():
    input_df = testing.get_test_example()
    hierarchy = ['category', 'product', 'state', 'store']
    
    # test 1: 2-day rolling average
    rolling_mean_df = dataframes.calc_rolling_agg(df=input_df,
                                                  hierarchy=hierarchy, 
                                                  rolling_window=2, 
                                                  target_var='sales_int', 
                                                  agg_func='mean')
    
    answer = pd.Series([
        113,
        (10000 + 113)/2,
        (10000 + 102)/2,
        (102 + 123)/2,
        5,
        (800 + 5)/2,
        (800 + 0)/2,
        (0 + -20)/2
    ])
    result = (rolling_mean_df['mean_2_sales_int'])
    assert (answer == result).all(), print(answer, result, input_df) 

    # test 2: three-day rolling average
    rolling_mean_df = dataframes.calc_rolling_agg(df=input_df,
                                                  hierarchy=hierarchy,
                                                  rolling_window=3,
                                                  target_var='sales_int',
                                                  agg_func='mean')

    answer = pd.Series([
        113,
        (10000 + 113)/2,
        (10000 + 102 + 113)/3,
        (10000 + 102 + 123)/3,
        5,
        (800 + 5)/2,
        (800 + 5 + 0)/3,
        (0 + -20 + 800)/3
    ])

    result = (rolling_mean_df['mean_3_sales_int'])
    assert (answer == result).all(), print(answer, result, input_df) 


def test_create_outlier_mask():
    input_df = testing.get_test_example()

    # use custom sales column since the outliers are too pronounced
    input_df['sales_int'] = [4, 5, 1, 2, 4, 6, 100, 4]

    # test 1: flag outliers where a sales value is greater than two st devs.
    grouped_flag_mask = dataframes.create_outlier_mask(
        input_df,
        target_var='sales_int',
        grouping_cols='product', 
        number_of_stds=1
        )

    answer = [True, False, False, True, True, True, False, True]
    assert((grouped_flag_mask == answer).all())

    # test 2: flag outliers for salves overall
    flag_mask = dataframes.create_outlier_mask(
        input_df,
        target_var='sales_int',
        number_of_stds=1
        )

    answer = [True]*6 + [False, True]
    assert((flag_mask == answer).all())


def test_check_key_mismatches():
    keys = ['product', 'state', 'store']
    
    first_test_df = testing.get_test_example()
    second_test_df = pd.DataFrame(
           [
            ["Prod_3", "CA", "Store_1", 2],
            ["Prod_3", "TX", "Store_1", 23.12],
           ],
           columns=keys + ['price']
    )

    answer = pd.DataFrame(
        [
            ["Prod_3", "CA", "Store_1", 'both'],
            ["Prod_4", "CA", "Store_1", 'left_only'],
            ["Prod_3", "TX", "Store_1", 'right_only'],

        ],
        columns=['product', 'stage', 'store', '_merge']
    )

    result = dataframes.check_key_mismatches(
        df1=first_test_df,
        df2=second_test_df,
        keys=keys,
        )

    assert np.array_equal(answer.values, result.values)


def test_count_elements_across_columns():
    input_df = pd.DataFrame(
        [["32", "12"], ["12", None], ["17", "21", "41"], ["41"], ["41"]],
        columns=["class00", "class01", "class002"]
    )

    expected = pd.Series(
        [3, 2, 1, 1, 1],
        index=["41", "12", "21", "32", "17"]
    )
    
    result = dataframes.count_elements_across_columns(input_df)
    
    assert (result.values == expected.values).all()



def test_expand_delimited_multiclass_series():
    input_series = pd.Series(
        ["32_12", "12", "17_21_41"],
        name="class"
    )

    expected = pd.DataFrame(
        [["32", "12"], ["12", None], ["17", "21", "41"]],
        columns=["class00", "class01", "class02"]
    )

    result = dataframes.expand_delimited_multiclass_series(input_series, "_")

    assert result.equals(expected)


if __name__ == "__main__":
    test_check_key_mismatches()
    test_remove_blank_cols()
    test_compress_dataframe()
    test_filter_using_multiindex()
    test_filter_using_dict()
    test_merge_by_concat()
    test_concatenate_dfs()
    test_print_memory_usage()
    test_index_features()
    test_deindex_features()
    test_auto_convert_datetime()
    test_get_memory_usage()
    test_convert_pandas_to_dask()
    test_get_columns_of_type()
    test_distribute_dask_df()
    test_profile_dask_client()
    test_calc_rolling_agg()
    test_create_outlier_mask()
    test_count_elements_across_columns()
    test_expand_delimited_multiclass_series()