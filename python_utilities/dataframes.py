import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
from python_utilities import helpers, io
import pytest


def compress_dataframe(df):
    """
    Downcast dataframe and convert objects to categories to save memory
    """

    def handle_numeric_downcast(array, type_):
        return array.apply(pd.to_numeric, downcast=type_)

    numeric_lookup_dict = {
        "integer": np.integer,
        "float": np.floating,
        "object": "object",
    }

    for type_ in ["integer", "float", "object"]:
        column_list = get_columns_of_type(df, numeric_lookup_dict[type_])
        if not column_list:
            continue

        if type_ == "object":
            df[column_list] = df[column_list].astype("category")
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
    Filter a pandas or dask dataframe using conditions supplied via a dict
    """
    filter_conditions = tuple(
        df_pd[column] == column_dict[column] for column in column_dict.keys()
    )

    mask = pd.DataFrame(filter_conditions).transpose().all(axis=1)

    return df_pd.loc[mask]


def remove_blank_cols(df):
    """
    Remove blank 'Unnamed' columns that occassionally appear when importing csv
    """
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


def get_columns_of_type(df, type_):
    """
    Generate a list of columns from a given df that match a certain type
    """
    return list(df.select_dtypes(include=type_).columns)


def get_memory_usage(pandas_df):
    """
    Returns the number of bytes used by a pandas dataframe
    """
    return pandas_df.memory_usage(deep=True).sum()


def print_memory_usage(pandas_df):
    """
    Returns the number of bytes used by a pd df in a formatted string
    """
    return helpers.format_bytes(get_memory_usage(pandas_df))


def merge_by_concat(big_df, small_df, index_cols, how="left"):
    """
    Merge two dataframes by concatenation. Avoids trying to join two
    high-dimensionality dataframes in memory by joining them on an 
    index and then adding in the other columns later.
    """

    merged_df = big_df[index_cols].merge(small_df, on=index_cols, how=how)
    merged_df.drop(index_cols, axis=1, inplace=True)

    return pd.concat([big_df, merged_df], axis=1)


def concatenate_dfs(df1, df2):
    """
    Safely concatenates dataframes, handling for the different treatment 
    between panda and dask implementations
    """
    from python_utilities.helpers import is_dask_df, is_pandas_df

    if is_pandas_df(df1) & is_pandas_df(df2):
        return pd.concat([df1, df2])

    elif is_dask_df(df1) & is_dask_df(df2):
        return dd.concat([df1, df2])

    else:
        raise Exception("DataFrames of wrong class or of different classes")


def index_features(df, path_spec=None, name="index_mapping"):
    """
    Index string columns and return a DataFrame that is ready for modeling.
    """

    def factorize_columns(pandas_df, columns):
        """
        Transforms string or category columns into factors (e.g., replacing 
        all of the strings with an integer encoding, i.e. dummy variable)
        and outputs a label dict for future use
        """

        def factorize_column(series):
            array, labels = series.factorize()
            return {"array": array, "labels": list(labels)}

        label_dict = {}

        for column in columns:
            factorize_dict = factorize_column(pandas_df[column])
            pandas_df[column + "_index"] = pd.to_numeric(
                factorize_dict["array"], downcast="integer"
            )

            label_dict.update({column + "_index": factorize_dict["labels"]})

        return pandas_df, label_dict

    string_features = get_columns_of_type(df, ["object", "category"])
    assert string_features, "No object or category columns found."

    factorized_df, labels = factorize_columns(df, string_features)
    new_columns = [col + "_index" for col in string_features]

    mapping_array = factorized_df[string_features + new_columns]\
        .drop_duplicates()

    path = os.path.abspath(
        os.path.join(os.path.dirname("."), "mappings/" + name + ".pkl")
    )

    io.save_df(mapping_array, path)

    factorized_df.drop(string_features, inplace=True, axis=1)

    return factorized_df


def deindex_features(df, name="index_mapping"):
    """
    Deindex columns in a dataframe using a pre-saved array
    """
    path = os.path.abspath(
        os.path.join(os.path.dirname("."), "mappings/" + name + ".pkl")
    )

    mapping_array = io.load_df(path)

    indexed_features = [col for col in list(df.columns) if "_index" in col]
    mapped_indexed_features = [
        col for col in list(mapping_array.columns) if "_index" in col
    ]

    missing_from_mapping_array = [
        col for col in indexed_features if col not in mapped_indexed_features
    ]
    if missing_from_mapping_array:
        print(
            "The following columns are missing in your mapping array \
                and won't be deindexed: %s"
            % missing_from_mapping_array
        )

    deindexed_df = df.merge(mapping_array, 
                            on=mapped_indexed_features, 
                            how="inner")                           
    deindexed_df = deindexed_df.drop(mapped_indexed_features, axis=1)

    return deindexed_df


@pytest.mark.skip(reason="requires cluster to test")
def distribute_dask_df(dask_df):
    """
    Distribute a dask dataframe over a client that's accessible via
    the global DASK_CLIENT
    """
    from distributed import Client

    global DASK_CLIENT
    DASK_CLIENT = Client()

    dask_df = DASK_CLIENT.persist(dask_df)

    return dask_df


@pytest.mark.skip(reason="requires cluster to test")
def profile_dask_client():
    """
    Print scheduler statistics
    """

    assert DASK_CLIENT, "No dask client has been defined globally."
    return DASK_CLIENT.profile()


@pytest.mark.skip(reason="shortcut for default dask behavior")
def convert_pandas_to_dask(df, npartitions=4, partition_size="100MB",
                           *args, **kwargs):

    """
    Convert a pandas dataframe to a distributed dask dataframe, enabling
    lazy evaluation that can be prompted using .compute() or .persist()
    """

    dask_df = dd.from_pandas(df, npartitions=npartitions)

    if partition_size:
        dask_df = dask_df.repartition(partition_size=partition_size)

    return dask_df


def auto_convert_datetime(df):
    """
    Try to auto-convert a datetime column (usually upon import)
    """
    for col in df.columns:
        if df[col].dtype in ['object']:
            try:
                df[col] = pd.to_datetime(df[col])
            except ValueError:
                pass
    return df


def calc_rolling_agg(df, target_var, rolling_window, 
                     hierarchy, agg_func="mean", min_periods=1):
    """
    Calculates rolling aggregated count of orders by rolling window timeframe.
    This function assumes that ther aren't any time gaps within your hierarchy.
    """
    def rolling_calc(df, calc_col, rolling_window, agg_func):
        column_name = f'{agg_func}_{rolling_window}_{target_var}'

        df[column_name] = df[calc_col].rolling(window=rolling_window, 
                                               min_periods=min_periods)\
                                      .agg(agg_func)
                                
        return df

    rolling_df = df.groupby(hierarchy).apply(rolling_calc, target_var, 
                                             rolling_window, agg_func)

    return rolling_df


def create_outlier_mask(df, target_var, number_of_stds, grouping_cols=None):
    """
    Create a row-wise mask to filter-out outliers based on target_var. 
    Optionally allows you to filter outliers by group for hier. data.
    """
    def flag_outliers_within_groups(df, target_var, 
                                    grouping_cols, number_of_stds):
        groups = df.groupby(grouping_cols)
        means = groups[target_var].transform('mean')
        stds = groups[target_var].transform('std')

        upper_bound = means + stds * number_of_stds
        lower_bound = means - stds * number_of_stds

        return df[target_var].between(lower_bound, upper_bound)

    def flag_outliers_without_groups(df, target_var, number_of_stds):
        
        mean_val = df[target_var].mean() 
        std_val = df[target_var].std()

        upper_bound = (mean_val + (std_val * number_of_stds))
        lower_bound = (mean_val - (std_val * number_of_stds))
        
        return (df[target_var] > lower_bound) & (df[target_var] < upper_bound)
        
    if grouping_cols:
        mask = flag_outliers_within_groups(
            df=df, target_var=target_var, 
            number_of_stds=number_of_stds, grouping_cols=grouping_cols
            )

    else:
        mask = flag_outliers_without_groups(
                df=df, target_var=target_var, 
                number_of_stds=number_of_stds
                )

    return mask
