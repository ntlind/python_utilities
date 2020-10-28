import pandas as pd
import numpy as np
import pytest


@pytest.mark.skip(reason="shortcut for default pandas behavior")
def get_test_example(convert_dtypes=True):
    """
    Return a made-up dataframe that can be used for testing purposes
    """

    column_names = [
        "datetime",
        "category",
        "sales_int",
        "product",
        "state",
        "store",
        "float_col",
    ]

    example = pd.DataFrame(
        [
            ["2020-01-01", "Cat_1", 113, "Prod_3", "CA", "Store_1", -0.4],
            ["2020-01-02", "Cat_1", 10000, "Prod_3", "CA", "Store_1", -1],
            ["2020-01-03", "Cat_1", 102, "Prod_3", "CA", "Store_1", 1],
            ["2020-01-04", "Cat_1", 123, "Prod_3", "CA", "Store_1", np.nan],
            ["2019-12-30", "Cat_2", 5, "Prod_4", "CA", "Store_1", -0.9],
            ["2019-12-31", "Cat_2", 800, "Prod_4", "CA", "Store_1", 0.8],
            ["2020-01-01", "Cat_2", 0, "Prod_4", "CA", "Store_1", np.nan],
            ["2020-01-02", "Cat_2", -20, "Prod_4", "CA", "Store_1", np.nan],
        ],
        columns=column_names,
    )

    if convert_dtypes:
        example["datetime"] = pd.to_datetime(example["datetime"])
        example["sales_int"] = example['sales_int'].astype('int')

    return example


@pytest.mark.skip(reason="shortcut for default timer")
def print_runtime(func, create_global_dict=True):
    """
    A timer decorator that creates a global dict for reporting times across multiple runs
    """
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        import time
        from collections import defaultdict

        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()

        runtime = end - start

        print(f"The runtime for {func.__name__} took {round(runtime, 2)} \
            seconds to complete")

        return value

    return function_timer


@pytest.mark.skip(reason="shortcut for default profiler")
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
        profiler.print_stats(sort="cumulative")

        return result

    return wrap


@pytest.mark.skip(reason="shortcut for default behavior")
def profile_memory_usage(func, *args, **kwargs):
    """
    Profile the amount of memory used in a python function
    """
    from memory_profiler import profile

    return profile(func(*args, **kwargs))


@pytest.mark.skip(reason="shortcut for default behavior")
def profile_line_runtime(func, *args, **kwargs):
    """
    Profile the runtime for individual lines in a given func
    """
    from line_profiler import LineProfiler

    lp = LineProfiler()
    lp_wrapper = lp(func)
    lp_wrapper(*args, **kwargs)
    
    return lp.print_stats()


def exp_increase_df_size(df, n):
    """
    Exponentially multiples dataframe size for feasibility testing
    """
    for i in range(n):
        df = pd.concat([df, df], axis=0)

    return df
