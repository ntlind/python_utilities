
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
  

def exp_increase_df_size(df, n):
  """
  Exponentially multiples dataframe size for feasibility testing
  """
  for i in range(n):
    df = pd.concat([df, df], axis=0)
  
  return df