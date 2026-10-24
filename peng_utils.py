import functools

"""Utility functions used throughout the peng-robinson and moody programs
"""

# We memoize some functions so that they do not get repeatedly called with
# the same arguments. Yet still be retain a more obvius way of writing the program.

# HOWEVER:
# Yes, but your current custom memoize decorator in peng_utils.py has a major 
# performance bottleneck: it builds a string via str(args) + str(kwargs) on every 
# single call. String serialization of tuples and dictionaries is computationally 
# expensive in Python and can completely negate the performance gain of caching 
# inside tight nested loops. 

 # Furthermore, using @functools.lru_cache or a specialized  fast key-generator will drastically speed up both your Coutinho-Tsonopoulos model and any virial evaluation.

def memoize(func):
    """Standard memoize function to use in a decorator, see
    https://medium.com/@nkhaja/memoization-and-decorators-with-python-32f607439f84
    """
    cache = func.cache = {}
    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return memoized_func