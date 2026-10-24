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

def freeze(obj):
    if isinstance(obj, dict):
        return tuple(sorted((k, freeze(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(freeze(item) for item in obj)
    return obj

def memoize(func):
    cache = func.cache = {}
    
    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        try:
            frozen_args = tuple(freeze(arg) for arg in args)
            frozen_kwargs = tuple(sorted((k, freeze(v)) for k, v in kwargs.items())) if kwargs else ()
            key = (frozen_args, frozen_kwargs)
            
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]
        except TypeError:
            # Fallback if any argument contains an unhashable type that couldn't be frozen
            return func(*args, **kwargs)
            
    return memoized_func