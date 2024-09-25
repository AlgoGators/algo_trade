import warnings
import functools
from typing import Callable, Tuple, TypeVar, Any, Dict

# Define a generic type for the return value of the function
R = TypeVar('R')

def deprecated(func: Callable[..., R]) -> Callable[..., R]:
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args: Tuple[Any], **kwargs: Dict[str, Any]) -> R:
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(f"Call to deprecated function {func.__name__}.",
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func
