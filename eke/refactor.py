import warnings as _warnings

def new_to_old(new_func, old_name):
    def old_func(*args):
        _warnings.warn("{} is deprecated. Please use {} instead".format(
            old_name, new_func.__name__),
                       DeprecationWarning)
        return new_func(*args)
    return old_func
