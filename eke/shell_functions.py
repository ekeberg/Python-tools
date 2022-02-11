"""Implements a few shell features that are not available natively in python"""
import os as _os
import errno as _errno


def mkdir_p(path):
    """Create a directory without returning an error if it already exists."""
    try:
        _os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == _errno.EEXIST and _os.path.isdir(path):
            pass
        else:
            raise IOError(f"Problem creating directory: {path}")
