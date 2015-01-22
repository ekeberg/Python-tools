"""Implements a few shell features that are not available natively in python"""
import os, errno

def mkdir_p(path):
    """Create a directory without returning an error if it already exists."""
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise
