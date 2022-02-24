"""Implements a few shell features that are not available natively in python"""
import os as _os
import errno as _errno
import pathlib as _pathlib
import re as _re

def mkdir_p(path):
    """Create a directory without returning an error if it already exists."""
    try:
        _os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == _errno.EEXIST and _os.path.isdir(path):
            pass
        else:
            raise IOError(f"Problem creating directory: {path}")



def remove_all_but_last(path, ext=None, quiet=False):
    path = _pathlib.Path(path)
    dir_part = path.parent
    file_part = path.name

    file_and_index = []
    for f in dir_part.glob("*"):
        if ext is not None:
            match = _re.search(f"{file_part}([0-9]*)\.{ext}$", f.as_posix())
        else:
            match = _re.search(f"{file_part}([0-9]*)\.", f.as_posix())
        if match:
            index = int(match.groups()[0])
            file_and_index.append((f, index))

    file_and_index = sorted(file_and_index, key=lambda x: x[1])

    if not quiet:
        print(f"Removing {len(file_and_index[:-1])} files")

    for f in file_and_index[:-1]:
        f[0].unlink()



