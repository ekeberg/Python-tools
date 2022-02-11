import os as _os
import numpy as _numpy


def get_array_index(noerr=False):
    try:
        return int(_os.environ["SLURM_ARRAY_TASK_ID"])
    except KeyError:
        if noerr:
            return 0
        else:
            raise EnvironmentError("This process is not part of a slurm array")


def get_array_size(noerr=False):
    try:
        return (int(_os.environ["SLURM_ARRAY_TASK_MAX"])
                - int(_os.environ["SLURM_ARRAY_TASK_MIN"]))
    except KeyError:
        if noerr:
            return 1
        else:
            raise EnvironmentError("This process is not part of a slurm array")


def get_array_min(noerr=False):
    try:
        return int(_os.environ["SLURM_ARRAY_TASK_MIN"])
    except KeyError:
        if noerr:
            return 0
        else:
            raise EnvironmentError("This process is not part of a slurm array")


def get_array_max(noerr=False):
    try:
        return int(_os.environ["SLURM_ARRAY_TASK_MAX"])
    except KeyError:
        if noerr:
            return 1
        else:
            raise EnvironmentError("This process is not part of a slurm array")


def index_to_multiple(index, lengths):
    """Provide a single index (such as array number) and get
    multiple indices back."""
    lengths = _numpy.int64(_numpy.array(lengths))
    if index < 0 or index >= _numpy.prod(lengths):
        raise ValueError("Index is out of range")
    result = []
    for i, length in enumerate(lengths):
        div = _numpy.prod(lengths[i+1:])
        mod = length
        result.append((index // div) % mod)
    return tuple(result)


def multiple_to_index(multiple, lengths):
    """Provide a tuple with multiple indices (such as array number) and get
    a single index back."""
    if (
            (_numpy.array(multiple) >= _numpy.array(lengths)).max() or
            (_numpy.array(multiple) < 0).max()
    ):
        raise ValueError("indices are out of range")
    lengths = _numpy.int64(_numpy.array(lengths))
    index = 0
    for i, length in enumerate(lengths):
        index += _numpy.prod(lengths[i+1:]) * multiple[i]
    return index
