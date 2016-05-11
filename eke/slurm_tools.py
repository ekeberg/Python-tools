import os as _os

def get_array_index():
    try:
        return _os.environ["SBATCH_ARRAY_TASK_ID"]
    except KeyError:
        raise EnvironmentError("This process is not part of a slurm array")

def get_array_size():
    try:
        return _os.environ["SBATCH_ARRAY_TASK_MAX"] - _os.environ["SBATCH_ARRAY_TASK_MIN"]
    except KeyError:
        raise EnvironmentError("This process is not part of a slurm array")

def get_array_min():
    try:
        return _os.environ["SBATCH_ARRAY_TASK_MIN"]
    except KeyError:
        raise EnvironmentError("This process is not part of a slurm array")

def get_array_max():
    try:
        return _os.environ["SBATCH_ARRAY_TASK_MAX"]
    except KeyError:
        raise EnvironmentError("This process is not part of a slurm array")


