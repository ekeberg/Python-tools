import h5py as _h5py
import numpy as _numpy


def dictionary_to_hdf5(file_name, dict_in):
    """Write a dictionary to an HDF5 file."""
    def write_group(group_handle, dict_in):
        for name, value in dict_in.items():
            if isinstance(value, dict):
                new_group = group_handle.create_group(name)
                write_group(new_group, value)
            else:
                group_handle.create_dataset(name, data=_numpy.array(value))
    with _h5py.File(file_name, "w") as file_handle:
        write_group(file_handle, dict_in)


def hdf5_to_dictionary(file_name):
    """Read the entire contents of an HDF5 file into a dictionary."""
    def read_group(group):
        return_dictionary = {}
        for name, value in group.items():
            if isinstance(value, _h5py.Dataset):
                return_dictionary[name] = value[...]
            elif isinstance(value, _h5py.Group):
                return_dictionary[name] = read_group(value)
            else:
                raise IOError(f"Can't handle {type(value)} in file "
                              f"{file_name}")
        return return_dictionary
    with _h5py.File(file_name, "r") as file_handle:
        return_dictionary = read_group(file_handle)
    return return_dictionary


def save_numpy(filename, array):
    """Save a numpy array to an HDF5 file."""
    raise DeprecationWarning("Use write_dataset instead")
    write_dataset(filename, "data", array)


def read_numpy(filename):
    """Read a numpy array from an HDF5 file."""
    raise DeprecationWarning("Use read_dataset instead")
    return read_dataset(filename, "data")


def read_dataset(filename, loc):
    """Read a dataset from an HDF5 file."""
    with _h5py.File(filename, "r") as file_handle:
        array = file_handle[loc][...]
    return array


def write_dataset(filename, loc, data):
    """Write a dataset to an HDF5 file."""
    with _h5py.File(filename, "a") as file_handle:
        file_handle.create_dataset(loc, data=data)
