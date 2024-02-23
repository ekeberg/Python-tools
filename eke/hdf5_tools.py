import re
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


def write_dataset(filename, loc, data, overwrite=False):
    """Write a dataset to an HDF5 file."""
    with _h5py.File(filename, "a") as file_handle:
        if loc in file_handle and not overwrite:
            raise IOError(f"Dataset {loc} already exists in file {filename}")
        elif loc in file_handle and overwrite:
            del file_handle[loc]
        file_handle.create_dataset(loc, data=data)


def parse_name_and_key(name_and_key):
    """Parse a string of the form name/key into name and key."""
    match = re.search(r"^(.+\.h5)(/.+)?$", name_and_key)
    if match is None:
        raise IOError(f"Can't parse {name_and_key}")
    input_file, input_key = match.groups()
    return input_file, input_key

def list_datasets(file_name, dimensions=None):
    """List datasets in an HDF5 file. Optionally filter by number of dimensions."""
    datasets = []
    def visitor_func(name, node):
        if isinstance(node, _h5py.Dataset):
            if dimensions is None or len(node.shape) == dimensions:
                datasets.append(node.name)
    with _h5py.File(file_name, "r") as file_handle:
        file_handle.visititems(visitor_func)
    return datasets

        
