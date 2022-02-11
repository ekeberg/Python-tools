import h5py as _h5py
import numpy as _numpy


def dictionary_to_hdf5(file_name, dict_in):
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
    with _h5py.File(filename, "w") as file_handle:
        file_handle.create_dataset("data", data=_numpy.array(array))


def read_numpy(filename):
    with _h5py.File(filename, "r") as file_handle:
        array = file_handle["data"][...]
    return array


def read_dataset(filename, loc):
    with _h5py.File(filename, "r") as file_handle:
        array = file_handle[loc][...]
    return array


def write_datset(filename, loc, data):
    with _h5py.File(filename, "a") as file_handle:
        file_handle.create_dataset(loc, data=data)
