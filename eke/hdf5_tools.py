import h5py as _h5py
import numpy as _numpy

def dictionary_to_hdf5(file_name, dict_in):
    def write_group(group_handle, dict_in):
        for name, value in dict_in.iteritems():
            if isinstance(value, dict):
                print "create group: {0}".format(name)
                new_group = group_handle.create_group(name)
                write_group(new_group, value)
            else:
                print "create dataset {0}".format(name)
                group_handle.create_dataset(name, data=_numpy.array(value))
    with _h5py.File(file_name, "w") as file_handle:
        write_group(file_handle, dict_in)

def hdf5_to_dictionary(file_name):
    def read_group(group):
        return_dictionary = {}
        for name, value in group.iteritems():
            if isinstance(value, _h5py.Dataset):
                print "read dataset {0}".format(name)
                return_dictionary[name] = value[...]
            elif isinstance(value, _h5py.Group):
                print "read group {0}".format(name)
                return_dictionary[name] = read_group(value)
            else:
                raise IOError("Can't handle {0} in file {1}".format(str(type(value))), file_name)
        return return_dictionary
    with _h5py.File(file_name, "r") as file_handle:
        return_dictionary = read_group(file_handle)
    return return_dictionary
