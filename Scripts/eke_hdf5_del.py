import numpy
import h5py
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str)

args = parser.parse_args()

#print(args.input_file)
input_file, input_key = re.search("^(.+\.h5)(/.+)?$", args.file).groups()

if input_key is None:
    raise ValueError(f"Must provide a location in the hdf5 file: file.h5/location")

with h5py.File(input_file, "r+") as file_handle:
    if input_key not in file_handle.keys():
        raise ValueError(f"Dataset or group {input_key} doew not exist in {input_file}")
    del file_handle[input_key]
