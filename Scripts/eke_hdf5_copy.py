import numpy
import h5py
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=str)
parser.add_argument("output_file", type=str)
parser.add_argument("--slice", type=str, default=None)
parser.add_argument("--overwrite", type=bool, default=False)

args = parser.parse_args()

#print(args.input_file)
input_file, input_key = re.search("^(.+\.h5)(/.+)?$", args.input_file).groups()
output_file, output_key = re.search("^(.+\.h5)(/.+)?$", args.output_file).groups()

if input_key is None:
    raise ValueError(f"Must provide a location in the hdf5 file: file.h5/location")

if output_key is None:
    output_key = input_key
    
# Read data
with h5py.File(input_file, "r") as file_handle:
    data_set = file_handle[input_key]
    if args.slice:
        input_data = eval(f"data_set[{args.slice}]")
    else:
        input_data = data_set[...]

        
with h5py.File(output_file, "a") as file_handle:
    if output_key in file_handle.keys():
        if args.overwrite:
            del file_handle[output_key]
        else:
            raise ValueError(f"Dataset or group {output_key} already exist in {output_file}")
    else:
        file_handle.create_dataset(output_key, data=input_data)
