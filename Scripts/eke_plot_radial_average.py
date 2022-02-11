import h5py
import argparse
import re
import matplotlib.pyplot
from eke import tools


def split_file_and_key(arg):
    search_string = r"^(.+\.h5)(/.+)?$"
    search = re.search(search_string, args.filename)
    input_file, input_key = search.groups()
    return input_file, input_key


parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--mask")

args = parser.parse_args()

data_file, data_key = split_file_and_key(args.filename)
with h5py.File(data_file, "r") as file_handle:
    data = file_handle[data_key][...]

if args.mask is not None:
    mask_file, mask_key = split_file_and_key(args.filename)
    with h5py.File(data_file, "r") as file_handle:
        data = file_handle[data_key][...]
else:
    mask = None

data_radial_average = tools.radial_average(data, mask=mask)

fig = matplotlib.pyplot.figure("Radial average")
ax = fig.add_subplot(111)
ax.plot(data_radial_average)
matplotlib.pyplot.show()
