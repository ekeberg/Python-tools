import numpy
import h5py
from eke import vtk_tools
import argparse
import re
import sys


def print_3d_datasets(file_name):
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            if len(node.shape) == 3:
                print(f"   {node.name}")
    with h5py.File(file_name, "r") as file_handle:
        file_handle.visititems(visitor_func)


parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("-s", "--surface", action="store_true",
                    help="Plot surface")
parser.add_argument("-l", "--log", action="store_true",
                    help="Plot in log scale.")
parser.add_argument("-a", "--abs", action="store_true",
                    help="Plot in absolue value.")
args = parser.parse_args()

input_file, input_key = re.search(r"^(.+\.h5)(/.+)?$", args.filename).groups()
if input_key is None:
    print("You must specify a dataset to plot. Available 3D datasets are:")
    print_3d_datasets(input_file)
    sys.exit(1)

with h5py.File(input_file, "r") as file_handle:
    data = numpy.float32(file_handle[input_key][...])

if args.abs:
    data = abs(data)

if args.surface:
    vtk_tools.plot_isosurface_interactive(data)
else:
    vtk_tools.plot_planes(data, log=args.log)
