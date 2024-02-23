import argparse
import sys
import pathlib
import numpy
import h5py
from eke import vtk_tools
from eke import hdf5_tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-s", "--surface", action="store_true",
                        help="Plot surface")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Plot in log scale.")
    parser.add_argument("-a", "--abs", action="store_true",
                        help="Plot in absolue value.")
    args = parser.parse_args()

    input_file, input_key = hdf5_tools.parse_name_and_key(args.filename)

    if not pathlib.Path(input_file).exists():
        print(f"File {input_file} does not exist.")
        sys.exit(1)

    if input_key is None:
        print("You must specify a dataset to plot. Available 3D datasets are:")
        print("\n".join(hdf5_tools.list_datasets(args.filename, dimensions=3)))
        sys.exit(1)

    if input_key not in hdf5_tools.list_datasets(input_file, dimensions=3):
        print(f"Dataset {input_key} not found in file {input_file}")
        sys.exit(1)

    with h5py.File(input_file, "r") as file_handle:
        data = numpy.float32(file_handle[input_key][...])

    if args.abs:
        data = abs(data)

    if args.surface:
        vtk_tools.plot_isosurface_interactive(data)
    else:
        vtk_tools.plot_planes(data, log=args.log)


if __name__ == "__main__":
    main()
