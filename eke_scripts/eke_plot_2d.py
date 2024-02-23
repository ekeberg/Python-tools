import h5py
import numpy
import argparse
import re
import matplotlib.pyplot
import sys
from eke import matplotlib_tools
from eke import hdf5_tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Plot in log scale.")
    parser.add_argument("-a", "--abs", action="store_true",
                        help="Plot abs of complex numbers.")
    parser.add_argument("-p", "--phase", action="store_true",
                        help="Plot phases of complex numbers.")
    parser.add_argument("-r", "--real", action="store_true",
                        help="Plot real values of complex numbers.")
    parser.add_argument("-i", "--imag", action="store_true",
                        help="Plot imaginary values of complex numbers.")
    parser.add_argument("-s", "--shift", action="store_true",
                        help="FFT-shift image.")
    parser.add_argument("--min", type=float,
                        help="Lower limit of plot values")
    parser.add_argument("--max", type=float,
                        help="Upper limit of plot values")
    parser.add_argument("--slice", type=str, default=None)

    args = parser.parse_args()

    input_file, input_key = hdf5_tools.parse_name_and_key(args.filename)
    if input_key is None:
        print("You must specify a dataset to plot. Available 2D datasets are:")
        print("\n".join(hdf5_tools.list_datasets(input_file, dimensions=2)))
        sys.exit(1)

    with h5py.File(input_file, "r") as file_handle:
        # data = file_handle[input_key][...]
        data_set = file_handle[input_key]
        if args.slice:
            data = eval(f"data_set[{args.slice}]")
        else:
            data = data_set[...]

    show_colorbar = True

    if args.phase:
        data = numpy.angle(data)
    elif args.real:
        data = numpy.real(data)
    elif args.imag:
        data = numpy.imag(data)
    elif args.abs:
        data = abs(data)
    elif numpy.iscomplexobj(data):
        data = matplotlib_tools.complex_to_rgb(data)
        show_colorbar = False

    if args.shift:
        data = numpy.fft.fftshift(data)

    fig = matplotlib.pyplot.figure(input_file)
    ax = fig.add_subplot(111)
    if args.log:
        im = ax.imshow(data, vmin=args.min, vmax=args.max,
                    norm=matplotlib.colors.LogNorm())
    else:
        im = ax.imshow(data, vmin=args.min, vmax=args.max)

    if show_colorbar:
        fig.colorbar(im)


    matplotlib.pyplot.show()


if __name__ == "__main__":
    main()
