#!/usr/bin/env python
from eke import sphelper
import numpy
import matplotlib
import matplotlib.pyplot
import argparse


def plot_image(in_file, function, plot_mask, plot_log, plot_shifted):
    image, mask = sphelper.import_spimage(in_file, ['image', 'mask'])

    colormap = "jet"
    if function == "phase":
        colormap = "hsv"

    if plot_shifted:
        image = numpy.fft.fftshift(image)
        mask = numpy.fft.fftshift(mask)

    def no_log(x):
        return x

    if plot_log:
        log_function = numpy.log
    else:
        log_function = no_log

    if plot_mask:
        plot_input = mask
    else:
        plot_input = image

    function_dict = {"abs": abs,
                     "phase": numpy.angle,
                     "real": numpy.real,
                     "imag": numpy.imag}

    matplotlib.pyplot.imshow(log_function(function_dict[function](plot_input)),
                             cmap=colormap,
                             origin="lower",
                             interpolation="nearest")
    matplotlib.pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-m", "--mask", action="store_true",
                        help="Plot the mask.")
    parser.add_argument("-f", "--function",
                        choices=("abs", "phase", "real", "imag"),
                        default="abs",
                        help="What to plot in the image.")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Plot in log scale.")
    parser.add_argument("-s", "--shift", action="store_true",
                        help="Shift image.")
    args = parser.parse_args()

    plot_image(args.file, args.function, args.mask, args.log, args.shift)
