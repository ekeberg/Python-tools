#!/usr/bin/env python
import numpy
import matplotlib
import matplotlib.pyplot
import argparse


class Setup:
    def __init__(self, w, d, p, n):
        self.wavelength = w
        self.distance = d
        self.pixel_size = p
        self.number_of_images = n
        if self.wavelength > 0.0 and\
                self.distance > 0.0 and\
                self.pixel_size > 0.0:
            self.known = True
        else:
            self.known = False

    def convert(self, pixels):
        if self.known:
            return pixels * self.pixel_size / self.distance / self.wavelength
        else:
            return pixels

    def rescale(self, value):
        if self.number_of_images:
            return ((value - 1 / numpy.sqrt(self.number_of_images))
                    / (1 - 1 / numpy.sqrt(self.number_of_images)))
        else:
            return value


def plot_prtf(filename, setup=0):
    data = numpy.transpose(numpy.loadtxt(filename))

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111)

    ax.plot([setup.convert(i) for i in data[0, :]],
            [setup.rescale(i) for i in data[1, :]])
    ax.plot([setup.convert(data[0, 0]), setup.convert(data[0, -1])],
            [numpy.exp(-1), numpy.exp(-1)])

    for i in range(len(data[1, :])):
        if setup.rescale(data[1, i]) < numpy.exp(-1):
            break
    ax.plot([setup.convert(data[0, i]), setup.convert(data[0, i])],
            [0, numpy.exp(-1)])
    ax.set_ylim([0, 1])

    if setup.known:
        ax.text(0.95*ax.get_xlim()[1], 0.95*ax.get_ylim()[1],
                rf"$R_{{\mathrm{{crystall}}}} = {1./setup.convert(data[0,i])} "
                r"nm \quad R_{{\mathrm{{optical}}}} = "
                rf"{.5/setup.convert(data[0,i])} nm$",
                va='top', ha='right', fontsize=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    res = parser.add_argument_group("Resolution")
    res.add_argument("-w", "--wavelength", type=float,
                     help="Wavlength used in the experiment [nm]")
    res.add_argument("-d", "--distance", type=float,
                     help="Distance from interaction region to detector. "
                     "Unit must be the same as for the pixel size.")
    res.add_argument("-p", "--pixel_size", type=float,
                     help="Pixel size. Unit must be the same as for the "
                     "detector distance.")
    parser.add_argument("-n", "--number_of_reconstructions", type=int,
                        help="Number of images included in the PRTF. If this "
                        "option is specified the curve is scaled down to "
                        "compensate for low counts.")
    args = parser.parse_args()

    setup = 0

    setup = Setup(args.wavelength, args.distance,
                  args.pixel_size, args.number_of_reconstructions)

    plot_prtf(args.file, setup)
    matplotlib.pyplot.show()
