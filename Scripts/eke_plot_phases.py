#!/usr/bin/env python
import pylab
import spimage
import argparse


HISTOGRAM, PHASES = list(range(2))


def plot_phases(in_file, plot_type, plot_log):
    plot_flag = 0

    def no_log(x):
        return x

    fig = pylab.figure(1)
    ax = fig.add_subplot(111)

    try:
        img = spimage.sp_image_read(in_file, 0)
    except IOError:
        raise IOError("Can't read %s." % in_file)

    values = img.image.reshape(pylab.size(img.image))

    if plot_log:
        log_function = pylab.log
    else:
        log_function = no_log

    if plot_type == PHASES:
        hist = pylab.histogram(pylab.angle(values), bins=500)
        ax.plot((hist[1][:-1] + hist[1][1:]) / 2, log_function(hist[0]))
    elif plot_flag == HISTOGRAM:
        hist = pylab.histogram2d(pylab.real(values),
                                 pylab.imag(values),
                                 bins=500)
        ax.imshow(log_function(hist[0]),
                  extent=(hist[2][0], hist[2][-1], -hist[1][-1], -hist[1][0]),
                  interpolation='nearest')
    else:
        ax.plot(pylab.real(values), pylab.imag(values), '.')
    return fig


if __name__ == "__main__":
    """This script probably doesn't really work."""
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-m", "--mode", choices=("histogram", "phases"),
                        default="phases",
                        help="plot phase histogram or phaseplot")
    parser.add_argument("-l", "--log", action="store_true",
                        help="Plot in log scale")
    args = parser.parse_args()

    plot_type_dict = {"histogram": HISTOGRAM, "phases": PHASES}

    plot_phases(args.file, plot_type_dict[args.mode], args.log)
    pylab.show()
