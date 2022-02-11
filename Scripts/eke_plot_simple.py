#!/usr/bin/env python
import numpy
import matplotlib
import matplotlib.pyplot
import argparse


def plot_simple(in_file):
    try:
        data = numpy.loadtxt(in_file)
    except IOError:
        raise IOError(f"Can't read {in_file}.")

    for i in numpy.transpose(data):
        matplotlib.pyplot.plot(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    plot_simple(args.file)
    matplotlib.pyplot.show()
