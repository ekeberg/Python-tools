#!/usr/bin/env python
import h5py
import matplotlib
import matplotlib.pyplot
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()

    view_pnccd(args.file)


def view_pnccd(filename):
    with h5py.File(filename, "r") as file_handle:
        data1 = file_handle.values()[1].values()[3].value
        data2 = file_handle.values()[1].values()[4].value

    fig = matplotlib.pyplot.figure(1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax2.imshow(data1)
    ax1.imshow(data2)
    matplotlib.pyplot.show()


if __name__ == '__main__':
    main()
