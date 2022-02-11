#!/usr/bin/env python
import numpy
import os
import re
import argparse


def get_errors(input_dir=".", error_type="fourier"):
    column_numbers = {"fourier": 3, "real": 2}
    if error_type not in column_numbers.keys():
        print("Error: invalid error_type")
        exit(1)
    ls_out = os.listdir(input_dir)
    dirs = [d for d in ls_out if re.search('^[0-9]{6}$', d)]

    err = numpy.zeros(len(dirs))

    for i in range(len(dirs)):
        uwrapc_log = numpy.loadtxt("%s/uwrapc.log" % (dirs[i]), skiprows=47)
        err[i] = uwrapc_log[-1][column_numbers[error_type]]

    err.sort()
    for i in err:
        print(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", default=".",
                        help="The directory in which to search for "
                        "reconstructions. Default is .")
    parser.add_argument("-e", "--error_type", default="fourier",
                        help="Type of error to use: fourier or real")
    args = parser.parse_args()
    if not os.path.isdir(args.indir):
        print("Error: %s is not a directory" % args.indir)
        exit(1)
    get_errors(args.indir, args.error_type)
