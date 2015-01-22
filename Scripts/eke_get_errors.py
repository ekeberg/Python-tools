#!/bin/env python
from pylab import *
import os
import re
import sys
from optparse import OptionParser

def get_errors(input_dir=".",error_type="fourier"):
    column_numbers = {"fourier" : 3, "real" : 2}
    if not error_type in column_numbers.keys():
        print "Error: invalid error_type"
        exit(1)
    ls_out = os.listdir(input_dir)
    dirs = [d for d in ls_out if re.search('^[0-9]{6}$', d)]

    err = zeros(len(dirs))

    for i in range(len(dirs)):
        l = loadtxt("%s/uwrapc.log" % (dirs[i]), skiprows=47)
        err [i] = l[-1][column_numbers[error_type]]

    err.sort()
    for i in err:
        print i

if __name__ == "__main__":
    parser = OptionParser(usage="%prog [-p DIRECTORY]")
    parser.add_option("-i", action="store", type="string", dest="input_dir", default=".",
                      help="The directory in which to search for reconstructions. Default is .")
    parser.add_option("-e", action="store", type="string", dest="error_type", default="fourier",
                      help="Type of error to use: fourier or real")
    (options,argv) = parser.parse_args()
    if not os.path.isdir(options.input_dir):
        print "Error: %s is not a directory" % options.input_dir
        exit(1)
    get_errors(options.input_dir,options.error_type)
