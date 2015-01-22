#!/bin/env python
import sys
import pylab
from optparse import OptionParser

def plot_simple(in_file):
    try:
        data = pylab.loadtxt(sys.argv[1])
    except:
        raise IOError("Can't read %s." % in_file)

    for i in pylab.transpose(data):
        pylab.plot(i)


if __name__ == "__main__":
    parser = OptionParser(usage="%prog <data>")
    (options, args) = parser.parse_args()
    if len(args) == 0: raise IOError("No datafile provided")

    plot_simple(args[0])
    pylab.show()

