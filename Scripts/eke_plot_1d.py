#!/bin/env python
import sys
import pylab
from optparse import OptionParser

def plot_1d(arguments):
    if isinstance(arguments,str):
        arguments = [arguments]
    if len(arguments) < 1:
        print "Need at least one data set"
        sys.exit(1)

    fig = pylab.figure(1)
    ax = fig.add_subplot(111)

    for f in arguments:
        try:
            data = pylab.loadtxt(f)
        except:
            print "Error %s is not a readable file.\n" % (f)
            sys.exit(1)

        #data = pylab.transpose(data)

        ax.plot(data,label=f)

    if len(arguments) > 1:
        ax.legend()

    return data

if __name__ == "__main__":
    parser = OptionParser(usage="%prog <data>")
    (options, args) = parser.parse_args()
    plot_1d(args[0])
    pylab.show()
