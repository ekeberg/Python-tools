#! /usr/bin/python

import sys
import pylab

def plot_simple(in_file):
    try:
        data = pylab.loadtxt(sys.argv[1])
    except:
        print "Error %s is not a readable file.\n" % (sys.argv[1])
        return

    for i in pylab.transpose(data):
        pylab.plot(i)


if __name__ == "__main__":
    try:
        plot_simple(sys.argv[1])
        pylab.show()
    except:
        print "Need a data set"

