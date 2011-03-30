#! /usr/bin/python

import sys
import pylab

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
    plot_1d(sys.argv[1:])
    pylab.show()
