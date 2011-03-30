#! /usr/bin/python

import sys
import pylab

def read_data(f):
    try:
        data = pylab.loadtxt(f)
    except:
        print "Error %s is not a readable file.\n" % (f)
        return 0

    data = pylab.transpose(data)

    if len(data) < 2:
        print "Data %s doesn't have at least two dimensions\n" % (f)
        return 0
    return data

def plot_2d(*arguments):
    if len(sys.argv) < 2:
        print "Need at least one data set"
        return

    fig = pylab.figure(1)
    ax = fig.add_subplot(111)

    for f in sys.argv[1:]:
        data = read_data(f)
        if data == None:
            return
        reference = data[0]
        plots = data[1:]

        for i in plots:
            ax.plot(reference,i,label=f)

    if len(sys.argv) > 2:
        ax.legend()

    return data

if __name__ == "__main__":
    plot_2d(*sys.argv[1:])
    pylab.show()
