#! /usr/bin/python
import sys, h5py, pylab

def view_pnccd(filename):
    try:
        f = h5py.File(filename)
    except:
        print "Error reading file %s. It may not be a pnCCD file." % filename
        exit(1)

    data1 = f.values()[1].values()[3].value
    data2 = f.values()[1].values()[4].value

    fig = pylab.figure(1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax2.imshow(data1)
    ax1.imshow(data2)
    pylab.show()

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print """
Usage: view_pnccd <filename.h5>

This program is used to view the output from cass.

"""
        exit(0)
    
    view_pnccd(sys.argv[1])
