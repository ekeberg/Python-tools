
import sys, h5py, pylab
from optparse import OptionParser

def view_pnccd(filename):
    f = h5py.File(filename)

    data1 = f.values()[1].values()[3].value
    data2 = f.values()[1].values()[4].value

    fig = pylab.figure(1)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax2.imshow(data1)
    ax1.imshow(data2)
    pylab.show()

if __name__ == '__main__':
    parser = OptionParser(usage="%prog PNCCD_FILE")
    (options, args) = parser.parse_args()
    
    if len(args) == 0:
        print "You must provide a pnCCD file"
        exit(0)
    
    view_pnccd(args[0])
