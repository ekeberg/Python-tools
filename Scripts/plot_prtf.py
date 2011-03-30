#! /usr/bin/python

import sys,math
sys.path.append("/home/ekeberg/Scripts")
from plot_2d import *
from optparse import OptionParser
from optparse import OptionGroup

class Setup:

    def __init__(self,w,d,p,n):
        self.wavelength = w
        self.distance = d
        self.pixel_size = p
        self.number_of_images = n
        if self.wavelength > 0.0 and\
                self.distance > 0.0 and\
                self.pixel_size > 0.0:
            self.known = True
        else:
            self.known = False


    def convert(self,pixels):
        if self.known:
            return pixels*self.pixel_size/self.distance/self.wavelength
        else:
            return pixels
            
    def rescale(self,value):
        if self.number_of_images:
            return (value-1.0/math.sqrt(float(self.number_of_images)))/(1.0-1.0/math.sqrt(float(self.number_of_images)))
        else:
            return value
    

def plot_prtf(filename,setup = 0):
    try:
        data = pylab.transpose(pylab.loadtxt(filename))
    except:
        print "Can't open file %s" % filename
        exit(1)

    fig = pylab.figure()
    ax = fig.add_subplot(111)
        
    ax.plot([setup.convert(i) for i in data[0,:]],
            [setup.rescale(i) for i in data[1,:]])
    ax.plot([setup.convert(data[0,0]),setup.convert(data[0,-1])],
               [pylab.exp(-1),pylab.exp(-1)])

    for i in range(len(data[1,:])):
        if setup.rescale(data[1,i]) < pylab.exp(-1):
            break
    ax.plot([setup.convert(data[0,i]),setup.convert(data[0,i])],[0,pylab.exp(-1)])
    ax.set_ylim([0.0,1.0])

    if setup.known:
        #pylab.text(setup.convert(data[0,-1]),0.95,
        ax.text(0.95*ax.get_xlim()[1],0.95*ax.get_ylim()[1],
                   r'$R_{\mathrm{crystall}} = %g nm \quad R_{\mathrm{optical}} = %g nm$' % (1.0/setup.convert(data[0,i]),0.5/setup.convert(data[0,i])),
                   va='top',ha='right',fontsize = 15)


if __name__ == "__main__":
    parser = OptionParser(usage="%prog filename [options]")
    res = OptionGroup(parser,"Resolution options",
                      "These options are all needed to calculate the resolution. By themselves they are ineffective.")
    res.add_option("-w", action="store", type="float", dest="wavelength",
                      help="Wavlength used in the experiment [nm]")
    res.add_option("-d", action="store", type="float", dest="distance",
                      help="Distance from interaction region to detector. Unit must be the same as for the pixel size.")
    res.add_option("-p", action="store", type="float", dest="pixel_size",
                      help="Pixel size. Unit must be the same as for the detector distance.")
    parser.add_option_group(res)
    parser.add_option("-n", action="store", type="int", dest="number",
                      help="Number of images included in the PRTF. If this option is specified the curve is scaled down to compensate for low counts.")
    (options,args) = parser.parse_args()
    
    if len(args) < 1:
        parser.error("A filename must be specified")

    setup = 0
    try:
        setup = Setup(options.wavelength, options.distance,
                      options.pixel_size, options.number)
    except:
        print "Error in arguments"
        exit(1)
    plot_prtf(args[0],setup)
    pylab.show()

