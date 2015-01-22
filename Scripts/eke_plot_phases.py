#!/bin/env python
import sys, pylab, spimage
from optparse import OptionParser

HISTOGRAM, PHASES = range(2)

def plot_phases(in_file, plot_type, plot_log):
    flags = ['histogram','phases']
    plot_flag = 0
    log_flag = 0

    def no_log(x):
        return x

    fig = pylab.figure(1)
    ax = fig.add_subplot(111)

    try:
        img = spimage.sp_image_read(in_file,0)
    except:
        raise IOError("Can't read %s." % in_file)

    values = img.image.reshape(pylab.size(img.image))

    if plot_log:
        log_function = pylab.log
    else:
        log_function = no_log

    if plot_type == PHASES:
        hist = pylab.histogram(pylab.angle(values),bins=500)
        ax.plot((hist[1][:-1]+hist[1][1:])/2.0,log_function(hist[0]))
    elif plot_flag == HISTOGRAM:
        hist = pylab.histogram2d(pylab.real(values),pylab.imag(values),bins=500)
        ax.imshow(log_function(hist[0]),extent=(hist[2][0],hist[2][-1],-hist[1][-1],-hist[1][0]),interpolation='nearest')
    else:
        ax.plot(pylab.real(values),pylab.imag(values),'.')
    return fig
    


if __name__ == "__main__":
    """This script probably doesn't really work."""
    parser = OptionParser(usage="%prog <image.h5>")
    parser.add_option("-m", "--mode", action="store", dest="mode", type="choice",
                      help="plot phase histogram or phaseplot", choices=("histogram", "phases"),
                      default="phases")
    parser.add_option("-l", "--log", action="store_true", dest="log", help="Plot in log scale")
    (options, args) = parser.parse_args()
    
    if len(args) == 0: raise IOError("No input image.")

    plot_type_dict = {"histogram" : HISTOGRAM, "phases" : PHASES}

    plot_phases(args[0], plot_type_dict[options.mode], options.log)
    pylab.show()

