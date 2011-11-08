
import matplotlib
#matplotlib.use('WxAgg')
matplotlib.interactive(True)
from pylab import *
import spimage
import sys
import numpy
import time
from enthought.mayavi import mlab
from optparse import OptionParser

def plot_image_3d(image_file, plot_shifted, plot_log, plot_mask):
    img = None
    shift = False
    log = False
    plot_mask = False
    try:
        img = spimage.sp_image_read(sys.argv[1],0)
    except:
        raise InputError("%s is not a readable h5 image." % image_file)

    if plot_shifted:
        img_s = spimage.sp_image_shift(img)
        spimage.sp_image_free(img)
        img = img_s

    if plot_mask:
        plot_array = img.mask
    else:
        plot_array = abs(img.image)

    if plot_log:
        s = mlab.pipeline.scalar_field(log10(0.001*max(plot_array.flatten())+plot_array))
    else:
        s = mlab.pipeline.scalar_field(plot_array)

    mlab.pipeline.image_plane_widget(s,plane_orientation='x_axes',
                                     slice_index=shape(img.image)[0]/2)
    mlab.pipeline.image_plane_widget(s,plane_orientation='y_axes',
                                     slice_index=shape(img.image)[1]/2)

    #mlab.outline()
    mlab.show()

if __name__ == "__main__":
    parser = OptionParser(usage="%prog <3d_file.h5>")
    parser.add_option("-s", action="store_true", dest="shift", help="Shift image.")
    parser.add_option("-l", action="store_true", dest="log", help="Plot in log scale.")
    parser.add_option("-m", action="store_true", dest="mask", help="Plot mask.")
    (options, args) = parser.parse_args()
    
    if len(args) == 0: raise InputError("No input image")

    plot_image_3d(args[0], options.shift, options.log, options.mask)
