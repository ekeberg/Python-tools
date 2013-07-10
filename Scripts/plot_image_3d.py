
#import matplotlib
#matplotlib.use('WxAgg')
#matplotlib.interactive(True)
from pylab import *
#import spimage
import sphelper
import sys
from mayavi import mlab
from optparse import OptionParser

def read_image(image_file, mask):
    if mask:
        field = 'mask'
    else:
        field = 'image'
    try:
        img = abs(sphelper.import_spimage(image_file, [field]))
    except:
        raise InputError("%s is not a readable h5 image." % image_file)
    return img


def plot_image_3d(image, plot_shifted, plot_log, plot_surface):
    shift = False
    log = False
    plot_mask = False

    if plot_shifted:
        image = fftshift(image)
        
    if plot_log:
        s = mlab.pipeline.scalar_field(log10(0.001*max(image.flatten())+img))
    else:
        s = mlab.pipeline.scalar_field(image)

    if plot_surface:
        mlab.pipeline.iso_surface(s)
    else:
        mlab.pipeline.image_plane_widget(s,plane_orientation='x_axes',
                                         slice_index=image.shape[0]/2)
        mlab.pipeline.image_plane_widget(s,plane_orientation='y_axes',
                                         slice_index=image.shape[1]/2)

    mlab.show()

#def plot_image_3d_surface(image_file

if __name__ == "__main__":
    parser = OptionParser(usage="%prog <3d_file.h5>")
    parser.add_option("-s", action="store_true", dest="shift", help="Shift image.")
    parser.add_option("-l", action="store_true", dest="log", help="Plot in log scale.")
    parser.add_option("-m", action="store_true", dest="mask", help="Plot mask.")
    parser.add_option("-S", action="store_true", dest="surface", help="Plot surface")
    (options, args) = parser.parse_args()
    
    if len(args) == 0: raise InputError("No input image")

    image = read_image(args[0], options.mask)
    
    plot_image_3d(image, options.shift, options.log, options.surface)
