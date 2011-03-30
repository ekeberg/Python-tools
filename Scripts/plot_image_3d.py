#! /usr/bin/python

import matplotlib
matplotlib.use('WxAgg')
matplotlib.interactive(True)
from pylab import *
import spimage
import sys
import numpy
import time
from enthought.mayavi import mlab

img = None
shift = False
log = False
plot_mask = False
try:
    img = spimage.sp_image_read(sys.argv[1],0)
except:
    print "Must provide h5 image to read"

#img.image[img.image < 0.0] = 0.0

options = ['shift','log','mask']

for o in sys.argv:
    if o in options:
        if o.lower() == "shift":
            shift = True
        if o.lower() == "log":
            log = True
        if o.lower() == "mask":
            plot_mask = True

if shift:
    img_s = spimage.sp_image_shift(img)
    spimage.sp_image_free(img)
    img = img_s

if plot_mask:
    plot_array = img.mask
else:
    plot_array = abs(img.image)

if log:
    s = mlab.pipeline.scalar_field(log10(0.001*max(plot_array.flatten())+plot_array))
else:
    s = mlab.pipeline.scalar_field(plot_array)

#mlab.figure(size=(1000,900))

mlab.pipeline.image_plane_widget(s,plane_orientation='x_axes',
                                 slice_index=shape(img.image)[0]/2)
mlab.pipeline.image_plane_widget(s,plane_orientation='y_axes',
                                 slice_index=shape(img.image)[1]/2)

mlab.outline()
mlab.show()
