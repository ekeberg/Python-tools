#!/usr/bin/env python
#import spimage, pylab, sys, re
from eke import sphelper
import pylab, sys, re
from optparse import OptionParser

def plot_image(in_file, function, plot_mask, plot_log, plot_shifted):
    
    # try:
    #     #img = spimage.sp_image_read(in_file,0)
    image, mask = sphelper.import_spimage(in_file, ['image', 'mask'])
    # except:
    #     raise TypeError("Error: %s is not a readable .h5 file\n" % in_file)

    plot_flags = ['abs','mask','phase','real','imag']
    shift_flags = ['shift']
    log_flags = ['log']

    plot_flag = 0
    shift_flag = 0
    log_flag = 0

    colormap = "jet"
    if function == "phase":
        colormap = "hsv"

    if plot_shifted:
        #img = spimage.sp_image_shift(img)
        image = pylab.fftshift(image)
        mask = pylab.fftshift(mask)

    def no_log(x):
        return x

    if plot_log:
        log_function = pylab.log
    else:
        log_function = no_log

    if plot_mask:
        plot_input = mask
    else:
        plot_input = image
        
    function_dict = {"abs" : abs, "phase" : pylab.angle, "real" : pylab.real, "imag" : pylab.imag}
    
    pylab.imshow(log_function(function_dict[function](plot_input)), cmap=colormap, origin="lower", interpolation="nearest")
    pylab.show()

    # if (plot_flag == "mask"):
    #     pylab.imshow(img.mask,origin='lower',interpolation="nearest")
    # elif(plot_flag == "phase"):
    #     pylab.imshow(pylab.angle(img.image),cmap='hsv',origin='lower',interpolation="nearest")
    # elif(plot_flag == "real"):
    #     pylab.imshow(log_function(pylab.real(img.image)),origin='lower',interpolation="nearest")
    # elif(plot_flag == "imag"):
    #     pylab.imshow(log_function(pylab.imag(img.image)),origin='lower',interpolation="nearest")
    # else:
    #     pylab.imshow(log_function(abs(img.image)),origin='lower',interpolation="nearest")

    # pylab.show()

if __name__ == "__main__":
    parser = OptionParser(usage="%prog image")
    parser.add_option("-m", action="store_true", dest="mask", help="Plot the mask.")
    parser.add_option("-f", action="store", type="choice", dest="function", help="What to plot in the image.",
                      choices=("abs", "phase", "real", "imag"), default="abs")
    parser.add_option("-l", action="store_true", dest="log", help="Plot in log scale.")
    parser.add_option("-s", action="store_true", dest="shift", help="Shift image.")
    (options, args) = parser.parse_args()
    
    plot_image(args[0], options.function, options.mask, options.log, options.shift)
    
#     try:
#         plot_image(str(sys.argv[1]),*(sys.argv[2:]))
#     except:
#         print """
# Usage:  python_script_plot_image <image.h5> [arguments]

# Arguments:
# abs    plot the absolute value of the image (default)
# mask   plot the image mask
# phase  plot the phase of the image
# log    plot the image in log scale 
# shift  shift the image before plotting

# Several or none of the above arguments might be combined

# """
