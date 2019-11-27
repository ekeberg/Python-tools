#!/usr/bin/env python
from eke import sphelper
import numpy
import matplotlib
import matplotlib.pyplot
import sys
import re
import argparse

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
        image = numpy.fft.fftshift(image)
        mask = numpy.fft.fftshift(mask)

    def no_log(x):
        return x

    if plot_log:
        log_function = numpy.log
    else:
        log_function = no_log

    if plot_mask:
        plot_input = mask
    else:
        plot_input = image
        
    function_dict = {"abs" : abs, "phase" : numpy.angle, "real" : numpy.real, "imag" : numpy.imag}
    
    matplotlib.pyplot.imshow(log_function(function_dict[function](plot_input)), cmap=colormap, origin="lower", interpolation="nearest")
    matplotlib.pyplot.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-m", "--mask", action="store_true", help="Plot the mask.")
    parser.add_argument("-f", "--function", choices=("abs", "phase", "real", "imag"), default="abs",
                        help="What to plot in the image.")
    parser.add_argument("-l", "--log", action="store_true", help="Plot in log scale.")
    parser.add_argument("-s", "--shift", action="store_true", help="Shift image.")
    args = parser.parse_args()
    
    plot_image(args.file, args.function, args.mask, args.log, args.shift)
    
