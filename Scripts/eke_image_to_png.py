#!/usr/bin/env python
import spimage, pylab, sys
from optparse import OptionParser

def image_to_png(in_filename, out_filename, colorscale, shift):
    try:
        img = spimage.sp_image_read(in_filename,0)
    except:
        raise TypeError("%s is not a readable file" % in_filename)

    if shift == 1:
        img = spimage.sp_image_shift(img)

    try:
        spimage.sp_image_write(img,out_filename,colorscale)
    except:
        raise TypeError("Can not write %s" % out_filename)

if __name__ == "__main__":
    parser = OptionParser(usage="%prog [-c colorscale] -i <image_in.h5> -o <image_out.png>")
    parser.add_option("-i", action="store", type="string", dest="infile", help="HDF5 file to convert to png.")
    parser.add_option("-o", action="store", type="string", dest="outfile", help="Output file.")
    parser.add_option("-l", action="store_true", dest="log", help="Use log scale.")
    parser.add_option("-s", action="store_true", dest="shift", help="Shift image.")
    parser.add_option("-m", action="store_true", dest="mask", help="Output mask.")
    parser.add_option("-c", action="store", type="choice", dest="colorscale", help="Colorscale of output image.",
                      choices=("gray", "jet", "phase", "hot", "rainbow", "traditional", "weighted_phase"), default="jet")
    (options, args) = parser.parse_args()
    
    colorscale_dict = {"gray" : spimage.SpColormapGrayScale,
                       "jet" : spimage.SpColormapJet,
                       "phase" : spimage.SpColormapPhase,
                       "hot": spimage.SpColormapHot,
                       "rainbow" : spimage.SpColormapRainbow,
                       "traditional" : spimage.SpColormapTraditional,
                       "weighted_phase" : spimage.SpColormapWeightedPhase}

    output_flag = colorscale_dict[options.colorscale]
    if options.log: output_flag |= spimage.SpColormapLogScale
    if options.mask: output_flag |= spimage.SpColormapMask

    image_to_png(options.infile, options.outfile, output_flag, options.shift)
