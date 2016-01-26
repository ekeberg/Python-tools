#!/usr/bin/env python
"""
This program converts all .h5 files in the current
directory to .png using the HAWK program
image_to_png
"""
import os, re, sys, spimage
import scripts
from optparse import OptionParser
from optparse import OptionGroup

def to_png(*arguments):
    if len(arguments) <= 0:
        print """
    This program converts all h5 files in the curren directory to png.
    Usage:  python_script_new_to_png [colorscale]

    Colorscales:
    Jet
    Gray
    PosNeg
    InvertedPosNeg
    Phase
    InvertedPhase
    Log (can be combined with the others)
    Shift (can be combined with the others)
    Support

    """
        return
    elif not (isinstance(arguments,list) or isinstance(arguments,tuple)):
        print "function to_png takes must have a list or string input"
        return


    #l = os.popen('ls').readlines()
    l = os.listdir('.')

    expr = re.compile('.h5$')
    h5_files = filter(expr.search,l)

    expr = re.compile('.png$')
    png_files = filter(expr.search,l)

    files = [f for f in h5_files if f[:-2]+"png" not in png_files]
    files.sort()

    print "Converting %d files" % len(files)

    log_flag = 0
    shift_flag = 0
    support_flag = 0
    color = 16

    for flag in arguments:
        if flag == 'PosNeg':
            color = 8192
        elif flag == 'InvertedPosNeg':
            color = 16384
        elif flag == 'Phase':
            color = 256
        elif flag == 'InvertedPhase':
            color = 4096
        elif flag == 'Jet':
            color = 16
        elif flag == 'Gray':
            color = 1
        elif flag == 'Log':
            log_flag = 1
        elif flag == 'Shift':
            shift_flag = 1
        elif flag == 'Support':
            support_flag = 1
        else:
            print "unknown flag %s" % flag

    if log_flag == 1:
        color += 128

    # for f in files:
    #     img = spimage.sp_image_read(f[:-1],0)

    def shift_function(img):
        return img

    if shift_flag:
        def shift_function(img):
            ret = spimage.sp_image_shift(img)
            spimage.sp_image_free(img)
            return ret

    if support_flag:
        for f in files:
            img = spimage.sp_image_read(f,0)
            spimage.sp_image_mask_to_image(img,img)
            img = shift_function(img)
            spimage.sp_image_write(img,f[:-2]+"png",color)
            spimage.sp_image_free(img)
    else:
        for f in files:
            img = spimage.sp_image_read(f,0)
            img = shift_function(img)
            spimage.sp_image_write(img,f[:-2]+"png",color)
            spimage.sp_image_free(img)

def read_files(in_dir, out_dir):
    in_files = os.listdir(in_dir)
    h5_files = [f for f in in_files if  re.search('.h5$',f)]
    out_files = os.listdir(in_dir)
    png_files = [f for f in out_files if  re.search('.png$',f)]
    files = [in_dir+'/'+f for f in h5_files if f[:-2]+"png" not in png_files]
    files.sort()
    return files

if __name__ == "__main__":
    parser = OptionParser(usage="%prog [options]")
    parser.add_option("-c", "--color", action="store", type="string", dest="colorscale", default="jet",
                      help="Colorscale")
    parser.add_option("-l", "--log", action="store_true", dest="log", default=False,
                      help="Log scale")
    parser.add_option("-s", "--shift", action="store_true", dest="shift", default=False,
                      help="Shift image")
    parser.add_option("-m", "--mask", action="store_true", dest="mask", default=False,
                      help="Plot mask")
    parser.add_option("-i", "--input", action="store", type="string", dest="input", default=".",
                      help="Input directory")
    parser.add_option("-o", "--output", action="store", type="string", dest="output", default=".",
                      help="Output directory")
    (options,args) = parser.parse_args()

    plot_setup = scripts.to_png.PlotSetup()
    plot_setup.set_color(options.colorscale)
    print options.log
    plot_setup.set_log(options.log)
    plot_setup.set_shift(options.shift)
    plot_setup.set_mask(options.mask)

    files = read_files(options.input, options.output)

    scripts.to_png.to_png_parallel(files, options.output, plot_setup)

    #scripts.to_png.to_png(*sys.argv[1:])

