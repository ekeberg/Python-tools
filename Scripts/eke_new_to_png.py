#!/usr/bin/env python
"""
This program converts all .h5 files in the current
directory to .png using the HAWK program
image_to_png
"""
from __future__ import print_function
import os
import re
import sys
import spimage
from eke import scripts
import argparse

def to_png(*arguments):
    if len(arguments) <= 0:
        print("""
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

    """)
        return
    elif not (isinstance(arguments,list) or isinstance(arguments,tuple)):
        print("function to_png takes must have a list or string input")
        return


    #l = os.popen('ls').readlines()
    l = os.listdir('.')

    expr = re.compile('.h5$')
    h5_files = list(filter(expr.search,l))

    expr = re.compile('.png$')
    png_files = list(filter(expr.search,l))

    files = [f for f in h5_files if f[:-2]+"png" not in png_files]
    files.sort()

    print("Converting %d files" % len(files))

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
            print("unknown flag %s" % flag)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("indir", default=".", help="Input directory")
    parser.add_argument("outdir", default=".", help="Output directory")    
    parser.add_argument("-c", "--colorscale", default="jet", help="Colorscale")
    parser.add_argument("-l", "--log", action="store_true", help="Log scale")
    parser.add_argument("-s", "--shift", action="store_true", help="Shift image")
    parser.add_argument("-m", "--mask", action="store_true", help="Plot mask")
    args = parser.parse_args()

    plot_setup = scripts.to_png.PlotSetup()
    plot_setup.set_color(args.colorscale)
    print(args.log)
    plot_setup.set_log(args.log)
    plot_setup.set_shift(args.shift)
    plot_setup.set_mask(args.mask)

    files = read_files(args.indir, args.outdir)

    scripts.to_png.to_png_parallel(files, args.output, plot_setup)

