#! /usr/bin/python

import spimage, sys

def shift_image(in_file,out_file = 0):
    try:
        img = spimage.sp_image_read(in_file,0)
    except:
        print "Error: Can not read image %s" % in_file
        return

    img_s = spimage.sp_image_shift(img)

    if out_file == 0:
        out_file = in_file

    try:
        spimage.sp_image_write(img_s,out_file,0)
    except:
        print "Error: Can not write to %s" % out_file

if __name__ == "__main__":
    try:
        if len(sys.argv) > 2:
            shift_image(sys.argv[1],sys.argv[2])
        else:
            shift_image(sys.argv[1])
    except:
        print """
Usage:  python_script_shift_image <in.h5> [out.h5]

If no output image is given the origninal image will
be overwritten.
"""
