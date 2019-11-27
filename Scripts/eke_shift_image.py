#!/usr/bin/env python
import spimage
import sys
import argparse

def shift_image(in_file,out_file = 0):
    try:
        img = spimage.sp_image_read(in_file,0)
    except:
        raise IOError("Can't read %s" % in_file)

    img_s = spimage.sp_image_shift(img)

    if out_file == 0:
        out_file = in_file

    try:
        spimage.sp_image_write(img_s,out_file,0)
    except:
        print("Error: Can not write to %s" % out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outfile")    
    args = parser.parse_args()

    shift_image(args.infile, args.outfile)
