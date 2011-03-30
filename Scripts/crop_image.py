#! /usr/bin/python

import spimage, pylab, sys

def crop_image(in_file, out_file, sideX, sideY = 0):
    if sideY == 0:
        sideY = sideX
    try:
        img = spimage.sp_image_read(in_file,0)
    except:
        print "Error: %s is not a readable .h5 file\n" % in_file
        exit(1)

    shifted = 0
    if img.shifted:
        shifted = 1
        img = spimage.sp_image_shift(img)

    print "shifted = ", shifted

    lowX = img.detector.image_center[0]-(sideX/2.0-0.5)
    highX = img.detector.image_center[0]+(sideX/2.0-0.5)
    lowY = img.detector.image_center[1]-(sideY/2.0-0.5)
    highY = img.detector.image_center[1]+(sideY/2.0-0.5)

    print lowX, " ", highX
    print lowY, " ", highY

    if lowX != pylab.floor(lowX):
        lowX = int(pylab.floor(lowX))
        highX = int(pylab.floor(highX))
        img.detector.image_center[0] -= 0.5
    else:
        lowX = int(lowX)
        highX = int(highX)
    if lowY != pylab.floor(lowY):
        lowY = int(pylab.floor(lowY))
        highY = int(pylab.floor(highY))
        img.detector.image_center[1] -= 0.5
    else:
        lowY = int(lowY)
        highY = int(highY)

    cropped = spimage.rectangle_crop(img,lowX,lowY,highX,highY)

    print "did crop"

    if shifted:
        cropped = spimage.sp_image_shift(cropped)

    print "shifted (or not)"

    print "write ", out_file

    #print "orientation = ", cropped.detector.orientation
    #print spimage.sp_3matrix_get(cropped.detector.orientation,0,0,0)

    try:
        spimage.sp_image_write(cropped,out_file,16)
    except:
        print "Error: can not write to %s\n" % out_file

    print "end"

if __name__ == "__main__":
    try:
        try:
            yside = sys.argv[4]
        except:
            yside = 0
        crop_image(str(sys.argv[1]),str(sys.argv[2]),int(sys.argv[3]),yside)
    except:
        print """
Usage:  python_script_crop_image <in.h5> <out.h5> xside [yside]

Crops the image to the specified size symmetrically
around the image center. If only one side is given
xside is used for both sides.
"""
