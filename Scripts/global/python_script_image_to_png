#! /usr/bin/python

import spimage, pylab, sys

def image_to_png(arguments):
    if isinstance(arguments,str):
        arguments = [arguments]
    elif not isinstance(arguments,list):
        print "function to_png takes must have a list or string input"
        return

    if len(sys.argv) <= 1:
        print """
    Usage:  python_script_image_to_png <image_in.h5> <image_out.h5> [colorscale]

    Colorscales:
    Jet
    Gray
    PosNeg
    InvertedPosNeg
    Phase
    InvertedPhase
    Log (can be combined with the others)
    Shift (can be combined with the others)

    """
        exit(1)

    try:
        img = spimage.sp_image_read(arguments[0],0)
    except:
        print "Error: %s is not a readable .h5 file\n" % arguments[0]
        exit(1)

    log_flag = 0
    shift_flag = 0

    for flag in arguments[2:]:
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
        else:
            print "unknown flag %s" % flag

    if log_flag == 1:
        color += 128

    if shift_flag == 1:
        img = spimage.sp_image_shift(img)

    try:
        spimage.sp_image_write(img,arguments[1],color)
    except:
        print "Error: Can not write %s\n" % arguments[1]
        exit(1)

if __name__ == "__main__":
    image_to_png(sys.argv[1:])

