
import spimage, sys
from optparse import OptionParser

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
        print "Error: Can not write to %s" % out_file

if __name__ == "__main__":
    parser = OptionParser(usage="%prog <input_image.h5> [<output_image.h5>]")
    
    (options, args) = parser.parse_args()
    if len(args) == 0:
        print "An input image is needed"
        exit(0)

    in_file = args[0]
    
    if len(args) == 1:
        out_file = in_file
    else:
        out_file = args[1]

    shift_image(in_file, out_file)
