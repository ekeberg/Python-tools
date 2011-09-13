"""Script to crop an h5 image and pad with zeros around it"""
import spimage
import sys
import image_manipulation

def crop_image(in_file, out_file, side, center=None):
    """Function to crop an h5 image and pad with zeros around it"""

    img = spimage.sp_image_read(in_file, 0)

    if not center:
        center = img.detector.image_center[:2]

    shifted = 0
    if img.shifted:
        shifted = 1
        img = spimage.sp_image_shift(img)

    print "shifted = ", shifted

    #cropped = spimage.rectangle_crop(img,lowX,lowY,highX,highY)
    cropped = spimage.sp_image_alloc(side, side)
    cropped.image[:,:] = image_manipulation.crop_and_pad(img.image, center, side)
    cropped.mask[:,:] = image_manipulation.crop_and_pad(img.mask, center, side)

    print "did crop"

    if shifted:
        cropped = spimage.sp_image_shift(cropped)

    print "shifted (or not)"

    print "write ", out_file

    #print "orientation = ", cropped.detector.orientation
    #print spimage.sp_3matrix_get(cropped.detector.orientation,0,0,0)

    spimage.sp_image_write(cropped, out_file, 16)

    spimage.sp_image_free(img)
    spimage.sp_image_free(cropped)

    print "end"

if __name__ == "__main__":
    from optparse import OptionParser
    from optparse import OptionGroup
    parser = OptionParser(usage="%prog filename -i infile -o outfile -s side [-c center]")
    parser.add_option("-i", action="store", type="string", dest="infile",
                      help="Input file")
    parser.add_option("-o", action="store", type="float", dest="outfile",
                      help="Output file")
    parser.add_option("-s", action="store", type="int", dest="side",
                      help="New side in pixels.")
    parser.add_option("-c", action="store", type="string", dest="number", default=None,
                      help="Image center to crop around. If not given the center specified in the image is used.")
    (options,args) = parser.parse_args()

    if options.center:
        center = center.split('x')
        if len(center) != 2:
            raise ValueError("crop_image: Center must be of the form form %fx%f.")
        center = array([float(center[0]), float(center[1])])
    else:
        center = None

    try:
        crop_image(options.infile, options.outfile, options.side, center)
    except:
        print """
Usage:  python_script_crop_image <in.h5> <out.h5> side

Crops the image to the specified size symmetrically
around the image center. If only one side is given
xside is used for both sides.
"""
