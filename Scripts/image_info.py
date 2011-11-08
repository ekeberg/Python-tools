
import spimage, sys
from optparse import OptionParser

def image_info(filename):
    try:
        img = spimage.sp_image_read(filename,0)
    except:
        print "Error: Can not read image %s" % filename
        exit(1)

    print "%s: (%dD image)" % (filename,img.num_dimensions)
    if img.num_dimensions == 3:
        print "size = %d x %d x %d" % (spimage.sp_image_x(img),
                                       spimage.sp_image_y(img),
                                       spimage.sp_image_z(img))
        print "center = %g x %g x %g" % (img.detector.image_center[0],
                                         img.detector.image_center[1],
                                         img.detector.image_center[2])
    else:
        print "size = %d x %d" % (spimage.sp_image_x(img),
                                  spimage.sp_image_y(img))
        print "center = %g x %g" % (img.detector.image_center[0],
                                    img.detector.image_center[1])

    print "shifted = %d" % img.shifted
    print "scaled = %d" % img.scaled
    print "phased = %d" % img.phased
    print "wavelength = %g" % img.detector.wavelength
    print "detector distance = %g" % img.detector.detector_distance
    if img.num_dimensions == 3:
        print "pixel size = %g x %g x %g" % (img.detector.pixel_size[0],
                                             img.detector.pixel_size[1],
                                             img.detector.pixel_size[2])
    else:
        print "pixel size = %g x %g" % (img.detector.pixel_size[0],
                                        img.detector.pixel_size[1])

if __name__ == "__main__":
    parser = OptionParser(usage="%prog <image.h5>")
    (options,args) = parser.parse_args()
    if len(args) < 1:
        print "You have to provide an h5 image file."

    image_info(args[0])


