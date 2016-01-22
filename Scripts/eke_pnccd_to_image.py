#!/usr/bin/env python
import sys, h5py, pylab, spimage
from optparse import OptionParser

def pnccd_to_image(infile, outfile):
    try:
        f = h5py.File(infile)
    except:
        raise IOError("Can't read %s. It may not be a pnCCD file." % filename)

    i1 = f.keys().index('data')
    i2 = f.values()[i1].keys().index('data1')

    data = f.values()[i1].values()[i2].value

    img = spimage.sp_image_alloc(pylab.shape(data)[0],pylab.shape(data)[1],1)
    img.image[:,:] = data[:,:]
    spimage.sp_image_write(img,outfile,0)
    spimage.sp_image_free(img)

if __name__ == '__main__':
    parser = OptionParser(usage="%prog -i <pnccd_file.h5> -o <output_file>")
    parser.add_option("-i", "--input", action="store", type="string", dest="input",
                      help="Name of image to convert.")
    parser.add_option("-o", "--output", action="store", type="string", dest="output",
                      help="Writes output to this file.")
    (options, args) = parser.parse_args()
    
    pnccd_to_image(options.input, options.output)


