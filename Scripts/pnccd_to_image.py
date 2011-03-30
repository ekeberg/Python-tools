#! /usr/bin/python
import sys, h5py, pylab, spimage

def pnccd_to_image(infile, outfile):
    try:
        f = h5py.File(infile)
    except:
        print "Error reading file %s. It may not be a pnCCD file." % filename
        exit(1)

    i1 = f.keys().index('data')
    i2 = f.values()[i1].keys().index('data1')

    data = f.values()[i1].values()[i2].value

    img = spimage.sp_image_alloc(pylab.shape(data)[0],pylab.shape(data)[1],1)
    img.image[:,:] = data[:,:]
    spimage.sp_image_write(img,outfile,0)
    spimage.sp_image_free(img)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print """
Usage: pnccd_to_image <infile.h5> <outfile.h5>

This program is used to convert pnccd files to hawk files.

"""
        exit(0)
    
    pnccd_to_image(sys.argv[1],sys.argv[2])
