
import sys, h5py, pylab, spimage
from optparse import OptionParser

def split_pnccd(filename):
    f = h5py.File(filename)

    i1 = f.keys().index('data')
    i2 = f.values()[i1].keys().index('data1')
    i2 = f.values()[i1].keys().index('data1')

    data2 = f.values()[i1].values()[i2].value

    data2_1 = spimage.sp_image_alloc(data2.shape[1]/2,data2.shape[0],1)
    data2_2 = spimage.sp_image_alloc(data2.shape[1]/2,data2.shape[0],1)
    data2_1.image[:,:] = data2[:,:(data2.shape[1]/2)]
    data2_2.image[:,:] = data2[:,(data2.shape[1]/2):]

    spimage.sp_image_write(data2_1,filename[:-3]+"_part1.h5",0)
    spimage.sp_image_write(data2_2,filename[:-3]+"_part2.h5",0)
    

if __name__ == '__main__':
    parser = OptionParser(usage="%prog PNCCD_FILE")
    (options, args) = parser.parse_args()
    
    if len(args) == 0:
        print "You must provide a pnCCD file"
        exit(1)
    
    split_pnccd(args[0])
