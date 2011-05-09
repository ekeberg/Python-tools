
from pylab import *
from spimage import *
import sys

def center_image(filename,outfile):
    sigma = 3
    img = sp_image_read(filename,0)
    side = shape(img.image)[0]
    x = arange(shape(img.image)[0],dtype='float64') -\
        shape(img.image)[0]/2.0 + 0.5
    y = arange(shape(img.image)[1],dtype='float64') -\
        shape(img.image)[1]/2.0 + 0.5
    z = arange(shape(img.image)[2],dtype='float64') -\
        shape(img.image)[2]/2.0 + 0.5
    #X,Y,Z = meshgrid(x,y,z)
    #kernel = X**2+Y**2
    #kernel = exp(-(X**2+Y**2+Z**2)/2.0/sigma**2)
    kernel = exp(-(x[:,newaxis,newaxis]**2+y[newaxis,:,newaxis]**2+
                   y[newaxis,newaxis,:]**2)/2.0/sigma**2)

    #img.image[:,:] = kernel[:,:]
    #sp_image_write(img,"foo_kernel.h5",0)
    #exit(0)

    img_ft = fft2(fftshift(img.image))
    kernel_ft = fft2(fftshift(kernel))
    kernel_ft *= conj(img_ft)
    bt = ifft2(kernel_ft)

    min_v = 0.
    min_x = 0
    min_y = 0
    min_z = 0
    for x in range(shape(bt)[0]):
        for y in range(shape(bt)[1]):
            for z in range(shape(bt)[2]):
                if abs(bt[z,y,x]) > min_v:
                    min_v = abs(bt[z,y,x])
                    min_x = x
                    min_y = y
                    min_z = z
    print min_x, min_y, min_z
    sp_image_translate(img,-(-min_z + shape(bt)[0]/2),
                       -(-min_y + shape(bt)[1]/2),
                       -(-min_x + shape(bt)[2]/2), SP_TRANSLATE_WRAP_AROUND)
    #shift = sp_image_shift(img)
    shift = img

    sp_image_write(shift,outfile,0)
    

if __name__ == "__main__":
    center_image(sys.argv[1],sys.argv[2])
