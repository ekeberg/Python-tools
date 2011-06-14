#from pylab import *
#from spimage import *
import pylab
import spimage
import sys

def center_image_2d(img):
    sigma = 3
    #img = sp_image_read(filename,0)
    side = pylab.shape(img.image)[0]
    x = pylab.arange(pylab.shape(img.image)[0],dtype='float64') -\
        pylab.shape(img.image)[0]/2.0 + 0.5
    y = pylab.arange(pylab.shape(img.image)[1],dtype='float64') -\
        pylab.shape(img.image)[1]/2.0 + 0.5
    #X,Y,Z = meshgrid(x,y,z)
    #kernel = X**2+Y**2
    #kernel = exp(-(X**2+Y**2+Z**2)/2.0/sigma**2)
    kernel = pylab.exp(-(x[:,pylab.newaxis]**2+y[pylab.newaxis,:]**2)/2.0/sigma**2)

    #img.image[:,:] = kernel[:,:]
    #sp_image_write(img,"foo_kernel.h5",0)
    #exit(0)

    img_ft = pylab.fft2(pylab.fftshift(img.image))
    kernel_ft = pylab.fft2(pylab.fftshift(kernel))
    kernel_ft *= pylab.conj(img_ft)
    bt = pylab.ifft2(kernel_ft)

    min_v = 0.
    min_x = 0
    min_y = 0
    for x in range(pylab.shape(bt)[0]):
        for y in range(pylab.shape(bt)[1]):
            if abs(bt[y,x]) > min_v:
                    min_v = abs(bt[y,x])
                    min_x = x
                    min_y = y
    print min_x, min_y
    spimage.sp_image_translate(img,-(-min_x + pylab.shape(bt)[0]/2),
                               -(-min_y + pylab.shape(bt)[1]/2),
                               0, SP_TRANSLATE_WRAP_AROUND)
    shift = spimage.sp_image_shift(img)
    spimage.sp_image_free(img)
    #shift = img
    #sp_image_write(shift,outfile,0)
    return shift
    

def center_image_3d(img):
    sigma = 3
    #img = sp_image_read(filename,0)
    side = pylab.shape(img.image)[0]
    x = pylab.arange(pylab.shape(img.image)[0],dtype='float64') -\
        pylab.shape(img.image)[0]/2.0 + 0.5
    y = pylab.arange(pylab.shape(img.image)[1],dtype='float64') -\
        pylab.shape(img.image)[1]/2.0 + 0.5
    z = pylab.arange(pylab.shape(img.image)[2],dtype='float64') -\
        pylab.shape(img.image)[2]/2.0 + 0.5
    #X,Y,Z = meshgrid(x,y,z)
    #kernel = X**2+Y**2
    #kernel = exp(-(X**2+Y**2+Z**2)/2.0/sigma**2)
    kernel = pylab.exp(-(x[:,pylab.newaxis,pylab.newaxis]**2+y[pylab.newaxis,:,pylab.newaxis]**2+
                   z[pylab.newaxis,pylab.newaxis,:]**2)/2.0/sigma**2)

    #img.image[:,:] = kernel[:,:]
    #sp_image_write(img,"foo_kernel.h5",0)
    #exit(0)

    img_ft = pylab.fft2(pylab.fftshift(img.image))
    kernel_ft = pylab.fft2(pylab.fftshift(kernel))
    kernel_ft *= pylab.conj(img_ft)
    bt = pylab.ifft2(kernel_ft)

    min_v = 0.
    min_x = 0
    min_y = 0
    min_z = 0
    for x in range(pylab.shape(bt)[0]):
        for y in range(pylab.shape(bt)[1]):
            for z in range(pylab.shape(bt)[2]):
                if abs(bt[z,y,x]) > min_v:
                    min_v = abs(bt[z,y,x])
                    min_x = x
                    min_y = y
                    min_z = z
    print min_x, min_y, min_z
    # spimage.sp_image_translate(img,-(-min_z + pylab.shape(bt)[0]/2),
    #                            -(-min_y + pylab.shape(bt)[1]/2),
    #                            -(-min_x + pylab.shape(bt)[2]/2), SP_TRANSLATE_WRAP_AROUND)
    spimage.sp_image_translate(img,-(-min_z + pylab.shape(bt)[0]/2),
                               -(-min_y + pylab.shape(bt)[1]/2),
                               -(-min_x + pylab.shape(bt)[2]/2), 1)
    #shift = sp_image_shift(img)
    shift = img

    #sp_image_write(shift,outfile,0)
    return shift

def scale_image_2d(img, factor):
    """Scales up the image by the scaling factor. No cropping is done.
    Input:
    image
    factor"""
    sizeX = pylab.shape(img.image)[0]
    sizeY = pylab.shape(img.image)[1]
    centerX = sizeX/2
    centerY = sizeY/2
    windowX = int(sizeX/factor)
    windowY = int(sizeY/factor)
    ft = pylab.fft2(img.image[centerX-windowX/2:centerX+windowX/2,
                              centerY-windowY/2:centerY+windowY/2])
    rs = abs(pylab.ifftn(pylab.fftshift(ft),[sizeX,sizeY]))
    return rs
    #img.image[:,:] = rs[:,:]

def scale_image_3d(img, factor):
    """Scales up the image by the scaling factor.
    Input:
    image
    factor"""
    sizeX = pylab.shape(img.image)[0]
    sizeY = pylab.shape(img.image)[1]
    sizeZ = pylab.shape(img.image)[2]
    centerX = sizeX/2
    centerY = sizeY/2
    centerZ = sizeZ/2
    windowX = int(sizeX/factor)
    windowY = int(sizeY/factor)
    windowZ = int(sizeZ/factor)
    ft = pylab.fftn(img.image[centerX-windowX/2:centerX+windowX/2,
                              centerY-windowY/2:centerY+windowY/2,
                              centerZ-windowZ/2:centerZ+windowZ/2],
                    [sizeX,sizeY,sizeZ])
    rs = abs(pylab.ifftn(pylab.fftshift(ft),[sizeX,sizeY,sizeZ]))
    #return rs
    img.image[:,:] = rs[:,:]

