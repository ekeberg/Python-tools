#from pylab import *
#from spimage import *
import pylab
import spimage
import sys

def center_image_2d(img, radius):
    """For an image with an object surounded by empty space, this function
    puts it in the center. A rough idea about the size of the object is
    needed in the form of the variable radius."""
    #sigma = 3
    sigma = radius
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
    

def center_image_3d(img, radius):
    """For an image with an object surounded by empty space, this function
    puts it in the center. A rough idea about the size of the object is
    needed in the form of the variable radius."""
    #sigma = 3
    sigma = radius
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
    kernel = pylab.exp(-(x[:,pylab.newaxis,pylab.newaxis]**2+
                         y[pylab.newaxis,:,pylab.newaxis]**2+
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

def crop_and_pad(image, center, side):
    """Crops the image around the center to the side given. If the
    cropped area is larger than the original image it is padded with zeros"""
    dims = len(pylab.shape(image))
    if dims != 3 and dims != 2:
        raise ValueError("crop_and_pad: Input image must be 2 or three dimensional")
    if len(center) != dims:
        raise ValueError("crop_and_pad: Center must be same length as image dimensions")

    ret = pylab.zeros((side,)*dims,dtype=image.dtype)
    
    low_in = pylab.array(center)-side/2.0-0.5
    high_in = pylab.array(center)+side/2.0-0.5

    low_out = pylab.zeros(dims)
    high_out = pylab.array((side,)*dims)

    for i in range(dims):
        if low_in[i] < 0:
            low_out[i] += abs(low_in[i])
            low_in[i] = 0
        if high_in[i] > pylab.shape(image)[i]:
            high_out[i] -= abs(pylab.shape(image)[i] - high_in[i])
            high_in[i] = pylab.shape(image)[i]

    if dims == 2:
        ret[low_out[0]:high_out[0],low_out[1]:high_out[1]] = image[low_in[0]:high_in[0],low_in[1]:high_in[1]]
    else:
        ret[low_out[0]:high_out[0],low_out[1]:high_out[1],low_out[2]:high_out[2]] = \
            image[low_in[0]:high_in[0],low_in[1]:high_in[1],low_in[2]:high_in[2]]

    return ret
