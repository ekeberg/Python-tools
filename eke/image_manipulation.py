"""A collection of image manipulation functions. Some of them uses spimage objects
and some numpy arrays."""
import pylab
import spimage

def center_image_2d(img, radius):
    """For an image with an object surounded by empty space, this function
    puts it in the center. A rough idea about the size of the object is
    needed in the form of the variable radius."""
    sigma = radius
    x_coordinates = pylab.arange(pylab.shape(img.image)[0], dtype='float64') -\
                    pylab.shape(img.image)[0]/2.0 + 0.5
    y_coordinates = pylab.arange(pylab.shape(img.image)[1], dtype='float64') -\
                    pylab.shape(img.image)[1]/2.0 + 0.5
    kernel = pylab.exp(-(x_coordinates[:, pylab.newaxis]**2 +
                         y_coordinates[pylab.newaxis, :]**2)/2.0/sigma**2)

    img_ft = pylab.fft2(pylab.fftshift(img.image))
    kernel_ft = pylab.fft2(pylab.fftshift(kernel))
    kernel_ft *= pylab.conj(img_ft)
    img_bt = pylab.ifft2(kernel_ft)

    min_v = 0.
    min_x = 0
    min_y = 0
    for x_index in range(pylab.shape(img_bt)[0]):
        for y_index in range(pylab.shape(img_bt)[1]):
            if abs(img_bt[y_index, x_index]) > min_v:
                min_v = abs(img_bt[y_index, x_index])
                min_x = x_index
                min_y = y_index
    print min_x, min_y
    spimage.sp_image_translate(img, -(-min_x + pylab.shape(img_bt)[0]/2),
                               -(-min_y + pylab.shape(img_bt)[1]/2),
                               0, spimage.SP_TRANSLATE_WRAP_AROUND)
    shift = spimage.sp_image_shift(img)
    spimage.sp_image_free(img)
    return shift

def center_image_3d(img, radius):
    """For an image with an object surounded by empty space, this function
    puts it in the center. A rough idea about the size of the object is
    needed in the form of the variable radius."""
    sigma = radius
    x_coordinates = pylab.arange(pylab.shape(img.image)[0], dtype='float64') -\
                    pylab.shape(img.image)[0]/2.0 + 0.5
    y_coordinates = pylab.arange(pylab.shape(img.image)[1], dtype='float64') -\
                    pylab.shape(img.image)[1]/2.0 + 0.5
    z_coordinates = pylab.arange(pylab.shape(img.image)[2], dtype='float64') -\
                    pylab.shape(img.image)[2]/2.0 + 0.5
    kernel = pylab.exp(-(x_coordinates[:, pylab.newaxis, pylab.newaxis]**2+
                         y_coordinates[pylab.newaxis, :, pylab.newaxis]**2+
                         z_coordinates[pylab.newaxis, pylab.newaxis, :]**2)/2.0/sigma**2)

    img_ft = pylab.fft2(pylab.fftshift(img.image))
    kernel_ft = pylab.fft2(pylab.fftshift(kernel))
    kernel_ft *= pylab.conj(img_ft)
    img_bt = pylab.ifft2(kernel_ft)

    min_v = 0.
    min_x = 0
    min_y = 0
    min_z = 0
    for x_index in range(pylab.shape(img_bt)[0]):
        for y_index in range(pylab.shape(img_bt)[1]):
            for z_index in range(pylab.shape(img_bt)[2]):
                if abs(img_bt[z_index, y_index, x_index]) > min_v:
                    min_v = abs(img_bt[z_index, y_index, x_index])
                    min_x = x_index
                    min_y = y_index
                    min_z = z_index
    print min_x, min_y, min_z
    spimage.sp_image_translate(img, -(-min_z + pylab.shape(img_bt)[0]/2),
                               -(-min_y + pylab.shape(img_bt)[1]/2),
                               -(-min_x + pylab.shape(img_bt)[2]/2), 1)
    shift = img

    return shift

def scale_image_2d(image, factor):
    """Scales up the image by the scaling factor. No cropping is done.
    Input:
    image
    factor"""
    size_x = pylab.shape(image)[0]
    size_y = pylab.shape(image)[1]
    center_x = size_x/2
    center_y = size_y/2
    window_x = int(size_x/factor)
    window_y = int(size_y/factor)
    image_ft = pylab.fft2(image[center_x-window_x/2:center_x+window_x/2,
                                center_y-window_y/2:center_y+window_y/2])
    image_scaled = abs(pylab.ifftn(pylab.fftshift(image_ft), [size_x, size_y]))

    return image_scaled

def scale_image_3d(image, factor):
    """Scales up the image by the scaling factor.
    Input:
    image
    factor"""
    size_x = pylab.shape(image)[0]
    size_y = pylab.shape(image)[1]
    size_z = pylab.shape(image)[2]
    center_x = size_x/2
    center_y = size_y/2
    center_z = size_z/2
    window_x = int(size_x/factor)
    window_y = int(size_y/factor)
    window_z = int(size_z/factor)
    image_ft = pylab.fftn(image[center_x-window_x/2:center_x+window_x/2,
                                center_y-window_y/2:center_y+window_y/2,
                                center_z-window_z/2:center_z+window_z/2],
                          [size_x, size_y, size_z])
    image_scaled = abs(pylab.ifftn(pylab.fftshift(image_ft), [size_x, size_y, size_z]))
    return image_scaled

def crop_and_pad(image, center, side):
    """Crops the image around the center to the side given. If the
    cropped area is larger than the original image it is padded with zeros"""
    dims = len(pylab.shape(image))
    if dims != 3 and dims != 2:
        raise ValueError("crop_and_pad: Input image must be 2 or three dimensional")
    if len(center) != dims:
        raise ValueError("crop_and_pad: Center must be same length as image dimensions")

    ret = pylab.zeros((side, )*dims, dtype=image.dtype)

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
        ret[low_out[0]:high_out[0], low_out[1]:high_out[1]] = image[low_in[0]:high_in[0], low_in[1]:high_in[1]]
    else:
        ret[low_out[0]:high_out[0], low_out[1]:high_out[1], low_out[2]:high_out[2]] = \
            image[low_in[0]:high_in[0], low_in[1]:high_in[1], low_in[2]:high_in[2]]

    return ret
