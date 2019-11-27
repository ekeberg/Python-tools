"""A collection of image manipulation functions. Some of them uses spimage objects
and some numpy arrays."""
import numpy as _numpy

def scale_image_2d(image, factor):
    """Scales up the image by the scaling factor. No cropping is done.
    Input:
    image
    factor"""
    size_x = image.shape[0]
    size_y = image.shape[1]
    center_x = size_x//2
    center_y = size_y//2
    window_x = int(size_x//factor)
    window_y = int(size_y//factor)
    image_ft = _numpy.fft2(image[center_x-window_x//2:center_x+window_x//2,
                                center_y-window_y//2:center_y+window_y//2])
    image_scaled = abs(_numpy.fft.ifftn(_numpy.fft.fftshift(image_ft), [size_x, size_y]))

    return image_scaled

def scale_image_3d(image, factor):
    """Scales up the image by the scaling factor.
    Input:
    image
    factor"""
    size_x = image.shape[0]
    size_y = image.shape[1]
    size_z = image.shape[2]
    center_x = size_x//2
    center_y = size_y//2
    center_z = size_z//2
    window_x = int(size_x//factor)
    window_y = int(size_y//factor)
    window_z = int(size_z//factor)
    image_ft = _numpy.fft.fftn(image[center_x-window_x//2:center_x+window_x//2,
                                center_y-window_y//2:center_y+window_y//2,
                                center_z-window_z//2:center_z+window_z//2],
                          [size_x, size_y, size_z])
    image_scaled = abs(_numpy.fft.ifftn(_numpy.fft.fftshift(image_ft), [size_x, size_y, size_z]))
    return image_scaled

def crop_and_pad(image, center, side):
    """Crops the image around the center to the side given. If the
    cropped area is larger than the original image it is padded with zeros"""
    dims = len(image.shape)
    if dims != 3 and dims != 2:
        raise ValueError("crop_and_pad: Input image must be 2 or three dimensional")
    if len(center) != dims:
        raise ValueError("crop_and_pad: Center must be same length as image dimensions ({0} != {1})".format(len(center), dims))
    center = tuple(round(center_element-0.5)+0.5 for center_element in center)

    ret = _numpy.zeros((side, )*dims, dtype=image.dtype)

    low_in = _numpy.float64(center)-side/2.0+0.5
    high_in = _numpy.float64(center)+side/2.0+0.5

    low_out = _numpy.zeros(dims)
    high_out = _numpy.array((side,)*dims)

    for i in range(dims):
        if low_in[i] < 0:
            low_out[i] += abs(low_in[i])
            low_in[i] = 0
        if high_in[i] > image.shape[i]:
            high_out[i] -= abs(image.shape[i] - high_in[i])
            high_in[i] = image.shape[i]

    low_in = _numpy.int32(low_in)
    high_in = _numpy.int32(high_in)
    low_out = _numpy.int32(low_out)
    high_out = _numpy.int32(high_out)
    if dims == 2:
        ret[low_out[0]:high_out[0], low_out[1]:high_out[1]] = image[low_in[0]:high_in[0], low_in[1]:high_in[1]]
    else:
        ret[low_out[0]:high_out[0], low_out[1]:high_out[1], low_out[2]:high_out[2]] = \
            image[low_in[0]:high_in[0], low_in[1]:high_in[1], low_in[2]:high_in[2]]

    return ret
