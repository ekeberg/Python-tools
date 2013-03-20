
def get_h5_in_dir(path):
    "Returns a list of all the h5 files in a directory"
    import os
    import re
    l = os.listdir(path)
    files = ["%s/%s" % (path,f) for f in l if re.search("\.h5$",f)]
    return files
#instead use scipy.ndimage.filters.gaussian_filter    
# def gaussian_blur(image, sigma):
#     """Returns a blured version of the image. The kernel is a gaussian with radius sigma."""
#     import pylab
#     size = pylab.shape(image)
#     #axes = [(pylab.arange(axis_size) - axis_size/2.0 + 0.5)/axis_size for axis_size in size]
#     axes = [(pylab.arange(axis_size) - axis_size/2.0)/axis_size for axis_size in size]
#     # x = pylab.arange(size[0]) - size[0]/2.0 + 0.5
#     # y = pylab.arange(size[1]) - size[1]/2.0 + 0.5
#     # z = pylab.arange(size[2]) - size[2]/2.0 + 0.5
#     # kernel = pylab.fftshift(pylab.exp(-2.0*sigma**2*pylab.pi**2*((x/size[0])**2 +
#     #                                                              (y[:,pylab.newaxis]/size[1])**2 +
#     #                                                              (z[:,pylab.newaxis,pylab.newaxis]/size[2])**2)))
#     kernel = axes[0]**2
#     for this_axis in axes[1:]:
#         kernel = kernel.reshape(kernel.shape+(1,)) + this_axis**2
#     image_ft = pylab.fftn(image)
#     image_ft*= kernel
#     product = pylab.ifftn(image_ft)
#     return product

def gaussian_blur_nonperiodic(a, sigma):
    """Only 1d at this point"""
    import pylab
    small_size = len(a)
    pad_size = int(sigma*10)
    large_size = small_size+2*pad_size
    large_a = pylab.zeros(large_size)
    large_a[pad_size:(small_size+pad_size)] = a
    large_a[:pad_size] = a[0]
    large_a[(small_size+pad_size):] = a[-1]
    
    x = pylab.arange(-large_size/2, large_size/2)
    #kernel = large_size*pylab.sqrt(pylab.pi/2./pylab.pi/sigma)/sigma*pylab.fftshift(pylab.exp(-2.*sigma**2*pylab.pi**2*(x/large_size)**2))
    kernel = large_size/pylab.sqrt(2.*pylab.pi)/sigma*pylab.fftshift(pylab.exp(-2.*sigma**2*pylab.pi**2*(x/large_size)**2))
    image_ft = pylab.fftn(large_a)
    image_ft *= kernel
    blured_large_a = pylab.ifftn(image_ft)
    return blured_large_a[pad_size:-pad_size]
    #return blured_large_a

def translate(image, dist):
    pass

def circular_mask(side, radius = None):
    import pylab
    if not radius:
        radius = side/2.
    x = pylab.arange(-side/2.+0.5, side/2.+0.5)
    radius2 = x**2 + x[:,pylab.newaxis]**2
    mask = radius2 < radius**2
    return mask

def spherical_mask(side, radius = None):
    import pylab
    if not radius:
        radius = side/2.
    x = pylab.arange(-side/2.+0.5, side/2.+0.5)
    radius2 = x**2 + x[:, pylab.newaxis]**2 + x[:, pylab.newaxis, pylab.newaxis]**2
    mask = radius2 < radius**2
    return mask

def remove_duplicates(input_list):
    seen = []
    for index, value in enumerate(input_list):
        if value in seen:
            input_list.pop(index)
        else:
            seen.append(value)

def factorial(n):
    from scipy.special import gamma as gamma_
    return gamma_(n+1)

def bincoef(n, k):
    from scipy.special import binom
    return binom(n,k)

def radial_average(image, mask=None):
    """Calculates the radial average of an array of any shape, the center is assumed to be at the physical center."""
    import pylab
    if mask == None:
        mask = pylab.ones(image.shape, dtype='bool8')
    else:
        mask = pylab.bool8(mask)
    x = [pylab.arange(l) - l/2. + 0.5 for l in image.shape]
    r = pylab.zeros((image.shape[-1]))
    for i in range(len(image.shape)):
        r = r + (x[-(1+i)][(slice(0,None),) + (pylab.newaxis,)*i])**2
    r = pylab.int32(pylab.sqrt(r))
    number_of_bins = r[mask].max() + 1
    radial_sum = pylab.zeros(number_of_bins)
    weight = pylab.zeros(number_of_bins)
    for v, r in zip(image[mask], r[mask]):
        radial_sum[r] += v
        weight[r] += 1.
    return radial_sum / weight

def radial_average_simple(image, mask=None):
    """Calculates the radial average from the center to the edge (corners are not included)"""
    import pylab
    image_shape = pylab.shape(image)
    if len(image_shape) == 2:
        if image_shape[0] != image_shape[1]:
            raise ValueError("Image must be square")
        side = image_shape[0]
        x = pylab.arange(-side/2.+0.5, side/2.+0.5)
        radius = pylab.int32(pylab.sqrt(x**2 + x[:, pylab.newaxis]**2))
        in_range = radius < side/2.

        radial_average = pylab.zeros(side/2)
        weight = pylab.zeros(side/2)
        for v, r in zip(image[in_range], radius[in_range]):
            radial_average[r] += v
            weight[r] += 1
        radial_average /= weight
        return radial_average
    elif len(image_shape) == 3:
        if image_shape[0] != image_shape[1] or image_shape[0] != image_shape[1]:
            raise ValueError("Image must be a cube")
        side = image_shape[0]
        x = pylab.arange(-side/2.+0.5, side/2.+0.5)
        radius = pylab.int32(pylab.sqrt(x**2 + x[:, pylab.newaxis]**2 + x[:, pylab.newaxis, pylab.newaxis]**2))
        in_range = radius < side/2.

        radial_average = pylab.zeros(side/2)
        weight = pylab.zeros(side/2)
        for v, r in zip(image[in_range], radius[in_range]):
            radial_average[r] += v
            weight[r] += 1
        radial_average /= weight
        return radial_average
    else:
        raise ValueError("Image must be a 2d or 3d array")

def correlation(image1, image2):
    import pylab
    image1_ft = pylab.fft2(image1)
    image2_ft = pylab.fft2(image2)
    correlation = abs(pylab.fftshift(pylab.ifft2(pylab.conjugate(image1_ft)*image2_ft)))
    return correlation

def convolution(image1, image2):
    import pylab
    image1_ft = pylab.fft2(image1)
    image2_ft = pylab.fft2(image2)
    convolution = abs(pylab.fftshift(pylab.ifft2(image1_ft*image2_ft)))
    return convolution

def pearson_correlation(data_1, data_2):
    import pylab
    return (((data_1-pylab.average(data_1)) * (data_2-pylab.average(data_2))).sum() / 
            pylab.sqrt(((data_1-pylab.average(data_1))**2).sum()) /
            pylab.sqrt(((data_2-pylab.average(data_2))**2).sum()))
            
def sorted_indices(a):
    return [i[0] for i in sorted(enumerate(a), key=lambda x: x[1])]

def random_diffraction(image_size, object_size):
    if image_size < object_size:
        raise ValueError("image_size must be larger than object size")
    import pylab
    image_real = pylab.zeros((image_size, )*2)
    lower_bound = pylab.floor(object_size/2.)
    higher_bound = pylab.ceil(object_size/2.)
    image_real[image_size/2-lower_bound:image_size/2+higher_bound,
               image_size/2-lower_bound:image_size/2+higher_bound] = pylab.random((object_size, )*2)
    image_fourier = pylab.fftshift(pylab.fft2(pylab.fftshift(image_real)))
    return image_fourier
    
