"""A mix of tools that didn't fit anywhere else."""

def get_h5_in_dir(path):
    "Returns a list of all the h5 files in a directory"
    import os
    import re
    all_files = os.listdir(path)
    hdf5_files = ["%s/%s" % (path, this_file) for this_file in all_files if re.search(".h5$", this_file)]
    return hdf5_files

def gaussian_blur_nonperiodic(array, sigma):
    """Only 1d at this point"""
    import pylab
    small_size = len(array)
    pad_size = int(sigma*10)
    large_size = small_size+2*pad_size
    large_array = pylab.zeros(large_size)
    large_array[pad_size:(small_size+pad_size)] = array
    large_array[:pad_size] = array[0]
    large_array[(small_size+pad_size):] = array[-1]

    x_array = pylab.arange(-large_size/2, large_size/2)
    kernel = (large_size/pylab.sqrt(2.*pylab.pi)/sigma*
              pylab.fftshift(pylab.exp(-2.*sigma**2*pylab.pi**2*(x_array/large_size)**2)))
    image_ft = pylab.fftn(large_array)
    image_ft *= kernel
    blured_large_array = pylab.ifftn(image_ft)
    return blured_large_array[pad_size:-pad_size]
    #return blured_large_array

def circular_mask(side, radius = None):
    """Returns a 2D bool array with a circular mask. If no radius is specified half of the
    array side is used."""
    import pylab
    if not radius:
        radius = side/2.
    x_array = pylab.arange(-side/2.+0.5, side/2.+0.5)
    radius2 = x_array[pylab.newaxis, :]**2 + x_array[:, pylab.newaxis]**2
    mask = radius2 < radius**2
    return mask

def spherical_mask(side, radius = None):
    """Returns a 3D bool array with a spherical mask. If no radius is specified, half of the
    array side is used."""
    import pylab
    if not radius:
        radius = side/2.
    x_array = pylab.arange(-side/2.+0.5, side/2.+0.5)
    radius2 = (x_array[pylab.newaxis, pylab.newaxis, :]**2 +
               x_array[pylab.newaxis, :, pylab.newaxis]**2 +
               x_array[:, pylab.newaxis, pylab.newaxis]**2)
    mask = radius2 < radius**2
    return mask

def remove_duplicates(input_list):
    """Return the same list but with only the first entry of every duplicate."""
    seen = []
    for index, value in enumerate(input_list):
        if value in seen:
            input_list.pop(index)
        else:
            seen.append(value)

def factorial(value):
    """Uses scipy.special.gamma."""
    from scipy.special import gamma as gamma_
    return gamma_(value+1)

def bincoef(n, k):
    """Binomial coefficient (n, k)."""
    from scipy.special import binom
    return binom(n, k)

def radial_average(image, mask=None):
    """Calculates the radial average of an array of any shape,
    the center is assumed to be at the physical center."""
    import pylab
    if mask == None:
        mask = pylab.ones(image.shape, dtype='bool8')
    else:
        mask = pylab.bool8(mask)
    axis_values = [pylab.arange(l) - l/2. + 0.5 for l in image.shape]
    radius = pylab.zeros((image.shape[-1]))
    for i in range(len(image.shape)):
        radius = radius + (axis_values[-(1+i)][(slice(0, None), ) + (pylab.newaxis, )*i])**2
    radius = pylab.int32(pylab.sqrt(radius))
    number_of_bins = radius[mask].max() + 1
    radial_sum = pylab.zeros(number_of_bins)
    weight = pylab.zeros(number_of_bins)
    for value, this_radius in zip(image[mask], radius[mask]):
        radial_sum[this_radius] += value
        weight[this_radius] += 1.
    return radial_sum / weight

def radial_average_simple(image, mask=None):
    """Calculates the radial average from the center to the edge (corners are not included)"""
    import pylab
    image_shape = pylab.shape(image)
    if mask is None:
        mask = True
    if len(image_shape) == 2:
        if image_shape[0] != image_shape[1]:
            raise ValueError("Image must be square")
        side = image_shape[0]
        x_array = pylab.arange(-side/2.+0.5, side/2.+0.5)
        radius = pylab.int32(pylab.sqrt(x_array**2 + x_array[:, pylab.newaxis]**2))
        in_range = radius < side/2.

        radial_average = pylab.zeros(side/2)
        weight = pylab.zeros(side/2)
        for value, this_radius in zip(image[in_range*mask], radius[in_range*mask]):
            radial_average[this_radius] += value
            weight[this_radius] += 1
        radial_average /= weight
        return radial_average
    elif len(image_shape) == 3:
        if image_shape[0] != image_shape[1] or image_shape[0] != image_shape[1]:
            raise ValueError("Image must be a cube")
        side = image_shape[0]
        x_array = pylab.arange(-side/2.+0.5, side/2.+0.5)
        radius = pylab.int32(pylab.sqrt(x_array**2 + x_array[:, pylab.newaxis]**2 +
                                        x_array[:, pylab.newaxis, pylab.newaxis]**2))
        in_range = radius < side/2.

        radial_average = pylab.zeros(side/2)
        weight = pylab.zeros(side/2)
        for value, this_radius in zip(image[in_range*mask], radius[in_range*mask]):
            radial_average[this_radius] += value
            weight[this_radius] += 1
        radial_average /= weight
        return radial_average
    else:
        raise ValueError("Image must be a 2d or 3d array")

def correlation(image1, image2):
    """Mathematical correlation F-1(F(i1) F(i1)*) (not Pearson correlation)"""
    import pylab
    image1_ft = pylab.fft2(image1)
    image2_ft = pylab.fft2(image2)
    return abs(pylab.fftshift(pylab.ifft2(pylab.conjugate(image1_ft)*image2_ft)))

def convolution(image1, image2):
    """Mathematical concvolution F-1(F(i1) F(i2))."""
    import pylab
    image1_ft = pylab.fft2(image1)
    image2_ft = pylab.fft2(image2)
    return abs(pylab.fftshift(pylab.ifft2(image1_ft*image2_ft)))

def pearson_correlation(data_1, data_2):
    """Pearson correlation."""
    import pylab
    return (((data_1-pylab.average(data_1)) * (data_2-pylab.average(data_2))).sum() /
            pylab.sqrt(((data_1-pylab.average(data_1))**2).sum()) /
            pylab.sqrt(((data_2-pylab.average(data_2))**2).sum()))

def sorted_indices(input_list):
    """Sorts the list a and returns a list of the original position in the list."""
    return [i[0] for i in sorted(enumerate(input_list), key=lambda x: x[1])]

def random_diffraction(image_size, object_size):
    """Calculate a diffraction pattern from a random object. Sizes are given in pixels."""
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
    
def insert_array_at_center(large_image, small_image):
    s = []
    for large_size, small_size in zip(large_image.shape, small_image.shape):
        if small_size%2:
            s.append(slice(large_size/2-small_size/2, large_size/2+small_size/2+1))
        else:
            s.append(slice(large_size/2-small_size/2, large_size/2+small_size/2))
    large_image[s] = small_image

def insert_array(large_image, small_image, center):
    s = []
    for this_center, small_size in zip(center, small_image.shape):
        if small_size%2:
            s.append(slice(this_center-small_size/2, this_center+small_size/2+1))
        else:
            s.append(slice(this_center-small_size/2, this_center+small_size/2))
    large_image[s] = small_image

def enum(**enums):
    """Gives enumerate functionality to python."""
    return type('Enum', (), enums)

def log_range(min_value, max_value, steps):
    """A range that has the values logarithmically distributed"""
    return pylab.exp(pylab.arange(pylab.log(min_value), pylab.log(max_value), (pylab.log(max_value) - pylab.log(min_value))/steps))
