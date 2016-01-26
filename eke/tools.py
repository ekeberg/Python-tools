"""A mix of tools that didn't fit anywhere else."""
import numpy

class Resampler(object):
    """Can be used to resample arbitrarily positioned
    points to another set of arbitrarily positioned points."""
    def __init__(self, input_points, output_points):
        self._set_input_points(intput_points)
        self._set_output_points(output_points)
        self._calculate_table()

    def _set_input_points(self, grid):
        """Doesn't update the table, useful in constructor"""
        self._input_points = numpy.array(grid)
        if (len(self._input_points.shape) != 2 or self._input_points.shape[1] != 3):
            raise ValueError("Grid must be (n, 3) dimensional, {0} received.".format(str(self._input_points.shape)))
        self._calculate_table()

    def set_input_points(self, grid):
        """Updates table, useful when n is already set"""
        self._set_input_points(grid)
        self._calculate_table()

    def _set_output_points(self, grid):
        """Doesn't update the table, useful in constructor"""
        self._output_points = numpy.array(grid)
        if (len(self._output_points.shape) != 2 or self._output_points.shape[1] != 3):
            raise ValueError("Grid must be (n, 3) dimensional, {0} received.".format(str(self._output_points.shape)))
        self._calculate_table()

    def set_output_points(self, grid):
        """Updates table, useful when n is already set"""
        self._set_output_points(grid)
        self._calculate_table()

    def _calculate_table(self):
        def closest_coordinate(coordinate, points):
            """Calculate the point in points closest to the given coordinate."""
            return (((points - coordinate)**2).sum(axis=1)).argmax()

        number_of_input_points = len(self._input_points)
        number_of_output_points = len(self._output_points)
        self._table = numpy.zeros(number_of_input_points, dtype="float64")
        self._weights = numpy.zeros(number_of_output_points, dtype="float64")

        for i, input_point in enumerate(self._input_points):
            self._table[i] = closest_coordinate(input_point, self._output_points)
            self._weights[self._table[i]] += 1.
        self._output_points_without_input = self._weights == 0

    def remap(self, input_values):
        output_values = numpy.empty(len(self._output_points), dtype="float64")
        self.remap_in_place(input_values, output_values)
        return output_values

    def remap_in_place(self, input_values, output_values):
        if len(input_values) != len(self._input_points):
            raise ValuesError("Array size ({0}) different thatn precalculated ({1})".format(len(input_values), len(self._input_points)))
        output_values[:] = 0.
        for i, input_value in enumerate(input_values):
            output_values[self._table[i]] = input_values[i]
        output_values[-self._output_points_without_input] /= self._weights[-self._output_points_without_input]
        output_values[self._output_points_without_input] = 0.

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

def circular_mask(side, radius=None):
    """Returns a 2D bool array with a circular mask. If no radius is specified half of the
    array side is used."""
    import pylab
    if not radius:
        radius = side/2.
    x_array = pylab.arange(-side/2.+0.5, side/2.+0.5)
    radius2 = x_array[pylab.newaxis, :]**2 + x_array[:, pylab.newaxis]**2
    mask = radius2 < radius**2
    return mask

def ellipsoidal_mask(side, large_radius, small_radius, direction):
    """Not very well tested yet"""
    import numpy
    direction = direction / numpy.sqrt(direction[0]**2+direction[1]**2)
    x_array = numpy.arange(-side/2.+0.5, side/2.+0.5)
    radius2 = (((x_array[numpy.newaxis, :]*direction[1] + x_array[:, numpy.newaxis]*direction[0])/large_radius)**2 +
               ((x_array[numpy.newaxis, :]*(-direction[0]) + x_array[:, numpy.newaxis]*direction[1])/small_radius)**2)
    mask = radius2 < 1.
    return mask

def spherical_mask(side, radius=None):
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
    if mask is None:
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

        radial_average_out = pylab.zeros(side/2)
        weight = pylab.zeros(side/2)
        for value, this_radius in zip(image[in_range*mask], radius[in_range*mask]):
            radial_average_out[this_radius] += value
            weight[this_radius] += 1
        radial_average_out /= weight
        return radial_average_out

    elif len(image_shape) == 3:
        if image_shape[0] != image_shape[1] or image_shape[0] != image_shape[1]:
            raise ValueError("Image must be a cube")
        side = image_shape[0]
        x_array = pylab.arange(-side/2.+0.5, side/2.+0.5)
        radius = pylab.int32(pylab.sqrt(x_array**2 + x_array[:, pylab.newaxis]**2 +
                                        x_array[:, pylab.newaxis, pylab.newaxis]**2))
        in_range = radius < side/2.

        radial_average_out = pylab.zeros(side/2)
        weight = pylab.zeros(side/2)
        for value, this_radius in zip(image[in_range*mask], radius[in_range*mask]):
            radial_average_out[this_radius] += value
            weight[this_radius] += 1
        radial_average_out /= weight
        return radial_average_out

    else:
        raise ValueError("Image must be a 2d or 3d array")

def downsample(image, factor):
    """For now don't use a mask to make it speedier. For a mask
    a C implementation might be a better option to preserve speed."""
    import itertools
    
    output_size = numpy.array(image.shape) / factor
    end_index = output_size*factor

    reshape_parameters = []
    for this_output_size in output_size:
        reshape_parameters += [this_output_size, factor]
    
    image_view = image[[slice(0, index) for index in end_index]].reshape(*reshape_parameters)
    output_image = image_view.sum(axis=tuple(numpy.arange(len(image.shape))*2+1))
    
    return output_image

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
    """The central part of large_array is replaced by small_array."""
    this_slice = []
    for large_size, small_size in zip(large_image.shape, small_image.shape):
        if small_size%2:
            this_slice.append(slice(large_size/2-small_size/2, large_size/2+small_size/2+1))
        else:
            this_slice.append(slice(large_size/2-small_size/2, large_size/2+small_size/2))
    large_image[this_slice] = small_image

def insert_array(large_image, small_image, center):
    """Part of large_array centered around center is replaced by small_array."""
    this_slice = []
    for this_center, small_size in zip(center, small_image.shape):
        if small_size%2:
            this_slice.append(slice(this_center-small_size/2, this_center+small_size/2+1))
        else:
            this_slice.append(slice(this_center-small_size/2, this_center+small_size/2))
    large_image[this_slice] = small_image

def pad_with_zeros(small_image, size):
    """Put the image in the center of a new array with the specified size"""
    if len(size) != len(small_image.shape):
        raise ValueError("Input image is {0} dimensional and size is {1} dimensional".format(len(small_image.shape), len(size)))
    large_image = numpy.zeros(size, dtype=small_image.dtype)
    insert_array_at_center(large_image, small_image)
    return large_image

def enum(**enums):
    """Gives enumerate functionality to python."""
    return type('Enum', (), enums)

def log_range(min_value, max_value, steps):
    """A range that has the values logarithmically distributed"""
    import pylab
    return pylab.exp(pylab.arange(pylab.log(min_value), pylab.log(max_value), (pylab.log(max_value) - pylab.log(min_value))/steps))

def required_number_of_orientations(particle_size, resolution, prob):
    r = particle_size / resolution
    K = 4.*numpy.pi*(r-0.5)**2/2.
    k = 2.*numpy.pi*(r-0.5)/2.
    return numpy.log(1.-prob**(1./K)) / numpy.log(1.-k/K)

def central_slice(large_array_shape, small_array_shape):
    """Intended usage large_array[central_slice(large_array.shape, small_array.shape)] = small_array"""
    return [slice(l0/2-s0/2, l0/2+s0/2+s0%2) for l0, s0 in zip(large_array_shape, small_array_shape)]
