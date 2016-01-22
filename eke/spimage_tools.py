"""Collection of tools to make it easier to work with the spimage python interface."""
import numpy as _numpy
import spimage as _spimage

def image_from_array(image, mask=None):
    """Create spimage image type from numpy array with optional mask"""
    if len(image.shape) == 2:
        img = _spimage.sp_image_alloc(image.shape[0], image.shape[1], 1)
    elif len(image.shape) == 3:
        img = _spimage.sp_image_alloc(image.shape[0], image.shape[1], image.shape[2])
    else:
        raise ValueError("Array must be 2 or 3 dimensional")
    img.image[:] = image
    if mask != None:
        img.mask[:] = _numpy.int32(mask)
    else:
        img.mask[:] = 1
    img.shifted = 0
    img.phased = int(bool(_numpy.iscomplex(image).sum() > 0))
    return img

def allocate_image(shape):
    """Allocate spimage image object from numpy type shape"""
    if len(shape) == 2:
        img = _spimage.sp_image_alloc(shape[0], shape[1], 1)
    elif len(shape) == 3:
        img = _spimage.sp_image_alloc(shape[0], shape[1], shape[2])
    else:
        raise ValueError("Array must be 2 or 3 dimensional")
    return img

def get_basic_phaser(amplitudes, support, mask=None):
    """This function is not finished"""
    if mask == None:
        amplitudes_sp = image_from_array(amplitudes)
    else:
        amplitudes_sp = image_from_array(amplitudes, mask)
    beta = _spimage.sp_smap_alloc(1)
    _spimage.sp_smap_insert(beta, 0, 0.9)
    phase_alg = _spimage.sp_phasing_hio_alloc(beta, _spimage.SpNoConstraints)
    support_sp = image_from_array(support)
    sup_alg = _spimage.sp_support_array_init(_spimage.sp_support_static_alloc(), 20)

    phaser = _spimage.sp_phaser_alloc()
    _spimage.sp_phaser_init(phaser, phase_alg, sup_alg, _spimage.SpEngineCUDA)
    _spimage.sp_phaser_set_amplitudes(phaser, amplitudes_sp)
    _spimage.sp_phaser_init_model(phaser, None, _spimage.SpModelRandomPhases)
    _spimage.sp_phaser_init_support(phaser, support_sp, 0, 0)

    return phaser

def smap(values):
    """Create spimage smap object from iterable of (iteration, value) pairs."""
    import collections
    if not isinstance(values, collections.Iterable):
        values = [(0, values)]

    smap_out = _spimage.sp_smap_alloc(len(values))
    for i in values:
        _spimage.sp_smap_insert(smap_out, i[0], i[1])

    return smap_out

def get_constraints(real=False, positive=False):
    """It is easy to forget the name of the constraints variable. This function
    returns the appropriate one."""
    if real and positive:
        return _spimage.SpPositiveRealObject
    elif real:
        return _spimage.SpRealObject
    elif positive:
        return _spimage.SpPositiveComplexObject
    else:
        return _spimage.SpNoConstraints

def algorithm_hio(beta, real=False, positive=False):
    """Create an HIO algorithm object"""
    beta_sp = smap(beta)
    constraints = get_constraints(real, positive)
    algorithm = _spimage.sp_phasing_hio_alloc(beta_sp, constraints)
    return algorithm

def algorithm_raar(beta, real=False, positive=False):
    """Create a RAAR algorithm object"""
    beta_sp = smap(beta)
    constraints = get_constraints(real, positive)
    algorithm = _spimage.sp_phasing_raar_alloc(beta_sp, constraints)
    return algorithm

def algorithm_er(real=False, positive=False):
    """Create an ER algorithm object"""
    constraints = get_constraints(real, positive)
    algorithm = _spimage.sp_phasing_er_alloc(constraints)
    return algorithm

def support_static():
    """Create a support update array with only a static support."""
    algorithm = _spimage.sp_support_array_init(_spimage.sp_support_static_alloc(), 20)
    return algorithm

def support_area(area, blur_radius):
    """Create a support update array with only an area support."""
    area_sp = smap(area)
    blur_radius_sp = smap(blur_radius)
    algorithm = _spimage.sp_support_array_init(_spimage.sp_support_area_alloc(blur_radius_sp, area_sp), 20)
    return algorithm

def support_threshold(threshold, blur_radius):
    """Create a support update array with only a threshold support."""
    threshold_sp = smap(threshold)
    blur_radius_sp = smap(blur_radius)
    algorithm = _spimage.sp_support_array_init(_spimage.sp_support_threshold_alloc(blur_radius_sp, threshold_sp), 20)
    return algorithm

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
