"""Generate binary contrast particles with a tunable feature size."""
from __future__ import absolute_import
import numpy as _numpy
import scipy.fft as _fft
from eke import tools


def elser_particle(array_size, particle_size,
                   feature_size, return_blured=True):
    """Return a binary contrast particle. 'particle_size' and 'feature_size'
    should both be given in pixels."""
    if particle_size > array_size-2:
        raise ValueError("Particle size must be <= array_size is "
                         f"({particle_size} > {array_size}-2)")

    x_coordinates = _numpy.arange(array_size) - array_size/2. + 0.5
    y_coordinates = _numpy.arange(array_size) - array_size/2. + 0.5
    z_coordinates = _numpy.arange(array_size) - array_size/2. + 0.5
    radius = _numpy.sqrt(x_coordinates[_numpy.newaxis, _numpy.newaxis, :]**2 +
                         y_coordinates[_numpy.newaxis, :, _numpy.newaxis]**2 +
                         z_coordinates[:, _numpy.newaxis, _numpy.newaxis]**2)
    particle_mask = radius > particle_size/2.
    kernel_scaling = float(feature_size)**2 / float(array_size)**2
    lp_kernel = _fft.fftshift(_numpy.exp(-radius**2 * kernel_scaling))

    particle = _numpy.random.random((array_size, )*3)

    for _ in range(4):
        # binary constrast
        particle_average = _numpy.median(particle[~particle_mask])
        particle[particle > particle_average] = 1.
        particle[particle <= particle_average] = 0.
        particle[particle_mask] = 0.
        # smoothen
        particle_ft = _fft.fftn(particle)
        particle_ft *= lp_kernel
        particle[:, :] = abs(_fft.ifftn(particle_ft))

    if not return_blured:
        particle_average = _numpy.median(particle[~particle_mask])
        particle[particle > particle_average] = 1.
        particle[particle <= particle_average] = 0.

    return particle


def elser_particle_nd(array_shape, feature_size,
                      mask=None, return_blured=True):
    """Return a binary contrast particle of arbitrary dimensionality and shape.
    Feature size is given in pixels. If no mask is provided the paritlce will
    be spherical."""
    radius = tools.radial_distance(array_shape)
    if mask is None:
        particle_size = min(array_shape)-2
        mask = radius < particle_size/2.
    elif mask.shape != array_shape:
        raise ValueError("array_shape and mask.shape are different "
                         f"({array_shape} != {mask.shape})")
    coordinates = [_numpy.arange(this_shape) - this_shape/2 + 0.5
                   for this_shape in array_shape]
    scaling = float(feature_size)**2/float(len(coordinates[0]))**2
    component_exp = [_numpy.exp(-this_coordinate**2*scaling)
                     for this_coordinate in coordinates]
    lp_kernel = _numpy.ones(array_shape)
    for index, this_exp in enumerate(component_exp):
        this_slice = [_numpy.newaxis]*len(array_shape)
        this_slice[index] = slice(None)
        this_slice = tuple(this_slice)
        lp_kernel *= this_exp[this_slice]
    lp_kernel = _fft.fftshift(lp_kernel)

    particle = _numpy.random.random(array_shape)
    for _ in range(4):
        particle_average = _numpy.median(particle[mask])
        particle[particle > particle_average] = 1.
        particle[particle <= particle_average] = 0.
        particle[~mask] = 0.
        particle_ft = _fft.fftn(particle)
        particle_ft *= lp_kernel
        particle[:] = abs(_fft.ifftn(particle_ft))

    if not return_blured:
        particle_average = _numpy.median(particle[mask])
        particle[particle > particle_average] = 1.
        particle[particle <= particle_average] = 0.
    return particle


def oval_elser_particle(array_size, particle_size, rotation,
                        feature_size, return_blured=True):
    raise NotImplementedError("Not yet implemented")


def _elser_particle_old(resolution):
    """This is the original function from Veit Elser without the
    tunable feature size."""
    x_coordinates = _numpy.arange(resolution+2) - (resolution+2)/2.0 + 0.5
    y_coordinates = _numpy.arange(resolution+2) - (resolution+2)/2.0 + 0.5
    z_coordinates = _numpy.arange(resolution+2) - (resolution+2)/2.0 + 0.5
    radius = _numpy.sqrt(x_coordinates[_numpy.newaxis, _numpy.newaxis, :]**2 +
                         y_coordinates[_numpy.newaxis, :, _numpy.newaxis]**2 +
                         z_coordinates[:, _numpy.newaxis, _numpy.newaxis]**2)
    particle_mask = radius > resolution/2.
    lp_mask = _fft.fftshift(radius > resolution/4.)

    particle = _numpy.random.random((resolution+2, resolution+2, resolution+2))

    for _ in range(4):
        particle[particle_mask] = 0.0
        particle_ft = _fft.fftn(particle)
        particle_ft[lp_mask] = 0.0
        particle[:, :] = abs(_fft.ifftn(particle))
        particle_average = _numpy.average(particle.flatten())
        particle[particle > 0.5*particle_average] = 1.0
        particle[particle <= 0.5*particle_average] = 0.0

    particle[particle_mask] = 0.0
    return particle
