"""Generate binary contrast particles with a tunable feature size."""
from __future__ import absolute_import
import numpy as numpy
import scipy
from eke import tools


def elser_particle(array_shape, feature_size,
                   mask=None, return_blured=True):
    """Return a binary contrast particle of arbitrary dimensionality and shape.
    Feature size is given in pixels and is sigma*4. If no mask is provided the
    paritcle will be spherical."""
    radius = tools.radial_distance(array_shape)
    if mask is None:
        particle_size = min(array_shape)-2
        mask = radius < particle_size/2.
    elif mask.shape != array_shape:
        raise ValueError("array_shape and mask.shape are different "
                         f"({array_shape} != {mask.shape})")

    particle = numpy.random.random(array_shape)
    num_iterations = 4
    for _ in range(num_iterations):
        particle_average = numpy.median(particle[mask])
        particle[particle > particle_average] = 1.
        particle[particle <= particle_average] = 0.
        particle[~mask] = 0.

        particle = scipy.ndimage.gaussian_filter(particle, feature_size/4)

    if not return_blured:
        particle_average = numpy.median(particle[mask])
        particle[particle > particle_average] = 1.
        particle[particle <= particle_average] = 0.

    return particle


def elser_particle_nd(array_shape, feature_size,
                      mask=None, return_blured=True):
    raise DeprecationWarning("Use elser_particle instead")
    return elser_particle(array_shape, feature_size, mask, return_blured)
