"""Generate binary contrast particles with a tunable feature size."""
#import pylab
import numpy as _numpy

def elser_particle(array_size, particle_size, feature_size, return_blured=True):
    """Return a binary contrast particle. 'particle_size' and 'feature_size'
    should both be given in pixels."""
    if particle_size > array_size-2:
        raise ValueError("Particle size must be <= array_size is (%d > %d-2)" % (particle_size, array_size))

    x_coordinates = _numpy.arange(array_size) - array_size/2. + 0.5
    y_coordinates = _numpy.arange(array_size) - array_size/2. + 0.5
    z_coordinates = _numpy.arange(array_size) - array_size/2. + 0.5
    radius = _numpy.sqrt(x_coordinates[_numpy.newaxis, _numpy.newaxis, :]**2 +
                        y_coordinates[_numpy.newaxis, :, _numpy.newaxis]**2 +
                        z_coordinates[:, _numpy.newaxis, _numpy.newaxis]**2)
    particle_mask = radius > particle_size/2.
    #lp_mask = _numpy.fftn.fftshift(radius > array_size/(feature_size*4.))
    lp_kernel = _numpy.fft.fftshift(_numpy.exp(-radius**2*float(feature_size)**2/float(array_size)**2))

    particle = _numpy.random.random((array_size, )*3)

    for _ in xrange(4):
        # binary constrast
        particle_average = _numpy.median(particle[-particle_mask])
        particle[particle > particle_average] = 1.
        particle[particle <= particle_average] = 0.
        particle[particle_mask] = 0.
        # smoothen
        particle_ft = _numpy.fft.fftn(particle)
        particle_ft *= lp_kernel
        particle[:, :] = abs(_numpy.fft.ifftn(particle_ft))

    if not return_blured:
        particle_average = _numpy.median(particle[-particle_mask])
        particle[particle > particle_average] = 1.
        particle[particle <= particle_average] = 0.

    return particle

def oval_elser_particle(array_size, particle_size, rotation, feature_size, return_blured=True):
    raise NotImplementedError("Not yet implemented")
    import rotations
    x_coordinates = _numpy.arange(array_size) - array_size/2. + 0.5
    y_coordinates = _numpy.arange(array_size) - array_size/2. + 0.5
    z_coordinates = _numpy.arange(array_size) - array_size/2. + 0.5
    
    

def _elser_particle_old(resolution):
    """This is the original function from Veit Elser without the
    tunable feature size."""
    x_coordinates = _numpy.arange(resolution+2) - (resolution+2)/2.0 + 0.5
    y_coordinates = _numpy.arange(resolution+2) - (resolution+2)/2.0 + 0.5
    z_coordinates = _numpy.arange(resolution+2) - (resolution+2)/2.0 + 0.5
    radius = _numpy.sqrt(x_coordinates[_numpy.newaxis, _numpy.newaxis, :]**2 +
                        y_coordinates[_numpy.newaxis, :, _numpy.newaxis]**2 +
                        z_coordinates[:, _numpy.newaxis, _numpy.newaxis]**2)
    particle_mask = radius > resolution/2.0
    lp_mask = _numpy.fft.fftshift(radius > resolution/4.0)

    particle = _numpy.random.random((resolution+2, resolution+2, resolution+2))

    for _ in xrange(4):
        particle[particle_mask] = 0.0
        particle_ft = _numpy.fft.fftn(particle)
        particle_ft[lp_mask] = 0.0
        particle[:, :] = abs(_numpy.fft.ifftn(particle))
        particle_average = _numpy.average(particle.flatten())
        particle[particle > 0.5*particle_average] = 1.0
        particle[particle <= 0.5*particle_average] = 0.0

    particle[particle_mask] = 0.0
    return particle


# if __name__ == "__main__":
#     images = []
#     images_big = []
#     images_binary = []
#     sides = (2**_numpy.arange(7))[1:]
#     for s in sides:
#         print "generate side %d particle" % s
#         images.append(elser_particle(s))
#         image_ft = _numpy.fft.fftn(images[-1])
#         images_big.append(abs(_numpy.fft.ifftn(_numpy.fft.fftshift(image_ft), [sides[-1], sides[-1], sides[-1]])))
#         images_binary.append(images_big[-1] > 1.5*_numpy.average(images_big[-1].flatten()))


