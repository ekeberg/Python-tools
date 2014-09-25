import pylab

def elser_particle(array_size, particle_size, feature_size):
    """Return a binary contrast particle. 'particle_size' and 'feature_size' should both be given in pixels."""
    if (particle_size > array_size-2): raise ValueError("Particle size must be <= array_size is (%d > %d-2)" % (particle_size, array_size))

    x = pylab.arange(array_size) - array_size/2. + 0.5
    y = pylab.arange(array_size) - array_size/2. + 0.5
    z = pylab.arange(array_size) - array_size/2. + 0.5
    r = pylab.sqrt(x**2 + y[:,pylab.newaxis]**2 + z[:,pylab.newaxis,pylab.newaxis]**2)
    particle_mask = r > particle_size/2.
    lp_mask = pylab.fftshift(r > array_size/(feature_size*4.))
    lp_kernel = pylab.fftshift(pylab.exp(-r**2*float(feature_size)**2/float(array_size)**2))

    particle = pylab.random((array_size, )*3)

    for i in range(4):
        # binary constrast
        particle_average = pylab.median(particle[-particle_mask])
        particle[particle > particle_average] = 1.
        particle[particle <= particle_average] = 0.
        particle[particle_mask] = 0.
        # smoothen
        particle_ft = pylab.fftn(particle)
        particle_ft *= lp_kernel
        particle[:,:] = abs(pylab.ifftn(particle_ft))
        
    return particle

def elser_particle_old(resolution):
    x = pylab.arange(resolution+2) - (resolution+2)/2.0 + 0.5
    y = pylab.arange(resolution+2) - (resolution+2)/2.0 + 0.5
    z = pylab.arange(resolution+2) - (resolution+2)/2.0 + 0.5
    r = pylab.sqrt(x**2+y[:,pylab.newaxis]**2+z[:,pylab.newaxis,pylab.newaxis]**2)
    particle_mask = r > resolution/2.0
    lp_mask = pylab.fftshift(r > resolution/4.0)
    
    particle = pylab.random((resolution+2,resolution+2,resolution+2))


    for i in range(4):
        particle[particle_mask] = 0.0
        particle_ft = pylab.fftn(particle)
        particle_ft[lp_mask] = 0.0
        particle[:,:] = abs(pylab.ifftn(particle))
        particle_average = pylab.average(particle.flatten())
        particle[particle > 0.5*particle_average] = 1.0
        particle[particle <= 0.5*particle_average] = 0.0

    particle[particle_mask] = 0.0
    return particle


if __name__ == "__main__":
    images = []
    images_big = []
    images_binary = []
    sides = (2**pylab.arange(7))[1:]
    for s in sides:
        print "generate side %d particle" % s
        images.append(elser_particle(s))
        image_ft = pylab.fftn(images[-1])
        images_big.append(abs(pylab.ifftn(pylab.fftshift(image_ft),[sides[-1],sides[-1],sides[-1]])))
        images_binary.append(images_big[-1] > 1.5*pylab.average(images_big[-1].flatten()))


