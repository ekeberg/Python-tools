import pylab

def elser_particle(resolution):
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

    
