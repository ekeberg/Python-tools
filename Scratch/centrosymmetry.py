from pylab import *
from IPython.Shell import IPShellEmbed


def centro():
    print "first"
    side = 1024
    radius = 10.0

    x = arange(side) - side/2.0 + 0.5
    y = arange(side) - side/2.0 + 0.5
    X,Y = meshgrid(x,y)
    r = sqrt(X**2+Y**2)

    #particle = 1.0*(r < radius) + 1.0*(r < radius)*(X < 0.0)
    #particle = 1.0*(abs(X) < radius) * (abs(Y) < radius) + 1.0*(abs(X) < radius) * (abs(Y) < radius) * X
    #particle = 1.0*(abs(X) < radius) * (abs(Y) < radius) + 1.0*(abs(X-radius/2.0) < radius/4.0) * (abs(Y) < radius/4.0)
    #particle = 1.0*(abs(X) < radius) * (abs(Y) < radius) * random((side,side))
    #particle = exp(-(X**2+Y**2)/2.0/radius**2) + exp(-((X-20)**2+Y**2)/2.0/(radius/3.0)**2)

    particle_cent = zeros((side,side))
    rand = random((int(radius),int(radius)*2))
    particle_cent[side/2-int(radius):side/2,side/2-int(radius):side/2+int(radius)] = rand
    particle_cent[side/2:side/2+int(radius),side/2-int(radius):side/2+int(radius)] = rand[::-1,::-1]
    particle_non_cent = zeros((side,side))
    particle_non_cent[side/2-int(radius):side/2+int(radius),side/2-int(radius):side/2+int(radius)] = random((2*int(radius),2*int(radius)))

    print "second"
    ft_cent = fft2(fftshift(particle_cent))
    ft_non_cent = fft2(fftshift(particle_non_cent))

    fig = figure(1)
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    cutoff = 50
    cent_center = ft_cent[side/2-cutoff:side/2+cutoff,side/2-cutoff:side/2+cutoff].flatten()
    non_cent_center = ft_non_cent[side/2-cutoff:side/2+cutoff,side/2-cutoff:side/2+cutoff].flatten()
    ax1.plot(real(cent_center),imag(cent_center),'.')
    ax2.plot(real(non_cent_center),imag(non_cent_center),'.')
    show()

    print "hey"
    return locals()

    # ft = fft2(fftshift(particle))

    # fig = figure(1)
    # fig.clear()
    # ax1 = fig.add_subplot(121)
    # ax1.imshow(particle[side/2-radius*4:side/2+radius*4,side/2-radius*4:side/2+radius*4],interpolation='nearest')
    # ax2 = fig.add_subplot(122)
    # ax2.imshow((angle(fftshift(ft)))[512-100:512+100,512-100:512+100],interpolation='nearest')

    # show()


