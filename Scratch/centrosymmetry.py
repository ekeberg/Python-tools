from pylab import *

side = 256
radius = 10.0

x = arange(side) - side/2.0 + 0.5
y = arange(side) - side/2.0 + 0.5
X,Y = meshgrid(x,y)
r = sqrt(X**2+Y**2)

#particle = 1.0*(r < radius) + 1.0*(r < radius)*(X < 0.0)
#particle = 1.0*(abs(X) < radius) * (abs(Y) < radius) + 1.0*(abs(X) < radius) * (abs(Y) < radius) * X
particle = 1.0*(abs(X) < radius) * (abs(Y) < radius) + 1.0*(abs(X-radius/3.0) < radius/4.0) * (abs(Y) < radius/4.0)

ft = fft2(fftshift(particle))

fig = figure(1)
fig.clear()
ax1 = fig.add_subplot(121)
ax1.imshow(particle,interpolation='nearest')
ax2 = fig.add_subplot(122)
ax2.imshow(log(abs(fftshift(ft))),interpolation='nearest')

show()
