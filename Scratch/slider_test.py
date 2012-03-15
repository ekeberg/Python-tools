from pylab import *
from gui import manipulate, manipulate_multi

# matplotlib must run in interactive mode
ion()

plot_range = arange(0.,10,0.01)
sigma_range = (0.02,2.)
center_range = (0.,10.)

fig = figure(1)
fig.clear()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

def plot_gaussian(sigma, center):
    dist = exp(-(plot_range-center)**2/sigma**2)
    cum = cumsum(dist[::-1])[::-1]
    ax1.clear()
    ax1.plot(plot_range, dist)
    ax2.clear()
    ax2.plot(plot_range, cum/cum[0])
    draw()


manipulate_multi(plot_gaussian, (sigma_range, center_range), ('Sigma', 'Center'))
