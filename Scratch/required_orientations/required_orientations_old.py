from pylab import *
from scipy.special import binom
import pickle


save_plots = False

# changes
# divide N and n by 2 to account for the fact that the two hemispheres will be identically covered.

D = 460. #nm
d = 83. #nm
d_array = arange(1., 200) #nm
N = 4.*pi*(D/d)**2/2.
n = 2.*pi*(D/d)/2.
number_of_images = 261
number_of_images_array = arange(1, 400)

average_coverage = pickle.load(open('average_coverage.p', 'rb'))

p_per_speckle = (1.-n/N)**number_of_images
print p_per_speckle

p = (1.-p_per_speckle)**D
print p

close('all')
fig1 = figure(1, figsize=(6,4), dpi=100)
fig1.subplots_adjust(bottom=0.13)
fig1.clear()
ax1 = fig1.add_subplot(111)
#d = d_array
number_of_images = number_of_images_array
#ax.plot(d_array, (1. - (1-(2.*pi*(D/d))/(8.*pi*(D/d)**2))**number_of_images)**N)
#single_hit_prob =  (1. - (1-(2.*pi*(D/d))/(8.*pi*(D/d)**2))**number_of_images)**N
single_hit_prob = (1. - (1.-n/N)**number_of_images)**N
ax1.plot(number_of_images_array, single_hit_prob, label='analytic')
ax1.plot(number_of_images_array[:len(average_coverage)], average_coverage, label='simulated')
single_hit_old = (1. - (1.-n/N)**number_of_images)**(N*2.)
ax1.plot(number_of_images_array, single_hit_old, label='old analytic')
ax1.legend()
ax1.plot([261, 261], [0, 1], '--', color='black', )

ax1.set_xlabel(r'Number of patterns')
ax1.set_ylabel(r'Probability of total coverage ($p$)')

if save_plots:
    fig1.savefig('/Users/ekeberg/Work/Random results/plot_image_count_vs_p.png', dpi=300)

def multi_hit_prob(max_hits):
    m = arange(max_hits)
    return (1. - sum(binom(number_of_images_array, m[:,newaxis]) * (n/N)**m[:,newaxis] * (1.-n/N)**(number_of_images_array-m[:,newaxis]), axis=0))**N

number_of_images_array = arange(1, 2000)

fig2 = figure(2, figsize=(6,2), dpi=100)
fig2.subplots_adjust(bottom=0.22)
fig2.subplots_adjust(left=0.14)
fig2.clear()
ax2 = fig2.add_subplot(111)

hits = arange(0, 101, 10)
hits[0] = 1
for h in hits:
    ax2.plot(number_of_images_array, multi_hit_prob(h), label='%d hits' % h)

#ax2.legend()
ax2.set_xlabel(r'Number of patterns')
ax2.set_ylabel(r'Probability of' '\n' r'total coverage')

if save_plots:
    fig2.savefig("/Users/ekeberg/Work/Random results/multi_hits.png", dpi=300)

fig3 = figure(3, figsize=(6,4), dpi=100)
fig3.subplots_adjust(bottom=0.13)
fig3.subplots_adjust(left=0.13)
fig3.clear()
ax3 = fig3.add_subplot(111)
p = 0.95
D = 460. #nm
d_inv = arange(1./460., 1./1., 1./460.)
N = 4.*pi*(D*d_inv)**2/2.
n = 2.*pi*(D*d_inv)/2.
images_required = log(1.-p**(1/N)) / log(1.-n/N)
ax3.plot(d_inv, images_required)
ax3.plot([1./83.,1./83.], [0., ax3.get_ylim()[1]], '--', color='black')

ax3.set_xlabel(r'Resolution [nm$^{-1}$]')
ax3.set_ylabel(r'Number of images')

if save_plots:
    fig3.savefig('/Users/ekeberg/Work/Random results/images_required.png', dpi=300)

show()


