#import matplotlib
#import sys
#sys.path.append("/Applications/Gimp.app/Contents/Resources/share/pygtk/2.0/codegen/")
#matplotlib.use("WxAgg")
from pylab import *
import random as python_random

ALGORITHM_ER, ALGORITHM_RAAR, ALGORITHM_HIO = range(3)

python_random.seed(4)

ion()

SAMPLE_SIDE = 50
PATTERN_SIDE = 500
POISSON_NOISE = 0.1
GAUSSIAN_NOISE = 0.1
MAX_ITERATIONS = 300
END_ER_ITERATIONS = 10
BETA = 0.9
ALGORITHM = ALGORITHM_HIO

class ReconstructionState(object):
    def __init__(self, algorithm = ALGORITHM_HIO, algorithm_parameters = {"beta" : 0.9}, amplitudes = None, support = None):
        self.algorithm = algorithm
        self.algorithm_parameters = algorithm_parameters #dict
        self.amplitudes = amplitudes
        self.support = support
        self.real_space = None
        self.fourier_space = None
    def is_ready(self):
        return (self.algorithm != None and self.algorithm_parameters != None and
                self.amplitudes != None and self.support != None)
    def support_inside(self):
        return self.support == 0
    def support_outside(self):
        return self.support > 0

def square_support(pattern_side, sample_side):
    """Returns a shifted square support"""
    support = zeros((pattern_side, pattern_side),dtype='int32')
    support[pattern_side/2-sample_side/2:pattern_side/2+sample_side/2,
            pattern_side/2-sample_side/2:pattern_side/2+sample_side/2] = 1
    return fftshift(support)

def lena(sample_side):
    original_image = imread('lenabw.jpg')
    bw_image = sum(lena, axis=2)[:507,:]
    bw_image /= average(bw_image)
    sample = abs(ifft2(fftshift(fft2(lenabw))[507/2-sample_side/2:507/2+sample_side/2,
                                              507/2-sample_side/2:507/2+sample_side/2]))

def get_pattern(sample, pattern_side):
    padded_sample = zeros((pattern_side,)*2, dtype='float64')
    sample_side_x, sample_side_y = shape(sample)
    padded_sample[pattern_side/2-sample_side_x/2:pattern_side/2+sample_side_x/2,
                  pattern_side/2-sample_side_y/2:pattern_side/2+sample_side_y/2] = sample
    ft = fft2(fftshift(padded_sample))
    pattern = abs(pattern)**2
    pattern /= average(pattern)
    return pattern

def noisy_pattern(pattern, average_gaussian_noise, average_poisson_noise):
    noisy_pattern = (poisson(pattern*average_poisson_noise**2/sum(sqrt(pattern))**2) +
                     normal(0., average_gaussian_noise, size=shape(pattern)))
    return noisy_pattern

def random_phases(amplitudes):
    return amplitudes*exp(-2.j*pi*random(shape(amplitudes)))


start_state = ReconstructionState()
start_state.support = square_support(PATTERN_SIDE, SAMPLE_SIDE)
start_state.amplitudes = sqrt(noisy_pattern(get_pattern(lena(SAMPLE_SIDE), PATTERN_SIDE), GAUSSIAN_NOISE, POISSON_NOISE))
start_state.fourier_space = random_phases(start_state.amplitudes)

close('all')
fig1 = figure(1,figsize=(8,12))
fig1.clear()
fig1_cols = 2; fig1_rows = 3
real_space_plot = fig1.add_subplot(fig1_rows, fig1_cols, 1)
sample_plot = fig1.add_subplot(fig1_rows, fig1_cols, 2)
fourier_space_plot = fig1.add_subplot(fig1_rows, fig1_cols, 3)
pattern_plot = fig1.add_subplot(fig1_rows, fig1_cols, 4)
fourier_phase_plot = fig1.add_subplot(fig1_rows, fig1_cols, 5)
fig2 = figure(2)
fig2.clear()
error_plot = fig2.add_subplot(1, 1, 1)

lena = imread('lenabw.jpg')
lenabw = sum(lena,axis=2)[:507,:]
sample = abs(ifft2(fftshift(fft2(lenabw))[507/2-SAMPLE_SIDE/2:507/2+SAMPLE_SIDE/2,507/2-SAMPLE_SIDE/2:507/2+SAMPLE_SIDE/2]))
#sample = random((SAMPLE_SIDE, SAMPLE_SIDE))
#sample = array([python_random.random() for i in range(SAMPLE_SIDE**2)]).reshape((SAMPLE_SIDE, SAMPLE_SIDE))
ft = fftn(sample, [PATTERN_SIDE, PATTERN_SIDE])
pattern = abs(ft)
pattern /= average(pattern)
#pattern = poisson(pattern*100.*average(pattern))
pattern += normal(0., .01, size=(PATTERN_SIDE, PATTERN_SIDE))
support_inside = support > 0
support_outside = support == 0
sample_plot.imshow(sample)
pattern_plot.imshow(log(fftshift(pattern)))
#fig1.show()
#fig2.show()
#show()
draw()

#set random phases
#fourier_space = pattern*exp(-2.j*pi*random((PATTERN_SIDE, PATTERN_SIDE)))
fourier_space = fft2(support)

def reconstruct(fourier_space, max_iterations, algorithm):
    err = []
    fourier_history = []
    real_history = []
    
    last_real = zeros((PATTERN_SIDE, PATTERN_SIDE))
    for iteration in range(max_iterations):
        if iteration == max_iterations - END_ER_ITERATIONS: algorithm = ALGORITHM_ER
        print "iteration " + str(iteration)
        #back fourer transform
        real_space = real(ifft2(fourier_space))
        real_space_plot.clear()
        real_space_plot.imshow(abs(fftshift(real_space)[PATTERN_SIDE/2-SAMPLE_SIDE:PATTERN_SIDE/2+SAMPLE_SIDE,
                                                        PATTERN_SIDE/2-SAMPLE_SIDE:PATTERN_SIDE/2+SAMPLE_SIDE]))
        #real-space constraint
        if algorithm == ALGORITHM_RAAR:
            good_pixels = support_inside & ((2.0*real_space - last_real) > 0.)
            bad_pixels = good_pixels == False
            real_space[good_pixels] = real_space[good_pixels]
            real_space[bad_pixels] = (BETA*last_real[bad_pixels] - (1.0 - 2.0*BETA)*real_space[bad_pixels])
        elif algorithm == ALGORITHM_HIO:
            good_pixels = support_inside & ((2.0*real_space - last_real) > 0.)
            bad_pixels = good_pixels == False
            real_space[good_pixels] = real_space[good_pixels]
            real_space[bad_pixels] = (last_real[bad_pixels] - BETA*real_space[bad_pixels])*1.#*(1.0 - 0.3*float(iteration)/float(max_iterations))
        elif algorithm == ALGORITHM_ER:
            real_space[support_outside] = 0.
        last_real = real_space.copy()
        #real_space[bad_pixels] = 0.0
        #fourier transform
        fourier_space = fft2(real_space)
        fourier_space_plot.clear()
        fourier_space_plot.imshow(log(abs(fftshift(fourier_space))))
        fourier_phase_plot.clear()
        fourier_phase_plot.imshow(angle(fftshift(fourier_space)))
        err.append(average((abs(fourier_space) - pattern)**2) / average(pattern**2))
        #err.append(average(abs(abs(fourier_space) - pattern) / abs(abs(fourier_space) + pattern)))
        #fourier constraint
        fourier_space = pattern*fourier_space/(abs(fourier_space)+0.0001)
        #fig1.show()
        #figure(1)
        fig1.canvas.draw()
        #figure(2)
        error_plot.clear()
        error_plot.plot(err)
        fig2.canvas.draw()
        #fig2.show()
        fourier_history.append(fourier_space)
        real_history.append(real_space) 

    return err, fourier_history, real_history

err, fourier_history, real_history = reconstruct(fourier_space, MAX_ITERATIONS, ALGORITHM)

error_plot.plot(err)
draw()
show()

def minima_map():
    p = pattern
    #u = real_space[support_inside]
    u = real_space

    ft_u = fft2(u)
    abs_ft_u = abs(ft_u)
    abs_operator = conj(ft_u/abs(ft_u))

    x = range(PATTERN_SIDE)
    y = range(PATTERN_SIDE)
    X,Y = meshgrid(x,y)

    support_pixel_num = sum(support)
    der_matrix = zeros((support_pixel_num, support_pixel_num),dtype='float64')
    for i1, (x1, y1) in enumerate(zip(X[support_inside], Y[support_inside])):
        print i1
        for i2, (x2, y2) in enumerate(zip(X[support_inside], Y[support_inside])):
            #for c in (1., 1.j):
            c = 1.
            u_i_1 = zeros((PATTERN_SIDE, PATTERN_SIDE), dtype='complex128')
            u_i_1[x1,y1] = c
            ft_u_i_1 = fft2(u_i_1)
            u_i_2 = zeros((PATTERN_SIDE, PATTERN_SIDE), dtype='complex128')
            u_i_2[x2,y2] = c
            ft_u_i_2 = fft2(u_i_2)
            der_matrix[i1, i2] += c*sum((real(abs_operator*ft_u_i_2) - p)*real(abs_operator*ft_u_i_1))


    eigen_values, eigen_vectors = linalg.eig(der_matrix)

    eig_values_abs = abs(eigen_values)
    eig_values_abs_sorted = sort(eig_values_abs)
    index1 = eig_values_abs.tolist().index(eig_values_abs_sorted[0])
    index2 = eig_values_abs.tolist().index(eig_values_abs_sorted[-1])

    # der = diag(der_matrix)
    # abs_d = abs(array(der))
    # sort_abs_d = sort(abs_d)
    # index1 = abs_d.tolist().index(sort_abs_d[0])
    # index2 = abs_d.tolist().index(sort_abs_d[-11])

    x1 = X[support_inside][index1/2]
    y1 = Y[support_inside][index1/2]
    x2 = X[support_inside][index2/2]
    y2 = Y[support_inside][index2/2]

    # i1_array = arange(-0.5, 0.5, 0.01, dtype='complex128')
    # i2_array = arange(-0.5, 0.5, 0.01, dtype='complex128')
    i1_array = arange(-0.05, 0.05, 0.001, dtype='float64')*100.
    i2_array = arange(-0.05, 0.05, 0.001, dtype='float64')*100.
    # i1_c = index1%2
    # i2_c = index2%2
    # if i1_c: i1_array *= 1.j
    # if i2_c: i2_array *= 1.j

    error_map = zeros((len(i1_array), len(i2_array)))
    perturbed_u = u.copy()

    # perturbation_1 = random(100)+random(100)*1.j
    # perturbation_2 = random(100)+random(100)*1.j
    perturbation_1 = eigen_vectors[:,index1]
    perturbation_2 = eigen_vectors[:,index2]

    for i1_i, i1 in enumerate(i1_array):
        for i2_i, i2 in enumerate(i2_array):
            #perturbed_u[x1,y1] = u[x1,y1] + i1
            #perturbed_u[x2,y2] = u[x2,y2] + i2
            perturbed_u[support_inside] = u[support_inside] + perturbation_1*i1 + perturbation_2*i2
            error_map[i1_i,i2_i] = average((abs(fft2(perturbed_u)) - pattern)**2) / average(pattern**2)

    imshow(error_map)
    colorbar()
    show()
            

def map_1st_derivative():
    der = []
    for x,y in zip(X[support_inside], Y[support_inside]):
        for c in (1., 1.j):
            u_i = zeros((PATTERN_SIDE, PATTERN_SIDE),dtype='complex128')
            u_i[x,y] = c
            ft_u_i = fft2(u_i)
            abs_operator = conj(ft_u/abs_ft_u)
            der.append(sum((abs_ft_u - p) * real(abs_operator*ft_u_i)) / sum(p**2))

    abs_d = abs(array(der))
    sort_abs_d = sort(abs_d)
    index1 = abs_d.tolist().index(sort_abs_d[0])
    index2 = abs_d.tolist().index(sort_abs_d[1])

    x1 = X[support_inside][index1/2]
    y1 = Y[support_inside][index1/2]
    x2 = X[support_inside][index2/2]
    y2 = Y[support_inside][index2/2]

    i1_array = arange(-0.5, 0.5, 0.01, dtype='complex128')
    i2_array = arange(-0.5, 0.5, 0.01, dtype='complex128')
    i1_c = index1%2
    i2_c = index2%2
    if i1_c: i1_array *= 1.j
    if i2_c: i2_array *= 1.j

    error_map = zeros((len(i1_array), len(i2_array)))
    perturbed_u = u.copy()
    for i1_i, i1 in enumerate(i1_array):
        for i2_i, i2 in enumerate(i2_array):
            perturbed_u[x1,y1] = u[x1,y1] + i1
            perturbed_u[x2,y2] = u[x2,y2] + i2
            error_map[i1_i,i2_i] = average((abs(fft2(perturbed_u)) - pattern)**2) / average(pattern**2)

    imshow(error_map)
    show()

