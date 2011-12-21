#import matplotlib
import sys
#sys.path.append("/Applications/Gimp.app/Contents/Resources/share/pygtk/2.0/codegen/")
#matplotlib.use("WxAgg")
from pylab import *
import random as python_random
import function_probe

ALGORITHM_ER, ALGORITHM_RAAR, ALGORITHM_HIO, ALGORITHM_ANDREW, ALGORITHM_DM = range(5)

python_random.seed(4)

ion()

SAMPLE_SIDE = 20
PATTERN_SIDE = 80
POISSON_NOISE = 0.0
GAUSSIAN_NOISE = 0.0
BEAMSTOP_RADIUS = 0.0
MAX_ITERATIONS = 300
END_ER_ITERATIONS = 0
BETA = 1.0
ALGORITHM = ALGORITHM_ER

class ReconstructionState(object):
    def __init__(self, algorithm = ALGORITHM_HIO, algorithm_parameters = {"beta" : 0.9}, amplitudes = None, support = None):
        self.algorithm = algorithm
        self.algorithm_parameters = algorithm_parameters #dict
        self.amplitudes = amplitudes
        self.support = support
        self.real_space = None
        self.fourier_space = None
        self.mask = None
    def is_ready(self):
        return (self.algorithm != None and self.algorithm_parameters != None and
                self.amplitudes != None and self.support != None and self.fourier_space != None)
    def support_inside(self):
        return self.support > 0
    def support_outside(self):
        return self.support == 0
    def mask_inside(self):
        if self.mask != None:
            return self.mask > 0
        else:
            return zeros(shape(self.amplitudes), dtype='bool')
    def mask_outside(self):
        if self.mask != None:
            return self.mask == 0
        else:
            return ones(shape(self.amplitudes), dtype='bool')
    def project_real(self, real_in):
        #ret = real_in.copy()
        #ret = real(real_in).copy()
        ret = zeros(shape(real_in))
        pixels_to_copy = self.support_inside() & (real(real_in) > 0.)
        ret[pixels_to_copy] = real(real_in[pixels_to_copy])
        #ret[self.support_outside()] = 0.
        return ret
    def project_fourier(self, real_in):
        ft = fft2(real_in)
        ft[self.mask_outside()] *= self.amplitudes[self.mask_outside()]/abs(ft[self.mask_outside()])
        ret = ifft2(ft)
        return ret


def square_support(pattern_side, sample_side):
    """Returns a shifted square support"""
    support = zeros((pattern_side, pattern_side),dtype='int32')
    support[pattern_side/2-sample_side/2:pattern_side/2+sample_side/2,
            pattern_side/2-sample_side/2:pattern_side/2+sample_side/2] = 1
    return fftshift(support)

def round_mask(pattern_side, hole_radius):
    mask = zeros((pattern_side,)*2, dtype='int32')
    x = arange(pattern_side) - pattern_side/2.0# + 0.5
    y = arange(pattern_side) - pattern_side/2.0# + 0.5
    X, Y = meshgrid(x, y)
    mask[X**2 + Y**2 < hole_radius**2] = 1
    return fftshift(mask)

def lena(sample_side, scale = 1.):
    original_image = imread('lenabw.jpg')
    # bw_image = float64(sum(original_image, axis=2)[:507,:])
    # bw_image /= average(bw_image)
    # sample = abs(ifft2(fftshift(fft2(bw_image))[507/2-sample_side/2:507/2+sample_side/2,
    #                                             507/2-sample_side/2:507/2+sample_side/2]))

    if scale > 1. or scale <= 0.:
        raise ValueError("Scale parameter is not in (0,1]")
    original_side = 507
    new_side = 2*int32(original_side*scale/2.0) #only even numbers
    bw_image = float64(sum(original_image, axis=2)[:507,:])
    bw_image_cropped = bw_image[original_side/2-new_side/2:original_side/2+new_side/2,
                                original_side/2-new_side/2:original_side/2+new_side/2]
    sample = abs(ifft2(fftshift(fft2(bw_image_cropped))[new_side/2-sample_side/2:new_side/2+sample_side/2,
                                                new_side/2-sample_side/2:new_side/2+sample_side/2]))
    sample /= average(sample)
    return sample
    #return ones((sample_side,)*2)

def get_pattern(sample, pattern_side):
    padded_sample = zeros((pattern_side,)*2, dtype='float64')
    sample_side_x, sample_side_y = shape(sample)
    padded_sample[pattern_side/2-sample_side_x/2:pattern_side/2+sample_side_x/2,
                  pattern_side/2-sample_side_y/2:pattern_side/2+sample_side_y/2] = sample
    ft = fft2(fftshift(padded_sample))
    pattern = abs(ft)**2
    pattern /= average(pattern)
    return pattern

def noisy_pattern(pattern, average_gaussian_noise, average_poisson_noise):
    try:
        poisson_pattern = poisson(pattern*(sum(sqrt(pattern))**2/sum(pattern)**2/average_poisson_noise**2))
    except ValueError:
        poisson_pattern = pattern
    if average_gaussian_noise > 0.:
        noisy_pattern = poisson_pattern + normal(0., average_gaussian_noise, size=shape(pattern))
    else:
        noisy_pattern = poisson_pattern
    noisy_pattern[noisy_pattern < 0.] = 0.
    return noisy_pattern

def random_phases(amplitudes):
    return amplitudes*exp(-2.j*pi*random(shape(amplitudes)))


close('all')
fig1 = figure(1,figsize=(8,12))
fig1.clear()
fig1_cols = 2; fig1_rows = 3
real_space_plot = fig1.add_subplot(fig1_rows, fig1_cols, 1)
sample_plot = fig1.add_subplot(fig1_rows, fig1_cols, 2)
fourier_space_plot = fig1.add_subplot(fig1_rows, fig1_cols, 3)
pattern_plot = fig1.add_subplot(fig1_rows, fig1_cols, 4)
fourier_phase_plot = fig1.add_subplot(fig1_rows, fig1_cols, 5)
andrew_plot = fig1.add_subplot(fig1_rows, fig1_cols, 6)

fig2 = figure(2)
fig2.clear()
error_plot = fig2.add_subplot(1, 1, 1)

#set random phases
#fourier_space = pattern*exp(-2.j*pi*random((PATTERN_SIDE, PATTERN_SIDE)))
#fourier_space = fft2(support)

def reconstruct(start_state, max_iterations):
    err = []
    fourier_history = []
    real_history = []

    pattern_plot.imshow(fftshift(start_state.amplitudes))
    state = start_state

    if state.real_space != None:
        last_real = state.real_space
    else:
        last_real = zeros(shape(state.support))
            
    for iteration in range(max_iterations):
        if iteration == max_iterations - END_ER_ITERATIONS: state.algorithm = ALGORITHM_ER
        print "iteration " + str(iteration)
        #back fourer transform
        state.real_space = real(ifft2(state.fourier_space))
        real_space_plot.clear()
        real_space_plot.imshow(abs(fftshift(state.real_space)[PATTERN_SIDE/2-SAMPLE_SIDE:PATTERN_SIDE/2+SAMPLE_SIDE,
                                                              PATTERN_SIDE/2-SAMPLE_SIDE:PATTERN_SIDE/2+SAMPLE_SIDE]))
        #real-space constraint
        if state.algorithm == ALGORITHM_RAAR:
            good_pixels = state.support_inside() & ((2.0*state.real_space - last_real) > 0.)
            bad_pixels = good_pixels == False
            state.real_space[good_pixels] = state.real_space[good_pixels]
            state.real_space[bad_pixels] = (state.algorithm_parameters['beta']*last_real[bad_pixels] - (1.0 - 2.0*state.algorithm_parameters['beta'])*state.real_space[bad_pixels])
        elif state.algorithm == ALGORITHM_HIO:
            #good_pixels = state.support_inside() & ((2.0*state.real_space - last_real) > 0.)
            good_pixels = state.support_inside() & (state.real_space > 0.)
            bad_pixels = good_pixels == False
            state.real_space[good_pixels] = state.real_space[good_pixels]
            state.real_space[bad_pixels] = (last_real[bad_pixels] - state.algorithm_parameters['beta']*state.real_space[bad_pixels])*1.0#*(1.0 - 0.3*float(iteration)/float(max_iterations))
        elif state.algorithm == ALGORITHM_ER:
            state.real_space[state.support_outside()] = 0.
        elif state.algorithm == ALGORITHM_ANDREW:
            #good_pixels = state.support_inside() & ((2.0*state.real_space - last_real) > 0.)
            good_pixels = state.support_inside() & (state.real_space > 0.)
            bad_pixels = good_pixels == False
            hio_pixels = bad_pixels & (abs(state.real_space) > state.algorithm_parameters['threshold'])
            er_pixels = bad_pixels & (abs(state.real_space) <= state.algorithm_parameters['threshold'])
            state.real_space[good_pixels] = state.real_space[good_pixels]
            state.real_space[hio_pixels] = (last_real[hio_pixels] -
                                            state.algorithm_parameters['beta']*state.real_space[hio_pixels])
            state.real_space[er_pixels] = 0.
            andrew_plot.imshow(fftshift(hio_pixels))
            
        last_real = state.real_space.copy()
        #fourier transform
        state.fourier_space = fft2(state.real_space)
        fourier_space_plot.clear()
        fourier_space_plot.imshow(log(abs(fftshift(state.fourier_space))))
        fourier_phase_plot.clear()
        fourier_phase_plot.imshow(angle(fftshift(state.fourier_space)))
        err.append(average((abs(state.fourier_space[state.mask_outside()]) - state.amplitudes[state.mask_outside()])**2) / average(state.amplitudes[state.mask_outside()]**2))
        #err.append(average(abs(abs(fourier_space) - pattern) / abs(abs(fourier_space) + pattern)))
        #fourier constraint
        #state.fourier_space = state.amplitudes*state.fourier_space/(abs(state.fourier_space)+0.00001)
        state.fourier_space[state.mask_outside()] = (state.amplitudes[state.mask_outside()]*
                                                     state.fourier_space[state.mask_outside()]/
                                                     (abs(state.fourier_space[state.mask_outside()])+0.00001))
        fig1.canvas.draw()
        error_plot.clear()
        error_plot.plot(err)
        fig2.canvas.draw()
        if iteration % 20 == 0:
            fourier_history.append(state.fourier_space)
            real_history.append(state.real_space) 

    return err, fourier_history, real_history

def reconstruction_iteration_dm(state):
    # state.real_space += (state.project_real(state.real_space) -
    #                      state.project_fourier(state.fourier_space) +
    #                      (state.algorithm_parameters['beta'] - 1.) * state.project_real(state.project_fourier(state.real_space)) -
    #                      (state.algorithm_parameters['beta'] - 1.) * state.project_fourier(state.project_real(state.real_space)))
    b = state.algorithm_parameters['beta']
    g = state.algorithm_parameters['gamma']
    # difference map
    state.real_space += (g/b*state.project_real(state.real_space) -
                         g/b*state.project_fourier(state.fourier_space) +
                         (g - g/b) * state.project_real(state.project_fourier(state.real_space)) -
                         (g - g/b) * state.project_fourier(state.project_real(state.real_space)))
    # error reduction
    #state.real_space = state.project_fourier(state.project_real(state.real_space))
    #state.real_space = state.project_real(state.project_fourier(state.real_space))
    # hio
    #state.real_space += state.project_fourier(2.*state.project_real(state.real_space) - state.real_space) - state.project_real(state.real_space)
    #state.real_space += state.project_real(2.*state.project_fourier(state.real_space) - state.real_space) - state.project_fourier(state.real_space)
    # I = state.real_space
    # Pr = state.real_space.copy(); Pr[state.support_outside()] = 0.
    # Pf = fft2(state.real_space); Pf[state.mask_outside()] *= state.amplitudes[state.mask_outside()]/abs(Pf[state.mask_outside()]); Pf = ifft2(Pf)
    # PrPf = Pf.copy(); Pr[state.support_outside()] = 0.
    # PfPr = fft2(Pr); PfPr[state.mask_outside()] *= state.amplitudes[state.mask_outside()]/abs(PfPr[state.mask_outside()]); PfPr = ifft2(Pf)
    # state.real_space =  I + Pr - Pf + (state.algorithm_parameters['beta'] - 1.)*PrPf - (state.algorithm_parameters['beta'] - 1.)*PfPr

def reconstruct_dm(state, max_iterations):
    err =[]
    fourier_history = []
    real_history = []
    pattern_plot.imshow(log(fftshift(state.amplitudes)))
    last_real = zeros(shape(state.support))
    state.real_space = ifft2(state.fourier_space)

    for iteration in range(max_iterations):
        print iteration
        reconstruction_iteration_dm(state)
        real_space_plot.clear()
        real_space_plot.imshow(abs(fftshift(state.real_space)[PATTERN_SIDE/2-SAMPLE_SIDE:PATTERN_SIDE/2+SAMPLE_SIDE,
                                                              PATTERN_SIDE/2-SAMPLE_SIDE:PATTERN_SIDE/2+SAMPLE_SIDE]))
        fourier_space_plot.clear()
        fourier_space_plot.imshow(log(abs(fftshift(fft2(state.real_space)))))
        real_history.append(state.real_space.copy())
        fig1.canvas.draw()

    return err, fourier_history, real_history
        

def reconstruct_noisy(start_state, max_iterations):
    err = []
    fourier_history = []
    real_history = []

    pattern_plot.imshow(log(fftshift(start_state.amplitudes)))
    state = start_state

    if state.real_space != None:
        last_real = state.real_space
    else:
        last_real = zeros(shape(state.support))
            
    for iteration in range(max_iterations):
        if iteration == max_iterations - END_ER_ITERATIONS: state.algorithm = ALGORITHM_ER
        print "iteration " + str(iteration)
        #back fourer transform
        state.real_space = real(ifft2(state.fourier_space))
        real_space_plot.clear()
        real_space_plot.imshow(abs(fftshift(state.real_space)[PATTERN_SIDE/2-SAMPLE_SIDE:PATTERN_SIDE/2+SAMPLE_SIDE,
                                                              PATTERN_SIDE/2-SAMPLE_SIDE:PATTERN_SIDE/2+SAMPLE_SIDE]))
        real_outside = state.real_space.copy()
        real_outside[state.support_inside()] = 0.
        fourier_outside = fft2(real_outside)
        
        #real-space constraint
        if state.algorithm == ALGORITHM_RAAR:
            good_pixels = state.support_inside() & ((2.0*state.real_space - last_real) > 0.)
            bad_pixels = good_pixels == False
            state.real_space[good_pixels] = state.real_space[good_pixels]
            state.real_space[bad_pixels] = (state.algorithm_parameters['beta']*last_real[bad_pixels] - (1.0 - 2.0*state.algorithm_parameters['beta'])*state.real_space[bad_pixels])
        elif state.algorithm == ALGORITHM_HIO:
            #good_pixels = state.support_inside() & ((2.0*state.real_space - last_real) > 0.)
            good_pixels = state.support_inside() & (state.real_space > 0.)
            bad_pixels = good_pixels == False
            state.real_space[good_pixels] = state.real_space[good_pixels]
            state.real_space[bad_pixels] = (last_real[bad_pixels] - state.algorithm_parameters['beta']*state.real_space[bad_pixels])*1.0#*(1.0 - 0.3*float(iteration)/float(max_iterations))
        elif state.algorithm == ALGORITHM_ER:
            state.real_space[state.support_outside()] = 0.
        elif state.algorithm == ALGORITHM_ANDREW:
            #good_pixels = state.support_inside() & ((2.0*state.real_space - last_real) > 0.)
            good_pixels = state.support_inside() & (state.real_space > 0.)
            bad_pixels = good_pixels == False
            hio_pixels = bad_pixels & (abs(state.real_space) > state.algorithm_parameters['threshold'])
            er_pixels = bad_pixels & (abs(state.real_space) <= state.algorithm_parameters['threshold'])
            state.real_space[good_pixels] = state.real_space[good_pixels]
            state.real_space[hio_pixels] = (last_real[hio_pixels] -
                                            state.algorithm_parameters['beta']*state.real_space[hio_pixels])
            state.real_space[er_pixels] = 0.
            andrew_plot.imshow(fftshift(hio_pixels))

        last_real = state.real_space.copy()
        #fourier transform
        state.fourier_space = fft2(state.real_space)

        #fourier_outside *= minimum(state.amplitudes*state.algorithm_parameters['threshold'], abs(fourier_outside))/abs(fourier_outside)
        fourier_outside *= minimum(average(state.amplitudes)*state.algorithm_parameters['threshold'], abs(fourier_outside))/abs(fourier_outside)

        err.append(average((abs((state.fourier_space)[state.mask_outside()]) - state.amplitudes[state.mask_outside()])**2) / average(state.amplitudes[state.mask_outside()]**2))

        state.fourier_space[state.mask_outside()] = (state.amplitudes[state.mask_outside()]*
                                                     (state.fourier_space[state.mask_outside()])/
                                                     (abs(state.fourier_space[state.mask_outside()]) + 0.0000)-
                                                     fourier_outside[state.mask_outside()])

        #state.fourier_space = state.amplitudes*(fourier_inside+fourier_outside)/(abs(fourier_inside+fourier_outside)) - fourier_outside

        andrew_plot.clear()
        andrew_plot.imshow(abs(fftshift(fourier_outside)))
        
        #state.fourier_space = fft2(state.real_space)
        fourier_space_plot.clear()
        fourier_space_plot.imshow(log(abs(fftshift(state.fourier_space))))
        fourier_phase_plot.clear()
        fourier_phase_plot.imshow(angle(fftshift(state.fourier_space)))

        #err.append(average(abs(abs(fourier_space) - pattern) / abs(abs(fourier_space) + pattern)))
        #fourier constraint
        #state.fourier_space = state.amplitudes*state.fourier_space/(abs(state.fourier_space)+0.00001)
        
        # state.fourier_space[state.mask_outside()] = (state.amplitudes[state.mask_outside()]*
        #                                              state.fourier_space[state.mask_outside()]/
        #                                              (abs(state.fourier_space[state.mask_outside()])+0.00001))
        fig1.canvas.draw()
        error_plot.clear()
        error_plot.plot(err)
        fig2.canvas.draw()
        if iteration % 20 == 0:
            fourier_history.append(state.fourier_space)
            real_history.append(state.real_space) 

    return err, fourier_history, real_history


def minima_map(pattern, real_space, support, mask, var_cont):
    p = pattern
    #u = real_space[support_inside]
    u = real_space

    ft_u = fft2(u)
    abs_ft_u = abs(ft_u)
    abs_operator = conj(ft_u/abs(ft_u))

    x = range(shape(pattern)[0])
    y = range(shape(pattern)[1])
    X,Y = meshgrid(x,y)

    support_pixel_count = sum(support)
    der_matrix = zeros((support_pixel_count,)*2,dtype='float64')
    for i1, (x1, y1) in enumerate(zip(X[support > 0], Y[support > 0])):
        print i1
        for i2, (x2, y2) in enumerate(zip(X[support > 0], Y[support > 0])):
            #for c in (1., 1.j):
            c = 1.
            u_i_1 = zeros((PATTERN_SIDE,)*2, dtype='complex128')
            u_i_1[x1,y1] = c
            ft_u_i_1 = fft2(u_i_1)
            u_i_2 = zeros((PATTERN_SIDE,)*2, dtype='complex128')
            u_i_2[x2,y2] = c
            ft_u_i_2 = fft2(u_i_2)
            der_matrix[i1, i2] += c*sum((real(abs_operator[mask]*ft_u_i_2[mask]) - p[mask])*real(abs_operator[mask]*ft_u_i_1[mask]))


    eigen_values, eigen_vectors = linalg.eig(der_matrix)

    eig_values_abs = abs(eigen_values)
    eig_values_abs_sorted = sort(eig_values_abs)
    index1 = eig_values_abs.tolist().index(eig_values_abs_sorted[0])
    index2 = eig_values_abs.tolist().index(eig_values_abs_sorted[1])

    # der = diag(der_matrix)
    # abs_d = abs(array(der))
    # sort_abs_d = sort(abs_d)
    # index1 = abs_d.tolist().index(sort_abs_d[0])
    # index2 = abs_d.tolist().index(sort_abs_d[-11])

    x1 = X[support > 0][index1/2]
    y1 = Y[support > 0][index1/2]
    x2 = X[support > 0][index2/2]
    y2 = Y[support > 0][index2/2]

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
            perturbed_u[support > 0] = u[support > 0] + perturbation_1*i1 + perturbation_2*i2
            error_map[i1_i,i2_i] = average((abs(fft2(perturbed_u))[mask] - pattern[mask])**2) / average(pattern[mask]**2)

    exec var_cont._get_code("var_cont")
    return error_map

func_probe = function_probe.container()
#imshow(error_map)
#colorbar()
#show()

sample = lena(SAMPLE_SIDE, 0.25)+1.
sample_padded = zeros((SAMPLE_SIDE*2,)*2)
sample_padded[SAMPLE_SIDE/2:SAMPLE_SIDE/2*3,SAMPLE_SIDE/2:SAMPLE_SIDE/2*3] = sample
sample_plot.imshow(sample_padded)

start_state = ReconstructionState()
start_state.support = square_support(PATTERN_SIDE, SAMPLE_SIDE)
start_state.amplitudes = sqrt(noisy_pattern(get_pattern(sample, PATTERN_SIDE), GAUSSIAN_NOISE, POISSON_NOISE))
start_state.fourier_space = random_phases(start_state.amplitudes)
start_state.mask = round_mask(PATTERN_SIDE, BEAMSTOP_RADIUS)
start_state.algorithm = ALGORITHM
start_state.algorithm_parameters['threshold'] = 0.007
start_state.algorithm_parameters['beta'] = BETA
start_state.algorithm_parameters['gamma'] = 1.

start_state.amplitudes[start_state.mask_inside()] = 0.

err, fourier_history, real_history = reconstruct_dm(start_state, MAX_ITERATIONS)
#err, fourier_history, real_history = reconstruct(start_state, MAX_ITERATIONS)

#error_map = minima_map(start_state.amplitudes, real_history[-1], start_state.support, start_state.mask_outside(), func_probe)

