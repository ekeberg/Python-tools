from pylab import *
import random as python_random

python_random.seed(4)

SAMPLE_SIDE = 10
PATTERN_SIDE = 100
MAX_ITERATIONS = 10

fig1 = figure(1)
fig1.clear()
real_space_plot = fig1.add_subplot(2, 2, 1)
sample_plot = fig1.add_subplot(2, 2, 2)
fourier_space_plot = fig1.add_subplot(2, 2, 3)
pattern_plot = fig1.add_subplot(2, 2, 4)
fig2 = figure(2)
fig2.clear()
error_plot = fig2.add_subplot(1, 1, 1)

#sample = random((SAMPLE_SIDE, SAMPLE_SIDE))
sample = array([python_random.random() for i in range(SAMPLE_SIDE**2)]).reshape((SAMPLE_SIDE, SAMPLE_SIDE))
ft = fftn(sample, [PATTERN_SIDE, PATTERN_SIDE])
pattern = abs(ft)
support = zeros((PATTERN_SIDE, PATTERN_SIDE),dtype='int32')
support[PATTERN_SIDE/2-SAMPLE_SIDE/2:PATTERN_SIDE/2+SAMPLE_SIDE/2,
        PATTERN_SIDE/2-SAMPLE_SIDE/2:PATTERN_SIDE/2+SAMPLE_SIDE/2] = 1
support_inside = support > 0
support_outside = support == 0
sample_plot.imshow(sample)
pattern_plot.imshow(fftshift(pattern))
draw()

err = []

#set random phases
fourier_space = pattern*exp(-2.j*pi*random((PATTERN_SIDE, PATTERN_SIDE)))
for iteration in range(MAX_ITERATIONS):
    print "iteration " + str(iteration)
    #back fourer transform
    real_space = ifft2(fourier_space)
    real_space_plot.imshow(abs(real_space))
    #real-space constraint
    real_space[support_outside] = 0.
    #fourier transform
    fourier_space = fft2(real_space)
    fourier_space_plot.imshow(abs(fftshift(fourier_space)))
    err.append(average((abs(fourier_space) - pattern)**2) / average(pattern**2))
    #err.append(average(abs(abs(fourier_space) - pattern) / abs(abs(fourier_space) + pattern)))
    #fourier constraint
    fourier_space = pattern*fourier_space/abs(fourier_space)
    draw()

error_plot.plot(err)

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
der_matrix = zeros((support_pixel_num, support_pixel_num))
for i1, (x1, y1) in enumerate(zip(X[support_inside], Y[support_inside])):
    print i1
    for i2, (x2, y2) in enumerate(zip(X[support_inside], Y[support_inside])):
        for c in (1., 1.j):
            u_i_1 = zeros((PATTERN_SIDE, PATTERN_SIDE), dtype='complex128')
            u_i_1[x1,y1] = c
            ft_u_i_1 = fft2(u_i_1)
            u_i_2 = zeros((PATTERN_SIDE, PATTERN_SIDE), dtype='complex128')
            u_i_2[x2,y2] = c
            ft_u_i_2 = fft2(u_i_2)
            der_matrix[i1, i2] = sum((real(abs_operator*ft_u_i_2) - p)*real(abs_operator*ft_u_i_1))


eigen_values, eigen_vectors = linalg.eig(der_matrix)

eig_values_abs = abs(eigen_values)
eig_values_abs_sorted = sort(eig_values_abs)
index1 = eig_values_abs.tolist().index(eig_values_abs_sorted[0])

der = diag(der_matrix)
abs_d = abs(array(der))
sort_abs_d = sort(abs_d)
index1 = abs_d.tolist().index(sort_abs_d[0])
index2 = abs_d.tolist().index(sort_abs_d[-11])

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

perturbation_1 = random(100)+random(100)*1.j
perturbation_2 = random(100)+random(100)*1.j

for i1_i, i1 in enumerate(i1_array):
    for i2_i, i2 in enumerate(i2_array):
        #perturbed_u[x1,y1] = u[x1,y1] + i1
        #perturbed_u[x2,y2] = u[x2,y2] + i2
        perturbed_u[support_inside] = u[support_inside] + perturbation_1*i1
        perturbed_u[support_inside] = u[support_inside] + perturbation_2*i2
        error_map[i1_i,i2_i] = average((abs(fft2(perturbed_u)) - pattern)**2) / average(pattern**2)

imshow(error_map)
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

