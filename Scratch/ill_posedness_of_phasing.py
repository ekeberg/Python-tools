from pylab import *


N = 1600
particle_size = 40
mask_size = 40

# def dft_1d(N):
#     i = arange(N); j = arange(N)
#     i_mesh, j_mesh = meshgrid(i,j)
#     o = exp(-2.0j*pi/N)
#     dft = matrix(o**(i_mesh*j_mesh)/sqrt(N))
#     return dft

# dft = dft_1d(N)
# support = zeros(N)
# support[:particle_size] = 1
# support_indices = support > 0
# support_indices_inverted = support == 0
# particle = transpose(matrix(zeros(N)))
# particle[:particle_size] = transpose(matrix(random(particle_size)))

# pattern = dft*particle
# amplitudes = abs(pattern)
# phases = angle(pattern)
# mask = zeros(N)
# mask[:mask_size] = 1
# mask_indices = mask > 0
# mask_indices_inverted = mask == 0

# print svd(dft[support_indices][:,mask_indices_inverted])[1]

# #dft_SMi = dft[support_indices][:,mask_indices_inverted]
# dft_SMi = dft[mask_indices_inverted][:,support_indices]

# # [row,column]
# #phase_linearization_matrix = matrix(zeros((N-particle_size,N-mask_size)),dtype='complex128')
# phase_linearization_matrix = matrix(zeros((N-mask_size,N-mask_size)),dtype='complex128')
# pattern_normalized = pattern/abs(pattern)
# for i in range((N-mask_size)/2):
#     phase_linearization_matrix[i,2*i] = 1.0/pattern_normalized[mask_size+2*i][0,0]
#     phase_linearization_matrix[i,2*i+1] = 1.0/pattern_normalized[mask_size+2*i+1][0,0]/1.0j
#     #phase_linearization_matrix[(N-mask_size)/2+i,2*i] = 1.0/(pattern_normalized[mask_size+2*i][0,0]*1.0j)
#     #phase_linearization_matrix[(N-mask_size)/2+i,2*i+1] = 1.0/(pattern_normalized[mask_size+2*i+1][0,0]*1.0j)/1.0j
#     phase_linearization_matrix[(N-mask_size)/2+i,2*i] = sqrt(1.0-phase_linearization_matrix[i,2*i]**2)

# tot_mat = (phase_linearization_matrix*dft_SMi)


# s = svd(tot_mat[:(N-mask_size)/2])

# print s[1]


def dft_1d_complex(N):
    """
    The dft matrix that is returnd works on complex vectors
    that are stored as real ones as [a_real,a_imag,b_real,b_imag,...]
    """
    o = exp(-2.0j*pi/N)
    i = arange(N); j = arange(N)
    i_mesh, j_mesh = meshgrid(i,j)
    dft = matrix(zeros((2*N,2*N)))
    for i in range(N):
        for j in range(N):
            coeff = o**(i*j)/sqrt(N)
            dft[2*i,2*j] = real(coeff)
            dft[2*i,2*j+1] = -imag(coeff)
            dft[2*i+1,2*j] = imag(coeff)
            dft[2*i+1,2*j+1] = real(coeff)
    return dft

dft = dft_1d_complex(N)

support = zeros(2*N)
support[:particle_size*2] = 1
support_indices = support > 0
support_indices_inverted = support == 0
particle = transpose(matrix(zeros(2*N)))
particle[:particle_size*2] = transpose(matrix(random(particle_size*2)))

pattern = dft*particle
#amplitudes = pattern
#phases = angle(pattern)
mask = zeros(2*N)
mask[:mask_size*2] = 1
mask_indices = mask > 0
mask_indices_inverted = mask == 0

s = svd(dft[support_indices][:,mask_indices_inverted])
print ", ".join(["%f" % f for f in reversed(sqrt(1.0-s[1][2*arange(len(s[1])/2)]**2))])

dft_SMi = dft[mask_indices_inverted][:,support_indices]

plz = matrix(zeros((2*(N-mask_size),2*(N-mask_size))))

pattern_unmasked = pattern[2*mask_size:]
for i in range((N-mask_size)/2):
    plz[4*i,2*i] = pattern_unmasked[4*i]/sqrt(pattern_unmasked[4*i]**2+pattern_unmasked[4*i+1]**2)
    plz[4*i+1,2*i] = pattern_unmasked[4*i+1]/sqrt(pattern_unmasked[4*i]**2+pattern_unmasked[4*i+1]**2)
    plz[4*i+2,2*i+1] = pattern_unmasked[4*i+2]/sqrt(pattern_unmasked[4*i+2]**2+pattern_unmasked[4*i+3]**2)
    plz[4*i+3,2*i+1] = pattern_unmasked[4*i+3]/sqrt(pattern_unmasked[4*i+2]**2+pattern_unmasked[4*i+3]**2)

    plz[4*i,(N-mask_size)+2*i] = -pattern_unmasked[4*i+1]/sqrt(pattern_unmasked[4*i]**2+pattern_unmasked[4*i+1]**2)
    plz[4*i+1,(N-mask_size)+2*i] = pattern_unmasked[4*i]/sqrt(pattern_unmasked[4*i]**2+pattern_unmasked[4*i+1]**2)
    plz[4*i+2,(N-mask_size)+2*i+1] = -pattern_unmasked[4*i+3]/sqrt(pattern_unmasked[4*i+2]**2+pattern_unmasked[4*i+3]**2)
    plz[4*i+3,(N-mask_size)+2*i+1] = pattern_unmasked[4*i+2]/sqrt(pattern_unmasked[4*i+2]**2+pattern_unmasked[4*i+3]**2)

tot_mat = transpose(plz)*dft_SMi

s = svd(tot_mat[:(N-mask_size)])
print ", ".join(["%f" % f for f in reversed(sqrt(1.0-s[1][2*arange(len(s[1])/2)]**2))])
