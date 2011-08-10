from pylab import *

#data generatin constants
image_size = 30
sample_size = 4
scaling = 4
number_of_images = 400

#diffusion map parameters
epsilon = 0.002
alpha = 1.0
number_of_eigenvectors = 5

sample = random((sample_size,sample_size))
sample[::2,:] = 0.0; sample[:,::2] = 0.0
sample /= sum((sample**2).flatten())

image_2d = fftn(sample,[image_size*scaling,image_size*scaling])/float(sample_size)

def get_2d_slice(image_2d,angle,scaling):
    length = shape(image_2d)[0]/scaling
    pixel = arange(length) - length/2.0 + 0.5
    x = scaling*pixel*cos(angle)
    y = scaling*pixel*sin(angle)
    result = image_2d[int32(round_(x)),int32(round_(y))]
    return result

s = abs(get_2d_slice(image_2d,1.0,2.0))

def get_dataset_1d_diffraction(number_of_samples, image_2d, scaling, angles=None):
    images = []
    if angles == None:
        angles = 2.0*pi*random(number_of_samples)
    for angle in angles:
        images.append(abs(get_2d_slice(image_2d, angle, scaling)))
    return images, angles

#data = [(sin(s_angle/2)+1).*cos(s_angle*5);(sin(s_angle/2)+1).*sin(s_angle*5);sin(s_angle)];

def get_dataset_simple(number_of_samples, angles=None):
    if angles == None:
        angles = 2.0*pi*random(number_of_samples)
    data = transpose(array([(sin(angles/2.0)+1.0)*cos(angles*5.0),(sin(angles/2.0)+1.0)*sin(angles*5.0),sin(angles)]))
    return data, angles

get_dataset = get_dataset_simple

images, angles = get_dataset(number_of_images, arange(0.0,2.0*pi,2.0*pi/number_of_images))
#images, angles = get_dataset(number_of_images, image_2d, scaling,arange(0.0,1.0*pi,1.0*pi/number_of_images))
#images, angles = get_dataset(number_of_images, image_2d, scaling)

#images = array(images)
#images = array(transpose([images[:,i]/sum(images[:,i]) for i in range(image_size)]))

def pairwise_distance(image1, image2):
    #return exp(-sum(((image1 - image2)**2).flatten())/epsilon)
    return exp(-norm(((image1 - image2)**2).flatten())/epsilon)

# euklidian_distance_matrix = zeros((number_of_images, number_of_images))
# for image1,image1_index in zip(images,range(number_of_images)):
#     for image2,image2_index in zip(images,range(number_of_images)):
#         euklidian_distance_matrix[image1_index,image2_index] = sum(((image1-image2)**2).flatten())
# euklidian_distance_matrix[euklidian_distance_matrix == 0.0] = inf

distance_matrix = zeros((number_of_images, number_of_images))
for image1,image1_index in zip(images,range(number_of_images)):
    for image2,image2_index in zip(images[:image1_index+1],range(number_of_images)[:image1_index+1]):
        dist = pairwise_distance(image1,image2)
        distance_matrix[image1_index,image2_index] = dist
        distance_matrix[image2_index,image1_index] = dist
    
def normalize_distance_matrix(distance_matrix, alpha):
    normalization_constants = sum(distance_matrix,axis=0)**alpha
    distance_matrix = transpose(transpose(distance_matrix) / normalization_constants)
    distance_matrix /= normalization_constants
    normalization_constants = sum(distance_matrix,axis=0)
    distance_matrix = distance_matrix / normalization_constants
    return distance_matrix

def normalize_distance_matrix(distance_matrix, alpha):
    q = sum(distance_matrix, axis=0)
    distance_matrix = matrix(diag(q**-alpha))*matrix(distance_matrix)*matrix(diag(q**-alpha))
    d = squeeze(array(sum(distance_matrix, axis=0)))
    distance_matrix = matrix(diag(1.0/d))*matrix(distance_matrix)
    return distance_matrix

P = normalize_distance_matrix(distance_matrix, alpha)
    
eigenvalues_unsorted, eigenvectors_unsorted = eigh(P)
eig_zip = zip(eigenvalues_unsorted,range(len(eigenvalues_unsorted)))
eig_zip.sort()
eig_zip.reverse()
eigenvalue_order = int32(array(eig_zip)[:,1])
eigenvalues = eigenvalues_unsorted[eigenvalue_order]
#eigenvectors = [eigenvectors_unsorted[i] for
eigenvectors = eigenvectors_unsorted[:,eigenvalue_order]
scaled_eigenvectors = eigenvectors*diag(eigenvalues)
#scaled_eigenvectors = diag(eigenvalues)*eigenvectors

low_d_space = zeros((number_of_images,number_of_eigenvectors))
for image_index in range(number_of_images):
    for v in range(number_of_eigenvectors):
        low_d_space[image_index, v] = scaled_eigenvectors[image_index,v+1]
