from pylab import *
import rotations
import pickle

# model[x_rotated, y_rotated, z_rotated] = 1.
D = 460. #nm
d = 83. #nm
Dd = D/d
Dd = 40/2
d_array = arange(1., 200) #nm
N = 4.*pi*(Dd)**2
n = 2.*pi*(Dd)
number_of_images = 261
number_of_images_array = arange(1, 400)


p_per_speckle = (1.-n/N)**number_of_images
print p_per_speckle

p = (1.-p_per_speckle)**D
print p

close('all')
fig = figure(1, figsize=(6,4), dpi=100)
fig.subplots_adjust(bottom=0.13)
fig.clear()
ax = fig.add_subplot(111)
#d = d_array
number_of_images = number_of_images_array
#ax.plot(d_array, (1. - (1-(2.*pi*(D/d))/(8.*pi*(D/d)**2))**number_of_images)**N)
ax.plot(number_of_images_array, (1. - (1-(2.*pi*(Dd))/(8.*pi*(Dd)**2))**number_of_images)**N)
ax.plot([261, 261], [0, 1], '--', color='black', )

ax.set_xlabel(r'Number of patterns')
ax.set_ylabel(r'Probability of total coverage ($p$)')


class SliceInserter(object):
    def __init__(self, model_side, image_side):
        self.model_side = model_side
        self.image_side = image_side

        x_range = arange(-image_side/2+0.5, image_side/2+0.5)
        y_range = arange(-image_side/2+0.5, image_side/2+0.5)
        self.x_coordinates, self.y_coordinates = meshgrid(x_range, y_range)

        
        model_x = arange(-model_side/2+0.5, model_side/2+0.5)
        model_y = arange(-model_side/2+0.5, model_side/2+0.5)
        model_z = arange(-model_side/2+0.5, model_side/2+0.5)
        self.mask = model_x**2 + model_y[:, newaxis]**2 + model_z[:, newaxis, newaxis]**2 > (self.model_side/2.)**2
        #self.mask = (self.x_coordinates**2 + self.y_coordinates**2) > (self.model_side/2.)**2

    def insert(self, model, rotation_matrix):
        x_rotated = int32(self.x_coordinates*rotation_matrix[0,0] + self.y_coordinates*rotation_matrix[0,1] +
                          image_side/2 + 0.5)
        y_rotated = int32(self.x_coordinates*rotation_matrix[1,0] + self.y_coordinates*rotation_matrix[1,1] +
                          image_side/2 + 0.5)
        z_rotated = int32(self.x_coordinates*rotation_matrix[2,0] + self.y_coordinates*rotation_matrix[2,1] +
                          image_side/2 + 0.5)


        x_out_of_range = (x_rotated < 0) + (x_rotated >= image_side)
        y_out_of_range = (y_rotated < 0) + (y_rotated >= image_side)
        z_out_of_range = (z_rotated < 0) + (z_rotated >= image_side)

        out_of_range = x_out_of_range + y_out_of_range + z_out_of_range
        x_rotated[out_of_range] = image_side/2
        y_rotated[out_of_range] = image_side/2
        z_rotated[out_of_range] = image_side/2

        model[x_rotated, y_rotated, z_rotated] += 1.
        model[self.mask] = 0.

model_side = 40
image_side = 40

inserter = SliceInserter(model_side, image_side)

max_sum = sum(-inserter.mask)
number_of_iterations = 10
number_of_images = 1000

coverage = zeros(number_of_iterations)

full_coverage = []

full_coverage_1 = []
full_coverage_2 = []
full_coverage_3 = []
full_coverage_4 = []

for i in range(number_of_iterations):
    print i
    model = zeros((model_side,)*3)
    full_coverage_1.append(zeros(number_of_images))
    full_coverage_2.append(zeros(number_of_images))
    full_coverage_3.append(zeros(number_of_images))
    full_coverage_4.append(zeros(number_of_images))
    
    for image_n in range(number_of_images):
        inserter.insert(model, rotations.quaternion_to_matrix(rotations.random_quaternion()))

        if sum(model >= 1) == max_sum:
            full_coverage_1[i][image_n] = 1.
        if sum(model >= 2) == max_sum:
            full_coverage_2[i][image_n] = 1.
        if sum(model >= 3) == max_sum:
            full_coverage_3[i][image_n] = 1.
        if sum(model >= 4) == max_sum:
            full_coverage_4[i][image_n] = 1.



# average_coverage = average(array(full_coverage), axis=0)


# ax.plot(average_coverage)
# pickle.dump(average_coverage, open('average_coverage.p', 'wb'))

show()
