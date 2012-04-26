from pylab import *
import rotations

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

resolution = 40 #must be integer
model_side = resolution*2
image_side = resolution*2

number_of_images = 500
repeats = 100

coverage = zeros((repeats, number_of_images))
inserter = SliceInserter(model_side, image_side)

max_sum = sum(-inserter.mask) # - is boolean not

for repetition in range(repeats):
    model = zeros((model_side,)*3)
    print 'repetition ', repetition
    for image_n in range(number_of_images):
        inserter.insert(model, rotations.quaternion_to_matrix(rotations.random_quaternion()))
        coverage[repetition,image_n] = float(sum(model > 0))/float(max_sum)


coverage_average = average(coverage, axis=0)

d_array = arange(1., number_of_images) #nm
Ks = 4.*pi*(resolution)**2/2.
ks = 2.*pi*(resolution)/2.

N = arange(number_of_images)
coverage_analytical = 1. - exp(-N*ks/Ks)

fig = figure(1, figsize=(8,6), dpi=100)
fig.clear()
ax = fig.add_subplot(111)
ax.plot(N, coverage_average, label='simulated')
ax.plot(N, coverage_analytical, label='analytical')
ax.legend()
ax.set_xlabel('Number of images')
ax.set_ylabel('Coverage')

show()
