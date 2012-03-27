from pylab import *
from scipy.special import binom
import pickle
import parallel
import rotations

class Setup(object):
    def __init__(self, object_diameter, resolution):
        self._object_diameter = object_diameter
        self._resolution = resolution
        self.Ks = 4.*pi*(self._object_diameter/self._resolution)**2/2.
        self.ks = 2.*pi*(self._object_diameter/self._resolution)/2.

    def get_int_ratio(self):
        return int(self._object_diameter/self._resolution*2.)

class Analytic(object):
    def __init__(self, setup):
        self._setup = setup

    def full_coverage(self, number_of_images):
        return (1. - (1.-self._setup.ks/self._setup.Ks)**number_of_images)**self._setup.Ks

    def coverage(self, number_of_images):
        return 1. - exp(-number_of_images*self._setup.ks/self._setup.Ks)



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
        self.max_sum = sum(-self.mask) # - is boolean not

        #the ring mask is inverted with respect to the other mask (the ring is truth)
        self.shell_mask = (-self.mask) * (model_x**2 + model_y[:, newaxis]**2 + model_z[:, newaxis, newaxis]**2 > (self.model_side/2.-1.)**2) # * is boolean and
        self.shell_max_sum = sum(self.shell_mask)
        #self.mask = (self.x_coordinates**2 + self.y_coordinates**2) > (self.model_side/2.)**2

    def insert(self, model, rotation_matrix):
        x_rotated = int32(self.x_coordinates*rotation_matrix[0,0] + self.y_coordinates*rotation_matrix[0,1] +
                          self.image_side/2 + 0.5)
        y_rotated = int32(self.x_coordinates*rotation_matrix[1,0] + self.y_coordinates*rotation_matrix[1,1] +
                          self.image_side/2 + 0.5)
        z_rotated = int32(self.x_coordinates*rotation_matrix[2,0] + self.y_coordinates*rotation_matrix[2,1] +
                          self.image_side/2 + 0.5)


        x_out_of_range = (x_rotated < 0) + (x_rotated >= self.image_side)
        y_out_of_range = (y_rotated < 0) + (y_rotated >= self.image_side)
        z_out_of_range = (z_rotated < 0) + (z_rotated >= self.image_side)

        out_of_range = x_out_of_range + y_out_of_range + z_out_of_range
        x_rotated[out_of_range] = self.image_side/2
        y_rotated[out_of_range] = self.image_side/2
        z_rotated[out_of_range] = self.image_side/2

        model[x_rotated, y_rotated, z_rotated] += 1.
        model[self.mask] = 0.


class Simulation(object):
    def __init__(self, setup):
        self._setup = setup
        self._model_side = setup.get_int_ratio()

    def coverage(self, max_images, number_of_repeats):
        """Calculate the persent of pixels in the outer shell that are covered"""
        model_side = self._setup.get_int_ratio()
        image_side = self._setup.get_int_ratio()
        inserter = SliceInserter(model_side, image_side)

        def single_repeat(inserter):
            model = zeros((model_side,)*3)
            coverage = zeros(max_images)
            for image_n in range(max_images):
                inserter.insert(model, rotations.quaternion_to_matrix(rotations.random_quaternion()))
                #coverage[image_n] = float(sum(model > 0)) / float(inserter.max_sum)
                coverage[image_n] = float(sum(model*inserter.shell_mask > 0)) / float(inserter.shell_max_sum)
            return coverage

        jobs = [inserter]*number_of_repeats
        job_results = parallel.run_parallel(jobs, single_repeat)
        return average(job_results, axis=0)
        
    def full_coverage(self, max_images, number_of_repeats):
        model_side = self._setup.get_int_ratio()
        image_side = self._setup.get_int_ratio()
        inserter = SliceInserter(model_side, image_side)

        def single_repeat(inserter):
            model = zeros((model_side,)*3)
            full_coverage = zeros(max_images, dtype='bool')
            for image_n in range(max_images):
                inserter.insert(model, rotations.quaternion_to_matrix(rotations.random_quaternion()))
                full_coverage[image_n] = (sum(model > 0) == inserter.max_sum)
            return full_coverage

        jobs = [inserter]*number_of_repeats
        job_results = parallel.run_parallel(jobs, single_repeat)
        return average(job_results, axis=0)
