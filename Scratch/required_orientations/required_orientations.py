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
        self.Ks_int = int(self.Ks)
        self.ks_int = int(self.ks)

    def get_int_ratio(self):
        return int(self._object_diameter/self._resolution*2.)

class Analytic(object):
    def __init__(self, setup):
        self._setup = setup

    def full_coverage(self, number_of_images):
        return (1. - (1.-self._setup.ks/self._setup.Ks)**number_of_images)**self._setup.Ks

    def coverage(self, number_of_images):
        return 1. - exp(-number_of_images*self._setup.ks/self._setup.Ks)

    def N_from_coverage(self, coverage):
        return -self._setup.Ks/self._setup.ks*log(1.-coverage)

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

        # adding a -1.0 to the mask radius makes it fit quite good with the analytical case. Is this a coincidence?
        self.mask = (model_x**2 + model_y[:, newaxis]**2 + model_z[:, newaxis, newaxis]**2 >
                     (self.model_side/2.-0.0)**2)
        self.max_sum = sum(-self.mask) # - is boolean not

        #the ring mask is inverted with respect to the other mask (the ring is truth)
        self.shell_mask = (-self.mask) * (model_x**2 + model_y[:, newaxis]**2 + model_z[:, newaxis, newaxis]**2 >
                                          (self.model_side/2.-1.)**2) # * is boolean and
        self.shell_max_sum = sum(self.shell_mask)
        #self.mask = (self.x_coordinates**2 + self.y_coordinates**2) > (self.model_side/2.)**2

    def insert(self, model, rotation_matrix):
        x_rotated = int32(self.x_coordinates*rotation_matrix[0,0] + self.y_coordinates*rotation_matrix[0,1] +
                          (self.image_side/2-0.5) + 0.5)
        y_rotated = int32(self.x_coordinates*rotation_matrix[1,0] + self.y_coordinates*rotation_matrix[1,1] +
                          (self.image_side/2-0.5) + 0.5)
        z_rotated = int32(self.x_coordinates*rotation_matrix[2,0] + self.y_coordinates*rotation_matrix[2,1] +
                          (self.image_side/2-0.5) + 0.5)


        x_out_of_range = (x_rotated < 0) + (x_rotated >= self.image_side)
        y_out_of_range = (y_rotated < 0) + (y_rotated >= self.image_side)
        z_out_of_range = (z_rotated < 0) + (z_rotated >= self.image_side)

        out_of_range = x_out_of_range + y_out_of_range + z_out_of_range
        x_rotated[out_of_range] = self.image_side/2
        y_rotated[out_of_range] = self.image_side/2
        z_rotated[out_of_range] = self.image_side/2

        model[x_rotated, y_rotated, z_rotated] += 1.
        model[self.mask] = 0.


class SimulationRealistic(object):
    def __init__(self, setup):
        self._setup = setup

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

        jobs = [(inserter,)]*number_of_repeats
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
                full_coverage[image_n] = (sum(model > 0) >= inserter.max_sum)
            return full_coverage

        jobs = [(inserter,)]*number_of_repeats
        job_results = parallel.run_parallel(jobs, single_repeat)
        return average(job_results, axis=0)

class SimulationModel(object):
    def __init__(self, setup):
        self._setup = setup

    def coverage(self, max_images, number_of_repeats):

        # def single_repeat(input):
        #     setup = input[0]; max_images = input[1]
        def single_repeat(setup, max_images):
            model = zeros(setup.Ks_int)
            coverage = zeros(max_images)
            
            def get_n_unique(n, index_range):
                index_array = -ones(n, dtype='int32')
                for i, value in enumerate(index_array):
                    new_value = random_integers(index_range)-1
                    while new_value in index_array:
                        new_value = random_integers(index_range)-1
                    index_array[i] = new_value
                return index_array
            
            for image_n in range(max_images):
                indices = get_n_unique(setup.ks_int, setup.Ks_int)
                model[indices] = 1.
                coverage[image_n] = float(sum(model)) / float(setup.Ks_int)
            return coverage

        jobs = [(self._setup, max_images)]*number_of_repeats
        job_results = parallel.run_parallel(jobs, single_repeat)
        return average(job_results, axis=0)

    def full_coverage(self, max_images, number_of_repeats):
        # def single_repeat(input):
        #     setup = input[0]; max_images = input[1]
        def single_repeat(setup, max_images):
            model = zeros(setup.Ks_int)
            full_coverage = zeros(max_images)

            def get_n_unique(n, index_range):
                index_array = -ones(n, dtype='int32')
                for i, value in enumerate(index_array):
                    new_value = random_integers(index_range)-1
                    while new_value in index_array:
                        new_value = random_integers(index_range)-1
                    index_array[i] = new_value
                return index_array
            
            for image_n in range(max_images):
                indices = get_n_unique(setup.ks_int, setup.Ks_int)
                model[indices] = 1.
                full_coverage[image_n] = sum(model) == setup.Ks_int
            return full_coverage

        jobs = [(self._setup, max_images)]*number_of_repeats
        job_results = parallel.run_parallel(jobs, single_repeat)
        return average(job_results, axis=0)
                

if __name__ == "__main__":
    setup = Setup(83.*20, 83.) #415
    a = Analytic(setup)
    sr = SimulationRealistic(setup)
    sm = SimulationModel(setup)

    max_images = 600
    an_full_coverage = a.full_coverage(arange(max_images))
    sr_full_coverage = sr.full_coverage(max_images, 100)
    sm_full_coverage = sm.full_coverage(max_images, 100)

    plot(an_full_coverage, label='analytical')
    plot(sr_full_coverage, label='realistic sim')
    plot(sm_full_coverage, label='model sim')
    legend()
    show()
