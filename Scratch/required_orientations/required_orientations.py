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
        # the adjusted values are lower since they take inte account that the outer shell has a thickness of 1.
        self.Ks_adjusted = 4.*pi*(self._object_diameter/self._resolution-0.5)**2/2.
        self.ks_adjusted = 2.*pi*(self._object_diameter/self._resolution-0.5)/2.
        self.Ks_adjusted_int = int(self.Ks_adjusted)
        self.ks_adjusted_int = int(self.ks_adjusted)

    def set_Ks(self, Ks):
        self.Ks = Ks
        self.Ks_int = int32(Ks)
        self.Ks_adjusted = 2.*pi*(sqrt(Ks/2./pi) - 0.5)**2
        self.Ks_adjusted_int = int(self.Ks_adjusted)

    def set_ks(self, ks):
        self.ks = ks
        self.ks_int = int32(ks)
        self.ks_adjusted = ks - pi/2.
        self.ks_adjusted_int = int(self.ks_adjusted)

    def get_int_ratio(self):
        return int(self._object_diameter/self._resolution*2.)

class Analytic(object):
    def __init__(self, setup):
        self._setup = setup

    def full_coverage(self, number_of_images):
        return (1. - (1.-self._setup.ks_adjusted/self._setup.Ks_adjusted)**number_of_images)**self._setup.Ks_adjusted

    def coverage(self, number_of_images):
        return 1. - exp(-number_of_images*self._setup.ks_adjusted/self._setup.Ks_adjusted)

    def N_from_coverage(self, coverage):
        return -self._setup.Ks_adjusted/self._setup.ks_adjusted*log(1.-coverage)

    def at_most_n_missing(self, n, number_of_images):
        p_single = 1. - (1. - float(self._setup.ks_adjusted_int)/float(self._setup.Ks_adjusted_int))**number_of_images
        return sum([binom(self._setup.Ks_adjusted_int, k)*p_single**k*(1.-p_single)**(self._setup.Ks_adjusted_int-k) for k in range(self._setup.Ks_adjusted_int-n, self._setup.Ks_adjusted_int+1)], axis=0)
        

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
        
    def full_coverage(self, max_images, number_of_repeats, max_missing=0):
        model_side = self._setup.get_int_ratio()
        image_side = self._setup.get_int_ratio()
        inserter = SliceInserter(model_side, image_side)

        def single_repeat(inserter):
            model = zeros((model_side,)*3)
            full_coverage = zeros(max_images, dtype='bool')
            for image_n in range(max_images):
                inserter.insert(model, rotations.quaternion_to_matrix(rotations.random_quaternion()))
                #full_coverage[image_n] = (sum(model > 0) >= inserter.max_sum)
                full_coverage[image_n] = (sum(model*inserter.shell_mask > 0) >= inserter.shell_max_sum-max_missing)
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
            model = zeros(setup.Ks_adjusted_int)
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
                indices = get_n_unique(setup.ks_adjusted_int, setup.Ks_adjusted_int)
                model[indices] = 1.
                coverage[image_n] = float(sum(model)) / float(setup.Ks_adjusted_int)
            return coverage

        jobs = [(self._setup, max_images)]*number_of_repeats
        job_results = parallel.run_parallel(jobs, single_repeat)
        return average(job_results, axis=0)

    def full_coverage(self, max_images, number_of_repeats, max_missing=0):
        # def single_repeat(input):
        #     setup = input[0]; max_images = input[1]
        def single_repeat(setup, max_images):
            model = zeros(setup.Ks_adjusted_int)
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
                indices = get_n_unique(setup.ks_adjusted_int, setup.Ks_adjusted_int)
                model[indices] = 1.
                full_coverage[image_n] = (sum(model) >= setup.Ks_adjusted_int-max_missing)
            return full_coverage

        jobs = [(self._setup, max_images)]*number_of_repeats
        job_results = parallel.run_parallel(jobs, single_repeat)
        return average(job_results, axis=0)
                

if __name__ == "__main__":
    # setup = Setup(83.*15, 83.) #415
    # a = Analytic(setup)
    # sm = SimulationModel(setup)
    # setup = Setup(83.*16, 83.) #415
    # sr = SimulationRealistic(setup)


    max_images = 1300
    # an_full_coverage = a.full_coverage(arange(max_images))
    # sr_full_coverage = sr.full_coverage(max_images, 200)
    # sm_full_coverage = sm.full_coverage(max_images, 200)
    #an_full_coverage = a.at_most_n_missing(0, arange(max_images))
    #sr_full_coverage = sr.full_coverage(max_images, 1000)
    #sm_full_coverage = sm.full_coverage(max_images, 1000)


    setup = Setup(83.*32, 83.)
    a = Analytic(setup)
    an_full_coverage = a.full_coverage(arange(max_images))
    
    sm = SimulationModel(setup)
    sm_full_coverage = sm.full_coverage(max_images, 1000)
    
    print "Ks = %d, ks = %d" % (setup.Ks, setup.ks)
    
    sr = SimulationRealistic(setup)
    sr_full_coverage = sr.full_coverage(max_images, 1000)

    print "old Ks = %d" % setup.Ks
    setup.set_Ks(setup.Ks*1.1) #this seems to work pretty well (for some reason)
    print "new Ks = %d" % setup.Ks
    
    #setup.set_ks(48.)
    sm2 = SimulationModel(setup)
    sm2_full_coverage = sm.full_coverage(max_images, 1000)

    plot(an_full_coverage, label='analytical')
    plot(sr_full_coverage, label='realistic sim')
    plot(sm_full_coverage, label='model sim')
    plot(sm2_full_coverage, label='model sim adjusted')
    legend()
    show()
