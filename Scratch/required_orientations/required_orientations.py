from pylab import *
from scipy.special import binom
import pickle
import parallel
import rotations
import nice_plots

nice_plots.apply_standard_features()

#the weird_factor is a measure of the correlation between pixel placement.
weird_factor = 2. #should be 2  (0.6 looks better but not good. This would imply a lower correlation between the placement of pixels from one slice. I think only values above 2. makes sense.)
#pixel_scaling = 1./sqrt(2) # 0.7
pixel_scaling = 1.

class Setup(object):
    def __init__(self, object_diameter, resolution, s = 1.):
        self._object_diameter = object_diameter
        self._resolution = resolution
        self.Ks = 4.*pi*(s*self._object_diameter/self._resolution*pixel_scaling)**2/weird_factor
        self.ks = 2.*pi*(s*self._object_diameter/self._resolution*pixel_scaling)/weird_factor
        self.Ks_int = int(self.Ks)
        self.ks_int = int(self.ks)
        # the adjusted values are lower since they take inte account that the outer shell has a thickness of 1.
        self.Ks_adjusted = 4.*pi*(s*self._object_diameter/self._resolution*pixel_scaling-0.5)**2/weird_factor
        self.ks_adjusted = 2.*pi*(s*self._object_diameter/self._resolution*pixel_scaling-0.5)/weird_factor
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
        return log(1.-coverage**(1./self._setup.Ks_adjusted)) / log(1.-self._setup.ks_adjusted/self._setup.Ks_adjusted)
        #return -self._setup.Ks_adjusted/self._setup.ks_adjusted*log(1.-coverage)

    def at_most_n_missing(self, u, number_of_images):
        #p_single = 1. - (1. - float(self._setup.ks_adjusted_int)/float(self._setup.Ks_adjusted_int))**number_of_images
        #return sum([binom(self._setup.Ks_adjusted_int, k)*p_single**k*(1.-p_single)**(self._setup.Ks_adjusted_int-k) for k in range(self._setup.Ks_adjusted_int-n, self._setup.Ks_adjusted_int+1)], axis=0)
        p_single = 1. - (1. - float(self._setup.ks_adjusted)/float(self._setup.Ks_adjusted))**number_of_images
        return sum([binom(self._setup.Ks_adjusted, v)*p_single**v*(1.-p_single)**(self._setup.Ks_adjusted-v) for v in arange(self._setup.Ks_adjusted-u, self._setup.Ks_adjusted+1)], axis=0)

    def at_least_n_samplings(self, n, number_of_images):
        m = arange(n)
        return (1. - sum(binom(number_of_images, m[:,newaxis]) * (self._setup.ks_adjusted/self._setup.Ks_adjusted)**m[:,newaxis] * (1.-self._setup.ks_adjusted/self._setup.Ks_adjusted)**(number_of_images-m[:,newaxis]), axis=0))**self._setup.Ks_adjusted

        

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

    def get_shell_count(self):
        x = arange(-self.model_side/2+0.5, self.model_side/2+0.5)
        r = sqrt(x**2 + x[:, newaxis]**2 + x[:, newaxis, newaxis]**2)
        return sum((r > (self.model_side/2.-1.)) * (r < self.model_side/2.))

    def get_circle_count(self):
        x = arange(-self.model_side/2+0.5, self.model_side/2+0.5)
        r = sqrt(x**2 + x[:, newaxis]**2)
        return sum((r > (self.model_side/2.-1.)) * (r < self.model_side/2.))

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
        job_results = parallel.run_parallel(jobs, single_repeat, quiet=True)
        return average(job_results, axis=0)
        
    def full_coverage(self, max_images, number_of_repeats, max_missing=0):
        model_side = self._setup.get_int_ratio()
        image_side = self._setup.get_int_ratio()
        self.inserter = SliceInserter(model_side, image_side)

        def single_repeat(inserter):
            model = zeros((model_side,)*3)
            full_coverage = zeros(max_images, dtype='bool')
            for image_n in range(max_images):
                inserter.insert(model, rotations.quaternion_to_matrix(rotations.random_quaternion()))
                #full_coverage[image_n] = (sum(model > 0) >= inserter.max_sum)
                full_coverage[image_n] = (sum(model*inserter.shell_mask > 0) >= inserter.shell_max_sum-max_missing)
            return full_coverage

        jobs = [(self.inserter,)]*number_of_repeats
        job_results = parallel.run_parallel(jobs, single_repeat, quiet=True)
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
        job_results = parallel.run_parallel(jobs, single_repeat, quiet=True)
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
        job_results = parallel.run_parallel(jobs, single_repeat, quiet=True)
        return average(job_results, axis=0)
                
OUTPUT_DIR = '/Users/ekeberg/Work/Random results/'

def plot_missing_pixels():
    resolution = 10
    number_of_images = 300
    setup = Setup(83.*resolution, 83., s = 2.**(1./3.))
    an = Analytic(setup)
    missing_pixels = range(5)

    number_of_images_array = arange(number_of_images)

    fig = figure(1, dpi=150)
    fig.clear()
    fig.subplots_adjust(left=0.13, bottom=0.13, right=0.9, top=0.9)
    ax = fig.add_subplot(111)
    
    for missing in missing_pixels:
        ax.plot(number_of_images_array, an.at_most_n_missing(missing, number_of_images_array), label="{} pixels missing".format(missing))

    ax.set_xlabel("Number of diffraction patterns")
    ax.set_ylabel("Probability of full coverage")
    ax.legend()
    ax.set_ylim((0., 1.))

    fig.savefig('%s/missing_pixels.png' % OUTPUT_DIR, dpi=300)
    fig.savefig('%s/missing_pixels.pdf' % OUTPUT_DIR)
        

def plot_full_coverage_analytic():
    resolution_array = [10]
    number_of_images = 400
    setup_array = [Setup(83.*resolution, 83, s = 2.**(1./3.)) for resolution in resolution_array]
    an_array = [Analytic(setup) for setup in setup_array]
    #an = Analytic(setup)
    number_of_images_array = arange(number_of_images)

    fig = figure(1, dpi=150)
    fig.clear()
    fig.subplots_adjust(left=0.13, bottom=0.13, right=0.9, top=0.9)
    ax = fig.add_subplot(111)

    #ax.plot(number_of_images_array, an.full_coverage(number_of_images_array))
    [ax.plot(number_of_images_array, an.full_coverage(number_of_images_array)) for an in an_array]
    ax.set_xlim((0, number_of_images))
    ax.set_ylim((0, 1))
    ax.set_xlabel(r'Number of diffraction patterns')
    ax.set_ylabel(r'Probability of full coverage')
    fig.savefig('%s/full_coverage_analytic.png' % OUTPUT_DIR, dpi=300)
    fig.savefig('%s/full_coverage_analytic.pdf' % OUTPUT_DIR)

def plot_images_required():
    resolution = arange(0.01, 100, 0.01)
    fig = figure(1, dpi=150)
    fig.clear()
    fig.subplots_adjust(left=0.13, bottom=0.13, right=0.9, top=0.9)
    ax = fig.add_subplot(111)
    
    p = 0.95
    K = 4.*pi*(resolution-0.5)**2/2.
    k = 2.*pi*(resolution-0.5)/2.
    images_required = log(1.-p**(1./K)) / log(1.-k/K)

    ax.plot(resolution, images_required)
    ax.set_xlim((0, resolution[-1]))
    ax.set_ylim((0, 3000))
    ax.set_xlabel(r'Resolution ($R$)')
    ax.set_ylabel(r'Number of diffraction patterns')
    fig.savefig('%s/images_required.png' % OUTPUT_DIR, dpi=300)
    fig.savefig('%s/images_required.pdf' % OUTPUT_DIR)

def plot_missing_pixels():
    resolution = 10
    number_of_images = 350
    setup = Setup(83.*resolution, 83., s = 2.**(1./3.))
    an = Analytic(setup)
    number_of_images_array = arange(number_of_images)

    fig = figure(1, dpi=150)
    fig.clear()
    fig.subplots_adjust(left=0.13, bottom=0.13, right=0.9, top=0.9)
    ax = fig.add_subplot(111)

    missing_pixels_array = array([0, 1, 2, 3, 4, 5])
    graphs = []
    for missing_pixels in missing_pixels_array:
        graphs.append(an.at_most_n_missing(missing_pixels, number_of_images_array))
        ax.plot(number_of_images_array, graphs[-1], 'blue')

    text_placement_hor = 215
    text_placement_ver = arange(0.75, 0.75-6*0.08, -0.08)
    #arrow_placement_hor = [140, 117, 110, 105, 100, 95]
    arrow_placement_ver = text_placement_ver

    for i in range(6):
        ax.text(text_placement_hor+2, text_placement_ver[i], r'%d pixels missing' % missing_pixels_array[i], backgroundcolor='white',
                va='center', size='large')
        ax.arrow(text_placement_hor, text_placement_ver[i],
                 number_of_images_array[abs(graphs[i]-arrow_placement_ver[i]).argmin()]-text_placement_hor,
                 arrow_placement_ver[i]-text_placement_ver[i],
                 zorder=100, shape='full', label='%d' % missing_pixels_array[i],
                 length_includes_head=True, color='black', width=0.002, head_width=0.02, head_length=10.)
        
        # ax.arrow(text_placement_hor, text_placement_ver[i], arrow_placement_hor[i]-text_placement_hor,
        #          graphs[i][list(number_of_images_array).index(arrow_placement_hor[i])]-text_placement_ver[i])

    ax.set_xlabel(r'Number of diffraction patterns')
    ax.set_ylabel(r'Probability of full coverage')
    fig.savefig('%s/missing_pixels.png' % OUTPUT_DIR, dpi=300)
    fig.savefig('%s/missing_pixels.pdf' % OUTPUT_DIR)

def plot_multi_hits():
    resolution = 10
    number_of_images = 5000
    setup = Setup(resolution, 1, s = 2.**(1./3.))
    an = Analytic(setup)
    number_of_images_array = arange(number_of_images)

    fig = figure(1)
    fig.clear()
    fig.subplots_adjust(left=0.13, bottom=0.13, right=0.9, top=0.9)
    ax = fig.add_subplot(111)

    number_of_samplings = arange(1, 100, 1)
    # for n in number_of_samplings:
    #     ax.plot(number_of_images_array, an.at_least_n_samplings(n, number_of_images_array))
    prob_limit = 0.95
    N = []
    for n in number_of_samplings:
        prob = an.at_least_n_samplings(n, number_of_images_array)
        N.append(list(prob > prob_limit).index(True))
    ax.plot(number_of_samplings, N)

    ax.set_xlabel(r'Number of samplings per pixel ($n$)')
    ax.set_ylabel(r'Number of diffraction patterns')
    fig.savefig('%s/multi_hits.png' % OUTPUT_DIR, dpi=300)
    fig.savefig('%s/multi_hits.pdf' % OUTPUT_DIR)
    draw()

def plot_sim_vs_analytic():
    max_images = 600
    number_of_trials = 1000
    model_side_list = range(5, 36, 2)

    fig = figure(1)
    fig.clear()
    ax = fig.add_subplot(111)

    for model_side in model_side_list:
        setup = Setup(83.*model_side, 83., s = 2.**(1./3.))
        an = Analytic(setup)
        an_full_coverage = an.full_coverage(arange(max_images))
        ax.plot(an_full_coverage, color='blue')

    from exact_simulation import Gaps

    exact_simulation_filename = 'exact_600.p'
    file_handle = open(exact_simulation_filename, 'rb')
    gaps = pickle.load(file_handle)
    file_handle.close()

    resolution_list = array(model_side_list)*2
    full_coverage_plot = []
    for resolution in resolution_list:
        conversion_factor = resolution/2. # nyquist = coordinate * conversion_factor
        gaps_nyquist_all = gaps.gaps() * conversion_factor
        full_coverage_all = gaps_nyquist_all <= 0.7 #looks ok at 0.7
        full_coverage_probability = average(full_coverage_all, axis=1)
        full_coverage_plot.append(full_coverage_probability)

    for i, p in enumerate(full_coverage_plot):
        plot(gaps.number_of_images(), p, label=str(resolution_list[i]), color='green')

    ax.set_xlabel('Number of diffraction patterns')
    ax.set_ylabel('Probability of full coverage')
    fig.savefig('%s/sim_vs_analytic.png' % OUTPUT_DIR, dpi=300)
    fig.savefig('%s/sim_vs_analytic.pdf' % OUTPUT_DIR)
    show()

    

def make_all_plots():
    print "plot multi_hits"
    plot_multi_hits()
    print "plot missing_pixels"
    plot_missing_pixels()
    print "plot images_required"
    plot_images_required()
    print "plot full_coverage_analytic"
    plot_full_coverage_analytic()
    print "plot missing_pixels"
    plot_missing_pixels()
    print "done"

if __name__ == "__main__a":
    # setup = Setup(83.*15, 83., s = 2.**(1./3.)) #415
    # a = Analytic(setup)
    # sm = SimulationModel(setup)
    # setup = Setup(83.*16, 83., s = 2.**(1./3.)) #415
    # sr = SimulationRealistic(setup)



    # an_full_coverage = a.full_coverage(arange(max_images))
    # sr_full_coverage = sr.full_coverage(max_images, 200)
    # sm_full_coverage = sm.full_coverage(max_images, 200)
    #an_full_coverage = a.at_most_n_missing(0, arange(max_images))
    #sr_full_coverage = sr.full_coverage(max_images, 1000)
    #sm_full_coverage = sm.full_coverage(max_images, 1000)
    
    max_images = 600
    number_of_trials = 1000
    #model_side_list = [6, 12, 18, 24]
    #model_side_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
    model_side_list = range(5, 36, 2)

    fig = figure(1)
    fig.clear()
    ax = fig.add_subplot(111)

    for model_side in model_side_list:
        setup = Setup(83.*model_side, 83., s = 2.**(1./3.))
        an = Analytic(setup)
        an_full_coverage = an.full_coverage(arange(max_images))
    
        print "side = %d" % model_side
        print "Ks = %d, ks = %d" % (setup.Ks, setup.ks)

        # sr = SimulationRealistic(setup)
        # sr_full_coverage = sr.full_coverage(max_images, number_of_trials)

        # print "Ks = %d, ks = %d" % (sr.inserter.get_shell_count()/2, sr.inserter.get_circle_count()/2)

        # inserter = SliceInserter(model_side*2, model_side*2)

        # setup.set_Ks(inserter.get_shell_count()/2.)
        # setup.set_ks(inserter.get_circle_count()/2.)
        # an_adjust = Analytic(setup)
        # an_adjust_full_coverage = an.full_coverage(arange(max_images))

        # print "Ks = %d, ks = %d" % (setup.Ks, setup.ks)
        
        if model_side == model_side_list[0]:
            ax.plot(an_full_coverage, label='analytical', color='blue')
            #ax.plot(sr_full_coverage, label='realistic sim', color='green')
            #ax.plot(an_adjust_full_coverage, label='adjusted analytical', color='red')
        else:
            ax.plot(an_full_coverage, color='blue')
            #ax.plot(sr_full_coverage, color='green')
            #ax.plot(an_adjust_full_coverage, color='red')

    from exact_simulation import Gaps

    # plot result from exact simulation as well
    exact_simulation_filename = 'exact_600.p'
    file_handle = open(exact_simulation_filename, 'rb')
    gaps = pickle.load(file_handle)
    file_handle.close()
    
    #gaps.number_of_images()
    
    #resolution_list = [10., 15., 20., 25., 30.] #This is the number of resolution elments along the object
    resolution_list = array(model_side_list)*2
    full_coverage_plot = []
    for resolution in resolution_list:
        conversion_factor = resolution/2. # nyquist = coordinate * conversion_factor
        gaps_nyquist_all = gaps.gaps() * conversion_factor
        full_coverage_all = gaps_nyquist_all <= 0.7 #looks ok at 0.7
        full_coverage_probability = average(full_coverage_all, axis=1)
        full_coverage_plot.append(full_coverage_probability)

    for i, p in enumerate(full_coverage_plot):
        plot(gaps.number_of_images(), p, label=str(resolution_list[i]), color='green')


    #ax.set_xlim((0,800))
    #ax.legend()
    ax.set_xlabel('Number of images')
    ax.set_ylabel('Probability of full coverage')
    show()

    #ax.plot(sm_full_coverage, label='model sim')
    #ax.plot(sm2_full_coverage, label='model sim adjusted')
    #ax.plot(an_adjust_full_coverage, label='analytic adjusted')
    #sm = SimulationModel(setup)
    #sm_full_coverage = sm.full_coverage(max_images, number_of_trials)

    # inserter = SliceInserter(model_side*2, model_side*2)
    # setup2 = Setup(83.*model_side, 83., s = 2.**(1./3.))
    # setup2.set_Ks(inserter.get_shell_count()/2)
    # setup2.set_ks(inserter.get_circle_count()/2)
    # an_adjust = Analytic(setup2)
    # an_adjust_full_coverage = an_adjust.full_coverage(arange(max_images))

    # print "old Ks = %d" % setup.Ks
    # setup.set_Ks(setup.Ks*1.1) #this seems to work pretty well (for some reason)
    # print "new Ks = %d" % setup.Ks
    
    # #setup.set_ks(48.)
    # sm2 = SimulationModel(setup)
    # sm2_full_coverage = sm.full_coverage(max_images, number_of_trials)


