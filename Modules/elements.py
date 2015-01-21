"""X-ray material properties. Calculate cross section, attenuation length and more."""
import pylab as _pylab
import pickle as _pickle
import conversions as _conversions
import constants as _constants
import os as _os
import _info

_ELEMENTS_FILE = open(_os.path.join(_info.install_directory, "/Resources/elements.dat")
ATOMIC_MASS, SCATTERING_FACTORS = _pickle.load(_ELEMENTS_FILE)
_ELEMENTS_FILE.close()
ELEMENTS = ATOMIC_MASS.keys()

class ChemicalComposition(object):
    """This class contains a set of relative abundances of materials."""
    def __init__(self, **kwargs):
        self._elements = kwargs
        bad_elements = []
        for key in self._elements:
            if key not in ATOMIC_MASS.keys():
                bad_elements.append(key)
                raise IndexError("%s is not an element." % key)
        for element in bad_elements:
            self._elements.pop(element)

    def element_ratios(self):
        """Returns relative ratios of elements (not relative masses)"""
        return self._elements

    def element_ratio_sum(self):
        """Returns the sum of all relative ratios of elements (not relative masses)"""
        return sum(self._elements.values())

    def element_mass_ratios(self):
        """Return relative masses of all elements in the material"""
        mass_ratios = {}
        for element in self._elements:
            mass_ratios[element] = self._elements[element]*ATOMIC_MASS[element]
        return mass_ratios

    def element_mass_ratio_sum(self):
        """Returns the sum of all relative mass ratios returned bu element_mass_ratios()"""
        mass_ratios = self.element_mass_ratios()
        return sum(mass_ratios.values())


class Material(object):
    """Contains both the density and the chemical composition of the material"""
    def __init__(self, density, **kwargs):
        self._density = density
        self._chemical_composition = ChemicalComposition(**kwargs)

    def element_ratios(self):
        """Returns relative ratios of elements (not relative masses)"""
        return self._chemical_composition.element_ratios()

    def element_ratio_sum(self):
        """Returns the sum of all relative ratios of elements (not relative masses)"""
        return self._chemical_composition.element_ratio_sum()

    def element_mass_ratios(self):
        """Return relative masses of all elements in the material"""
        return self._chemical_composition.element_mass_ratios()

    def element_mass_ratio_sum(self):
        """Returns the sum of all relative mass ratios returned bu element_mass_ratios()"""
        return self._chemical_composition.element_mass_ratio_sum()

    def material_density(self):
        """Return the density of the material."""
        return self._density

    def __repr__(self):
        ratios = self.element_ratios()
        return ' '.join([i+":"+str(ratios[i]) for i in ratios]) + ", density: " + str(self._density)

MATERIALS = {"protein" : Material(1350, H=86, C=52, N=13, O=15, P=0, S=3),
             "water" : Material(1000, O=1, H=2),
             "virus" : Material(1455, H=72.43, C=47.52, N=13.55, O=17.17, P=1.11, S=0.7),
             "cell" : Material(1000, H=23, C=3, N=1, O=10, P=0, S=1)}

def get_scattering_factor(element, photon_energy):
    """
    get the scattering factor for an element through linear interpolation. Photon energy is given in eV.
    """
    f_1 = _pylab.interp(photon_energy, SCATTERING_FACTORS[element][:, 0], SCATTERING_FACTORS[element][:, 1])
    f_2 = _pylab.interp(photon_energy, SCATTERING_FACTORS[element][:, 0], SCATTERING_FACTORS[element][:, 2])
    #return [f1,f2]
    return f_1+f_2*1.j

def get_scattering_power(photon_energy, material, complex_scattering_factor=False):
    """Returns the scattering factor for a volume of 1 m^3 of the material"""
    average_density = 0.0
    total_atomic_ammounts = 0.0
    f_1 = 0.0
    f_2 = 0.0
    for element, value in material.element_ratios().iteritems():
        # sum up average atom density
        average_density += value*ATOMIC_MASS[element]*_constants.u
        total_atomic_ammounts += value
        # sum up average scattering factor
        scattering_factor = get_scattering_factor(element, photon_energy)
#        f_1 += value*scattering_factor[0]
#        f_2 += value*scattering_factor[1]
        f_1 += value*_pylab.real(scattering_factor)
        f_2 += value*_pylab.imag(scattering_factor)

    average_density /= total_atomic_ammounts
    f_1 /= total_atomic_ammounts
    f_2 /= total_atomic_ammounts

    #n0 = material.material_density()/average_density
    refractive_index = material.material_density()/average_density
    #return [refractive_index*f1,n0*f2,n0]
    if complex_scattering_factor:
        return refractive_index*f_1 + 1.0j*refractive_index*f_2
    else:
        return refractive_index*f_1

def get_attenuation_length(photon_energy, material):
    """Returns the attenuation length for the material in meters"""
    average_density = 0.0
    total_atomic_ammounts = 0.0
    f_2 = 0.0
    for element, value in material.element_ratios().iteritems():
        # sum up average atom density
        average_density += value*ATOMIC_MASS[element]*_constants.u
        total_atomic_ammounts += value
        # sum up average scattering factor
        scattering_factor = get_scattering_factor(element, photon_energy)
        f_2 += value*_pylab.imag(scattering_factor)
    average_density /= total_atomic_ammounts
    f_2 /= total_atomic_ammounts

    return 1.0/(2.0*_constants.re*_conversions.ev_to_nm(photon_energy)*
                1e-9*f_2*material.material_density()/average_density)

def get_index_of_refraction(photon_energy, material):
    """Returns the refractive index of the material"""
    average_density = 0.
    total_atomic_ammounts = 0.
    f_1 = 0.
    f_2 = 0.
    for element, value in material.element_ratios().iteritems():
        average_density += value*ATOMIC_MASS[element]*_constants.u
        total_atomic_ammounts += value
        # sum up average scattering factor
        scattering_factor = get_scattering_factor(element, photon_energy)
        f_1 += value*_pylab.real(scattering_factor)
        f_2 += value*_pylab.imag(scattering_factor)
    average_density /= total_atomic_ammounts
    f_1 /= total_atomic_ammounts
    f_2 /= total_atomic_ammounts

    refractive_index = material.material_density()/average_density

    return 1. - refractive_index*_constants.re*_conversions.ev_to_m(photon_energy)**2/(2.*_pylab.pi)*(f_1+1.j*f_2)

def size_to_nyquist_angle(size, wavelength):
    """Takes the size (diameter) in nm and returns the angle of a nyquist pixel"""
    return wavelength/size

# class Atom5G(object):
#     """This class is used as a container for the CCP4 scattering factors.
#     They are 2D scattering factors using the 5 gaussian model at the Cu K-alpha
#     wavelength (1.5418 A)."""
#     def __init__(self, element):
#         self.element = element
#         self.coefs_a = None
#         self.coefs_b = None
#         self.coefs_c = None

def atomic_scattering_factor(atom, scattering_vector, b_factor=0.):
    """Evaluate the scattering factor. s should be given in 1/A"""
    return ((atom["a"][0]*_pylab.exp(-atom["b"][0]*scattering_vector**2) +
             atom["a"][1]*_pylab.exp(-atom["b"][1]*scattering_vector**2) +
             atom["a"][2]*_pylab.exp(-atom["b"][2]*scattering_vector**2) +
             atom["a"][3]*_pylab.exp(-atom["b"][3]*scattering_vector**2)) *
            _pylab.exp(-b_factor*scattering_vector**2/4.) + atom["c"])

def read_atomsf():
    """Read pickled element properties"""
    file_name = '%s/Work/Python/Resources/atomsf.dat' % (_os.path.expanduser("~"))
    with open(file_name, "r") as _atomsf_file:
        atomsf_data = _pickle.load(_atomsf_file)
    return atomsf_data

ATOMSF_DATA = read_atomsf()

def write_atomsf(data_dict):
    """Pickle element properties (read by parse_atomsf)"""
    file_name = '%s/Work/Python/Resources/atomsf.dat' % (_os.path.expanduser("~"))
    with open(file_name, "w") as _atomsf_file:
        _pickle.dump(data_dict, _atomsf_file)

def parse_atomsf(file_path='atomsf.lib'):
    """Read atomsf.lib libraray"""
    file_handle = open(file_path)
    file_lines = file_handle.readlines()
    data_lines = [line for line in file_lines if line[:2] != 'AD']

    data_dict = {}

    for i in range(len(data_lines)/5):
        element = data_lines[i*5].split()[0]
        #atom_cont = Atom5G(element)
        atom_cont = {}

        variables = _pylab.float32(data_lines[i*5+1].split())
        atom_cont["weight"] = variables[0]
        atom_cont["number_of_electrons"] = variables[1]
        atom_cont["c"] = variables[2]

        variables = _pylab.float32(data_lines[i*5+2].split())
        atom_cont["a"] = variables

        variables = _pylab.float32(data_lines[i*5+3].split())
        atom_cont["b"] = variables

        variables = _pylab.float32(data_lines[i*5+4].split())
        atom_cont["cu_f"] = complex(variables[0], variables[1])
        atom_cont["mo_f"] = complex(variables[2], variables[3])

        data_dict[element] = atom_cont

    return data_dict

# def plot_stuff():
#     distance = _pylab.arange(50, 200, 0.1)
#     pixel_size = 0.015
#     pixels_radially = 2048
#     wavelength = 5.7
#     gaps = _pylab.array([1.0, 2.0, 3.0, 4.0, 3.0*2.0])
#     particle_size = 500.0
#     missing_limit = 2.8
#     binning = 4
#     fig = _pylab.figure(1)
#     fig.clear()

#     missing_data_plot = fig.add_subplot(411)
#     maximum_resolution_plot = fig.add_subplot(412)
#     combined_plot = fig.add_subplot(413)
#     sampling_plot = fig.add_subplot(414)

#     for g in gaps:
#         missing_data_plot.plot(distance, g/distance/size_to_nyquist_angle(particle_size, wavelength),
#                                label="%g mm gap" % g)
#     missing_data_plot.plot([distance[0], distance[-1]], [missing_limit, missing_limit], color='black')
#     missing_data_plot.legend()
#     missing_data_plot.set_ylabel("Missing nyquist pixels")

#     maximum_resolution_plot.plot(distance, wavelength/2.0/(pixels_radially*pixel_size/distance))
#     maximum_resolution_plot.set_ylabel("Maximum resolution [nm]")
#     maximum_resolution_plot.set_xlabel("Detector distance [mm]")

#     def resolution(distance):
#         return wavelength/2.0/(pixels_radially*pixel_size/distance)
#     for g in gaps:
#         combined_plot.plot(resolution(distance), g/distance/size_to_nyquist_angle(particle_size, wavelength),
#                            label="%g mm gap" % g)
#     combined_plot.plot([resolution(distance)[0], resolution(distance)[-1]],
#                        [missing_limit, missing_limit], color='black')
#     combined_plot.legend()
#     combined_plot.set_ylabel("Missing nyquist pixels")
#     combined_plot.set_xlabel("Resolution at edge [nm]")

#     sampling_plot.plot(distance, size_to_nyquist_angle(particle_size, wavelength)/(pixel_size*binning/distance))
#     sampling_plot.set_ylabel("Sampling ratio")
#     sampling_plot.set_xlabel("Detector distance [mm]")

#     _pylab.show()

# def calculate_pattern():
#     photon_energy = 540

#     virus_size = 275.0e-9
#     virus_total_scattering_factor = (2.0*virus_size)**3*get_scattering_power(photon_energy, material_virus)
#     virus_total_cross_section = _constants.re**2*virus_total_scattering_factor**2

#     water_size = 1500.0e-9
#     water_total_scattering_factor = (2.0*water_size)**3*get_scattering_power(photon_energy, material_water)
#     water_total_cross_section = _constants.re**2*water_total_scattering_factor**2

#     print "Virus cross section = ", virus_total_cross_section
#     print "Water cross section = ", water_total_cross_section
#     print "Ratio = ", virus_total_cross_section/water_total_cross_section

#     # scattering from ball

#     # wavelength = _conversions.ev_to_nm(photon_energy)
#     # q_x = _pylab.arange(-0.075*512/740/wavelength,0.075*512/740/wavelength,0.075/740/wavelength)
#     # q_y = _pylab.arange(-0.075*512/740/wavelength,0.075*512/740/wavelength,0.075/740/wavelength)

#     # q_x_grid, q_y_grid = _pylab.meshgrid(q_x,q_y)
#     # q = _pylab.sqrt(q_x_grid**2+q_y_grid**2)
#     # s = _pylab.float128(2.0*_pylab.pi*virus_size*q*0.01)
#     # scattering = (_pylab.sin(s) - s*_pylab.cos(s))/3.0/s**3
#     # #scattering = (_pylab.sin(s) - s)/3.0/s**3
#     # _pylab.clf()
#     # _pylab.imshow(_pylab.float64(scattering))

#     fov = 16000.0e-9 #m
#     fs_n_of_pixels = 1024
#     fs_pixel_size = fov/fs_n_of_pixels
#     # scattering_factor_virus = _pylab.zeros((fs_n_of_pixels,fs_n_of_pixels,fs_n_of_pixels))
#     # scattering_factor_virus_water = _pylab.zeros((fs_n_of_pixels,fs_n_of_pixels,fs_n_of_pixels))

#     virus_scattering_factor = get_scattering_power(photon_energy, material_virus)*fs_pixel_size**3
#     water_scattering_factor = get_scattering_power(photon_energy, material_water)*fs_pixel_size**3

#     # for x,x_i in zip((_pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
#     #     #print x_i
#     #     for y,y_i in zip((_pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
#     #         for z,z_i in zip((_pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
#     #             if x**2+y**2+z**2 < virus_size**2:
#     #                 scattering_factor_virus[x_i,y_i,z_i] = virus_scattering_factor
#     #                 scattering_factor_virus_water[x_i,y_i,z_i] = virus_scattering_factor
#     #             elif x**2 + y**2 < water_size**2:
#     #                 scattering_factor_virus_water[x_i,y_i,z_i] = water_scattering_factor
#     #                 scattering_factor_virus[x_i,y_i,z_i] = 0.0
#     #             else:
#     #                 scattering_factor_virus_water[x_i,y_i,z_i] = 0.0


#     # projected_SCATTERING_FACTORS_virus = _pylab.sum(scattering_factor_virus,axis=1)
#     # projected_SCATTERING_FACTORS_virus_water = _pylab.sum(scattering_factor_virus_water,axis=1)

#     detector_size = 1024
#     oversampling = 1
#     input_intensities = 1e13 #photons
#     beam_width = 1500e-9

#     _center = detector_size*oversampling/2

#     scattering_factor_virus = _pylab.zeros((fs_n_of_pixels, fs_n_of_pixels))
#     scattering_factor_virus_water = _pylab.zeros((fs_n_of_pixels, fs_n_of_pixels))

#     tmp_water_size = water_size
#     for y, y_i in zip((_pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size, range(fs_n_of_pixels)):
#         if y_i%50 == 0: print y_i
#         tmp_water_size = tmp_water_size*(1.0+0.05*(-1.0+2.0*_pylab.rand()))
#         for x, x_i in zip((_pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size, range(fs_n_of_pixels)):
#             if x**2 + y**2 < virus_size**2:
#                 thickness = 2.0*_pylab.sqrt(virus_size**2 - x**2 - y**2)/fs_pixel_size
#                 scattering_factor_virus[x_i, y_i] = virus_scattering_factor*thickness
#                 scattering_factor_virus_water[x_i, y_i] = virus_scattering_factor*thickness
#                 #water_thickness = (2.0*_pylab.sqrt(water_size**2 - x**2)/fs_pixel_size - thickness)*(1.0+0.05*_pylab.rand())
#                 water_thickness = (2.0*_pylab.sqrt(tmp_water_size**2 - x**2)/fs_pixel_size - thickness)
#                 scattering_factor_virus_water[x_i, y_i] += water_scattering_factor*water_thickness
#             elif x**2 < tmp_water_size**2:
#                 #thickness = 2.0*_pylab.sqrt(water_size**2 - x**2)/fs_pixel_size*(1.0+0.05*_pylab.rand())
#                 thickness = 2.0*_pylab.sqrt(tmp_water_size**2 - x**2)/fs_pixel_size
#                 scattering_factor_virus_water[x_i, y_i] = water_scattering_factor*thickness
#                 scattering_factor_virus[x_i, y_i] = 0.0
#             else:
#                 scattering_factor_virus_water[x_i, y_i] = 0.0
#             r = _pylab.sqrt(x**2+y**2)
#             scattering_factor_virus[x_i, y_i] *= _pylab.exp(-r**2/2.0/beam_width**2)
#             scattering_factor_virus_water[x_i, y_i] *= _pylab.exp(-r**2/2.0/beam_width**2)

#     amplitudes_virus = _pylab.fftshift(abs(_pylab.fftn(scattering_factor_virus,
#                                                        [oversampling*detector_size]*2)))[_center-detector_size/2:_center+detector_size/2,
#                                                                                          _center-detector_size/2:_center+detector_size/2]
#     amplitudes_virus_water = _pylab.fftshift(abs(_pylab.fftn(scattering_factor_virus_water,
#                                                              [oversampling*detector_size]*2)))[_center-detector_size/2:_center+detector_size/2,
#                                                                                                _center-detector_size/2:_center+detector_size/2]

#     intensities_virus = input_intensities*_constants.re**2*amplitudes_virus**2
#     intensities_virus_water = input_intensities*_constants.re**2*amplitudes_virus_water**2
