"""X-ray material properties. Calculate cross section, attenuation length and more."""
import numpy as _numpy
import pickle as _pickle
import conversions as _conversions
import constants as _constants
import os as _os

_ELEMENTS_FILE = open(_os.path.join(_os.path.split(__file__)[0], "elements.dat"), "r")
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
             "cell" : Material(1000, H=23, C=3, N=1, O=10, P=0, S=1),
             "silicon_nitride": Material(3440, Si=3, N=4)}

def get_scattering_factor(element, photon_energy):
    """
    get the scattering factor for an element through linear interpolation. Photon energy is given in eV.
    """
    f_1 = _numpy.interp(photon_energy, SCATTERING_FACTORS[element][:, 0], SCATTERING_FACTORS[element][:, 1])
    f_2 = _numpy.interp(photon_energy, SCATTERING_FACTORS[element][:, 0], SCATTERING_FACTORS[element][:, 2])
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
        f_1 += value*_numpy.real(scattering_factor)
        f_2 += value*_numpy.imag(scattering_factor)

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
        f_2 += value*_numpy.imag(scattering_factor)
    average_density /= total_atomic_ammounts
    f_2 /= total_atomic_ammounts

    return 1.0/(2.0*_constants.re*_conversions.ev_to_nm(photon_energy)*
                1e-9*f_2*material.material_density()/average_density)

def get_transmission(photon_energy, material, thickness):
    """Return the transmission of the given material of the given thickness."""
    attenuation_length = get_attenuation_length(photon_energy, material)
    return _numpy.exp(-thickness / attenuation_length)

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
        f_1 += value*_numpy.real(scattering_factor)
        f_2 += value*_numpy.imag(scattering_factor)
    average_density /= total_atomic_ammounts
    f_1 /= total_atomic_ammounts
    f_2 /= total_atomic_ammounts

    refractive_index = material.material_density()/average_density

    return 1. - refractive_index*_constants.re*_conversions.ev_to_m(photon_energy)**2/(2.*_numpy.pi)*(f_1+1.j*f_2)

def get_phase_shift(photon_energy, material, distance):
    """Returns the phase shift that that amount of material will cause."""
    index_of_refraction = get_index_of_refraction(photon_energy, material)
    phase_shift_per_period = (1.-_numpy.real(index_of_refraction)) * (2.*_numpy.pi)
    number_of_wavelengths = distance / (_conversions.ev_to_m(photon_energy) *
                                        _numpy.real(index_of_refraction))
    total_phase_shift = number_of_wavelengths*phase_shift_per_period
    return total_phase_shift
    

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
    return ((atom["a"][0]*_numpy.exp(-atom["b"][0]*scattering_vector**2) +
             atom["a"][1]*_numpy.exp(-atom["b"][1]*scattering_vector**2) +
             atom["a"][2]*_numpy.exp(-atom["b"][2]*scattering_vector**2) +
             atom["a"][3]*_numpy.exp(-atom["b"][3]*scattering_vector**2)) *
            _numpy.exp(-b_factor*scattering_vector**2/4.) + atom["c"])

def read_atomsf():
    """Read pickled element properties"""
    file_name = _os.path.join(_os.path.split(__file__)[0], "atomsf.dat")
    with open(file_name, "r") as _atomsf_file:
        atomsf_data = _pickle.load(_atomsf_file)
    return atomsf_data

ATOMSF_DATA = read_atomsf()

def write_atomsf(data_dict):
    """Pickle element properties (read by parse_atomsf)"""
    file_name = _os.path.join(_os.path.split(__file__)[0], "atomsf.dat")
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

        variables = _numpy.float32(data_lines[i*5+1].split())
        atom_cont["weight"] = variables[0]
        atom_cont["number_of_electrons"] = variables[1]
        atom_cont["c"] = variables[2]

        variables = _numpy.float32(data_lines[i*5+2].split())
        atom_cont["a"] = variables

        variables = _numpy.float32(data_lines[i*5+3].split())
        atom_cont["b"] = variables

        variables = _numpy.float32(data_lines[i*5+4].split())
        atom_cont["cu_f"] = complex(variables[0], variables[1])
        atom_cont["mo_f"] = complex(variables[2], variables[3])

        data_dict[element] = atom_cont

    return data_dict

