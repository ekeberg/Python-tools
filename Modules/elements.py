import pylab as _pylab
import pickle as _pickle
import conversions as _conversions
import constants as _constants
import sys as _sys
import os as _os

_elements_file = open('%s/Python/Resources/elements.dat' % (_os.path.expanduser("~")))
atomic_mass,scattering_factors = _pickle.load(_elements_file)
_elements_file.close()
elements = atomic_mass.keys()

class ChemicalComposition(object):
    """This class contains a set of relative abundances of materials."""
    def __init__(self, **kwargs):
        self._elements = kwargs
        bad_elements = []
        for key in self._elements:
            if key not in atomic_mass.keys():
                bad_elements.append(key)
                raise IndexError("%s is not an element." % key)
        for element in bad_elements:
            self._elements.pop(b)

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
            mass_ratios[element] = self._elements[element]*atomic_mass[element]
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
        return self._chemical_composition.element_mass_ratio()
        
    def element_mass_ratio_sum(self):
        """Returns the sum of all relative mass ratios returned bu element_mass_ratios()"""
        return self._chemical_composition.element_mass_ratio_sum()

    def material_density(self):
        return self._density
           
materials = {"protein" : Material(1350,H=86,C=52,N=13,O=15,P=0,S=3),
             "water" : Material(1000,O=1,H=2),
             "virus" : Material(1455,H=72.43,C=47.52,N=13.55,O=17.17,P=1.11,S=0.7),
             "cell" : Material(1000,H=23,C=3,N=1,O=10,P=0,S=1)}

def get_scattering_factor(element,photon_energy):
    """
    get the scattering factor for an element through linear interpolation. Photon energy is given in eV.
    """
    f1 = _pylab.interp(photon_energy,scattering_factors[element][:,0],scattering_factors[element][:,1])
    f2 = _pylab.interp(photon_energy,scattering_factors[element][:,0],scattering_factors[element][:,2])
#return [f1,f2] 
    return f1+f2*1.j

def get_scattering_power(photon_energy,material,complex=False):
    """Returns the scattering factor for a volume of 1 m^3 of the material"""
    average_density = 0.0
    total_atomic_ammounts = 0.0
    f1 = 0.0; f2 = 0.0
    for element,value in material.element_ratios().iteritems():
        # sum up average atom density
        average_density += value*atomic_mass[element]*_constants.u
        total_atomic_ammounts += value
        # sum up average scattering factor
        f = get_scattering_factor(element,photon_energy)
#        f1 += value*f[0]
#        f2 += value*f[1]
        f1 += value*_pylab.real(f)
        f2 += value*_pylab.imag(f)

    average_density /= total_atomic_ammounts
    f1 /= total_atomic_ammounts
    f2 /= total_atomic_ammounts

    n0 = material.material_density()/average_density
    #return [n0*f1,n0*f2,n0]
    if complex:
        return n0*f1 + 1.0j*n0*f2
    else:
        return n0*f1

def get_attenuation_length(photon_energy,material):
    """Returns the attenuation length for the material in meters"""
    average_density = 0.0
    total_atomic_ammounts = 0.0
    f1 = 0.0; f2 = 0.0
    for element,value in material.element_ratios().iteritems():
        # sum up average atom density
        average_density += value*atomic_mass[element]*_constants.u
        total_atomic_ammounts += value
        # sum up average scattering factor
        f = get_scattering_factor(element,photon_energy)
#        f1 += value*f[0]
#        f2 += value*f[1]
        f1 += value*_pylab.real(f)
        f2 += value*_pylab.imag(f)
    average_density /= total_atomic_ammounts
    f1 /= total_atomic_ammounts
    f2 /= total_atomic_ammounts

    n0 = material.material_density()/average_density

    #print 2.0*_constants.re*_conversions.ev_to_nm(photon_energy)*1e-9*f2*material[0]/average_density
    return 1.0/(2.0*_constants.re*_conversions.ev_to_nm(photon_energy)*1e-9*f2*material.material_density()/average_density)

def size_to_nyquist_angle(size, wavelength):
    """Takes the size (diameter) in nm and returns the angle of a nyquist pixel"""
    return wavelength/size

class Atom5G(object):
    """This class is used as a container for the CCP4 scattering factors.
    They are 2D scattering factors using the 5 gaussian model at the Cu K-alpha
    wavelength (1.5418 A)."""
    def __init__(self, element):
        self.element = element

    def scattering_factor(self, s, b_factor = 0.):
        """Evaluate the scattering factor. s should be given in 1/A"""
        return (self.a[0]*_pylab.exp(-self.b[0]*s**2) +
                self.a[1]*_pylab.exp(-self.b[1]*s**2) +
                self.a[2]*_pylab.exp(-self.b[2]*s**2) +
                self.a[3]*_pylab.exp(-self.b[3]*s**2)) * _pylab.exp(-b_factor*s**2/4.) + self.c
    
_atomsf_file = open('%s/Python/Resources/atomsf.dat' % (_os.path.expanduser("~")))
atomsf_data = _pickle.load(_atomsf_file)
_atomsf_file.close()


def parse_atomsf(file_path='atomsf.lib'):
    file_handle = open(file_path)
    file_lines = file_handle.readlines()
    data_lines = [line for line in file_lines if line[:2] != 'AD']

    data_dict = {}

    for i in range(len(data_lines)/5):
        element = data_lines[i*5].split()[0]
        atom_cont = Atom5G(element)
        
        variables = _pylab.float32(data_lines[i*5+1].split())
        atom_cont.weight = variables[0]
        atom_cont.number_of_electrons = variables[1]
        atom_cont.c = variables[2]

        variables = _pylab.float32(data_lines[i*5+2].split())
        atom_cont.a = variables

        variables = _pylab.float32(data_lines[i*5+3].split())
        atom_cont.b = variables

        variables = _pylab.float32(data_lines[i*5+4].split())
        atom_cont.cu_f = complex(variables[0], variables[1])
        atom_cont.mo_f = complex(variables[2], variables[3])

        data_dict[element] = atom_cont

    return data_dict

def plot_stuff():
    distance = _pylab.arange(50,200,0.1)
    pixel_size = 0.015
    pixels_radially = 2048
    wavelength = 5.7
    gaps = _pylab.array([1.0,2.0,3.0,4.0,3.0*2.0])
    particle_size = 500.0
    missing_limit = 2.8
    binning = 4
    fig = _pylab.figure(1)
    fig.clear()
    
    missing_data_plot = fig.add_subplot(411)
    maximum_resolution_plot = fig.add_subplot(412)
    combined_plot = fig.add_subplot(413)
    sampling_plot = fig.add_subplot(414)
    
    for g in gaps:
        missing_data_plot.plot(distance, g/distance/size_to_nyquist_angle(particle_size, wavelength),
                               label="%g mm gap" % g)
    missing_data_plot.plot([distance[0],distance[-1]],[missing_limit,missing_limit],color='black')
    missing_data_plot.legend()
    missing_data_plot.set_ylabel("Missing nyquist pixels")

    maximum_resolution_plot.plot(distance, wavelength/2.0/(pixels_radially*pixel_size/distance))
    maximum_resolution_plot.set_ylabel("Maximum resolution [nm]")
    maximum_resolution_plot.set_xlabel("Detector distance [mm]")

    def resolution(distance):
        return wavelength/2.0/(pixels_radially*pixel_size/distance)
    for g in gaps:
        combined_plot.plot(resolution(distance), g/distance/size_to_nyquist_angle(particle_size, wavelength),
                           label="%g mm gap" % g)
    combined_plot.plot([resolution(distance)[0],resolution(distance)[-1]],[missing_limit,missing_limit],color='black')
    combined_plot.legend()
    combined_plot.set_ylabel("Missing nyquist pixels")
    combined_plot.set_xlabel("Resolution at edge [nm]")

    sampling_plot.plot(distance,size_to_nyquist_angle(particle_size, wavelength)/(pixel_size*binning/distance))
    sampling_plot.set_ylabel("Sampling ratio")
    sampling_plot.set_xlabel("Detector distance [mm]")

    _pylab.show()

def calculate_pattern():
    photon_energy = 540

    virus_size = 275.0e-9
    virus_total_scattering_factor = (2.0*virus_size)**3*get_scattering_power(photon_energy,material_virus)
    virus_total_cross_section = _constants.re**2*virus_total_scattering_factor**2

    water_size = 1500.0e-9
    water_total_scattering_factor = (2.0*water_size)**3*get_scattering_power(photon_energy,material_water)
    water_total_cross_section = _constants.re**2*water_total_scattering_factor**2

    print "Virus cross section = ", virus_total_cross_section
    print "Water cross section = ", water_total_cross_section
    print "Ratio = ", virus_total_cross_section/water_total_cross_section

    # scattering from ball

    # wavelength = _conversions.ev_to_nm(photon_energy)
    # q_x = _pylab.arange(-0.075*512/740/wavelength,0.075*512/740/wavelength,0.075/740/wavelength)
    # q_y = _pylab.arange(-0.075*512/740/wavelength,0.075*512/740/wavelength,0.075/740/wavelength)

    # q_x_grid, q_y_grid = _pylab.meshgrid(q_x,q_y)
    # q = _pylab.sqrt(q_x_grid**2+q_y_grid**2)
    # s = _pylab.float128(2.0*_pylab.pi*virus_size*q*0.01)
    # scattering = (_pylab.sin(s) - s*_pylab.cos(s))/3.0/s**3
    # #scattering = (_pylab.sin(s) - s)/3.0/s**3
    # _pylab.clf()
    # _pylab.imshow(_pylab.float64(scattering))

    fov = 16000.0e-9 #m
    fs_n_of_pixels = 1024
    fs_pixel_size = fov/fs_n_of_pixels
    # scattering_factor_virus = _pylab.zeros((fs_n_of_pixels,fs_n_of_pixels,fs_n_of_pixels))
    # scattering_factor_virus_water = _pylab.zeros((fs_n_of_pixels,fs_n_of_pixels,fs_n_of_pixels))

    virus_scattering_factor = get_scattering_power(photon_energy,material_virus)*fs_pixel_size**3
    water_scattering_factor = get_scattering_power(photon_energy,material_water)*fs_pixel_size**3

    # for x,x_i in zip((_pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
    #     #print x_i
    #     for y,y_i in zip((_pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
    #         for z,z_i in zip((_pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
    #             if x**2+y**2+z**2 < virus_size**2:
    #                 scattering_factor_virus[x_i,y_i,z_i] = virus_scattering_factor
    #                 scattering_factor_virus_water[x_i,y_i,z_i] = virus_scattering_factor
    #             elif x**2 + y**2 < water_size**2:
    #                 scattering_factor_virus_water[x_i,y_i,z_i] = water_scattering_factor
    #                 scattering_factor_virus[x_i,y_i,z_i] = 0.0
    #             else:
    #                 scattering_factor_virus_water[x_i,y_i,z_i] = 0.0


    # projected_scattering_factors_virus = _pylab.sum(scattering_factor_virus,axis=1)
    # projected_scattering_factors_virus_water = _pylab.sum(scattering_factor_virus_water,axis=1)

    detector_size = 1024
    oversampling = 1
    input_intensities = 1e13 #photons
    beam_width = 1500e-9

    _center = detector_size*oversampling/2

    # amplitudes = _pylab.fftshift(abs(_pylab.fftn(projected_scattering_factors,[oversampling*detector_size]*2)))[_center-detector_size/2:_center+detector_size/2,_center-detector_size/2:_center+detector_size/2]

    # intensities = input_intensities*_constants.re**2*amplitudes**2


    scattering_factor_virus = _pylab.zeros((fs_n_of_pixels,fs_n_of_pixels))
    scattering_factor_virus_water = _pylab.zeros((fs_n_of_pixels,fs_n_of_pixels))

    tmp_water_size = water_size
    for y,y_i in zip((_pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
        if y_i%50 == 0: print y_i
        tmp_water_size = tmp_water_size*(1.0+0.05*(-1.0+2.0*_pylab.rand()))
        for x,x_i in zip((_pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
            if x**2 + y**2 < virus_size**2:
                thickness = 2.0*_pylab.sqrt(virus_size**2 - x**2 - y**2)/fs_pixel_size
                scattering_factor_virus[x_i,y_i] = virus_scattering_factor*thickness
                scattering_factor_virus_water[x_i,y_i] = virus_scattering_factor*thickness
                #water_thickness = (2.0*_pylab.sqrt(water_size**2 - x**2)/fs_pixel_size - thickness)*(1.0+0.05*_pylab.rand())
                water_thickness = (2.0*_pylab.sqrt(tmp_water_size**2 - x**2)/fs_pixel_size - thickness)
                scattering_factor_virus_water[x_i,y_i] += water_scattering_factor*water_thickness
            elif x**2 < tmp_water_size**2:
                #thickness = 2.0*_pylab.sqrt(water_size**2 - x**2)/fs_pixel_size*(1.0+0.05*_pylab.rand())
                thickness = 2.0*_pylab.sqrt(tmp_water_size**2 - x**2)/fs_pixel_size
                scattering_factor_virus_water[x_i,y_i] = water_scattering_factor*thickness
                scattering_factor_virus[x_i,y_i] = 0.0
            else:
                scattering_factor_virus_water[x_i,y_i] = 0.0
            r = _pylab.sqrt(x**2+y**2)
            scattering_factor_virus[x_i,y_i] *= _pylab.exp(-r**2/2.0/beam_width**2)
            scattering_factor_virus_water[x_i,y_i] *= _pylab.exp(-r**2/2.0/beam_width**2)

    amplitudes_virus = _pylab.fftshift(abs(_pylab.fftn(scattering_factor_virus,[oversampling*detector_size]*2)))[_center-detector_size/2:_center+detector_size/2,_center-detector_size/2:_center+detector_size/2]
    amplitudes_virus_water = _pylab.fftshift(abs(_pylab.fftn(scattering_factor_virus_water,[oversampling*detector_size]*2)))[_center-detector_size/2:_center+detector_size/2,_center-detector_size/2:_center+detector_size/2]

    intensities_virus = input_intensities*_constants.re**2*amplitudes_virus**2
    intensities_virus_water = input_intensities*_constants.re**2*amplitudes_virus_water**2
