import pylab
import pickle
import conversions
import constants
import sys
import os

_elements_file = open('%s/Python/Resources/elements.dat' % (os.path.expanduser("~")))
atomic_mass,scattering_factors = pickle.load(_elements_file)
_elements_file.close()
elements = atomic_mass.keys()

class Material:
    def __init__(self,density,**kwargs):
        self.density = density
        self.elements = kwargs
        bad_elements = []
        for key in self.elements:
            if key not in atomic_mass.keys():
                #self.elements.append((key,kwargs[key]))
                #self.elements.pop(key)
                bad_elements.append(key)
                print "%s is not an element. Ignoring it." % key
            #else:
        for b in bad_elements:
            self.elements.pop(b)
        
            
materials = {"protein" : Material(1350,H=86,C=52,N=13,O=15,P=0,S=3),
             "water" : Material(1000,O=1,H=2),
             "virus" : Material(1455,H=72.43,C=47.52,N=13.55,O=17.17,P=1.11,S=0.7),
             "cell" : Material(1000,H=23,C=3,N=1,O=10,P=0,S=1)}

def get_scattering_factor(element,photon_energy):
    """
    get the scattering factor for an element through linear interpolation.
    """
    f1 = pylab.interp(photon_energy,scattering_factors[element][:,0],scattering_factors[element][:,1])
    f2 = pylab.interp(photon_energy,scattering_factors[element][:,0],scattering_factors[element][:,2])
    return [f1,f2] 

def get_scattering_power(photon_energy,material,complex=False):
    """Returns the scattering factor for a volume of 1 m^3 of the material"""
    average_density = 0.0
    total_atomic_ammounts = 0.0
    f1 = 0.0; f2 = 0.0
    for element,value in material.elements.iteritems():
        # sum up average atom density
        average_density += value*atomic_mass[element]*constants.u
        total_atomic_ammounts += value
        # sum up average scattering factor
        f = get_scattering_factor(element,photon_energy)
        f1 += value*f[0]
        f2 += value*f[1]

    average_density /= total_atomic_ammounts
    f1 /= total_atomic_ammounts
    f2 /= total_atomic_ammounts

    n0 = material.density/average_density
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
    for element,value in material.elements.iteritems():
        # sum up average atom density
        average_density += value*atomic_mass[element]*constants.u
        total_atomic_ammounts += value
        # sum up average scattering factor
        f = get_scattering_factor(element,photon_energy)
        f1 += value*f[0]
        f2 += value*f[1]
    average_density /= total_atomic_ammounts
    f1 /= total_atomic_ammounts
    f2 /= total_atomic_ammounts

    n0 = material.density/average_density

    #print 2.0*constants.re*conversions.ev_to_nm(photon_energy)*1e-9*f2*material[0]/average_density
    return 1.0/(2.0*constants.re*conversions.ev_to_nm(photon_energy)*1e-9*f2*material.density/average_density)


def calculate_pattern():
    photon_energy = 540

    virus_size = 275.0e-9
    virus_total_scattering_factor = (2.0*virus_size)**3*get_scattering_power(photon_energy,material_virus)
    virus_total_cross_section = constants.re**2*virus_total_scattering_factor**2

    water_size = 1500.0e-9
    water_total_scattering_factor = (2.0*water_size)**3*get_scattering_power(photon_energy,material_water)
    water_total_cross_section = constants.re**2*water_total_scattering_factor**2

    print "Virus cross section = ", virus_total_cross_section
    print "Water cross section = ", water_total_cross_section
    print "Ratio = ", virus_total_cross_section/water_total_cross_section

    # scattering from ball

    # wavelength = conversions.ev_to_nm(photon_energy)
    # q_x = pylab.arange(-0.075*512/740/wavelength,0.075*512/740/wavelength,0.075/740/wavelength)
    # q_y = pylab.arange(-0.075*512/740/wavelength,0.075*512/740/wavelength,0.075/740/wavelength)

    # q_x_grid, q_y_grid = pylab.meshgrid(q_x,q_y)
    # q = pylab.sqrt(q_x_grid**2+q_y_grid**2)
    # s = pylab.float128(2.0*pylab.pi*virus_size*q*0.01)
    # scattering = (pylab.sin(s) - s*pylab.cos(s))/3.0/s**3
    # #scattering = (pylab.sin(s) - s)/3.0/s**3
    # pylab.clf()
    # pylab.imshow(pylab.float64(scattering))

    fov = 16000.0e-9 #m
    fs_n_of_pixels = 1024
    fs_pixel_size = fov/fs_n_of_pixels
    # scattering_factor_virus = pylab.zeros((fs_n_of_pixels,fs_n_of_pixels,fs_n_of_pixels))
    # scattering_factor_virus_water = pylab.zeros((fs_n_of_pixels,fs_n_of_pixels,fs_n_of_pixels))

    virus_scattering_factor = get_scattering_power(photon_energy,material_virus)*fs_pixel_size**3
    water_scattering_factor = get_scattering_power(photon_energy,material_water)*fs_pixel_size**3

    # for x,x_i in zip((pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
    #     #print x_i
    #     for y,y_i in zip((pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
    #         for z,z_i in zip((pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
    #             if x**2+y**2+z**2 < virus_size**2:
    #                 scattering_factor_virus[x_i,y_i,z_i] = virus_scattering_factor
    #                 scattering_factor_virus_water[x_i,y_i,z_i] = virus_scattering_factor
    #             elif x**2 + y**2 < water_size**2:
    #                 scattering_factor_virus_water[x_i,y_i,z_i] = water_scattering_factor
    #                 scattering_factor_virus[x_i,y_i,z_i] = 0.0
    #             else:
    #                 scattering_factor_virus_water[x_i,y_i,z_i] = 0.0


    # projected_scattering_factors_virus = pylab.sum(scattering_factor_virus,axis=1)
    # projected_scattering_factors_virus_water = pylab.sum(scattering_factor_virus_water,axis=1)

    detector_size = 1024
    oversampling = 1
    input_intensities = 1e13 #photons
    beam_width = 1500e-9

    _center = detector_size*oversampling/2

    # amplitudes = pylab.fftshift(abs(pylab.fftn(projected_scattering_factors,[oversampling*detector_size]*2)))[_center-detector_size/2:_center+detector_size/2,_center-detector_size/2:_center+detector_size/2]

    # intensities = input_intensities*constants.re**2*amplitudes**2


    scattering_factor_virus = pylab.zeros((fs_n_of_pixels,fs_n_of_pixels))
    scattering_factor_virus_water = pylab.zeros((fs_n_of_pixels,fs_n_of_pixels))

    tmp_water_size = water_size
    for y,y_i in zip((pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
        if y_i%50 == 0: print y_i
        tmp_water_size = tmp_water_size*(1.0+0.05*(-1.0+2.0*pylab.rand()))
        for x,x_i in zip((pylab.arange(fs_n_of_pixels)-fs_n_of_pixels/2+0.5)*fs_pixel_size,range(fs_n_of_pixels)):
            if x**2 + y**2 < virus_size**2:
                thickness = 2.0*pylab.sqrt(virus_size**2 - x**2 - y**2)/fs_pixel_size
                scattering_factor_virus[x_i,y_i] = virus_scattering_factor*thickness
                scattering_factor_virus_water[x_i,y_i] = virus_scattering_factor*thickness
                #water_thickness = (2.0*pylab.sqrt(water_size**2 - x**2)/fs_pixel_size - thickness)*(1.0+0.05*pylab.rand())
                water_thickness = (2.0*pylab.sqrt(tmp_water_size**2 - x**2)/fs_pixel_size - thickness)
                scattering_factor_virus_water[x_i,y_i] += water_scattering_factor*water_thickness
            elif x**2 < tmp_water_size**2:
                #thickness = 2.0*pylab.sqrt(water_size**2 - x**2)/fs_pixel_size*(1.0+0.05*pylab.rand())
                thickness = 2.0*pylab.sqrt(tmp_water_size**2 - x**2)/fs_pixel_size
                scattering_factor_virus_water[x_i,y_i] = water_scattering_factor*thickness
                scattering_factor_virus[x_i,y_i] = 0.0
            else:
                scattering_factor_virus_water[x_i,y_i] = 0.0
            r = pylab.sqrt(x**2+y**2)
            scattering_factor_virus[x_i,y_i] *= pylab.exp(-r**2/2.0/beam_width**2)
            scattering_factor_virus_water[x_i,y_i] *= pylab.exp(-r**2/2.0/beam_width**2)

    amplitudes_virus = pylab.fftshift(abs(pylab.fftn(scattering_factor_virus,[oversampling*detector_size]*2)))[_center-detector_size/2:_center+detector_size/2,_center-detector_size/2:_center+detector_size/2]
    amplitudes_virus_water = pylab.fftshift(abs(pylab.fftn(scattering_factor_virus_water,[oversampling*detector_size]*2)))[_center-detector_size/2:_center+detector_size/2,_center-detector_size/2:_center+detector_size/2]

    intensities_virus = input_intensities*constants.re**2*amplitudes_virus**2
    intensities_virus_water = input_intensities*constants.re**2*amplitudes_virus_water**2
