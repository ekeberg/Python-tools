import numpy as _numpy
from . import rotmodule as _rotmodule
from . import constants as _constants
from . import conversions as _conversions

def shannon_angle(size, wavelength):
    """Takes the size (diameter) in nm and returns the angle of a nyquist pixel"""
    return wavelength/size

def ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size):
    x0_pixels_1d = _numpy.arange(image_shape[0]) - image_shape[0]/2. + 0.5
    x1_pixels_1d = _numpy.arange(image_shape[1]) - image_shape[1]/2. + 0.5
    x1_pixels, x0_pixels = _numpy.meshgrid(x0_pixels_1d, x1_pixels_1d, indexing="ij")
    x0_meters = x0_pixels*pixel_size
    x1_meters = x1_pixels*pixel_size
    #radius_meters = x0_meters[:, _numpy.newaxis]**2 + x1_meters[_numpy.newaxis, :]**2
    radius_meters = _numpy.sqrt(x0_meters**2 + x1_meters**2)

    scattering_angle = _numpy.arctan(radius_meters / detector_distance)
    x2 = -1./wavelength*(1. - _numpy.cos(scattering_angle))
    radius_fourier = _numpy.sqrt(1./wavelength**2 - (1./wavelength - abs(x2))**2)

    x0 = x0_meters * radius_fourier / radius_meters
    x1 = x1_meters * radius_fourier / radius_meters

    output_coordinates = _numpy.zeros((_numpy.prod(image_shape), 3))
    output_coordinates[:, 0] = x0.flatten() #0
    output_coordinates[:, 1] = x1.flatten() #1
    output_coordinates[:, 2] = x2.flatten()
    return output_coordinates

def calculate_diffraction(scattering_factor_density, density_pixel_size, rotation,
                          image_shape, wavelength, detector_distance, pixel_size):
    import nfft as _nfft
    base_coordinates = ewald_coordinates(
        image_shape, wavelength,
        detector_distance, pixel_size)
    rotated_coordinates = _rotmodule.rotate_array(rotation, base_coordinates)
    diffraction = _nfft.nfft(scattering_factor_density, density_pixel_size,
                             rotated_coordinates)
    return diffraction.reshape(image_shape)

def sphere_diffraction(diameter, material, number_of_fringes=10):
    raise NotImplementedError("half-written function. Should complete it later.")
    pixel_size = 75e-6
    detector_shape = (1024, 1024)
    detector_distance = 730e-3
    coords = [_numpy.arange(this_detector_shape) - this_detector_shape/2.+0.5]
    r = _numpy.sqrt(coords[0][:, _numpy.newaxis]**2 + coords[1][_numpy.newaxis, :]**2)
    



def klein_nishina(energy, scattering_angle, polarization_angle):
    """The cross section of a free electron. Energy given in eV, angles given in radians."""
    energy_in_joules = _conversions.ev_to_J(energy)
    relative_energy_change = 1. / (1. + (energy_in_joules / (_constants.me*_constants.c**2)) *
                                   (1. - _numpy.cos(scattering_angle)))
    return ((_constants.re**2 * relative_energy_change**2)/2. *
            (relative_energy_change + 1./relative_energy_change -
             _numpy.sin(scattering_angle)**2*_numpy.cos(polarization_angle)**2))

