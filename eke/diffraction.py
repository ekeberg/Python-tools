import numpy as _numpy
import nfft as _nfft
from . import rotations as _rotations
from . import constants as _constants
from . import conversions as _conversions

def shannon_angle(size, wavelength):
    """Takes the size (diameter) in nm and returns the angle of a nyquist pixel"""
    return wavelength/size

def ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size):
    pixels_to_im = pixel_size/detector_distance/wavelength
    x0_pixels = _numpy.arange(image_shape[0]) - image_shape[0]/2 + 0.5
    x1_pixels = _numpy.arange(image_shape[1]) - image_shape[1]/2 + 0.5
    x0 = x0_pixels*pixels_to_im
    x1 = x1_pixels*pixels_to_im
    r_pixels = _numpy.sqrt(x0_pixels[:, _numpy.newaxis]**2 + x1_pixels[_numpy.newaxis, :]**2)
    theta = _numpy.arctan(r_pixels*pixel_size / detector_distance)
    x2 = 1./wavelength*(1 - _numpy.cos(theta))

    x0_2d, x1_2d = _numpy.meshgrid(x0, x1, indexing="ij")
    output_coordinates = _numpy.zeros((_numpy.prod(image_shape), 3))
    output_coordinates[:, 0] = x0_2d.flatten()
    output_coordinates[:, 1] = x1_2d.flatten()
    output_coordinates[:, 2] = x2.flatten()
    return output_coordinates


def calculate_diffraction(scattering_factor_density, density_pixel_size, rotation,
                          image_shape, wavelength, detector_distance, pixel_size):
    base_coordinates = ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size)
    rotated_coordinates = _rotations.rotate_array(rotation, base_coordinates)
    largest_distance_in_fourier_space = 1./(2.*density_pixel_size)
    scaled_coordinates = rotated_coordinates * 0.5 / largest_distance_in_fourier_space
    diffraction = _nfft.nfft(scattering_factor_density, scaled_coordinates)
    return diffraction.reshape(image_shape)


def klein_nishina(energy, scattering_angle, polarization_angle):
    """The cross section of a free electron. Energy given in eV, angles given in radians."""
    energy_in_joules = _conversions.ev_to_J(energy)
    relative_energy_change = 1. / (1. + (energy_in_joules / (_constants.me*_constants.c**2)) *
                                   (1. - _numpy.cos(scattering_angle)))
    return ((_constants.re**2 * relative_energy_change**2)/2. *
            (relative_energy_change + 1./relative_energy_change -
             _numpy.sin(scattering_angle)**2*_numpy.cos(polarization_angle)**2))
