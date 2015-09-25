import numpy as _numpy
import rotations as _rotations
import nfft as _nfft

def ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size):
    pixels_to_im = pixel_size/detector_distance/wavelength
    x0_pixels = _numpy.arange(image_shape[0]) - image_shape[0]/2 + 0.5
    x1_pixels = _numpy.arange(image_shape[1]) - image_shape[1]/2 + 0.5
    x0 = x0_pixels*pixels_to_im
    x1 = x1_pixels*pixels_to_im
    r_pixels = _numpy.sqrt(x0_pixels[:, _numpy.newaxis]**2 + x1_pixels[_numpy.newaxis, :]**2)
    #r = r_pixels*pixels_to_im
    theta = _numpy.arctan(r_pixels*pixel_size / detector_distance)
    x2 = 1./wavelength*(1 - _numpy.cos(theta))

    x0_2d, x1_2d = _numpy.meshgrid(x0, x1, indexing="ij")
    output_coordinates = _numpy.zeros((_numpy.prod(image_shape), 3))
    output_coordinates[:, 0] = x0_2d.flatten()
    output_coordinates[:, 1] = x1_2d.flatten()
    output_coordinates[:, 2] = x2.flatten()
    return output_coordinates

def calculate_diffraction(electron_density, density_pixel_size, rotation, image_shape, wavelength, detector_distance, pixel_size):
    base_coordinates = ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size)
    rotated_coordinates = _rotations.rotate_array(rotation, base_coordinates)
    largest_distance_in_fourier_space = 1./(2.*density_pixel_size)
    scaled_coordinates = rotated_coordinates * 0.5 / largest_distance_in_fourier_space
    diffraction = _nfft.nfft(electron_density, scaled_coordinates)
    return diffraction.reshape(image_shape)

def calculate_diffraction(scattering_factor_density, density_pixel_size, rotation, image_shape, wavelength, detector_distance, pixel_size):
    base_coordinates = ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size)
    rotated_coordinates = _rotations.rotate_array(rotation, base_coordinates)
    largest_distance_in_fourier_space = 1./(2.*density_pixel_size)
    scaled_coordinates = rotated_coordinates * 0.5 / largest_distance_in_fourier_space
    diffraction = _nfft.nfft(electron_density, scaled_coordinates)
    return diffraction.reshape(image_shape)