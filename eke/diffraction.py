import numpy as _numpy
from . import rotmodule as _rotmodule
from . import constants as _constants
from . import conversions as _conversions
from . import elements as _elements


def shannon_angle(size, wavelength):
    """Takes the size (diameter) in nm and returns the angle of a nyquist
    pixel
    """
    return wavelength/size


def ewald_coordinates(image_shape, wavelength, detector_distance, pixel_size,
                      edge_distance=None):
    """Returns posisions of detector pixels in units of pixels"""
    try:
        pixel_size[1]
    except TypeError:
        pixel_size = (pixel_size, pixel_size)
    image_shape = tuple(image_shape)
    if edge_distance is None:
        edge_distance = image_shape[0]/2.
    x_pixels_1d = _numpy.arange(image_shape[1]) - image_shape[1]/2. + 0.5
    y_pixels_1d = _numpy.arange(image_shape[0]) - image_shape[0]/2. + 0.5
    y_pixels, x_pixels = _numpy.meshgrid(y_pixels_1d, x_pixels_1d,
                                         indexing="ij")
    x_meters = x_pixels*pixel_size[0]
    y_meters = y_pixels*pixel_size[1]
    radius_meters = _numpy.sqrt(x_meters**2 + y_meters**2)

    scattering_angle = _numpy.arctan(radius_meters / detector_distance)
    z = -1./wavelength*(1. - _numpy.cos(scattering_angle))
    radius_fourier = _numpy.sqrt(1./wavelength**2
                                 - (1./wavelength - abs(z))**2)

    x = x_meters * radius_fourier / radius_meters
    y = y_meters * radius_fourier / radius_meters

    x[radius_meters == 0.] = 0.
    y[radius_meters == 0.] = 0.

    output_coordinates = _numpy.zeros((3, ) + image_shape)
    output_coordinates[0, :, :] = x
    output_coordinates[1, :, :] = y
    output_coordinates[2, :, :] = z

    # Rescale so that edge pixels match.
    furthest_edge_coordinate = _numpy.sqrt(x[0, image_shape[1]//2]**2
                                           + y[0, image_shape[1]//2]**2
                                           + z[0, image_shape[1]//2]**2)
    rescale_factor = edge_distance/furthest_edge_coordinate
    output_coordinates *= rescale_factor

    return output_coordinates


def ewald_coordinates_fourier(image_shape, wavelength, detector_distance,
                              pixel_size):
    """Returns posisions of detector pixels in Fourier units"""
    try:
        pixel_size[1]
    except TypeError:
        pixel_size = (pixel_size, pixel_size)
    image_shape = tuple(image_shape)
    x_pixels_1d = _numpy.arange(image_shape[1]) - image_shape[1]/2. + 0.5
    y_pixels_1d = _numpy.arange(image_shape[0]) - image_shape[0]/2. + 0.5
    y_pixels, x_pixels = _numpy.meshgrid(y_pixels_1d, x_pixels_1d,
                                         indexing="ij")
    x_meters = x_pixels*pixel_size[0]
    y_meters = y_pixels*pixel_size[1]
    radius_meters = _numpy.sqrt(x_meters**2 + y_meters**2)

    scattering_angle = _numpy.arctan(radius_meters / detector_distance)
    z = -1./wavelength*(1. - _numpy.cos(scattering_angle))
    radius_fourier = _numpy.sqrt(1./wavelength**2
                                 - (1./wavelength - abs(z))**2)

    x = x_meters * radius_fourier / radius_meters
    y = y_meters * radius_fourier / radius_meters

    x[radius_meters == 0.] = 0.
    y[radius_meters == 0.] = 0.

    output_coordinates = _numpy.zeros((3, ) + image_shape)
    output_coordinates[0, :, :] = x
    output_coordinates[1, :, :] = y
    output_coordinates[2, :, :] = z

    return output_coordinates


def pixel_solid_angle(image_shape, detector_distance, pixel_size):
    try:
        pixel_size[1]
    except TypeError:
        pixel_size = (pixel_size, pixel_size)
    image_shape = tuple(image_shape)
    x_pixels_1d = _numpy.arange(image_shape[1]) - image_shape[1]/2. + 0.5
    y_pixels_1d = _numpy.arange(image_shape[0]) - image_shape[0]/2. + 0.5
    y_pixels, x_pixels = _numpy.meshgrid(y_pixels_1d, x_pixels_1d,
                                         indexing="ij")
    x_meters = x_pixels*pixel_size[0]
    y_meters = y_pixels*pixel_size[1]
    z_meters = detector_distance
    radius_meters = _numpy.sqrt(x_meters**2 + y_meters**2 + z_meters**2)

    solid_angle = pixel_size[0]*pixel_size[1] / radius_meters**2
    return solid_angle


def calculate_diffraction(scattering_factor_density, density_pixel_size,
                          rotation, image_shape, wavelength, detector_distance,
                          pixel_size):
    import nfft as _nfft
    base_coordinates = ewald_coordinates(
        image_shape, wavelength,
        detector_distance, pixel_size)
    rotated_coordinates = _rotmodule.rotate_array(rotation, base_coordinates)
    diffraction = _nfft.nfft(scattering_factor_density, density_pixel_size,
                             rotated_coordinates)
    return diffraction.reshape(image_shape)


def klein_nishina(energy, scattering_angle, polarization_angle):
    """The cross section of a free electron. Energy given in eV, angles
    given in radians."""
    energy_in_joules = _conversions.ev_to_J(energy)
    rescaled_energy = energy_in_joules / (_constants.me*_constants.c**2)

    relative_energy_change = 1. / (1. + rescaled_energy *
                                   (1. - _numpy.cos(scattering_angle)))
    cross_section = ((_constants.re**2 * relative_energy_change**2)/2. *
                     (relative_energy_change + 1./relative_energy_change -
                      (_numpy.sin(scattering_angle)**2
                       * _numpy.cos(polarization_angle)**2)))
    return cross_section


def sphere_diffraction(diameter, material, wavelength, image_shape,
                       detector_distance, pixel_size, intensity):
    """Return the wave of the diffraction from a uniform sphere. To get
    the diffracted intensity, take sqrt(d)**2"""
    photon_energy = _conversions.m_to_ev(wavelength)
    intensity_photons = _conversions.J_to_ev(intensity) / photon_energy
    r = diameter/2
    F0 = _numpy.sqrt(intensity_photons)*2*_numpy.pi/wavelength**2
    V = 4/3*_numpy.pi*r**3
    dn = 1-_elements.get_index_of_refraction(photon_energy, material)
    K = (F0*V*dn)**2
    coordinates = ewald_coordinates_fourier(image_shape, wavelength,
                                            detector_distance, pixel_size)
    q = _numpy.sqrt((coordinates**2).sum(axis=0))*2*_numpy.pi
    Omega_p = pixel_solid_angle(image_shape, detector_distance, pixel_size)

    epsilon = _numpy.finfo("float64").eps
    intensity = 3 * _numpy.sqrt(abs(K))
    diffraction = (intensity * (_numpy.sin(q*r) - q*r*_numpy.cos(q*r))
                   / ((q * r)**3 + epsilon))
    diffraction *= _numpy.sqrt(Omega_p)
    return diffraction
