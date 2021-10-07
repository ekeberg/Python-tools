import numpy
from eke import rotmodule

def get_arc_positions(relative_orientation, wavelength, pixel_size, detector_distance, image_shape, step=1):

    # relative_orientation = rotmodule.random(1000)
    # relative_orientation = rotmodule.random()

    multiple = len(relative_orientation.shape) > 1

    # wavelength = 1e-9 * 10
    # detector_distance = 0.74/10
    # pixel_size = 8*75e-6

    # image_shape = (128, 128)
    # step=1

    euler_angles = rotmodule.matrix_to_euler_angle(rotmodule.quaternion_to_matrix(relative_orientation), order="zyz")

    if multiple:
        relative_angle = euler_angles[...,1]
    else:
        relative_angle = numpy.array([euler_angles[...,1]])


    s_max = 2/wavelength*numpy.sin(0.5*numpy.arctan(min(image_shape)/2*pixel_size/detector_distance))
    arccos_argument = 1 - 0.5*s_max**2*wavelength**2/numpy.cos(relative_angle/2)**2

    phi_max = numpy.pi * numpy.ones_like(arccos_argument)
    arccos_mask = (arccos_argument > -1) * (arccos_argument < 1)
    phi_max[arccos_mask] = numpy.arccos(arccos_argument[arccos_mask])

    phi = numpy.linspace(-1, 1, min(image_shape))[numpy.newaxis, :] * phi_max[:, numpy.newaxis]

    denominator = 1 - numpy.cos(relative_angle[:, numpy.newaxis]/2)**2*(1 - numpy.cos(phi))
    x = detector_distance*numpy.cos(relative_angle[:, numpy.newaxis]/2)*numpy.sin(phi) / denominator
    y = detector_distance*numpy.cos(relative_angle[:, numpy.newaxis]/2)*numpy.sin(relative_angle[:, numpy.newaxis]/2)*(1-numpy.cos(phi)) / denominator

    if multiple:
        in_plane_0 = -euler_angles[..., 0]
        in_plane_1 = euler_angles[..., 2]
    else:
        in_plane_0 = -numpy.array([euler_angles[..., 0]])
        in_plane_1 = numpy.array([euler_angles[..., 2]])

    x_0 = numpy.cos(in_plane_0[:, numpy.newaxis])*x - numpy.sin(in_plane_0[:, numpy.newaxis])*(-y)
    y_0 = numpy.sin(in_plane_0[:, numpy.newaxis])*x + numpy.cos(in_plane_0[:, numpy.newaxis])*(-y)
    x_1 = numpy.cos(in_plane_1[:, numpy.newaxis])*x - numpy.sin(in_plane_1[:, numpy.newaxis])*(y)
    y_1 = numpy.sin(in_plane_1[:, numpy.newaxis])*x + numpy.cos(in_plane_1[:, numpy.newaxis])*(y)


    x_pixel_0 = x_0/pixel_size + image_shape[0]/2 - 0.5
    y_pixel_0 = y_0/pixel_size + image_shape[1]/2 - 0.5
    x_pixel_1 = x_1/pixel_size + image_shape[0]/2 - 0.5
    y_pixel_1 = y_1/pixel_size + image_shape[1]/2 - 0.5

    if multiple:
        return_array = numpy.zeros((x_pixel_0.shape[0], ) + (2, 2) + (x_pixel_0.shape[1], ), dtype=x_pixel_0.dtype)
    else:
        return_array = numpy.zeros((2, 2) + (x_pixel_0.shape[1], ), dtype=x_pixel_0.dtype)
    return_array[..., 0, 0, :] = x_pixel_0
    return_array[..., 0, 1, :] = y_pixel_0
    return_array[..., 1, 0, :] = x_pixel_1
    return_array[..., 1, 1, :] = y_pixel_1

    return return_array

