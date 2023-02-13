import numpy as _numpy
from scipy.interpolate import RegularGridInterpolator
from . import rotmodule
from . import compare_rotations


class MapRotater:
    """Rotate a 3D map by a given quaternion. This is tested to be
    consistent with how rotations are used in EMC.

    """
    def __init__(self, data, center=None, fill_value=0):
        self._data = data

        if center is None:
            self._center = [s/2 - 0.5 for s in self._data.shape]
        else:
            self._center = center
            if len(self._center) != len(self._data.shape):
                raise ValueError("Center must have one element for each "
                                 "dimension of data")

        grid_points = [_numpy.arange(s) - c
                       for s, c in zip(self._data.shape, self._center)]
        # grid_points = [_numpy.arange(-s/2 + 0.5, s/2+0.5)
        #                for s in self._data.shape]
        self._interpolater = RegularGridInterpolator(
            grid_points, self._data, fill_value=fill_value, bounds_error=False)

        grid_1d = (_numpy.arange(-s/2+0.5, s/2+0.5) for s in self._data.shape)
        x, y, z = _numpy.meshgrid(*grid_1d, indexing="ij")
        self._points = _numpy.vstack((x.flatten(), y.flatten(), z.flatten()))

    def rotate(self, rot):
        "Rotate the map by the given rotation."
        rotated_points = rotmodule.rotate(rot, self._points).T
        rotated_data = self._interpolater(rotated_points).reshape(
            self._data.shape)
        return rotated_data


def rotate_map(data, rot, center=None):
    """Rotate the data by the given quaternion."""
    return MapRotater(data, center=center).rotate(rot)


def align_map(data, correct_rotations, recovered_rotations,
              symmetry_operations=((1, 0, 0, 0), ), shallow_ewald=True,
              return_rot_error=False):
    """Rotate the map back to the orientation used for simulation by
    providing the correct and recovered orientations. This function exists
    to a large degree to help

    """
    relative_rot = (
        compare_rotations.average_relative_orientation(correct_rotations,
                                                       recovered_rotations,
                                                       symmetry_operations,
                                                       shallow_ewald))
    rotated_data = rotate_map(data, relative_rot)
    if return_rot_error:
        aligned_rots = rotmodule.multiply(relative_rot, recovered_rotations)
        errors = rotmodule.relative_angle(correct_rotations, aligned_rots)
        average_error = _numpy.minimum(errors, _numpy.pi-errors).mean()
        return rotated_data, average_error
    else:
        return rotated_data

