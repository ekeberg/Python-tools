import numpy as _numpy
from scipy.interpolate import RegularGridInterpolator
from . import rotmodule


class MapRotater:
    def __init__(self, data, fill_value=0):
        self._data = data

        grid_points = [_numpy.arange(-s/2 + 0.5, s/2+0.5)
                       for s in self._data.shape]
        self._interpolater = RegularGridInterpolator(
            grid_points, self._data, fill_value=fill_value, bounds_error=False)

        grid_1d = (_numpy.arange(-s/2+0.5, s/2+0.5) for s in self._data.shape)
        x, y, z = _numpy.meshgrid(*grid_1d, indexing="ij")
        self._points = _numpy.vstack((x.flatten(), y.flatten(), z.flatten()))

    def rotate(self, rot):
        rotated_points = rotmodule.rotate(rot, self._points).T
        rotated_data = self._interpolater(rotated_points).reshape(
            self._data.shape)
        return rotated_data


def rotate_map(data, rot):
    return MapRotater(data).rotate(rot)

