import numpy as _numpy
import scipy.fft as _fft


def get_edge_kernel(image_shape, center=None):
    """The image must be centered before applying this kernel"""
    a = _numpy.array([0.25, 0.5, 0.25])
    kernel = a[:, _numpy.newaxis] * a[_numpy.newaxis, :]
    return abs(_fft.fftshift(_fft.fftn(_fft.fftshift(kernel), image_shape)))
