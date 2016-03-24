import numpy as _numpy

def get_edge_kernel(image_shape, center=None):
    """The image must be centered before applying this kernel"""
    a = _numpy.array([0.25, 0.5, 0.25])
    kernel = a[:, _numpy.newaxis] * a[_numpy.newaxis, :]
    return abs(_numpy.fft.fftshift(_numpy.fft.fftn(_numpy.fft.fftshift(kernel), image_shape)))
