"""Create DFT matrices for 1D and 2D Fourier transforms"""
import numpy as _numpy
import scipy.fft as _fft
from . import tools as _tools
import matplotlib as _matplotlib
from matplotlib.colors import LogNorm as _LogNorm


def dft_1d(array_length):
    """
    The dft matrix that is returnd works on complex vectors
    and returns a complex vector
    """
    omega = _numpy.exp(-2.j*_numpy.pi/array_length)
    i = _numpy.arange(array_length)
    j = _numpy.arange(array_length)
    dft = omega**(i[:, _numpy.newaxis]*j[_numpy.newaxis, :])
    return dft


def dft_1d_real(array_length):
    """
    The dft matrix that is returnd works on real vectors
    and returns complex numbers stored as a real vector
    as [a_real, a_imag, b_real, b_imag, ...]
    """
    dft_full = dft_1d(array_length)
    dft = _numpy.zeros((2*dft_full.shape[0], dft_full.shape[1]),
                       dtype="float64")
    dft[0::2, :] = _numpy.real(dft_full)
    dft[1::2, :] = _numpy.imag(dft_full)
    return dft


def dft_2d(y_side, x_side):
    """
    The dft matrix that is returnd works on complex vectors
    and returns a complex vector. Data is stored consistent with
    numpys flatten().
    """
    o_1 = _numpy.exp(-2.j*_numpy.pi/y_side)
    o_2 = _numpy.exp(-2.j*_numpy.pi/x_side)
    i = _numpy.zeros(x_side*y_side)
    j = _numpy.zeros(x_side*y_side)
    for k in range(y_side):
        j[x_side*k:x_side*(k+1)] = _numpy.arange(x_side)
    for k in range(x_side):
        i[k::x_side] = _numpy.arange(y_side)
    dft = (o_1**(i[_numpy.newaxis, :]*i[:, _numpy.newaxis]) *
           o_2**(j[_numpy.newaxis, :]*j[:, _numpy.newaxis]))
    return dft


def dft_2d_real(y_side, x_side):
    """
    The dft matrix that is returnd works on real vectors
    and returns complex numbers stored as a real vector
    as [a_real, a_imag, b_real, b_imag, ...]. Data is stored
    consistent with numpys flatten().
    """
    dft_full = dft_2d(y_side, x_side)
    dft = _numpy.zeros((2*dft_full.shape[0], dft_full.shape[1]),
                       dtype="float64")
    dft[0::2, :] = _numpy.real(dft_full)
    dft[1::2, :] = _numpy.imag(dft_full)
    return dft


def dft_2d_masked(y_side, x_side, mask_real, mask_fourier):
    """
    The dft matrix that is returnd works on complex vectors
    and returns a complex vector. Data is stored consistent with
    numpys flatten(). Only the cols and rows corresponding to pixels
    in the real and Fourier mask respectively are calculated.
    """
    o_1 = _numpy.exp(-2.j*_numpy.pi/y_side)
    o_2 = _numpy.exp(-2.j*_numpy.pi/x_side)
    i = _numpy.zeros(x_side*y_side)
    j = _numpy.zeros(x_side*y_side)
    for k in range(y_side):
        j[x_side*k:x_side*(k+1)] = _numpy.arange(x_side)
    for k in range(x_side):
        i[k::x_side] = _numpy.arange(y_side)
    i_mask_real = i[_numpy.bool8(mask_real.flatten())]
    i_mask_fourier = i[_numpy.bool8(mask_fourier.flatten())]
    j_mask_real = j[_numpy.bool8(mask_real.flatten())]
    j_mask_fourier = j[_numpy.bool8(mask_fourier.flatten())]
    dft = (o_1**(i_mask_real[:, _numpy.newaxis] *
                 i_mask_fourier[_numpy.newaxis, :]) *
           o_2**(j_mask_real[:, _numpy.newaxis] *
                 j_mask_fourier[_numpy.newaxis, :]))
    return dft


def dft_2d_masked_real(y_side, x_side, mask_real, mask_fourier):
    """
    The dft matrix that is returnd works on real vectors
    and returns complex numbers stored as a real vector
    as [a_real, a_imag, b_real, b_imag, ...]. Data is stored
    consistent with numpys flatten(). Only the cols and rows
    corresponding to pixels in the real and Fourier mask respectively
    are calculated.
    """
    dft_full = dft_2d_masked(y_side, x_side, mask_real, mask_fourier)
    dft = _numpy.zeros((dft_full.shape[0], 2*dft_full.shape[1]),
                       dtype="float64")
    dft[:, ::2] = _numpy.real(dft_full)
    dft[:, 1::2] = _numpy.imag(dft_full)
    return dft


def dft_2d_masked_complex(y_side, x_side, mask_real, mask_fourier):
    """
    The dft matrix that is returnd works on complex vectors stored as real
    vectors as [a_real, a_imag, b_real, b_imag, ...] and returns vectors
    of the same type. Data is stored consistent with numpys flatten(). Only
    the cols and rows corresponding to pixels in the real and Fourier mask
    respectively are calculated.
    """
    dft_full = dft_2d_masked(y_side, x_side, mask_real, mask_fourier)
    dft = _numpy.zeros((2*dft_full.shape[0], 2*dft_full.shape[1]),
                       dtype="float64")
    dft[0::2, 0::2] = _numpy.real(dft_full)
    dft[1::2, 1::2] = _numpy.real(dft_full)
    dft[0::2, 1::2] = _numpy.imag(dft_full)
    dft[1::2, 0::2] = -_numpy.imag(dft_full)
    return dft


def dft(*shape):
    """
    The dft matrix that is returnd works on complex vectors
    and returns a complex vector. Data is stored consistent with
    numpys flatten().
    """
    omega = [_numpy.exp(-2.j*_numpy.pi/s) for s in shape]
    index = [element.flatten()
             for element in _numpy.meshgrid(*[_numpy.arange(s) for s in shape],
                                            indexing="ij")]

    dft = _numpy.ones((_numpy.prod(shape), )*2, dtype=_numpy.complex128)
    # from IPython.core.debugger import Tracer
    # Tracer()()

    for i, this_omega, this_index in zip(range(len(shape)), omega, index):
        dft *= this_omega**(this_index[_numpy.newaxis, :]
                            * this_index[:, _numpy.newaxis])
    return dft


def dft_real(*shape):
    """
    The dft matrix that is returnd works on real vectors
    and returns complex numbers stored as a real vector
    as [a_real, a_imag, b_real, b_imag, ...]. Data is stored
    consistent with numpys flatten().
    """
    dft_full = dft(*shape)
    real_dft = _numpy.zeros((2*_numpy.prod(shape),
                             _numpy.prod(shape)), dtype=_numpy.float64)
    real_dft[0::2, :] = _numpy.real(dft_full)
    real_dft[1::2, :] = _numpy.imag(dft_full)
    return real_dft


def dft_masked(mask_real, mask_fourier):
    """
    The dft matrix that is returnd works on complex vectors
    and returns a complex vector. Data is stored consistent with
    numpys flatten(). Only the cols and rows corresponding to pixels
    in the real and Fourier mask respectively are calculated.
    """
    shape = mask_real.shape
    mask_real = _numpy.bool8(mask_real)
    mask_fourier = _numpy.bool8(mask_fourier)
    omega = [_numpy.exp(-2.j*_numpy.pi/s) for s in shape]
    index = [element.flatten()
             for element in _numpy.meshgrid(
                     *[_numpy.arange(s) for s in shape], indexing="ij")]

    dft = _numpy.ones((mask_fourier.sum(), mask_real.sum()),
                      dtype=_numpy.complex128)
    index_mask_real = [this_index[_numpy.bool8(mask_real.flatten())]
                       for this_index in index]
    index_mask_fourier = [this_index[_numpy.bool8(mask_fourier.flatten())]
                          for this_index in index]
    for i, this_omega, this_index_mask_real, this_index_mask_fourier in zip(
            range(len(shape)), omega, index_mask_real, index_mask_fourier):
        dft *= this_omega**(this_index_mask_fourier[:, _numpy.newaxis]
                            * this_index_mask_real[_numpy.newaxis, :])
    return dft


def dft_masked_real(mask_real, mask_fourier):
    """
    The dft matrix that is returnd works on real vectors
    and returns complex numbers stored as a real vector
    as [a_real, a_imag, b_real, b_imag, ...]. Data is stored
    consistent with numpys flatten().
    """
    mask_real = _numpy.bool8(mask_real)
    mask_fourier = _numpy.bool8(mask_fourier)
    dft_full = dft_masked(mask_real, mask_fourier)
    dft = _numpy.zeros((2*mask_fourier.sum(), mask_real.sum()),
                       dtype=_numpy.float64)

    dft[0::2, :] = _numpy.real(dft_full)
    dft[1::2, :] = _numpy.imag(dft_full)
    return dft


def _test_dft_1d():
    """Print result of DFT multiplication and build in fft."""
    image_side = 4
    sample = _numpy.random.random(image_side)
    ft_true = _fft.fft(sample)
    dft = dft_1d(image_side)
    ft_new = dft*_numpy.matrix(sample).transpose()
    print("built in")
    print(ft_true)
    print("my matrix")
    print(ft_new)


def _test_dft_2d():
    """Print result of DFT multiplication and build in fft."""
    image_side = 2
    sample = _numpy.random.random((image_side, image_side*2))
    ft_true = _fft.fft2(sample)
    dft = dft_2d(image_side, image_side*2)
    ft_flat = dft*_numpy.matrix(sample.flatten()).transpose()
    ft_new = ft_flat.reshape((image_side, image_side*2))
    print("built in")
    print(ft_true)
    print("my matrix")
    print(ft_new)


def _test_dft_2d_visually():
    """Plot result of DFT multiplication and build in fft."""
    image_side = 100
    sample = _fft.fftshift(_numpy.random.random((image_side, )*2))
    sample *= _fft.fftshift(_tools.circular_mask(image_side, 10))
    ft_true = _fft.fft2(sample)
    dft = dft_2d(image_side, image_side)
    ft_flat = dft*_numpy.matrix(sample.flatten()).transpose()
    ft_new = ft_flat.reshape((image_side, image_side))

    fig = _matplotlib.pyplot.figure(1)
    fig.clear()
    ax1 = fig.add_subplot(221)
    ax1.imshow(abs(ft_true), norm=_LogNorm())
    ax2 = fig.add_subplot(222)
    ax2.imshow(abs(ft_new), norm=_LogNorm())
    ax3 = fig.add_subplot(223)
    ax3.imshow(_numpy.angle(ft_true), cmap="hsv")
    ax4 = fig.add_subplot(224)
    ax4.imshow(_numpy.angle(ft_new), cmap="hsv")
    fig.canvas.draw()


def _test_dft():
    side = 4

    dft_1d_1 = dft_1d(side)
    dft_1d_2 = dft((side, ))
    print(abs((dft_1d_1 - dft_1d_2)).sum())

    dft_2d_1 = dft_2d(side, side)
    dft_2d_2 = dft((side, side))
    print(abs((dft_2d_1 - dft_2d_2)).sum())

    dft_1d_real_1 = dft_1d_real(side)
    dft_1d_real_2 = dft_real((side, ))
    print(abs((dft_1d_real_1 - dft_1d_real_2)).sum())

    dft_2d_real_1 = dft_2d_real(side, side)
    dft_2d_real_2 = dft_real((side, side))
    print(abs((dft_2d_real_1 - dft_2d_real_2)).sum())


# def _test_dft_3d():
#     """Print result of DFT multiplication and build in fft."""
#     image_side = 3
#     sample = _numpy.random.random((image_side, image_side*2, image_side))
#     ft_true = _fft.fftn(sample)
#     dft = dft((image_side, image_side*2, image_side))
#     ft_flat = _numpy.array(dft*_numpy.matrix(sample.flatten()).transpose())
#     ft_new = ft_flat.reshape((image_side, image_side*2, image_side))
#     print("built in")
#     print(ft_true)
#     print("my matrix")
#     print(ft_new)
#     print("diff = {0}".format(abs(ft_true - ft_new).sum()))
