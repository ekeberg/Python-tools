"""Create DFT matrices for 1D and 2D Fourier transforms"""
import numpy
import tools
import matplotlib
from matplotlib.colors import LogNorm


def dft_1d(array_length):
    """
    The dft matrix that is returnd works on complex vectors
    and returns a complex vector
    """
    omega = numpy.exp(-2.j*numpy.pi/array_length)
    i = numpy.arange(array_length)
    j = numpy.arange(array_length)
    dft = omega**(i[:, numpy.newaxis]*j[numpy.newaxis, :])
    return dft


def dft_1d_real(array_length):
    """
    The dft matrix that is returnd works on real vectors
    and returns complex numbers stored as a real vector
    as [a_real, a_imag, b_real, b_imag, ...]
    """
    dft_full = dft_1d(array_length)
    dft = numpy.zeros((dft_full.shape[0], 2*dft_full.shape[1]),
                      dtype="float64")
    dft[:, ::2] = numpy.real(dft_full)
    dft[:, 1::2] = numpy.imag(dft_full)
    return dft


def dft_2d(y_side, x_side):
    """
    The dft matrix that is returnd works on complex vectors
    and returns a complex vector. Data is stored consistent with
    numpys flatten().
    """
    o_1 = numpy.exp(-2.j*numpy.pi/y_side)
    o_2 = numpy.exp(-2.j*numpy.pi/x_side)
    i = numpy.zeros(x_side*y_side)
    j = numpy.zeros(x_side*y_side)
    for k in xrange(y_side):
        j[x_side*k:x_side*(k+1)] = numpy.arange(x_side)
    for k in xrange(x_side):
        i[k::x_side] = numpy.arange(y_side)
    dft = (o_1**(i[numpy.newaxis, :]*i[:, numpy.newaxis]) *
           o_2**(j[numpy.newaxis, :]*j[:, numpy.newaxis]))
    return dft


def dft_2d_real(y_side, x_side):
    """
    The dft matrix that is returnd works on real vectors
    and returns complex numbers stored as a real vector
    as [a_real, a_imag, b_real, b_imag, ...]. Data is stored
    consistent with numpys flatten().
    """
    dft_full = dft_2d(y_side, x_side)
    dft = numpy.zeros((dft_full.shape[0], 2*dft_full.shape[1]),
                      dtype="float64")
    dft[:, ::2] = numpy.real(dft_full)
    dft[:, 1::2] = numpy.imag(dft_full)
    return dft


def dft_2d_masked(y_side, x_side, mask_real, mask_fourier):
    """
    The dft matrix that is returnd works on complex vectors
    and returns a complex vector. Data is stored consistent with
    numpys flatten(). Only the cols and rows corresponding to pixels
    in the real and Fourier mask respectively are calculated.
    """
    o_1 = numpy.exp(-2.j*numpy.pi/y_side)
    o_2 = numpy.exp(-2.j*numpy.pi/x_side)
    i = numpy.zeros(x_side*y_side)
    j = numpy.zeros(x_side*y_side)
    for k in xrange(y_side):
        j[x_side*k:x_side*(k+1)] = numpy.arange(x_side)
    for k in xrange(x_side):
        i[k::x_side] = numpy.arange(y_side)
    i_mask_real = i[numpy.bool8(mask_real.flatten())]
    i_mask_fourier = i[numpy.bool8(mask_fourier.flatten())]
    j_mask_real = j[numpy.bool8(mask_real.flatten())]
    j_mask_fourier = j[numpy.bool8(mask_fourier.flatten())]
    dft = (o_1**(i_mask_real[:, numpy.newaxis] *
                 i_mask_fourier[numpy.newaxis, :]) *
           o_2**(j_mask_real[:, numpy.newaxis] *
                 j_mask_fourier[numpy.newaxis, :]))
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
    dft = numpy.zeros((dft_full.shape[0], 2*dft_full.shape[1]),
                      dtype="float64")
    dft[:, ::2] = numpy.real(dft_full)
    dft[:, 1::2] = numpy.imag(dft_full)
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
    dft = numpy.zeros((2*dft_full.shape[0], 2*dft_full.shape[1]),
                      dtype="float64")
    dft[0::2, 0::2] = numpy.real(dft_full)
    dft[1::2, 1::2] = numpy.real(dft_full)
    dft[0::2, 1::2] = numpy.imag(dft_full)
    dft[1::2, 0::2] = -numpy.imag(dft_full)
    return dft


def _test_dft_1d():
    """Print result of DFT multiplication and build in fft."""
    image_side = 4
    sample = numpy.random.random(image_side)
    ft_true = numpy.fft.fft(sample)
    dft = dft_1d(image_side)
    ft_new = dft*numpy.matrix(sample).transpose()
    print "built in"
    print ft_true
    print "my matrix"
    print ft_new


def _test_dft_2d():
    """Print result of DFT multiplication and build in fft."""
    image_side = 2
    sample = numpy.random.random((image_side, image_side*2))
    ft_true = numpy.fft.fft2(sample)
    dft = dft_2d(image_side, image_side*2)
    ft_flat = dft*numpy.matrix(sample.flatten()).transpose()
    ft_new = ft_flat.reshape((image_side, image_side*2))
    print "built in"
    print ft_true
    print "my matrix"
    print ft_new


def _test_dft_2d_visually():
    """Plot result of DFT multiplication and build in fft."""
    image_side = 100
    sample = numpy.fft.fftshift(numpy.random.random((image_side, )*2))
    sample *= numpy.fft.fftshift(tools.circular_mask(image_side, 10))
    ft_true = numpy.fft.fft2(sample)
    dft = dft_2d(image_side, image_side)
    ft_flat = dft*numpy.matrix(sample.flatten()).transpose()
    ft_new = ft_flat.reshape((image_side, image_side))

    fig = matplotlib.pyplot.figure(1)
    fig.clear()
    ax1 = fig.add_subplot(221)
    ax1.imshow(abs(ft_true), norm=LogNorm())
    ax2 = fig.add_subplot(222)
    ax2.imshow(abs(ft_new), norm=LogNorm())
    ax3 = fig.add_subplot(223)
    ax3.imshow(numpy.angle(ft_true), cmap="hsv")
    ax4 = fig.add_subplot(224)
    ax4.imshow(numpy.angle(ft_new), cmap="hsv")
    fig.canvas.draw()
