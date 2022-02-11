"""Calculate weakly constrained modes for a Fourier phase
retrieval problem using singular value decomposition"""
from . import dft as _dft
import numpy as _numpy
import scipy.fft as _fft


def calculate_modes(real_mask, fourier_mask, reality_constraint=False,
                    number_of_modes=None):
    """Calculate weakly constrained modes for a Fourier phase retrieval
    problem using singular value decomposition
    """
    if real_mask.shape != fourier_mask.shape:
        raise ValueError("Masks must have the same shape")

    real_mask_shifted = _fft.fftshift(real_mask)
    fourier_mask_shifted = _fft.fftshift(fourier_mask)

    if reality_constraint:
        sub_matrix = _dft.dft_nd_masked_real(real_mask_shifted,
                                             fourier_mask_shifted)
    else:
        sub_matrix = _dft.dft_nd_masked(real_mask_shifted,
                                        fourier_mask_shifted)

    sub_matrix /= _numpy.product([_numpy.sqrt(s) for s in real_mask.shape])
    u, s, v = _numpy.linalg.linalg.svd(sub_matrix, full_matrices=False)
    singular_values = s
    fourier_modes_raw = u
    real_modes_raw = v

    # We can have values slightly above 1. due to numerical errors but
    # if they are very large somthing bad is going on
    singular_values[(singular_values > 1.) * (singular_values < 1.1)] = 1.
    inverse_singular_values = _numpy.sqrt(1.-singular_values**2)

    if not number_of_modes:
        number_of_modes = len(singular_values)

    output_shape = (number_of_modes, ) + real_mask_shifted.shape
    real_modes_shifted = _numpy.zeros(output_shape, dtype="complex128")
    if reality_constraint:
        real_modes_shifted[:, real_mask_shifted] = (
            real_modes_raw[:number_of_modes, ::2]
            + 1.j*real_modes_raw[:number_of_modes, 1::2])
        real_modes_shifted[:, real_mask_shifted] = (
            real_modes_raw[:number_of_modes, :])
    else:
        real_modes_shifted[:, real_mask_shifted] = (
            real_modes_raw[:number_of_modes, :])
    real_modes = _fft.fftshift(real_modes_shifted, axes=(1, 2))
    fourier_modes_shifted = _numpy.zeros(output_shape, dtype="complex128")
    fourier_modes_shifted[:, fourier_mask_shifted] = (
        fourier_modes_raw.T[:number_of_modes])
    fourier_modes = _fft.fftshift(fourier_modes_shifted, axes=(1, 2))

    return inverse_singular_values[:number_of_modes], real_modes, fourier_modes
